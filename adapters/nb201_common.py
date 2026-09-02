"""Shared helpers for NB201 / NATS-TSS AZ-NAS search + 200-ep train adapters.

Locks (see plan): C=16, N=5, max_nodes=4, seed=0; NB201 ZeroShotProxy +
``extract_cell_features`` + TinyNetwork only (never MBV2 ``run_score_matrix``);
rank aggregate Σ log(rank/n) over expressivity, progressivity, trainability,
complexity (P0-measured FLOPs); W&B option A with local JSON as SoT.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# Locked macro (NB201 / NATS-TSS).
NB201_C = 16
NB201_N = 5
NB201_MAX_NODES = 4
NB201_SEED = 0
TSS_FULL_SIZE = 15625

RANK_KEYS = ("expressivity", "progressivity", "trainability", "complexity")

RANKING_KEY = "az_nas_rank_sum"
RANKING_KEY_DOC = (
    "az_nas_rank_sum: for architectures scored in one nb201_search_az.py run, "
    "treat that set as a population. For each key in "
    "{expressivity, progressivity, trainability, complexity}, rank values with "
    "scipy.stats.rankdata (higher raw value = higher rank), then sum "
    "log(rank / population_size) across keys. complexity = FLOPs measured at "
    "P0 (C,H,W) via xautodl.utils.get_model_infos — never NATS API CIFAR FLOPs. "
    "Only comparable within one search population (same dataset, seed, n)."
)

DEFAULT_WANDB_ENTITY = "EXPERIMENT_WANDB_ENTITY"
DEFAULT_WANDB_PROJECT = "aznas-nb201-grow"

# Pilot / paper P0 dataset names.
P0_DATASETS = (
    "multnist",
    "cifartile",
    "geoclassing",
    "gutenberg",
    "chesseract",
)

# Score pad expectations (search always; train when ``train_uses_score_pad``).
SCORE_PAD_EXPECTATIONS: dict[str, tuple[int, int, int]] = {
    "multnist": (3, 32, 32),
    "cifartile": (3, 64, 64),
    "geoclassing": (3, 64, 64),
    "gutenberg": (1, 32, 32),
    "chesseract": (12, 8, 8),  # board12 family; stem in_channels=12 later
}

# Datasets whose *native* H/W break NB201 ``ResNetBasicblock`` stride-2 residuals
# (Conv3×3 pad=1 → ceil-ish vs AvgPool2d pad=0 → floor on odd sides). Gutenberg
# native ``(1, 27, 18)`` yields 14 vs 13 on dim 2 at the first reduction. Align
# train to the same score pad (32×32) as search — do not rerank.
TRAIN_SCORE_PAD_DATASETS: frozenset[str] = frozenset({"gutenberg"})


def train_uses_score_pad(dataset: str) -> bool:
    """Whether 200-ep train should apply the search score pad for ``dataset``."""
    return dataset in TRAIN_SCORE_PAD_DATASETS


def resolve_az_nas_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_nb201_root(az_nas_root: Path | None = None) -> Path:
    root = az_nas_root if az_nas_root is not None else resolve_az_nas_root()
    return root / "NB201"


class _Nb201Context:
    """Push ``NB201/`` onto ``sys.path`` and chdir there for local imports."""

    def __init__(self, nb201_root: Path):
        self.nb201_root = nb201_root
        self._prev_cwd: str | None = None
        self._path_added = False

    def __enter__(self) -> "_Nb201Context":
        root_str = str(self.nb201_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            self._path_added = True
        self._prev_cwd = os.getcwd()
        os.chdir(root_str)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._prev_cwd is not None:
            os.chdir(self._prev_cwd)
        if self._path_added:
            try:
                sys.path.remove(str(self.nb201_root))
            except ValueError:
                pass


def nb201_context(az_nas_root: Path | None = None) -> _Nb201Context:
    return _Nb201Context(resolve_nb201_root(az_nas_root))


def tss_op_names() -> list[str]:
    """NATS-TSS / NAS-Bench-201 operation names (locked search space)."""
    return [
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]


def random_genotype(max_nodes: int, op_names: Sequence[str], *, rng: random.Random):
    """Sample one NB201 cell genotype (Structure), matching tss_general.ipynb."""
    from xautodl.models.cell_searchs.genotypes import Structure  # noqa: PLC0415

    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            op_name = rng.choice(list(op_names))
            xlist.append((op_name, j))
        genotypes.append(tuple(xlist))
    return Structure(genotypes)


def generate_all_archs(max_nodes: int = NB201_MAX_NODES):
    """Enumerate the full TSS (15 625) cell genotypes in canonical order."""
    from xautodl.models.cell_searchs.genotypes import Structure  # noqa: PLC0415

    return Structure.gen_all(tss_op_names(), max_nodes, False)


def genotype_to_str(arch: Any) -> str:
    if hasattr(arch, "tostr"):
        return str(arch.tostr())
    return str(arch)


def genotype_short(arch_or_str: Any, *, n: int = 24) -> str:
    """Short, filesystem-safe genotype id for run names / ckpt dirs."""
    s = genotype_to_str(arch_or_str)
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    compact = (
        s.replace("|", "")
        .replace("+", "_")
        .replace("~", "")
        .replace("nor_conv_", "c")
        .replace("skip_connect", "sk")
        .replace("avg_pool_3x3", "ap")
        .replace("none", "no")
    )
    if len(compact) > n:
        compact = compact[:n]
    return f"{compact}_{digest}"


def parse_genotype(xstr: str):
    from xautodl.models.cell_searchs.genotypes import Structure  # noqa: PLC0415

    return Structure.str2structure(xstr)


def build_tiny_network(
    genotype: Any,
    *,
    num_classes: int,
    in_channels: int = 3,
    C: int = NB201_C,
    N: int = NB201_N,
):
    """Build NB201 TinyNetwork with configurable stem ``in_channels``.

    Pilot RGB datasets use ``in_channels=3``. gutenberg (1) and chesseract (12)
    are wired for Phase 3 via this stem parameter.
    """
    import torch.nn as nn  # noqa: PLC0415
    from xautodl.models.cell_infers.cells import InferCell  # noqa: PLC0415
    from xautodl.models.cell_operations import ResNetBasicblock  # noqa: PLC0415

    class TinyNetwork(nn.Module):
        """NB201 macro with ``extract_cell_features`` (custom/tss_model.py)."""

        def __init__(self) -> None:
            super().__init__()
            self._C = C
            self._layerN = N
            self._in_channels = in_channels
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
            layer_reductions = (
                [False] * N + [True] + [False] * N + [True] + [False] * N
            )
            C_prev = C
            self.cells = nn.ModuleList()
            for C_curr, reduction in zip(layer_channels, layer_reductions):
                if reduction:
                    cell = ResNetBasicblock(C_prev, C_curr, 2, True)
                else:
                    cell = InferCell(genotype, C_prev, C_curr, 1)
                self.cells.append(cell)
                C_prev = cell.out_dim
            self._Layer = len(self.cells)
            self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(C_prev, num_classes)

        def extract_cell_features(self, inputs):
            cell_features = []
            feature = self.stem(inputs)
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)
            for cell in self.cells:
                feature = cell(feature)
                if feature.requires_grad:
                    feature.retain_grad()
                cell_features.append(feature)
            return cell_features

        def forward(self, inputs):
            feature = self.stem(inputs)
            for cell in self.cells:
                feature = cell(feature)
            out = self.lastact(feature)
            out = self.global_pooling(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)
            return out, logits

    return TinyNetwork()


def measure_flops_params(model: Any, batch_shape_chw: Sequence[int]) -> tuple[float, float]:
    """P0-measured FLOPs (M) and params (MB) via ``get_model_infos``.

    ``batch_shape_chw`` is ``(C, H, W)`` — never NATS API CIFAR FLOPs.
    """
    from xautodl.utils import get_model_infos  # noqa: PLC0415

    c, h, w = (int(x) for x in batch_shape_chw)
    flops, params = get_model_infos(model, [1, c, h, w])
    return float(flops), float(params)


def compute_az_nas_rank_sum(
    infos: Sequence[dict[str, float]],
    *,
    keys: Sequence[str] = RANK_KEYS,
) -> list[float]:
    """Population rank-sum aggregate; higher is better."""
    from scipy import stats  # noqa: PLC0415

    n = len(infos)
    if n == 0:
        return []
    total = np.zeros(n, dtype=float)
    for key in keys:
        values = np.array([float(info.get(key, np.nan)) for info in infos], dtype=float)
        # Non-finite → very small so they rank worst under higher-is-better.
        values = np.where(np.isfinite(values), values, -np.inf)
        ranks = stats.rankdata(values)
        total = total + np.log(ranks / n)
    return [float(x) for x in total]


def non_finite_info() -> dict[str, float]:
    return {
        "expressivity": float("-inf"),
        "progressivity": float("-inf"),
        "trainability": float("-inf"),
        "complexity": float("-inf"),
    }


def score_one_arch(
    network: Any,
    *,
    gpu: int | None,
    cached_batches: list[list[Any]],
    resolution: int,
    batch_size: int,
) -> dict[str, float]:
    """Call NB201 ``compute_az_nas_score``; map failures / ``g_in is None`` to -inf.

    Stock scorer may raise when ``autograd.grad`` returns ``None`` (g_in) or
    PixelUnshuffle geometry fails; treat as non-finite trainability / scores.
    """
    from ZeroShotProxy import compute_az_nas_score  # noqa: PLC0415

    try:
        info = compute_az_nas_score.compute_nas_score(
            model=network,
            gpu=gpu,
            trainloader=cached_batches,
            resolution=resolution,
            batch_size=batch_size,
        )
    except Exception:
        return non_finite_info()

    out: dict[str, float] = {}
    for key in ("expressivity", "progressivity", "trainability"):
        val = float(info.get(key, float("-inf")))
        if not np.isfinite(val):
            val = float("-inf")
        out[key] = val
    return out


def wandb_run_kwargs(
    *,
    run_name: str,
    config: dict[str, Any],
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    entity = os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    project = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    mode = os.environ.get("WANDB_MODE", "online")
    return {
        "entity": entity if entity and entity != DEFAULT_WANDB_ENTITY else None,
        "project": project,
        "name": run_name,
        "config": config,
        "tags": list(tags or []),
        "mode": mode,
        "reinit": True,
    }


@contextlib.contextmanager
def maybe_wandb(
    *,
    enabled: bool,
    run_name: str,
    config: dict[str, Any],
    tags: Sequence[str] | None = None,
) -> Iterator[Any]:
    """W&B option A: online by default; no-op when disabled or import fails."""
    if not enabled:
        yield None
        return
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        print("[nb201] wandb not installed; continuing without W&B", flush=True)
        yield None
        return

    kwargs = wandb_run_kwargs(run_name=run_name, config=config, tags=tags)
    # Avoid failing hard on placeholder entity — omit entity so default user applies.
    init_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[nb201] wandb.init failed ({exc}); continuing without W&B", flush=True)
        yield None
        return
    try:
        yield run
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# Documented 6-run train policy per dataset (top-3 + random; top-1 × 3 seeds).
TRAIN_RUN_POLICY_DOC = (
    "Per dataset: top-1 × 3 seeds; top-2 × 1 seed; top-3 × 1 seed; "
    "1 random control × 1 seed → 6 train runs total."
)


def expand_train_jobs(
    ranked_archs: Sequence[dict[str, Any]],
    *,
    top_k: int = 3,
    include_random: bool = True,
    top1_seeds: Sequence[int] = (0, 1, 2),
    other_seed: int = 0,
    random_seed: int = 0,
) -> list[dict[str, Any]]:
    """Expand a ranked search list into the locked 6-run train job list."""
    if len(ranked_archs) < top_k:
        raise ValueError(
            f"Need at least top_k={top_k} ranked archs, got {len(ranked_archs)}"
        )
    jobs: list[dict[str, Any]] = []
    for seed in top1_seeds:
        jobs.append(
            {
                "role": "top-1",
                "rank": 1,
                "seed": int(seed),
                "arch": ranked_archs[0],
            }
        )
    for k in range(2, top_k + 1):
        jobs.append(
            {
                "role": f"top-{k}",
                "rank": k,
                "seed": int(other_seed),
                "arch": ranked_archs[k - 1],
            }
        )
    if include_random:
        # Prefer an explicit random pick from the search artifact when present.
        random_arch = None
        for row in ranked_archs:
            if row.get("selection_role") == "random":
                random_arch = row
                break
        if random_arch is None:
            # Fall back: last entry or a mid-ranked cell.
            random_arch = ranked_archs[min(len(ranked_archs) - 1, top_k)]
        jobs.append(
            {
                "role": "random",
                "rank": None,
                "seed": int(random_seed),
                "arch": random_arch,
            }
        )
    return jobs
