"""Shared helpers for AZ-NAS grow adapters (`smoke_score.py`, `run_score_matrix.py`).

Both scoring entry points need the same handful of things: resolve sibling
paths, get dual git provenance (SHA + dirty flag) for the result JSON, hand
off to `ImageNet_MBV2` for model building / `ZeroShotProxy` scoring, and
write a result JSON that matches
`experimental_grow/experiments/AZ-NAS/result_schema.md`. This module holds
that shared logic once so neither script has to reimplement it.

Import contract (mirrors `spaces/_common.py`): this module is safe to import
from *outside* the MBV2 venv (no `torch`/`Masternet` import at module load
time beyond stdlib + `numpy`) -- MBV2-local imports (`torch`, `Masternet`,
`ZeroShotProxy`) only happen lazily inside the functions that actually need
them, after the caller has entered :func:`mbv2_context`.

Real-batch bridging (the locked-but-unresolved question flagged in
`SETUP.md`'s `grow-data-adapter` section): the five matrix datasets
(multnist, cifartile, geoclassing, gutenberg, chesseract) are backed by
`tools.datasets.NpyWebDataset` subclasses, which import `gromo` at module
load time. `gromo` is grow's own package, version-pinned against a much
newer `torch` than the MBV2 venv's pinned `torch==2.0.1` -- installing it
into the MBV2 venv would silently fight that pin (see `SETUP.md`). So
`grow_data.load()` itself must run inside grow's own `uv` environment, in a
*separate process* from the MBV2-venv process that builds `MasterNet` and
calls `ZeroShotProxy`. :func:`load_real_batches` shells out to
`_bridge_grow_batches.py` via `uv run --project <grow_root>` and exchanges
plain `.npy`/JSON on disk (never `torch.save`/pickle) so the two processes
never need pickle-compatible torch versions.

That bridge also gates network access: `NpyWebDataset.__init__` downloads
its Figshare zip unconditionally whenever it isn't already on disk,
regardless of any `download=` flag (see
`experimental_grow/tools/datasets.py::NpyWebDataset._download_and_extract`).
`_bridge_grow_batches.py` checks for the on-disk zip *before* ever
instantiating the dataset and refuses to proceed (reporting
`status="skipped_missing_data"`) unless `--allow_download` is passed, so
routine matrix runs never trigger a multi-hundred-MB download by surprise.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_BRIDGE_SCRIPT = Path(__file__).resolve().parent / "_bridge_grow_batches.py"


class DataUnavailableError(RuntimeError):
    """Raised by :func:`load_real_batches` when the bridge reports its
    `status="skipped_missing_data"` sentinel (the dataset's raw zip isn't on
    disk and `--allow_download` wasn't set). Callers should treat this as a
    documented skip, not a scoring failure.
    """

    def __init__(self, message: str, *, meta: dict[str, Any]):
        super().__init__(message)
        self.meta = meta


class NoCudaError(RuntimeError):
    """Raised when a caller explicitly requested a specific `--gpu` index but
    no CUDA device is visible on this host.
    """


# ---------------------------------------------------------------------------
# Path resolution (mirrors grow_data._resolve_grow_root / spaces/_common.py)
# ---------------------------------------------------------------------------


def resolve_az_nas_root() -> Path:
    """`AZ-NAS` repo root; this file lives at `<AZ-NAS-root>/adapters/adapter_utils.py`."""
    return Path(__file__).resolve().parents[1]


def resolve_mbv2_root(az_nas_root: Path | None = None) -> Path:
    root = az_nas_root if az_nas_root is not None else resolve_az_nas_root()
    return root / "ImageNet_MBV2"


def _looks_like_grow_root(root: Path) -> bool:
    return (root / "hydra_script" / "configs").is_dir()


def _documented_grow_candidates() -> list[Path]:
    """Documented dual-worktree grow paths (never the AZ-NAS.worktrees sibling)."""
    home = Path.home()
    return [
        home / ".cursor" / "worktrees" / "experimental_grow" / "aznas",
        home / "experimental_grow.worktrees" / "aznas-compare",
        home / "Projects" / "research" / "experimental_grow.worktrees" / "aznas-compare",
    ]


def resolve_grow_root() -> Path:
    """Resolve the ``experimental_grow`` repo root used for Hydra datasets.

    Order:
      1. ``EXPERIMENTAL_GROW_ROOT`` if set (must contain ``hydra_script/configs``)
      2. Documented worktree paths (Cursor-native, cluster ``$HOME/.../aznas-compare``,
         local legacy sibling worktree)
      3. Main ``experimental_grow`` sibling of an AZ-NAS *main* clone only

    Deliberately refuses the silent footgun
    ``<AZ-NAS.worktrees>/experimental_grow`` (wrong sibling under the AZ-NAS
    worktrees parent). Cluster jobs must set ``EXPERIMENTAL_GROW_ROOT`` to
    ``$HOME/experimental_grow.worktrees/aznas-compare``.
    """
    env_root = os.environ.get("EXPERIMENTAL_GROW_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not _looks_like_grow_root(root):
            raise FileNotFoundError(
                f"EXPERIMENTAL_GROW_ROOT={root!s} lacks hydra_script/configs. "
                "Set it to a real experimental_grow checkout "
                "(cluster: $HOME/experimental_grow.worktrees/aznas-compare)."
            )
        return root

    for cand in _documented_grow_candidates():
        if cand.is_dir() and _looks_like_grow_root(cand):
            return cand.resolve()

    az_nas_root = resolve_az_nas_root()
    az_parent = az_nas_root.parent
    # Refuse AZ-NAS.worktrees/experimental_grow (common wrong sibling).
    if az_parent.name == "AZ-NAS.worktrees" or "AZ-NAS.worktrees" in az_parent.parts:
        raise FileNotFoundError(
            "Cannot resolve experimental_grow from an AZ-NAS worktree without "
            "EXPERIMENTAL_GROW_ROOT. Refusing silent sibling "
            f"{az_parent / 'experimental_grow'!s}. Set EXPERIMENTAL_GROW_ROOT to "
            "$HOME/experimental_grow.worktrees/aznas-compare (cluster) or "
            "~/.cursor/worktrees/experimental_grow/aznas (local Cursor)."
        )

    sibling = az_parent / "experimental_grow"
    if _looks_like_grow_root(sibling):
        return sibling.resolve()

    raise FileNotFoundError(
        f"Cannot find an experimental_grow checkout near {az_nas_root!s} "
        "(expected hydra_script/configs). Set EXPERIMENTAL_GROW_ROOT to override."
    )


def resolve_results_dir(grow_root: Path | None = None) -> Path:
    root = grow_root if grow_root is not None else resolve_grow_root()
    results = root / "experiments" / "AZ-NAS" / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results


# Module-level convenience constants (cheap Path arithmetic only, no I/O) --
# `smoke_score.py` uses these directly (`adapter_utils.AZ_NAS_ROOT`,
# `adapter_utils.MBV2_ROOT`) instead of calling the resolver functions.
AZ_NAS_ROOT: Path = resolve_az_nas_root()
MBV2_ROOT: Path = resolve_mbv2_root(AZ_NAS_ROOT)


def _ensure_grow_on_syspath(grow_root: Path) -> None:
    """Prepend `grow_root` to `sys.path` so `import grow_data` (and, in turn,
    grow's own `hydra_script`/`tools` packages) resolve. Idempotent.
    """
    root_str = str(grow_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# Git provenance (dual SHAs + dirty flags, per result_schema.md)
# ---------------------------------------------------------------------------

_PROVENANCE_REQUIRED_KEYS = (
    "experimental_grow_sha",
    "experimental_grow_dirty",
    "az_nas_sha",
    "az_nas_dirty",
)


def git_sha_and_dirty(repo_root: Path) -> tuple[str, bool]:
    """Best-effort `git rev-parse HEAD` + dirty-tree check for `repo_root`.

    Returns `("unknown", True)` on any failure (not a repo, `git` missing,
    timeout, ...) rather than raising -- provenance is best-effort and must
    never abort a scoring run.

    For paper corpus rows, prefer :func:`load_provenance_freeze` /
    ``--provenance-file`` / ``AZ_NAS_PROVENANCE_FILE`` so SHA+dirty are frozen
    at job start — do **not** rely on this live call alone after scoring starts.
    """
    try:
        sha_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if sha_proc.returncode != 0:
            return "unknown", True
        sha = sha_proc.stdout.strip()
    except Exception:
        return "unknown", True

    try:
        status_proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = status_proc.returncode != 0 or bool(status_proc.stdout.strip())
    except Exception:
        dirty = True

    return sha or "unknown", dirty


def git_sha(repo_root: Path) -> str:
    """Single-value convenience wrapper around :func:`git_sha_and_dirty` for
    callers (e.g. `smoke_score.py`) that want the SHA and dirty flag as two
    separate calls rather than one tuple.
    """
    return git_sha_and_dirty(repo_root)[0]


def git_dirty(repo_root: Path) -> bool:
    """See :func:`git_sha`."""
    return git_sha_and_dirty(repo_root)[1]


def resolve_provenance_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve a job-start provenance freeze file path, or None if unset."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("AZ_NAS_PROVENANCE_FILE")
    if env:
        return Path(env).expanduser().resolve()
    return None


def load_provenance_freeze(
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load dual-SHA+dirty freeze from ``--provenance-file`` / env, or None.

    When a path is set (CLI or ``AZ_NAS_PROVENANCE_FILE``), the file **must**
    exist and contain the four required SHA/dirty keys — missing freeze is a
    hard error (paper mode must not silently fall back to live porcelain).
    """
    resolved = resolve_provenance_path(path)
    if resolved is None:
        return None
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Provenance freeze file not found: {resolved}. "
            "Pass --provenance-file or set AZ_NAS_PROVENANCE_FILE to the "
            "job-start provenance.json written before scoring."
        )
    data = json.loads(resolved.read_text())
    missing = [k for k in _PROVENANCE_REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Provenance freeze {resolved} missing required key(s): {missing}"
        )
    return data


def resolve_dual_sha_dirty(
    grow_root: Path,
    az_nas_root: Path,
    *,
    provenance: dict[str, Any] | None = None,
    provenance_path: str | Path | None = None,
) -> tuple[str, bool, str, bool]:
    """Return (grow_sha, grow_dirty, az_sha, az_dirty).

    Prefer a job-start freeze (``provenance`` dict, or load from
    ``provenance_path`` / ``AZ_NAS_PROVENANCE_FILE``). Only fall back to live
    :func:`git_sha_and_dirty` when no freeze is configured (local/dev).
    """
    freeze = provenance
    if freeze is None:
        freeze = load_provenance_freeze(provenance_path)
    if freeze is not None:
        return (
            str(freeze["experimental_grow_sha"]),
            bool(freeze["experimental_grow_dirty"]),
            str(freeze["az_nas_sha"]),
            bool(freeze["az_nas_dirty"]),
        )
    grow_sha, grow_dirty = git_sha_and_dirty(grow_root)
    az_sha, az_dirty = git_sha_and_dirty(az_nas_root)
    return grow_sha, grow_dirty, az_sha, az_dirty


# ---------------------------------------------------------------------------
# MasterNet seeding + torch/CUDA device extras (paper-mode pre-submit gate)
# ---------------------------------------------------------------------------


def seed_for_masternet(seed: int) -> None:
    """Seed numpy / torch / CUDA RNGs **before** MasterNet build (paper protocol)."""
    import torch  # noqa: PLC0415

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_device_extras(*, device: str) -> dict[str, Any]:
    """Record ``torch_version`` / ``torch_cuda_version`` / ``gpu_name`` for JSON."""
    import torch  # noqa: PLC0415

    extras: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": None,
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        idx = 0
        if ":" in str(device):
            try:
                idx = int(str(device).split(":", 1)[1])
            except ValueError:
                idx = 0
        extras["gpu_name"] = torch.cuda.get_device_name(idx)
    return extras


def assert_scipy_available() -> None:
    """Hard-fail helper: ``az_nas_rank_sum`` needs scipy in the MBV2 venv."""
    try:
        from scipy import stats  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "scipy is required for az_nas_rank_sum (matrix appendix). "
            "Install scipy into the MBV2 .venv-mbv2 (Phase 1 bootstrap)."
        ) from exc


def assert_paper_ready() -> None:
    """Machine-checkable paper-mode gate (seed + extras + provenance freeze API).

    Used by ``submit_cluster.sh`` / ``--paper-ready`` before paper-eligible
    submit. Ops-only debug may skip this gate but cannot be promoted to the
    paper corpus.
    """
    import inspect  # noqa: PLC0415

    if not callable(seed_for_masternet):
        raise AssertionError("seed_for_masternet missing")
    if not callable(load_provenance_freeze):
        raise AssertionError("load_provenance_freeze missing")
    if not callable(torch_device_extras):
        raise AssertionError("torch_device_extras missing")
    src = inspect.getsource(build_result_record)
    for marker in ("torch_version", "torch_cuda_version", "gpu_name", "resolve_dual_sha_dirty"):
        if marker not in src:
            raise AssertionError(
                f"build_result_record does not reference {marker!r} "
                "(paper-mode extras / provenance freeze not wired)"
            )
    assert_scipy_available()


# ---------------------------------------------------------------------------
# MBV2 sys.path / cwd context (mirrors spaces/_common.py::self_check)
# ---------------------------------------------------------------------------


class _Mbv2Context:
    def __init__(self, mbv2_root: Path):
        self.mbv2_root = mbv2_root
        self._prev_cwd: str | None = None
        self._path_added = False

    def __enter__(self) -> Path:
        if not self.mbv2_root.is_dir():
            raise FileNotFoundError(f"ImageNet_MBV2 not found at {self.mbv2_root}")
        self._prev_cwd = os.getcwd()
        root_str = str(self.mbv2_root)
        self._path_added = root_str not in sys.path
        if self._path_added:
            sys.path.insert(0, root_str)
        os.chdir(self.mbv2_root)
        return self.mbv2_root

    def __exit__(self, *exc_info: object) -> None:
        if self._prev_cwd is not None:
            os.chdir(self._prev_cwd)
        if self._path_added:
            try:
                sys.path.remove(str(self.mbv2_root))
            except ValueError:
                pass


def mbv2_context(az_nas_root: Path | None = None) -> _Mbv2Context:
    """Context manager: chdir to `ImageNet_MBV2` and put it on `sys.path` for
    the duration of the `with` block (restores both on exit), so `import
    Masternet` / `from ZeroShotProxy import ...` resolve. Required before
    calling :func:`build_model_from_space` or :func:`compute_zero_shot_score`.
    """
    return _Mbv2Context(resolve_mbv2_root(az_nas_root))


# ---------------------------------------------------------------------------
# Real-batch bridging (grow's uv env -> this process, via disk, no torch pickle)
# ---------------------------------------------------------------------------


def load_real_batches(
    dataset_config: str,
    *,
    grow_root: Path,
    batch_size: int = 64,
    seed: int = 0,
    maxbatch: int = 2,
    num_workers: int = 0,
    device: "Any" = "cpu",
    uv_bin: str = "uv",
    allow_download: bool = False,
    timeout: float = 1800.0,
) -> tuple[list[list[Any]], dict[str, Any]]:
    """Collect up to `maxbatch` real training batches for `dataset_config` by
    shelling out to `_bridge_grow_batches.py` under grow's own `uv` env, then
    load the resulting `.npz`/JSON back into this process as plain
    `torch.Tensor`s already moved to `device` (both `x` and `y`, per the
    plan's `rand_input=False` contract -- gradient proxies need labels).

    Returns `(batches, meta)` where `batches` is a list of `[x, y]` pairs
    (mirrors the `trainbatches` shape stock `evolution_search_az.py` builds
    for `rand_input=False`) and `meta` is `grow_data.load`'s metadata dict
    (`dataset`, `num_classes`, `batch_shape`, `split_train_val`, `seed`,
    `transforms`, ...) plus `num_batches`/`status`.

    Raises `DataUnavailableError` (with `.meta` set) if the bridge reports
    `status="skipped_missing_data"` -- callers should treat this as a
    documented skip, not a failure. Raises `RuntimeError` for any other
    non-zero bridge exit (a real bug/config problem, not a data-availability
    skip).
    """
    import torch  # noqa: PLC0415 (lazy: only needed once batches are read back)

    with tempfile.TemporaryDirectory(prefix="az_nas_grow_batches_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            uv_bin,
            "run",
            "--project",
            str(grow_root),
            "python",
            str(_BRIDGE_SCRIPT),
            "--dataset_config",
            dataset_config,
            "--batch_size",
            str(batch_size),
            "--seed",
            str(seed),
            "--maxbatch",
            str(maxbatch),
            "--num_workers",
            str(num_workers),
            "--out",
            str(tmp_path),
        ]
        if allow_download:
            cmd.append("--allow_download")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        meta_path = tmp_path / "meta.json"
        if result.returncode != 0 or not meta_path.is_file():
            raise RuntimeError(
                f"grow_data bridge failed for dataset_config={dataset_config!r} "
                f"(exit={result.returncode}, cmd={' '.join(cmd)}):\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )

        meta: dict[str, Any] = json.loads(meta_path.read_text())
        if meta.get("status") == "skipped_missing_data":
            raise DataUnavailableError(
                f"{dataset_config}: {meta.get('reason', 'data unavailable')}",
                meta=meta,
            )
        if meta.get("status") != "ok":
            raise RuntimeError(
                f"grow_data bridge returned unexpected status for "
                f"{dataset_config!r}: {meta!r}"
            )

        npz_path = tmp_path / "batches.npz"
        with np.load(npz_path) as npz:
            num_batches = int(meta["num_batches"])
            batches = []
            for i in range(num_batches):
                x = torch.from_numpy(np.array(npz[f"x{i}"])).to(device)
                y = torch.from_numpy(np.array(npz[f"y{i}"])).to(device)
                batches.append([x, y])

    return batches, meta


def load_batch(
    dataset_config: str,
    *,
    batch_size: int = 64,
    seed: int = 0,
    num_workers: int = 0,
    device: Any = "cpu",
    grow_root: Path | None = None,
    uv_bin: str = "uv",
    allow_download: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return one real `(x, y, meta)` training batch for `dataset_config`
    (the single-batch convenience form `smoke_score.py` uses).

    Tries loading in-process first, in the *current* interpreter -- this
    works whenever every `grow_data.load` dependency for this particular
    `dataset_config` is already importable here (e.g. cifar10's
    torchvision-only path works directly in the MBV2 venv, per `SETUP.md`).
    Falls back to the grow-env subprocess bridge (:func:`load_real_batches`,
    `maxbatch=1`) on any exception from the in-process attempt -- in
    particular the five `gromo`-backed matrix datasets, whose Hydra
    `instantiate()` raises (wrapping a `ModuleNotFoundError` for `gromo`,
    which is deliberately not installed in the MBV2 venv; see this module's
    docstring) as soon as they're instantiated in-process.
    """
    root = grow_root if grow_root is not None else resolve_grow_root()
    try:
        _ensure_grow_on_syspath(root)
        import grow_data  # noqa: PLC0415

        train_loader, meta = grow_data.load(
            dataset_config,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            device=device,
        )
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        meta = {k: v for k, v in meta.items() if k not in ("val_loader", "test_loader")}
        return x, y, meta
    except Exception:  # noqa: BLE001 - broad on purpose: any in-process failure
        # (missing gromo wrapped in hydra.errors.InstantiationException, a
        # plain ImportError, etc.) means "try the grow-env bridge instead",
        # not "this dataset is unavailable" -- that determination happens
        # inside the bridge subprocess itself (DataUnavailableError).
        batches, meta = load_real_batches(
            dataset_config,
            grow_root=root,
            batch_size=batch_size,
            seed=seed,
            maxbatch=1,
            num_workers=num_workers,
            device=device,
            uv_bin=uv_bin,
            allow_download=allow_download,
        )
        x, y = batches[0]
        return x, y, meta


# ---------------------------------------------------------------------------
# Model building (needs MBV2 on sys.path -- call inside mbv2_context())
# ---------------------------------------------------------------------------


def build_model_from_space(
    space_cfg: Any, num_classes: int, *, seed: int | None = None
) -> Any:
    """Build a `MasterNet` from `space_cfg.init_plainnet_str`. Must be called
    inside :func:`mbv2_context`. `num_classes` must come from grow's Hydra
    `cfg.dataset_config.num_classes` (the "Replace getmisc()" contract in the
    plan), never from `space_cfg.num_classes_hint`.

    When ``seed`` is given, seeds numpy/torch/CUDA RNGs via
    :func:`seed_for_masternet` **before** constructing MasterNet (paper protocol).
    """
    import Masternet  # noqa: PLC0415 (MBV2-local; only valid inside mbv2_context)

    if seed is not None:
        seed_for_masternet(seed)

    return Masternet.MasterNet(
        num_classes=num_classes,
        plainnet_struct=space_cfg.init_plainnet_str,
        no_create=False,
    )


# ---------------------------------------------------------------------------
# Zero-shot scoring (needs MBV2 on sys.path -- call inside mbv2_context())
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def cpu_safe_cuda_patch(*, active: bool):
    """Temporarily no-op `torch.nn.Module.cuda` while `active`.

    `ZeroShotProxy.compute_az_nas_score.compute_nas_score` calls
    `model.cuda()` unconditionally (no CPU branch upstream -- see
    `ImageNet_MBV2/ZeroShotProxy/compute_az_nas_score.py`), which raises on
    a CPU-only torch build. Patching `nn.Module.cuda` to return `self`
    unchanged is a call-site-only workaround (never touches the AZ-NAS
    source tree, never used for real `--gpu` runs) and is restored
    immediately after the `with` block, even on exception. Mirrors
    `smoke_score.py`'s own `_cpu_safe_cuda_patch` (kept as a separate,
    independent implementation there; duplicated here rather than imported
    so neither script depends on the other's private helpers).
    """
    if not active:
        yield
        return
    import torch.nn as nn  # noqa: PLC0415

    original_cuda = nn.Module.cuda
    nn.Module.cuda = lambda self, device=None: self  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.cuda = original_cuda


def compute_zero_shot_score(
    model: Any,
    *,
    batches: list[list[Any]],
    resolution: int,
    batch_size: int,
    gpu: int | None = None,
    zero_shot_score: str = "az_nas",
) -> dict[str, float]:
    """Mirror `ImageNet_MBV2/evolution_search_az.py::compute_nas_score`,
    adapted to take an already-built model and a pre-collected list of real
    `[x, y]` batches (never `None` here -- this project's locked
    `rand_input=False`) instead of building the model from a structure
    string and reading a raw `trainloader` internally. Must be called inside
    :func:`mbv2_context` so `from ZeroShotProxy import ...` resolves.

    CUDA handling: if `gpu` is given, requires CUDA (raises `NoCudaError` if
    unavailable) and runs the real GPU path. If `gpu` is `None`: runs on GPU
    if CUDA happens to be visible (stock `model.cuda()` picks the current
    default device), else falls back to a CPU-safe path via
    :func:`cpu_safe_cuda_patch` (stock `compute_nas_score` otherwise has no
    CPU branch at all).

    Raises `FloatingPointError` if any returned score component is
    NaN/+-inf (matches the plan's "NaN/FloatingPointError: log and
    continue" contract -- callers should catch this per-dataset).
    """
    if zero_shot_score.lower() != "az_nas":
        raise NotImplementedError(
            f"zero_shot_score={zero_shot_score!r} not implemented; only 'az_nas' "
            "is wired up (matches the plan's smoke/matrix scope)."
        )

    import torch  # noqa: PLC0415

    cuda_available = torch.cuda.is_available()
    if gpu is not None:
        if not cuda_available:
            raise NoCudaError(
                f"--gpu {gpu} was requested but torch.cuda.is_available() is False "
                "on this host; omit --gpu to run the CPU-safe path."
            )
        torch.cuda.set_device(gpu)
    use_cpu_patch = not cuda_available

    from ZeroShotProxy import compute_az_nas_score  # noqa: PLC0415

    with cpu_safe_cuda_patch(active=use_cpu_patch):
        info: dict[str, float] = compute_az_nas_score.compute_nas_score(
            model=model,
            gpu=gpu,
            trainloader=batches,
            resolution=resolution,
            batch_size=batch_size,
        )

    non_finite = {k: v for k, v in info.items() if not np.isfinite(v)}
    if non_finite:
        raise FloatingPointError(
            f"non-finite zero-shot score component(s) {non_finite!r} in full info={info!r}"
        )
    return info


# ---------------------------------------------------------------------------
# Result schema (matches experiments/AZ-NAS/result_schema.md)
# ---------------------------------------------------------------------------


def build_result_record(
    *,
    dataset: str,
    seed: int,
    meta: dict[str, Any],
    rand_input: bool,
    zero_shot_score: str,
    ranking_key: str,
    info: dict[str, float],
    structure_str: str,
    grow_root: Path,
    az_nas_root: Path,
    skip_latency: bool,
    device: str,
    extra: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one result dict matching the minimum JSON contract in
    `experiments/AZ-NAS/result_schema.md`. `extra` may add adapter-specific
    fields (e.g. `run_score_matrix.py`'s `matrix_rank_sum`) beyond the
    required set -- the schema explicitly allows this.

    Dual SHA+dirty fields prefer a job-start provenance freeze
    (``provenance`` / ``provenance_path`` / ``AZ_NAS_PROVENANCE_FILE``) over
    live :func:`git_sha_and_dirty`. Always records ``torch_version``,
    ``torch_cuda_version``, and ``gpu_name`` (paper-mode extras locus).
    """
    grow_sha, grow_dirty, az_sha, az_dirty = resolve_dual_sha_dirty(
        grow_root,
        az_nas_root,
        provenance=provenance,
        provenance_path=provenance_path,
    )
    record: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "batch_shape": list(meta["batch_shape"]),
        "num_classes": int(meta["num_classes"]),
        "split_train_val": float(meta["split_train_val"]),
        "rand_input": bool(rand_input),
        "transforms": meta.get("transforms", "standard"),
        "zero_shot_score": zero_shot_score,
        "ranking_key": ranking_key,
        "info": info,
        "structure_str": structure_str,
        "experimental_grow_sha": grow_sha,
        "experimental_grow_dirty": grow_dirty,
        "az_nas_sha": az_sha,
        "az_nas_dirty": az_dirty,
        "skip_latency": skip_latency,
        "device": device,
    }
    record.update(torch_device_extras(device=device))
    if extra:
        record.update(extra)
    return record


def write_result_json(
    record: dict[str, Any], results_dir: Path, *, filename: str | None = None
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"{record['dataset']}_seed{record['seed']}.json"
    out_path = results_dir / name
    out_path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Population-based ranking aggregate (documented `ranking_key` formula)
# ---------------------------------------------------------------------------

AZ_NAS_RANK_SUM_KEY = "az_nas_rank_sum"

AZ_NAS_RANK_SUM_DOC = (
    "az_nas_rank_sum: for the datasets successfully scored in one "
    "run_score_matrix.py invocation, treat that set as a population "
    "(mirroring stock evolution_search_az.py's own per-iteration "
    "combination: for each key in info, rank values with "
    "scipy.stats.rankdata -- higher raw value = higher rank -- then sum "
    "log(rank / population_size) across keys). Only comparable *within* "
    "one matrix run (same population, same seed); not a global "
    "architecture-quality constant, and not meaningful across families "
    "with different input resolutions/channel counts. Degenerates to 0 "
    "for every entry when fewer than 2 datasets were successfully scored "
    "(rankdata over 1 element always yields rank=1, so log(1/1)=0)."
)


def compute_matrix_rank_sum(records: list[dict[str, Any]]) -> dict[str, float]:
    """Population-based `az_nas_rank_sum` aggregate across `records`
    (dataset -> result record with an `info` dict), keyed by `dataset`. See
    `AZ_NAS_RANK_SUM_DOC` for the exact formula and its caveats.
    """
    from scipy import stats  # noqa: PLC0415 (MBV2-venv dependency, lazy)

    if not records:
        return {}

    keys = sorted({k for r in records for k in r["info"].keys()})
    n = len(records)
    total = np.zeros(n)
    for key in keys:
        values = np.array([r["info"].get(key, np.nan) for r in records], dtype=float)
        ranks = stats.rankdata(values)
        total = total + np.log(ranks / n)
    return {r["dataset"]: float(total[i]) for i, r in enumerate(records)}


def as_json_default(obj: Any) -> Any:
    """`json.dumps(..., default=as_json_default)` helper for numpy scalars."""
    if isinstance(obj, np.generic):
        return obj.item()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
