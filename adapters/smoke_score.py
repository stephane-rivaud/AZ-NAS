"""AZ-NAS proxy zero-shot smoke score for one `experimental_grow` dataset.

Run inside the MBV2 venv (`AZ-NAS/.venv-mbv2`), from `AZ-NAS/adapters/`:

    cd /Users/strivaud/Projects/research/AZ-NAS/adapters
    ../.venv-mbv2/bin/python smoke_score.py --dataset_config cifar10 --seed 0
    ../.venv-mbv2/bin/python smoke_score.py --dataset_config multnist --seed 0

Algorithm (per the plan's locked `smoke_score.py` contract):

1. Resolve paths (grow root / AZ-NAS root / `ImageNet_MBV2` root) and real
   git provenance -- via `adapter_utils`, the module shared with
   `run_score_matrix.py` (see that module's docstring for why it exists).
2. `adapter_utils.load_real_batches(dataset_config, ...)` -> up to
   `--maxbatch` real training batches + metadata, via grow's Hydra
   `get_dataloaders` (`transforms="standard"`), bridged from grow's own `uv`
   environment (needed for the five `gromo`-backed P0 datasets -- multnist,
   cifartile, geoclassing, gutenberg, chesseract -- which cannot import in
   the MBV2 venv; see `adapter_utils.py`/`_bridge_grow_batches.py`).
3. Build a `MasterNet` from the dataset's family `SpaceConfig.init_plainnet_str`
   (`adapters/spaces/get_space`), via `adapter_utils.build_model_from_space`
   inside `adapter_utils.mbv2_context()` (chdir + `sys.path` for
   `Masternet`/`PlainNet`/`ZeroShotProxy`).
4. `ZeroShotProxy.compute_az_nas_score.compute_nas_score(..., rand_input=False)`
   with real `datax, datay` both moved to the target device first.
   - `--gpu <n>`: delegates to `adapter_utils.compute_zero_shot_score`
     (identical code path to `run_score_matrix.py`).
   - No `--gpu` (default, CPU): the plan's smoke-score contract explicitly
     requires a CPU path ("--gpu optional (CPU OK)"), so this script -- and
     only this script, not `adapter_utils.py` -- temporarily monkeypatches
     `nn.Module.cuda` to a no-op for the duration of the call. Stock
     `compute_nas_score` calls `model.cuda()` unconditionally with no CPU
     branch, which otherwise crashes any CPU-only torch build outright; the
     patch is call-site-only, restored immediately after (even on
     exception), and never touches the AZ-NAS source tree.
     `run_score_matrix.py`/`adapter_utils.compute_zero_shot_score` deliberately
     do *not* do this (they raise `NoCudaError` instead) since that runner
     targets real GPU hosts; smoke-testing on a CPU dev machine is this
     script's specific, narrower use case.
5. Persist the full multi-key `info` dict (never a single invented scalar)
   as JSON under `experiments/AZ-NAS/results/`, via
   `adapter_utils.build_result_record`/`write_result_json`, matching
   `experiments/AZ-NAS/result_schema.md`'s field contract exactly (as
   `smoke_<dataset>_seed<seed>.json`, distinct from `run_score_matrix.py`'s
   `<dataset>_seed<seed>.json`, so the two entry points never clobber each
   other's output for datasets they both cover, e.g. multnist).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

import adapter_utils
from spaces import get_space

# See module docstring: AZ-NAS's own population-level "az_nas" aggregate
# (evolution_search_az.py) rank-sums expressivity/progressivity/trainability
# across a *population* of candidate structures scored on the same dataset;
# there is no single-network "az_nas" scalar inside `compute_nas_score`'s
# returned `info` dict. Documented verbatim per result_schema.md's
# "ranking_key must be spelled out, not left implicit" requirement.
RANKING_KEY_DOC = (
    "not a single-scalar formula for a lone network: AZ-NAS's own 'az_nas' "
    "aggregate (evolution_search_az.py) rank-sums expressivity+progressivity"
    "+trainability across a *population* of candidate structures scored on "
    "the same dataset; this smoke run scores exactly one structure, so no "
    "population exists to rank against. See info.{expressivity,"
    "progressivity,trainability,complexity} for the raw per-run values. "
    "(run_score_matrix.py's multi-dataset runs use "
    f"'{adapter_utils.AZ_NAS_RANK_SUM_KEY}' instead -- see that module.)"
)


@contextlib.contextmanager
def _cpu_safe_cuda_patch(*, active: bool):
    """Temporarily no-op `torch.nn.Module.cuda` while `active`.

    See the module docstring's step 4 for why this script (uniquely, not
    `adapter_utils.py`) needs this. Never activated for `--gpu` runs.
    """
    if not active:
        yield
        return
    import torch.nn as nn

    original_cuda = nn.Module.cuda
    nn.Module.cuda = lambda self, device=None: self  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.cuda = original_cuda


def _assert_finite(info: dict[str, float], *, dataset_config: str) -> None:
    """Mirror `adapter_utils.compute_zero_shot_score`'s non-finite check for
    the CPU path (which bypasses that function -- see module docstring).
    """
    non_finite = {k: v for k, v in info.items() if not (v == v and abs(v) != float("inf"))}
    if non_finite:
        raise FloatingPointError(
            f"{dataset_config}: non-finite zero-shot score component(s) "
            f"{non_finite!r} in full info={info!r}"
        )


def run_smoke_score(
    dataset_config: str,
    *,
    seed: int = 0,
    batch_size: int = 64,
    maxbatch: int = 1,
    rand_input: bool = False,
    gpu: int | None = None,
    grow_root: Path | None = None,
    az_nas_root: Path | None = None,
    allow_download: bool = False,
    uv_bin: str = "uv",
    provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one AZ-NAS proxy smoke score and return the full result record.

    `rand_input=False` is the only supported mode for scoring an actual
    `experimental_grow` dataset (the plan's locked contract): passing
    `rand_input=True` here always raises, since AZ-NAS's own `--rand_input`
    (random-noise scoring) is a different, unrelated code path
    (`trainloader=None` in `evolution_search_az.py`) that this adapter does
    not wire up -- there would be nothing dataset-specific left to smoke.
    """
    assert not rand_input, (
        "rand_input=True is not supported by smoke_score.py: the plan locks "
        "rand_input=False for every 'our dataset' scoring claim. Use AZ-NAS's "
        "own --rand_input flag directly (e.g. evolution_search_az.py) if you "
        "specifically want a random-noise-input score."
    )

    az_nas_root = az_nas_root or adapter_utils.resolve_az_nas_root()
    grow_root = grow_root or adapter_utils.resolve_grow_root()
    provenance = adapter_utils.load_provenance_freeze(provenance_path)
    space_cfg = get_space(dataset_config)

    use_cuda = gpu is not None
    if use_cuda:
        assert torch.cuda.is_available(), (
            f"--gpu {gpu} was requested but torch.cuda.is_available() is False "
            "on this host; omit --gpu to run the CPU-safe path."
        )
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # Locked contract: both features and labels move to the target device,
    # even though `compute_az_nas_score.compute_nas_score` itself only
    # consumes `datax` -- matches `evolution_search_az.py`'s own
    # `datax, datay = batch[0].cuda(), batch[1].cuda()` real-input assembly.
    batches, meta = adapter_utils.load_real_batches(
        dataset_config,
        grow_root=grow_root,
        batch_size=batch_size,
        seed=seed,
        maxbatch=maxbatch,
        device=device,
        uv_bin=uv_bin,
        allow_download=allow_download,
    )
    num_classes = int(meta["num_classes"])
    batch_size_used = int(batches[0][0].shape[0])

    start = time.time()
    with adapter_utils.mbv2_context(az_nas_root):
        net = adapter_utils.build_model_from_space(space_cfg, num_classes, seed=seed)

        if use_cuda:
            net = net.to(device)
            info = adapter_utils.compute_zero_shot_score(
                net,
                batches=batches,
                resolution=int(meta["input_image_size"]),
                batch_size=batch_size_used,
                gpu=gpu,
                zero_shot_score="az_nas",
            )
        else:
            from ZeroShotProxy import compute_az_nas_score

            with _cpu_safe_cuda_patch(active=True):
                info = compute_az_nas_score.compute_nas_score(
                    model=net,
                    gpu=None,
                    trainloader=batches,
                    resolution=int(meta["input_image_size"]),
                    batch_size=batch_size_used,
                )
            _assert_finite(info, dataset_config=dataset_config)
    elapsed = time.time() - start

    record = adapter_utils.build_result_record(
        dataset=meta["dataset"],
        seed=seed,
        meta=meta,
        rand_input=False,
        zero_shot_score="az_nas",
        ranking_key=RANKING_KEY_DOC,
        info=info,
        structure_str=space_cfg.init_plainnet_str,
        grow_root=grow_root,
        az_nas_root=az_nas_root,
        skip_latency=True,
        device=str(device),
        provenance=provenance,
        provenance_path=provenance_path,
        extra={
            "space_family": space_cfg.family,
            "batch_size_used": batch_size_used,
            "maxbatch": maxbatch,
            "elapsed_seconds": elapsed,
            "python_version": platform.python_version(),
        },
    )
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_config",
        required=False,
        default=None,
        help="grow Hydra dataset_config name, e.g. cifar10 (required unless --paper-ready)",
    )
    parser.add_argument(
        "--rand_input",
        type=lambda v: v.lower() in ("1", "true", "yes", "y", "t"),
        default=False,
        help="must stay False (asserted below) for any 'our dataset' claim",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--maxbatch", type=int, default=1, help="number of real batches to collect")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index; omit for CPU (default)")
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="let the grow-env bridge fetch a missing dataset's Figshare zip (off by default)",
    )
    parser.add_argument("--uv_bin", default="uv")
    parser.add_argument("--grow_root", default=None)
    parser.add_argument("--az_nas_root", default=None)
    parser.add_argument(
        "--results_dir",
        default=None,
        help="override output directory (default: experimental_grow/experiments/AZ-NAS/results/)",
    )
    parser.add_argument(
        "--provenance-file",
        default=None,
        dest="provenance_file",
        help="job-start provenance.json freeze (or set AZ_NAS_PROVENANCE_FILE)",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="run machine-checkable paper-mode probe (seed+extras+provenance+scipy) and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.paper_ready:
        adapter_utils.assert_paper_ready()
        print("[smoke_score] paper-ready: ok")
        return 0
    if not args.dataset_config:
        raise SystemExit("--dataset_config is required unless --paper-ready")
    if args.rand_input:
        raise AssertionError(
            "--rand_input true is not allowed for smoke_score.py (locked decision, "
            "see the plan): this script only ever makes 'our dataset' claims."
        )

    grow_root = Path(args.grow_root).expanduser().resolve() if args.grow_root else None
    az_nas_root = Path(args.az_nas_root).expanduser().resolve() if args.az_nas_root else None

    try:
        record = run_smoke_score(
            args.dataset_config,
            seed=args.seed,
            batch_size=args.batch_size,
            maxbatch=args.maxbatch,
            rand_input=args.rand_input,
            gpu=args.gpu,
            grow_root=grow_root,
            az_nas_root=az_nas_root,
            allow_download=args.allow_download,
            uv_bin=args.uv_bin,
            provenance_path=args.provenance_file,
        )
    except adapter_utils.DataUnavailableError as exc:
        # Missing data: exit 0, write no JSON (matrix summary records skips).
        print(f"[skip] {args.dataset_config}: {exc}", file=sys.stderr)
        return 0

    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else adapter_utils.resolve_results_dir(grow_root or adapter_utils.resolve_grow_root())
    )
    out_path = adapter_utils.write_result_json(
        record, results_dir, filename=f"smoke_{record['dataset']}_seed{record['seed']}.json"
    )
    print(
        f"[smoke_score] dataset={record['dataset']} seed={record['seed']} "
        f"device={record['device']} elapsed={record['elapsed_seconds']:.2f}s"
    )
    print(f"[smoke_score] info={json.dumps(record['info'])}")
    print(f"[smoke_score] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
