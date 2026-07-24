"""AZ-NAS zero-shot proxy scoring across the five grow "matrix" datasets.

Implements the `matrix-runner` todo from
`.cursor/plans/az-nas_grow_adapters_0b7a938b.plan.md`: for each of
`multnist`, `cifartile`, `geoclassing`, `gutenberg`, `chesseract`, build the
dataset's shrunk-space family's named initial `plainnet` structure (see
`spaces/`), score it with `ZeroShotProxy.compute_az_nas_score` against real
batches (`rand_input=False`, `skip_latency=True`, locked decisions), and
persist one result JSON per dataset under
`experimental_grow/experiments/AZ-NAS/results/` matching
`result_schema.md`.

Run inside the MBV2 venv (needs `torch`/`Masternet`/`ZeroShotProxy`):

    cd AZ-NAS/adapters
    ../.venv-mbv2/bin/python run_score_matrix.py --gpu 0
    # or, via the grow-side launcher (activates env var plumbing for you):
    #   experiments/AZ-NAS/launchers/score_matrix.sh

By default this never downloads data and never crashes when data is missing
or CUDA is unavailable: each dataset independently ends up `ok`,
`skipped_missing_data`, or (only if a specific `--gpu` was requested but
isn't visible) `skipped_no_cuda`. Without `--gpu`, scoring runs on GPU if
CUDA happens to be visible, else falls back to a CPU-safe path (stock
`compute_az_nas_score.compute_nas_score` has no CPU branch of its own; see
`adapter_utils.cpu_safe_cuda_patch`). A `FloatingPointError`/non-finite
score for one dataset is logged and does not abort the rest of the matrix
(locked decision -- "NaN/FloatingPointError: log and continue").

CLI:

    --dataset_configs   comma-separated list (default: the five matrix
                        datasets above)
    --seed              default 0
    --batch_size        default 64 (stock AZ-NAS default)
    --maxbatch          default 2  (stock AZ-NAS default; only the first
                        collected batch is actually read by
                        compute_az_nas_score, but multiple are collected for
                        parity with stock `rand_input=False` runs and future
                        multi-batch proxies)
    --gpu               CUDA device index (default: None -> cuda:0 if
                        available, else a CPU-safe fallback path -- see
                        `adapter_utils.cpu_safe_cuda_patch`). Passing an
                        explicit `--gpu` requires that device to actually be
                        visible (raises if not).
    --rand_input        must stay False (default) for this "our dataset"
                        matrix runner; passing --rand_input true raises
                        immediately (locked decision, see the plan).
    --allow_download    let the grow-env bridge fetch a missing dataset's
                        Figshare zip. Off by default -- never download large
                        datasets by surprise.
    --results_dir       override the output directory (default:
                        experimental_grow/experiments/AZ-NAS/results/)
    --grow_root / --az_nas_root  override sibling-path resolution (default:
                        EXPERIMENTAL_GROW_ROOT env var / sibling directory)
    --uv_bin            `uv` binary used for the grow-env bridge subprocess
                        (default: "uv")
    --dry_run           skip every dataset unconditionally (no subprocess,
                        no model build) -- useful for exercising the CLI /
                        result-writing plumbing without any data or CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import adapter_utils
from spaces import get_space

MATRIX_DATASETS: tuple[str, ...] = (
    "multnist",
    "cifartile",
    "geoclassing",
    "gutenberg",
    "chesseract",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the five grow matrix datasets with AZ-NAS's az_nas proxy."
    )
    parser.add_argument(
        "--dataset_configs",
        default=",".join(MATRIX_DATASETS),
        help="Comma-separated grow dataset_config names (default: the five matrix datasets).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--maxbatch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--rand_input",
        type=lambda v: v.lower() in ("1", "true", "yes", "y", "t"),
        default=False,
        help="Must stay False for this matrix runner (locked decision); True raises.",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow the grow-env bridge to fetch a missing dataset's Figshare zip.",
    )
    parser.add_argument("--zero_shot_score", default="az_nas")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--grow_root", default=None)
    parser.add_argument("--az_nas_root", default=None)
    parser.add_argument("--uv_bin", default="uv")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip every dataset unconditionally; exercises CLI/result plumbing only.",
    )
    return parser.parse_args()


def _score_one_dataset(
    dataset_config: str,
    *,
    args: argparse.Namespace,
    grow_root: Path,
    az_nas_root: Path,
    cuda_available: bool,
) -> dict[str, Any]:
    """Run the full per-dataset pipeline; always returns a status dict, never raises
    (all failure modes -- missing data, missing CUDA, non-finite scores, unexpected
    errors -- are caught here and turned into a `status` field per the plan's
    "NaN/FloatingPointError: log and continue" contract).
    """
    entry: dict[str, Any] = {"dataset": dataset_config}

    try:
        space_cfg = get_space(dataset_config)
    except KeyError as exc:
        entry["status"] = "skipped_unknown_family"
        entry["message"] = str(exc)
        return entry

    if args.dry_run:
        entry["status"] = "skipped_dry_run"
        entry["message"] = "--dry_run passed; no subprocess/model build attempted."
        return entry

    try:
        batches, meta = adapter_utils.load_real_batches(
            dataset_config,
            grow_root=grow_root,
            batch_size=args.batch_size,
            seed=args.seed,
            maxbatch=args.maxbatch,
            num_workers=args.num_workers,
            device="cpu",
            uv_bin=args.uv_bin,
            allow_download=args.allow_download,
        )
    except adapter_utils.DataUnavailableError as exc:
        entry["status"] = "skipped_missing_data"
        entry["message"] = str(exc)
        entry["expected_zip"] = exc.meta.get("expected_zip")
        return entry
    except Exception as exc:  # noqa: BLE001 - report and continue, never abort the matrix
        entry["status"] = "error_loading_data"
        entry["message"] = f"{exc.__class__.__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
        return entry

    if args.gpu is not None:
        device_str = f"cuda:{args.gpu}"
    else:
        device_str = "cuda:0" if cuda_available else "cpu"

    try:
        with adapter_utils.mbv2_context(az_nas_root):
            import torch  # noqa: PLC0415

            device = torch.device(device_str)
            batches_on_device = [[x.to(device), y.to(device)] for x, y in batches]
            model = adapter_utils.build_model_from_space(space_cfg, meta["num_classes"])
            model = model.to(device)

            info = adapter_utils.compute_zero_shot_score(
                model,
                batches=batches_on_device,
                resolution=space_cfg.input_image_size,
                batch_size=meta["batch_size"],
                gpu=args.gpu,
                zero_shot_score=args.zero_shot_score,
            )
    except adapter_utils.NoCudaError as exc:
        entry["status"] = "skipped_no_cuda"
        entry["message"] = str(exc)
        return entry
    except FloatingPointError as exc:
        entry["status"] = "skipped_nan_score"
        entry["message"] = str(exc)
        return entry
    except Exception as exc:  # noqa: BLE001 - report and continue, never abort the matrix
        entry["status"] = "error_scoring"
        entry["message"] = f"{exc.__class__.__name__}: {exc}"
        if isinstance(exc, RuntimeError) and "cannot be multiplied" in str(exc):
            entry["message"] += (
                " (known upstream limitation: compute_az_nas_score.py's trainability "
                "score assumes every adjacent layer-feature resolution ratio is an "
                "exact power-of-2 pixel_unshuffle stride -- e.g. rgb28/gray_sq's final "
                "7->3 / 6->3 floor-division stage isn't exact. This is a stock "
                "AZ-NAS bug at small/non-power-of-2 resolutions, not specific to "
                "run_score_matrix.py; not patched here per the plan's no-vendoring "
                "rule. rgb64/board12 families are unaffected.)"
            )
        traceback.print_exc(file=sys.stderr)
        return entry

    record = adapter_utils.build_result_record(
        dataset=dataset_config,
        seed=args.seed,
        meta=meta,
        rand_input=False,
        zero_shot_score=args.zero_shot_score,
        ranking_key=adapter_utils.AZ_NAS_RANK_SUM_KEY,
        info=info,
        structure_str=space_cfg.init_plainnet_str,
        grow_root=grow_root,
        az_nas_root=az_nas_root,
        skip_latency=True,
        device=device_str,
        extra={"family": space_cfg.family},
    )
    entry["status"] = "ok"
    entry["record"] = record
    return entry


def main() -> int:
    args = _parse_args()

    if args.rand_input:
        raise AssertionError(
            "--rand_input true is not allowed for run_score_matrix.py: this runner "
            "makes 'our dataset' claims, which the plan locks to rand_input=False "
            "(real batches, labels included). Use the (not-yet-implemented) "
            "evolution-search entry point for random-input exploration instead."
        )

    dataset_configs = [d.strip() for d in args.dataset_configs.split(",") if d.strip()]
    if not dataset_configs:
        raise SystemExit("--dataset_configs produced an empty list")

    az_nas_root = (
        Path(args.az_nas_root).expanduser().resolve()
        if args.az_nas_root
        else adapter_utils.resolve_az_nas_root()
    )
    grow_root = (
        Path(args.grow_root).expanduser().resolve()
        if args.grow_root
        else adapter_utils.resolve_grow_root()
    )
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else adapter_utils.resolve_results_dir(grow_root)
    )

    cuda_available = False
    if not args.dry_run:
        with adapter_utils.mbv2_context(az_nas_root):
            import torch  # noqa: PLC0415

            cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print(
                "[warn] No CUDA device visible on this host. "
                "compute_az_nas_score.compute_nas_score() unconditionally calls "
                "model.cuda(); scoring below runs via a call-site-only CPU patch "
                "(adapter_utils.cpu_safe_cuda_patch, mirrors smoke_score.py's own "
                "workaround) instead of the real GPU path. See "
                "AZ-NAS/adapters/SETUP.md 'Blockers' for the CUDA assumption this "
                "documents around.",
                file=sys.stderr,
            )

    print(f"EXPERIMENTAL_GROW_ROOT={grow_root}")
    print(f"AZ_NAS_ROOT={az_nas_root}")
    print(f"results_dir={results_dir}")
    print(f"dataset_configs={dataset_configs}")

    entries: list[dict[str, Any]] = []
    for dataset_config in dataset_configs:
        entry = _score_one_dataset(
            dataset_config,
            args=args,
            grow_root=grow_root,
            az_nas_root=az_nas_root,
            cuda_available=cuda_available,
        )
        entries.append(entry)
        status = entry["status"]
        print(f"[{status}] {dataset_config}: {entry.get('message', '')}".rstrip(": "))

    ok_entries = [e for e in entries if e["status"] == "ok"]
    ok_records = [e["record"] for e in ok_entries]
    rank_sums = adapter_utils.compute_matrix_rank_sum(ok_records)

    written_paths: list[Path] = []
    for entry in ok_entries:
        record = entry["record"]
        record["matrix_rank_sum"] = rank_sums[record["dataset"]]
        record["matrix_rank_sum_population"] = sorted(rank_sums.keys())
        path = adapter_utils.write_result_json(record, results_dir)
        written_paths.append(path)
        print(f"wrote {path}")

    summary = {
        "seed": args.seed,
        "zero_shot_score": args.zero_shot_score,
        "ranking_key": adapter_utils.AZ_NAS_RANK_SUM_KEY,
        "ranking_key_doc": adapter_utils.AZ_NAS_RANK_SUM_DOC,
        "results": sorted(
            (
                {
                    "dataset": e["dataset"],
                    "status": e["status"],
                    "message": e.get("message"),
                    "matrix_rank_sum": rank_sums.get(e["dataset"]),
                    "info": e.get("record", {}).get("info"),
                }
                for e in entries
            ),
            key=lambda r: (r["matrix_rank_sum"] is None, -(r["matrix_rank_sum"] or 0.0)),
        ),
    }
    summary_path = results_dir / f"matrix_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")

    num_ok = len(ok_entries)
    num_total = len(entries)
    print(f"done: {num_ok}/{num_total} dataset(s) scored ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
