#!/usr/bin/env python
"""NB201 / NATS-TSS AZ-NAS zero-cost ranking on grow P0 batches.

Primary search path (Phase 1). Uses **NB201** ``ZeroShotProxy`` +
``extract_cell_features`` + TinyNetwork only — never MBV2
``run_score_matrix`` / PlainNet.

Run from ``AZ-NAS/adapters`` with a torch env that can import NB201
(``.venv-mbv2`` is fine) and grow's ``uv`` available for the batch bridge:

    export EXPERIMENTAL_GROW_ROOT=...
    ../.venv-mbv2/bin/python nb201_search_az.py \\
        --dataset_config multnist --n_samples 8 --gpu 0

Gates
-----
* Default ``--n_samples`` is small (smoke). Full TSS (15 625) is **refused**
  unless ``--allow_full_tss`` is set (still ask a human + timed-smoke GPU-h
  estimate before cluster submit).
* Figshare / NATS downloads: bridge refuses missing zips unless
  ``--allow_download`` (ask before mass download).

Resume
------
Periodic scored-index shards under ``--shard_dir``; resume skips already
scored cell indices. Local JSON remains source of truth; W&B is online
observability (option A).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import adapter_utils
import nb201_common as nb


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_config", required=True)
    p.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of random TSS cells to score (default: 8 smoke). "
        f"Use --n_samples {nb.TSS_FULL_SIZE} with --allow_full_tss for full space.",
    )
    p.add_argument(
        "--allow_full_tss",
        action="store_true",
        help="Required to score the full NATS-TSS (15625) or n_samples>=15625. "
        "Do not set without human approval + timed-smoke GPU-h estimate.",
    )
    p.add_argument(
        "--full_tss",
        action="store_true",
        help="Score the canonical full TSS enumeration (implies n_samples=15625; "
        "still requires --allow_full_tss).",
    )
    p.add_argument("--seed", type=int, default=nb.NB201_SEED)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Figshare zip download if missing (ask before using).",
    )
    p.add_argument("--channel", type=int, default=nb.NB201_C)
    p.add_argument("--num_cells", type=int, default=nb.NB201_N)
    p.add_argument("--max_nodes", type=int, default=nb.NB201_MAX_NODES)
    p.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="Stem in_channels override (default: from batch). Pilot RGB=3; "
        "gutenberg=1 / chesseract=12 for later phases.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Final ranked JSON path (default: results/nb201_search_<ds>_seed<s>.json).",
    )
    p.add_argument(
        "--shard_dir",
        type=Path,
        default=None,
        help="Directory for periodic scored shards (default: alongside --out).",
    )
    p.add_argument("--shard_every", type=int, default=25)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed cell index in shard_dir.",
    )
    p.add_argument(
        "--timed_smoke",
        action="store_true",
        help="Print cells/sec after scoring (for GPU-h preflight estimate).",
    )
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--provenance-file", type=Path, default=None)
    p.add_argument(
        "--uv_bin",
        default=None,
        help="uv binary for grow batch bridge (default: UV_BIN env or 'uv').",
    )
    p.add_argument(
        "--include_random_control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tag one mid/low-ranked cell as selection_role=random for train.",
    )
    return p.parse_args(argv)


def _refuse_full_if_needed(args: argparse.Namespace) -> None:
    n = nb.TSS_FULL_SIZE if args.full_tss else int(args.n_samples)
    if n >= nb.TSS_FULL_SIZE and not args.allow_full_tss:
        raise SystemExit(
            f"Refusing full TSS scoring (n_samples={n} >= {nb.TSS_FULL_SIZE}) "
            "without --allow_full_tss. Run a small --n_samples smoke / timed_smoke "
            "first, write GPU-h estimate to preflight_gpu_estimate.md, and get "
            "explicit user approval before full-space jobs."
        )


def _shard_path(shard_dir: Path, dataset: str, seed: int) -> Path:
    return shard_dir / f"nb201_search_{dataset}_seed{seed}_shards.jsonl"


def _load_scored_indices(shard_path: Path) -> dict[int, dict[str, Any]]:
    scored: dict[int, dict[str, Any]] = {}
    if not shard_path.is_file():
        return scored
    with shard_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scored[int(row["cell_index"])] = row
    return scored


def _append_shard(shard_path: Path, row: dict[str, Any]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _refuse_full_if_needed(args)

    az_root = adapter_utils.resolve_az_nas_root()
    grow_root = adapter_utils.resolve_grow_root()
    results_dir = adapter_utils.resolve_results_dir(grow_root)
    uv_bin = args.uv_bin or __import__("os").environ.get("UV_BIN", "uv")

    out_path = args.out or (
        results_dir / f"nb201_search_{args.dataset_config}_seed{args.seed}.json"
    )
    shard_dir = args.shard_dir or (out_path.parent / "nb201_shards")
    shard_path = _shard_path(shard_dir, args.dataset_config, args.seed)

    import torch  # noqa: PLC0415

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise adapter_utils.NoCudaError(
                f"--gpu {args.gpu} requested but CUDA is unavailable"
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        gpu = args.gpu
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gpu = 0 if device.type == "cuda" else None

    # Real grow batches (score pad) via bridge — never MBV2 path.
    try:
        batches, meta = adapter_utils.load_real_batches(
            args.dataset_config,
            grow_root=grow_root,
            batch_size=args.batch_size,
            seed=args.seed,
            maxbatch=1,
            num_workers=args.num_workers,
            device=device,
            uv_bin=uv_bin,
            allow_download=args.allow_download,
        )
    except adapter_utils.DataUnavailableError as exc:
        print(f"[nb201_search] skipped_missing_data: {exc}", flush=True)
        skip = {
            "status": "skipped_missing_data",
            "dataset": args.dataset_config,
            "seed": args.seed,
            "reason": str(exc),
            "meta": getattr(exc, "meta", {}),
        }
        nb.write_json(out_path, skip)
        return 0

    # Cache one eval batch (list so next(iter(...)) is stable).
    cached_batches = [batches[0]]
    batch_shape = tuple(int(x) for x in meta["batch_shape"])
    in_channels = int(args.in_channels or batch_shape[0])
    resolution = int(batch_shape[1])
    num_classes = int(meta["num_classes"])
    expected = nb.SCORE_PAD_EXPECTATIONS.get(args.dataset_config)
    if expected is not None and batch_shape != expected:
        print(
            f"[nb201_search] warning: batch_shape={batch_shape} != expected "
            f"score pad {expected} for {args.dataset_config}",
            flush=True,
        )

    rng = random.Random(args.seed)
    with nb.nb201_context(az_root):
        if args.full_tss:
            archs = nb.generate_all_archs(args.max_nodes)
            assert len(archs) == nb.TSS_FULL_SIZE, (
                f"TSS enumeration size {len(archs)} != {nb.TSS_FULL_SIZE}"
            )
            work: list[tuple[int, Any]] = list(enumerate(archs))
        else:
            work = []
            for i in range(int(args.n_samples)):
                arch = nb.random_genotype(args.max_nodes, nb.tss_op_names(), rng=rng)
                work.append((i, arch))

        scored = _load_scored_indices(shard_path) if args.resume else {}
        if scored and not args.resume:
            scored = {}

        wandb_config = {
            "dataset": args.dataset_config,
            "seed": args.seed,
            "n_samples": len(work),
            "full_tss": bool(args.full_tss),
            "C": args.channel,
            "N": args.num_cells,
            "max_nodes": args.max_nodes,
            "batch_shape": list(batch_shape),
            "in_channels": in_channels,
            "ranking_key": nb.RANKING_KEY,
        }
        run_name = f"nb201/{args.dataset_config}/search"

        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        n_new = 0

        with nb.maybe_wandb(
            enabled=args.wandb,
            run_name=run_name,
            config=wandb_config,
            tags=["nb201", "search", args.dataset_config],
        ) as wb:
            for cell_index, arch in work:
                if cell_index in scored:
                    rows.append(scored[cell_index])
                    continue

                arch_str = nb.genotype_to_str(arch)
                network = nb.build_tiny_network(
                    arch,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    C=args.channel,
                    N=args.num_cells,
                )
                network = network.to(device)
                network.train()

                info = nb.score_one_arch(
                    network,
                    gpu=gpu,
                    cached_batches=cached_batches,
                    resolution=resolution,
                    batch_size=args.batch_size,
                )
                try:
                    flops, params = nb.measure_flops_params(network, batch_shape)
                except Exception:
                    flops, params = float("-inf"), float("-inf")
                info["complexity"] = float(flops) if np.isfinite(flops) else float("-inf")

                row = {
                    "cell_index": cell_index,
                    "genotype": arch_str,
                    "genotype_short": nb.genotype_short(arch_str),
                    "info": info,
                    "flops_m": flops if np.isfinite(flops) else None,
                    "params_mb": params if np.isfinite(params) else None,
                }
                rows.append(row)
                _append_shard(shard_path, row)
                scored[cell_index] = row
                n_new += 1

                del network
                _cleanup_cuda()

                if args.shard_every > 0 and n_new % args.shard_every == 0:
                    print(
                        f"[nb201_search] shard checkpoint: {len(scored)}/{len(work)} "
                        f"cells scored ({shard_path})",
                        flush=True,
                    )

            # Rank population (including resumed rows).
            infos = [r["info"] for r in rows]
            rank_sums = nb.compute_az_nas_rank_sum(infos)
            for row, rs in zip(rows, rank_sums):
                row["az_nas_rank_sum"] = rs

            rows_sorted = sorted(rows, key=lambda r: r["az_nas_rank_sum"], reverse=True)
            for rank_i, row in enumerate(rows_sorted, start=1):
                row["rank"] = rank_i

            if args.include_random_control and rows_sorted:
                # Pick a non-top-3 cell when possible as the random control.
                pick = rows_sorted[min(len(rows_sorted) - 1, max(3, len(rows_sorted) // 2))]
                pick["selection_role"] = "random"

            elapsed = time.perf_counter() - t0
            # Throughput over newly scored cells only (resume-safe).
            throughput_n = n_new if n_new > 0 else len(rows)
            cells_per_sec = (throughput_n / elapsed) if elapsed > 0 else 0.0

            if args.timed_smoke:
                print(
                    f"[timed_smoke] dataset={args.dataset_config} "
                    f"cells_scored={throughput_n} elapsed_s={elapsed:.3f} "
                    f"cells_per_sec={cells_per_sec:.4f} "
                    f"sec_per_cell={(elapsed / throughput_n) if throughput_n else float('nan'):.4f}",
                    flush=True,
                )

            provenance = adapter_utils.load_provenance_freeze(args.provenance_file)
            grow_sha, grow_dirty, az_sha, az_dirty = adapter_utils.resolve_dual_sha_dirty(
                grow_root,
                az_root,
                provenance=provenance,
                provenance_path=args.provenance_file,
            )

            record: dict[str, Any] = {
                "status": "ok",
                "dataset": args.dataset_config,
                "seed": args.seed,
                "batch_shape": list(batch_shape),
                "num_classes": num_classes,
                "split_train_val": float(meta.get("split_train_val", 0.05)),
                "rand_input": False,
                "transforms": "standard",
                "pad_for_proxy": True,
                "zero_shot_score": "az_nas",
                "ranking_key": nb.RANKING_KEY,
                "ranking_key_doc": nb.RANKING_KEY_DOC,
                "structure_str": "nb201_tss",
                "info": {},
                "experimental_grow_sha": grow_sha,
                "experimental_grow_dirty": grow_dirty,
                "az_nas_sha": az_sha,
                "az_nas_dirty": az_dirty,
                "skip_latency": True,
                "device": str(device),
                "macro": {
                    "C": args.channel,
                    "N": args.num_cells,
                    "max_nodes": args.max_nodes,
                    "in_channels": in_channels,
                },
                "n_scored": len(rows_sorted),
                "full_tss": bool(args.full_tss),
                "allow_full_tss": bool(args.allow_full_tss),
                "shard_path": str(shard_path),
                "search_wall_s": elapsed,
                "cells_per_sec": cells_per_sec,
                "train_run_policy": nb.TRAIN_RUN_POLICY_DOC,
                "architectures": rows_sorted,
            }
            record.update(adapter_utils.torch_device_extras(device=str(device)))
            nb.write_json(out_path, record)
            print(f"[nb201_search] wrote {out_path} ({len(rows_sorted)} archs)", flush=True)

            if wb is not None:
                try:
                    wb.log(
                        {
                            "search/n_scored": len(rows_sorted),
                            "search/wall_s": elapsed,
                            "search/cells_per_sec": cells_per_sec,
                            "search/best_rank_sum": rows_sorted[0]["az_nas_rank_sum"]
                            if rows_sorted
                            else None,
                        }
                    )
                    wb.save(str(out_path), policy="now")
                except Exception as exc:  # noqa: BLE001
                    print(f"[nb201_search] wandb log warning: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
