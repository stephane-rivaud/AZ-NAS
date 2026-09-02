#!/usr/bin/env python
"""Train selected NB201 TinyNetwork cells under grow's 200-epoch matrix recipe.

Locked recipe: ``sgd_nesterov`` (lr=0.1, momentum=0.9, nesterov) +
``linear5warmup_cosine195``, ``nb_step=200``. CE on **logits** (forward returns
``features, logits``). Native grow shapes by default; datasets in
``nb201_common.TRAIN_SCORE_PAD_DATASETS`` (gutenberg) reuse the search score
pad so residual reductions stay even. Augmented train loaders where the
dataset YAML defines ``transforms.augmented``.

Run inside grow's ``uv`` env (needs ``gromo`` for P0 datasets) with NB201 on
``sys.path`` via this script's context:

    export EXPERIMENTAL_GROW_ROOT=...
    export AZ_NAS_ROOT=...
    cd "$AZ_NAS_ROOT/adapters"
    uv run --project "$EXPERIMENTAL_GROW_ROOT" python nb201_train_selected.py \\
        --dataset_config multnist \\
        --search_json .../nb201_search_multnist_seed0.json \\
        --policy locked6 \\
        --gpu 0

Or train one genotype:

    uv run --project "$EXPERIMENTAL_GROW_ROOT" python nb201_train_selected.py \\
        --dataset_config cifartile --genotype '...' --seed 0 --epochs 2 --dry_run

Selection policy (locked): top-1 × 3 seeds; top-2 × 1; top-3 × 1; random × 1
→ **6 runs / dataset**. Epoch checkpoints + best-val + resume; W&B online
option A; local JSON remains SoT.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import adapter_utils
import grow_data
import nb201_common as nb


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_config", required=True)
    p.add_argument(
        "--search_json",
        type=Path,
        default=None,
        help="Ranked search artifact from nb201_search_az.py.",
    )
    p.add_argument(
        "--policy",
        choices=("locked6", "top-k", "single"),
        default="single",
        help="locked6 = 6 runs/dataset protocol; top-k = top-k × seeds; "
        "single = one genotype/seed.",
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--genotype", type=str, default=None, help="Genotype string (single).")
    p.add_argument("--seed", type=int, default=nb.NB201_SEED)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for top-1 (locked6 default: 0,1,2).",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--channel", type=int, default=nb.NB201_C)
    p.add_argument("--num_cells", type=int, default=nb.NB201_N)
    p.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="Stem in_channels (default: from native batch). Pilot RGB=3.",
    )
    p.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Figshare zip download if missing (ask before using).",
    )
    p.add_argument(
        "--ckpt_dir",
        type=Path,
        default=None,
        help="Checkpoint root (default: results/nb201_ckpts/<dataset>/...).",
    )
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--provenance-file", type=Path, default=None)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Build model + one batch forward/backward only; no full train.",
    )
    p.add_argument(
        "--timed_smoke",
        action="store_true",
        help="Train a few epochs and print epochs/hour estimate.",
    )
    p.add_argument(
        "--timed_smoke_epochs",
        type=int,
        default=2,
        help="Epochs to run under --timed_smoke (default 2).",
    )
    return p.parse_args(argv)


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    if args.policy == "locked6":
        return [0, 1, 2]
    return [int(args.seed)]


def _jobs_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.policy == "single":
        if not args.genotype and not args.search_json:
            raise SystemExit("single policy needs --genotype or --search_json")
        if args.genotype:
            arch = {
                "genotype": args.genotype,
                "genotype_short": nb.genotype_short(args.genotype),
                "rank": None,
            }
        else:
            payload = nb.load_json(args.search_json)
            archs = payload.get("architectures") or []
            if not archs:
                raise SystemExit(f"no architectures in {args.search_json}")
            arch = archs[0]
        return [{"role": "single", "rank": arch.get("rank"), "seed": args.seed, "arch": arch}]

    if args.search_json is None:
        raise SystemExit(f"--policy {args.policy} requires --search_json")
    payload = nb.load_json(args.search_json)
    archs = payload.get("architectures") or []
    if not archs:
        raise SystemExit(f"no architectures in {args.search_json}")

    if args.policy == "locked6":
        return nb.expand_train_jobs(
            archs,
            top_k=args.top_k,
            include_random=True,
            top1_seeds=_parse_seeds(args),
            other_seed=int(args.seed),
            random_seed=int(args.seed),
        )

    # top-k: each of top_k × each seed in --seeds / --seed
    seeds = _parse_seeds(args)
    jobs: list[dict[str, Any]] = []
    for k in range(min(args.top_k, len(archs))):
        for seed in seeds:
            jobs.append(
                {
                    "role": f"top-{k + 1}",
                    "rank": k + 1,
                    "seed": int(seed),
                    "arch": archs[k],
                }
            )
    return jobs


def _accuracy(logits, targets) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item())


def _evaluate(model, loader, device, criterion) -> tuple[float, float]:
    import torch  # noqa: PLC0415

    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _feat, logits = model(x)
            loss = criterion(logits, y)
            bs = int(y.size(0))
            total_loss += float(loss.item()) * bs
            total_correct += float((logits.argmax(dim=1) == y).float().sum().item())
            total_n += bs
    if total_n == 0:
        return float("nan"), float("nan")
    return total_loss / total_n, total_correct / total_n


def _build_optimizer_scheduler(model, args: argparse.Namespace):
    import torch  # noqa: PLC0415

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    warmup = int(args.warmup_epochs)
    cosine_epochs = max(int(args.epochs) - warmup, 1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0e-6,
                end_factor=1.0,
                total_iters=warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=1.0e-6,
            ),
        ],
        milestones=[warmup],
    )
    return optimizer, scheduler


def _ckpt_paths(ckpt_dir: Path) -> dict[str, Path]:
    return {
        "last": ckpt_dir / "last.pt",
        "best": ckpt_dir / "best.pt",
    }


def _save_ckpt(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    meta: dict[str, Any],
) -> None:
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "meta": meta,
        },
        path,
    )


def _gate_download(dataset_config: str, grow_root: Path, allow_download: bool) -> None:
    """Refuse missing NpyWebDataset zips unless --allow_download (ask first)."""
    import grow_data as gd  # noqa: PLC0415

    cfg = gd._compose_dataset_cfg(grow_root, dataset_config, seed=0, num_workers=0)
    dataset_cfg = cfg.dataset_config
    target = str(dataset_cfg.dataset.get("_target_", ""))
    if not target.startswith("tools.datasets."):
        return
    class_name = target.rsplit(".", 1)[-1]
    data_root = str(dataset_cfg.dataset.get("root", dataset_cfg.get("path", "")))
    zip_path = Path(data_root).expanduser() / f"{class_name}.zip"
    if zip_path.is_file():
        return
    if allow_download:
        print(f"[nb201_train] --allow_download: missing {zip_path}; grow may fetch", flush=True)
        return
    raise SystemExit(
        f"skipped_missing_data: {zip_path} not on disk. Pass --allow_download "
        "only after explicit approval for Figshare download."
    )


def train_one_job(
    job: dict[str, Any],
    *,
    args: argparse.Namespace,
    grow_root: Path,
    az_root: Path,
    results_dir: Path,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    arch = job["arch"]
    genotype_str = arch["genotype"]
    genotype_short = arch.get("genotype_short") or nb.genotype_short(genotype_str)
    seed = int(job["seed"])
    role = job["role"]

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise adapter_utils.NoCudaError(
                f"--gpu {args.gpu} requested but CUDA is unavailable"
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _gate_download(args.dataset_config, grow_root, args.allow_download)

    # Native shapes by default; gutenberg (and any TRAIN_SCORE_PAD_DATASETS)
    # reuse search pad so ResNetBasicblock stride-2 residuals match.
    # Augmented train where YAML defines it.
    pad_for_proxy = nb.train_uses_score_pad(args.dataset_config)
    train_loader, meta = grow_data.load(
        args.dataset_config,
        batch_size=args.batch_size,
        seed=seed,
        num_workers=args.num_workers,
        device=device,
        grow_root=grow_root,
        pad_for_proxy=pad_for_proxy,
        train_transforms="auto",
        download=True if args.allow_download else None,
    )
    val_loader = meta.get("val_loader")
    test_loader = meta.get("test_loader")
    batch_shape = tuple(int(x) for x in meta["batch_shape"])
    in_channels = int(args.in_channels or meta["in_channels"])
    num_classes = int(meta["num_classes"])

    out_dir = args.out_dir or results_dir
    if args.ckpt_dir is not None:
        ckpt_root = (
            args.ckpt_dir
            / args.dataset_config
            / genotype_short
            / f"seed{seed}"
        )
    else:
        ckpt_root = (
            results_dir
            / "nb201_ckpts"
            / args.dataset_config
            / genotype_short
            / f"seed{seed}"
        )
    paths = _ckpt_paths(ckpt_root)
    result_path = (
        out_dir
        / f"nb201_train_{args.dataset_config}_{genotype_short}_seed{seed}.json"
    )

    epochs = int(args.epochs)
    if args.timed_smoke:
        epochs = min(epochs, int(args.timed_smoke_epochs))
    if args.dry_run:
        epochs = 0

    with nb.nb201_context(az_root):
        genotype = nb.parse_genotype(genotype_str)
        model = nb.build_tiny_network(
            genotype,
            num_classes=num_classes,
            in_channels=in_channels,
            C=args.channel,
            N=args.num_cells,
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = _build_optimizer_scheduler(model, args)

        start_epoch = 0
        best_val_acc = -math.inf
        if args.resume and paths["last"].is_file():
            ckpt = torch.load(paths["last"], map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc = float(ckpt.get("best_val_acc", -math.inf))
            print(
                f"[nb201_train] resume {genotype_short} seed{seed} "
                f"from epoch {start_epoch}",
                flush=True,
            )

        if args.dry_run:
            model.train()
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            _feat, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            record = {
                "status": "dry_run_ok",
                "dataset": args.dataset_config,
                "seed": seed,
                "role": role,
                "genotype": genotype_str,
                "genotype_short": genotype_short,
                "batch_shape": list(batch_shape),
                "in_channels": in_channels,
                "loss": float(loss.item()),
                "acc_batch": _accuracy(logits.detach(), y),
                "train_run_policy": nb.TRAIN_RUN_POLICY_DOC,
            }
            nb.write_json(result_path, record)
            return record

        run_name = f"nb201/{args.dataset_config}/train/{genotype_short}/seed{seed}"
        wandb_config = {
            "dataset": args.dataset_config,
            "seed": seed,
            "role": role,
            "genotype": genotype_str,
            "genotype_short": genotype_short,
            "epochs": epochs,
            "batch_shape": list(batch_shape),
            "in_channels": in_channels,
            "optimizer": "sgd_nesterov",
            "lr_scheduler": "linear5warmup_cosine195",
            "nb_step": 200,
        }

        t0 = time.perf_counter()
        history: list[dict[str, Any]] = []

        with nb.maybe_wandb(
            enabled=args.wandb,
            run_name=run_name,
            config=wandb_config,
            tags=["nb201", "train", args.dataset_config, role],
        ) as wb:
            for epoch in range(start_epoch, epochs):
                model.train()
                running_loss = 0.0
                running_correct = 0.0
                running_n = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    _feat, logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    bs = int(y.size(0))
                    running_loss += float(loss.item()) * bs
                    running_correct += float(
                        (logits.argmax(dim=1) == y).float().sum().item()
                    )
                    running_n += bs
                scheduler.step()

                train_loss = running_loss / max(running_n, 1)
                train_acc = running_correct / max(running_n, 1)
                val_loss, val_acc = (
                    _evaluate(model, val_loader, device, criterion)
                    if val_loader is not None
                    else (float("nan"), float("nan"))
                )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                if wb is not None:
                    try:
                        wb.log(
                            {
                                "train/loss": train_loss,
                                "train/acc": train_acc,
                                "val/loss": val_loss,
                                "val/acc": val_acc,
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "epoch": epoch,
                            }
                        )
                    except Exception:
                        pass

                meta_ckpt = {
                    "dataset": args.dataset_config,
                    "genotype": genotype_str,
                    "seed": seed,
                    "role": role,
                }
                _save_ckpt(
                    paths["last"],
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    meta=meta_ckpt,
                )
                if val_acc == val_acc and val_acc > best_val_acc:  # not NaN
                    best_val_acc = val_acc
                    _save_ckpt(
                        paths["best"],
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_val_acc=best_val_acc,
                        meta=meta_ckpt,
                    )

            # Load best-val for final test when available.
            if paths["best"].is_file():
                best = torch.load(paths["best"], map_location=device)
                model.load_state_dict(best["model"])
                best_val_acc = float(best.get("best_val_acc", best_val_acc))

            test_loss, test_acc = (
                _evaluate(model, test_loader, device, criterion)
                if test_loader is not None
                else (float("nan"), float("nan"))
            )
            elapsed = time.perf_counter() - t0
            epochs_done = max(epochs - start_epoch, 1)
            epochs_per_hour = (epochs_done / elapsed) * 3600.0 if elapsed > 0 else 0.0

            if args.timed_smoke:
                print(
                    f"[timed_smoke] dataset={args.dataset_config} "
                    f"genotype={genotype_short} seed={seed} "
                    f"epochs={epochs_done} elapsed_s={elapsed:.3f} "
                    f"epochs_per_hour={epochs_per_hour:.4f} "
                    f"min_per_epoch={(elapsed / epochs_done) / 60.0:.4f}",
                    flush=True,
                )

            try:
                flops, params = nb.measure_flops_params(model, batch_shape)
            except Exception:
                flops, params = float("nan"), float("nan")

            provenance = adapter_utils.load_provenance_freeze(args.provenance_file)
            grow_sha, grow_dirty, az_sha, az_dirty = adapter_utils.resolve_dual_sha_dirty(
                grow_root,
                az_root,
                provenance=provenance,
                provenance_path=args.provenance_file,
            )

            record = {
                "status": "ok",
                "dataset": args.dataset_config,
                "seed": seed,
                "role": role,
                "rank": job.get("rank"),
                "genotype": genotype_str,
                "genotype_short": genotype_short,
                "batch_shape": list(batch_shape),
                "num_classes": num_classes,
                "in_channels": in_channels,
                "pad_for_proxy": pad_for_proxy,
                "transforms": meta.get("transforms"),
                "epochs": epochs,
                "optimizer": "sgd_nesterov",
                "lr_scheduler": "linear5warmup_cosine195",
                "nb_step": 200,
                "best_val_acc": best_val_acc if best_val_acc != -math.inf else None,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "flops_m": flops,
                "params_mb": params,
                "train_wall_s": elapsed,
                "epochs_per_hour": epochs_per_hour,
                "ckpt_dir": str(ckpt_root),
                "train_run_policy": nb.TRAIN_RUN_POLICY_DOC,
                "history": history,
                "experimental_grow_sha": grow_sha,
                "experimental_grow_dirty": grow_dirty,
                "az_nas_sha": az_sha,
                "az_nas_dirty": az_dirty,
                "device": str(device),
                "zero_shot_score": "az_nas",
                "ranking_key": nb.RANKING_KEY,
                "structure_str": genotype_str,
                "rand_input": False,
                "split_train_val": float(meta.get("split_train_val", 0.05)),
                "info": {},
                "skip_latency": True,
            }
            record.update(adapter_utils.torch_device_extras(device=str(device)))
            nb.write_json(result_path, record)
            print(f"[nb201_train] wrote {result_path}", flush=True)

            if wb is not None:
                try:
                    wb.log(
                        {
                            "test/acc": test_acc,
                            "test/loss": test_loss,
                            "best_val_acc": best_val_acc,
                            "train/wall_s": elapsed,
                            "train/epochs_per_hour": epochs_per_hour,
                        }
                    )
                    wb.save(str(result_path), policy="now")
                except Exception as exc:  # noqa: BLE001
                    print(f"[nb201_train] wandb log warning: {exc}", flush=True)

        return record


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    az_root = adapter_utils.resolve_az_nas_root()
    grow_root = adapter_utils.resolve_grow_root()
    results_dir = adapter_utils.resolve_results_dir(grow_root)

    jobs = _jobs_from_args(args)
    print(
        f"[nb201_train] {len(jobs)} job(s); policy={args.policy}; "
        f"{nb.TRAIN_RUN_POLICY_DOC}",
        flush=True,
    )
    summaries = []
    for job in jobs:
        summaries.append(
            train_one_job(
                job,
                args=args,
                grow_root=grow_root,
                az_root=az_root,
                results_dir=results_dir,
            )
        )

    summary_path = (
        (args.out_dir or results_dir)
        / f"nb201_train_summary_{args.dataset_config}_seed{args.seed}.json"
    )
    nb.write_json(
        summary_path,
        {
            "dataset": args.dataset_config,
            "policy": args.policy,
            "train_run_policy": nb.TRAIN_RUN_POLICY_DOC,
            "n_jobs": len(summaries),
            "jobs": [
                {
                    "role": s.get("role"),
                    "seed": s.get("seed"),
                    "genotype_short": s.get("genotype_short"),
                    "status": s.get("status"),
                    "test_acc": s.get("test_acc"),
                    "best_val_acc": s.get("best_val_acc"),
                }
                for s in summaries
            ],
        },
    )
    print(f"[nb201_train] summary {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
