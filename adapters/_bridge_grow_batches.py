#!/usr/bin/env python
"""Grow-env side of the `adapter_utils.load_real_batches` bridge.

Run this script with grow's own interpreter (`uv run --project
<experimental_grow> python _bridge_grow_batches.py ...`), never with the
AZ-NAS MBV2 venv's Python. It is the *only* place that imports `grow_data`
directly for the five `gromo`-backed matrix datasets (multnist, cifartile,
geoclassing, gutenberg, chesseract) -- see `adapter_utils.py`'s module
docstring and `SETUP.md`'s `grow-data-adapter` section for why this must be
a separate process from the MBV2-venv one that builds `MasterNet`.

Safety gate (locked requirement -- never download large datasets by
surprise): `tools.datasets.NpyWebDataset.__init__` downloads its Figshare
zip unconditionally whenever the zip isn't already on disk, regardless of
any `download=` flag passed through Hydra. Before ever calling
`grow_data.load()` for a `tools.datasets.*`-backed config, this script
composes just enough of the Hydra config to read the resolved `root` and
`_target_`, and refuses to proceed (writing
`{"status": "skipped_missing_data", ...}` to `meta.json` and exiting 0)
unless the expected zip already exists on disk or `--allow_download` is
passed.

Output contract (written to `--out`, a directory):

- `meta.json`: JSON dict. Always has a `status` key
  (`"ok"` | `"skipped_missing_data"`). On `"ok"`, also has every key from
  `grow_data.load`'s `meta` dict (minus the non-serializable `val_loader`/
  `test_loader` DataLoader objects) plus `num_batches`.
- `batches.npz` (only written on `"ok"`): `x0`, `y0`, `x1`, `y1`, ... one
  `(x, y)` pair per collected batch, as plain `float32`/int numpy arrays
  (never `torch.save`/pickle -- the reading side may be on a different
  torch version, see `adapter_utils.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grow_data  # noqa: E402


def _resolve_target_and_root(
    dataset_config: str, *, seed: int, num_workers: int
) -> tuple[str, str]:
    """Compose just the Hydra config (no dataset instantiation, no I/O
    beyond reading YAML) to find the resolved dataset `_target_` and `root`,
    so the download gate below never has to guess a path.
    """
    root = grow_data._resolve_grow_root()
    cfg = grow_data._compose_dataset_cfg(
        root, dataset_config, seed=seed, num_workers=num_workers
    )
    dataset_cfg = cfg.dataset_config
    target = str(dataset_cfg.dataset.get("_target_", ""))
    data_root = str(dataset_cfg.dataset.get("root", dataset_cfg.get("path", "")))
    return target, data_root


def _npy_webdataset_zip_path(target: str, data_root: str) -> Path | None:
    """For `tools.datasets.<Class>` (`NpyWebDataset` subclass) targets,
    return the on-disk zip path that gates the unconditional download (see
    `NpyWebDataset._download_and_extract`: it only skips the HTTP GET when
    this exact file already exists). Returns `None` for other targets (e.g.
    cifar10's `torchvision.datasets.CIFAR10`, which isn't handled by this
    gate -- it honors its own `download=` flag cleanly, per `SETUP.md`).
    """
    if not target.startswith("tools.datasets."):
        return None
    class_name = target.rsplit(".", 1)[-1]
    return Path(data_root).expanduser() / f"{class_name}.zip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxbatch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help=(
            "Allow NpyWebDataset's always-on download-if-missing behavior for "
            "this dataset_config. Off by default."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target, data_root = _resolve_target_and_root(
        args.dataset_config, seed=args.seed, num_workers=args.num_workers
    )
    zip_path = _npy_webdataset_zip_path(target, data_root)
    if zip_path is not None and not zip_path.exists() and not args.allow_download:
        meta_out = {
            "status": "skipped_missing_data",
            "dataset_config": args.dataset_config,
            "target": target,
            "expected_zip": str(zip_path),
            "reason": (
                f"{zip_path} does not exist; refusing to call grow_data.load() "
                "because tools.datasets.NpyWebDataset downloads its Figshare zip "
                "unconditionally once instantiated, regardless of any download= "
                "flag (see experimental_grow/tools/datasets.py). Pass "
                "--allow_download once you intend to fetch it, or place the zip "
                "at the path above."
            ),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
        print(f"[skip] {args.dataset_config}: {meta_out['reason']}")
        return 0

    train_loader, meta = grow_data.load(
        args.dataset_config,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i, (x, y) in enumerate(train_loader):
        if i >= args.maxbatch:
            break
        xs.append(x.detach().cpu().numpy().astype("float32"))
        ys.append(y.detach().cpu().numpy())

    if not xs:
        raise RuntimeError(
            f"{args.dataset_config}: train_loader produced zero batches "
            f"(maxbatch={args.maxbatch})"
        )

    npz_payload: dict[str, np.ndarray] = {}
    for i, (x_arr, y_arr) in enumerate(zip(xs, ys)):
        npz_payload[f"x{i}"] = x_arr
        npz_payload[f"y{i}"] = y_arr
    np.savez(out_dir / "batches.npz", **npz_payload)

    meta_out: dict[str, Any] = {
        k: v for k, v in meta.items() if k not in ("val_loader", "test_loader")
    }
    meta_out["batch_shape"] = list(meta_out["batch_shape"])
    meta_out["image_shape"] = list(meta_out["image_shape"])
    meta_out["num_batches"] = len(xs)
    meta_out["status"] = "ok"
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[ok] {args.dataset_config}: wrote {len(xs)} batches to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
