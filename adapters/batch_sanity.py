"""One-batch sanity check for ``grow_data.load`` across the smoke + P0 datasets.

Step 3.5 of the plan: for cifar10 (smoke) plus the five P0 matrix datasets
(multnist, cifartile, geoclassing, gutenberg, chesseract), pull one training
batch through ``grow_data.load`` and check shape/dtype/labels/NaN.

This never triggers a network download: it first checks whether grow's data
root (``$HOME/datasets`` by default, overridable via ``GROW_DATASETS_ROOT``)
exists at all, and skips every dataset with a clear message if it doesn't.
Individual datasets are still allowed to fail independently (e.g. only some
have been prefetched) -- one dataset's missing/broken data does not abort the
rest of the run.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import grow_data

# cifar10 first (smoke order), then the five P0 / matrix datasets (one per
# shrunk-space family: rgb28, rgb64, rgb60, gray_sq, board12 -- see the plan's
# search-space table).
DATASETS: tuple[str, ...] = (
    "cifar10",
    "multnist",
    "cifartile",
    "geoclassing",
    "gutenberg",
    "chesseract",
)


def _data_root() -> Path:
    override = os.environ.get("GROW_DATASETS_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "datasets"


def _check_one(dataset_config: str) -> tuple[bool, str]:
    try:
        train_loader, meta = grow_data.load(dataset_config, batch_size=8)
        x, y = next(iter(train_loader))
        grow_data.assert_batch_sane(x, y, dataset_name=dataset_config)
        return True, (
            f"ok  batch_shape={meta['batch_shape']} num_classes={meta['num_classes']} "
            f"x.dtype={x.dtype} y.shape={tuple(y.shape)}"
        )
    except FileNotFoundError as exc:
        return False, f"skip (data missing): {exc}"
    except Exception as exc:  # noqa: BLE001 - report and continue, never abort the loop
        # Hydra wraps dataset-constructor errors in InstantiationException; a
        # missing/corrupted-dataset message from the underlying constructor
        # (e.g. torchvision's "Dataset not found or corrupted") is still a
        # graceful skip, not an adapter bug.
        if "not found" in str(exc).lower():
            return False, f"skip (data missing): {exc}"
        return False, f"error: {exc.__class__.__name__}: {exc}"


def main() -> int:
    root = _data_root()
    if not root.is_dir():
        print(
            f"[skip] Data root {root} does not exist; skipping batch sanity for all "
            f"{len(DATASETS)} datasets ({', '.join(DATASETS)}). Set GROW_DATASETS_ROOT "
            "or populate $HOME/datasets to run this check for real."
        )
        return 0

    any_failed = False
    for name in DATASETS:
        ok, message = _check_one(name)
        status = "[ok]  " if ok else "[skip]"
        print(f"{status} {name}: {message}")
        if not ok and "skip (data missing)" not in message:
            any_failed = True
            traceback.print_exc(file=sys.stderr)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
