"""Bridge from AZ-NAS adapters to ``experimental_grow``'s Hydra dataloaders.

This module never imports anything from the AZ-NAS tree (one-way import
direction, see the plan). It resolves the sibling ``experimental_grow``
checkout, composes a Hydra ``dataset_config`` exactly like
``hydra_script/train_and_grow.py`` does, and returns plain
``torch.utils.data.DataLoader``/metadata objects that the (separately
implemented) ``smoke_score.py`` / ``run_score_matrix.py`` scripts consume.

Locked contract (see ``.cursor/plans/az-nas_grow_adapters_0b7a938b.plan.md``):

- Path resolution: ``EXPERIMENTAL_GROW_ROOT`` env var, else sibling of the
  AZ-NAS repo root (``<AZ-NAS-repo-root>/../experimental_grow``).
- Always ``get_dataloaders`` with ``transforms="standard"`` for every split
  (train/val/test) -- never ``create_dataloaders`` / the ``"augmented"``
  pipeline, so proxy scoring never sees training-time augmentation.
- Gutenberg and multnist are padded to a square ``32x32`` in the adapter's
  own transform chain (not in grow's YAML) -- **not** their native shapes
  (``(1, 27, 18)`` / ``(3, 28, 28)``) -- when ``pad_for_proxy=True`` (default,
  zero-cost search). GeoClassing is padded to ``64x64`` from its measured
  native ``(3, 60, 60)``. This is the AZ-NAS proxy geometry policy (see
  ``adapters/spaces/README.md``): AZ-NAS's ``compute_az_nas_score``
  trainability term relies on ``nn.PixelUnshuffle``, which needs even
  stride-2 ratios. Full 200-ep train usually passes ``pad_for_proxy=False``
  (native shapes) and ``train_transforms="auto"`` (augmented when YAML
  defines it). **Exception:** gutenberg train also pads → ``32x32`` (same as
  search) because NB201 ``ResNetBasicblock`` stride-2 residuals disagree on
  odd heights (Conv3×3 pad=1 → 14 vs AvgPool2d → 13 for H=27). Native shapes
  on disk are unchanged; only the adapter-facing transform pipeline pads them.
- The first training batch is asserted to be ``BCHW`` float with labels and
  no NaN/Inf before being handed back to the caller.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

# AZ-NAS proxy geometry policy (see module docstring): pad selected datasets
# to the next PixelUnshuffle-safe power-of-two square -- not their native
# shapes -- so every stride-2 downsample AZ-NAS's trainability term needs
# stays evenly divisible. torchvision.transforms.Pad's padding order is
# (left, top, right, bottom).
_PAD_TARGET_SIZE = 32  # gutenberg / multnist
_GEOCLASSING_PAD_TARGET_SIZE = 64

# Gutenberg native standard-pipeline output is (1, 27, 18): width needs
# +14 (18 -> 32, split 7/7), height needs +5 (27 -> 32, split 2/3 since the
# remainder is odd).
_GUTENBERG_PAD_LTRB: tuple[int, int, int, int] = (7, 2, 7, 3)

# Multnist native standard-pipeline output is (3, 28, 28): both dimensions
# need +4 (28 -> 32), split evenly 2/2.
_MULTNIST_PAD_LTRB: tuple[int, int, int, int] = (2, 2, 2, 2)

# GeoClassing measured native is (3, 60, 60): both dimensions need +4
# (60 -> 64), split evenly 2/2.
_GEOCLASSING_PAD_LTRB: tuple[int, int, int, int] = (2, 2, 2, 2)

# dataset_config.name -> expected (H, W) after adapter pad (proxy only).
_PADDED_SQUARE_EXPECTATIONS: dict[str, int] = {
    "gutenberg": _PAD_TARGET_SIZE,
    "multnist": _PAD_TARGET_SIZE,
    "geoclassing": _GEOCLASSING_PAD_TARGET_SIZE,
}


def _resolve_grow_root() -> Path:
    """Resolve the ``experimental_grow`` repo root.

    Order: ``EXPERIMENTAL_GROW_ROOT`` if set, else documented worktree paths
    (Cursor-native / cluster ``aznas-compare`` / local legacy), else a main
    ``experimental_grow`` sibling of an AZ-NAS *main* clone. Never silently
    picks ``AZ-NAS.worktrees/experimental_grow`` (wrong sibling footgun).
    Mirrors ``adapter_utils.resolve_grow_root``.
    """
    env_root = os.environ.get("EXPERIMENTAL_GROW_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "hydra_script" / "configs").is_dir():
            raise FileNotFoundError(
                f"EXPERIMENTAL_GROW_ROOT={root!s} lacks hydra_script/configs. "
                "Set it to a real experimental_grow checkout "
                "(cluster: $HOME/experimental_grow.worktrees/aznas-compare)."
            )
        return root

    home = Path.home()
    candidates = [
        home / ".cursor" / "worktrees" / "experimental_grow" / "aznas",
        home / "experimental_grow.worktrees" / "aznas-compare",
        home / "Projects" / "research" / "experimental_grow.worktrees" / "aznas-compare",
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "hydra_script" / "configs").is_dir():
            return cand.resolve()

    az_nas_root = Path(__file__).resolve().parents[1]
    az_parent = az_nas_root.parent
    if az_parent.name == "AZ-NAS.worktrees" or "AZ-NAS.worktrees" in az_parent.parts:
        raise FileNotFoundError(
            "Cannot resolve experimental_grow from an AZ-NAS worktree without "
            "EXPERIMENTAL_GROW_ROOT. Refusing silent sibling "
            f"{az_parent / 'experimental_grow'!s}. Set EXPERIMENTAL_GROW_ROOT to "
            "$HOME/experimental_grow.worktrees/aznas-compare (cluster) or "
            "~/.cursor/worktrees/experimental_grow/aznas (local Cursor)."
        )

    sibling = az_parent / "experimental_grow"
    if (sibling / "hydra_script" / "configs").is_dir():
        return sibling.resolve()

    raise FileNotFoundError(
        f"Cannot find an experimental_grow checkout at {sibling!s} "
        "(expected hydra_script/configs under it). Set EXPERIMENTAL_GROW_ROOT "
        "to override the resolved path."
    )


def _ensure_grow_on_syspath(grow_root: Path) -> None:
    """Prepend the grow root to ``sys.path`` so ``hydra_script``/``tools`` import."""
    root_str = str(grow_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _compose_dataset_cfg(
    grow_root: Path,
    dataset_config: str,
    *,
    seed: int,
    num_workers: int,
) -> DictConfig:
    """Compose grow's main Hydra config with ``dataset_config`` overridden.

    Uses ``initialize_config_dir`` (a context manager) so ``GlobalHydra`` is
    cleared on exit and repeated calls (e.g. one per dataset in a batch-sanity
    loop) don't collide.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(grow_root / "hydra_script" / "configs")
    overrides = [
        f"dataset_config={dataset_config}",
        f"general.seed={seed}",
        f"general.num_workers={num_workers}",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    # compose() locks the tree in struct mode; we need to rewrite
    # dataset_config.transforms.standard for gutenberg below.
    OmegaConf.set_struct(cfg, False)
    return cfg


def _apply_square_pad(dataset_cfg: DictConfig, pad_ltrb: tuple[int, int, int, int]) -> None:
    """Rewrite the ``standard`` transform pipeline to append a fixed square pad.

    AZ-NAS proxy geometry policy (see module docstring): gutenberg/multnist
    must only ever be scored at the padded square 32x32, and geoclassing at
    padded 64x64, never their native shapes. This mutates only the composed
    in-memory config for this run, never the checked-in YAML -- native data
    on disk is untouched.
    """
    standard = dataset_cfg.transforms.standard
    transforms_list: list[dict[str, Any]] = OmegaConf.to_container(
        standard.transforms, resolve=True
    )  # type: ignore[assignment]

    already_padded = any(
        t.get("_target_") == "torchvision.transforms.Pad" for t in transforms_list
    )
    if not already_padded:
        transforms_list.append(
            {
                "_target_": "torchvision.transforms.Pad",
                "padding": list(pad_ltrb),
            }
        )

    dataset_cfg.transforms.standard = OmegaConf.create(
        {"_target_": standard["_target_"], "transforms": transforms_list}
    )


def _build_splits(
    split_train_val: float,
    *,
    train_transforms: str = "standard",
) -> dict[str, dict[str, Any]]:
    """Train/val/test split descriptors for ``get_dataloaders``.

    Proxy / zero-cost search: every split uses ``transforms="standard"``
    (``pad_for_proxy`` path). Full 200-ep train may pass
    ``train_transforms="augmented"`` (or ``"auto"`` resolved by the caller)
    for the train split only — val/test stay ``standard``. ``val`` is omitted
    when ``split_train_val<=0``.
    """
    splits: dict[str, dict[str, Any]] = {
        "train": {
            "source": "train",
            "proportion": 1.0 - split_train_val if split_train_val > 0 else 1.0,
            "transforms": train_transforms,
            "shuffle": True,
            "drop_last": True,
        },
        "test": {
            "source": "test",
            "proportion": 1.0,
            "transforms": "standard",
            "shuffle": False,
            "drop_last": False,
        },
    }
    if split_train_val > 0:
        splits["val"] = {
            "source": "train",
            "proportion": split_train_val,
            "transforms": "standard",
            "shuffle": False,
            "drop_last": False,
        }
    return splits


def assert_batch_sane(x: torch.Tensor, y: torch.Tensor, *, dataset_name: str) -> None:
    """Assert a batch is ``BCHW`` float with matching labels and no NaN/Inf.

    Shared by :func:`load` (checked on the train split before returning) and
    ``batch_sanity.py`` (checked per split/per dataset).
    """
    assert isinstance(x, torch.Tensor), (
        f"{dataset_name}: expected a torch.Tensor batch, got {type(x)!r}"
    )
    assert x.ndim == 4, f"{dataset_name}: expected BCHW batch, got shape {tuple(x.shape)}"
    assert x.dtype.is_floating_point, (
        f"{dataset_name}: expected a floating-point tensor, got dtype {x.dtype}"
    )
    assert y is not None and len(y) == x.shape[0], (
        f"{dataset_name}: labels missing or batch-size mismatch "
        f"(inputs={x.shape[0]}, labels={0 if y is None else len(y)})"
    )
    assert not torch.isnan(x).any(), f"{dataset_name}: NaN values found in input batch"
    assert not torch.isinf(x).any(), f"{dataset_name}: Inf values found in input batch"


def load(
    dataset_config: str,
    *,
    batch_size: int = 64,
    seed: int = 0,
    num_workers: int = 0,
    device: torch.device | str = "cpu",
    grow_root: str | os.PathLike[str] | None = None,
    download: bool | None = None,
    pad_for_proxy: bool = True,
    train_transforms: str = "standard",
) -> tuple[torch.utils.data.DataLoader, dict[str, Any]]:
    """Load grow dataloaders for ``dataset_config``.

    Parameters
    ----------
    dataset_config
        Name of a Hydra ``dataset_config`` group entry, e.g. ``"cifar10"``,
        ``"multnist"``, ``"gutenberg"``.
    batch_size
        Shared batch size for all splits.
    seed
        Used both for ``general.seed`` (Hydra override, informational) and as
        the ``get_dataloaders`` split/shuffle seed (reproducible splitting).
    num_workers
        DataLoader worker count. Forced to ``0`` on CPU by ``get_dataloaders``
        regardless of this value.
    device
        Target device; only affects ``pin_memory``/``num_workers`` in the
        returned loaders. Tensors are **not** moved to this device here --
        callers (e.g. ``smoke_score.py``) are responsible for
        ``x, y = x.to(device), y.to(device)`` per the plan's
        ``rand_input=False`` contract.
    grow_root
        Override for the resolved ``experimental_grow`` root (mainly for
        tests); defaults to :func:`_resolve_grow_root`.
    download
        If not ``None``, overrides ``dataset_config.dataset.download``.
        **Caveat:** for the NAS small-benchmark datasets backed by
        ``tools.datasets.NpyWebDataset`` (multnist, cifartile, geoclassing,
        gutenberg, chesseract), this flag does not actually gate the initial
        Figshare fetch -- ``NpyWebDataset.__init__`` always calls
        ``_download_and_extract()`` regardless of ``download``, and only
        skips the network request if the zip is already on disk. It only
        gates torchvision-style datasets (e.g. cifar10) cleanly. Callers
        that must never touch the network (like ``batch_sanity.py``) check
        filesystem existence themselves instead of relying on this flag.
    pad_for_proxy
        When True (default), apply AZ-NAS proxy geometry pads (multnist /
        gutenberg → 32, geoclassing → 64) for zero-cost scoring. When False,
        keep **native** grow shapes (200-ep train default). Callers that need
        residual-safe train geometry for gutenberg pass True via
        ``nb201_common.train_uses_score_pad``.
    train_transforms
        Transform key for the train split: ``"standard"`` (proxy/search),
        ``"augmented"`` (train when YAML defines it), or ``"auto"`` (use
        grow's ``resolve_train_transform_key``).

    Returns
    -------
    tuple[DataLoader, dict]
        ``(train_loader, meta)``. ``meta`` contains at least ``dataset``,
        ``num_classes``, ``batch_shape`` (``(C, H, W)``), ``split_train_val``,
        and ``seed``, plus convenience fields (``val_loader``, ``test_loader``,
        ``in_channels``, ``input_image_size``, ``image_shape``, ``transforms``).
    """
    root = Path(grow_root) if grow_root is not None else _resolve_grow_root()
    _ensure_grow_on_syspath(root)

    from hydra_script.data_handling.datasets import (
        get_dataloaders,
        resolve_train_transform_key,
    )

    cfg = _compose_dataset_cfg(root, dataset_config, seed=seed, num_workers=num_workers)
    dataset_cfg = cfg.dataset_config
    if download is not None:
        dataset_cfg.dataset.download = download

    if pad_for_proxy:
        if dataset_cfg.name == "gutenberg":
            _apply_square_pad(dataset_cfg, _GUTENBERG_PAD_LTRB)
        elif dataset_cfg.name == "multnist":
            _apply_square_pad(dataset_cfg, _MULTNIST_PAD_LTRB)
        elif dataset_cfg.name == "geoclassing":
            _apply_square_pad(dataset_cfg, _GEOCLASSING_PAD_LTRB)

    if train_transforms == "auto":
        train_tf = str(resolve_train_transform_key(dataset_cfg))
    else:
        train_tf = train_transforms

    split_train_val = float(dataset_cfg.split_train_val)
    splits = _build_splits(split_train_val, train_transforms=train_tf)
    device_t = torch.device(device)

    loaders = get_dataloaders(
        dataset_cfg,
        splits=splits,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_t,
        seed=seed,
    )

    train_loader = loaders["train"]
    batch_x, batch_y = next(iter(train_loader))
    assert_batch_sane(batch_x, batch_y, dataset_name=dataset_cfg.name)

    num_channels, height, width = (int(d) for d in batch_x.shape[1:])
    if pad_for_proxy and dataset_cfg.name in _PADDED_SQUARE_EXPECTATIONS:
        pad_side = _PADDED_SQUARE_EXPECTATIONS[dataset_cfg.name]
        assert (height, width) == (pad_side, pad_side), (
            f"{dataset_cfg.name}: expected padded square ({pad_side}, "
            f"{pad_side}), got ({height}, {width})"
        )

    meta: dict[str, Any] = {
        "dataset": str(dataset_cfg.name),
        "num_classes": int(dataset_cfg.num_classes),
        "class_num": int(dataset_cfg.num_classes),
        "batch_shape": (num_channels, height, width),
        "split_train_val": split_train_val,
        "seed": seed,
        "transforms": train_tf,
        "pad_for_proxy": bool(pad_for_proxy),
        "in_channels": num_channels,
        "input_image_size": height,
        "image_shape": (num_channels, height, width),
        "batch_size": batch_size,
        "device": str(device_t),
        "val_loader": loaders.get("val"),
        "test_loader": loaders.get("test"),
    }
    return train_loader, meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ad hoc one-batch check for a single grow dataset_config."
    )
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_loader, meta = load(
        args.dataset_config,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(f"dataset={meta['dataset']} num_classes={meta['num_classes']} "
          f"batch_shape={meta['batch_shape']} split_train_val={meta['split_train_val']}")
    x, y = next(iter(train_loader))
    print(f"train batch: x.shape={tuple(x.shape)} x.dtype={x.dtype} y.shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
