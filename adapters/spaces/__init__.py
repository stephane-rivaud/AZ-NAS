"""`AZ-NAS/adapters/spaces` — shrunk search-space family registry.

See `README.md` in this directory for the full import contract. Quick start:

    from adapters.spaces import get_space, get_space_for_dataset

    cfg = get_space("multnist")          # by grow dataset_config name
    cfg = get_space_for_dataset("rgb28")  # or by family name directly
    print(cfg.init_plainnet_str, cfg.budget_flops, cfg.max_layers)
"""

from . import board12, gray_sq, rgb28, rgb32, rgb64
from ._common import SpaceConfig

FAMILIES = {
    "rgb28": rgb28.CONFIG,
    "rgb64": rgb64.CONFIG,
    "rgb32": rgb32.CONFIG,
    "gray_sq": gray_sq.CONFIG,
    "board12": board12.CONFIG,
}

_DATASET_TO_FAMILY = {
    dataset: family for family, cfg in FAMILIES.items() for dataset in cfg.datasets
}


def get_space(family_or_dataset: str) -> SpaceConfig:
    """Look up a `SpaceConfig` by family name (e.g. `"rgb28"`) or by grow
    `dataset_config` name (e.g. `"multnist"`). Raises `KeyError` with the
    full list of known keys if neither matches.
    """
    if family_or_dataset in FAMILIES:
        return FAMILIES[family_or_dataset]
    if family_or_dataset in _DATASET_TO_FAMILY:
        return FAMILIES[_DATASET_TO_FAMILY[family_or_dataset]]
    known = sorted(set(FAMILIES) | set(_DATASET_TO_FAMILY))
    raise KeyError(f"Unknown family/dataset {family_or_dataset!r}; known: {known}")


# Alias: some callers look up by dataset explicitly for readability.
get_space_for_dataset = get_space

__all__ = ["FAMILIES", "SpaceConfig", "get_space", "get_space_for_dataset"]
