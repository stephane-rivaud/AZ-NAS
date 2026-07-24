"""Shrunk search-space family: `rgb28` — 3x28x28 RGB, e.g. `multnist`.

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python rgb28.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="rgb28",
    datasets=("multnist",),
    native_shape=(3, 28, 28),
    in_channels=3,
    input_image_size=28,
    # Stem SuperConvK3BNRELU keeps stride=1 (28x28 is already small: an
    # ImageNet-style stride-2 stem would immediately halve a tiny image).
    # Three SuperResIDWE4K3 stride-2 stages take 28 -> 14 -> 7 -> 3; the
    # stride-1 SuperConvK1BNRELU head keeps the final 3x3 before global
    # average pooling. Verified resolution trace (self_check):
    # [28, 28, 14, 7, 3, 3].
    init_plainnet_str=(
        "SuperConvK3BNRELU(3,8,1,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperConvK1BNRELU(48,128,1,1)"
    ),
    budget_flops=5e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1; body SuperResIDWE4K3 stride=[2,2,2] (28->14->7->3); "
        "head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=10,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
