"""Shrunk search-space family: `rgb64` — 3x64x64 RGB, e.g. `cifartile`, `geoclassing`.

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python rgb64.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="rgb64",
    datasets=("cifartile", "geoclassing"),
    native_shape=(3, 64, 64),
    in_channels=3,
    input_image_size=64,
    # Stem SuperConvK3BNRELU stride=2 (64x64 can afford one stride-2 stem,
    # unlike the 28x28/27x27 families). Four SuperResIDWE4K3 stages: three
    # stride=2 (64->32->16->8) plus one stride=1 refine stage at 8x8, then
    # the stride-1 SuperConvK1BNRELU head. Verified resolution trace
    # (self_check): [64, 32, 16, 8, 4, 4, 4].
    init_plainnet_str=(
        "SuperConvK3BNRELU(3,8,2,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperResIDWE4K3(48,64,1,32,1)"
        "SuperConvK1BNRELU(64,256,1,1)"
    ),
    budget_flops=15e6,
    max_layers=12,
    stride_policy=(
        "stem stride=2; body SuperResIDWE4K3 stride=[2,2,2,1] (64->32->16->8->4); "
        "head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=None,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
