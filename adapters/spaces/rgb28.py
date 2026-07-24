"""Shrunk search-space family: `rgb28` — 3x32x32 RGB (padded), e.g. `multnist`.

Multnist's native shape is `3x28x28`. AZ-NAS proxy geometry policy (see
`spaces/README.md`): `grow_data.py` pads it to a square `32x32` before
scoring, because AZ-NAS's `compute_az_nas_score` trainability term uses
`nn.PixelUnshuffle` to reconcile mismatched feature-map resolutions between
adjacent layers, which requires every stride-2 downsample to divide evenly.
Native `28x28` hits a non-power-of-two spatial size partway through a
standard 3x stride-2 body (28 -> 14 -> 7 -> 3, where the last stride hits an
odd input and `PixelUnshuffle` raises). `32x32` keeps every stride-2 step
exact (32 -> 16 -> 8 -> 4), matching `rgb32`'s already-verified geometry.
This module's `input_image_size=32` assumes that padding has already
happened upstream; the family name (`rgb28`) is kept as the historical
family identifier and does not reflect the scored resolution.

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
    native_shape=(3, 28, 28),  # pre-pad; grow_data.py pads to (3, 32, 32)
    in_channels=3,
    input_image_size=32,  # square side AFTER grow_data.py pads 28x28 -> 32x32
    # Stem SuperConvK3BNRELU keeps stride=1 (small inputs shouldn't lose
    # detail immediately -- same convention as rgb32/CIFAR). Three
    # SuperResIDWE4K3 stride-2 stages take 32 -> 16 -> 8 -> 4 (every step
    # exact, no PixelUnshuffle divisibility failure); the stride-1
    # SuperConvK1BNRELU head keeps the final 4x4 before global average
    # pooling. Verified resolution trace (self_check): [32, 32, 16, 8, 4, 4].
    init_plainnet_str=(
        "SuperConvK3BNRELU(3,8,1,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperConvK1BNRELU(48,128,1,1)"
    ),
    budget_flops=6e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1; body SuperResIDWE4K3 stride=[2,2,2] (32->16->8->4, "
        "all exact -- PixelUnshuffle-safe); head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=10,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
