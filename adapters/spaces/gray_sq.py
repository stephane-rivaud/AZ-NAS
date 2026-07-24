"""Shrunk search-space family: `gray_sq` — 1x32x32 grayscale (padded), `gutenberg`.

Gutenberg's native shape is `(1, 27, 18)`. AZ-NAS proxy geometry policy (see
`spaces/README.md`): `grow_data.py` pads it to a square `32x32` before
scoring -- not the previously-locked `27x27` -- because AZ-NAS's
`compute_az_nas_score` trainability term uses `nn.PixelUnshuffle` to
reconcile mismatched feature-map resolutions between adjacent layers, which
requires every stride-2 downsample to divide evenly. `27x27` hits a
non-power-of-two spatial size on the very first stride-2 stage
(27 -> 13, `PixelUnshuffle` requires the larger side divisible by the
stride and raises). `32x32` keeps every stride-2 step exact
(32 -> 16 -> 8 -> 4), matching `rgb32`'s already-verified geometry. This
module's `input_image_size=32` assumes that padding has already happened
upstream.

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python gray_sq.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="gray_sq",
    datasets=("gutenberg",),
    native_shape=(1, 27, 18),  # pre-pad; grow_data.py pads to (1, 32, 32)
    in_channels=1,
    input_image_size=32,  # square side AFTER grow_data.py pads 27x18 -> 32x32
    # C=1 grayscale stem (SuperConvK3BNRELU in_channels=1), stride=1 (small
    # inputs shouldn't lose detail immediately -- same convention as
    # rgb32/CIFAR). Three SuperResIDWE4K3 stride=2 stages take
    # 32 -> 16 -> 8 -> 4 (every step exact, no PixelUnshuffle divisibility
    # failure), then the stride-1 head. Verified resolution trace
    # (self_check): [32, 32, 16, 8, 4, 4].
    init_plainnet_str=(
        "SuperConvK3BNRELU(1,8,1,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperConvK1BNRELU(48,128,1,1)"
    ),
    budget_flops=6e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1, in_channels=1 (grayscale); "
        "body SuperResIDWE4K3 stride=[2,2,2] (32->16->8->4, all exact -- "
        "PixelUnshuffle-safe); head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=6,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
