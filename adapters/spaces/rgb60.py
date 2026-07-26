"""Shrunk search-space family: `rgb60` — 3x64x64 RGB (padded), `geoclassing`.

GeoClassing's **measured** native shape is `3x60x60` (on-disk `train_x.npy`
`(N,3,60,60)` float32; grow `batch_shape` after `ToTensor` is `(3,60,60)`).
The zip README/`metadata` claim of `64x64` is stale and wrong.

AZ-NAS proxy geometry policy (see `spaces/README.md`): `grow_data.py` pads
`60x60` to a square `64x64` before scoring — same PixelUnshuffle-safety
rationale as multnist/gutenberg pad→32. With the rgb64-style stride-2 stem,
native `60` yields `60→30→15→8`, and the `15→8` adjacent-map ratio is not an
exact power-of-two PixelUnshuffle stride (job 408708:
`(16x14400)@(4096x16)` at batch=64 ⇒ 15×15 vs 8×8). Pad→64 keeps every
stride-2 step exact (`64→32→16→8→4`).

This module reuses the same init plainnet string / FLOPs budget as `rgb64`
(scored geometry is identical after pad). `input_image_size=64` assumes
padding has already happened upstream; the family name (`rgb60`) records
the measured native side length.

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python rgb60.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="rgb60",
    datasets=("geoclassing",),
    native_shape=(3, 60, 60),  # measured pre-pad; grow_data.py pads to (3, 64, 64)
    in_channels=3,
    input_image_size=64,  # square side AFTER grow_data.py pads 60x60 -> 64x64
    # Same architecture as rgb64 (scored size is identical after pad):
    # stem SuperConvK3BNRELU stride=2; body SuperResIDWE4K3 stride=[2,2,2,1];
    # head SuperConvK1BNRELU stride=1. Verified resolution trace (self_check):
    # [64, 32, 16, 8, 4, 4, 4].
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
        "stem stride=2; body SuperResIDWE4K3 stride=[2,2,2,1] (64->32->16->8->4, "
        "all exact after pad 60->64 -- PixelUnshuffle-safe); "
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
