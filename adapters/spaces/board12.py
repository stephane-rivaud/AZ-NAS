"""Shrunk search-space family: `board12` — 12x8x8 board-encoded, `chesseract`.

Locked constraint (plan): **no stride-2 stem/stage that collapses the 8x8
spatial size below 2.** This family therefore uses at most one stride-2
stage total, preferring stride-1 sub-layers otherwise, so an 8x8 chess-board
encoding is never downsampled into a degenerate 1x1 (or 0x0) feature map
before global average pooling.

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python board12.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="board12",
    datasets=("chesseract",),
    native_shape=(12, 8, 8),
    in_channels=12,
    input_image_size=8,
    # C=12 stem (SuperConvK3BNRELU in_channels=12), stride=1. Only the
    # *second* body stage uses stride=2 (8->4); the stem and the other two
    # SuperResIDWE4K3 stages plus the head all stay stride=1, so spatial
    # size never drops below 4 (well clear of the "<2" floor). Verified
    # resolution trace (self_check): [8, 8, 8, 4, 4, 4].
    init_plainnet_str=(
        "SuperConvK3BNRELU(12,16,1,1)"
        "SuperResIDWE4K3(16,32,1,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperResIDWE4K3(48,64,1,32,1)"
        "SuperConvK1BNRELU(64,192,1,1)"
    ),
    budget_flops=8e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1, in_channels=12 (board planes); "
        "body SuperResIDWE4K3 stride=[1,2,1] — exactly one stride-2 stage "
        "(8->8->4->4), no collapse below 2; head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=3,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
