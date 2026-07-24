"""Shrunk search-space family: `gray_sq` — 1x27x27 grayscale, `gutenberg`.

Gutenberg's native shape is `(1, 27, 18)` (locked in the plan: pad to square
`27x27` in `grow_data.py` before scoring — never pass native `27x18` or `H`
alone). This module's `input_image_size=27` assumes that padding has already
happened upstream.

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
    native_shape=(1, 27, 18),  # pre-pad; grow_data.py pads to (1, 27, 27)
    in_channels=1,
    input_image_size=27,  # square side AFTER grow_data.py pads 27x18 -> 27x27
    # C=1 grayscale stem (SuperConvK3BNRELU in_channels=1), stride=1 (27x27
    # is small and not a power of two; an early stride-2 stem would waste
    # detail on the padded rows). Three SuperResIDWE4K3 stride=2 stages take
    # 27 -> 13 -> 6 -> 3 (floor division per PlainNet's ConvKX/ConvDW
    # `get_output_resolution`), then the stride-1 head. Verified resolution
    # trace (self_check): [27, 27, 13, 6, 3, 3].
    init_plainnet_str=(
        "SuperConvK3BNRELU(1,8,1,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperConvK1BNRELU(48,128,1,1)"
    ),
    budget_flops=5e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1, in_channels=1 (grayscale); "
        "body SuperResIDWE4K3 stride=[2,2,2] (27->13->6->3, floor division); "
        "head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=6,
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
