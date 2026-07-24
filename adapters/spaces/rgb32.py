"""Shrunk search-space family: `rgb32` — 3x32x32 RGB, e.g. `cifar10`, `cifar100` (P1).

See `spaces/README.md` for the import contract. Run this module directly
(inside the MBV2 venv) to self-check the init string:

    cd AZ-NAS/adapters/spaces && ../../.venv-mbv2/bin/python rgb32.py
"""

try:
    from ._common import SpaceConfig, self_check
except ImportError:  # running as a standalone script, not `python -m`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import SpaceConfig, self_check

CONFIG = SpaceConfig(
    family="rgb32",
    datasets=("cifar10", "cifar100"),
    native_shape=(3, 32, 32),
    in_channels=3,
    input_image_size=32,
    # CIFAR-like: stem SuperConvK3BNRELU stride=1 (standard CIFAR practice —
    # no early stride-2 stem, unlike ImageNet's 224x224 pipeline). Three
    # SuperResIDWE4K3 stride=2 stages take 32 -> 16 -> 8 -> 4, then the
    # stride-1 SuperConvK1BNRELU head. Verified resolution trace
    # (self_check): [32, 32, 16, 8, 4, 4].
    init_plainnet_str=(
        "SuperConvK3BNRELU(3,8,1,1)"
        "SuperResIDWE4K3(8,16,2,8,1)"
        "SuperResIDWE4K3(16,32,2,16,1)"
        "SuperResIDWE4K3(32,48,2,24,1)"
        "SuperConvK1BNRELU(48,192,1,1)"
    ),
    budget_flops=8e6,
    max_layers=10,
    stride_policy=(
        "stem stride=1 (CIFAR convention, no early downsample); "
        "body SuperResIDWE4K3 stride=[2,2,2] (32->16->8->4); "
        "head SuperConvK1BNRELU stride=1"
    ),
    search_space_py="SearchSpace/search_space_IDW_fixfc.py",
    skip_latency=True,
    num_classes_hint=None,  # cifar10=10, cifar100=100: read from dataset_config
)


if __name__ == "__main__":
    import json

    result = self_check(CONFIG, num_classes=CONFIG.num_classes_hint or 10)
    print(json.dumps(result, indent=2))
