"""Shared helpers for `AZ-NAS/adapters/spaces/*` family modules.

This module is intentionally free of any AZ-NAS/`ImageNet_MBV2`-only imports
at module load time (no `import Masternet`, no `import PlainNet`) so that a
family module can be imported from anywhere (e.g. by `experimental_grow`
docs tooling that only wants the dataclass metadata) without needing
`ImageNet_MBV2` on `sys.path` or CWD set to that directory.

`Masternet`/`PlainNet` are only imported lazily, inside `self_check()`,
which is the one place that actually needs to build a real model.
See `spaces/README.md` for the full import contract.
"""

from __future__ import annotations

import dataclasses
import os
import sys


@dataclasses.dataclass(frozen=True)
class SpaceConfig:
    """One shrunk search-space family, per the plan's "Shrunk search spaces" table.

    Every field here is plain data (str/int/float/bool/tuple) so it can be
    imported and inspected (or `json.dumps(dataclasses.asdict(cfg))`-ed)
    without ever constructing a `torch.nn.Module`.
    """

    family: str
    datasets: tuple  # grow `dataset_config` names this family serves
    native_shape: tuple  # (C, H, W) as produced by grow's raw dataset, pre-pad
    in_channels: int
    input_image_size: int  # square side length AFTER any padding (H == W)
    init_plainnet_str: str  # named initial `plainnet` structure string
    budget_flops: float
    max_layers: int
    stride_policy: str  # human-readable description of stride/downsample plan
    search_space_py: str  # path, relative to `ImageNet_MBV2/`, of the AZ-NAS
    # SearchSpace module (`gen_search_space`) to pair with this init string
    # for block-mutation search (reused as-is; not reimplemented here)
    skip_latency: bool = True
    num_classes_hint: int | None = None  # documentation only; smoke_score
    # must read the authoritative value from `cfg.dataset_config.num_classes`

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _resolve_mbv2_root() -> str:
    """`<AZ-NAS-repo-root>/ImageNet_MBV2`, derived from this file's location.

    `spaces/` lives at `AZ-NAS/adapters/spaces/`, so `ImageNet_MBV2` is a
    sibling of `adapters/` one level up from this file's parent.
    """
    adapters_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    az_nas_root = os.path.dirname(adapters_dir)
    return os.path.join(az_nas_root, "ImageNet_MBV2")


def self_check(cfg: SpaceConfig, num_classes: int = 10, batch: int = 2) -> dict:
    """Build a real `MasterNet` from `cfg.init_plainnet_str` and sanity-check it.

    Requires running inside the MBV2 venv (`AZ-NAS/.venv-mbv2`), since it
    imports `Masternet`/`PlainNet`/`torch` from `ImageNet_MBV2`. Not used by
    the grow side; this is a spaces-only self-test helper (also invoked by
    `python -m adapters.spaces.<family>`).

    Returns a dict with flops/params/layers/resolution-trace/output-shape so
    callers (or `smoke_score.py` later) can assert the family stays inside
    its documented budget without duplicating this wiring.
    """
    mbv2_root = _resolve_mbv2_root()
    prev_cwd = os.getcwd()
    path_added = mbv2_root not in sys.path
    if path_added:
        sys.path.insert(0, mbv2_root)
    os.chdir(mbv2_root)
    try:
        import torch  # noqa: PLC0415 (intentionally lazy, see module docstring)
        import Masternet  # noqa: PLC0415

        net = Masternet.MasterNet(
            num_classes=num_classes, plainnet_struct=cfg.init_plainnet_str, no_create=False
        )
        net.eval()

        flops = net.get_FLOPs(cfg.input_image_size)
        params = net.get_model_size()
        layers = net.get_num_layers()

        resolution_trace = [cfg.input_image_size]
        r = cfg.input_image_size
        for block in net.block_list:
            r = block.get_output_resolution(r)
            resolution_trace.append(r)
        assert min(resolution_trace) >= 2, (
            f"{cfg.family}: stride policy collapses spatial size below 2 "
            f"(trace={resolution_trace})"
        )

        x = torch.randn(batch, cfg.in_channels, cfg.input_image_size, cfg.input_image_size)
        with torch.no_grad():
            y = net(x)
        assert tuple(y.shape) == (batch, num_classes)

        if flops > cfg.budget_flops:
            raise AssertionError(
                f"{cfg.family}: init string FLOPs ({flops:.3e}) exceed "
                f"budget_flops ({cfg.budget_flops:.3e})"
            )
        if layers > cfg.max_layers:
            raise AssertionError(
                f"{cfg.family}: init string layers ({layers}) exceed "
                f"max_layers ({cfg.max_layers})"
            )

        return {
            "family": cfg.family,
            "flops": flops,
            "params": params,
            "layers": layers,
            "resolution_trace": resolution_trace,
            "output_shape": tuple(y.shape),
        }
    finally:
        os.chdir(prev_cwd)
        if path_added:
            sys.path.remove(mbv2_root)
