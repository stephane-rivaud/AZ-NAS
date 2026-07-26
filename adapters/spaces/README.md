# `adapters/spaces/` — shrunk search-space families

Executable, dependency-light Python modules describing one shrunk PlainNet
search space per dataset family from the plan's "Shrunk search spaces"
table. These are **not** AZ-NAS `SearchSpace/search_space_*.py` block-mutation
files (those already exist upstream and are reused as-is, referenced by
`search_space_py` below) — this directory instead answers the question
*"what named initial `plainnet` structure, FLOPs budget, and layer/stride
policy should `smoke_score.py` / `run_score_matrix.py` start from for
dataset X?"*.

## Files

| File | Family | Datasets | Native shape | Locked geometry | in_channels |
|------|--------|----------|---------------|------------------|-------------|
| `rgb28.py` | `rgb28` | `multnist` | `(3,28,28)` | pad → 32×32 | 3 |
| `rgb64.py` | `rgb64` | `cifartile` | `(3,64,64)` | 64×64 | 3 |
| `rgb60.py` | `rgb60` | `geoclassing` | `(3,60,60)` | pad → 64×64 | 3 |
| `rgb32.py` | `rgb32` | `cifar10`, `cifar100` | `(3,32,32)` | 32×32 | 3 |
| `gray_sq.py` | `gray_sq` | `gutenberg` | `(1,27,18)` | pad → 32×32 | 1 |
| `board12.py` | `board12` | `chesseract` | `(12,8,8)` | 8×8, no stride-2 collapse | 12 |

**AZ-NAS proxy geometry policy (`rgb28`/`gray_sq` pad to 32×32;
`rgb60`/geoclassing pad to 64×64):** AZ-NAS's `compute_az_nas_score`
trainability term uses `nn.PixelUnshuffle` to reconcile mismatched
feature-map resolutions between adjacent layers, which requires every
stride-2 downsample in the resolution trace to divide evenly
(`PixelUnshuffle` raises otherwise). Multnist's native `28x28` and
gutenberg's native `27x18`/padded-`27x27` both hit a non-power-of-two
spatial size partway through a standard 3x stride-2 body
(`28->14->7->3` and `27->13->6->3`), crashing the trainability
computation before it can even run. GeoClassing's **measured** native
`60x60` (not the stale zip README `64x64`) under an rgb64-style stride-2
stem yields `60->30->15->8`; the `15→8` adjacent-map ratio is not an exact
power-of-two unshuffle stride. Padding multnist/gutenberg to `32x32` and
geoclassing to `64x64` keeps every stride-2 step exact
(`32->16->8->4` / `64->32->16->8->4`). This padding happens only in
`grow_data.py`'s in-memory transform pipeline for proxy/search purposes --
native data on disk, and the `native_shape` field documented above, are
unchanged.

Plus `_common.py` (shared `SpaceConfig` dataclass + `self_check()` helper,
no `ImageNet_MBV2` imports at module load time) and `__init__.py` (registry).

## Import contract

Each family module (`rgb28.py`, `rgb64.py`, `rgb60.py`, `rgb32.py`, `gray_sq.py`,
`board12.py`) exposes exactly one public symbol:

```python
CONFIG: _common.SpaceConfig
```

`SpaceConfig` is a frozen dataclass with these fields (all plain
str/int/float/bool/tuple — importing a family module never requires
`ImageNet_MBV2` on `sys.path`, `torch`, or a CUDA device):

| Field | Type | Meaning |
|-------|------|---------|
| `family` | `str` | family name, matches the module's filename stem |
| `datasets` | `tuple[str, ...]` | grow `dataset_config` names this family serves |
| `native_shape` | `tuple[int,int,int]` | `(C,H,W)` as grow produces it, pre-pad |
| `in_channels` | `int` | stem `in_channels` (matches native/padded `C`) |
| `input_image_size` | `int` | square side length AFTER any padding (`H == W`); the resolution to pass as `resolution=` / `input_image_size` to `compute_nas_score` / `get_FLOPs` |
| `init_plainnet_str` | `str` | named initial `plainnet` structure string, loadable by `Masternet.MasterNet(plainnet_struct=...)` |
| `budget_flops` | `float` | FLOPs budget at `input_image_size`, shrunk vs. ImageNet's 450M (see table below) |
| `max_layers` | `int` | max `get_num_layers()` allowed during evolution (init string is well under this, leaving search headroom) |
| `stride_policy` | `str` | human-readable stride/downsample description |
| `search_space_py` | `str` | path relative to `ImageNet_MBV2/` of the upstream `SearchSpace/search_space_*.py` (`gen_search_space`) to pair with this init string for block mutations — reused unmodified, not reimplemented here |
| `skip_latency` | `bool` | always `True` for these families (matrix/smoke runs skip latency; the stock latency helper hardcodes `in_channels=3`, which is wrong for `gray_sq` (`C=1`) and `board12` (`C=12`)) |
| `num_classes_hint` | `int \| None` | **documentation only** — `None` when a family serves multiple datasets with different class counts (`rgb32`). Consumers MUST read the authoritative value from `cfg.dataset_config.num_classes`, never from this field, per the plan's `getmisc()` replacement contract |

### How `smoke_score.py` (or any consumer) should use this

```python
# from adapters/ (smoke_score.py's own directory), after argv parsing,
# BEFORE os.chdir(ImageNet_MBV2) — this import only touches plain data:
from spaces import get_space

cfg = get_space(args.dataset_config)  # looks up by dataset name OR family name
# ... resolve num_classes from grow's Hydra cfg, NOT cfg.num_classes_hint ...

os.chdir(mbv2_root)  # only now do MBV2-local imports become valid
import Masternet
the_model = Masternet.MasterNet(
    num_classes=resolved_num_classes,
    plainnet_struct=cfg.init_plainnet_str,
    no_create=False,
)
# resolution=cfg.input_image_size, skip_latency=cfg.skip_latency, etc.
```

`get_space(name)` (from `spaces/__init__.py`) accepts either a family name
(`"rgb28"`) or a grow `dataset_config` name (`"multnist"`) and returns the
matching `SpaceConfig`.

### Running a family's self-check directly

Each module also has a `if __name__ == "__main__":` block that builds a
real `MasterNet` from `init_plainnet_str`, verifies FLOPs/layers stay inside
budget, checks the resolution never collapses below 2, and runs a forward
pass — this requires the MBV2 venv (needs `torch`, `Masternet`, `PlainNet`):

```bash
cd AZ-NAS/adapters/spaces
../../.venv-mbv2/bin/python rgb28.py     # or rgb64.py / rgb60.py / rgb32.py / gray_sq.py / board12.py
```

## Design notes

- **Block type:** all five families use `SuperResIDWE4K3` (inverted
  depthwise-separable, expansion=4, kernel=3) body stages, paired with
  `SearchSpace/search_space_IDW_fixfc.py` for block-mutation search (same
  search-space file the stock ImageNet `flops450M` script uses). `E4` (not
  `E6`, the ImageNet default) was chosen because these shrunk inputs don't
  need as much per-block capacity; the search space still explores
  `E1/E2/E4/E6` and `K3/K5/K7` variants when mutated.
- **FLOPs budgets are shrunk 30–90x vs. ImageNet's 450M** — actual init-string
  FLOPs (measured via `self_check()`) are ~1.7M–3.7M; budgets leave headroom
  (5M–15M) for evolution to grow channels/sub-layers without approaching
  ImageNet-scale compute:

  | Family | Init FLOPs (measured) | `budget_flops` |
  |--------|------------------------|----------------|
  | `rgb28` | ~2.8M (at padded 32×32) | 6M |
  | `rgb64` | ~3.7M | 15M |
  | `rgb60` | ~3.7M (at padded 64×64; same init as `rgb64`) | 15M |
  | `rgb32` | ~2.9M | 8M |
  | `gray_sq` | ~2.7M (at padded 32×32) | 6M |
  | `board12` | ~2.2M | 8M |

- **Stride policy is resolution-aware, not copy-pasted from ImageNet:**
  ImageNet's init string uses a stride-2 stem plus four stride-2 body
  stages (32x total downsample, 224→7). None of these shrunk families use
  that pattern verbatim:
  - `rgb28`/`rgb32`/`gray_sq`: stride-1 stem (small inputs shouldn't lose
    detail immediately), 3 stride-2 body stages.
  - `rgb64`/`rgb60`: stride-2 stem is affordable at (padded) 64x64; 3
    stride-2 body stages + 1 stride-1 refine stage.
  - `board12`: **only one** stride-2 stage total (locked constraint — 8x8
    board encoding must never collapse below 2x2); stem and two of three
    body stages are stride-1.
- **`rgb28`/`gray_sq`/`rgb60` assume padding already happened upstream.**
  `grow_data.py` pads multnist's native `28x28` and gutenberg's native
  `27x18` both to square `32x32`, and geoclassing's measured native
  `60x60` to square `64x64`, before each space's `input_image_size` is
  used -- **not** their native sizes. This is the AZ-NAS proxy geometry
  policy (see above): AZ-NAS's PixelUnshuffle-based trainability term
  requires every stride-2 downsample to divide evenly, which those native
  sizes fail under their families' stride traces, while the padded squares
  do not.
- **Every `SpaceConfig` self-check asserts the resolution trace never drops
  below 2** at any point (not just at the end), matching the plan's
  `board12` constraint but enforced generically for all families.
