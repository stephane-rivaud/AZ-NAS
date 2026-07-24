# AZ-NAS sibling setup (Phase 1: `setup-sibling`)

## Repo

- Path: `/Users/strivaud/Projects/research/AZ-NAS` (sibling of `experimental_grow`, cloned from `https://github.com/cvlab-yonsei/AZ-NAS`; not vendored/submoduled into `experimental_grow`).
- Pinned SHA: `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47`
  - Also written to `experimental_grow/experiments/AZ-NAS/results/.az_nas_sha`.

## Dual-worktree path contract (pinned)

This `adapters/` tree is committed **only** on the `feat/grow-adapters`
branch, checked out at a dedicated worktree — never on `master` in the main
`AZ-NAS/` checkout. This mirrors the grow-side contract in
[`experiments/AZ-NAS/README.md`](../../experimental_grow/experiments/AZ-NAS/README.md);
see that file for the full layout diagram and cluster mirror.

```text
~/Projects/research/                                    # (or $HOME on the cluster)
├── experimental_grow/                                   # main checkout — day-to-day branches
├── experimental_grow.worktrees/aznas-compare/            # branch: feat/aznas-compare (pinned)
├── AZ-NAS/                                               # main checkout — stays on master, no adapters/
└── AZ-NAS.worktrees/grow-adapters/                       # branch: feat/grow-adapters (pinned) — you are here
    └── adapters/                                         # this directory
```

- Run everything from this worktree (`AZ-NAS.worktrees/grow-adapters`), not
  from the main `AZ-NAS/` checkout — `master` is kept clean of adapter code
  intentionally, so it stays free for unrelated AZ-NAS upstream work.
- The `.venv-mbv2` environment described below can live in either the main
  checkout or this worktree; each worktree has its own working tree, so
  build it wherever you're actually running from (`AZ_NAS_ROOT/.venv-mbv2`,
  where `AZ_NAS_ROOT` defaults to this worktree — see the grow-side
  launchers under `experiments/AZ-NAS/launchers/`). `.venv-mbv2/` is
  `.gitignore`d repo-wide, independent of which worktree it's built in.
- Env vars for launchers/adapters: `EXPERIMENTAL_GROW_ROOT` →
  `experimental_grow.worktrees/aznas-compare`, `AZ_NAS_ROOT` → this worktree
  (`AZ-NAS.worktrees/grow-adapters`). Both are overridable; see the grow-side
  README for the fallback rules.

## MBV2 venv

- Location: `AZ-NAS/.venv-mbv2` (gitignored via the cloned repo's own `.gitignore` — matches `.venv`/`venv/`/`env/` patterns; also never tracked by `experimental_grow` git).
- Created with `uv venv --python 3.11 .venv-mbv2` (Python 3.11.13). Python 3.11 was chosen deliberately instead of the system default (3.13/3.14 on this machine) because `torch==2.0.1` (pinned in `ImageNet_MBV2/requirements.txt`) has no wheels for those newer interpreters.
- Activate: `source /Users/strivaud/Projects/research/AZ-NAS/.venv-mbv2/bin/activate`

### Installed (from `ImageNet_MBV2/requirements.txt`, best-effort)

Installed at pinned versions:

- `torch==2.0.1` (CPU build — this machine is macOS arm64, no CUDA; verified `torch.cuda.is_available() == False`)
- `torchvision==0.15.2`
- `numpy==1.23.5`
- `ptflops==0.7`
- `tqdm==4.65.0`
- `matplotlib==3.7.1`
- `pandas==2.0.2`
- `scikit_learn==1.2.2`

### Skipped / gaps

- `apex==0.1` — NVIDIA Apex requires a CUDA toolchain to build from source (see repo `Dockerfile`, which builds it with `--cuda_ext`). Not installable/needed on this CPU-only macOS host. Not imported by the score-computation path (`Masternet.py`, `PlainNet/*`, `ModelLoader/*`, `ZeroShotProxy/compute_az_nas_score.py`) — only referenced by training scripts (`train_image_classification.py`, `ts_train_image_classification.py`) which are out of scope for this phase (proxy smoke + matrix scoring only).
- `horovod==0.28.0` — requires MPI/NCCL and a working CUDA-enabled torch build to compile against; not installable on this host. Same scope note as apex: only used by distributed training scripts, not by the proxy-scoring path.
- `tensorflow==2.12.0` — heavy dependency, not imported anywhere in the proxy-scoring import chain (`Masternet`, `PlainNet`, `ModelLoader`, `ZeroShotProxy/compute_az_nas_score`). Only pulled in by `global_utils.py`'s TensorBoard-adjacent logging helpers and `xautodl/log_utils/*` (training/logging utilities), and by some `ModelLoader/geffnet/*` modules used for pretrained EfficientNet/MobileNetV3 loading, not the plainnet/MasterNet path this project needs. Skipped; not required for smoke/matrix scoring.

### Verified working

Ran a direct import smoke check (not the adapter script — that's a later phase) inside `ImageNet_MBV2/` using the new venv:

```bash
cd /Users/strivaud/Projects/research/AZ-NAS/ImageNet_MBV2
../.venv-mbv2/bin/python -c "
import torch, torchvision
import Masternet
import PlainNet
from ZeroShotProxy import compute_az_nas_score
"
```

Result: imports succeed; `torch==2.0.1`, `cuda available: False`.

### `grow_data.py` deps (Phase 2: `grow-data-adapter` / `batch-sanity`)

Added to the MBV2 venv, minimal set needed for Hydra config composition:

```bash
uv pip install --python .venv-mbv2/bin/python hydra-core omegaconf
```

Installed: `hydra-core==1.3.4`, `omegaconf==2.3.1`, `antlr4-python3-runtime==4.9.3`,
`pyyaml==6.0.3` (transitive). `requests`, `numpy`, `torch`, `torchvision` were
already present from the Phase 1 install above.

**What works in the MBV2 venv with just those added:** `grow_data.load("cifar10", ...)`
end-to-end (path resolution → `sys.path` → `initialize_config_dir`/`compose` →
`get_dataloaders` → dataset instantiation) — verified it reaches all the way to
`torchvision.datasets.CIFAR10`'s own "Dataset not found" error when the data
isn't present (i.e. the whole adapter plumbing is exercised; only the actual
Torchvision data read is missing).

**What does not work in the MBV2 venv:** the five P0 / matrix datasets
(multnist, cifartile, geoclassing, gutenberg, chesseract) all route through
`tools.datasets.<Class>`, and `tools/datasets.py` does
`from gromo.utils.utils import global_device` at *module import time* — Hydra's
`instantiate()` on `dataset_config.dataset` triggers that import as soon as any
of these five configs is composed and instantiated, regardless of `download`.
In the MBV2 venv this fails with:

```text
hydra.errors.InstantiationException: Error locating target 'tools.datasets.MultNIST', ...
  caused by: ImportError: Error loading 'tools.datasets.MultNIST':
    ModuleNotFoundError("No module named 'gromo.utils'")
```

**Decision — do not install `gromo` into the MBV2 venv.** `gromo` is grow's own
package (pinned via git SHA in `experimental_grow/pyproject.toml`) and grow's
own `uv` env resolves it against `torch==2.11.0` / Python 3.13.11 — both far
ahead of MBV2's pinned `torch==2.0.1` / Python 3.11.13 (chosen in Phase 1
specifically because `torch==2.0.1` has no wheels for newer interpreters).
Installing `gromo` here would either fail to resolve, or silently upgrade/pin
a different torch into the score-computation venv, which is exactly what
Phase 1's dedicated-venv isolation was meant to prevent.

**Chosen approach:** `grow_data.py` itself is venv-agnostic (no AZ-NAS imports,
works anywhere `hydra-core`/`omegaconf`/`torch`/`torchvision` are importable).
For actually exercising it against the five `gromo`-backed P0 datasets — e.g.
`batch_sanity.py`'s full loop, or any future data-loading test/dev work — run
it inside grow's own `uv` environment, which already has every dependency
(`hydra-core`, `omegaconf`, `gromo`, `requests`, `torch`, `torchvision`)
installed and version-matched:

```bash
cd /Users/strivaud/Projects/research/experimental_grow
uv run python /Users/strivaud/Projects/research/AZ-NAS/adapters/batch_sanity.py
uv run python /Users/strivaud/Projects/research/AZ-NAS/adapters/grow_data.py --dataset_config cifar10
```

The MBV2 venv remains the env for the not-yet-implemented `smoke_score.py` /
`run_score_matrix.py` (`MasterNet`/`ZeroShotProxy` scoring). Whoever implements
those needs to decide how the two environments hand off batches for the five
`gromo`-backed datasets (e.g. run `grow_data.load` out-of-process from grow's
venv and pass tensors/metadata across, or accept CPU-only proxy scores using
only `cifar10`-style loaders directly importable in MBV2). That decision is
explicitly **not** made here — flagging it for the `smoke-score` todo.

### Blockers / follow-ups for later phases

- **No CUDA on this dev machine.** `rand_input=False` scoring and any latency benchmarking will run CPU-only here; real GPU runs (or the `skip_latency=True` matrix runner) need to target a CUDA host. Document this in `experiments/AZ-NAS/README.md` dual-env / CUDA-assumption section (owned by `scaffold-experiments` / `docs-disclaimer` todos).
- apex/horovod/tensorflow gaps above are believed non-blocking for smoke + matrix scoring (proxy score path doesn't import them), but flagging here in case a later phase's import graph pulls in `global_utils.py` code paths that need them — if so, install `tensorflow==2.12.0` (pip-installable, no CUDA needed for logging-only use) as a follow-up; apex/horovod remain out of reach on this host without a CUDA toolchain.
