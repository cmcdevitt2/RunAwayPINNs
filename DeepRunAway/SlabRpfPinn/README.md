# SlabRpfPinn

## Overview

CPU finite-volume and JAX neural-network tools for the steady relativistic
runaway-electron first-passage problem: generate an FV dataset, train a
model (MLP or DeepONet, data-driven or physics-informed), then validate it.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runs fully on CPU. GPUs are optional and only change performance. To use a
GPU, install the matching CUDA plugin in place of the plain `jax` line, e.g.:

```bash
pip install "jax[cuda13]"
```

See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) for the full package list.

## Workflow

```bash
python generate_fv_dataset.py
python train_model.py
python validate_model.py
```

Each script reads its own JSON config: `run_configs/fv_dataset.json`,
`run_configs/train.json`, `run_configs/validate_model.json`. Python config
schemas validate training values, and comments in the loader code explain
cross-stage requirements. Each new run needs new
`run_dir`/`output_dir`/`checkpoint_dir` values; completed runs are protected
from overwrite.
