# SlabRpfPinn

## Overview

CPU finite-volume and JAX neural-network tools for the steady relativistic
runaway-electron first-passage problem: generate an FV dataset, train a
model (MLP or DeepONet, data-driven or physics-informed), then validate it.

## Install

```bash
if [ ! -x .venv/bin/python ]; then
  python -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
source .venv/bin/activate
```

FV generation runs on CPU. Training and validation require GPU JAX.
`requirements.txt` contains the CUDA 13 JAX stack and the Git sources required
for SOAP and SSBroyden.

## Workflow

For production, submit tracked Slurm scripts:

```bash
mkdir -p logs
sbatch generate_fv_dataset.sbatch
sbatch train_model.sbatch
sbatch validate_model.sbatch
```

For interactive validation, run drivers directly inside an allocation:

```bash
python generate_fv_dataset.py
python train_model.py
python validate_model.py
```

Each script reads its own JSON config: `configs/fv_dataset.json`,
`configs/train.json`, `configs/validate_model.json`. Configuration validation
and manifest checks enforce cross-stage requirements. Each new run needs new
`run_dir`/`output_dir`/`checkpoint_dir` values; completed runs are protected
from overwrite.

## Code layout

`core/` separates model architecture, sampling, optimizers, PDE physics,
supervised and physics training, evaluation, workflow orchestration, plotting,
and artifact persistence. Root Python files only select configuration and call
those workflow modules.
