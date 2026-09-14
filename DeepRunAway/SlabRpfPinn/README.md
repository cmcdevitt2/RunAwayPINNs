# SlabRpfPinn

CPU finite-volume and JAX neural-network tools for the steady relativistic
runaway-electron first-passage problem.

Production workflow is script-driven. Slurm provides resources; JSON files
provide run configuration; Python scripts generate data, train models, and
validate results. No notebook state is required.

## Project structure

User entry points:

- `generate_fv_dataset.py` — generate and analyze FV training data.
- `train_model.py` — train selected mode from `run_configs/train.json`.
- `validate_model.py` — fresh CPU-FV validation and model plots.

Automatic analytics helpers:

- `core/fv_dataset.py` — dataset analytics and coverage plots.
- `core/training_artifacts.py` — automatic loss-history plots.

Core implementation (`core/`):

- `core/rpf_fv_cpu.py` — CPU FV grids, coefficients, operators, and adjoint solve.
- `core/fv_dataset.py` — FV generation and memory-mapped dataset storage.
- `core/model.py` — MLP/DeepONet architectures and prediction transforms.
- `core/pde.py` — coefficients, residuals, boundaries, and collocation sampling.
- `core/training.py` — shared data, physics, SOAP, SSBroyden, and active loops.
- `core/training_config.py` — one hierarchical model/loss/optimizer schema.
- `core/training_artifacts.py` — checkpoints, histories, manifests, and plots.

Distributed orchestration:

- `core/fv_distributed.py` — one FV rank per node.

Other scientific code:

- `FokkerPlanck-Plasma0d/` — PETSc forward-solver prototype.
- `rpf_fv_pinn_scientific_reference_v1.tex` — scientific reference.
- `docs/` — cluster and dependency notes.

## Environment

On Perlmutter:

```bash
module purge
module load python
conda activate /pscratch/sd/j/jsarnaud/conda-envs/deeprunaway-warp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/pscratch/sd/j/jsarnaud/matplotlib-cache
mkdir -p "$MPLCONFIGDIR"
```

Verify inside allocation:

```bash
which python
python --version
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

JAX must report `gpu` for model training and validation.

## Configuration

Edit these files:

- `run_configs/fv_dataset.json` — parameter domain, FV grid, Sobol sampling,
  dataset path, and dataset run directory.
- `run_configs/train.json` — mode, architecture, losses, optimizer,
  checkpoints, dataset manifest, optional low-p fill count, and training output
  paths. It does not duplicate FV-generation settings.
- `run_configs/validate_model.json` — model path, optional model and saved-
  dataset manifests, fresh-case count, PDE chunk size, validation output, and
  plot basename.

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the meaning of every
available field.

Select training mode in `train.json`:

```json
"mode": "data"
```

or:

```json
"mode": "physics"
```

Data mode supports MLP and DeepONet. Physics mode uses pointwise MLP PDE and
boundary losses with optional SSBroyden refinement.

Every new training run needs new `run_dir`, `output_dir`, and `checkpoint_dir`
values. Existing completed runs are protected from overwrite.

## Workflow

Obtain Slurm allocation manually. Then run scripts without runtime arguments.

### Generate FV dataset

```bash
python generate_fv_dataset.py
```

For multi-node allocations, script launches one CPU task per node. It writes a
directory-backed memory-mapped dataset under `data/`, then automatically writes
dataset analytics and coverage plots under `runs/`.

### Train model

```bash
python train_model.py
```

Script detects visible GPUs. Data/DeepONet and SOAP physics mode use all
visible local GPUs and launch one synchronized JAX process per node on
multi-node allocations. In a multi-node physics run, SSBroyden refinement is
performed by rank 0 and its parameters are broadcast to the other ranks.

Training writes final parameters, metadata, summary, explicit loss history,
loss plot, periodic checkpoints, and a hashed completion manifest.

### Validate model

Edit `run_configs/validate_model.json`, then run:

```bash
python validate_model.py
```

Validation uses fresh CPU-FV cases by default. Set `dataset_path` in
`validate_model.json` to evaluate a saved dataset instead. It saves scalar
metrics, prediction correlations, and selected-case FV RPF/model RPF/error/
PDE-residual plots.

## Artifact ownership

- `data/` contains active FV datasets.
- `runs/` contains active dataset and model artifacts.

Manifests and checksums prevent training against incomplete or mismatched
datasets. Atomic writes prevent readers from seeing partial artifacts.

## Numerical contract

CPU solver uses logarithmic momentum cells, uniform theta cells with
`xi = cos(theta)`, adaptive FV low-energy placement, absorbing low-energy
failure, and successful upper-energy escape. Preserve this contract when
changing sampling, architecture, or loss configuration.
