# NERSC Perlmutter workflow

This is the Perlmutter runbook for the current config-driven workflow. It
does not use a notebook, GPU FV solver, cuDSS, or legacy TOML entry points.
CPU finite-volume data generation and GPU model training are separate stages.

## Environment and allocation

Use login nodes only for editing, configuration checks, and job submission.
Run dataset generation and model training inside an allocation.

```bash
salloc --nodes=1 --qos=interactive --time=01:00:00 \
  --constraint=gpu --gpus=4 --account=<GPU_ACCOUNT>
```

Use the account and constraint currently available to the project. For
production, request `regular` and an appropriate walltime instead of the
interactive QOS. Do not start another interactive allocation while one is
already active.

After the allocation is granted, enter a compute-node shell and run the
config-driven scripts there:

```bash
srun --ntasks=1 --gpus-per-task=4 --cpus-per-task=128 \
  --cpu-bind=cores --pty bash
```

Inside that shell:

```bash
cd /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn
module purge
module load python
conda activate /pscratch/sd/j/jsarnaud/conda-envs/deeprunaway-warp

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

Training requires a JAX GPU backend. FV generation is CPU work and should use
the CPUs assigned by Slurm.

## Batch jobs

For repeatable production runs, create a small Slurm wrapper. The root scripts
read the JSON configuration and launch their own multi-node `srun` steps, so
the wrapper should invoke them directly from the batch shell.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=slab-rpf-train
#SBATCH --account=<GPU_ACCOUNT>
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load python
conda activate /pscratch/sd/j/jsarnaud/conda-envs/deeprunaway-warp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

python train_model.py
```

Submit it after creating the log directory:

```bash
mkdir -p logs
sbatch train_model.sbatch
squeue -j <JOBID>
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES
```

For a multi-node data or SOAP-physics run, change `--nodes` and
`--gpus-per-node`; the driver detects the allocation and launches one process
per node. For FV generation, use a CPU allocation instead (`--constraint=cpu`)
and run `python generate_fv_dataset.py` from the batch shell.

## Configure one run

Edit only these JSON files:

- `run_configs/fv_dataset.json` — parameter domain, Sobol count, FV grid,
  CPU worker count, dataset path, and dataset run directory.
- `run_configs/train.json` — `mode`, architecture, loss terms, optimizer,
  GPU training, checkpoint, output settings, and optional low-p fill count. It
  intentionally does not duplicate FV-generation settings.
- `run_configs/validate_model.json` — model path, optional saved-dataset
  manifest, validation cases, CPU FV resolution, and plot output.

Every run must use new `run_dir`, `output_dir`, and `checkpoint_dir` values.
Completed run directories are intentionally not overwritten.

## Workflow

### 1. Generate the FV dataset

```bash
python generate_fv_dataset.py
```

The script reads `run_configs/fv_dataset.json`, generates CPU FV cases, writes
a directory-backed dataset under `data/`, and automatically writes coverage
analytics under `runs/`. With a multi-node allocation it launches one CPU
task per node through `srun`; do not manually start one generator per node.

### 2. Train the model

Set `mode` in `run_configs/train.json`, then run:

```bash
python train_model.py
```

For `mode: "data"` and SOAP `mode: "physics"`, the driver uses local GPUs
and, on a multi-node allocation, launches one synchronized JAX process per
node. In multi-node physics mode, SSBroyden runs on rank 0 and broadcasts the
refined parameters.

The training stage writes model parameters, metadata, loss history, loss
plots, periodic checkpoints, and a completion manifest under `runs/`.

### 3. Validate and analyze

Edit `run_configs/validate_model.json`, then run:

```bash
python validate_model.py
```

Validation loads the saved model, generates fresh CPU FV cases unless
`dataset_path` is set, and writes metrics plus correlation and selected-case
plots. It also evaluates the same JAX PDE residual used by the model.

## Monitoring

```bash
squeue -u "$USER"
sacct -j <JOBID> \
  --format=JobID,JobName,AllocTRES,State,ExitCode,Elapsed,MaxRSS
```

Keep stdout/stderr and generated artifacts under approved project or scratch
storage. Do not run production workloads on login nodes.
