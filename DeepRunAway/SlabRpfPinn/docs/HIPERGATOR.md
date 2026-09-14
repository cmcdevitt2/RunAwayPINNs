# HiPerGator GPU workflow

This page contains only HiPerGator resource guidance. The Python workflow is
the same config-driven workflow described in the root `README.md`; resource
syntax and environment modules are cluster-specific.

## Environment and resources

Use HiPerGator login nodes for editing and submission. Run FV generation and
model training through Slurm. Confirm the current account, QOS, partition,
GPU type, and Python/CUDA modules with the site scheduler before submitting.

```bash
sinfo
module spider python
module spider cuda
```

Example placeholder allocation:

```bash
salloc --nodes=1 --ntasks=1 --gpus-per-task=4 \
  --cpus-per-task=64 --account=<ACCOUNT> --qos=<QOS> --time=<HH:MM:SS>
```

Inside the allocation, activate the validated project environment and disable
JAX preallocation:

```bash
cd /path/to/DeepRunAway/SlabRpfPinn
source /path/to/validated/environment/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

Use the CUDA/JAX package versions validated for the allocated GPU. Do not
assume Perlmutter modules or GPU names are portable to HiPerGator.

## Config-driven workflow

Edit:

- `run_configs/fv_dataset.json` for CPU FV generation;
- `run_configs/train.json` for model mode, architecture, losses, and outputs;
- `run_configs/validate_model.json` for validation and plots.

Then run the stages without CLI arguments:

```bash
python generate_fv_dataset.py
python train_model.py
python validate_model.py
```

For a batch run, place the same environment setup in a site-specific Slurm
wrapper and submit it with `sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=slab-rpf-train
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /path/to/validated/environment/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python train_model.py
```

```bash
mkdir -p logs
sbatch train_model.sbatch
```

The root driver launches its own one-process-per-node `srun` step when the
allocation spans multiple nodes. Do not add a second competing multi-node
launcher around it.

The FV stage uses CPUs. Data/DeepONet and SOAP physics training can use one
synchronized JAX process per node when launched from a multi-node allocation.
For multi-node physics training, SSBroyden runs on rank 0 and broadcasts the
refined parameters.

Each stage writes durable artifacts and a manifest. Use new run directories
for new dataset, training, or validation runs; existing completed runs are
protected from overwrite.

## Monitoring

```bash
squeue -u "$USER"
sacct -j <JOBID> \
  --format=JobID,JobName,Partition,Account,QOS,AllocTRES,State,ExitCode,Elapsed,MaxRSS
```

Record the allocation, environment, configuration files, model metadata,
loss history, and peak memory with production results.
