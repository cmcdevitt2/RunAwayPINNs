# Perlmutter

Internal runbook for production FV generation and GPU training/validation.
Use project-local `.venv`. Run production work inside Slurm allocations.

## Environment

Use existing project environment:

```bash
cd /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn
source .venv/bin/activate
```

If `.venv` does not exist, create it once and install `requirements.txt`.

For CPU FV generation, use a CPU allocation. For training and validation,
use a GPU allocation.

## Interactive allocation

```bash
salloc --nodes=1 --qos=interactive --time=01:00:00 \
  --constraint=gpu --gpus=4 --account=m5276
srun --ntasks=1 --gpus-per-task=4 --cpus-per-task=128 \
  --cpu-bind=cores --pty bash
```

Inside the compute shell:

```bash
cd /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn
module purge
module load python
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

## Batch scripts

Use one Slurm task per node. The root drivers launch required multi-node
steps. Set paths to new run directories for every production run.

```bash
mkdir -p logs
sbatch generate_fv_dataset.sbatch
sbatch train_model.sbatch
sbatch validate_model.sbatch
squeue -j <JOBID>
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES
```

Edit resource directives for walltime or node count before production runs.

## Production workflow

Edit these JSON files before each run:

- `configs/fv_dataset.json`
- `configs/train.json`
- `configs/validate_model.json`

Use new `run_dir`, `output_dir`, and `checkpoint_dir` values each run.

FV generation uses CPU workers. Training and validation use GPU JAX. Active
FV acquisition requires one JAX process. Multi-node runs require one task per
node, shared paths, and reachable hosts.
