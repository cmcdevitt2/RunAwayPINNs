#!/usr/bin/env bash
# Perlmutter CPU debug smoke template. No explicit partition: use public QOS plus constraint.
# Absolute paths keep submission independent of the caller's working directory.
#SBATCH --job-name=slab-petsc
#SBATCH --account=m5276
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=4
# Perlmutter CPU: 2 * floor(128 logical CPUs / 4 tasks) = 64.
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/FokkerPlanck-Plasma0d/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/FokkerPlanck-Plasma0d/logs/%x-%j.err

set -euo pipefail
cd /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/FokkerPlanck-Plasma0d

module purge
module load python/3.12-26.1.0
unset LD_LIBRARY_PATH

# Force the Cray-MPICH runtime into CPU mode; no GPU is allocated or used.
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export SLURM_CPU_BIND=cores

exec srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --cpu-bind=cores \
  /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/.petsc-cpu-venv/bin/python -u \
  /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/FokkerPlanck-Plasma0d/petsc_forward_solver.py \
  --coupled \
  --pmax 100.0 \
  --Np 1024 --Nxi 256 --t-end 0.1 --dtau-initial 1e-7
