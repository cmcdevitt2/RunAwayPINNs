#!/bin/bash -l
# Perlmutter A100 template for SlabRpfPinn.
#
# Default: one-node, one-GPU debug smoke/preflight job (30-minute QOS limit).
# For production-length work, override --qos and --time at submission; do not
# use this default debug request for a production campaign.
#
# Select the A100 memory class at submission if needed:
#   sbatch --constraint='gpu&hbm40g' perlmutter_a100.sbatch fv
#   sbatch --constraint='gpu&hbm80g' perlmutter_a100.sbatch fv
# Plain --constraint=gpu allows either A100 memory class.
#
# Modes:
#   fv [config]       adjoint_fv_solver.py
#   pinn [config]     pinn_training.py
#   campaign [config] fv_robustness_campaign.py

#SBATCH --job-name=slab-rpf-a100
#SBATCH --account=m5276
# NERSC selects the GPU partition from the constraint and QOS.
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Use the checked-out local environment by default. Set SLAB_ENV_DIR only if
# a separately validated environment is required.
env_dir="${SLAB_ENV_DIR:-$SLURM_SUBMIT_DIR/.venv}"

module purge
module load python/3.12-26.1.0
source "$env_dir/bin/activate"

# JAX/nvmath use pip-provided CUDA 13 libraries in this environment. Do not
# allow a site cudatoolkit LD_LIBRARY_PATH to override those libraries.
unset LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
export SLURM_CPU_BIND=cores

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | sed -n '1p')"
if [[ "$gpu_name" != *"A100"* ]]; then
    echo "Perlmutter A100 allocation required, got: $gpu_name" >&2
    exit 1
fi
if [[ "$gpu_memory" != *"40"* && "$gpu_memory" != *"80"* ]]; then
    echo "expected a 40/80 GB A100, got: $gpu_name ($gpu_memory)" >&2
    exit 1
fi

echo "GPU: $gpu_name ($gpu_memory)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "environment=$env_dir"

# The preflight must pass before any solver or training work is started.
srun --ntasks=1 --cpus-per-task=28 --gpus-per-task=1 \
    --cpu-bind=cores python - <<'PY'
import importlib.metadata as metadata
import jax
import nvmath
import warp
from nvmath.bindings import cudss

print("JAX:", metadata.version("jax"), jax.default_backend(), jax.devices())
print("Warp:", metadata.version("warp-lang"), warp.get_devices())
print("nvmath-python:", metadata.version("nvmath-python"))
print("cuDSS:", metadata.version("nvidia-cudss-cu13"))

if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX GPU backend required, got {jax.default_backend()!r}")
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise SystemExit(f"JAX CUDA device required, got {jax.devices()!r}")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit(f"Warp CUDA device required, got {warp.get_devices()!r}")

major = cudss.get_property(nvmath.LibraryPropertyType.MAJOR_VERSION)
minor = cudss.get_property(nvmath.LibraryPropertyType.MINOR_VERSION)
patch = cudss.get_property(nvmath.LibraryPropertyType.PATCH_LEVEL)
if (major, minor, patch) != (0, 8, 0):
    raise SystemExit(f"cuDSS 0.8.x required, got {major}.{minor}.{patch}")
PY

mode="${1:-fv}"
config="${2:-}"
case "$mode" in
    fv)
        config="${config:-adjoint_fv_solver.toml}"
        command=(adjoint_fv_solver.py --config "$config")
        ;;
    pinn)
        config="${config:-pinn_training.toml}"
        command=(pinn_training.py --config "$config")
        ;;
    campaign)
        config="${config:-fv_robustness_campaign.toml}"
        command=(fv_robustness_campaign.py --config "$config")
        ;;
    *)
        echo "usage: sbatch [Slurm overrides] perlmutter_a100.sbatch [fv|pinn|campaign] [config]" >&2
        exit 2
        ;;
esac

if [[ ! -f "$config" ]]; then
    echo "configuration file not found: $config" >&2
    exit 2
fi

echo "mode=$mode config=$config"
srun --ntasks=1 --cpus-per-task=28 --gpus-per-task=1 \
    --cpu-bind=cores python "${command[@]}"
