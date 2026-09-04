#!/usr/bin/env bash
# Perlmutter A100 batch template for the SlabFokkerPlanck forward solver.
# Submit from this directory after creating logs/.
#
# Default: one GPU, one serial process, debug QOS (30-minute limit).
# For an 80-GB A100, submit with:
#   sbatch --constraint='gpu&hbm80g' perlmutter_a100.sbatch <config>
# For a standard 40-GB A100, use the default gpu&hbm40g constraint below.
# Perlmutter selects its internal partition from constraint and QOS. This
# script intentionally contains no --partition directive.

#SBATCH --job-name=slab-fp-a100
#SBATCH --account=m5276
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.13-26.8.0
source .venv/bin/activate

config="${1:-forward_fv_solver.toml}"
if [[ ! -f "$config" ]]; then
    echo "configuration file not found: $config" >&2
    exit 2
fi

# The environment uses pip-managed CUDA 13 libraries. Prevent the site CUDA
# module path from overriding them at runtime.
unset LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)"
if [[ "$gpu_name" != *"A100"* ]]; then
    echo "Perlmutter A100 allocation required, got: $gpu_name" >&2
    exit 1
fi
echo "GPU: $gpu_name ($gpu_memory)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
# These are scheduler-selected observations, not requested directives.
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-<unset>}"
echo "git_commit=$(git rev-parse HEAD)"
echo "config=$config"
sha256sum "$config"
python --version
python -m pip freeze
module -t list 2>&1 || true

# Do not start the solver unless CUDA, Warp, JAX, FP64, and cuDSS are usable.
srun --ntasks=1 --cpus-per-task=128 --gpus-per-task=1 \
    --cpu-bind=cores python - <<'PY'
import importlib.metadata as metadata
import jax
import jax.numpy as jnp
import nvmath
import warp as wp
from nvmath.bindings import cudss

wp.init()
warp_device = wp.get_device("cuda:0")
if not warp_device.is_cuda:
    raise SystemExit(f"Warp CUDA device required, got {warp_device!r}")
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX GPU backend required, got {jax.default_backend()!r}")
jax.config.update("jax_enable_x64", True)
jax_value = jnp.ones(1, dtype=jnp.float64)
jax_value.block_until_ready()
warp_value = wp.zeros(1, dtype=wp.float64, device="cuda:0")
major = cudss.get_property(nvmath.LibraryPropertyType.MAJOR_VERSION)
minor = cudss.get_property(nvmath.LibraryPropertyType.MINOR_VERSION)
patch = cudss.get_property(nvmath.LibraryPropertyType.PATCH_LEVEL)
print("Warp:", metadata.version("warp-lang"), warp_device)
print("Warp FP64:", warp_value.dtype, warp_value.device)
print("JAX:", metadata.version("jax"), jax.default_backend(), jax.devices("gpu"))
print("JAX FP64:", jax_value.dtype, jax_value.devices())
print("nvmath-python:", metadata.version("nvmath-python"))
print("cuDSS:", f"{major}.{minor}.{patch}")
PY

srun --ntasks=1 --cpus-per-task=128 --gpus-per-task=1 \
    --cpu-bind=cores python forward_fv_solver.py --config "$config"
