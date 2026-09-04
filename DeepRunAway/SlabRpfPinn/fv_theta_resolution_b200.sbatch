#!/usr/bin/env bash
# Angular-grid resolution study; one B200 job, 28 total CPU threads.
# Submit from SlabRpfPinn.
#SBATCH --job-name=slab-rpf-theta-resolution
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$gpu_name" != *"B200"* ]]; then
    echo "B200 allocation required, got: $gpu_name" >&2
    exit 1
fi

python - <<'PY'
import jax
import warp
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX GPU backend required, got {jax.default_backend()!r}")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit(f"Warp CUDA device required, got {warp.get_devices()!r}")
print("GPU:", __import__("subprocess").check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip())
print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())
PY

python fv_robustness_campaign.py --config fv_theta_resolution.toml
