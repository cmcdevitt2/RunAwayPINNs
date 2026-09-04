# Hipergator GPU batch execution

Run the forward solver through Slurm on a GPU node. Login nodes are for
editing, inspection, syntax/config checks, and submission only.

## Environment and preflight

From `DeepRunAway/SlabFokkerPlanck`:

```bash
source ../.venv/bin/activate
python -m py_compile forward_fv_solver.py
python - <<'PY'
import tomllib
from pathlib import Path
with Path("forward_fv_solver.toml").open("rb") as stream:
    tomllib.load(stream)
print("TOML: OK")
PY
```

Run the following only inside an allocation:

```bash
python - <<'PY'
import nvmath
import warp as wp
wp.init()
if not wp.is_cuda_available():
    raise SystemExit("Warp CUDA is unavailable")
print("Warp:", wp.get_devices())
print("nvmath:", nvmath.__version__)
PY
```

The forward solver is GPU-only. Keep FP64 enabled and do not hide a failed
allocation with a CPU fallback. If a future bulk model uses JAX, additionally
require `jax.default_backend() == "gpu"` and verify its CUDA device matches
Warp's device.

## Example B200 job

Create log directories before submission because Slurm opens output paths
before the script body runs. Adjust partition/account/time to local site policy.

```bash
mkdir -p logs outputs
cat > /tmp/slab-fp.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=slab-fp
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

nvidia-smi
python - <<'PY'
import warp as wp
import nvmath
wp.init()
if not wp.is_cuda_available():
    raise SystemExit("CUDA GPU required")
print("Warp:", wp.get_devices())
print("nvmath:", nvmath.__version__)
PY

python forward_fv_solver.py --config forward_fv_solver.toml
SLURM
sbatch /tmp/slab-fp.slurm
```

Use a site-approved account/QOS if required. Keep the batch script outside Git
unless it is made project-generic; never hard-code a personal scratch path.

## Resource guidance

The SlabRpfPinn cuDSS study gives planning evidence, not a guarantee for this
forward operator. On B200, representative adjoint peaks were approximately:

| mesh | peak device memory |
|---|---:|
| 1024x512 | 1.3 GiB |
| 2048x1024 | 3.0 GiB |
| 4096x2048 | 10.6 GiB |
| 8192x4096 | 41.4 GiB |
| 16384x8192 | 171 GiB, near the 178-GiB limit |

Forward augmented systems can differ. Start below production resolution,
measure peak memory and cuDSS factor fill, and leave headroom. Default cuDSS
nested dissection was the best tested production ordering; geometric ordering
and BTF/COLAMD had substantially worse fill or runtime. Do not claim multi-GPU
scaling from the raw API probe; the qualified path is one GPU.

## Monitoring and diagnostics

```bash
squeue -j <JOBID>
tail -f logs/slab-fp-<JOBID>.out
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Record job ID, Git commit, config copy, environment, GPU model, solver
residuals, state bounds, timing, and memory. Separate each case's output path.
Use `sacct` after completion; `squeue` alone does not expose failed jobs.

Common failures:

- missing TOML: pass the explicit `--config` path and use
  `cd "$SLURM_SUBMIT_DIR"`;
- invalid CUDA device: the job lacks a valid GPU allocation or environment;
- cuDSS out-of-memory: reduce mesh, request a larger GPU, or quantify fill;
- nonfinite/negative state: stop and investigate timestep, boundary, or
  coefficient assembly; do not train or couple from that output.
