# HiPerGator GPU batch execution

Run the forward solver through Slurm on a GPU node. Login nodes are for
editing, inspection, syntax/config checks, and submission only.

## Shared GPU execution contract

Production kinetic execution is GPU-only. There is no CPU kinetic operator,
CPU sparse solve, or CPU fallback. A failed GPU allocation or failed CUDA,
Warp, or cuDSS preflight is a failed run that must be fixed, not bypassed.
Warp, cuDSS, and any future JAX component must use the same allocated CUDA
device. The solver uses FP64 Warp buffers and cuDSS direct solves.

The supported project environment is `SlabFokkerPlanck/.venv`, addressed as
`./.venv` from this directory. Activate it before Python checks and inside
every batch script:

```bash
source .venv/bin/activate
```

Keep generated data, logs, caches, and model data out of Git. Coupled runs
require local, untracked OpenADAS bundles under `data/openadas/`; see
[OPENADAS.md](OPENADAS.md).

## Environment and dependency installation

HiPerGator site settings are not portable to Perlmutter. Consult UF's
[GPU resource](https://docs.rc.ufl.edu/resources/gpus/),
[GPU access](https://docs.rc.ufl.edu/scheduler/gpu_access/), and
[CUDA usage](https://docs.rc.ufl.edu/software/apps/cuda/usage/) pages before
choosing a module or GPU type. The B200 partition requires CUDA 12.8.1 or
newer when using a system CUDA installation.

Use a fresh project-local environment. Do not copy a Perlmutter module path
or a conda environment into HiPerGator:

```bash
module spider python
module spider cuda
module purge
module load python/<site-approved-version>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Preferred pip-managed CUDA 13 target, when the allocated B200 driver supports
CUDA 13:

```bash
python -m pip install \
  'nvmath-python[cu13]==1.0.0' \
  'jax[cuda13]==0.11.1' \
  'numpy==2.5.2' \
  'scipy==1.18.1' \
  'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0+cu13-py3-none-manylinux_2_28_x86_64.whl#sha256=0dc6460a0fbb8cb68cc354c4775f5002347c14e419af0028d36cafb0f6eb41b6'
python -m pip check
```

This project-local environment uses pip CUDA libraries. In the batch script,
run `unset LD_LIBRARY_PATH` after module setup so a system CUDA module does not
override them. If the B200 driver cannot run CUDA 13, use the UF-supported
system CUDA module and a separate environment with `jax[cuda12-local]`, the
PyPI Warp CUDA 12 package, and the matching nvmath system-CTK/cuDSS setup.
Verify every CUDA-dependent package source; never mix CUDA 12 and CUDA 13
packages or pip, conda, and system CUDA libraries without a compatibility test.

The standalone forward solver requires Warp, NumPy, SciPy, and nvmath/cuDSS.
JAX is not used by the standalone solver, but remains part of this validated
environment for future JAX/bulk components; when present, require its GPU
backend and the same visible CUDA device as Warp and cuDSS.

## Preflight

From `SlabFokkerPlanck`:

```bash
source .venv/bin/activate
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

Keep FP64 enabled. If a future bulk model uses JAX, require
`jax.default_backend() == "gpu"` and verify its device matches Warp and cuDSS.

## Example B200 job

Create log directories before submission because Slurm opens output paths
before the script body runs. Adjust partition, account, time, and memory to
local site policy.

```bash
mkdir -p logs outputs
cat > hipergator_b200.sbatch <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=slab-fp
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --account=<UFRC_GPU_ACCOUNT>
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

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

srun --ntasks=1 --cpus-per-task=14 --gpus-per-task=1 \
    --cpu-bind=cores python forward_fv_solver.py --config forward_fv_solver.toml
SLURM
sbatch hipergator_b200.sbatch
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
