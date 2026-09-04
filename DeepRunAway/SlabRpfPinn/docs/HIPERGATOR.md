# HiPerGator B200 workflow

This page contains HiPerGator-only resource and module guidance. It does not
apply to NERSC Perlmutter. Use [`PERLMUTTER.md`](PERLMUTTER.md) for Perlmutter
A100 jobs. Python solver code remains cluster-agnostic.

Install shared dependencies from [`DEPENDENCIES.md`](DEPENDENCIES.md). Use
direct VCS installation for the SSBFGS Optimistix branch; do not keep an
Optimistix source repository as an environment dependency.

## Login node and environment

Use HiPerGator login nodes for editing, repository inspection, small syntax or
TOML checks, environment installation, and job submission. Run Warp, JAX,
cuDSS, FV, and PINN workloads only through Slurm.

From this directory, use the project environment or a validated replacement:

```bash
cd /path/to/DeepRunAway/SlabRpfPinn
module spider python
module spider cuda
module spider nvmath
module spider cudss
source ../.venv/bin/activate
```

Select Python/CUDA modules supported by HiPerGator at submission time. Do not
copy Perlmutter `module purge`, CUDA 13, or A100 assumptions without checking
the HiPerGator driver and module stack. Follow the CUDA profile in
[`DEPENDENCIES.md`](DEPENDENCIES.md) that matches `nvidia-smi`.

## B200 resource request

Existing HiPerGator B200 scripts use `hpg-b200`, `b200`, and
`gpu:b200:1`. Confirm current names with `sinfo` before submission. Replace
both placeholders below with values supplied by HiPerGator project settings;
do not guess account or QOS:

```bash
sinfo
sbatch --account=<HIPERGATOR_ACCOUNT> --qos=<HIPERGATOR_QOS> \
  fv_robustness_b200.sbatch
```

For ordinary one-GPU studies, retain 28 total host threads:

```bash
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
```

Use one serial batch job to cycle cases and meshes. Do not submit one job per
case unless a separate campaign design authorizes it.

## Direct FV or PINN batch template

Save this template as `hipergator_b200.sbatch` in an approved persistent
project/scratch directory, replace
the account, QOS, and time placeholders, then submit it. Do not save a script
under `/tmp` or use temporary directories for inputs or outputs.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=slab-rpf-hpg
#SBATCH --account=<HIPERGATOR_ACCOUNT>
#SBATCH --qos=<HIPERGATOR_QOS>
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
if [[ "$gpu_name" != *"B200"* ]]; then
    echo "HiPerGator B200 allocation required, got: $gpu_name" >&2
    exit 1
fi

python - <<'PY'
import importlib.metadata as metadata
import jax
import nvmath
import warp
from nvmath.bindings import cudss

print("JAX:", metadata.version("jax"), jax.default_backend(), jax.devices())
print("Warp:", metadata.version("warp-lang"), warp.get_devices())
print("nvmath-python:", metadata.version("nvmath-python"))
for package_name in ("nvidia-cudss-cu13", "nvidia-cudss-cu12"):
    try:
        print("cuDSS package:", package_name, metadata.version(package_name))
        break
    except metadata.PackageNotFoundError:
        pass
else:
    raise SystemExit("cuDSS runtime package required")
if jax.default_backend() != "gpu":
    raise SystemExit("JAX GPU backend required")
if not any(getattr(d, "is_cuda", False) for d in warp.get_devices()):
    raise SystemExit("Warp CUDA device required")
major = cudss.get_property(nvmath.LibraryPropertyType.MAJOR_VERSION)
minor = cudss.get_property(nvmath.LibraryPropertyType.MINOR_VERSION)
if (major, minor) != (0, 8):
    raise SystemExit(f"cuDSS 0.8 required, got {major}.{minor}")
PY

MODE="${MODE:-fv}"
CONFIG="${CONFIG:-adjoint_fv_solver.toml}"
case "$MODE" in
    fv)       PROGRAM=(adjoint_fv_solver.py) ;;
    pinn)     PROGRAM=(pinn_training.py) ;;
    campaign) PROGRAM=(fv_robustness_campaign.py) ;;
    *)        echo "MODE must be fv, pinn, or campaign" >&2; exit 2 ;;
esac
srun --ntasks=1 --cpus-per-task=28 --gpus-per-task=1 \
    --cpu-bind=cores python "${PROGRAM[@]}" --config "$CONFIG"
```

Create log directories before submission because Slurm opens output paths
before executing the script:

```bash
mkdir -p logs outputs
MODE=fv CONFIG=adjoint_fv_solver.toml sbatch hpg_b200.sbatch
MODE=pinn CONFIG=pinn_training_smoke.toml sbatch hpg_b200.sbatch
```

For committed campaign scripts, submit exact script name and pass account/QOS
overrides explicitly:

```bash
mkdir -p logs outputs/fv_robustness
python fv_robustness_campaign.py --config fv_robustness_campaign.toml --dry-run
sbatch --account=<HIPERGATOR_ACCOUNT> --qos=<HIPERGATOR_QOS> \
  fv_robustness_b200.sbatch
```

## Interactive smoke test

Use HiPerGator interactive GPU mechanism only if current scheduler policy
permits it. Request one B200, then run same preflight and persistent
`pinn_training_smoke.toml` configuration with `srun`. Keep output in
`outputs/`, `logs/`, or approved `$SCRATCH`/project storage.

## Monitoring and results

```bash
squeue -u "$USER"
squeue -j <JOBID>
tail -f logs/<job-name>-<JOBID>.out
sacct -j <JOBID> \
  --format=JobID,JobName,Partition,Account,QOS,AllocTRES,State,ExitCode,Elapsed,MaxRSS
```

Record job ID, effective account/QOS/partition, GPU name, driver, Python
package versions, Git commit, TOML, residuals, probability bounds, and peak
memory. Retain stdout/stderr with outputs. Never use `/tmp`, `$TMPDIR`,
`mktemp`, or node-local temporary directories for runtime files.

## Stop conditions

Stop when GPU, Warp, JAX, or cuDSS preflight fails. Do not set
`JAX_PLATFORMS=cpu` and do not substitute a CPU sparse solver. Stop when B200
memory or cuDSS workspace is insufficient; qualify a smaller grid or use an
approved resource option without changing solver mathematics.
