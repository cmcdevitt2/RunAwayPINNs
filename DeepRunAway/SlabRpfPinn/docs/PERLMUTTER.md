# NERSC Perlmutter GPU workflow

This page is the canonical NERSC Perlmutter runbook. It is intentionally separate from
[`HIPERGATOR.md`](HIPERGATOR.md), whose examples target Hipergator B200 nodes.
Do not combine the two systems' partitions, constraints, GPU names, or module
assumptions.

The commands below have been exercised on a Perlmutter A100 allocation. They
are execution checks, not FV-grid or PINN-quality qualification; production
qualification still requires the numerical checks in the scientific reference.

## Hardware and resource requests

NERSC's current Perlmutter documentation describes the GPU partition as NVIDIA
A100 (Ampere): one AMD EPYC 7763 CPU with 64 physical cores, four GPUs per
node, 256 GB of host memory, and either 40 GiB or 80 GiB of GPU-attached
memory per GPU. The 80 GiB nodes are selected with `gpu&hbm80g`; the ordinary
nodes are selected with `gpu` (or `gpu&hbm40g`). Perlmutter is not a B200
system according to the current NERSC hardware pages.

Perlmutter examples use the GPU constraint and an explicit GPU resource
request. They do not use the Hipergator `hpg-b200` partition or
`gpu:b200:1` GRES:

```bash
# One GPU, one serial Python process, 28 total host threads.
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5276
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
```

The live scheduler currently displays the ordinary GPU partition as
`gpu_ss11`, but that is not a script setting. The accepted user-facing form
selects the partition through the GPU constraint and public QOS name and does
not set a partition. In a live job record, `--qos=debug` appears as the
internal `gpu_debug` QOS and `--qos=interactive` appears as
`gpu_interactive`; both route to the site-selected GPU partition. Explicit
partition or internal-QOS directives are not portable and were rejected by
the current policy. Do not add them to the template.

Use `regular` for production FV or PINN work. The current GPU QOS limits list
`regular` at 48 hours, `shared` at 48 hours, `interactive` at 4 hours, and
`debug` at 30 minutes. The current node limits are 8 for `debug` and 4 for
`interactive`; this project template requests one node. NERSC says that
`debug` is for development/testing, not production. Jobs using one or two
GPUs may use `shared`, but the current policy gives a one-GPU shared job 16
CPU cores and 64 GB of host RAM; that is not enough for this project's
ordinary 28-thread contract. Use a whole-node QOS/resource shape when 28 host
threads or cuDSS workspace require it.

Project scheduling rule: run at most one active interactive job. Request no
more than half of any finite current QOS node limit. Therefore current
`interactive` limit 4 permits at most 2 project-requested nodes, and current
`debug` limit 8 permits at most 4. Recheck NERSC policy when limits change;
never increase project limits automatically. Use one node for ordinary
SlabRpfPinn FV/PINN work unless a separate resource plan authorizes more.

The `--cpus-per-task` value is in logical CPUs on Perlmutter (two hardware
threads per physical core). The project contract is 28 total host threads,
so keep one task, request 28 CPUs, and cap threaded libraries explicitly:

```bash
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
export SLURM_CPU_BIND=cores
```

For a memory-sensitive qualification, `--constraint="gpu&hbm80g"` is the
current NERSC spelling. Quote that expression on an interactive command line
because `&` is a shell operator. A 4096 x 1024 FP64 FV state and its cuDSS
factorization can require substantially more memory than the raw CSR values;
measure peak usage on the target A100 rather than extrapolating from CSR size.
Start with a reduced qualification case, then record `MaxRSS` and the GPU
memory requirement before attempting the recommended grid. Do not change the
physical domain or solver mathematics to fit a node.

## Login nodes versus compute nodes

Login nodes are for editing, source inspection, small syntax/configuration
checks, environment setup, and submitting/monitoring jobs. NERSC prohibits
compute- or memory-intensive applications there and applies per-user login
limits. In particular, a login-node Python import or CPU result is not a GPU
qualification and is not evidence that Warp/cuDSS is usable.

Use `salloc` for a short interactive GPU check and `sbatch` for production
FV/PINN work. This project has the GPU account `m5276`; Slurm records it as
`m5276_g`. The account and any project-storage path must be confirmed in Iris
if the allocation changes:

```bash
salloc --nodes=1 --qos=interactive --time=01:00:00 \
  --constraint='gpu&hbm40g' --gpus-per-task=1 --account=m5276
```

NERSC documents a GPU account requirement for interactive Perlmutter jobs;
use the GPU-enabled account shown for the user/project. The live allocation
for this project maps `m5276` to `m5276_g`; do not infer an account or QOS from
the Hipergator scripts. Do not start a second interactive allocation while one
project interactive job is active.

After `salloc` grants the node, run the setup and smoke checks in that shell:

```bash
cd /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn
module purge
module load python/3.12-26.1.0
unset LD_LIBRARY_PATH
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28

srun --ntasks=1 --cpus-per-task=28 --gpus-per-task=1 \
  --cpu-bind=cores python adjoint_fv_solver.py --config adjoint_fv_solver.toml
srun --ntasks=1 --cpus-per-task=28 --gpus-per-task=1 \
  --cpu-bind=cores python pinn_training.py --config pinn_training_smoke.toml
```

The first command is the small 256x128 FV case. The second command runs one
training and one validation case with one Adam and one SSBroyden step. It
checks the GPU execution path and output writing only; its validation error
must not be used as scientific evidence. Use `exit` to release the allocation.

## Batch and job-log workflow

Create Slurm output directories before submission because Slurm opens the
output paths before the script starts:

```bash
mkdir -p logs outputs
sbatch perlmutter_a100.sbatch
squeue -u "$USER"
squeue -j <JOBID>
tail -f logs/<job-name>-<JOBID>.out
sacct -j <JOBID> \
  --format=JobID,JobName,Partition,Account,AllocTRES,State,ExitCode,Elapsed,MaxRSS
```

Use `squeue` for pending/running jobs and `sacct` for the final state, exit
code, elapsed time, allocation, and memory. Keep the stdout/stderr files with
the result directory. The checked-in
[`perlmutter_a100.sbatch`](../perlmutter_a100.sbatch) template performs the
GPU/backend preflight and can dispatch FV or PINN mode:

```bash
mkdir -p logs outputs

# Default debug smoke/preflight (maximum 30 minutes):
sbatch perlmutter_a100.sbatch fv adjoint_fv_solver.toml

# PINN smoke/preflight:
sbatch perlmutter_a100.sbatch pinn pinn_training_smoke.toml

# Production FV run: override QOS/time, and optionally select 80 GiB A100s.
sbatch --qos=regular --time=06:00:00 \
  --constraint='gpu&hbm80g' perlmutter_a100.sbatch fv adjoint_fv_solver.toml
```

The template's `--mem=200G` and walltime are starting values, not a
qualification result. The extra CPUs available on an exclusive Perlmutter GPU
node are intentionally not consumed: the repository contract keeps ordinary
one-GPU studies at 28 total host threads. If the 40 GiB nodes are sufficient,
use `--constraint='gpu&hbm40g'` or leave the default `gpu` constraint.

For this repository, the checked-in template defaults to the public `debug`
QOS and one node. Submit it with `--qos=debug --time=00:30:00` for a smoke
check, or override to `--qos=regular` with a production walltime. Never add a
partition option. Create `logs/` before `sbatch`; Slurm opens the log paths
before the script body starts.

## Modules and Python environments

Perlmutter uses Lmod. Inspect the live installation rather than relying on a
remembered module version:

```bash
module list
module spider python
module spider cudatoolkit
module spider warp
module spider nvmath
module spider cudss
```

NERSC documents `module load python` for its supported Anaconda Python and
`cudatoolkit` for the CUDA toolkit, including `nvcc`. It also provides
`PrgEnv-gnu` and `PrgEnv-nvidia`; this Python-only workflow does not require a
compiler module unless a dependency must be built. Use one CUDA provider:
either the NERSC `cudatoolkit` module or CUDA packages inside the environment,
not both, unless a tested compatibility plan explicitly requires it.

Install all Python and GPU dependencies using the shared
[`DEPENDENCIES.md`](DEPENDENCIES.md) guide. The current validated Perlmutter
profile uses pip-provided CUDA 13 libraries. It installs the SSBFGS Optimistix
branch directly from its VCS URL; it does not require or create a persistent
Optimistix source checkout.

The project convention is the neighboring environment:

```bash
cd /path/to/DeepRunAway/SlabRpfPinn
module purge
module load python/3.12-26.1.0
unset LD_LIBRARY_PATH
source ../.venv/bin/activate
python -c 'import sys; print(sys.executable)'
```

If `../.venv` is absent or was built for another machine, the validated local
replacement is `./.venv` in this checkout. Build any other replacement under
`$PSCRATCH` or an appropriate project software area after selecting versions
with `module spider`. Keep the package/version manifest outside generated
output and validate the complete import set (`numpy`, `scipy`, JAX, Optax,
Equinox, the SSBFGS Optimistix branch, Warp, and nvmath-python) in a GPU
allocation. Do not install dependencies as a persistent repository checkout.
Do not assume the Hipergator B200 environment is portable to Perlmutter.

## JAX, Warp, and cuDSS gate

Run this probe inside the allocation before any expensive case. JAX must use
its GPU backend, Warp must expose the allocated CUDA device, and the cuDSS
binding must import. The FV solver must continue to report its
`gpu-warp-cudss` execution path; a CPU fallback is invalid.

```bash
srun --gpus=1 --cpu-bind=cores python - <<'PY'
import importlib.metadata as metadata
import jax
import warp
import nvmath
from nvmath.bindings import cudss

print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())
print("nvmath-python:", metadata.version("nvmath-python"))
print("cuDSS binding:", cudss)

if jax.default_backend() != "gpu":
    raise SystemExit("JAX GPU backend required")
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise SystemExit("JAX GPU device required")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit("Warp CUDA device required")
PY
```

NERSC's Perlmutter pages document CUDA and JAX, but do not publish a
Perlmutter `nvmath-python` or cuDSS module/version matrix. The current live
module probe used for this audit did not find modules named `nvmath` or
`cudss`, although it did expose `cudatoolkit` versions. NVIDIA's nvmath
documentation does expose `nvmath.bindings.cudss` and warns that its CUDA
minor-version and driver compatibility must match the installed libraries.
The validated stack for this checkout is JAX 0.11.1/jaxlib 0.11.1 with the
CUDA 13 plugin, Warp 1.17.0, nvmath-python 1.0.0, and cuDSS 0.8.0.10. With
`cudatoolkit/13.2` left on `LD_LIBRARY_PATH`, JAX and Warp still detected the
A100 but the cuDSS property probe reported major version 0; after `module
purge` and `unset LD_LIBRARY_PATH`, it reported cuDSS 0.8.0. Use the clean
module setup above and rerun the probe after any module or package change.
Record the Python package versions, cuDSS library version, CUDA toolkit,
driver, and GPU model in every qualification log. If the probe or the solver
cannot load cuDSS on A100, stop and resolve the environment; do not switch to
CPU or substitute another sparse solver.

## Storage and generated output

Use `$SCRATCH` or `$PSCRATCH` for active Perlmutter job inputs, logs, datasets,
checkpoints, and plots. NERSC describes Perlmutter scratch as high-performance
Lustre storage available on compute nodes, but it is purgeable after periods
of inactivity. Check quota with `showquota` on a login node and archive
irreplaceable results to HPSS or the project's permanent storage.

Use project Common/Community storage (for example,
`/global/common/software/<PROJECT>/` or `/global/cfs/cdirs/<PROJECT>/`, with
the actual project path supplied by NERSC) for shareable environments, source
artifacts, or longer-lived results as appropriate. Common storage is mounted
read-only on compute nodes, so do not place runtime-generated output there
without checking the intended access mode. Never hard-code another user's
path.

For this repository, keep generated `logs/`, `outputs/`, datasets, models,
plots, caches, and Slurm files under the checkout on `$PSCRATCH` or another
approved project result directory, not in Git. Do not use `/tmp`, `$TMPDIR`,
`mktemp`, or node-local temporary directories for inputs or outputs. Copy the
TOML and commit identifier beside each result.

## Sources and update rule

Consult these pages immediately before a production submission because
hardware, modules, and QOS policy can change:

- [Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)
- [Running jobs on Perlmutter](https://docs.nersc.gov/systems/perlmutter/running-jobs/)
- [Basics of running jobs](https://docs.nersc.gov/jobs/)
- [Queues and charges](https://docs.nersc.gov/jobs/policy/)
- [Interactive jobs](https://docs.nersc.gov/jobs/interactive/)
- [Resource usage policies](https://docs.nersc.gov/policies/resource-usage/)
- [Using Python on Perlmutter](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/)
- [CUDA at NERSC](https://docs.nersc.gov/development/programming-models/cuda/)
- [Lmod](https://docs.nersc.gov/environment/lmod/)
- [NERSC filesystems overview](https://docs.nersc.gov/filesystems/)
- [Perlmutter scratch](https://docs.nersc.gov/filesystems/perlmutter-scratch/)
- [NVIDIA nvmath-python installation](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html)
- [NVIDIA nvmath-python cuDSS bindings](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cudss.html)
- [CrunchOptimizer PINN Optimistix installation](https://github.com/CrunchOptimizer/PINNs)
- [Shared dependency installation](DEPENDENCIES.md)
