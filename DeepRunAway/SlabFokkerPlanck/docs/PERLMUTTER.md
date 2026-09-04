# NERSC Perlmutter GPU batch execution

Run the forward solver through Slurm on a Perlmutter GPU node. Login nodes are
for editing, inspection, syntax/config checks, and submission only; do not run
the kinetic solver on a login-node GPU.

## Shared GPU execution contract

Production kinetic execution is GPU-only. There is no CPU kinetic operator,
CPU sparse solve, or CPU fallback. A failed GPU allocation or failed CUDA,
Warp, or cuDSS preflight is a failed run that must be fixed, not bypassed.
Keep the configured `cuda:0` as the first device in the process-visible CUDA
namespace. Warp, cuDSS, and any future JAX component must use that same CUDA
device. The solver's cuDSS descriptors are built from Warp-owned FP64 buffers;
do not launch a separate CPU implementation or remap devices inside a job.

The supported environment is `SlabFokkerPlanck/.venv`, addressed as `./.venv`
from this directory. Activate it on the compute node inside every batch
script:

```bash
source .venv/bin/activate
```

The current Perlmutter target, installed on 2026-09-04, is Python 3.13 with a
single pip-managed CUDA 13 family:

```text
warp-lang       1.17.0+cu13
jax             0.11.1
jaxlib          0.11.1
nvmath-python   1.0.0 (installed with [cu13])
nvidia-cudss-cu13 0.8.0.10 (via nvmath-python[cu13])
numpy           2.5.2
scipy           1.18.1
```

For a fresh rebuild on Perlmutter, select the current Python 3.13 module shown
by `module spider python` and install the pinned target below. The Warp CUDA 13
wheel is hosted by NVIDIA's GitHub releases because the PyPI wheel may use a
different CUDA runtime:

```bash
module purge
module load python/3.13-26.8.0   # verify with: module spider python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'nvmath-python[cu13]==1.0.0' \
  'jax[cuda13]==0.11.1' \
  'numpy==2.5.2' \
  'scipy==1.18.1' \
  'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0+cu13-py3-none-manylinux_2_28_x86_64.whl#sha256=0dc6460a0fbb8cb68cc354c4775f5002347c14e419af0028d36cafb0f6eb41b6'
python -m pip check
python -m pip freeze > "$SCRATCH/slab-fp-python-${USER}.txt"
```

This recipe uses pip-provided CUDA 13 libraries, including cuDNN, cuBLAS,
cuSPARSE, NCCL, and cuDSS. Do not mix it with a CUDA 12 package, conda CUDA
package, or a conflicting system CUDA module. If a site module is loaded by
default, record it and remove conflicting CUDA library paths before testing;
the JAX CUDA wheel can otherwise be overridden by `LD_LIBRARY_PATH`. Keep the
freeze output in scratch rather than committing it.

Generated NPZ data, logs, caches, and GPU-memory samples belong in a run or
scratch directory, not in Git. Coupled runs additionally require the local,
untracked OpenADAS bundles under `data/openadas/`; those files are user-owned
inputs and are not committed. See [OPENADAS.md](OPENADAS.md).

### Allocation and resource syntax

Perlmutter uses Slurm constraints and QOS values for GPU jobs. Request
`--constraint=gpu` and explicitly request the GPUs. Do not substitute an
invented `--partition=gpu`: use `debug` for a job expected to finish within 30
minutes, or `regular` for a longer production case, subject to the current
[Perlmutter QOS policy](https://docs.nersc.gov/jobs/policy/).
Use `sinfo` and the [Perlmutter job guidance](https://docs.nersc.gov/systems/perlmutter/running-jobs/)
to check current availability and limits.

The project ID/account for this project is `m5276`. The live scheduler accepts
that account and records the GPU charge account as `m5276_g`; use `m5276` in
the batch template below and verify the recorded account with `sacct`. GPU and
CPU allocations are separate pools.

NERSC user-facing examples select the GPU pool with `--constraint=gpu` plus a
QOS and omit `--partition`. Scheduler-selected partition names can vary by
QOS, reservation, and system maintenance; inspect them with `sacct` after a
run. Do not put a partition directive in a Perlmutter batch script.

For one process using one GPU with the GPU node's 128 logical CPUs available,
the relevant options are:

```text
--account=m5276 --constraint=gpu&hbm40g --qos=debug
--nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=1
```

Perlmutter GPU nodes have one 64-core AMD CPU (128 logical CPUs) and four
A100s. The `--cpus-per-task=128` value above gives a single task the complete
logical CPU set available on the node; use `--cpu-bind=cores` when launching.
The solver itself remains a one-GPU process, so requesting additional nodes or
GPUs does not make one solver run distributed.

Use `--constraint=gpu&hbm40g` for the standard 40-GB A100 nodes and
`--constraint=gpu&hbm80g` for the 80-GB A100 nodes. In an `salloc` command,
quote the constraint because of the ampersand:

```bash
salloc --nodes=1 --qos=debug --time=00:30:00 \
  --constraint="gpu&hbm80g" --gpus=1 --account=m5276
```

The Perlmutter GPU `debug` QOS permits up to 8 nodes and 30 minutes. It is the
right choice for cases expected to finish within that limit. The `interactive`
QOS permits at most 2 submitted and 2 running interactive jobs per user, with a
4-node maximum and a longer walltime limit. The 4-node limit belongs to
`interactive`; it is not the debug-QOS node limit. Use `regular` for a
production case that needs more than 30 minutes, and check the current
[QOS policy](https://docs.nersc.gov/jobs/policy/) before submission.

### Login-node restrictions and interactive preflight

On a login node, inspect files, parse TOML, inspect modules, and submit jobs;
do not run the solver or treat the login node's A100 as a production
allocation. For a bounded interactive GPU preflight, NERSC's documented form
is:

```bash
salloc --nodes=1 --qos=interactive --time=01:00:00 \
  --constraint=gpu --gpus=1 --account=m5276

# srun must request GPU resources explicitly, too.
srun --nodes=1 --ntasks=1 --cpus-per-task=128 \
  --gpus-per-task=1 --cpu-bind=cores nvidia-smi
source .venv/bin/activate
```

Then run the CUDA/Warp/cuDSS/JAX preflight below. The interactive QOS has a
bounded wait and walltime policy; use a batch job for production. If `salloc`
reports that the request does not match policy, check the GPU account, the
`gpu` constraint, and the current QOS policy. See NERSC's
[interactive-job guidance](https://docs.nersc.gov/jobs/interactive/).

### Environment and device preflight

Perlmutter uses [Lmod](https://docs.nersc.gov/environment/lmod/). Inspect the
site-provided modules and record the result; do not put an unverified module
version in this repository:

```bash
module list
module spider python
module spider cudatoolkit
source .venv/bin/activate
python --version
python -m pip check
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
```

The NERSC CUDA toolkit is exposed through `cudatoolkit`; see NERSC's
[CUDA guidance](https://docs.nersc.gov/development/programming-models/cuda/).
Use the exact module that was used to build and validate the project
environment, if one is needed, and record it with `module list`. Do not combine
a module-provided CUDA toolkit with a second environment-provided CUDA toolkit
without checking compatibility; NERSC's [Python guidance](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/)
specifically warns against loading both copies in one Python environment.
The repository environment is expected to provide the pinned CUDA-13
Warp/JAX/cuDSS Python packages above; do not install packages in a production
job. Use the same package family for every CUDA-dependent Python component.

Run this only after the GPU allocation is active:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv
python - <<'PY'
import importlib.metadata as metadata
import os

import jax
import nvmath
from nvmath.bindings import cudss
import warp as wp

wp.init()
device = wp.get_device("cuda:0")
if not device.is_cuda:
    raise SystemExit(f"CUDA GPU required, got {device}")
fp64 = wp.zeros(1, dtype=wp.float64, device="cuda:0")
major = cudss.get_property(nvmath.LibraryPropertyType.MAJOR_VERSION)
minor = cudss.get_property(nvmath.LibraryPropertyType.MINOR_VERSION)
patch = cudss.get_property(nvmath.LibraryPropertyType.PATCH_LEVEL)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("Warp device:", device)
print("Warp FP64 buffer:", fp64.dtype, "on", fp64.device)
print("Warp:", metadata.version("warp-lang"))
print("nvmath-python:", metadata.version("nvmath-python"))
print("cuDSS:", f"{major}.{minor}.{patch}")
print("JAX:", jax.__version__, "backend=", jax.default_backend())
if jax.default_backend() != "gpu":
    raise SystemExit("JAX GPU backend is unavailable")
if not jax.devices("gpu"):
    raise SystemExit("JAX sees no GPU devices")
jax.config.update("jax_enable_x64", True)
print("JAX devices:", jax.devices("gpu"))
PY
```

For this one-GPU solver, `cuda:0` means the first GPU visible to the process;
do not overwrite Slurm's `CUDA_VISIBLE_DEVICES`. The production solver will
reject a non-CUDA Warp device, and its cuDSS/Warp buffers must have identical
device identities. Set `JAX_PLATFORMS=cuda` in the job environment when JAX is
used so a missing GPU fails rather than silently selecting CPU. A JAX component
must use the same process-visible GPU as Warp and cuDSS, and must enable X64
before constructing arrays used with this FP64 workflow.

### Batch template

Use project-local [perlmutter_a100.sbatch](../perlmutter_a100.sbatch) as the
single source of truth. Create `logs/` before submission because Slurm opens
`--output` and `--error` before the script body runs:

```bash
mkdir -p logs

# Default: standard 40-GB A100, one GPU, debug QOS, 30-minute limit.
sbatch perlmutter_a100.sbatch outputs/gpu_check/forward_fv_solver.toml

# 80-GB A100 variant; no partition directive is needed.
sbatch --constraint='gpu&hbm80g' perlmutter_a100.sbatch \
  outputs/gpu_check_80/forward_fv_solver.toml
```

The script performs GPU/backend preflight, records the selected GPU/QOS, and
launches the solver with `srun`. It requests 128 logical CPUs by default;
override `--cpus-per-task=28` for a small one-process test when full CPU-node
allocation is unnecessary. Keep result paths under project `outputs/` or a
site scratch directory.

For an independent multi-case workflow, `--nodes=4` is valid within the debug
QOS limit (and `--nodes=8` is the documented maximum), but the present forward
solver is not a multi-node executable. Do not request multiple nodes for one
case unless a future distributed driver explicitly assigns one process/GPU per
node.

For a coupled case, keep the input TOML and its local `data/openadas/` paths
in a checkout visible to the compute node, or prepare a run-specific TOML with
absolute paths before submission. Copying a coupled TOML without its local
OpenADAS directory will produce an invalid run. Do not commit the batch script,
input copy, OpenADAS data, logs, or NPZ outputs unless a separate project
artifact policy explicitly requires it.

### Filesystems and run layout

Use `$SCRATCH` or `$PSCRATCH` rather than hard-coding a personal
`/pscratch/sd/...` path. NERSC recommends Perlmutter Scratch for active,
high-performance job I/O, but it is purgeable; data not accessed within eight
weeks may be deleted. Use the project directory in CFS, `$CFS/<project>`, for
shared source or medium-term project data, and back up irreplaceable results
to an appropriate persistent/archive location. Check quotas before a large
run. See NERSC's [filesystem overview](https://docs.nersc.gov/filesystems/),
[Community File System guidance](https://docs.nersc.gov/filesystems/community/),
and [Perlmutter Scratch guidance](https://docs.nersc.gov/filesystems/perlmutter-scratch/).

Keep `data/openadas/` local to the checkout or run input area. It contains
user-owned ADF11 files and manifests and remains untracked; it is not a
repository dependency or generated output. Preserve its manifest and verify
the bundle paths before a coupled run, but do not add the data files to Git.

### Provenance, logging, and monitoring

Before scaling a case, make one run directory such as
`$SCRATCH/slab-fp/<JOBID>` and retain:

- the Slurm job ID and `sacct` record;
- the exact Git commit and input TOML (the solver also stores both in its NPZ);
- `module list`, Python/package versions, and `pip freeze`;
- GPU model, UUID, total memory, driver, `CUDA_VISIBLE_DEVICES`, Warp device,
  FP64 check, and cuDSS version;
- solver stdout/stderr, residuals, finite-volume balance, accepted/rejected
  steps, state bounds (`min_N`, `max_f`), timings, and storage estimate;
- host memory from `sstat`/`sacct` and GPU-memory samples or a profiler for
  peak device memory. `MaxRSS` is host memory, not GPU memory.

The NPZ output contains the input TOML, Git commit, runtime versions/device,
state diagnostics, residuals, balance data, timing data, and (for coupled
runs) OpenADAS/restart provenance. The batch-side provenance file is still
needed for the job ID, module/environment snapshot, GPU model, and memory
sampling.

Use one `squeue`/`sqs` query at a time; NERSC asks users not to run aggressive
continuous scheduler polling. Typical commands are:

```bash
JOBID=<job-id>
squeue -j "$JOBID"
squeue --start -j "$JOBID"
tail -f "logs/slab-fp-${JOBID}.out"
sstat -j "${JOBID}.batch" --format=JobID,MaxRSS,AveRSS
sacct -j "$JOBID" \
  --format=JobID,JobName,Partition,Account,QOS,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Use `sacct` after completion or failure; `squeue` only describes active or
queued jobs. `scontrol show job "$JOBID"` provides the submitted resource
request and pending/failure context. See NERSC's
[monitoring guidance](https://docs.nersc.gov/jobs/monitoring/).

### Failure handling

- `InvalidAccount`, `InvalidQOS`, or “request does not match policy”: verify the
  GPU account from Iris, `--constraint=gpu`, explicit GPU count, QOS, and current
  NERSC policy; do not retry with CPU resources.
- `PENDING`: inspect `squeue --start -j`, its reason field, and `scontrol show
  job`; reduce requested time/resources only when scientifically acceptable.
- Missing CUDA/Warp/cuDSS or a non-CUDA device: stop, preserve the preflight and
  module logs, and repair the environment/allocation. There is no CPU fallback.
- cuDSS out-of-memory or timeout: stop the case, retain logs and memory samples,
  reduce mesh or timestep workload, or qualify an appropriate `hbm80g` request;
  do not treat a partial output as a result.
- Nonfinite or negative state, excessive residual, or failed balance check: do
  not use the output for training or coupling. Investigate timestep, boundary,
  coefficient, or resource/environment causes first.
- Cancel a stuck or invalid job with `scancel <JOBID>` and retain its `.out`,
  `.err`, provenance, and `sacct` diagnostics.
