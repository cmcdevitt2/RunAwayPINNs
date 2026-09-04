# Hipergator GPU batch execution

Run both programs through Slurm batch jobs on Hipergator. Login nodes are for
editing, small syntax checks, and job submission only; they are not a valid
place for Warp/cuDSS or JAX GPU production runs.

The production target is an NVIDIA B200 GPU. Request the dedicated
`hpg-b200` partition and a typed `gpu:b200:1` GRES in every batch script
(subject to the project's Slurm account/QOS access); retain the `b200`
constraint as an additional allocation check when adapting the templates below.

## Prepare environment

From the renamed `SlabRpfPinn` directory, use the neighboring environment:

```bash
cd /path/to/DeepRunAway/SlabRpfPinn
source ../.venv/bin/activate
python - <<'PY'
import jax
import warp
import nvmath
print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())
print("nvmath:", nvmath.__version__)
PY
```

JAX may report its GPU backend as `gpu`; confirm `jax.devices()` includes a CUDA
device, and Warp must list at least one CUDA device. If either check reports
CPU-only execution, stop the job and inspect
the allocated GPU, modules, environment, and CUDA library paths.

## Adjoint FV job

Save a site-specific batch script outside Git or adapt this template:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=slab-rpf-fv
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs
source ../.venv/bin/activate

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python - <<'PY'
import jax
import warp
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX GPU backend required, got {jax.default_backend()!r}")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit(f"Warp CUDA device required, got {warp.get_devices()!r}")
print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())
PY

python adjoint_fv_solver.py --config adjoint_fv_solver.toml
```

Submit and monitor:

```bash
mkdir -p logs outputs
sbatch fv.slurm
squeue -u "$USER"
tail -f logs/slab-rpf-fv-<JOBID>.out
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Use a separate output path per case. Do not let concurrent jobs overwrite the
same `.npz` or plot files.

## FV robustness campaign

Before generating PINN labels, run the committed B200 campaign script. It
samples complete plasma states across the configured PINN domain, repeats them
on the configured grid/domain panels, and writes a machine-readable report of
per-case solver diagnostics plus common-grid RPF differences. It deliberately
does not turn a chosen numerical threshold into an automatic label-promotion
decision.

```bash
source ../.venv/bin/activate
python fv_robustness_campaign.py --config fv_robustness_campaign.toml --dry-run
mkdir -p logs outputs/fv_robustness
sbatch fv_robustness_b200.sbatch
```

The script requests `hpg-b200`, `gpu:b200:1`, and `--constraint=b200`, then
rejects an allocation whose GPU name is not B200. Use `sacct` after completion and retain
`outputs/fv_robustness/fv_robustness_summary.json`, the TOML, and Slurm logs.
Review the failed-case list, residual/identity diagnostics, probability bounds,
and RPF differences from radial, pitch, lower-domain, and upper-domain panels
before increasing sampling coverage or accepting FV labels.

The higher-resolution follow-up retains the first report in a distinct output
directory and can be submitted with:

```bash
mkdir -p logs outputs/fv_robustness/refinement_1024x512
sbatch fv_robustness_b200.sbatch fv_robustness_refinement.toml
```

## PINN training job

PINN label generation can solve many FV cases and may require substantial GPU
memory, host memory, and wall time. Submit it as its own job:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=slab-rpf-pinn
#SBATCH --partition=hpg-b200
#SBATCH --constraint=b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs
source ../.venv/bin/activate

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python - <<'PY'
import jax
import warp
if jax.default_backend() != "gpu":
    raise SystemExit(f"JAX GPU backend required, got {jax.default_backend()!r}")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit(f"Warp CUDA device required, got {warp.get_devices()!r}")
print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())
PY

python pinn_training.py --config pinn_training.toml
```

Create the Slurm log directories before submission because Slurm opens output
paths before executing the script:

```bash
mkdir -p logs data outputs
sbatch pinn.slurm
```

Start with reduced `Np`, `Nxi`, case counts, collocation counts, and optimizer
steps. Increase them only after one complete batch run passes the backend and
coefficient-parity checks. Keep generated dataset/model paths on scratch or a
project data area, not in Git.

## Reproducibility and diagnostics

- Record Slurm job ID, Git commit, Python environment, GPU model, and TOML file.
- Keep stdout and stderr logs with each result directory.
- Verify JAX reports backend `gpu` with CUDA devices. Standalone FV logs should
  contain `execution=gpu-warp-cudss`; PINN logs should contain its JAX and Warp
  device probes.
- Check cuDSS residuals, transpose checks, escape identity, and probability
  bounds before treating an FV result as a training label.
- For PINN runs, retain the copied TOML, training history, summary, and dataset
  metadata together.
- Use `sacct` after completion; `squeue` alone does not report failed jobs.

## Common failure modes

`Invalid device identifier: cuda:0` means the script is running without a valid
GPU allocation or visible CUDA device. Resubmit with a GPU resource request.

`FileNotFoundError` for a TOML file means the job started in the wrong
directory or the explicit `--config` path is wrong. Always use
`cd "$SLURM_SUBMIT_DIR"` and pass the config explicitly.

Do not bypass a failed GPU check by setting `JAX_PLATFORMS=cpu`; that only hides
the allocation/configuration problem and cannot run the cuDSS FV path.
