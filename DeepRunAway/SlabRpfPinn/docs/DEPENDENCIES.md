# Shared Python and GPU dependencies

SlabRpfPinn Python code is cluster-agnostic. HiPerGator and Perlmutter
documentation select different GPU resources and module setup; both clusters
install same Python package roles in a validated environment.

## Required packages

Install these packages for both FV and PINN execution:

- Python 3.12 or newer.
- `numpy`, `scipy`, and `matplotlib` (FV solver writes summary plots).
- `jax` with one NVIDIA CUDA plugin.
- `warp-lang` for GPU kernel execution.
- `nvmath-python` with matching CUDA extra; this provides
  `nvmath.bindings.cudss` and the cuDSS runtime package.
- `optax`, `equinox`, and `optimistix` for PINN training.
- The SSBFGS Optimistix branch from the [CrunchOptimizer PINN project](https://github.com/CrunchOptimizer/PINNs).

Repository does not use CuPy, PyTorch, MPI, or a compiler module at runtime.
Warp, cuDSS, and JAX must use same visible CUDA device. Keep FP64 enabled.

## CUDA profile selection

Check cluster driver before installation:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
  --format=csv,noheader
```

Use one CUDA major version for JAX and nvmath. Do not mix `cu12` and `cu13`
Python CUDA packages in one environment.

CUDA 13 profile, validated on Perlmutter A100 driver 580.159.04:

```bash
python -m pip install --upgrade \
  'jax[cuda13]==0.11.1' \
  'warp-lang==1.17.0' \
  'nvmath-python[cu13]==1.0.0' \
  'numpy>=2.0,<2.4' scipy matplotlib optax equinox \
  'git+https://github.com/raj-brown/optimistix.git@SSBFGS'
```

This installs JAX 0.11.1/jaxlib 0.11.1, Warp 1.17.0, nvmath-python 1.0.0,
cuDSS 0.8.x, and repository PINN dependencies. VCS installation installs
Optimistix directly into environment; it does not create persistent source
checkout. The branch currently reports Optimistix 0.0.11 and supplies
`AbstractSSBroyden` and `SSBroyden`.

CUDA 12 profile, only when cluster driver and GPU support it:

```bash
python -m pip install --upgrade \
  'jax[cuda12]' \
  'warp-lang' \
  'nvmath-python[cu12]' \
  'numpy>=2.0,<2.4' scipy matplotlib optax equinox \
  'git+https://github.com/raj-brown/optimistix.git@SSBFGS'
```

JAX requires NVIDIA driver at least 525 for CUDA 12 and at least 580 for
CUDA 13. Warp package CUDA requirements and GPU architecture support can
change by release. Verify `nvidia-smi`, then run GPU preflight before any
solver work. Do not force CPU fallback when CUDA profile fails.

## Environment placement

Activate environment from cluster documentation:

- HiPerGator: neighboring `../.venv` from `SlabRpfPinn`, unless cluster
  documentation specifies validated replacement.
- Perlmutter: local `./.venv` or validated neighboring environment.

Keep environment and dependency metadata in approved persistent project or
scratch storage. Keep generated datasets, models, logs, plots, and caches in
ignored output paths on `$SCRATCH`/`$PSCRATCH` or approved project storage.
Do not use `/tmp`, `$TMPDIR`, `mktemp`, or node-local temporary directories for
inputs or outputs.

## GPU preflight

Run inside a Slurm GPU allocation after activating environment:

```bash
python - <<'PY'
import importlib.metadata as metadata
import jax
import nvmath
import warp
from nvmath.bindings import cudss

print("Python packages:")
for name in ("jax", "jaxlib", "warp-lang", "nvmath-python", "optimistix"):
    print(f"  {name}=={metadata.version(name)}")
print("JAX:", jax.default_backend(), jax.devices())
print("Warp:", warp.get_devices())

if jax.default_backend() != "gpu":
    raise SystemExit("JAX GPU backend required")
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise SystemExit("JAX CUDA device required")
if not any(getattr(device, "is_cuda", False) for device in warp.get_devices()):
    raise SystemExit("Warp CUDA device required")

version = tuple(cudss.get_property(property_type) for property_type in (
    nvmath.LibraryPropertyType.MAJOR_VERSION,
    nvmath.LibraryPropertyType.MINOR_VERSION,
    nvmath.LibraryPropertyType.PATCH_LEVEL,
))
print("cuDSS property:", version)
if version[0] != 0 or version[1] < 8:
    raise SystemExit(f"cuDSS 0.8 or newer required, got {version}")
PY
```

Successful imports on a login node do not qualify GPU execution. Run FV
qualification and PINN smoke checks only through the cluster's Slurm workflow.
