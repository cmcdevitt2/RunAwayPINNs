# SlabFokkerPlanck

GPU forward finite-volume solver for the spatially homogeneous relativistic
electron Fokker--Planck problem. The intended next stage is coupling its
prescribed bulk-plasma coefficients to a bulk plasma model.

## Contents

- `forward_fv_solver.py`: Warp/cuDSS forward FV transient solver.
- `forward_fv_solver.toml`: reproducible example configuration.
- `kinetic_model_reorganized_v15_4_fv_assisted_pinn.tex`: scientific reference
  retained from the model-development work.
- `docs/HIPERGATOR.md`: required Hipergator Slurm/GPU workflow.
- `docs/CODEX_WORKFLOW.md`: compact Codex development and verification guide.

The current executable advances a prescribed plasma state. It is not yet a
self-consistent bulk-plasma coupling driver. The low-level GPU run accepts a
`SolverConfig`, and `CudssDirectSystem.refactorize()` provides the structural
reuse seam needed when future coupling changes matrix values without changing
the mesh topology; the public time-step coupling API remains future work.

## Numerical/GPU contract

The solver uses FP64 Warp kernels, GPU-resident CSR assembly, and cuDSS direct
solves. Keep Warp, cuDSS, and any future JAX bulk model on the same allocated
CUDA device. The production cuDSS ordering is the default nested-dissection
path. The SlabRpfPinn stress campaign found that fill/workspace, not raw CSR
storage, controls scaling; a 4096x2048 mesh used about 10.6 GiB peak and an
8192x4096 mesh about 41.4 GiB on a B200. Treat larger cases as a memory study.

## Quick start

Run only inside an allocated GPU job, using the neighboring environment:

```bash
source ../.venv/bin/activate
python forward_fv_solver.py --config forward_fv_solver.toml
```

Login nodes are for inspection, syntax checks, and submission. Use
`docs/HIPERGATOR.md` for Slurm templates, resource sizing, diagnostics, and
failure handling. Do not use the legacy `runaway_0d2p_warp_cudss_gpu_r7_5.py`
entrypoint.

Generated outputs, logs, caches, and model data are intentionally excluded
from Git. Keep large results in scratch or the project data area.
