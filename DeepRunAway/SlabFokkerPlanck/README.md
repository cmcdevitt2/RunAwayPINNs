# SlabFokkerPlanck

GPU forward finite-volume solver for the spatially homogeneous relativistic
electron Fokker--Planck problem. The intended next stage is coupling its
prescribed bulk-plasma coefficients to a bulk plasma model.

## Contents

- `forward_fv_solver.py`: Warp/cuDSS forward FV transient solver.
- `forward_fv_solver.toml`: reproducible example configuration.
- `forward_fokker_planck_model.tex`: forward-only scientific and numerical
  reference, including the future bulk-plasma coupling scope.
- `openadas_data.py` and `scripts/fetch_openadas.py`: private OpenADAS ADF11
  acquisition/provenance and strict-range reader for the future bulk model.
- `bulk_plasma_model.py`: tested host-side 0D CR, bulk-energy, Ohm-law,
  lumped-induction, and stage-current exchange subsystem for the kinetic
  coupling driver.
- `docs/OPENADAS.md`: required ADF11 classes and data-handling workflow.
- `docs/HIPERGATOR.md`: required Hipergator Slurm/GPU workflow.
- `docs/CODEX_WORKFLOW.md`: compact Codex development and verification guide.

Standalone mode advances a prescribed plasma state. Optional `[bulk]`
configuration enables a self-consistent GPU kinetic/bulk driver: host-side CR,
energy, and induction stages exchange currents and plasma state with distinct
variable-coefficient kinetic TR--BDF2 operators. Coupled input requires local
OpenADAS ADF11 bundles plus authoritative state-resolved screening data.

## Numerical/GPU contract

The solver uses FP64 Warp kernels, GPU-resident CSR assembly, and cuDSS direct
solves. Keep Warp, cuDSS, and any future JAX bulk model on the same allocated
CUDA device. The production cuDSS ordering is the default nested-dissection
path. The SlabRpfPinn stress campaign found that fill/workspace, not raw CSR
storage, controls scaling; a 4096x2048 mesh used about 10.6 GiB peak and an
8192x4096 mesh about 41.4 GiB on a B200. Treat larger cases as a memory study.

## Quick start

The project virtual environment is `DeepRunAway/.venv`. From this directory it
is addressed as `../.venv`. Run the solver only inside an allocated GPU job:

```bash
source ../.venv/bin/activate
python forward_fv_solver.py --config forward_fv_solver.toml
```

Use the same `../.venv` activation in every Slurm batch script; do not use a
separate environment under `SlabFokkerPlanck`.

Login nodes are for inspection, syntax checks, and submission. Use
`docs/HIPERGATOR.md` for Slurm templates, resource sizing, diagnostics, and
failure handling. Do not use the legacy `runaway_0d2p_warp_cudss_gpu_r7_5.py`
entrypoint.

Generated outputs, logs, caches, and model data are intentionally excluded
from Git. Keep large results in scratch or the project data area.

The executable uses the documented Chang--Cooper face flux and TR--BDF2
stages. Adaptive TR--BDF2 is available through the embedded stiff error
estimate. Coupled runs preserve one fixed CSR topology, refactorize numerical
values with cuDSS, and write bulk/restart/OpenADAS provenance.
