# SlabRpfPinn

GPU finite-volume adjoint and physics-informed neural-network tools for a
steady 0D--2P relativistic runaway-electron first-passage problem.

## Contents

- `adjoint_fv_solver.py`: Warp/cuDSS finite-volume adjoint/RPF solver.
- `perlmutter_a100.sbatch`: Perlmutter A100 FV/PINN batch template.
- `pinn_training.py`: JAX PINN trainer using FV-generated labels.
- `pinn_training_smoke.toml`: persistent one-case GPU execution smoke test;
  not a training qualification.
- `adjoint_fv_solver.toml`: small direct-solver example case.
- `pinn_training.toml`: training configuration and parameter domain.
- `rpf_fv_pinn_scientific_reference_v1.tex`: scientific and numerical reference.
- `docs/HIPERGATOR.md`: Hipergator B200 GPU batch-job workflow.
- `docs/PERLMUTTER.md`: NERSC Perlmutter A100 GPU batch-job workflow.
- `docs/DEPENDENCIES.md`: shared cluster-agnostic Python/GPU dependencies.

The solver is a prescribed-parameter, spatially homogeneous test-particle model;
it is not a self-consistent plasma evolution code. Preserve the physical and
numerical contract represented by the source and configuration files.
The uniform-grid FV baseline uses Chang--Cooper exponential fitting for its
drift--diffusion fluxes. The LaTeX reference gives the detailed adjoint PDE,
FV discretization, and PINN qualification requirements.
The solver also supports optional `xi_mapping = "theta"`, formed by uniform
theta cells with `xi = -cos(theta)`, and `p_mapping = "log"` or
`"exponential"`; defaults remain uniform xi and uniform p.

## Software model

The FV solver assembles the local finite-volume generator on the GPU, forms its
algebraic transpose for the first-passage adjoint, and uses cuDSS for sparse
direct solves. The PINN uses JAX/XLA for automatic differentiation, batching,
and optimization. FV label generation remains outside JAX and uses the same
GPU-resident Warp/cuDSS path.

## Quick start on an allocated GPU node

Run from this directory on an allocated GPU node after following the selected
cluster guide and [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md):

```bash
python adjoint_fv_solver.py --config adjoint_fv_solver.toml
python pinn_training.py --config pinn_training.toml
```

These commands are intended for an allocated GPU node. Do not run production
cases on a login node. See `docs/HIPERGATOR.md` for Hipergator B200 or
`docs/PERLMUTTER.md` for NERSC Perlmutter A100 Slurm commands, resource
requests, environment setup, logging, and validation checks.

For a short Perlmutter execution check, use persistent
`pinn_training_smoke.toml` through `docs/PERLMUTTER.md`.

Training with `generate = true` writes the FV dataset and model outputs under
the configured paths. Those generated files are ignored by Git and should be
reproducible from the committed source and TOML configuration.

## FV label generation

`pinn_training.py` generates FV labels outside JAX using the configured GPU FV
solver, then trains the PINN from those labels. Run it only on an allocated GPU
node:

```bash
python pinn_training.py --config pinn_training.toml
```

Use `pinn_training_smoke.toml` for a short execution-path check. Keep generated
datasets, models, logs, and plots in ignored output paths or approved scratch
storage.
