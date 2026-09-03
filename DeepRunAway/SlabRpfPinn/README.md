# SlabRpfPinn

GPU finite-volume adjoint and physics-informed neural-network tools for a
steady 0D--2P relativistic runaway-electron first-passage problem.

## Contents

- `adjoint_fv_solver.py`: Warp/cuDSS finite-volume adjoint/RPF solver.
- `pinn_training.py`: JAX PINN trainer using FV-generated labels.
- `adjoint_fv_solver.toml`: small direct-solver example case.
- `pinn_training.toml`: training configuration and parameter domain.
- `docs/HIPERGATOR.md`: required Hipergator GPU batch-job workflow.

The LaTeX model specification is the numerical and physical contract for these
codes. The solver is a prescribed-parameter, spatially homogeneous test-particle
model; it is not a self-consistent plasma evolution code.

## Software model

The FV solver assembles the local finite-volume generator on the GPU, forms its
algebraic transpose for the first-passage adjoint, and uses cuDSS for sparse
direct solves. The PINN uses JAX/XLA for automatic differentiation, batching,
and optimization. FV label generation remains outside JAX and uses the same
GPU-resident Warp/cuDSS path.

## Quick start on an allocated GPU node

Run from this directory with the repository's neighboring environment:

```bash
source ../.venv/bin/activate
python adjoint_fv_solver.py --config adjoint_fv_solver.toml
python pinn_training.py --config pinn_training.toml
```

These commands are intended for an allocated GPU node. Do not run production
cases on a login node. See `docs/HIPERGATOR.md` for Slurm commands, resource
requests, environment setup, logging, and validation checks.

Training with `generate = true` writes the FV dataset and model outputs under
the configured paths. Those generated files are ignored by Git and should be
reproducible from the committed source and TOML configuration.
