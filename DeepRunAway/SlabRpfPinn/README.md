# SlabRpfPinn

GPU finite-volume adjoint and physics-informed neural-network tools for a
steady 0D--2P relativistic runaway-electron first-passage problem.

## Contents

- `adjoint_fv_solver.py`: Warp/cuDSS finite-volume adjoint/RPF solver.
- `fv_robustness_campaign.py`: B200 parameter-sweep and grid/domain-refinement FV qualification driver.
- `fv_theta_resolution.toml`, `fv_theta_resolution_b200.sbatch`: focused angular-grid resolution study.
- `pinn_training.py`: JAX PINN trainer using FV-generated labels.
- `adjoint_fv_solver.toml`: small direct-solver example case.
- `fv_robustness_campaign.toml`: first-pass domain-wide FV robustness campaign.
- `fv_robustness_refinement.toml`: higher-resolution follow-up campaign.
- `pinn_training.toml`: training configuration and parameter domain.
- `rpf_fv_pinn_scientific_reference_v1.tex`: scientific and numerical reference.
- `docs/HIPERGATOR.md`: required Hipergator GPU batch-job workflow.

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

## FV qualification before PINN labels

Run `fv_robustness_campaign.py` before generating parametric PINN labels. It
uses the configured six-dimensional parameter domain, executes every case on
each grid/domain panel, records per-case FV checks, and compares each panel
with the reference panel on a common phase-space probe grid. It is an evidence
campaign: the resulting scalar report does not by itself promote labels.

On a B200 allocation, first validate the plan without a GPU, then submit the
committed batch script:

```bash
source ../.venv/bin/activate
python fv_robustness_campaign.py --config fv_robustness_campaign.toml --dry-run
mkdir -p logs outputs/fv_robustness
sbatch fv_robustness_b200.sbatch
```

The job writes `outputs/fv_robustness/fv_robustness_summary.json`; retain it
with the Slurm logs and configuration. Increase Sobol coverage and enable
corner cases or higher-resolution panels only after reviewing the initial
campaign's memory, residual, and refinement evidence.
