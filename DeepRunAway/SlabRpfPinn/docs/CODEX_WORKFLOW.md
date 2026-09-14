# Development workflow

This document describes how to modify and verify the current script-driven
SlabRpfPinn project. The root `README.md` and cluster runbooks are the
user-facing execution documentation.

## Session start

```bash
pwd
git status --short --branch
rg --files -g '!data/**' -g '!runs/**' -g '!*.npz'
```

Read `README.md`, the relevant JSON config, and the relevant driver/core
module. Search symbols and callers with `rg` before opening unrelated files.

## Code ownership

- `core/rpf_fv_cpu.py` owns the CPU finite-volume discretization and adjoint
  solve.
- `core/fv_dataset.py` owns case generation, flattening/coarsening, storage,
  and loading.
- `core/fv_distributed.py` owns multi-node CPU dataset orchestration.
- `core/model.py` owns MLP/DeepONet architecture and prediction.
- `core/pde.py` owns normalized physical coefficients, residuals, boundaries,
  and collocation sampling.
- `core/training.py` owns data, physics, SOAP, SSBroyden, and active loops.
- `core/training_config.py` owns the hierarchical training schema.
- `core/training_artifacts.py` owns manifests, checksums, atomic writes, file
  barriers, checkpoints, and histories.
- The three root Python files are config-only user entry points.

Keep FV label generation outside JAX automatic differentiation. Preserve the
normalized eight-coordinate model interface, FP64, probability bounds, FV
boundary semantics, and parameter-domain definitions unless the task
explicitly changes the numerical model.

## Configuration contract

There are three active JSON files:

- `run_configs/fv_dataset.json` for FV data generation;
- `run_configs/train.json` for model mode, architecture, losses, optimizer,
  resource usage, and artifact paths;
- `run_configs/validate_model.json` for model loading and analytics.

Do not add runtime CLI options to the root drivers. A new run uses new artifact
directories; completed run directories are protected from overwrite.

## Verification ladder

1. Parse all active JSON configurations.
2. Compile root drivers and `core/` with the project Python.
3. Import core drivers and verify the JAX backend/FP64 inside an allocation.
4. For FV changes, run a small CPU case and check probability bounds, residual,
   and dataset manifest completion.
5. For training changes, run a short GPU training allocation, inspect checkpoint
   and loss-history files, and validate held-out cases.
6. For validation changes, verify model and optional dataset manifests plus
   their checksums before analytics.
7. For multi-node data training, verify one rank per node, global device count,
   synchronized completion, and a single final artifact set.

Do not run production CPU generation or GPU training on login nodes. Do not
claim GPU or multi-node qualification from syntax/import checks alone.

## Reporting

Report changed paths, verification performed, runtime limitations, and any
remaining cluster-specific requirement. Keep generated data, models, logs,
plots, and caches out of source control.
