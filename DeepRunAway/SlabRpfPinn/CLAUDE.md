# Claude instructions for SlabRpfPinn

Read [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) first: monorepo
context, local-vs-remote-HPC execution split, and self-contained `./.venv`.

This document describes how to modify and verify the current script-driven
SlabRpfPinn project. The root `README.md` and cluster runbooks are the
user-facing execution documentation.

## Session start

```bash
pwd
git status --short --branch
rg --files
```

`rg --files` respects `.gitignore`; `data/`, `runs/`, `.venv/`, and other
generated artifacts are already excluded, no extra globs needed.

Read `docs/PROJECT_CONTEXT.md` first if not already loaded, then only the
specific JSON config and specific driver/core module the task touches. Do
not read `core/training.py` end-to-end — it is 2060 lines; find the target
with `rg -n 'def <name>' core/training.py` first. Search symbols and callers
with `rg` before opening unrelated files.

## Delegation and agent workflow

Use the `Agent` tool and `TaskCreate`/`TaskUpdate`/`TaskList` to keep the main
context small and reduce token spend on routine work.

- **Read-only search** (find a symbol, caller, config field, or file across
  `core/`, `run_configs/`, or `docs/`): delegate to `Agent`
  `subagent_type: Explore`. Do not grep-and-read manually in the main
  context when a targeted search agent returns the same answer for less.
- **Bounded, well-scoped edits** (a single-file fix, a narrow doc update, a
  clear one-function change): delegate to `Agent` `subagent_type:
  general-purpose` with the exact file path, the exact change, and the
  verification step to run afterward. Do not delegate work whose scope is
  still being decided — decide scope first, then delegate.
- **Multi-file or multi-step work** (touches 3+ files, or has an ordered
  sequence of dependent steps such as this project's documentation layers or
  a coordinated `core/` refactor): use `TaskCreate` to track each step,
  mark `in_progress`/`completed` as you go, and delegate independent steps
  to parallel `Agent` calls when they do not depend on each other's output.
- **Do not delegate**: numerical/physics correctness judgment (FV contract,
  PDE residual meaning, probability-bounds interpretation), anything touching
  more than one active JSON config's meaning at once, or a task whose
  acceptance criteria you have not yet pinned down. Decide those directly.
- Never read `data/`, `runs/`, or `*.npz` contents into the main context or
  an agent's context to "check" a result — inspect summary fields (manifest
  JSON, loss-history JSON, printed metrics) instead. These directories are
  large, generated, and out of source control.
- When delegating, state the target file(s), the exact change or question,
  and what a correct result looks like. A vague delegated prompt costs more
  tokens in back-and-forth than doing the search yourself.

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

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) before editing
`core/training.py` — it has a known dead-code defect.

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
3. Import core drivers and verify the JAX backend/FP64 (CPU backend is a
   valid pass here — everything in this project runs on CPU; GPU only
   changes performance, never correctness).
4. For FV changes, run a small CPU case and check probability bounds, residual,
   and dataset manifest completion.
5. For training changes, first run a short CPU-only training pass (small
   Sobol count, few steps) and inspect checkpoint and loss-history files —
   this is the cheapest correctness check and does not need an allocation.
   Only escalate to a GPU allocation to confirm production-scale behavior or
   GPU-path-specific code (device sharding, multi-host init).
6. For validation changes, verify model and optional dataset manifests plus
   their checksums before analytics.
7. For multi-node data training, verify one rank per node, global device count,
   synchronized completion, and a single final artifact set.

Do not run production CPU generation or GPU training on login nodes. Do not
claim multi-node qualification from syntax/import checks alone.

## Reporting

Report changed paths, verification performed, runtime limitations, and any
remaining cluster-specific requirement. Keep generated data, models, logs,
plots, and caches out of source control.
