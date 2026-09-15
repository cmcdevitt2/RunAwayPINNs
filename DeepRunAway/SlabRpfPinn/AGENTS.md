# Codex instructions for SlabRpfPinn

Read `README.md` first, then relevant files: monorepo context,
local-vs-remote-HPC execution split, and self-contained `./.venv`.

## Mission

Maintain a compact, reproducible CPU finite-volume adjoint dataset generator
and JAX PINN training/validation workflow. Preserve the physical and
numerical contract represented by the source and configuration files. Treat
solver correctness, FP64, and reproducibility as first-class requirements.

## Operating mode

- Infer routine intent from the user request and repository context; act when
  scope is clear.
- Keep updates concise. Report only decisions, changed paths, checks, blockers,
  and required user choices.
- Use `rg`, `rg --files`, `git status`, and targeted reads before broad scans.
- Prefer the smallest responsible edit. Do not add abstractions, dependencies,
  refactors, tests, or output formats without a concrete need.
- Preserve unrelated worktree changes. Never use blanket staging or cleanup.
- Before any commit, inspect exact staged paths and `git diff --cached`.
- Do not push, submit external jobs, or delete user data unless explicitly
  requested.

## Context and token discipline

- Read `README.md`, then relevant documentation in `docs/`, followed by the
  relevant config/code; read large
  data, generated output, and reference documents only when the task requires
  them.
- Search symbols and call sites before opening whole files.
- Keep delegated messages narrow and ask for path/line findings or a small patch,
  not an essay.
- Do not repeat repository context already established in the conversation.
- Use normal prose for persistent documentation; use concise reporting in chat.

## Delegation and skill routing

Use skills conditionally. Do not load every skill for every task. Use the
efficient model available for deterministic work; reserve the strongest model
and main context for coupled numerical reasoning, architecture, and integration.

- `cavecrew`: delegate independent, bounded work when it saves time or context.
  - `cavecrew-investigator`: locate definitions, callers, references, or runtime assumptions;
  - `cavecrew-builder`: clear surgical edit touching at most two files;
  - `cavecrew-reviewer`: inspect a staged diff and return findings by path and line.
- `caveman`: keep agent communication and status reports terse when requested.
- `caveman-review`: compressed review of a diff or file.
- `investigate-first`: unknown failure, regression, or performance issue before
  editing.
- `surgical-patch`: narrow bug fix with focused regression proof.
- `safe-refactor`: behavior-preserving renames or structural changes.
- `lean-build`: new behavior or integration with meaningful overbuilding risk.
- `verify-and-stop`: validation-only task or final acceptance check.
- `openai-docs`: Codex, OpenAI API, model, or skill behavior questions.
- `find-skills` and `skill-installer`: only for explicit skill discovery or
  installation requests.

User instructions and higher-priority system/developer instructions override
this file and any skill guidance. If a skill changes scope or blocks work,
identify the exact skill and instruction before pausing.

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
- The three root Python files (`generate_fv_dataset.py`, `train_model.py`,
  `validate_model.py`) are config-only user entry points.

Keep FV label generation outside JAX automatic differentiation. Preserve the
normalized eight-coordinate model interface, FP64, probability bounds, FV
boundary semantics, and parameter-domain definitions unless the task
explicitly changes the numerical model.

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) before editing
`core/training.py` — it has a known dead-code defect.

## Environment

Self-contained `./.venv` inside this directory, on the local machine and on
every HPC cluster. Never activate an environment outside `SlabRpfPinn/`. See
`docs/DEPENDENCIES.md` for install steps and the
`jax[cuda13]` requirement on HPC.

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

## Completion report

State: result first; changed files; verification performed; runtime
limitations; any remaining cluster-specific requirement; commit hash if
committed. Keep output brief.
