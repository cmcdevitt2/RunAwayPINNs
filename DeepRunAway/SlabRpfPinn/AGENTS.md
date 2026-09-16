# Codex instructions for SlabRpfPinn

Read `README.md` first. Read `PERLMUTTER.md` only for HPC tasks, then read
relevant configuration and code. This directory is a nested monorepo project:
touch only paths under `SlabRpfPinn/` and preserve unrelated sibling changes.

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
- Start with `git status --short --branch -- .`; use `rg` and targeted reads
  before broad scans.
- Prefer the smallest responsible edit. Do not add abstractions, dependencies,
  refactors, or output formats without a concrete need. Add focused regression
  tests when behavior or numerical contracts change; avoid speculative tests.
- Preserve unrelated worktree changes. Never use blanket staging or cleanup.
- Before any commit, inspect exact staged paths and `git diff --cached`.
- Do not push, submit external jobs, or delete user data unless explicitly
  requested.

## Context and token discipline

- Read relevant files only. Never read large data, generated output, or
  reference documents unless the task requires them.
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
- `skill-installer`: only for explicit skill installation requests.

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
- The three root Slurm files (`generate_fv_dataset.sbatch`, `train_model.sbatch`,
  `validate_model.sbatch`) are production job entry points.

Keep FV label generation outside JAX automatic differentiation. Preserve the
normalized nine-coordinate model interface, FP64, probability bounds, FV
boundary semantics, and parameter-domain definitions unless the task
explicitly changes the numerical model.

## Environment

Self-contained `./.venv` inside this directory, on the local machine and on
every HPC cluster. Never activate an environment outside `SlabRpfPinn/`. See
`requirements.txt` for dependencies. Use `PERLMUTTER.md` only for HPC setup.

## Configuration contract

There are three production JSON files:

- `configs/fv_dataset.json` for FV data generation;
- `configs/train.json` for model mode, architecture, losses, optimizer,
  resource usage, and artifact paths;
- `configs/validate_model.json` for model loading and analytics.

Only these three JSON files are production configurations. Define each run in
one of these files and use fresh artifact paths.

Root drivers remain config-driven. `train_model.py` and `validate_model.py`
accept an optional positional config path; `generate_fv_dataset.py` uses its
fixed default config. Preserve this interface unless the task explicitly
changes it. A new run uses new artifact directories; completed run directories
are protected from overwrite.

## Verification ladder

1. Parse all three production JSON configurations. Run `bash -n` on changed
   Slurm scripts.
2. Compile root drivers and `core/` with `./.venv/bin/python`.
3. Import core drivers; verify CPU backend and JAX FP64. Training and
   validation require a visible GPU backend.
4. For FV changes, run a small CPU case and check probability bounds and
   residual. For storage or distributed changes, also verify manifest completion.
5. For training changes, run the defined training configuration in a GPU
   allocation. Inspect checkpoint and loss-history files. Use CPU checks only
   for compile, import, and configuration validation.
6. For validation changes, use a GPU allocation. Verify model and optional
   dataset manifests plus checksums before running analytics.
7. For multi-node data or physics training, verify one rank per node, global
   device count, synchronized completion, and one final rank-0 artifact set.
   Verify SSBroyden broadcast when enabled. Active FV acquisition requires one
   JAX process.

Do not run production FV generation, training, or validation on login nodes.
Use tracked Slurm scripts for production jobs.
Do not claim GPU or multi-node qualification from syntax/import checks alone.

## Completion report

State: result first; changed files; verification performed; runtime
limitations; any remaining cluster-specific requirement; commit hash if
committed. Keep output brief.
