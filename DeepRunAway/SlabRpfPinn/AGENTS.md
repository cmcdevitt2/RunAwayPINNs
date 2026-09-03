# Codex instructions for SlabRpfPinn

## Mission

Maintain a compact, reproducible GPU finite-volume adjoint and JAX PINN
workflow. Preserve the physical and numerical contract represented by the
source and configuration files. Treat solver correctness, GPU residency,
cuDSS usage, and reproducibility as first-class requirements.

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

- Read README and relevant config/code first; read large data, generated output,
  and reference documents only when the task requires them.
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

## Scientific and runtime rules

- `adjoint_fv_solver.py` owns FV assembly, algebraic transpose, and cuDSS solves.
- `pinn_training.py` dynamically loads the configured FV solver and keeps FV
  label generation outside JAX autodiff.
- Keep Warp, cuDSS, and JAX on the same allocated CUDA device. Preserve FP64.
- Run production solver/training workloads only in Slurm GPU batch jobs. Login
  nodes are for inspection, syntax checks, and submission.
- Use the neighboring environment at `../.venv` from this directory.
- Verify JAX devices and Warp CUDA devices before expensive work. CPU fallback is
  not an acceptable substitute for the cuDSS path.
- Keep generated datasets, models, logs, plots, and caches out of Git.

## Verification ladder

1. Syntax and TOML parsing.
2. Import/config and small CPU structural checks when GPU libraries permit.
3. GPU backend check inside a Slurm allocation.
4. Solver qualification: residual, transpose, escape identity, probability
   bounds, and refinement checks.
5. PINN checks: coefficient parity, training/validation separation, held-out
   cases, and saved metadata.

Match verification effort to change risk. Stop after required checks pass unless
new failures or unresolved concerns justify broader testing.

## Completion report

State: result first; changed files; verification performed; known limitations;
commit hash if committed. Keep output brief.
