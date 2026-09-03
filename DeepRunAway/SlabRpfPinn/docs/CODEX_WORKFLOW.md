# Codex workflow

This document expands the compact repository instructions in `AGENTS.md`.
It is a playbook for Codex sessions working on the SlabRpfPinn solver and PINN.

## Session start

Use this order:

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
rg --files -g '!data/**' -g '!outputs/**' -g '!models/**' -g '!*.npz'
```

Read `README.md`, then the relevant script and TOML file. Search definitions and
callers with `rg` before opening unrelated files. Confirm repository root before
staging because this directory is nested inside a larger public repository.

## Task routing

Classify work before acting:

- Locate or trace one symbol: use a fast investigator and request path/line output.
- Rename or edit one or two obvious files: use a bounded builder or make the
  surgical edit directly.
- Unknown numerical failure: investigate first; rank hypotheses by evidence.
- Cross-file solver change: keep design and integration in the main context;
  delegate independent searches or checks only.
- Completed diff: use a reviewer before commit.
- Validation request: run focused checks, then stop.

Use parallel delegation only for independent tasks. Never delegate two agents to
edit overlapping files. Prefer a fast/low-cost model for deterministic search,
format, syntax, or metadata work. Use a stronger model for numerical semantics,
architecture, or review of coupled changes. Do not hard-code a model name when
the active Codex environment exposes different model choices.

## Context budget

- Pass exact paths, symbols, line ranges, and acceptance criteria to agents.
- Ask agents for compressed findings, not broad explanations.
- Do not paste whole source files into prompts when a path and search term work.
- Cache stable facts in the task context; do not reread them without a reason.
- Keep progress messages short and reserve detail for the final handoff.
- If context grows, summarize decisions and unresolved issues before continuing.

## Code boundaries

`adjoint_fv_solver.py` is the direct FV reference path. Preserve its cell-content
state convention, algebraic transpose, GPU CSR topology, and cuDSS factorization
reuse. `pinn_training.py` is a steady PINN path. Preserve its separation between
JAX differentiation and FV label generation.

Do not silently change physics, boundary semantics, solver tolerances, schemas,
parameter ranges, or training protocol during cleanup. Such changes require a
separate task and a qualification plan.

## HPC execution

Use `docs/HIPERGATOR.md` for batch templates. Before submitting:

```bash
mkdir -p logs data outputs
sbatch <job-script>
```

Inside the job, use `cd "$SLURM_SUBMIT_DIR"`, activate `../.venv`, disable JAX
preallocation, and print JAX and Warp device information. JAX may report its GPU
backend as `gpu`; confirm CUDA execution through `jax.devices()` and Warp's
device list. Never interpret a login-node CPU result as a solver qualification.

Start with a reduced grid, case count, collocation set, and optimizer budget.
Scale only after one complete batch job passes backend, parity, residual, and
memory checks. cuDSS workspace and fill can dominate memory beyond raw CSR size.

## Verification and reporting

Use the narrowest meaningful check:

- documentation/config change: Markdown inspection, TOML parse, `bash -n` for
  actual batch files, and CLI help;
- code rename or CLI change: compile, import/config check, and help output;
- numerical change: focused structural test plus required GPU qualification;
- training change: coefficient parity, short training smoke test, validation,
  and metadata inspection.

Do not run a full 100-case, million-point training job merely to validate a
documentation or naming change. Report GPU tests separately from CPU checks.

## Git handoff

Inspect status before edits and again before staging. Stage exact paths:

```bash
git add <exact-paths>
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Keep generated files and unrelated parent-repository changes unstaged. Use one
focused Conventional Commit when requested. Do not push automatically.
