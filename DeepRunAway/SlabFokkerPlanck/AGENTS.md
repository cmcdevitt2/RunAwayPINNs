# Codex instructions for SlabFokkerPlanck

## Mission

Maintain a compact, reproducible GPU forward finite-volume Fokker--Planck
workflow. Preserve the physical and numerical contract represented by the
source, configuration, and LaTeX reference. Treat conservation, positivity,
GPU residency, cuDSS use, and reproducibility as first-class requirements.

## Operating mode

- Read `README.md`, the relevant TOML, and targeted symbols before broad scans.
- Keep changes narrow. Do not add dependencies, abstractions, or output schemas
  without a concrete requirement.
- Preserve unrelated parent-repository changes; stage exact paths only.
- Before a commit, inspect `git diff --cached --check`, `--stat`, and the exact
  staged diff. Do not push or submit jobs unless explicitly requested.
- Keep chat updates concise: result, paths, checks, blockers, choices.

## Delegation and skill routing

Use skills only when relevant:

- `cavecrew`: bounded investigation, one/two-file edit, or staged-diff review.
- `caveman`: terse agent communication and status reporting when requested.
- `investigate-first`: unknown numerical, runtime, or performance failure.
- `surgical-patch`: narrow bug fix with focused proof.
- `safe-refactor`: behavior-preserving rename or restructuring.
- `lean-build`: new coupling behavior with explicit stop conditions.
- `verify-and-stop`: validation-only work.
- `openai-docs`: Codex/OpenAI behavior questions.
- `find-skills`/`skill-installer`: only for explicit skill discovery or install.

Delegate deterministic searches, file inventories, syntax checks, and staged
diff review to an efficient model. Keep coupled numerical reasoning, coupling
design, and integration in the main context. Ask delegates for compressed
path/line findings or a small patch, not an essay; never give agents overlapping
edit ownership.

## Scientific and runtime rules

- `forward_fv_solver.py` owns forward FV assembly, boundary conditions, transient
  stepping, and cuDSS solves. Preserve FP64 and cell/state conventions.
- The current run uses a prescribed plasma state. Do not imply that it already
  exchanges state with a bulk plasma model.
- `CudssDirectSystem.refactorize()` reuses analyzed topology only; it is not a
  substitute for proving coefficient-update correctness or conservation.
- Keep Warp, cuDSS, and any future JAX component on the same CUDA device.
- Production runs require Slurm GPU batch jobs on Hipergator. Login nodes are for
  inspection, syntax/config checks, and submission. Never accept CPU fallback.
- Use the neighboring `../.venv` environment from this directory.
- Keep generated data, logs, plots, caches, and scratch outputs out of Git.

## Verification ladder

1. Python syntax, CLI help, TOML parsing, and import/config construction.
2. GPU allocation check: Warp CUDA device, cuDSS import, device identity, FP64.
3. Small forward GPU case: finite/nonnegative state, balance/normalization,
   linear residual, and stable timestep completion.
4. Qualification: equilibrium/no-drive behavior, collision-overlap and boundary
   checks, mesh/time refinement, `q_coll`/`q_init` sensitivity, and memory.
5. Coupling work: coefficient parity, conservation exchange, topology reuse vs
   rebuild, refactorization residuals, restart behavior, and held-out cases.

Adjoint RPF stress tests in `SlabRpfPinn` are useful evidence for cuDSS ordering,
fill, and memory, but they do not qualify this forward transient solver.

## Completion report

State result first; list changed files, checks performed, known limitations, and
commit hash when committed. Stop after the required verification passes.
