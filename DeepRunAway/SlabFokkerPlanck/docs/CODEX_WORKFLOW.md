# Codex workflow

This is the compact playbook for Codex sessions in `SlabFokkerPlanck`.

## Start small

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
rg --files -g '!outputs/**' -g '!logs/**' -g '!*.npz'
```

Read `README.md`, then the relevant function and TOML section. Search symbols
and callers with `rg`; do not read generated data or the full LaTeX reference
unless the task needs its equations or conventions.

## Route work

- Locate a definition or runtime assumption: `cavecrew-investigator`.
- Make a clear one/two-file edit: `cavecrew-builder` or a surgical patch.
- Diagnose an unknown numerical/performance issue: `investigate-first` first.
- Review a staged diff: `cavecrew-reviewer` or `caveman-review`.
- Validate existing work: `verify-and-stop`.
- Design a new bulk-plasma coupling slice: `lean-build`; keep the interface,
  update cadence, and stop condition explicit.

Use fast models for deterministic location, metadata, syntax, and review. Keep
physics, matrix semantics, GPU residency, and cross-file integration in the
main context. Delegated prompts should name exact paths/symbols and request
compressed findings.

## Scope boundaries

The stable executable is `forward_fv_solver.py`; the stable config is
`forward_fv_solver.toml`. Preserve its FP64 Warp kernels, GPU CSR topology,
cuDSS direct solve, transient scheme, boundary semantics, and output schema.
The solver currently consumes prescribed plasma parameters. A coupling change
must specify: exchanged state, update cadence, coefficient rebuild policy,
topology reuse/refactorization policy, conservation exchange, and restart data.
OpenADAS acquisition is handled separately through `scripts/fetch_openadas.py`;
downloaded ADF11 files and manifests stay outside Git.

Do not port adjoint-only RPF identities or PINN training checks as if they were
forward-solver qualification. Reuse the SlabRpfPinn stress evidence for cuDSS
ordering and memory planning, then add forward-specific tests.

## HPC and verification

Follow `docs/HIPERGATOR.md`. Production execution is a Slurm GPU batch job;
login-node execution is limited to static checks and submission. Begin with a
small case and record GPU, environment, Git commit, TOML, job ID, residuals,
state bounds, and peak memory before scaling.
The supported Python environment is `DeepRunAway/.venv`, which is `../.venv`
when the working directory is `DeepRunAway/SlabFokkerPlanck`; activate it
before Python checks and inside every submitted batch script.

For source/config/docs changes, run syntax, TOML, import/config, and `--help`
checks. For numerical changes, add a small GPU run and the relevant conservation,
residual, refinement, and memory checks. Do not run a large production case to
validate documentation.

## Git discipline

The directory is nested in the public `RunAwayPINNs` repository. Check the
repository root before staging and never use blanket staging:

```bash
git add DeepRunAway/SlabFokkerPlanck/<exact-paths>
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Keep generated results and unrelated parent-repository work unstaged. Use one
focused Conventional Commit for a coherent change.
