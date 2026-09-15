# Project context

This document states the premises that shape every other document in this
project. Read it before `CLAUDE.md` or `AGENTS.md`. It is agent-facing, not
part of the public-facing `README.md`.

## Repository context

`SlabRpfPinn` is a subdirectory of the larger `RunAwayPINNs` monorepo
(`DeepRunAway/SlabRpfPinn`). It is developed on a feature branch off `main`.
Consequences:

- Only touch paths under `SlabRpfPinn/`.
- Preserve unrelated changes elsewhere in the working tree; stage exact paths,
  never blanket-stage.

## Local machine vs. remote HPC execution

Development and remote execution are two different machines:

- **Local machine**: editing, configuration review, syntax/import checks, JSON
  validation, Slurm job submission.
- **Remote HPC cluster** (Perlmutter): production FV dataset generation and
  model training/validation run here, inside a Slurm allocation. The login
  node is for editing and submission only, same as the local machine — never
  run production workloads there.

This split is why the verification ladder in `CLAUDE.md`/`AGENTS.md` stops at
import/structural checks when working locally, and why claims of multi-node
qualification require an actual HPC allocation, not just a successful local
import.

## Self-contained environment

The Python environment lives *inside* this directory, on every machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Addressed as `./.venv` from `SlabRpfPinn/`. Never activate or reference an
environment outside this directory (e.g. a conda environment on `/pscratch`
or similar shared scratch paths). This project previously depended on such
an external environment; reaching outside the project directory on HPC was a
known mistake and must not be repeated.

JAX XLA-compiles to whatever hardware it sees: a plain
`pip install -r requirements.txt` runs correctly on CPU alone, for every
stage including model training/validation. GPU is optional and only changes
performance. To use one, install JAX and its CUDA plugin together, matched
to the allocated GPU's CUDA version. Perlmutter currently provides CUDA 13
GPUs, so a GPU install there replaces the plain `jax` line in
`requirements.txt` with:

```bash
pip install "jax[cuda13]"
```

See `docs/DEPENDENCIES.md` for the full package list and split between the
CPU-only FV-generation profile and the training/validation profile.

## Solver architecture

This project's solver is the CPU finite-volume code in `core/rpf_fv_cpu.py`.
JAX is used only for the PINN model (architecture, training, PDE residuals),
not for FV label generation, which stays outside JAX automatic
differentiation.
