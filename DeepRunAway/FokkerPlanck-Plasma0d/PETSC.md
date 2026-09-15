# PETSc CPU solver environment

Project-local environment is `.petsc-cpu-venv/`. It contains PETSc 3.25.5,
petsc4py 3.25.5, and a locally built mpi4py, all built against the Cray-MPICH
compiler wrappers for CPU-only execution.
The build includes Numba for compiled stencil filling, plus MUMPS, ScaLAPACK,
METIS, ParMETIS, and PT-Scotch.

Recreate the environment with the pinned packages and PETSc external-solver
configuration:

```bash
module purge
module load python/3.12-26.1.0
unset LD_LIBRARY_PATH
python -m venv .petsc-cpu-venv
export PATH="$PWD/.petsc-cpu-venv/bin:/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/bin:/opt/cray/pe/gcc-native/14/bin:/usr/bin:/bin"
export MPICC="$(command -v mpicc)"
export MPICXX="$(command -v mpicxx)"
export MPIFORT="$(command -v mpifort)"
export MPICH_GPU_SUPPORT_ENABLED=0
export PETSC_CONFIGURE_OPTIONS="--with-cuda=0 --with-hip=0 --with-kokkos=0 \
  --download-mumps --download-scalapack \
  --download-metis --download-parmetis --download-ptscotch"
.petsc-cpu-venv/bin/python -m pip install --no-cache-dir --no-binary=mpi4py \
  -r petsc-requirements.txt
```

The current Perlmutter Cray-MPICH stack enables GPU-aware initialization by
default. The CPU PETSc path must disable that runtime mode, even though no GPU
is allocated:

```bash
export MPICH_GPU_SUPPORT_ENABLED=0
source .petsc-cpu-venv/bin/activate
```

Run the forward solver inside a Slurm allocation:

```bash
MPICH_GPU_SUPPORT_ENABLED=0 srun -n 4 \
  /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/.petsc-cpu-venv/bin/python \
  /pscratch/sd/j/jsarnaud/git/RunAwayPINNs/DeepRunAway/SlabRpfPinn/FokkerPlanck-Plasma0d/petsc_forward_solver.py \
  --coupled --pmax 100.0 \
  --t-end 1.0e-4 --dtau-initial 2.5e-5
```

Use `--coupled` for adaptive electric-field/induction Newton coupling. Both
constant and coupled paths use the exact auxiliary Chiu--Harvey radial block.

PETSc selects MUMPS through `PC` type `lu` and factor solver type `mumps`.
The solver sets these options programmatically, so no PETSc installation is
needed on compute nodes beyond this environment and matching MPI libraries.
