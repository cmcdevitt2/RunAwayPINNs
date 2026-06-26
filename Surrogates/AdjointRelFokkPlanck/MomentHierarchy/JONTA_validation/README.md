# JONTA Validation — Runaway Electron Moment Pipeline

Monte Carlo validation of the runaway electron moment surrogates using a JAX-based
relativistic guiding-center solver (JONTA).  The pipeline has three stages driven by
the primary orchestration script `Jonta_bash_moments.sh`.

---

## Scripts

| File | Role |
|------|------|
| `Jonta_bash_moments.sh` | Orchestrator — sets plasma parameters, loops over configurations, and calls the three Python scripts in sequence for each run |
| `JaxScript_moments.py` | Stage 1 — integrates N particles forward in time using RK4 + stochastic pitch-angle scattering, writes HDF5 snapshots |
| `BinParticles_one_gpu_moments.py` | Stage 2 — reads HDF5 snapshots, CIC-bins the phase-space distribution f(γ, ξ) onto a uniform grid, writes `fDist_step_XXXXXX.txt` |
| `Plot_bins_moments.py` | Stage 3 — loads the binned distributions, computes moments (n, j, E, pressure), saves time-series to `data/` and generates plots |

---

## How to Run

Run from the **repo root** (one level above `JONTA_validation/`):

```bash
bash JONTA_validation/Jonta_bash_moments.sh
```

The script sets all parameters internally — no command-line arguments are needed.

---

## Plasma Parameters

Edited directly at the top of `Jonta_bash_moments.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `Ebar` | `2.5` | Normalized electric field E/E_c |
| `Z_eff` | `3.0` | Effective ion charge |
| `alpha` | `0.1` | Synchrotron radiation coefficient |
| `n_steps` | `20_000` | Number of RK4 time steps |
| `dt` | `1e-3` | Time step size (units of collision time τ_c) |
| `N` | `10_000_000` | Number of Monte Carlo particles |
| `Energy_threshold` | `1` MeV | Minimum energy for a particle to be counted as a runaway in moment calculations |

---

## Pipeline Details

### Stage 1 — `JaxScript_moments.py`

Integrates the relativistic guiding-center equations in (γ, ξ) space:

- **Deterministic push**: RK4 on γ̇ and ξ̇ driven by the electric field `Ebar`, synchrotron drag `alpha`, and friction `C_F = γ²/p²`.
- **Stochastic pitch-angle kick**: Rademacher-noise approximation to the Langevin equation with `ν_D = γ(Z_eff+1)/p³`.
- **Boundary conditions**:
  - Low boundary (thermalized): particle γ is reset to `p_thermal` and xi is held.
  - High boundary (runaway): particle γ is clamped at `g_max = (16 MeV/m_e c² + 1)`.
  - `freeze_mode=True` — particles that reach the high boundary are frozen (γ and ξ held fixed); models the absorbing boundary used by the adjoint solver.
  - `freeze_mode=False` — no high-energy absorbing boundary; particles continue evolving (full dynamics).

Snapshots of (γ, ξ, ids) are written as compressed HDF5 files:

```
JONTA_validation/particles/<run_id>/device_0/step_XXXXXX.h5
```

The initial pitch-angle distribution is controlled by `--ximin` / `--ximax`:

| Configuration | ximin | ximax | Description |
|--------------|-------|-------|-------------|
| `aligned` | -1.0 | -0.9 | Beam aligned with B field (ξ ≈ -1) |
| `opposed` | 0.9 | 1.0 | Beam opposed to B field (ξ ≈ +1) |
| `isotropic` | -1.0 | 1.0 | Uniform in ξ |

### Stage 2 — `BinParticles_one_gpu_moments.py`

Reads the HDF5 snapshots and produces the phase-space distribution on a 128×128
(γ, ξ) grid using Cloud-In-Cell (CIC) interpolation with a phase-space Jacobian
correction (2π p γ dγ dξ).  Grid bounds: E ∈ [1, 16] MeV.

Outputs per requested time step:

```
<run_id>/particles/fDist_step_XXXXXX.txt   # columns: energy_MeV  xi  f
<run_id>/particles/g_nodes.txt
<run_id>/particles/xi_nodes.txt
```

### Stage 3 — `Plot_bins_moments.py`

For each saved time step, computes four moments by direct particle summation from the
raw HDF5 files (faster and more accurate than integrating the binned distribution):

| Output file | Quantity |
|-------------|----------|
| `data/n_vs_time.txt` | Runaway fraction n/N |
| `data/j_vs_time.txt` | Normalized current ∑ v·ξ / N |
| `data/E_vs_time.txt` | Mean energy ∑ 0.511(γ-1) / N [MeV] |
| `data/pressure_vs_time.txt` | Pressure anisotropy ∑ v² P₂(ξ) / N |

Only particles with energy > `Energy_threshold` are counted as runaways.

2D distribution plots are saved to `plots/` and xi-integrated energy spectra to
`plots/energy_distrib_*.png`.

---

## Run Naming Convention

Each run is identified by a string like:

```
REmoments_E=2.50_Zeff=3.00_alpha=0.1000_isotropic
```

`_full` suffix → `freeze_mode=False`; no suffix → `freeze_mode=True`.

---

## What the Bash Script Runs

`Jonta_bash_moments.sh` executes the full pipeline for the following parameter sweep
(all at `Energy_threshold = 1 MeV`):

1. **Orientation study** at `(Ebar=2.5, Z_eff=3.0, alpha=0.1)`: aligned and opposed
   initializations, both with and without freeze_mode.
2. **Ebar scan** `{4.0, 2.5, 1.5}` at fixed `Z_eff=3.0`, `alpha=0.1`, isotropic IC.
3. **Z_eff scan** `{1.0, 5.0}` at fixed `Ebar=2.5`, `alpha=0.1`, isotropic IC.
4. **alpha scan** `{0.05, 0.2}` at fixed `Ebar=2.5`, `Z_eff=3.0`, isotropic IC.
5. **Energy distribution cases** (freeze_mode=False only):
   - `(Ebar=1.5, Z_eff=1.0, alpha=0.2)`
   - `(Ebar=2.5, Z_eff=3.0, alpha=0.05)`
   - `(Ebar=4.0, Z_eff=5.0, alpha=0.1)`

---

## Output Directory Layout

```
JONTA_validation/
└── particles/
    └── <run_id>/
        ├── device_0/
        │   ├── step_000000.h5
        │   ├── step_001000.h5
        │   └── ...
        ├── particles/
        │   ├── g_nodes.txt
        │   ├── xi_nodes.txt
        │   └── fDist_step_XXXXXX.txt
        ├── plots/
        │   ├── fDist_step_XXXXXX.png
        │   └── energy_distrib_XXXXXX.png
        ├── data/
        │   ├── n_vs_time.txt
        │   ├── j_vs_time.txt
        │   ├── E_vs_time.txt
        │   └── pressure_vs_time.txt
        └── figures/
```
