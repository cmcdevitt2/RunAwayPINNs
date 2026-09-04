#!/usr/bin/env python3
"""
Standalone GPU 0D-2P adjoint/runaway-probability-function solver (unified steady + optional time-dependent TR-BDF2).

Purpose
-------
This is a standalone research script for the 0D-2P adjoint / runaway probability
function (RPF).  It can solve either the direct steady first-passage RPF only, or
the steady RPF followed by the finite-horizon time-dependent adjoint.  It uses the same qualified local forward
finite-volume generator as the steady RPF script, including the fully-relativistic
asymptotically matched test-particle collision operator, Hesslow energy-dependent
Coulomb logarithms, optional Hesslow partial screening, optional radial energy
diffusion, electric acceleration, synchrotron radiation, and the same absorbing
p_min / successful-open-outflow p_max semantics.

The Chiu-Harvey source is deliberately excluded from the tagged-electron adjoint
generator.  It is a branching source and can be contracted with an RPF afterward.

Time-dependent adjoint
----------------------
Let dN/dtau = L0 N be the local forward semidiscrete cell-content system and let
r_RE be the successful p_max escape-rate vector.  If T is the terminal physical
time and s=T-t is lookback time, the finite-horizon RPF satisfies

    dP/ds = L0^T P + r_RE,
    P(s=0) = g_RE,

where g_RE is the configured discrete terminal response.  The script uses the
exact algebraic transpose of the forward finite-volume generator.  This is the
physical semidiscrete time-dependent adjoint. When requested, this solver integrates it with
TR-BDF2; it is not the reverse-mode transpose of a particular already-discretized
time-marching map.  The terminal response can be either the exact discontinuous
Heaviside H(p-p_RE), projected in the p^2 finite-volume measure, or a tanh-smoothed
Heaviside volume-averaged in that same measure.

As s -> infinity the transient terminal contribution decays and the solution tends
to the already-qualified steady first-passage RPF,

    (-L0^T) P_inf = r_RE.

Fixed-step implicit integration
-------------------------------
This experiment replaces the BE-started BDF2 march by standard fixed-step TR-BDF2
with gamma=2-sqrt(2).  The method is second order, one-step, and L-stable.  For this
choice of gamma, the trapezoidal and BDF2 substeps use exactly the same implicit
matrix

    M = I + (gamma/2) dtau A,   A=-L0^T,

so the transient requires one numerical factorization of M and two triangular solves
per full timestep.  The factorization is reused for the complete time march.  The terminal-response
choice is independent of the time integrator.  No physical coefficient, source,
boundary condition, operator term, or RPF state is otherwise modified.

The direct steady RPF is solved first on the GPU and retained there.  The transient
integration stops only after it agrees with that GPU-resident steady target in both
relative L2 and maximum absolute error for several consecutive checks.  These
comparisons use GPU reductions and return only scalar diagnostics to the host.  The
timestep is deliberately a direct numerical-resolution axis: finite-horizon RPF
slices must be repeated with smaller dtau to establish temporal convergence.

Configuration
-------------
Inputs default to the TOML file beside this script.  ``[solve].mode`` is either
``steady`` or ``steady_and_time``.  Use ``--config`` to select another case.
The reference normalization is n_ref=n_e
and lnLambda_ref=lnLambda_0, so E_over_Ec is exactly E/E_c.
"""

from __future__ import annotations

import ctypes
import argparse
from dataclasses import asdict, dataclass, field
import importlib.metadata as importlib_metadata
import json
import math
from pathlib import Path
import site
import time
import tomllib

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.special as sps

try:
    import warp as wp
except Exception:
    wp = None


MEC2_EV = const.m_e * const.c**2 / const.e
CONFIG_PATH = Path(__file__).with_suffix(".toml")

# Algebraic acceptance tolerances for per-run RPF qualification.  Spatial/domain
# convergence remains a separate multi-run requirement.
RPF_EXACT_REL_TOL = 5.0e-13
RPF_IDENTITY_REL_TOL = 5.0e-11
# The escape identity can be strongly cancellation-conditioned on very broad
# momentum domains.  Retain the strict physical relative check, but allow a
# scale-aware roundoff fallback when the componentwise backward error is tiny.
RPF_IDENTITY_BACKWARD_TOL = 5.0e-13
RPF_PROBABILITY_TOL = 1.0e-8


# Hesslow et al. (2018), Table 1: normalized DFT-matched screening length
# a_bar = 2 a / alpha, indexed by ion charge state Z0.  Arnaud et al.
# (2025) uses singly ionized argon in its fixed-composition RPF examples and
# neon in the disruption-oriented parameter scans, so both full charge-state
# sequences are embedded here.
HESSLOW_ABAR_BY_Z: dict[int, tuple[float, ...]] = {
    1: (190.0,),
    10: (111.0, 100.0, 90.0, 80.0, 71.0, 62.0, 52.0, 40.0, 24.0, 23.0),
    18: (96.0, 90.0, 84.0, 78.0, 72.0, 65.0, 59.0, 53.0, 47.0,
         44.0, 41.0, 38.0, 35.0, 32.0, 27.0, 21.0, 13.0, 13.0),
}

# Mean excitation energies I [eV], indexed by charge state Z0, for the same
# Ne and Ar sequences used with the Hesslow Bethe stopping model.
MEAN_EXCITATION_EV_BY_Z: dict[int, tuple[float, ...]] = {
    1: (14.99,),
    10: (137.2, 165.2, 196.9, 235.2, 282.8, 352.6, 475.0, 696.8, 1409.2, 1498.4),
    18: (188.5, 219.4, 253.8, 293.4, 339.1, 394.5, 463.4, 568.0, 728.0,
         795.9, 879.8, 989.9, 1138.1, 1369.5, 1791.2, 2497.0, 4677.2, 4838.2),
}


def tabulated_screening_parameters(Z: int, Z0: int) -> tuple[float, float]:
    """Return (I_eV, a_bar) for a partially ionized Ne or Ar charge state.

    Fully stripped ions have no bound-electron screening contribution and
    therefore return zeros.  For another partially ionized element, provide
    I_eV and a_bar explicitly in the TOML.
    """
    if Z0 >= Z:
        return 0.0, 0.0
    I_values = MEAN_EXCITATION_EV_BY_Z[Z]
    a_values = HESSLOW_ABAR_BY_Z[Z]
    return float(I_values[Z0]), float(a_values[Z0])



# =============================================================================
# Configuration, normalization, and grid
# =============================================================================

@dataclass(frozen=True)
class IonSpecies:
    """Resolved integer charge-state component used by the collision operator."""
    name: str
    Z: int
    Z0: int
    density_m3: float
    I_eV: float = 0.0
    a_bar: float = 0.0

    @property
    def n_bound(self) -> int:
        return self.Z - self.Z0


@dataclass(frozen=True)
class AtomicSpecies:
    """Atomic species specified by total density and possibly non-integer mean charge."""
    name: str
    Z: int
    Zavg: float
    density_m3: float


@dataclass(frozen=True)
class SolverConfig:
    te_eV: float
    E_over_Ec: float
    B_T: float
    species: tuple[AtomicSpecies, ...] = field(default_factory=tuple)

    energy_diffusion: bool = False

    # Unified solve workflow: steady only, or steady followed by time-dependent.
    solve_mode: str = "steady_and_time"

    Np: int = 512
    Nxi: int = 256
    pmin: float = 0.2
    pmax: float = 5.0
    p_mapping: str = "uniform"
    p_mapping_kappa: float = 4.0
    xi_mapping: str = "uniform"

    # Configurable terminal response centered at p_RE.  "heaviside" uses the
    # exact discontinuous FV projection.  "smoothed_heaviside" uses a tanh
    # smoothing with width terminal_smoothing_cells*dp, volume-averaged in p^2 dp.
    terminal_p: float = 1.25
    terminal_condition: str = "heaviside"
    terminal_smoothing_cells: float = 2.0

    # Fixed lookback-time integration controls for TR-BDF2.  dtau is an explicit
    # numerical convergence axis and is never changed during a run.
    dtau: float = 1.0e-3
    max_steps: int = 100000
    max_lookback_tau: float = 1.0e4
    probability_tol: float = 1.0e-8

    # Steady-state stopping criteria.  The direct steady RPF is solved first on
    # the GPU and kept resident there.  The transient is stopped when it agrees
    # with that target in relative L2 and maximum absolute error for several checks.
    steady_check_every: int = 10
    steady_match_rel_l2_tol: float = 1.0e-8
    steady_match_max_abs_tol: float = 1.0e-7
    steady_confirm_steps: int = 3

    device: str = "cuda:0"
    output: Path = Path("runaway_0d2p_adjoint_time.npz")
    write_output: bool = True
    save_every: int = 0


    def resolved_ions(self) -> tuple[IonSpecies, ...]:
        """Expand each mean charge into neighboring integer charge states.

        The weights are the linear shape functions

            w_q(Zavg) = max(0, 1 - |Zavg-q|),

        so integer Zavg is represented exactly by one state and non-integer Zavg
        by the two adjacent states.  The listed density is the total atomic
        density of that species.
        """
        ions: list[IonSpecies] = []
        for spc in self.species:
            for Z0 in range(spc.Z + 1):
                weight = max(0.0, 1.0 - abs(spc.Zavg - float(Z0)))
                if weight <= 0.0:
                    continue
                if Z0 < spc.Z:
                    I_eV, a_bar = tabulated_screening_parameters(spc.Z, Z0)
                else:
                    I_eV, a_bar = 0.0, 0.0
                ions.append(IonSpecies(
                    name=f"{spc.name}[Z0={Z0}]",
                    Z=spc.Z,
                    Z0=Z0,
                    density_m3=spc.density_m3 * weight,
                    I_eV=I_eV,
                    a_bar=a_bar,
                ))
        return tuple(ions)


@dataclass(frozen=True)
class DerivedPhysics:
    theta: float
    ln_lambda0: float
    z_eff: float
    n_ref_m3: float
    ln_lambda_ref: float
    tau_ref_s: float
    E_ref_Vm: float
    e_parallel_Vm: float
    Ebar: float
    tau_syn_s: float
    alpha: float
    free_density_from_ions_m3: float


@dataclass(frozen=True)
class Grid:
    p_faces: np.ndarray
    p_centers: np.ndarray
    xi_faces: np.ndarray
    xi_centers: np.ndarray
    dp: float
    dxi: float
    radial_volume: np.ndarray
    cell_volume: np.ndarray
    p_cell_widths: np.ndarray
    xi_cell_widths: np.ndarray
    p_face_spacing: np.ndarray
    xi_face_spacing: np.ndarray

    @property
    def Np(self) -> int:
        return self.p_centers.size

    @property
    def Nxi(self) -> int:
        return self.xi_centers.size

    @property
    def size(self) -> int:
        return self.Np * self.Nxi


@dataclass(frozen=True)
class CollisionData:
    cf_faces: np.ndarray
    ca_faces: np.ndarray
    nud_centers: np.ndarray
    phi_centers: np.ndarray
    psi_centers: np.ndarray
    ln_lambda_ee_faces: np.ndarray
    ln_lambda_ei_faces: np.ndarray
    ln_lambda_ee_centers: np.ndarray
    ln_lambda_ei_centers: np.ndarray
    p_switch: float
    overlap_rel: float


@dataclass(frozen=True)
class CSRTopology:
    n: int
    nnz_count: int
    row_ptr: object
    col_ind: object
    diag_slot: object
    pm_slot: object
    pp_slot: object
    xm_slot: object
    xp_slot: object

    @property
    def nnz(self) -> int:
        return self.nnz_count


def thermal_coulomb_log(ne_m3: float, te_eV: float) -> float:
    return 14.9 - 0.5 * math.log(ne_m3 / 1.0e20) + math.log(te_eV / 1.0e3)


def collision_time(ne_m3: float, ln_lambda: float) -> float:
    return 4.0 * math.pi * const.epsilon_0**2 * const.m_e**2 * const.c**3 / (
        ne_m3 * const.e**4 * ln_lambda
    )


def validate_rpf_config(cfg: SolverConfig) -> None:
    """Validate the unified steady / steady+time first-passage configuration."""
    scalars = {
        "plasma.te_eV": cfg.te_eV,
        "plasma.E_over_Ec": cfg.E_over_Ec,
        "plasma.B_T": cfg.B_T,
        "grid.pmin": cfg.pmin,
        "grid.pmax": cfg.pmax,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if cfg.solve_mode not in ("steady", "steady_and_time"):
        raise ValueError('solve.mode must be "steady" or "steady_and_time"')
    if cfg.te_eV <= 0.0:
        raise ValueError("plasma.te_eV must be positive")
    if cfg.B_T < 0.0:
        raise ValueError("plasma.B_T must be non-negative")
    if cfg.Np < 2 or cfg.Nxi < 2:
        raise ValueError("adjoint RPF requires grid.Np >= 2 and grid.Nxi >= 2")
    if cfg.pmin < 0.0:
        raise ValueError("first-passage RPF requires grid.pmin >= 0")
    if cfg.pmax <= cfg.pmin:
        raise ValueError("grid.pmax must be greater than grid.pmin")
    p_mapping = str(cfg.p_mapping).strip().lower()
    if p_mapping not in ("uniform", "log", "exponential"):
        raise ValueError("grid.p_mapping must be 'uniform', 'log', or 'exponential'")
    if p_mapping == "log" and cfg.pmin <= 0.0:
        raise ValueError("grid.p_mapping='log' requires grid.pmin > 0")
    if not math.isfinite(cfg.p_mapping_kappa) or cfg.p_mapping_kappa <= 0.0:
        raise ValueError("grid.p_mapping_kappa must be finite and positive")
    if str(cfg.xi_mapping).strip().lower() not in ("uniform", "theta"):
        raise ValueError("grid.xi_mapping must be 'uniform' or 'theta'")

    if cfg.solve_mode == "steady_and_time":
        time_scalars = {
            "time.terminal_p": cfg.terminal_p,
            "time.terminal_smoothing_cells": cfg.terminal_smoothing_cells,
            "time.dtau": cfg.dtau,
            "time.max_lookback_tau": cfg.max_lookback_tau,
        }
        for name, value in time_scalars.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not (cfg.pmin < cfg.terminal_p < cfg.pmax):
            raise ValueError("time.terminal_p must lie strictly inside (grid.pmin, grid.pmax)")
        if cfg.terminal_condition not in ("heaviside", "smoothed_heaviside"):
            raise ValueError(
                'time.terminal_condition must be "heaviside" or "smoothed_heaviside"'
            )
        if cfg.terminal_smoothing_cells < 0.0:
            raise ValueError("time.terminal_smoothing_cells must be non-negative")
        if cfg.terminal_condition == "smoothed_heaviside" and cfg.terminal_smoothing_cells <= 0.0:
            raise ValueError(
                "time.terminal_smoothing_cells must be positive for smoothed_heaviside"
            )
        if cfg.dtau <= 0.0:
            raise ValueError("time.dtau must be positive")
        if cfg.max_steps < 2:
            raise ValueError("time.max_steps must be at least 2")
        if cfg.max_lookback_tau <= 0.0:
            raise ValueError("time.max_lookback_tau must be positive")
        if cfg.probability_tol <= 0.0:
            raise ValueError("time.probability_tol must be positive")
        if cfg.steady_match_rel_l2_tol <= 0.0 or cfg.steady_match_max_abs_tol <= 0.0:
            raise ValueError("steady target-match tolerances must be positive")
        if cfg.steady_check_every < 1:
            raise ValueError("time.steady_check_every must be positive")
        if cfg.steady_confirm_steps < 1:
            raise ValueError("time.steady_confirm_steps must be positive")

    if cfg.save_every < 0:
        raise ValueError("output.save_every must be non-negative")
    if not cfg.species:
        raise ValueError("plasma.species must contain at least one atomic species")
    for spc in cfg.species:
        if spc.Z <= 0:
            raise ValueError(f"invalid nuclear charge for species {spc.name!r}: Z={spc.Z}")
        if not math.isfinite(spc.Zavg) or not (0.0 <= spc.Zavg <= spc.Z):
            raise ValueError(
                f"species {spc.name!r} requires 0 <= Zavg <= Z; got Zavg={spc.Zavg}, Z={spc.Z}"
            )
        if not math.isfinite(spc.density_m3) or spc.density_m3 < 0.0:
            raise ValueError(f"species {spc.name!r} density must be finite and non-negative")
    resolved = cfg.resolved_ions()
    ne_quasineutral = sum(x.density_m3 * x.Z0 for x in resolved)
    if not math.isfinite(ne_quasineutral) or ne_quasineutral <= 0.0:
        raise ValueError(
            "quasineutral free-electron density sum_s n_s <Z_s> must be positive"
        )
    for ion in resolved:
        if ion.Z <= 0 or ion.Z0 < 0 or ion.Z0 > ion.Z:
            raise ValueError(f"invalid charge state for ion {ion.name!r}: Z={ion.Z}, Z0={ion.Z0}")
        if not math.isfinite(ion.density_m3) or ion.density_m3 < 0.0:
            raise ValueError(f"ion {ion.name!r} density must be finite and non-negative")
        if ion.density_m3 > 0.0 and ion.n_bound > 0:
            if not math.isfinite(ion.I_eV) or ion.I_eV <= 0.0:
                raise ValueError(f"ion {ion.name!r} requires positive I_eV for partial screening")
            if not math.isfinite(ion.a_bar) or ion.a_bar <= 0.0:
                raise ValueError(f"ion {ion.name!r} requires positive a_bar for partial screening")

def derive_physics(cfg: SolverConfig) -> DerivedPhysics:
    """Use the physical collision time itself as the reference normalization.

    Therefore Ebar = E/Ec exactly, matching the steady-state RPF convention.
    """
    theta = cfg.te_eV / MEC2_EV
    ions = cfg.resolved_ions()
    free_from_ions = sum(x.density_m3 * x.Z0 for x in ions)
    if free_from_ions <= 0.0:
        raise ValueError("quasineutral free-electron density must be positive")

    # Quasineutrality: ne is derived from the atomic species densities and their
    # mean charge states.  Since resolved_ions() uses linear shape functions,
    # this is exactly sum_s n_s * Zavg_s for both integer and non-integer Zavg.
    ne_m3 = free_from_ions
    ln0 = thermal_coulomb_log(ne_m3, cfg.te_eV)
    if ln0 <= 0.0:
        raise ValueError(f"thermal Coulomb logarithm is non-positive: {ln0}")

    z_eff = sum(x.density_m3 * x.Z0**2 for x in ions) / ne_m3

    nref = ne_m3
    lnref = ln0
    tauref = collision_time(nref, lnref)
    eref = const.m_e * const.c / (const.e * tauref)
    e_parallel = cfg.E_over_Ec * eref

    if cfg.B_T == 0.0:
        taus = math.inf
        alpha = 0.0
    else:
        taus = 6.0 * math.pi * const.epsilon_0 * const.m_e**3 * const.c**3 / (
            const.e**4 * cfg.B_T**2
        )
        alpha = tauref / taus

    return DerivedPhysics(
        theta=theta,
        ln_lambda0=ln0,
        z_eff=z_eff,
        n_ref_m3=nref,
        ln_lambda_ref=lnref,
        tau_ref_s=tauref,
        E_ref_Vm=eref,
        e_parallel_Vm=e_parallel,
        Ebar=cfg.E_over_Ec,
        tau_syn_s=taus,
        alpha=alpha,
        free_density_from_ions_m3=free_from_ions,
    )


def build_grid(cfg: SolverConfig) -> Grid:
    mapping = str(cfg.p_mapping).strip().lower()
    if mapping == "uniform":
        pf = np.linspace(cfg.pmin, cfg.pmax, cfg.Np + 1, dtype=np.float64)
    elif mapping == "log":
        s = np.linspace(0.0, 1.0, cfg.Np + 1, dtype=np.float64)
        pf = np.exp(np.log(cfg.pmin) + s * (np.log(cfg.pmax) - np.log(cfg.pmin)))
    elif mapping == "exponential":
        kappa = float(cfg.p_mapping_kappa)
        s = np.linspace(0.0, 1.0, cfg.Np + 1, dtype=np.float64)
        pf = cfg.pmin + (cfg.pmax-cfg.pmin)*np.expm1(kappa*s)/np.expm1(kappa)
    else:
        raise ValueError(f"unsupported p_mapping {cfg.p_mapping!r}")
    p_widths = np.diff(pf)
    pc = 0.5*(pf[:-1] + pf[1:])
    dp = float(np.mean(p_widths))
    xi_mapping = str(cfg.xi_mapping).strip().lower()
    dxi = 2.0 / cfg.Nxi
    if xi_mapping == "uniform":
        xf = np.linspace(-1.0, 1.0, cfg.Nxi + 1, dtype=np.float64)
        xc = -1.0 + (np.arange(cfg.Nxi, dtype=np.float64) + 0.5) * dxi
    elif xi_mapping == "theta":
        theta_faces = np.linspace(0.0, math.pi, cfg.Nxi + 1, dtype=np.float64)
        theta_centers = 0.5 * (theta_faces[:-1] + theta_faces[1:])
        xf = -np.cos(theta_faces)
        xc = -np.cos(theta_centers)
    else:
        raise ValueError(f"unsupported xi_mapping {cfg.xi_mapping!r}")
    xi_widths = np.diff(xf)
    p_spacing = np.empty(cfg.Np + 1, dtype=np.float64)
    p_spacing[0] = pc[0] - pf[0]
    p_spacing[1:-1] = pc[1:] - pc[:-1]
    p_spacing[-1] = pf[-1] - pc[-1]
    if xi_mapping == "uniform":
        xi_spacing = np.full(cfg.Nxi + 1, dxi, dtype=np.float64)
    else:
        xi_spacing = np.empty(cfg.Nxi + 1, dtype=np.float64)
        xi_spacing[0] = xc[0] - xf[0]
        xi_spacing[1:-1] = xc[1:] - xc[:-1]
        xi_spacing[-1] = xf[-1] - xc[-1]
    rv = (2.0 * math.pi / 3.0) * (pf[1:]**3 - pf[:-1]**3)
    cv = np.repeat(rv, cfg.Nxi) * np.tile(xi_widths, cfg.Np)
    return Grid(
        pf, pc, xf, xc, dp, dxi, rv, cv,
        p_widths, xi_widths, p_spacing, xi_spacing,
    )


# =============================================================================
# Small-angle collision coefficients derived from the qualified forward solver
# with the requested Hesslow energy-dependent Coulomb-log correction
# =============================================================================

def energy_dependent_coulomb_logs(
    p: np.ndarray, phys: DerivedPhysics, k: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Hesslow thermal-to-relativistic matched Coulomb logarithms.

    The thermal speed parameter is u_Te = sqrt(2 Theta).  The electron-electron
    logarithm uses the kinetic-energy variable 2(gamma-1)/u_Te^2, while the
    electron-ion logarithm uses 2p/u_Te.  gamma-1 is evaluated as
    p^2/(gamma+1) to avoid cancellation at low momentum.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0.0):
        raise ValueError("momentum must be non-negative")
    if k <= 0.0:
        raise ValueError("Coulomb-log matching exponent k must be positive")

    u_te = math.sqrt(2.0*phys.theta)
    if u_te <= 0.0:
        raise ValueError("positive thermal speed is required")

    gamma = np.sqrt(1.0 + p*p)
    gamma_minus_one = p*p/(gamma + 1.0)
    with np.errstate(over="raise", invalid="raise"):
        q_ee = 2.0*gamma_minus_one/(u_te*u_te)
        q_ei = 2.0*p/u_te
        ln_ee = phys.ln_lambda0 + np.log1p(q_ee**(0.5*k))/k
        ln_ei = phys.ln_lambda0 + np.log1p(q_ei**k)/k
    return ln_ee, ln_ei


def screening_h_g(p: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p*p)
    beta2 = p*p/(gamma*gamma)
    h = np.zeros_like(p)
    g = np.zeros_like(p)
    for ion in cfg.resolved_ions():
        if ion.density_m3 == 0.0 or ion.n_bound == 0:
            continue
        rr = ion.density_m3 / phys.free_density_from_ions_m3
        Ibar = ion.I_eV / MEC2_EV
        q = p*np.sqrt(np.maximum(gamma - 1.0, 0.0))/Ibar
        h += rr*ion.n_bound*(0.2*np.log1p(q**5) - beta2)
        y = (p*ion.a_bar)**1.5
        g += rr*((2.0/3.0)*(ion.Z**2 - ion.Z0**2)*np.log1p(y)
                 - (2.0/3.0)*ion.n_bound**2*y/(1.0 + y))
    return h, g


def _matched_chandrasekhar_phi_psi(x: np.ndarray, return_branches: bool = False):
    x = np.asarray(x, dtype=np.float64)
    phi = sps.erf(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_dir = (phi - 2.0*x*np.exp(-x*x)/math.sqrt(math.pi))/(2.0*x*x)

    z = x*x
    poly = (((((-1.0/1560.0*z + 1.0/264.0)*z - 1.0/54.0)*z
               + 1.0/14.0)*z - 1.0/5.0)*z + 1.0/3.0)
    psi_ser = (2.0*x/math.sqrt(math.pi))*poly
    x_switch = 0.08
    low = np.abs(x) <= x_switch
    psi = np.where(low, psi_ser, psi_dir)
    if return_branches:
        return phi, psi, psi_dir, psi_ser, x_switch
    return phi, psi


def fully_relativistic_asymptotically_matched_collision_coefficients(
    p: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics
):
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p*p)
    delta = math.sqrt(2.0*phys.theta)
    if delta <= 0.0:
        raise ValueError("positive thermal speed is required")
    x = p/(delta*gamma)
    phi, psi, psi_dir, psi_ser, x_switch = _matched_chandrasekhar_phi_psi(
        x, return_branches=True
    )
    chi_c = (phys.free_density_from_ions_m3/phys.n_ref_m3)*(phys.ln_lambda0/phys.ln_lambda_ref)

    # Hesslow energy-dependent Coulomb-log corrections.  The matched drag is
    # electron-electron, while the deflection coefficient contains distinct
    # electron-electron and electron-ion pieces.  Radial energy diffusion is
    # unchanged by these logarithmic corrections.
    ln_ee, ln_ei = energy_dependent_coulomb_logs(p, phys)
    r_ee = ln_ee/phys.ln_lambda0
    r_ei = ln_ei/phys.ln_lambda0
    ee_deflection = phi - psi + delta*delta*p*p/(2.0*gamma*gamma)

    with np.errstate(divide="ignore", invalid="ignore"):
        cf = chi_c*r_ee*2.0*psi/(delta*delta)
        ca = chi_c*(gamma/p)*psi
        nud = chi_c*(gamma/p**3)*(
            phys.z_eff*r_ei + r_ee*ee_deflection
        )

    # Partial screening is a separate additive layer.  The h and g terms are
    # normalized to lnLambda0 and are not multiplied by r_ee or r_ei.
    h, g = screening_h_g(p, cfg, phys)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = cf + chi_c*(gamma*gamma/(p*p))*(h/phys.ln_lambda0)
        nud = nud + chi_c*(gamma/(p**3))*(g/phys.ln_lambda0)

    return cf, ca, nud, phi, psi, psi_dir, psi_ser, x_switch


def build_collision_data(cfg: SolverConfig, phys: DerivedPhysics, grid: Grid) -> CollisionData:
    """Evaluate the fixed fully-relativistic asymptotically matched coefficients."""
    cf_f = np.zeros(grid.Np + 1, dtype=np.float64)
    ca_f = np.zeros(grid.Np + 1, dtype=np.float64)
    face_start = 1 if grid.p_faces[0] == 0.0 else 0
    face_eval = grid.p_faces[face_start:]

    c = fully_relativistic_asymptotically_matched_collision_coefficients(
        grid.p_centers, cfg, phys
    )
    nud_c = c[2]
    phi_c, psi_c = c[3], c[4]

    f = fully_relativistic_asymptotically_matched_collision_coefficients(
        face_eval, cfg, phys
    )
    cf_f[face_start:] = f[0]
    ca_f[face_start:] = f[1]
    if not cfg.energy_diffusion:
        ca_f.fill(0.0)

    # Report the location and direct/series agreement of the small-x branch.
    delta = math.sqrt(2.0*phys.theta)
    x_switch = c[7]
    a = delta*x_switch
    if a >= 1.0:
        raise ValueError(
            "temperature is outside the non-relativistic-bulk ordering used by the "
            "fully-relativistic asymptotically matched test-particle model"
        )
    p_switch = a/math.sqrt(max(1.0-a*a, np.finfo(float).tiny))
    x_test = np.geomspace(x_switch*0.7, x_switch*1.3, 24)
    _, _, direct, series, _ = _matched_chandrasekhar_phi_psi(
        x_test, return_branches=True
    )
    overlap = float(
        np.max(np.abs(direct-series)/np.maximum(np.abs(series), 1e-300))
    )

    # The Coulomb logarithms themselves are regular at p=0, so evaluate them
    # on every face even when the coordinate-singular collision coefficient at
    # the origin is deliberately skipped.
    ln_ee_f, ln_ei_f = energy_dependent_coulomb_logs(grid.p_faces, phys)
    ln_ee_c, ln_ei_c = energy_dependent_coulomb_logs(grid.p_centers, phys)

    return CollisionData(
        cf_faces=cf_f,
        ca_faces=ca_f,
        nud_centers=nud_c,
        phi_centers=phi_c,
        psi_centers=psi_c,
        ln_lambda_ee_faces=ln_ee_f,
        ln_lambda_ei_faces=ln_ei_f,
        ln_lambda_ee_centers=ln_ee_c,
        ln_lambda_ei_centers=ln_ei_c,
        p_switch=p_switch,
        overlap_rel=overlap,
    )


def validate_collision_data(coll: CollisionData) -> None:
    arrays = {
        "C_F faces": coll.cf_faces,
        "C_A faces": coll.ca_faces,
        "nu_D centers": coll.nud_centers,
        "Phi centers": coll.phi_centers,
        "Psi centers": coll.psi_centers,
        "lnLambda_ee faces": coll.ln_lambda_ee_faces,
        "lnLambda_ei faces": coll.ln_lambda_ei_faces,
        "lnLambda_ee centers": coll.ln_lambda_ee_centers,
        "lnLambda_ei centers": coll.ln_lambda_ei_centers,
    }
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            bad = int(np.flatnonzero(~np.isfinite(arr))[0])
            raise FloatingPointError(f"non-finite {name} at local index {bad}")
        if np.any(arr < 0.0):
            bad = int(np.flatnonzero(arr < 0.0)[0])
            raise FloatingPointError(
                f"negative {name} at local index {bad}: {arr[bad]:.6e}"
            )
    if not math.isfinite(coll.p_switch) or coll.p_switch <= 0.0:
        raise FloatingPointError("low-p branch switch must be finite and positive")
    if not math.isfinite(coll.overlap_rel):
        raise FloatingPointError("collision direct/series overlap diagnostic is non-finite")


# =============================================================================
# Local five-point CSR graph
# =============================================================================

def _build_local_csr_row_ptr_host(grid: Grid) -> tuple[np.ndarray, int]:
    n = grid.size
    counts = np.full(n, 5, dtype=np.int32)
    counts[:grid.Nxi] -= 1
    counts[(grid.Np-1)*grid.Nxi:] -= 1
    counts[0:n:grid.Nxi] -= 1
    counts[grid.Nxi-1:n:grid.Nxi] -= 1

    row_ptr64 = np.empty(n + 1, dtype=np.int64)
    row_ptr64[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_ptr64[1:])
    nnz = int(row_ptr64[-1])
    if nnz > np.iinfo(np.int32).max:
        raise ValueError("CSR nnz exceeds int32 range required by the cuDSS path")
    return np.ascontiguousarray(row_ptr64.astype(np.int32)), nnz


# =============================================================================
# Warp kernels: graph, forward -L assembly, exact transpose, and RPF RHS
# =============================================================================

if wp is not None:

    @wp.func
    def chang_cooper_delta(w: wp.float64) -> wp.float64:
        """Stable Chang--Cooper exponential-fitting weight."""
        aw = wp.abs(w)
        if aw < wp.float64(1.0e-4):
            w2 = w*w
            return (
                wp.float64(0.5) - w/wp.float64(12.0)
                + w*w2/wp.float64(720.0)
                - w*w2*w2/wp.float64(30240.0)
            )
        if w > wp.float64(50.0):
            return wp.float64(1.0)/w
        if w < wp.float64(-50.0):
            return wp.float64(1.0) + wp.float64(1.0)/w
        return wp.float64(1.0)/w - wp.float64(1.0)/(wp.exp(w)-wp.float64(1.0))


    @wp.func
    def chang_cooper_left_coefficient(
        A: wp.float64, D: wp.float64, dq: wp.float64
    ) -> wp.float64:
        # J = A*((1-delta)*f_left + delta*f_right)
        #     - D*(f_right-f_left)/dq.
        if D > wp.float64(0.0):
            delta = chang_cooper_delta(A*dq/D)
            return A*(wp.float64(1.0)-delta) + D/dq
        # Continuous D -> 0 Chang--Cooper limit.
        if A >= wp.float64(0.0):
            return A
        return wp.float64(0.0)


    @wp.func
    def chang_cooper_right_coefficient(
        A: wp.float64, D: wp.float64, dq: wp.float64
    ) -> wp.float64:
        if D > wp.float64(0.0):
            delta = chang_cooper_delta(A*dq/D)
            return A*delta - D/dq
        # Continuous D -> 0 Chang--Cooper limit.
        if A < wp.float64(0.0):
            return A
        return wp.float64(0.0)

    @wp.kernel
    def fill_local_csr_topology_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        Np: int,
        Nxi: int,
        col_ind: wp.array(dtype=wp.int32),
        diag_slot: wp.array(dtype=wp.int32),
        pm_slot: wp.array(dtype=wp.int32),
        pp_slot: wp.array(dtype=wp.int32),
        xm_slot: wp.array(dtype=wp.int32),
        xp_slot: wp.array(dtype=wp.int32),
    ):
        row = wp.tid()
        i = row // Nxi
        j = row - i*Nxi
        pm_slot[row] = -1
        pp_slot[row] = -1
        xm_slot[row] = -1
        xp_slot[row] = -1
        diag_slot[row] = -1
        pos = row_ptr[row]
        if i > 0:
            col_ind[pos] = row-Nxi
            pm_slot[row] = pos
            pos += 1
        if j > 0:
            col_ind[pos] = row-1
            xm_slot[row] = pos
            pos += 1
        col_ind[pos] = row
        diag_slot[row] = pos
        pos += 1
        if j + 1 < Nxi:
            col_ind[pos] = row+1
            xp_slot[row] = pos
            pos += 1
        if i + 1 < Np:
            col_ind[pos] = row+Nxi
            pp_slot[row] = pos


    @wp.kernel
    def validate_csr_local_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        col_ind: wp.array(dtype=wp.int32),
        diag_slot: wp.array(dtype=wp.int32),
        pm_slot: wp.array(dtype=wp.int32),
        pp_slot: wp.array(dtype=wp.int32),
        xm_slot: wp.array(dtype=wp.int32),
        xp_slot: wp.array(dtype=wp.int32),
        n: int,
        Np: int,
        Nxi: int,
        flag: wp.array(dtype=wp.int32),
    ):
        row = wp.tid()
        i = row // Nxi
        j = row - i*Nxi
        lo = row_ptr[row]
        hi = row_ptr[row+1]

        if lo >= hi:
            wp.atomic_max(flag, 0, 1)
        else:
            for s in range(lo, hi):
                c = col_ind[s]
                if c < 0 or c >= n:
                    wp.atomic_max(flag, 0, 1)
                if s > lo and c <= col_ind[s-1]:
                    wp.atomic_max(flag, 0, 1)

        ds = diag_slot[row]
        if ds < lo or ds >= hi:
            wp.atomic_max(flag, 0, 1)
        elif col_ind[ds] != row:
            wp.atomic_max(flag, 0, 1)

        if i > 0:
            ss = pm_slot[row]
            if ss < lo or ss >= hi:
                wp.atomic_max(flag, 0, 1)
            elif col_ind[ss] != row-Nxi:
                wp.atomic_max(flag, 0, 1)
        elif pm_slot[row] != -1:
            wp.atomic_max(flag, 0, 1)

        if i + 1 < Np:
            ss = pp_slot[row]
            if ss < lo or ss >= hi:
                wp.atomic_max(flag, 0, 1)
            elif col_ind[ss] != row+Nxi:
                wp.atomic_max(flag, 0, 1)
        elif pp_slot[row] != -1:
            wp.atomic_max(flag, 0, 1)

        if j > 0:
            ss = xm_slot[row]
            if ss < lo or ss >= hi:
                wp.atomic_max(flag, 0, 1)
            elif col_ind[ss] != row-1:
                wp.atomic_max(flag, 0, 1)
        elif xm_slot[row] != -1:
            wp.atomic_max(flag, 0, 1)

        if j + 1 < Nxi:
            ss = xp_slot[row]
            if ss < lo or ss >= hi:
                wp.atomic_max(flag, 0, 1)
            elif col_ind[ss] != row+1:
                wp.atomic_max(flag, 0, 1)
        elif xp_slot[row] != -1:
            wp.atomic_max(flag, 0, 1)


    @wp.kernel
    def assemble_minus_forward_generator_kernel(
        p_faces: wp.array(dtype=wp.float64),
        p_centers: wp.array(dtype=wp.float64),
        xi_faces: wp.array(dtype=wp.float64),
        xi_centers: wp.array(dtype=wp.float64),
        radial_volume: wp.array(dtype=wp.float64),
        cell_volume: wp.array(dtype=wp.float64),
        xi_cell_widths: wp.array(dtype=wp.float64),
        p_face_spacing: wp.array(dtype=wp.float64),
        xi_face_spacing: wp.array(dtype=wp.float64),
        cf_faces: wp.array(dtype=wp.float64),
        ca_faces: wp.array(dtype=wp.float64),
        nud_centers: wp.array(dtype=wp.float64),
        diag_slot: wp.array(dtype=wp.int32),
        pm_slot: wp.array(dtype=wp.int32),
        pp_slot: wp.array(dtype=wp.int32),
        xm_slot: wp.array(dtype=wp.int32),
        xp_slot: wp.array(dtype=wp.int32),
        Np: int,
        Nxi: int,
        Ebar: wp.float64,
        alpha: wp.float64,
        values: wp.array(dtype=wp.float64),
    ):
        # Assemble -L with exactly the forward FV flux formulas.
        # pmin is absorbing failure; positive pmax outflow is successful escape.
        row = wp.tid()
        i = row // Nxi
        j = row - i*Nxi
        xi = xi_centers[j]
        dxi_cell = xi_cell_widths[j]
        one_minus = wp.float64(1.0) - xi*xi
        self_L = wp.float64(0.0)

        # Left radial face: +F_left.  At pmin the exterior state is zero;
        # inward drift and radial diffusion are therefore failure losses.
        if i > 0:
            pf = p_faces[i]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[i]
            D = ca_faces[i]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi_cell
            cL = fac*chang_cooper_left_coefficient(A, D, p_face_spacing[i])/cell_volume[row-Nxi]
            cR = fac*chang_cooper_right_coefficient(A, D, p_face_spacing[i])/cell_volume[row]
            values[pm_slot[row]] = -cL
            self_L += cR
        else:
            pf = p_faces[0]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[0]
            D = ca_faces[0]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi_cell
            boundary_coeff = chang_cooper_right_coefficient(
                A, D, p_face_spacing[0]
            )
            self_L += fac*boundary_coeff/cell_volume[row]

        # Right radial face: -F_right.  At pmax only positive outward drift is
        # allowed; its loss coefficient is also the successful-escape RHS.
        if i + 1 < Np:
            pf = p_faces[i+1]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[i+1]
            D = ca_faces[i+1]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi_cell
            cL = fac*chang_cooper_left_coefficient(A, D, p_face_spacing[i+1])/cell_volume[row]
            cR = fac*chang_cooper_right_coefficient(A, D, p_face_spacing[i+1])/cell_volume[row+Nxi]
            self_L -= cL
            values[pp_slot[row]] = cR
        else:
            pf = p_faces[Np]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[Np]
            if A > wp.float64(0.0):
                c = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi_cell*A/cell_volume[row]
                self_L -= c

        p = p_centers[i]
        gamc = wp.sqrt(wp.float64(1.0) + p*p)
        rv = radial_volume[i]

        # Lower pitch face: +F_lower.
        if j > 0:
            xf = xi_faces[j]
            om = wp.float64(1.0) - xf*xf
            A = om*(-Ebar/p + alpha*xf/gamc)
            D = wp.float64(0.5)*nud_centers[i]*om
            cL = rv*chang_cooper_left_coefficient(A, D, xi_face_spacing[j])/cell_volume[row-1]
            cR = rv*chang_cooper_right_coefficient(A, D, xi_face_spacing[j])/cell_volume[row]
            values[xm_slot[row]] = -cL
            self_L += cR

        # Upper pitch face: -F_upper.
        if j + 1 < Nxi:
            xf = xi_faces[j+1]
            om = wp.float64(1.0) - xf*xf
            A = om*(-Ebar/p + alpha*xf/gamc)
            D = wp.float64(0.5)*nud_centers[i]*om
            cL = rv*chang_cooper_left_coefficient(A, D, xi_face_spacing[j+1])/cell_volume[row]
            cR = rv*chang_cooper_right_coefficient(A, D, xi_face_spacing[j+1])/cell_volume[row+1]
            self_L -= cL
            values[xp_slot[row]] = cR

        values[diag_slot[row]] = -self_L


    @wp.kernel
    def transpose_local_values_kernel(
        forward_values: wp.array(dtype=wp.float64),
        diag_slot: wp.array(dtype=wp.int32),
        pm_slot: wp.array(dtype=wp.int32),
        pp_slot: wp.array(dtype=wp.int32),
        xm_slot: wp.array(dtype=wp.int32),
        xp_slot: wp.array(dtype=wp.int32),
        Np: int,
        Nxi: int,
        adjoint_values: wp.array(dtype=wp.float64),
    ):
        # Exact transpose on the structurally symmetric five-point graph.
        row = wp.tid()
        i = row // Nxi
        j = row - i*Nxi

        adjoint_values[diag_slot[row]] = forward_values[diag_slot[row]]
        if i > 0:
            c = row - Nxi
            adjoint_values[pp_slot[c]] = forward_values[pm_slot[row]]
        if i + 1 < Np:
            c = row + Nxi
            adjoint_values[pm_slot[c]] = forward_values[pp_slot[row]]
        if j > 0:
            c = row - 1
            adjoint_values[xp_slot[c]] = forward_values[xm_slot[row]]
        if j + 1 < Nxi:
            c = row + 1
            adjoint_values[xm_slot[c]] = forward_values[xp_slot[row]]


    @wp.kernel
    def build_success_rhs_kernel(
        p_faces: wp.array(dtype=wp.float64),
        xi_centers: wp.array(dtype=wp.float64),
        xi_cell_widths: wp.array(dtype=wp.float64),
        cell_volume: wp.array(dtype=wp.float64),
        cf_faces: wp.array(dtype=wp.float64),
        Np: int,
        Nxi: int,
        Ebar: wp.float64,
        alpha: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        rhs[row] = wp.float64(0.0)
        i = row // Nxi
        if i == Np - 1:
            j = row - i*Nxi
            xi = xi_centers[j]
            pf = p_faces[Np]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*(wp.float64(1.0)-xi*xi) - cf_faces[Np]
            if A > wp.float64(0.0):
                rhs[row] = (
                    wp.float64(2.0)*wp.float64(3.14159265358979323846)
                    *pf*pf*xi_cell_widths[j]*A/cell_volume[row]
                )




    @wp.kernel
    def build_failure_rhs_kernel(
        p_faces: wp.array(dtype=wp.float64),
        xi_centers: wp.array(dtype=wp.float64),
        p_face_spacing: wp.array(dtype=wp.float64),
        xi_cell_widths: wp.array(dtype=wp.float64),
        cell_volume: wp.array(dtype=wp.float64),
        cf_faces: wp.array(dtype=wp.float64),
        ca_faces: wp.array(dtype=wp.float64),
        Np: int,
        Nxi: int,
        Ebar: wp.float64,
        alpha: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        # Total loss through the shifted absorbing inner radial face.  The
        # same Chang--Cooper boundary flux as the matrix assembly is used;
        # the exterior absorbing distribution is zero.
        row = wp.tid()
        rhs[row] = wp.float64(0.0)
        i = row // Nxi
        if i == 0:
            j = row
            xi = xi_centers[j]
            pf = p_faces[0]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*(wp.float64(1.0)-xi*xi) - cf_faces[0]
            boundary_coeff = chang_cooper_right_coefficient(
                A, ca_faces[0], p_face_spacing[0]
            )
            loss_speed = wp.float64(0.0)
            if boundary_coeff < wp.float64(0.0):
                loss_speed = -boundary_coeff
            rhs[row] = (
                wp.float64(2.0)*wp.float64(3.14159265358979323846)
                *pf*pf*xi_cell_widths[j]*loss_speed/cell_volume[row]
            )


    @wp.kernel
    def copy_float64_kernel(
        src: wp.array(dtype=wp.float64),
        dst: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        dst[row] = src[row]


    @wp.kernel
    def build_backward_euler_values_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        diag_slot: wp.array(dtype=wp.int32),
        forward_minus_L_values: wp.array(dtype=wp.float64),
        dt: wp.float64,
        be_values: wp.array(dtype=wp.float64),
    ):
        # B = I + dt*(-L).  Solving B N_{n+1}=N_n is backward Euler for
        # dN/dtau=L N.
        row = wp.tid()
        for k in range(row_ptr[row], row_ptr[row+1]):
            be_values[k] = dt*forward_minus_L_values[k]
        be_values[diag_slot[row]] += wp.float64(1.0)


    @wp.kernel
    def zero_scalar_kernel(x: wp.array(dtype=wp.float64)):
        if wp.tid() == 0:
            x[0] = wp.float64(0.0)


    @wp.kernel
    def accumulate_response_kernel(
        state: wp.array(dtype=wp.float64),
        response_rate: wp.array(dtype=wp.float64),
        dt: wp.float64,
        accum: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        wp.atomic_add(accum, 0, dt*response_rate[row]*state[row])


    @wp.kernel
    def sum_state_kernel(
        state: wp.array(dtype=wp.float64),
        total: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        wp.atomic_add(total, 0, state[row])


    @wp.kernel
    def csr_residual_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        col_ind: wp.array(dtype=wp.int32),
        values: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        b: wp.array(dtype=wp.float64),
        residual: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        acc = wp.float64(0.0)
        for k in range(row_ptr[row], row_ptr[row+1]):
            acc += values[k]*x[col_ind[k]]
        residual[row] = acc - b[row]


    @wp.kernel
    def csr_matvec_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        col_ind: wp.array(dtype=wp.int32),
        values: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        acc = wp.float64(0.0)
        for k in range(row_ptr[row], row_ptr[row+1]):
            acc += values[k]*x[col_ind[k]]
        y[row] = acc


def build_local_csr_topology_gpu(grid: Grid, device: str) -> tuple[CSRTopology, dict[str, float]]:
    if wp is None:
        raise RuntimeError("GPU topology construction requires warp-lang")

    t0 = time.perf_counter()
    row_ptr_host, nnz = _build_local_csr_row_ptr_host(grid)
    rowptr_host_s = time.perf_counter() - t0

    dev_obj = wp.get_device(device)
    stream = wp.get_stream(dev_obj)
    row_ptr = wp.array(row_ptr_host, dtype=wp.int32, device=device)
    col_ind = wp.zeros(nnz, dtype=wp.int32, device=device)
    diag = wp.zeros(grid.size, dtype=wp.int32, device=device)
    pm = wp.zeros(grid.size, dtype=wp.int32, device=device)
    pp = wp.zeros(grid.size, dtype=wp.int32, device=device)
    xm = wp.zeros(grid.size, dtype=wp.int32, device=device)
    xp = wp.zeros(grid.size, dtype=wp.int32, device=device)

    t1 = time.perf_counter()
    wp.launch(
        fill_local_csr_topology_kernel,
        dim=grid.size,
        inputs=[row_ptr, grid.Np, grid.Nxi],
        outputs=[col_ind, diag, pm, pp, xm, xp],
        device=device,
    )
    wp.synchronize_stream(stream)
    fill_gpu_s = time.perf_counter() - t1

    flag = wp.zeros(1, dtype=wp.int32, device=device)
    t2 = time.perf_counter()
    wp.launch(
        validate_csr_local_kernel,
        dim=grid.size,
        inputs=[row_ptr, col_ind, diag, pm, pp, xm, xp,
                grid.size, grid.Np, grid.Nxi, flag],
        device=device,
    )
    wp.synchronize_stream(stream)
    validate_gpu_s = time.perf_counter() - t2
    if int(np.asarray(flag.numpy(), dtype=np.int32)[0]) != 0:
        raise RuntimeError("GPU local CSR topology validation failed")

    topo = CSRTopology(grid.size, nnz, row_ptr, col_ind, diag, pm, pp, xm, xp)
    return topo, {
        "rowptr_host_s": rowptr_host_s,
        "fill_gpu_s": fill_gpu_s,
        "validate_gpu_s": validate_gpu_s,
        "total_s": time.perf_counter() - t0,
    }


# =============================================================================
# Warp-owned cuDSS sparse-direct wrapper
# =============================================================================

def find_cudss_threading_layer() -> str | None:
    candidates: list[Path] = []
    for root in site.getsitepackages():
        candidates.extend(Path(root).glob("nvidia/cu*/lib/libcudss_mtlayer_gomp.so*"))
    return str(sorted(candidates)[0]) if candidates else None


def load_gpu_runtime(device: str):
    if wp is None:
        raise RuntimeError("GPU backend requires warp-lang")
    try:
        import nvmath
        from nvmath.bindings import cudss
    except Exception as exc:
        raise RuntimeError("GPU backend requires nvmath-python with cuDSS bindings") from exc
    wp.init()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise RuntimeError(f"GPU backend requires CUDA, got {dev}")
    try:
        wv = importlib_metadata.version("warp-lang")
    except Exception:
        wv = getattr(wp, "__version__", "unknown")
    try:
        nv = importlib_metadata.version("nvmath-python")
    except Exception:
        nv = "unknown"
    try:
        maj = cudss.get_property(nvmath.LibraryPropertyType.MAJOR_VERSION)
        minr = cudss.get_property(nvmath.LibraryPropertyType.MINOR_VERSION)
        pat = cudss.get_property(nvmath.LibraryPropertyType.PATCH_LEVEL)
        cv = f"{maj}.{minr}.{pat}"
    except Exception:
        cv = "unknown"
    return nvmath, cudss, dev, {"warp":str(wv), "nvmath":str(nv), "cudss":str(cv), "device":str(dev)}


class CudssDirectSystem:
    """Persistent cuDSS system whose matrix/vector buffers are owned by Warp."""

    def __init__(self, *, nvmath, cudss, row_ptr, col_ind, values, rhs, solution,
                 n: int, nnz: int, threading_layer: str | None):
        self.nvmath = nvmath
        self.cudss = cudss
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.values = values
        self.rhs = rhs
        self.solution = solution
        self.n = int(n)
        self.nnz = int(nnz)
        self.closed = False
        self.analyzed = False

        if int(row_ptr.size) != self.n + 1 or int(col_ind.size) != self.nnz or int(values.size) != self.nnz:
            raise ValueError("cuDSS CSR buffer sizes are inconsistent")
        if int(rhs.size) != self.n or int(solution.size) != self.n:
            raise ValueError("cuDSS dense vector sizes are inconsistent")
        if not (row_ptr.device == col_ind.device == values.device == rhs.device == solution.device):
            raise ValueError("all cuDSS/Warp buffers must reside on the same CUDA device")

        self.handle = None
        self.config = None
        self.data = None
        self.a_desc = None
        self.b_desc = None
        self.x_desc = None
        self.stream = None
        self.ev0 = None
        self.ev1 = None

        try:
            self.handle = cudss.create()
            if threading_layer:
                cudss.set_threading_layer(self.handle, threading_layer)
            self.stream = wp.get_stream(values.device)
            self.stream_handle = int(self.stream.cuda_stream)
            cudss.set_stream(self.handle, self.stream_handle)
            self.ev0 = wp.Event(device=values.device, enable_timing=True)
            self.ev1 = wp.Event(device=values.device, enable_timing=True)
            self.config = cudss.config_create()
            self.data = cudss.data_create(self.handle)
            self.a_desc = cudss.matrix_create_csr(
                self.n, self.n, self.nnz,
                row_ptr.ptr, 0, col_ind.ptr, values.ptr,
                nvmath.CudaDataType.CUDA_R_32I,
                nvmath.CudaDataType.CUDA_R_32I,
                nvmath.CudaDataType.CUDA_R_64F,
                cudss.MatrixType.GENERAL,
                cudss.MatrixViewType.FULL,
                cudss.IndexBase.ZERO,
            )
            self.b_desc = cudss.matrix_create_dn(
                self.n, 1, self.n, rhs.ptr,
                nvmath.CudaDataType.CUDA_R_64F, cudss.Layout.COL_MAJOR,
            )
            self.x_desc = cudss.matrix_create_dn(
                self.n, 1, self.n, solution.ptr,
                nvmath.CudaDataType.CUDA_R_64F, cudss.Layout.COL_MAJOR,
            )
        except Exception:
            self._destroy_resources(synchronize=False)
            raise

    def _execute(self, phase) -> float:
        if self.closed:
            raise RuntimeError("cuDSS system is closed")
        self.stream.record_event(self.ev0)
        self.cudss.execute(self.handle, phase, self.config, self.data,
                           self.a_desc, self.x_desc, self.b_desc)
        self.stream.record_event(self.ev1)
        ms = wp.get_event_elapsed_time(self.ev0, self.ev1, synchronize=True)
        dp = getattr(self.cudss, "DataParam", None)
        if dp is not None and hasattr(dp, "INFO"):
            try:
                info = ctypes.c_int(0)
                written = ctypes.c_size_t(0)
                self.cudss.data_get(
                    self.handle, self.data, dp.INFO,
                    ctypes.addressof(info), ctypes.sizeof(info),
                    ctypes.addressof(written),
                )
                if info.value != 0:
                    raise RuntimeError(f"cuDSS asynchronous error INFO={info.value} in phase {phase}")
            except RuntimeError:
                raise
            except Exception:
                pass
        return float(ms)*1e-3

    def analyze(self) -> float:
        t = self._execute(self.cudss.Phase.ANALYSIS)
        self.analyzed = True
        return t

    def factorize(self) -> float:
        if not self.analyzed:
            raise RuntimeError("cuDSS analysis must precede factorization")
        return self._execute(self.cudss.Phase.FACTORIZATION)

    def solve(self) -> float:
        if not self.analyzed:
            raise RuntimeError("cuDSS analysis must precede solve")
        return self._execute(self.cudss.Phase.SOLVE)

    def solve_async(self) -> None:
        """Enqueue a solve on the configured CUDA stream without host synchronization."""
        if self.closed:
            raise RuntimeError("cuDSS system is closed")
        if not self.analyzed:
            raise RuntimeError("cuDSS analysis must precede solve")
        self.cudss.execute(
            self.handle, self.cudss.Phase.SOLVE, self.config, self.data,
            self.a_desc, self.x_desc, self.b_desc,
        )

    def _destroy_resources(self, *, synchronize: bool) -> None:
        if self.closed:
            return
        if synchronize and self.stream is not None:
            try:
                wp.synchronize_stream(self.stream)
            except Exception:
                pass
        for name in ("x_desc", "b_desc", "a_desc"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    self.cudss.matrix_destroy(obj)
                except Exception:
                    pass
                setattr(self, name, None)
        if self.data is not None and self.handle is not None:
            try:
                self.cudss.data_destroy(self.handle, self.data)
            except Exception:
                pass
            self.data = None
        if self.config is not None:
            try:
                self.cudss.config_destroy(self.config)
            except Exception:
                pass
            self.config = None
        if self.handle is not None:
            try:
                self.cudss.destroy(self.handle)
            except Exception:
                pass
            self.handle = None
        self.closed = True

    def close(self) -> None:
        self._destroy_resources(synchronize=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def gpu_relative_residual(row_ptr, col_ind, values, x, b, residual, n: int, device: str) -> float:
    wp.launch(csr_residual_kernel, dim=n,
              inputs=[row_ptr, col_ind, values, x, b], outputs=[residual], device=device)
    rr = np.asarray(residual.numpy(), dtype=np.float64)
    bb = np.asarray(b.numpy(), dtype=np.float64)
    nr = np.linalg.norm(rr)
    nb = np.linalg.norm(bb)
    return float(nr/nb if nb else nr)


# =============================================================================
# Time-dependent finite-horizon RPF solve
# =============================================================================

def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    den = max(float(np.linalg.norm(aa)), float(np.linalg.norm(bb)), 1.0e-300)
    return float(np.linalg.norm(aa-bb)/den)


def _host_matrix_qualification(topo: CSRTopology, forward_values, adjoint_values):
    """Independently transpose the actual GPU CSR matrix with SciPy."""
    row_ptr = np.asarray(topo.row_ptr.numpy(), dtype=np.int32)
    col_ind = np.asarray(topo.col_ind.numpy(), dtype=np.int32)
    a_vals = np.asarray(forward_values.numpy(), dtype=np.float64)
    at_gpu_vals = np.asarray(adjoint_values.numpy(), dtype=np.float64)

    A = sp.csr_matrix((a_vals, col_ind, row_ptr), shape=(topo.n, topo.n))
    AT = A.transpose().tocsr()
    AT.sort_indices()
    structure_ok = bool(
        np.array_equal(AT.indptr.astype(np.int32, copy=False), row_ptr)
        and np.array_equal(AT.indices.astype(np.int32, copy=False), col_ind)
    )
    if not structure_ok:
        raise RuntimeError("independent host transpose changed the expected symmetric CSR graph")
    diff = at_gpu_vals - AT.data
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rel_l2 = _relative_l2(at_gpu_vals, AT.data)
    if max_abs != 0.0:
        raise RuntimeError(
            "GPU exact-transpose values disagree with independent SciPy transpose: "
            f"max_abs={max_abs:.3e}, rel_l2={rel_l2:.3e}"
        )

    diag = A.diagonal()
    coo = A.tocoo()
    off = coo.data[coo.row != coo.col]
    scale = max(float(np.max(np.abs(a_vals))) if a_vals.size else 0.0, 1.0)
    sign_tol = 128.0*np.finfo(np.float64).eps*scale
    diag_min = float(np.min(diag)) if diag.size else 0.0
    offdiag_max = float(np.max(off)) if off.size else 0.0
    if diag_min < -sign_tol or offdiag_max > sign_tol:
        raise RuntimeError(
            "-L0 violates the expected generator sign structure: "
            f"diag_min={diag_min:.3e}, offdiag_max={offdiag_max:.3e}, tol={sign_tol:.3e}"
        )
    return A, AT, {
        "transpose_structure_ok": structure_ok,
        "transpose_max_abs": max_abs,
        "transpose_rel_l2": rel_l2,
        "generator_diag_min": diag_min,
        "generator_offdiag_max": offdiag_max,
        "generator_sign_tol": sign_tol,
    }


def _host_success_rhs(grid: Grid, coll: CollisionData, phys: DerivedPhysics) -> tuple[np.ndarray, np.ndarray]:
    expected = np.zeros(grid.size, dtype=np.float64)
    pf = grid.p_faces[-1]
    gam = math.sqrt(1.0 + pf*pf)
    A = (
        -phys.Ebar*grid.xi_centers
        - phys.alpha*gam*pf*(1.0-grid.xi_centers**2)
        - coll.cf_faces[-1]
    )
    rates = (
        2.0*math.pi*pf*pf*grid.xi_cell_widths*np.maximum(A, 0.0)
        / grid.cell_volume[-grid.Nxi:]
    )
    expected[-grid.Nxi:] = rates
    return expected, A


def _chang_cooper_right_coefficient_host(A: float, D: float, dq: float) -> float:
    if D > 0.0:
        w = A*dq/D
        if abs(w) < 1.0e-4:
            w2 = w*w
            delta = 0.5 - w/12.0 + w*w2/720.0 - w*w2*w2/30240.0
        elif w > 50.0:
            delta = 1.0/w
        elif w < -50.0:
            delta = 1.0 + 1.0/w
        else:
            delta = 1.0/w - 1.0/math.expm1(w)
        return A*delta - D/dq
    return A if A < 0.0 else 0.0


def _host_failure_rhs(grid: Grid, coll: CollisionData, phys: DerivedPhysics) -> np.ndarray:
    expected = np.zeros(grid.size, dtype=np.float64)
    pf = grid.p_faces[0]
    gam = math.sqrt(1.0 + pf*pf)
    A = (
        -phys.Ebar*grid.xi_centers
        - phys.alpha*gam*pf*(1.0-grid.xi_centers**2)
        - coll.cf_faces[0]
    )
    loss_speed = np.asarray([
        max(-_chang_cooper_right_coefficient_host(float(a), float(coll.ca_faces[0]),
                                                   float(grid.p_face_spacing[0])), 0.0)
        for a in A
    ], dtype=np.float64)
    expected[:grid.Nxi] = (
        2.0*math.pi*pf*pf*grid.xi_cell_widths*loss_speed
        / grid.cell_volume[:grid.Nxi]
    )
    return expected


def _terminal_cell_width(cfg: SolverConfig, grid: Grid) -> float:
    cell = int(np.clip(np.searchsorted(grid.p_faces, cfg.terminal_p)-1, 0, grid.Np-1))
    return float(grid.p_cell_widths[cell])


def build_terminal_response(cfg: SolverConfig, grid: Grid) -> np.ndarray:
    """Build the configured terminal response in the exact p^2 FV measure."""
    lo = grid.p_faces[:-1]
    hi = grid.p_faces[1:]
    den = hi**3 - lo**3

    if cfg.terminal_condition == "heaviside":
        frac = np.zeros(grid.Np, dtype=np.float64)
        frac[cfg.terminal_p <= lo] = 1.0
        cut = (lo < cfg.terminal_p) & (cfg.terminal_p < hi)
        frac[cut] = (hi[cut]**3 - cfg.terminal_p**3)/den[cut]
    elif cfg.terminal_condition == "smoothed_heaviside":
        delta_p = cfg.terminal_smoothing_cells * _terminal_cell_width(cfg, grid)
        xq, wq = np.polynomial.legendre.leggauss(8)
        mid = 0.5*(lo + hi)
        half = 0.5*(hi - lo)
        pquad = mid[:, None] + half[:, None]*xq[None, :]
        gquad = 0.5*(1.0 + np.tanh((pquad - cfg.terminal_p)/delta_p))
        integral = half*np.sum(wq[None, :]*(pquad*pquad)*gquad, axis=1)
        frac = 3.0*integral/den
    else:
        raise RuntimeError(f"unsupported terminal condition {cfg.terminal_condition!r}")

    frac = np.clip(frac, 0.0, 1.0)
    return np.ascontiguousarray(np.repeat(frac, grid.Nxi), dtype=np.float64)


if wp is not None:

    @wp.kernel
    def build_shifted_adjoint_matrix_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        diag_slot: wp.array(dtype=wp.int32),
        adjoint_minus_Lt_values: wp.array(dtype=wp.float64),
        h: wp.float64,
        alpha0: wp.float64,
        values: wp.array(dtype=wp.float64),
    ):
        # A=-L0^T.  Build alpha0*I + h*A, which is the implicit matrix for
        # dP/ds = -A P + r_RE.
        row = wp.tid()
        for k in range(row_ptr[row], row_ptr[row+1]):
            values[k] = h*adjoint_minus_Lt_values[k]
        values[diag_slot[row]] += alpha0


    @wp.kernel
    def build_tr_stage1_rhs_kernel(
        p_n: wp.array(dtype=wp.float64),
        A_p_n: wp.array(dtype=wp.float64),
        success_rate: wp.array(dtype=wp.float64),
        ah: wp.float64,
        gamma_h: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        # TR substep over gamma*h for dP/ds=-A P+r:
        # (I + ah A) P_gamma = (I - ah A) P_n + gamma*h*r,
        # where ah=(gamma/2)h.
        i = wp.tid()
        rhs[i] = p_n[i] - ah*A_p_n[i] + gamma_h*success_rate[i]


    @wp.kernel
    def build_tr_stage2_rhs_kernel(
        p_gamma: wp.array(dtype=wp.float64),
        p_n: wp.array(dtype=wp.float64),
        success_rate: wp.array(dtype=wp.float64),
        c_gamma: wp.float64,
        c_n: wp.float64,
        beta_h: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        # Variable-step BDF2 substep from t_n+gamma*h to t_n+h:
        # (I + beta*h A) P_{n+1}
        #   = c_gamma P_gamma + c_n P_n + beta*h*r.
        # For gamma=2-sqrt(2), beta=gamma/2, so this uses the same matrix
        # as the trapezoidal substep.
        i = wp.tid()
        rhs[i] = c_gamma*p_gamma[i] + c_n*p_n[i] + beta_h*success_rate[i]


    @wp.kernel
    def monitor_probability_kernel(
        p: wp.array(dtype=wp.float64),
        tol: wp.float64,
        flag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        v = p[i]
        # 1 = probability bound violation; 2 = non-finite / overflow-like value.
        if v != v or wp.abs(v) > wp.float64(1.0e300):
            wp.atomic_max(flag, 0, 2)
        elif v < -tol or v > wp.float64(1.0) + tol:
            wp.atomic_max(flag, 0, 1)


    @wp.kernel
    def reset_float64_scalar_kernel(x: wp.array(dtype=wp.float64)):
        if wp.tid() == 0:
            x[0] = wp.float64(0.0)


    @wp.kernel
    def accumulate_sumsq_kernel(
        x: wp.array(dtype=wp.float64),
        sumsq: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        v = x[i]
        wp.atomic_add(sumsq, 0, v*v)


    @wp.kernel
    def compare_to_steady_kernel(
        current: wp.array(dtype=wp.float64),
        steady: wp.array(dtype=wp.float64),
        diff_sumsq: wp.array(dtype=wp.float64),
        diff_max: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        d = current[i] - steady[i]
        a = wp.abs(d)
        wp.atomic_add(diff_sumsq, 0, d*d)
        wp.atomic_max(diff_max, 0, a)



def _probability_bounds_ok(p: np.ndarray, tol: float) -> bool:
    return bool(np.min(p) >= -tol and np.max(p) <= 1.0 + tol)


def solve_adjoint_rpf(
    cfg: SolverConfig, phys: DerivedPhysics, grid: Grid, coll: CollisionData,
    topo: CSRTopology, gpu_context,
) -> dict:
    """Unified steady RPF solve with optional fixed-step TR-BDF2 transient.

    The direct steady RPF A P_ss = r_RE (A=-L0^T) is always solved first and retained
    on the GPU.  In solve_mode="steady" the routine returns immediately after that
    qualified solve.  In solve_mode="steady_and_time" the transient then advances
    with standard TR-BDF2 using
    gamma=2-sqrt(2).  Both implicit substeps use the same matrix

        M = I + (gamma/2) h A,

    which is factorized once and reused for every stage of every timestep.  Each
    full step therefore requires two cuDSS solves plus one GPU CSR matvec for the
    trapezoidal right-hand side.  There is no BE startup and no multistep history.
    """
    if wp is None:
        raise RuntimeError("this script requires warp-lang and a CUDA device")

    nvmath, cudss, dev_obj, runtime = gpu_context
    dev = cfg.device
    h = float(cfg.dtau)
    stream = wp.get_stream(dev_obj)

    gamma_tr = 2.0 - math.sqrt(2.0)
    beta_tr = (1.0 - gamma_tr)/(2.0 - gamma_tr)
    a_tr = 0.5*gamma_tr
    if not math.isclose(beta_tr, a_tr, rel_tol=0.0, abs_tol=8.0*np.finfo(np.float64).eps):
        raise RuntimeError("TR-BDF2 gamma does not produce the shared implicit matrix")
    c_gamma = 1.0/(gamma_tr*(2.0-gamma_tr))
    c_n = -((1.0-gamma_tr)**2)/(gamma_tr*(2.0-gamma_tr))
    ah = a_tr*h
    gamma_h = gamma_tr*h
    beta_h = beta_tr*h

    def wa(x, dtype):
        return wp.array(np.ascontiguousarray(x), dtype=dtype, device=dev)

    p_faces = wa(grid.p_faces, wp.float64)
    p_centers = wa(grid.p_centers, wp.float64)
    xi_faces = wa(grid.xi_faces, wp.float64)
    xi_centers = wa(grid.xi_centers, wp.float64)
    radial_volume = wa(grid.radial_volume, wp.float64)
    cell_volume = wa(grid.cell_volume, wp.float64)
    xi_cell_widths = wa(grid.xi_cell_widths, wp.float64)
    p_face_spacing = wa(grid.p_face_spacing, wp.float64)
    xi_face_spacing = wa(grid.xi_face_spacing, wp.float64)
    cf_faces = wa(coll.cf_faces, wp.float64)
    ca_faces = wa(coll.ca_faces, wp.float64)
    nud_centers = wa(coll.nud_centers, wp.float64)

    forward_values = wp.zeros(topo.nnz, dtype=wp.float64, device=dev)
    adjoint_values = wp.zeros(topo.nnz, dtype=wp.float64, device=dev)
    success_rhs = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    failure_rhs = wp.zeros(topo.n, dtype=wp.float64, device=dev)

    t0 = time.perf_counter()
    wp.launch(
        assemble_minus_forward_generator_kernel, dim=grid.size,
        inputs=[
            p_faces, p_centers, xi_faces, xi_centers,
            radial_volume, cell_volume, xi_cell_widths, p_face_spacing, xi_face_spacing,
            cf_faces, ca_faces, nud_centers,
            topo.diag_slot, topo.pm_slot, topo.pp_slot, topo.xm_slot, topo.xp_slot,
            grid.Np, grid.Nxi,
            wp.float64(phys.Ebar), wp.float64(phys.alpha),
        ],
        outputs=[forward_values], device=dev,
    )
    wp.launch(
        transpose_local_values_kernel, dim=grid.size,
        inputs=[
            forward_values,
            topo.diag_slot, topo.pm_slot, topo.pp_slot, topo.xm_slot, topo.xp_slot,
            grid.Np, grid.Nxi,
        ],
        outputs=[adjoint_values], device=dev,
    )
    wp.launch(
        build_success_rhs_kernel, dim=grid.size,
        inputs=[
            p_faces, xi_centers, xi_cell_widths, cell_volume, cf_faces,
            grid.Np, grid.Nxi,
            wp.float64(phys.Ebar), wp.float64(phys.alpha),
        ],
        outputs=[success_rhs], device=dev,
    )
    wp.launch(
        build_failure_rhs_kernel, dim=grid.size,
        inputs=[
            p_faces, xi_centers, p_face_spacing, xi_cell_widths, cell_volume,
            cf_faces, ca_faces, grid.Np, grid.Nxi,
            wp.float64(phys.Ebar), wp.float64(phys.alpha),
        ],
        outputs=[failure_rhs], device=dev,
    )
    wp.synchronize_stream(stream)
    assembly_s = time.perf_counter()-t0

    # One-time exact transpose / escape-vector qualification, outside the loop.
    _, A_adjoint_host, transpose_diag = _host_matrix_qualification(
        topo, forward_values, adjoint_values
    )
    success_host = np.asarray(success_rhs.numpy(), dtype=np.float64)
    failure_host = np.asarray(failure_rhs.numpy(), dtype=np.float64)
    expected_success, outer_U = _host_success_rhs(grid, coll, phys)
    expected_failure = _host_failure_rhs(grid, coll, phys)
    success_rel = _relative_l2(success_host, expected_success)
    failure_rel = _relative_l2(failure_host, expected_failure)
    if success_rel > RPF_EXACT_REL_TOL or failure_rel > RPF_EXACT_REL_TOL:
        raise RuntimeError(
            "escape-vector qualification failed: "
            f"success_rel={success_rel:.3e}, failure_rel={failure_rel:.3e}"
        )
    escape_rhs = success_host + failure_host
    total_escape = np.asarray(A_adjoint_host @ np.ones(topo.n), dtype=np.float64)
    escape_err = total_escape - escape_rhs
    total_escape_rel = _relative_l2(total_escape, escape_rhs)

    # On wide p domains the interior row sums of -L0^T can involve cancellation
    # between coefficients many orders of magnitude larger than the physical
    # boundary escape rate.  In that regime a raw relative norm against the
    # escape vector is not a reliable floating-point qualification by itself.
    # Use the standard componentwise backward error as a roundoff diagnostic.
    row_abs_scale = np.asarray(abs(A_adjoint_host) @ np.ones(topo.n), dtype=np.float64)
    comp_den = np.maximum(row_abs_scale + np.abs(escape_rhs), 1.0e-300)
    total_escape_backward = float(np.max(np.abs(escape_err) / comp_den))

    # Accept either the original forward relative check or the scale-aware
    # componentwise backward-error check.  On very wide/uniform-p grids the
    # physical escape vector can be tiny compared with cancelling interior
    # coefficients, so an arbitrary hard ceiling on the forward relative
    # error is not meaningful.  The independent transpose and separately
    # assembled success/failure escape-vector checks remain in force.
    identity_ok = (
        total_escape_rel <= RPF_IDENTITY_REL_TOL
        or total_escape_backward <= RPF_IDENTITY_BACKWARD_TOL
    )
    if not identity_ok:
        raise RuntimeError(
            "total escape identity (-L0^T)1=r_RE+r_F failed: "
            f"rel_l2={total_escape_rel:.3e} "
            f"backward_max={total_escape_backward:.3e}"
        )

    terminal_host = build_terminal_response(cfg, grid)
    if not _probability_bounds_ok(terminal_host, 0.0):
        raise RuntimeError("terminal response is outside [0,1]")

    P_n = wa(terminal_host, wp.float64)
    P_steady = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    A_Pn = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    rhs = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    solution = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    step_values = wp.zeros(topo.nnz, dtype=wp.float64, device=dev)
    residual = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    probability_flag = wp.zeros(1, dtype=wp.int32, device=dev)

    steady_norm2_gpu = wp.zeros(1, dtype=wp.float64, device=dev)
    terminal_norm2_gpu = wp.zeros(1, dtype=wp.float64, device=dev)
    diff_norm2_gpu = wp.zeros(1, dtype=wp.float64, device=dev)
    diff_max_gpu = wp.zeros(1, dtype=wp.float64, device=dev)

    history: list[dict[str, float | int]] = []
    snapshot_steps: list[int] = []
    snapshot_lookback: list[float] = []
    snapshots: list[np.ndarray] = []

    step = 0
    lookback = 0.0
    steady_count = 0
    factor_s_total = 0.0
    factor_count = 0
    solve_count = 0
    steady_factor_s = 0.0
    steady_solve_s = 0.0
    trbdf2_factor_s = 0.0
    trbdf2_loop_wall_s = 0.0
    tr_stage1_first_solve_s = 0.0
    tr_stage2_first_solve_s = 0.0
    tr_stage1_first_residual = math.nan
    tr_stage2_first_residual = math.nan
    def check_probability_flag() -> None:
        flag = int(np.asarray(probability_flag.numpy(), dtype=np.int32)[0])
        if flag >= 2:
            raise FloatingPointError("non-finite value occurred in the time-dependent RPF")
        if flag == 1:
            raise RuntimeError(
                "time-dependent RPF left [0,1] beyond time.probability_tol; "
                "reduce the fixed timestep and rerun"
            )

    def steady_match(current_gpu) -> dict[str, float | bool]:
        wp.launch(reset_float64_scalar_kernel, dim=1, inputs=[diff_norm2_gpu], device=dev)
        wp.launch(reset_float64_scalar_kernel, dim=1, inputs=[diff_max_gpu], device=dev)
        wp.launch(
            compare_to_steady_kernel, dim=topo.n,
            inputs=[current_gpu, P_steady, diff_norm2_gpu, diff_max_gpu], device=dev,
        )
        diff2 = float(np.asarray(diff_norm2_gpu.numpy(), dtype=np.float64)[0])
        max_abs = float(np.asarray(diff_max_gpu.numpy(), dtype=np.float64)[0])
        scaled_l2 = math.sqrt(max(diff2, 0.0)/max(steady_scale_norm2, 1.0e-300))
        ok = (scaled_l2 <= cfg.steady_match_rel_l2_tol and
              max_abs <= cfg.steady_match_max_abs_tol)
        return {"steady_match_rel_l2": scaled_l2,
                "steady_match_max_abs": max_abs,
                "steady_ok": bool(ok)}

    def append_history(step_number: int, metrics: dict[str, float | bool]) -> None:
        history.append({
            "step": int(step_number),
            "lookback_tau": float(step_number*h),
            "dtau": h,
            "steady_match_rel_l2": float(metrics["steady_match_rel_l2"]),
            "steady_match_max_abs": float(metrics["steady_match_max_abs"]),
        })

    threading_layer = find_cudss_threading_layer()
    with CudssDirectSystem(
        nvmath=nvmath, cudss=cudss,
        row_ptr=topo.row_ptr, col_ind=topo.col_ind, values=step_values,
        rhs=rhs, solution=solution, n=topo.n, nnz=topo.nnz,
        threading_layer=threading_layer,
    ) as direct:
        # 1. Direct steady solve A P_ss = r_RE; retain P_ss on GPU.
        wp.launch(copy_float64_kernel, dim=topo.nnz,
                  inputs=[adjoint_values], outputs=[step_values], device=dev)
        wp.launch(copy_float64_kernel, dim=topo.n,
                  inputs=[success_rhs], outputs=[rhs], device=dev)
        analysis_s = direct.analyze()
        steady_factor_s = direct.factorize(); factor_s_total += steady_factor_s; factor_count += 1
        steady_solve_s = direct.solve(); solve_count += 1
        steady_linear_residual = gpu_relative_residual(
            topo.row_ptr, topo.col_ind, step_values, solution, rhs, residual, topo.n, dev
        )
        wp.launch(copy_float64_kernel, dim=topo.n,
                  inputs=[solution], outputs=[P_steady], device=dev)
        wp.launch(reset_float64_scalar_kernel, dim=1, inputs=[steady_norm2_gpu], device=dev)
        wp.launch(accumulate_sumsq_kernel, dim=topo.n,
                  inputs=[P_steady, steady_norm2_gpu], device=dev)
        wp.launch(reset_float64_scalar_kernel, dim=1, inputs=[terminal_norm2_gpu], device=dev)
        wp.launch(accumulate_sumsq_kernel, dim=topo.n,
                  inputs=[P_n, terminal_norm2_gpu], device=dev)
        steady_norm2 = float(np.asarray(steady_norm2_gpu.numpy(), dtype=np.float64)[0])
        terminal_norm2 = float(np.asarray(terminal_norm2_gpu.numpy(), dtype=np.float64)[0])
        if not math.isfinite(steady_norm2) or steady_norm2 < 0.0:
            raise RuntimeError("invalid GPU steady-RPF norm")

        has_success_boundary = bool(np.any(outer_U > 0.0))
        if has_success_boundary and steady_norm2 <= 0.0:
            raise RuntimeError(
                "successful pmax boundary exists but the direct steady RPF has zero norm"
            )
        steady_target_mode = (
            "nonzero-steady-rpf" if has_success_boundary
            else "zero-steady-rpf-no-success-boundary"
        )

        if cfg.solve_mode == "steady":
            steady_host = np.asarray(P_steady.numpy(), dtype=np.float64)
            if not _probability_bounds_ok(steady_host, cfg.probability_tol):
                raise RuntimeError("direct steady RPF lies outside probability bounds")
            return {
                "solve_mode": "steady",
                "P_terminal": None,
                "P_final": steady_host.reshape(grid.Np, grid.Nxi),
                "P_steady_direct": steady_host.reshape(grid.Np, grid.Nxi),
                "success_rate": success_host.reshape(grid.Np, grid.Nxi),
                "failure_rate": failure_host.reshape(grid.Np, grid.Nxi),
                "outer_U_p": outer_U,
                "outer_success_mask": outer_U > 0.0,
                "history": [],
                "snapshot_steps": [],
                "snapshot_lookback": [],
                "snapshots": [],
                "steps": 0,
                "steady_confirm_count": 0,
                "steady_lookback_tau": 0.0,
                "runtime": runtime,
                "assembly_s": assembly_s,
                "analysis_s": analysis_s,
                "factor_s_total": factor_s_total,
                "factor_count": factor_count,
                "solve_count": solve_count,
                "steady_factor_s": steady_factor_s,
                "steady_solve_s": steady_solve_s,
                "trbdf2_factor_s": 0.0,
                "trbdf2_loop_wall_s": 0.0,
                "tr_stage1_first_solve_s": 0.0,
                "tr_stage2_first_solve_s": 0.0,
                "tr_stage1_first_residual": math.nan,
                "tr_stage2_first_residual": math.nan,
                "last_linear_residual": math.nan,
                "steady_linear_residual": steady_linear_residual,
                "final_steady_residual": steady_linear_residual,
                "success_rhs_rel_l2": success_rel,
                "failure_rhs_rel_l2": failure_rel,
                "total_escape_rel_l2": total_escape_rel,
                "total_escape_backward_max": total_escape_backward,
                "trbdf2": {
                    "gamma": gamma_tr,
                    "beta": beta_tr,
                    "shared_matrix_coefficient": a_tr,
                    "c_gamma": c_gamma,
                    "c_n": c_n,
                },
                "steady_target": {
                    "mode": steady_target_mode,
                    "direct_residual": float(steady_linear_residual),
                    "has_success_boundary": bool(has_success_boundary),
                    "success_pitch_cells": int(np.count_nonzero(outer_U > 0.0)),
                    "steady_norm2": float(steady_norm2),
                },
                **transpose_diag,
            }

        if not math.isfinite(terminal_norm2) or terminal_norm2 <= 0.0:
            raise RuntimeError("invalid terminal-condition norm")
        steady_scale_norm2 = steady_norm2 if has_success_boundary else terminal_norm2

        m0 = steady_match(P_n)
        append_history(0, m0)
        steady_count = 1 if m0["steady_ok"] else 0

        # 2. TR-BDF2 transient matrix.  With gamma=2-sqrt(2), both stages use
        #    M = I + (gamma/2) h A, so one factorization serves the full march.
        if steady_count < cfg.steady_confirm_steps:
            wp.launch(
                build_shifted_adjoint_matrix_kernel, dim=topo.n,
                inputs=[topo.row_ptr, topo.diag_slot, adjoint_values,
                        wp.float64(ah), wp.float64(1.0)],
                outputs=[step_values], device=dev,
            )
            trbdf2_factor_s = direct.factorize()
            factor_s_total += trbdf2_factor_s; factor_count += 1
            t_loop = time.perf_counter()

            while steady_count < cfg.steady_confirm_steps:
                if step >= cfg.max_steps:
                    raise RuntimeError(
                        f"time-dependent adjoint did not reach the steady GPU target within {cfg.max_steps} fixed steps"
                    )
                if lookback + h > cfg.max_lookback_tau*(1.0+1e-14):
                    raise RuntimeError(
                        f"time-dependent adjoint did not reach the steady GPU target by lookback tau={cfg.max_lookback_tau:g}"
                    )

                # Stage 1: trapezoidal rule over gamma*h.
                wp.launch(
                    csr_matvec_kernel, dim=topo.n,
                    inputs=[topo.row_ptr, topo.col_ind, adjoint_values, P_n],
                    outputs=[A_Pn], device=dev,
                )
                wp.launch(
                    build_tr_stage1_rhs_kernel, dim=topo.n,
                    inputs=[P_n, A_Pn, success_rhs,
                            wp.float64(ah), wp.float64(gamma_h)],
                    outputs=[rhs], device=dev,
                )

                if step == 0:
                    tr_stage1_first_solve_s = direct.solve(); solve_count += 1
                    tr_stage1_first_residual = gpu_relative_residual(
                        topo.row_ptr, topo.col_ind, step_values, solution, rhs,
                        residual, topo.n, dev
                    )
                else:
                    direct.solve_async(); solve_count += 1

                wp.launch(
                    monitor_probability_kernel, dim=topo.n,
                    inputs=[solution, wp.float64(cfg.probability_tol), probability_flag], device=dev,
                )

                # Stage 2: BDF2 over the remaining (1-gamma)h interval.
                wp.launch(
                    build_tr_stage2_rhs_kernel, dim=topo.n,
                    inputs=[solution, P_n, success_rhs,
                            wp.float64(c_gamma), wp.float64(c_n), wp.float64(beta_h)],
                    outputs=[rhs], device=dev,
                )

                if step == 0:
                    tr_stage2_first_solve_s = direct.solve(); solve_count += 1
                    tr_stage2_first_residual = gpu_relative_residual(
                        topo.row_ptr, topo.col_ind, step_values, solution, rhs,
                        residual, topo.n, dev
                    )
                else:
                    direct.solve_async(); solve_count += 1

                wp.launch(
                    monitor_probability_kernel, dim=topo.n,
                    inputs=[solution, wp.float64(cfg.probability_tol), probability_flag], device=dev,
                )

                next_step = step + 1
                do_check = (next_step == 1 or next_step % cfg.steady_check_every == 0)
                do_save = (cfg.save_every > 0 and next_step % cfg.save_every == 0)

                if do_check:
                    mm = steady_match(solution)
                    append_history(next_step, mm)
                    steady_count = steady_count + 1 if mm["steady_ok"] else 0
                    check_probability_flag()

                if do_save:
                    host = np.asarray(solution.numpy(), dtype=np.float64)
                    snapshot_steps.append(next_step)
                    snapshot_lookback.append(next_step*h)
                    snapshots.append(host.reshape(grid.Np, grid.Nxi).copy())

                wp.launch(copy_float64_kernel, dim=topo.n,
                          inputs=[solution], outputs=[P_n], device=dev)
                step = next_step
                lookback = step*h

            wp.synchronize_stream(stream)
            trbdf2_loop_wall_s = time.perf_counter()-t_loop

        check_probability_flag()
        final_metrics = steady_match(P_n)
        if history[-1]["step"] != step:
            append_history(step, final_metrics)

        final_host = np.asarray(P_n.numpy(), dtype=np.float64)
        steady_host = np.asarray(P_steady.numpy(), dtype=np.float64)
        if not _probability_bounds_ok(final_host, cfg.probability_tol):
            raise RuntimeError("converged time-dependent RPF lies outside probability bounds")

        final_steady_residual = gpu_relative_residual(
            topo.row_ptr, topo.col_ind, adjoint_values, P_n, success_rhs, residual, topo.n, dev
        )
        if step > 0:
            last_linear_residual = gpu_relative_residual(
                topo.row_ptr, topo.col_ind, step_values, P_n, rhs, residual, topo.n, dev
            )
        else:
            last_linear_residual = 0.0

    return {
        "solve_mode": "steady_and_time",
        "P_terminal": terminal_host.reshape(grid.Np, grid.Nxi),
        "P_final": final_host.reshape(grid.Np, grid.Nxi),
        "P_steady_direct": steady_host.reshape(grid.Np, grid.Nxi),
        "success_rate": success_host.reshape(grid.Np, grid.Nxi),
        "failure_rate": failure_host.reshape(grid.Np, grid.Nxi),
        "outer_U_p": outer_U,
        "outer_success_mask": outer_U > 0.0,
        "history": history,
        "snapshot_steps": snapshot_steps,
        "snapshot_lookback": snapshot_lookback,
        "snapshots": snapshots,
        "steps": step,
        "steady_confirm_count": steady_count,
        "steady_lookback_tau": lookback,
        "runtime": runtime,
        "assembly_s": assembly_s,
        "analysis_s": analysis_s,
        "factor_s_total": factor_s_total,
        "factor_count": factor_count,
        "solve_count": solve_count,
        "steady_factor_s": steady_factor_s,
        "steady_solve_s": steady_solve_s,
        "trbdf2_factor_s": trbdf2_factor_s,
        "trbdf2_loop_wall_s": trbdf2_loop_wall_s,
        "tr_stage1_first_solve_s": tr_stage1_first_solve_s,
        "tr_stage2_first_solve_s": tr_stage2_first_solve_s,
        "tr_stage1_first_residual": tr_stage1_first_residual,
        "tr_stage2_first_residual": tr_stage2_first_residual,
        "last_linear_residual": last_linear_residual,
        "steady_linear_residual": steady_linear_residual,
        "final_steady_residual": final_steady_residual,
        "success_rhs_rel_l2": success_rel,
        "failure_rhs_rel_l2": failure_rel,
        "total_escape_rel_l2": total_escape_rel,
        "total_escape_backward_max": total_escape_backward,
        "trbdf2": {
            "gamma": gamma_tr,
            "beta": beta_tr,
            "shared_matrix_coefficient": a_tr,
            "c_gamma": c_gamma,
            "c_n": c_n,
        },
        "steady_target": {
            "mode": steady_target_mode,
            "rel_l2": float(final_metrics["steady_match_rel_l2"]),
            "max_abs": float(final_metrics["steady_match_max_abs"]),
            "direct_residual": float(steady_linear_residual),
            "has_success_boundary": bool(has_success_boundary),
            "success_pitch_cells": int(np.count_nonzero(outer_U > 0.0)),
            "steady_norm2": float(steady_norm2),
            "terminal_norm2": float(terminal_norm2),
        },
        **transpose_diag,
    }


# =============================================================================
# Configuration, reporting, and output
# =============================================================================

def load_config(config_path: Path | None = None) -> tuple[SolverConfig, str, Path]:
    path = (config_path or CONFIG_PATH).resolve()
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)

    plasma = raw["plasma"]
    collisions = raw["collisions"]
    grid = raw["grid"]
    solve_cfg = raw.get("solve", {})
    time_cfg = raw.get("time", {})
    runtime = raw["runtime"]
    output = raw["output"]

    species_list = []
    for x in plasma["species"]:
        species_list.append(AtomicSpecies(
            name=str(x["name"]),
            Z=int(x["Z"]),
            Zavg=float(x["Zavg"]),
            density_m3=float(x["density_m3"]),
        ))

    output_path = Path(output["path"])
    if not output_path.is_absolute():
        output_path = path.parent/output_path

    cfg = SolverConfig(
        te_eV=float(plasma["te_eV"]),
        E_over_Ec=float(plasma["E_over_Ec"]),
        B_T=float(plasma["B_T"]),
        species=tuple(species_list),
        energy_diffusion=bool(collisions["energy_diffusion"]),
        solve_mode=str(solve_cfg.get("mode", "steady_and_time")).strip().lower(),
        Np=int(grid["Np"]), Nxi=int(grid["Nxi"]),
        pmin=float(grid["pmin"]), pmax=float(grid["pmax"]),
        p_mapping=str(grid.get("p_mapping", "uniform")).strip().lower(),
        p_mapping_kappa=float(grid.get("p_mapping_kappa", 4.0)),
        xi_mapping=str(grid.get("xi_mapping", "uniform")).strip().lower(),
        terminal_p=float(time_cfg.get("terminal_p", 1.25)),
        terminal_condition=str(time_cfg.get("terminal_condition", "heaviside")).strip().lower(),
        terminal_smoothing_cells=float(time_cfg.get("terminal_smoothing_cells", 2.0)),
        dtau=float(time_cfg.get("dtau", 1.0e-3)),
        max_steps=int(time_cfg.get("max_steps", 100000)),
        max_lookback_tau=float(time_cfg.get("max_lookback_tau", 1.0e4)),
        probability_tol=float(time_cfg.get("probability_tol", 1.0e-8)),
        steady_check_every=int(time_cfg.get("steady_check_every", 10)),
        steady_match_rel_l2_tol=float(time_cfg.get("steady_match_rel_l2_tol", 1.0e-8)),
        steady_match_max_abs_tol=float(time_cfg.get("steady_match_max_abs_tol", 1.0e-7)),
        steady_confirm_steps=int(time_cfg.get("steady_confirm_steps", 3)),
        device=str(runtime["device"]),
        output=output_path,
        write_output=bool(output["write"]),
        save_every=int(output.get("save_every", 0)),
    )
    return cfg, text, path


def print_problem(cfg: SolverConfig, phys: DerivedPhysics, grid: Grid,
                  coll: CollisionData, topo: CSRTopology) -> None:
    print("=== standalone 0D-2P adjoint/RPF solver: steady + optional TR-BDF2 ===")
    print(f"solve mode={cfg.solve_mode}")
    print(f"execution=gpu-warp-cudss  grid={grid.Np}x{grid.Nxi}  state={grid.size:,}")
    print(
        f"p=[{cfg.pmin:g},{cfg.pmax:g}]  p_mapping={cfg.p_mapping}  "
        f"dp=[{grid.p_cell_widths.min():.6e},{grid.p_cell_widths.max():.6e}]  "
        f"dxi={grid.dxi:.6e}"
    )
    print("radial boundaries: pmin=absorbing failure, pmax=open successful outflow where U_p>0")
    if cfg.solve_mode == "steady_and_time":
        if cfg.terminal_condition == "heaviside":
            print(
                f"terminal condition: discontinuous H(p-p_RE), p_RE={cfg.terminal_p:.8g} "
                "(exact FV volume fraction)"
            )
        else:
            print(
                f"terminal condition: smoothed H(p-p_RE), p_RE={cfg.terminal_p:.8g}  "
                f"width={cfg.terminal_smoothing_cells:.6g} cells = "
                f"{cfg.terminal_smoothing_cells*_terminal_cell_width(cfg, grid):.6e} in p "
                "(tanh, p^2-FV averaged)"
            )
        print("integration: backward in physical time = forward in lookback s=T-t")
    else:
        print("time-dependent solve=off (direct steady RPF only)")
    print(
        f"Te={cfg.te_eV:.6e} eV  ne(quasineutral)={phys.free_density_from_ions_m3:.6e} m^-3  "
        f"Z_eff={phys.z_eff:.8g}"
    )
    for spc in cfg.species:
        print(
            f"  species={spc.name} Z={spc.Z} <Z>={spc.Zavg:.8g} "
            f"n_atomic={spc.density_m3:.6e} m^-3"
        )
    print(f"lnLambda0={phys.ln_lambda0:.8g}  tau_c={phys.tau_ref_s:.6e} s  Ec={phys.E_ref_Vm:.6e} V/m")
    print(f"E/Ec={phys.Ebar:.8g}  E_parallel={phys.e_parallel_Vm:.6e} V/m")
    print(f"B={cfg.B_T:.6e} T  alpha=tau_c/tau_s={phys.alpha:.6e}")
    print("small-angle model=fully-relativistic-asymptotically-matched")
    print("energy-dependent Coulomb logs=on (Hesslow thermal/high-energy match, k=5)")
    print(
        f"lnLambda_ee centers=[{coll.ln_lambda_ee_centers.min():.8g},{coll.ln_lambda_ee_centers.max():.8g}]  "
        f"lnLambda_ei centers=[{coll.ln_lambda_ei_centers.min():.8g},{coll.ln_lambda_ei_centers.max():.8g}]"
    )
    print(f"partial screening={'on' if any(x.n_bound for x in cfg.resolved_ions()) else 'off'}")
    for ion in cfg.resolved_ions():
        extra = f"  I={ion.I_eV:.6g} eV  a_bar={ion.a_bar:.6g}" if ion.n_bound else ""
        print(f"  resolved={ion.name} Z={ion.Z} Z0={ion.Z0} N_bound={ion.n_bound} n={ion.density_m3:.6e} m^-3{extra}")
    print(f"radial energy diffusion={'on' if cfg.energy_diffusion else 'off'}")
    print(f"matched collision p_switch={coll.p_switch:.6e} Phi/Psi overlap={coll.overlap_rel:.3e}")
    print(f"CSR nnz={topo.nnz:,} avg nnz/row={topo.nnz/topo.n:.2f}")
    if cfg.solve_mode == "steady_and_time":
        print(
            "fixed-step TR-BDF2: "
            f"dtau={cfg.dtau:.6e}  gamma={2.0-math.sqrt(2.0):.12f}  "
            f"max_steps={cfg.max_steps}  steady_check_every={cfg.steady_check_every}"
        )
        print(
            "steady stop against pre-solved GPU RPF: "
            f"rel_l2_tol={cfg.steady_match_rel_l2_tol:.1e} "
            f"max_abs_tol={cfg.steady_match_max_abs_tol:.1e} "
            f"confirm={cfg.steady_confirm_steps}"
        )

def report_result(result: dict) -> None:
    print("=== result ===")
    st = result["steady_target"]
    if result["solve_mode"] == "steady":
        P = np.asarray(result["P_steady_direct"], dtype=np.float64)
        print("solve mode=steady")
        print(f"steady RPF min={P.min():.12e} max={P.max():.12e}")
        print(f"successful pmax pitch cells={st['success_pitch_cells']}  steady_target_mode={st['mode']}")
        print(f"steady direct residual={result['steady_linear_residual']:.3e}")
        print(
            "independent transpose: "
            f"structure={result['transpose_structure_ok']} max_abs={result['transpose_max_abs']:.3e} "
            f"rel_l2={result['transpose_rel_l2']:.3e}"
        )
        print(f"escape vectors: success_rel={result['success_rhs_rel_l2']:.3e} failure_rel={result['failure_rhs_rel_l2']:.3e}")
        print(
            f"total escape identity rel_l2={result['total_escape_rel_l2']:.3e} "
            f"backward_max={result['total_escape_backward_max']:.3e}"
        )
        print(
            f"assembly_s={result['assembly_s']:.6e} analysis_s={result['analysis_s']:.6e} "
            f"steady_factor_s={result['steady_factor_s']:.6e} steady_solve_s={result['steady_solve_s']:.6e}"
        )
        print(
            f"factor_s_total={result['factor_s_total']:.6e} ({result['factor_count']} factorization) "
            f"solve_count={result['solve_count']}"
        )
        print("per-run steady adjoint checks=PASS")
        return

    h = result["history"][-1]
    P_final = np.asarray(result["P_final"], dtype=np.float64)
    print("solve mode=steady_and_time")
    print(
        f"steps={result['steps']} dtau={h['dtau']:.6e} "
        f"steady_lookback_tau={result['steady_lookback_tau']:.12e}"
    )
    print(f"final RPF min={P_final.min():.12e} max={P_final.max():.12e}")
    print(f"successful pmax pitch cells={st['success_pitch_cells']}  steady_target_mode={st['mode']}")
    metric_name = "rel_l2_to_steady" if st["has_success_boundary"] else "rel_l2_decay_from_terminal"
    print(
        "GPU steady-target comparison: "
        f"mode={st['mode']} {metric_name}={st['rel_l2']:.3e} "
        f"max_abs={st['max_abs']:.3e} steady_direct_residual={st['direct_residual']:.3e}"
    )
    print(f"final steady-equation residual={result['final_steady_residual']:.3e}")
    print(
        "independent transpose: "
        f"structure={result['transpose_structure_ok']} max_abs={result['transpose_max_abs']:.3e} "
        f"rel_l2={result['transpose_rel_l2']:.3e}"
    )
    print(f"escape vectors: success_rel={result['success_rhs_rel_l2']:.3e} failure_rel={result['failure_rhs_rel_l2']:.3e}")
    print(
        f"total escape identity rel_l2={result['total_escape_rel_l2']:.3e} "
        f"backward_max={result['total_escape_backward_max']:.3e}"
    )
    print(
        f"TR stage-1 first residual={result['tr_stage1_first_residual']:.3e}  "
        f"TR stage-2 first residual={result['tr_stage2_first_residual']:.3e}  "
        f"final step residual={result['last_linear_residual']:.3e}"
    )
    print(
        f"assembly_s={result['assembly_s']:.6e} analysis_s={result['analysis_s']:.6e} "
        f"steady_factor_s={result['steady_factor_s']:.6e} steady_solve_s={result['steady_solve_s']:.6e} "
        f"TRBDF2_factor_s={result['trbdf2_factor_s']:.6e} "
        f"TRBDF2_loop_wall_s={result['trbdf2_loop_wall_s']:.6e}"
    )
    print(
        f"factor_s_total={result['factor_s_total']:.6e} ({result['factor_count']} factorizations) "
        f"solve_count={result['solve_count']}"
    )
    print("per-run steady + time-dependent adjoint checks=PASS")

def save_result(path: Path, result: dict, cfg: SolverConfig, phys: DerivedPhysics,
                grid: Grid, coll: CollisionData, config_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)
    config_dict["output"] = str(config_dict["output"])
    config_dict["species"] = [asdict(x) for x in cfg.species]
    config_dict["resolved_ions"] = [asdict(x) for x in cfg.resolved_ions()]

    if result["solve_mode"] == "steady":
        payload = dict(
            output_schema=np.asarray("steady-rpf-v1"),
            solve_mode=np.asarray("steady"),
            p_faces=grid.p_faces, p_centers=grid.p_centers,
            xi_faces=grid.xi_faces, xi_centers=grid.xi_centers,
            P_steady_direct=np.asarray(result["P_steady_direct"], dtype=np.float64),
            success_rate=np.asarray(result["success_rate"], dtype=np.float64),
            failure_rate=np.asarray(result["failure_rate"], dtype=np.float64),
            outer_U_p=np.asarray(result["outer_U_p"], dtype=np.float64),
            outer_success_mask=np.asarray(result["outer_success_mask"], dtype=np.bool_),
            cf_faces=coll.cf_faces, ca_faces=coll.ca_faces, nud_centers=coll.nud_centers,
            ln_lambda_ee_faces=coll.ln_lambda_ee_faces,
            ln_lambda_ei_faces=coll.ln_lambda_ei_faces,
            ln_lambda_ee_centers=coll.ln_lambda_ee_centers,
            ln_lambda_ei_centers=coll.ln_lambda_ei_centers,
            coulomb_log_matching_k=np.asarray(5.0),
            energy_dependent_coulomb_logs=np.asarray(True),
            small_angle_model=np.asarray("fully-relativistic-asymptotically-matched"),
            partial_screening=np.asarray(any(x.n_bound for x in cfg.resolved_ions())),
            input_config_toml=np.asarray(config_text),
            config_json=np.asarray(json.dumps(config_dict, sort_keys=True)),
            physics_json=np.asarray(json.dumps(asdict(phys), sort_keys=True)),
            runtime_json=np.asarray(json.dumps(result["runtime"], sort_keys=True)),
            steady_target_json=np.asarray(json.dumps(result["steady_target"], sort_keys=True)),
            transpose_max_abs=np.asarray(result["transpose_max_abs"]),
            transpose_rel_l2=np.asarray(result["transpose_rel_l2"]),
            total_escape_rel_l2=np.asarray(result["total_escape_rel_l2"]),
            total_escape_backward_max=np.asarray(result["total_escape_backward_max"]),
            assembly_s=np.asarray(result["assembly_s"]), analysis_s=np.asarray(result["analysis_s"]),
            factor_s_total=np.asarray(result["factor_s_total"]),
            factor_count=np.asarray(result["factor_count"], dtype=np.int64),
            solve_count=np.asarray(result["solve_count"], dtype=np.int64),
            steady_factor_s=np.asarray(result["steady_factor_s"]),
            steady_solve_s=np.asarray(result["steady_solve_s"]),
            steady_linear_residual=np.asarray(result["steady_linear_residual"]),
        )
        np.savez_compressed(path, **payload)
        return

    hist = result["history"]
    payload = dict(
        output_schema=np.asarray("time-rpf-v1"),
        solve_mode=np.asarray("steady_and_time"),
        p_faces=grid.p_faces, p_centers=grid.p_centers,
        xi_faces=grid.xi_faces, xi_centers=grid.xi_centers,
        P_terminal=np.asarray(result["P_terminal"], dtype=np.float64),
        P_final=np.asarray(result["P_final"], dtype=np.float64),
        success_rate=np.asarray(result["success_rate"], dtype=np.float64),
        failure_rate=np.asarray(result["failure_rate"], dtype=np.float64),
        outer_U_p=np.asarray(result["outer_U_p"], dtype=np.float64),
        outer_success_mask=np.asarray(result["outer_success_mask"], dtype=np.bool_),
        terminal_p=np.asarray(cfg.terminal_p),
        terminal_condition=np.asarray(cfg.terminal_condition),
        terminal_smoothing_cells=np.asarray(cfg.terminal_smoothing_cells),
        terminal_smoothing_dp=np.asarray(
            cfg.terminal_smoothing_cells*_terminal_cell_width(cfg, grid)
            if cfg.terminal_condition == "smoothed_heaviside" else 0.0
        ),
        diagnostic_step=np.asarray([x["step"] for x in hist], dtype=np.int64),
        lookback_tau=np.asarray([x["lookback_tau"] for x in hist], dtype=np.float64),
        dtau=np.asarray(cfg.dtau),
        steady_match_rel_l2=np.asarray([x["steady_match_rel_l2"] for x in hist], dtype=np.float64),
        steady_match_max_abs=np.asarray([x["steady_match_max_abs"] for x in hist], dtype=np.float64),
        snapshot_step=np.asarray(result["snapshot_steps"], dtype=np.int64),
        snapshot_lookback_tau=np.asarray(result["snapshot_lookback"], dtype=np.float64),
        steps=np.asarray(result["steps"], dtype=np.int64),
        steady_lookback_tau=np.asarray(result["steady_lookback_tau"]),
        cf_faces=coll.cf_faces, ca_faces=coll.ca_faces, nud_centers=coll.nud_centers,
        ln_lambda_ee_faces=coll.ln_lambda_ee_faces,
        ln_lambda_ei_faces=coll.ln_lambda_ei_faces,
        ln_lambda_ee_centers=coll.ln_lambda_ee_centers,
        ln_lambda_ei_centers=coll.ln_lambda_ei_centers,
        coulomb_log_matching_k=np.asarray(5.0),
        energy_dependent_coulomb_logs=np.asarray(True),
        small_angle_model=np.asarray("fully-relativistic-asymptotically-matched"),
        partial_screening=np.asarray(any(x.n_bound for x in cfg.resolved_ions())),
        input_config_toml=np.asarray(config_text),
        config_json=np.asarray(json.dumps(config_dict, sort_keys=True)),
        physics_json=np.asarray(json.dumps(asdict(phys), sort_keys=True)),
        runtime_json=np.asarray(json.dumps(result["runtime"], sort_keys=True)),
        steady_target_json=np.asarray(json.dumps(result["steady_target"], sort_keys=True)),
        transpose_max_abs=np.asarray(result["transpose_max_abs"]),
        transpose_rel_l2=np.asarray(result["transpose_rel_l2"]),
        total_escape_rel_l2=np.asarray(result["total_escape_rel_l2"]),
        total_escape_backward_max=np.asarray(result["total_escape_backward_max"]),
        assembly_s=np.asarray(result["assembly_s"]), analysis_s=np.asarray(result["analysis_s"]),
        factor_s_total=np.asarray(result["factor_s_total"]),
        factor_count=np.asarray(result["factor_count"], dtype=np.int64),
        solve_count=np.asarray(result["solve_count"], dtype=np.int64),
        steady_factor_s=np.asarray(result["steady_factor_s"]),
        steady_solve_s=np.asarray(result["steady_solve_s"]),
        trbdf2_factor_s=np.asarray(result["trbdf2_factor_s"]),
        trbdf2_loop_wall_s=np.asarray(result["trbdf2_loop_wall_s"]),
        tr_stage1_first_solve_s=np.asarray(result["tr_stage1_first_solve_s"]),
        tr_stage2_first_solve_s=np.asarray(result["tr_stage2_first_solve_s"]),
        tr_stage1_first_residual=np.asarray(result["tr_stage1_first_residual"]),
        tr_stage2_first_residual=np.asarray(result["tr_stage2_first_residual"]),
        last_linear_residual=np.asarray(result["last_linear_residual"]),
        steady_linear_residual=np.asarray(result["steady_linear_residual"]),
        final_steady_residual=np.asarray(result["final_steady_residual"]),
        trbdf2_json=np.asarray(json.dumps(result["trbdf2"], sort_keys=True)),
        time_integrator=np.asarray("fixed-step-tr-bdf2-gamma-2-sqrt2-gpu-steady-target"),
    )
    if result["P_steady_direct"] is not None:
        payload["P_steady_direct"] = np.asarray(result["P_steady_direct"], dtype=np.float64)
    for k, snap in enumerate(result["snapshots"]):
        payload[f"P_snapshot_{k:06d}"] = np.asarray(snap, dtype=np.float64)
    np.savez_compressed(path, **payload)


def plot_summary(path: Path, result: dict, cfg: SolverConfig, grid: Grid, phys: DerivedPhysics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result["solve_mode"] == "steady":
        plot_path = path.with_name(path.stem + "_steady_rpf.png")
        P = np.asarray(result["P_steady_direct"], dtype=np.float64).T
        fig, ax = plt.subplots(constrained_layout=True)
        cf = ax.contourf(grid.p_centers, grid.xi_centers, P, levels=50, cmap="turbo", extend="both")
        cbar = fig.colorbar(cf, ax=ax); cbar.set_label(r"runaway probability $P$")
        ax.set_xlabel(r"$p=P/(m_e c)$"); ax.set_ylabel(r"$\xi=P_\parallel/P$")
        ax.set_title(rf"Steady RPF: $E/E_c={phys.Ebar:.4g}$")
        ax.set_xlim(grid.p_faces[0], grid.p_faces[-1]); ax.set_ylim(-1.0, 1.0)
        fig.savefig(plot_path); plt.close(fig)
        print(f"wrote {plot_path}")
        return

    plot_path = path.with_name(path.stem + "_final_rpf.png")
    P = np.asarray(result["P_final"], dtype=np.float64).T
    fig, ax = plt.subplots(constrained_layout=True)
    cf = ax.contourf(grid.p_centers, grid.xi_centers, P, levels=50, cmap="turbo", extend="both")
    cbar = fig.colorbar(cf, ax=ax); cbar.set_label(r"runaway probability $P$")
    ax.set_xlabel(r"$p=P/(m_e c)$"); ax.set_ylabel(r"$\xi=P_\parallel/P$")
    ax.set_title(rf"Time-dependent RPF at steady limit: $E/E_c={phys.Ebar:.4g}$")
    ax.set_xlim(grid.p_faces[0], grid.p_faces[-1]); ax.set_ylim(-1.0, 1.0)
    fig.savefig(plot_path); plt.close(fig)
    print(f"wrote {plot_path}")

    conv_path = path.with_name(path.stem + "_convergence.png")
    hist = result["history"][1:]
    if hist:
        s = np.asarray([x["lookback_tau"] for x in hist], dtype=np.float64)
        r = np.asarray([x["steady_match_rel_l2"] for x in hist], dtype=np.float64)
        d = np.asarray([x["steady_match_max_abs"] for x in hist], dtype=np.float64)
        fig, ax = plt.subplots(constrained_layout=True)
        if result["steady_target"]["has_success_boundary"]:
            l2_label = r"$||P-P_{ss}||_2/||P_{ss}||_2$"
        else:
            l2_label = r"$||P||_2/||P_T||_2$"
        ax.semilogy(s, np.maximum(r, 1e-300), label=l2_label)
        ax.semilogy(s, np.maximum(d, 1e-300), label=r"$||P-P_{ss}||_\infty$")
        ax.set_xlabel(r"lookback time $s=(T-t)/\tau_c$")
        ax.set_ylabel("relative diagnostic")
        ax.legend()
        fig.savefig(conv_path); plt.close(fig)
        print(f"wrote {conv_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU finite-volume adjoint RPF solver")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="TOML case file (default: %(default)s)",
    )
    args = parser.parse_args()
    cfg, config_text, config_path = load_config(args.config)
    if wp is None:
        raise RuntimeError("adjoint_fv_solver.py requires warp-lang")

    print(f"loaded case: {config_path}")
    t0 = time.perf_counter()
    validate_rpf_config(cfg)
    phys = derive_physics(cfg)
    grid = build_grid(cfg)

    print("evaluating small-angle collision coefficients ...", flush=True)
    t = time.perf_counter()
    coll = build_collision_data(cfg, phys, grid)
    validate_collision_data(coll)
    print(f"collision coefficients: {time.perf_counter()-t:.3f} s")

    print("initializing Warp/cuDSS runtime ...", flush=True)
    gpu_context = load_gpu_runtime(cfg.device)

    print("building local five-point CSR topology on GPU ...", flush=True)
    topo, topo_timing = build_local_csr_topology_gpu(grid, cfg.device)
    print(
        "CSR topology: "
        f"rowptr_host={topo_timing['rowptr_host_s']:.3e} s  "
        f"fill_gpu={topo_timing['fill_gpu_s']:.3e} s  "
        f"validate_gpu={topo_timing['validate_gpu_s']:.3e} s  "
        f"total={topo_timing['total_s']:.3e} s"
    )

    print_problem(cfg, phys, grid, coll, topo)
    result = solve_adjoint_rpf(cfg, phys, grid, coll, topo, gpu_context)
    result["topology_timing"] = topo_timing
    report_result(result)

    if cfg.write_output:
        save_result(cfg.output, result, cfg, phys, grid, coll, config_text)
        print(f"wrote {cfg.output}")
    else:
        print("output writing disabled by [output].write=false")
    plot_summary(cfg.output, result, cfg, grid, phys)
    print(f"total wall={time.perf_counter()-t0:.3f} s")


if __name__ == "__main__":
    main()
