#!/usr/bin/env python3
"""
Standalone GPU 0D-2P relativistic electron kinetic finite-volume solver.

Scientific/numerical authority
------------------------------
    kinetic_model_reorganized_v15_4_fv_assisted_pinn.tex (September 2026)

Execution model
---------------
* NVIDIA GPU only for kinetic assembly, implicit TR--BDF2 time integration, and
  sparse direct solves. Warp owns the FP64 CSR matrix/vectors; nvmath-python/
  cuDSS consumes the Warp device pointers on the same CUDA stream.
* The physical state is the uniform finite-volume (p, xi) state. The optional
  Chiu-Harvey gain uses the qualified exact auxiliary pitch-reduction system.
* NumPy/SciPy are restricted to time-independent host preprocessing such as
  physical constants, one-dimensional coefficient work, Chiu-Harvey kinematics,
  initial projection, and compact row-pointer construction. There is no CPU
  kinetic operator or CPU sparse solve.

Selectable small-angle test-particle collision models
------------------------------------------------------
1. finite-temperature-relativistic
2. fully-relativistic-asymptotically-matched

Selectable large-angle secondary-electron gain
-----------------------------------------------
1. none
2. chiu-harvey

The production solver intentionally retains only the no-gain and Chiu-Harvey
large-angle configurations. The Chiu-Harvey model keeps the
finite-primary-energy knock-on physics already benchmarked against McDevitt 2019
Fig. B3(a). The R7 shifted-domain primary-cell mapping fix is retained.

CUDA-13 environment used by the project:
    python -m pip install numpy scipy warp-lang "nvmath-python[cu13]"

Run configuration
-----------------
All physics, numerical, runtime, and output settings default to the TOML file
beside this script. Use ``--config`` to select another case. Relative output
paths are resolved relative to the configuration file.

Partially ionized ions require positive ``I_eV`` and ``a_bar``. The same Hesslow
partial-screening corrections to collisional drag and pitch-angle deflection are
applied to either selectable complete-screening base small-angle model; radial
energy diffusion is not modified by the screening overlay.

This is a research solver, not a convergence claim. Delta-p, Delta-xi, p_max,
q_coll (where applicable), q_init, and Delta-tau remain independent numerical
qualification axes.
"""

from __future__ import annotations

import tomllib
import argparse
import ctypes
from dataclasses import asdict, dataclass, field
import importlib.metadata as importlib_metadata
import json
import math
from pathlib import Path
import site
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.constants as const
import scipy.special as sps

try:
    import warp as wp
except Exception:
    wp = None


MEC2_EV = const.m_e * const.c**2 / const.e
R_E = const.e**2 / (4.0 * math.pi * const.epsilon_0 * const.m_e * const.c**2)


# =============================================================================
# Configuration and geometry
# =============================================================================

@dataclass(frozen=True)
class IonSpecies:
    name: str
    Z: int
    Z0: int
    density_m3: float
    I_eV: float = 0.0
    a_bar: float = 0.0
    # Optional state-resolved inputs supplied by a coupled bulk model.  When
    # present, entries are ordered q=0,...,Z and replace the legacy scalar
    # effective-ion screening inputs above.
    charge_populations_m3: tuple[float, ...] | None = None
    I_eV_by_charge: tuple[float, ...] | None = None
    a_bar_by_charge: tuple[float, ...] | None = None

    @property
    def n_bound(self) -> int:
        return self.Z - self.Z0




@dataclass(frozen=True)
class SolverConfig:
    """Complete immutable case configuration loaded from TOML."""

    te_eV: float = 1000.0
    ne_m3: float = 1.0e20
    e_parallel_Vm: float = 0.0
    B_T: float = 0.0
    ions: tuple[IonSpecies, ...] = field(default_factory=tuple)
    nt_m3: float | None = None

    n_ref_m3: float | None = None
    ln_lambda_ref: float | None = None

    # Small-angle base operator and Chiu-Harvey avalanche cutoff.
    small_angle_model: str = "finite-temperature-relativistic"
    large_angle_model: str = "chiu-harvey"
    p_m: float = 0.1

    # Uniform finite-volume mesh and radial-boundary semantics.
    Np: int = 96
    Nxi: int = 64
    pmin: float = 0.0
    pmax: float = 5.0
    inner_boundary: str = "zero-flux"
    energy_diffusion: bool = True
    q_coll: int = 48
    q_init: int = 8

    # Dimensionless collision-time integration interval.
    dtau: float = 1.0e-3
    tau_end: float = 1.0e-2

    # Cell-integrated initial condition.
    init: str = "maxwell-juttner"
    init_te_eV: float | None = None
    init_density_m3: float | None = None
    gaussian_p0: float = 0.6
    gaussian_sigma_p: float = 0.15
    gaussian_xi0: float = -0.8
    gaussian_sigma_xi: float = 0.15

    # Optional pre-existing runaway seed. seed_fraction is a fraction of the
    # total represented initial density, not an additive density increase.
    seed_fraction: float = 0.0
    seed_p0: float = 1.0
    seed_sigma_p: float = 0.10
    seed_xi0: float = -0.95
    seed_sigma_xi: float = 0.05

    # Runtime/output policy. The production path uses the plain cuDSS direct solve.
    device: str = "cuda:0"
    adaptive: bool = False
    rtol: float = 1.0e-4
    atol: float = 1.0e-12
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    dtau_min: float = 1.0e-8
    dtau_max: float | None = None
    max_steps: int = 100000
    save_every: int = 0
    diag_every: int = 0
    output: Path = Path("runaway_0d2p_output.npz")
    write_output: bool = True
    balance_check: bool = True

    @property
    def include_ch(self) -> bool:
        return self.large_angle_model == "chiu-harvey"

    def resolved_ions(self) -> tuple[IonSpecies, ...]:
        return self.ions



@dataclass(frozen=True)
class DerivedPhysics:
    theta: float
    ln_lambda0: float
    z_eff: float
    n_ref_m3: float
    ln_lambda_ref: float
    tau_ref_s: float
    E_ref_Vm: float
    Ebar: float
    tau_syn_s: float
    alpha: float
    nt_m3: float
    nt_over_nref: float
    free_density_from_ions_m3: float


@dataclass(frozen=True)
class Grid:
    p_faces: np.ndarray
    p_centers: np.ndarray
    xi_faces: np.ndarray
    xi_centers: np.ndarray
    dp: float
    dxi: float
    radial_volume: np.ndarray  # (2*pi/3)*(p_hi^3-p_lo^3)
    cell_volume: np.ndarray    # flattened V_ij = radial_volume_i * dxi
    current_radial_weight: np.ndarray
    current_pitch_weight: np.ndarray

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
    model: str
    cf_faces: np.ndarray
    ca_faces: np.ndarray
    nud_centers: np.ndarray
    psi_s_centers: np.ndarray
    psi_d_centers: np.ndarray
    p_switch: float
    k2_scaled: float
    overlap_rel_psi_s: float
    overlap_rel_psi_d: float


@dataclass(frozen=True)
class CHGeometry:
    active_rows: np.ndarray
    primary_radial: np.ndarray
    row_coefficient: np.ndarray
    root_p: np.ndarray

    @property
    def active_count(self) -> int:
        return self.active_rows.size


@dataclass(frozen=True)
class CSRTopology:
    """GPU-resident fixed CSR graph and direct slot maps.

    ``physical_n`` is the number of finite-volume cell-content unknowns.  When
    Chiu-Harvey is active, ``n = physical_n + Np`` because one auxiliary radial
    pitch-integrated mass is appended per radial cell.  The auxiliary equations
    are algebraic constraints, not additional physical state variables.
    """
    n: int
    nnz_count: int
    row_ptr: object
    col_ind: object
    diag_slot: object
    pm_slot: object
    pp_slot: object
    xm_slot: object
    xp_slot: object
    physical_n: int
    formulation: str = "local"
    ch_aux_slot: object | None = None
    aux_n_start_slot: object | None = None
    aux_diag_slot: object | None = None

    @property
    def nnz(self) -> int:
        return self.nnz_count


def thermal_coulomb_log(ne_m3: float, te_eV: float) -> float:
    # Thermal Coulomb logarithm in the normalization used by the model.
    return 14.9 - 0.5 * math.log(ne_m3 / 1.0e20) + math.log(te_eV / 1.0e3)


def collision_time(ne_m3: float, ln_lambda: float) -> float:
    return 4.0 * math.pi * const.epsilon_0**2 * const.m_e**2 * const.c**3 / (
        ne_m3 * const.e**4 * ln_lambda
    )


def derive_physics(cfg: SolverConfig) -> DerivedPhysics:
    ions = cfg.resolved_ions()
    theta = cfg.te_eV / MEC2_EV
    ln0 = thermal_coulomb_log(cfg.ne_m3, cfg.te_eV)
    if ln0 <= 0.0:
        raise ValueError(f"thermal Coulomb logarithm is non-positive: {ln0}")
    free_from_ions = sum(x.density_m3 * x.Z0 for x in ions)
    z_eff = sum(x.density_m3 * x.Z0**2 for x in ions) / cfg.ne_m3
    nref = cfg.ne_m3 if cfg.n_ref_m3 is None else cfg.n_ref_m3
    if nref <= 0.0:
        raise ValueError("n_ref_m3 must be positive")
    lnref = ln0 if cfg.ln_lambda_ref is None else cfg.ln_lambda_ref
    if lnref <= 0.0:
        raise ValueError("ln_lambda_ref must be positive")
    tauref = collision_time(nref, lnref)
    eref = const.m_e * const.c / (const.e * tauref)
    ebar = cfg.e_parallel_Vm / eref
    if cfg.B_T == 0.0:
        taus = math.inf
        alpha = 0.0
    else:
        taus = 6.0 * math.pi * const.epsilon_0 * const.m_e**3 * const.c**3 / (
            const.e**4 * cfg.B_T**2
        )
        alpha = tauref / taus
    bound = sum(x.density_m3 * x.n_bound for x in ions)
    nt = cfg.ne_m3 + bound if cfg.nt_m3 is None else cfg.nt_m3
    if nt < 0.0:
        raise ValueError("nt_m3 must be non-negative")
    return DerivedPhysics(
        theta=theta, ln_lambda0=ln0, z_eff=z_eff,
        n_ref_m3=nref, ln_lambda_ref=lnref,
        tau_ref_s=tauref, E_ref_Vm=eref, Ebar=ebar,
        tau_syn_s=taus, alpha=alpha,
        nt_m3=nt, nt_over_nref=nt/nref,
        free_density_from_ions_m3=free_from_ions,
    )


def build_grid(cfg: SolverConfig) -> Grid:
    dp = (cfg.pmax - cfg.pmin) / cfg.Np
    dxi = 2.0 / cfg.Nxi
    pf = np.linspace(cfg.pmin, cfg.pmax, cfg.Np + 1, dtype=np.float64)
    pc = cfg.pmin + (np.arange(cfg.Np, dtype=np.float64) + 0.5) * dp
    xf = np.linspace(-1.0, 1.0, cfg.Nxi + 1, dtype=np.float64)
    xc = -1.0 + (np.arange(cfg.Nxi, dtype=np.float64) + 0.5) * dxi
    rv = (2.0 * math.pi / 3.0) * (pf[1:]**3 - pf[:-1]**3)
    vol = np.repeat(rv * dxi, cfg.Nxi)
    # Exact cell-integrated velocity/pitch weights for parallel current.
    gp = np.sqrt(1.0 + pf*pf)
    primitive = gp**3 / 3.0 - gp
    wv = 2.0 * math.pi * (primitive[1:] - primitive[:-1])
    wx = 0.5 * (xf[1:]**2 - xf[:-1]**2)
    return Grid(pf, pc, xf, xc, dp, dxi, rv, vol, wv, wx)


# =============================================================================
# Finite-temperature relativistic collision coefficients
# =============================================================================

def _low_p_coefficients(theta: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    t = theta
    a1 = (2*t*t + 2*t + 1)/(3*t)
    a3 = -(6*t**3 + 6*t*t + 5*t + 3)/(30*t*t)
    a5 = (30*t**4 + 30*t**3 + 27*t*t + 17*t + 5)/(280*t**3)
    a7 = -(210*t**5 + 210*t**4 + 195*t**3 + 125*t*t + 44*t + 7)/(3024*t**4)
    a9 = (1890*t**6 + 1890*t**5 + 1785*t**4 + 1155*t**3 + 435*t*t + 92*t + 9)/(38016*t**5)
    a11 = -(20790*t**7 + 20790*t**6 + 19845*t**5 + 12915*t**4 + 5040*t**3 + 1197*t*t + 167*t + 11)/(549120*t**6)
    a13 = (270270*t**8 + 270270*t**7 + 259875*t**6 + 169785*t**5 + 67725*t**4 + 17136*t**3 + 2786*t*t + 275*t + 13)/(8985600*t**7)

    b0 = (8*t*t + 5*t + 4)/12
    b2 = (64*t**3 + 79*t*t + 30*t - 8)/(240*t)
    b4 = -(768*t**4 + 873*t**3 + 274*t*t + 53*t - 12)/(3360*t*t)
    b6 = (49152*t**5 + 53877*t**4 + 15594*t**3 + 3830*t*t + 572*t - 80)/(241920*t**3)
    b8 = -(983040*t**6 + 1055805*t**5 + 292986*t**4 + 75031*t**3 + 16876*t*t + 1545*t - 140)/(5322240*t**4)
    b10 = (47185920*t**7 + 50023755*t**6 + 13539990*t**5 + 3487020*t**4 + 866196*t**3 + 136817*t*t + 8050*t - 504)/(276756480*t**5)
    b12 = -(440401920*t**8 + 462699195*t**7 + 123237750*t**6 + 31617235*t**5 + 8090500*t**4 + 1515314*t**3 + 167034*t*t + 6755*t - 308)/(2767564800*t**6)
    return (a1, a3, a5, a7, a9, a11, a13), (b0, b2, b4, b6, b8, b10, b12)


def _horner_even(p: np.ndarray, coeff: tuple[float, ...]) -> np.ndarray:
    z = p*p
    out = np.full_like(p, coeff[-1], dtype=np.float64)
    for c in reversed(coeff[:-1]):
        out = out*z + c
    return out


def _scaled_rapidity_integrals(p: np.ndarray, theta: float, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially scaled finite collision integrals using Gauss-Legendre rapidity quadrature."""
    p = np.asarray(p, dtype=np.float64)
    y1 = np.arcsinh(p)
    x, w = leggauss(q)
    out0 = np.empty_like(p)
    out1 = np.empty_like(p)
    # Chunking bounds temporary memory for very fine radial meshes.
    chunk = max(1, 1_000_000 // max(1, q))
    for lo in range(0, p.size, chunk):
        hi = min(p.size, lo + chunk)
        yy = 0.5*y1[lo:hi, None]*(x[None, :] + 1.0)
        ch = np.cosh(yy)
        expo = np.exp(-(ch - 1.0)/theta)
        jac = 0.5*y1[lo:hi, None]
        out0[lo:hi] = np.sum(jac*w[None, :]*expo, axis=1)
        out1[lo:hi] = np.sum(jac*w[None, :]*expo*ch, axis=1)
    return out0, out1


def collision_functions(p: np.ndarray, theta: float, q_coll: int, *, return_branches: bool = False):
    p = np.asarray(p, dtype=np.float64)
    if np.any(p <= 0.0):
        raise ValueError("collision functions must be evaluated only at p>0")
    gamma = np.sqrt(1.0 + p*p)
    psi0, psi1 = _scaled_rapidity_integrals(p, theta, q_coll)
    expg = np.exp(-(gamma - 1.0)/theta)
    k2s = float(sps.kve(2, 1.0/theta))  # exp(1/theta) K2(1/theta)

    psis_dir = (gamma*gamma*psi1 - theta*psi0 + (theta*gamma - 1.0)*p*expg) / (p*p*k2s)
    psid_dir = (
        (p*p*gamma*gamma + theta*theta)*psi0
        + theta*(2.0*p**4 - 1.0)*psi1
        + gamma*theta*(1.0 + theta*(2.0*p*p - 1.0))*p*expg
    ) / (2.0*gamma*p**3*k2s)

    aa, bb = _low_p_coefficients(theta)
    psis_ser = p*_horner_even(p, aa)/k2s
    psid_ser = _horner_even(p, bb)/k2s
    psw = 0.1*min(1.0, math.sqrt(theta))
    low = p <= psw
    psis = np.where(low, psis_ser, psis_dir)
    psid = np.where(low, psid_ser, psid_dir)
    if return_branches:
        return psis, psid, psis_dir, psid_dir, psis_ser, psid_ser, psw, k2s
    return psis, psid, psw, k2s


def screening_h_g(p: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p*p)
    beta2 = p*p/(gamma*gamma)
    h = np.zeros_like(p)
    g = np.zeros_like(p)
    for ion in cfg.resolved_ions():
        if ion.density_m3 == 0.0:
            continue
        if ion.charge_populations_m3 is not None:
            assert ion.I_eV_by_charge is not None
            assert ion.a_bar_by_charge is not None
            for charge, population in enumerate(ion.charge_populations_m3):
                bound = ion.Z - charge
                if population == 0.0 or bound == 0:
                    continue
                rr = population / cfg.ne_m3
                Ibar = ion.I_eV_by_charge[charge] / MEC2_EV
                qarg = p*np.sqrt(np.maximum(gamma - 1.0, 0.0))/Ibar
                h += rr*bound*(0.2*np.log1p(qarg**5) - beta2)
                y = (p*ion.a_bar_by_charge[charge])**1.5
                g += rr*((2.0/3.0)*(ion.Z**2 - charge**2)*np.log1p(y)
                         - (2.0/3.0)*bound**2*y/(1.0 + y))
            continue
        if ion.n_bound == 0:
            continue
        rr = ion.density_m3 / cfg.ne_m3
        Ibar = ion.I_eV / MEC2_EV
        qarg = p*np.sqrt(np.maximum(gamma - 1.0, 0.0))/Ibar
        h += rr*ion.n_bound*(0.2*np.log1p(qarg**5) - beta2)
        y = (p*ion.a_bar)**1.5
        g += rr*((2.0/3.0)*(ion.Z**2 - ion.Z0**2)*np.log1p(y)
                 - (2.0/3.0)*ion.n_bound**2*y/(1.0 + y))
    return h, g


def finite_temperature_relativistic_collision_coefficients(
    p: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics, q_coll: int
):
    p = np.asarray(p, dtype=np.float64)
    psis, psid, psis_dir, psid_dir, psis_ser, psid_ser, psw, k2s = collision_functions(
        p, phys.theta, q_coll, return_branches=True
    )
    gamma = np.sqrt(1.0 + p*p)
    ute = math.sqrt(2.0*phys.theta)
    k = 5.0
    ln_ee = phys.ln_lambda0 + np.log1p((2.0*(gamma - 1.0)/(ute*ute))**(k/2.0))/k
    ln_ei = phys.ln_lambda0 + np.log1p((2.0*p/ute)**k)/k
    h, g = screening_h_g(p, cfg, phys)
    tauc = collision_time(cfg.ne_m3, phys.ln_lambda0)

    nus = (gamma*gamma/p**3)/tauc/phys.ln_lambda0 * (
        ln_ee*(p*p/(gamma*gamma))*psis + h
    )
    nupar = (2.0*gamma*phys.theta/p**3)*psis/tauc
    nud = (gamma/p**3)/tauc/phys.ln_lambda0 * (
        ln_ee*(2.0*p/gamma)*psid + phys.z_eff*ln_ei + g
    )
    cf = p*phys.tau_ref_s*nus
    ca = 0.5*p*p*phys.tau_ref_s*nupar
    nudn = phys.tau_ref_s*nud
    return cf, ca, nudn, psis, psid, psis_dir, psid_dir, psis_ser, psid_ser, psw, k2s


def _matched_chandrasekhar_phi_psi(x: np.ndarray, return_branches: bool = False):
    """Stable Phi/Psi evaluation for the fully-relativistic matched TP model.

    Psi(x) = [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]/(2*x^2).  The direct
    expression loses precision as x->0, so a fixed Taylor branch is used.
    """
    x = np.asarray(x, dtype=np.float64)
    phi = sps.erf(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_dir = (phi - 2.0*x*np.exp(-x*x)/math.sqrt(math.pi))/(2.0*x*x)

    z = x*x
    poly = (((((-1.0/1560.0*z + 1.0/264.0)*z - 1.0/54.0)*z
               + 1.0/14.0)*z - 1.0/5.0)*z + 1.0/3.0)
    psi_ser = (2.0*x/math.sqrt(math.pi))*poly

    # The series is very accurate in this interval and comfortably overlaps
    # the direct expression in binary64.
    x_switch = 0.08
    low = np.abs(x) <= x_switch
    psi = np.where(low, psi_ser, psi_dir)
    if return_branches:
        return phi, psi, psi_dir, psi_ser, x_switch
    return phi, psi


def fully_relativistic_asymptotically_matched_collision_coefficients(
    p: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics
):
    """Fully-relativistic, asymptotically matched TP small-angle coefficients.

    This fully-relativistic momentum-space model is constructed by
    asymptotically matching the standard non-relativistic test-particle
    coefficients to the fully relativistic high-energy test-particle limit.
    It is historically the operator used in CODE, but CODE is provenance rather
    than the configuration name.  The formulas below define the complete-screening
    base operator; the same Hesslow partial-screening drag and pitch-deflection
    corrections used by the finite-temperature model are then added when bound
    electrons are present.  No field-particle term is included.
    """
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p*p)
    delta = math.sqrt(2.0*phys.theta)
    if delta <= 0.0:
        raise ValueError("positive thermal speed is required")
    x = p/(delta*gamma)
    phi, psi, psi_dir, psi_ser, x_switch = _matched_chandrasekhar_phi_psi(
        x, return_branches=True
    )
    chi_c = (cfg.ne_m3/phys.n_ref_m3)*(phys.ln_lambda0/phys.ln_lambda_ref)

    # Complete-screening base coefficients.
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = chi_c*2.0*psi/(delta*delta)
        ca = chi_c*(gamma/p)*psi
        nud = chi_c*(gamma/p**3)*(
            phys.z_eff + phi - psi + delta*delta*p*p/(2.0*gamma*gamma)
        )

    # Common Hesslow partial-screening overlay.  This uses the same h(p), g(p)
    # corrections as the finite-temperature relativistic base operator: drag
    # and pitch-angle deflection are modified, while C_A is unchanged.  For
    # fully stripped ions h=g=0 identically.
    h, g = screening_h_g(p, cfg, phys)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = cf + chi_c*(gamma*gamma/(p*p))*(h/phys.ln_lambda0)
        nud = nud + chi_c*(gamma/(p**3))*(g/phys.ln_lambda0)

    return cf, ca, nud, phi, psi, psi_dir, psi_ser, x_switch


def build_collision_data(cfg: SolverConfig, phys: DerivedPhysics, grid: Grid) -> CollisionData:
    # Centers: nu_D.  Radial faces: C_F and C_A.  At the pmin=0
    # zero-flux origin we deliberately do not evaluate the coordinate-singular
    # face coefficient.  For pmin>0 the lower face is regular and is evaluated
    # so an opt-in absorbing boundary can use the physical drift/diffusion.
    cf_f = np.zeros(grid.Np + 1, dtype=np.float64)
    ca_f = np.zeros(grid.Np + 1, dtype=np.float64)
    face_start = 1 if grid.p_faces[0] == 0.0 else 0
    face_eval = grid.p_faces[face_start:]

    if cfg.small_angle_model == "finite-temperature-relativistic":
        c = finite_temperature_relativistic_collision_coefficients(
            grid.p_centers, cfg, phys, cfg.q_coll
        )
        nud_c, psis_c, psid_c = c[2], c[3], c[4]
        f = finite_temperature_relativistic_collision_coefficients(
            face_eval, cfg, phys, cfg.q_coll
        )
        cf_f[face_start:] = f[0]
        ca_f[face_start:] = f[1]
        if not cfg.energy_diffusion:
            ca_f.fill(0.0)

        psw = c[9]
        ptest = np.geomspace(max(psw*0.7, np.finfo(float).eps), psw*1.3, 24)
        o = finite_temperature_relativistic_collision_coefficients(
            ptest, cfg, phys, max(cfg.q_coll, 64)
        )
        ds = np.max(np.abs(o[5]-o[7]) / np.maximum(np.abs(o[7]), 1e-300))
        dd = np.max(np.abs(o[6]-o[8]) / np.maximum(np.abs(o[8]), 1e-300))
        return CollisionData(
            cfg.small_angle_model, cf_f, ca_f, nud_c, psis_c, psid_c,
            psw, c[10], float(ds), float(dd)
        )

    if cfg.small_angle_model == "fully-relativistic-asymptotically-matched":
        c = fully_relativistic_asymptotically_matched_collision_coefficients(
            grid.p_centers, cfg, phys
        )
        nud_c = c[2]
        # Store Phi and Psi in the existing diagnostic arrays; they are labeled
        # explicitly in reporting/output for this configuration.
        phi_c, psi_c = c[3], c[4]
        f = fully_relativistic_asymptotically_matched_collision_coefficients(
            face_eval, cfg, phys
        )
        cf_f[face_start:] = f[0]
        ca_f[face_start:] = f[1]
        if not cfg.energy_diffusion:
            ca_f.fill(0.0)

        delta = math.sqrt(2.0*phys.theta)
        xsw = c[7]
        # x=p/(delta*gamma); invert exactly for x<1/delta.
        a = delta*xsw
        if a >= 1.0:
            raise ValueError(
                "temperature is outside the non-relativistic-bulk ordering used by the "
                "fully-relativistic asymptotically matched test-particle model"
            )
        psw = a/math.sqrt(max(1.0-a*a, np.finfo(float).tiny))
        x_test = np.geomspace(xsw*0.7, xsw*1.3, 24)
        _, _, d, s, _ = _matched_chandrasekhar_phi_psi(x_test, return_branches=True)
        overlap = float(np.max(np.abs(d-s)/np.maximum(np.abs(s), 1e-300)))
        return CollisionData(
            cfg.small_angle_model, cf_f, ca_f, nud_c, phi_c, psi_c,
            psw, 1.0, overlap, overlap
        )

    raise RuntimeError(f"unsupported small-angle model {cfg.small_angle_model!r}")


def validate_collision_data(coll: CollisionData) -> None:
    if coll.model == "finite-temperature-relativistic":
        arrays = {
            "C_F faces": coll.cf_faces,
            "C_A faces": coll.ca_faces,
            "nu_D centers": coll.nud_centers,
            "Psi_s centers": coll.psi_s_centers,
            "Psi_D centers": coll.psi_d_centers,
        }
    else:
        arrays = {
            "C_F faces": coll.cf_faces,
            "C_A faces": coll.ca_faces,
            "nu_D centers": coll.nud_centers,
            "Phi centers": coll.psi_s_centers,
            "Psi centers": coll.psi_d_centers,
        }
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            bad = int(np.flatnonzero(~np.isfinite(arr))[0])
            raise FloatingPointError(f"non-finite {name} at local index {bad}")
    for name, arr in arrays.items():
        if np.any(arr < 0.0):
            bad = int(np.flatnonzero(arr < 0.0)[0])
            raise FloatingPointError(f"negative {name} at local index {bad}: {arr[bad]:.6e}")
    if not math.isfinite(coll.k2_scaled) or coll.k2_scaled <= 0.0:
        raise FloatingPointError("collision normalization diagnostic must be finite and positive")
    if not math.isfinite(coll.p_switch) or coll.p_switch <= 0.0:
        raise FloatingPointError("low-p branch switch must be finite and positive")
    if not math.isfinite(coll.overlap_rel_psi_s) or not math.isfinite(coll.overlap_rel_psi_d):
        raise FloatingPointError("collision direct/series overlap diagnostic is non-finite")


# =============================================================================
# Chiu-Harvey finite-primary-energy knock-on gain
# =============================================================================

def ch_knockon_dsigma_bar_dp(primary_p: np.ndarray, secondary_p: np.ndarray) -> np.ndarray:
    """Normalized relativistic e-e knock-on differential cross section used by Chiu-Harvey."""
    pp, ps = np.broadcast_arrays(np.asarray(primary_p, dtype=np.float64), np.asarray(secondary_p, dtype=np.float64))
    gp = np.sqrt(1.0 + pp*pp)
    gs = np.sqrt(1.0 + ps*ps)
    beta = ps/gs
    ep = gp - 1.0
    es = gs - 1.0
    rem = gp - gs
    out = np.zeros_like(pp)
    good = (pp > 0.0) & (ps > 0.0) & (ep > 0.0) & (es > 0.0) & (rem > 0.0)
    if np.any(good):
        x = ep[good]**2/(es[good]*rem[good])
        bracket = x*x - 3.0*x + (ep[good]/gp[good])**2*(1.0 + x)
        out[good] = 2.0*math.pi*beta[good]*gp[good]**2 / (
            ep[good]**3*(gp[good] + 1.0)
        ) * bracket
    return out


def build_ch_geometry(cfg: SolverConfig, phys: DerivedPhysics, grid: Grid) -> CHGeometry:
    """Build the exact discrete Chiu-Harvey target map.

    This is algebraically identical to the former scalar (i,j) loop, but it
    evaluates target kinematics/coefficient arrays in bounded 2-D NumPy blocks.
    The ordering remains row-major, so active_rows, primary_radial, root_p, and
    row_coefficient define exactly the same first-order piecewise-constant CH
    operator used by the explicit CSR representation.

    The block cap avoids creating O(Np*Nxi) temporary host arrays on extreme
    grids while removing the expensive Python loop over every pitch cell.
    """
    if not cfg.include_ch or phys.nt_m3 == 0.0:
        z32 = np.empty(0, dtype=np.int32)
        z64 = np.empty(0, dtype=np.float64)
        return CHGeometry(z32, z32.copy(), z64, z64.copy())

    eps_m = math.sqrt(1.0 + cfg.p_m*cfg.p_m) - 1.0
    pref = phys.tau_ref_s*phys.nt_m3*const.c*R_E**2
    Nxi = grid.Nxi

    # Keep vectorized temporaries bounded to roughly one million target cells.
    # For very large Nxi this automatically reduces the radial block size.
    max_block_cells = 1_048_576
    block_rows = max(1, min(grid.Np, max_block_cells // max(1, Nxi)))

    xi = grid.xi_centers[None, :]
    xi2 = xi*xi
    xi_negative = xi < 0.0

    active_parts: list[np.ndarray] = []
    primary_parts: list[np.ndarray] = []
    coeff_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []

    for i0 in range(0, grid.Np, block_rows):
        i1 = min(grid.Np, i0 + block_rows)
        p = grid.p_centers[i0:i1, None]
        gamma = np.sqrt(1.0 + p*p)
        eps_s = gamma - 1.0

        den = 1.0 + xi2 - gamma*(1.0 - xi2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p0 = -2.0*p*xi/den

        mask = (
            (eps_s >= eps_m)
            & xi_negative
            & (den > 0.0)
            & np.isfinite(p0)
            & (p0 >= cfg.pmin)
            & (p0 < cfg.pmax)
        )

        gp = np.sqrt(1.0 + p0*p0)
        eps_p = gp - 1.0
        mask &= (eps_p > 0.0) & (eps_s <= 0.5*eps_p*(1.0 + 1.0e-14))

        # Convert the primary root to a radial cell index on the actual
        # [pmin,pmax] grid.  Always subtract pmin before converting the
        # physical primary momentum to a radial cell index.  Substituting pmin
        # for rejected roots prevents invalid casts without changing accepted targets.
        k_all = np.floor((np.where(mask, p0, cfg.pmin) - cfg.pmin)/grid.dp).astype(np.int64)
        mask &= (k_all >= 0) & (k_all < grid.Np)

        bi, bj = np.nonzero(mask)
        if bi.size == 0:
            continue

        target_i = i0 + bi
        p_target = p[bi, 0]
        p0v = p0[bi, bj]
        kval = k_all[bi, bj]
        xiv = grid.xi_centers[bj]

        ds = ch_knockon_dsigma_bar_dp(p0v, p_target)
        Vt = grid.radial_volume[target_i]*grid.dxi
        c = (
            Vt*pref/(p_target*p_target)
            * ds*(p0v**4)/np.abs(xiv)
            / grid.radial_volume[kval]
        )

        good = np.isfinite(ds) & (ds > 0.0) & np.isfinite(c) & (c > 0.0)
        if not np.all(good):
            target_i = target_i[good]
            bj = bj[good]
            p0v = p0v[good]
            kval = kval[good]
            c = c[good]

        active_parts.append(np.asarray(target_i*Nxi + bj, dtype=np.int32))
        primary_parts.append(np.asarray(kval, dtype=np.int32))
        coeff_parts.append(np.asarray(c, dtype=np.float64))
        root_parts.append(np.asarray(p0v, dtype=np.float64))

    if not active_parts:
        z32 = np.empty(0, dtype=np.int32)
        z64 = np.empty(0, dtype=np.float64)
        return CHGeometry(z32, z32.copy(), z64, z64.copy())

    return CHGeometry(
        np.ascontiguousarray(np.concatenate(active_parts)),
        np.ascontiguousarray(np.concatenate(primary_parts)),
        np.ascontiguousarray(np.concatenate(coeff_parts)),
        np.ascontiguousarray(np.concatenate(root_parts)),
    )


def validate_ch_geometry(cfg: SolverConfig, grid: Grid, ch: CHGeometry) -> None:
    n = ch.active_count
    if not (ch.primary_radial.size == n == ch.row_coefficient.size == ch.root_p.size):
        raise RuntimeError("inconsistent Chiu-Harvey geometry array lengths")
    if n == 0:
        return
    if np.any(ch.active_rows < 0) or np.any(ch.active_rows >= grid.size):
        raise RuntimeError("Chiu-Harvey target row outside represented state")
    if np.unique(ch.active_rows).size != n:
        raise RuntimeError("duplicate Chiu-Harvey target rows")
    if np.any(ch.primary_radial < 0) or np.any(ch.primary_radial >= grid.Np):
        raise RuntimeError("Chiu-Harvey primary radial index outside represented domain")
    if not np.all(np.isfinite(ch.root_p)) or np.any(ch.root_p <= 0.0) or np.any(ch.root_p >= cfg.pmax):
        raise RuntimeError("invalid Chiu-Harvey primary root")
    if not np.all(np.isfinite(ch.row_coefficient)) or np.any(ch.row_coefficient <= 0.0):
        raise RuntimeError("invalid Chiu-Harvey row coefficient")

    rows = ch.active_rows.astype(np.int64)
    ii = rows // grid.Nxi
    jj = rows - ii*grid.Nxi
    xi = grid.xi_centers[jj]
    ps = grid.p_centers[ii]
    if np.any(xi >= 0.0):
        raise RuntimeError("Chiu-Harvey support must lie at xi<0 in the aligned-primary reduction")
    kk = ch.primary_radial.astype(np.int64)
    tol = 32.0*np.finfo(np.float64).eps*max(1.0, cfg.pmax)
    if np.any(ch.root_p < grid.p_faces[kk]-tol) or np.any(ch.root_p >= grid.p_faces[kk+1]+tol):
        raise RuntimeError("Chiu-Harvey root does not lie in recorded primary radial cell")
    eps_m = math.sqrt(1.0 + cfg.p_m*cfg.p_m) - 1.0
    eps_s = np.sqrt(1.0 + ps*ps) - 1.0
    eps_p = np.sqrt(1.0 + ch.root_p*ch.root_p) - 1.0
    if np.any(eps_s < eps_m - 64.0*np.finfo(np.float64).eps):
        raise RuntimeError("Chiu-Harvey target violates the large-angle secondary cutoff")
    if np.any(eps_s > 0.5*eps_p*(1.0 + 1e-12)):
        raise RuntimeError("Chiu-Harvey target violates epsilon_s <= epsilon_p/2")


# =============================================================================






















# =============================================================================
# Fixed CSR graph: vectorized row pointers + GPU column/slot construction
# =============================================================================

def _build_local_csr_row_ptr_host(grid: Grid) -> tuple[np.ndarray, int]:
    """Build row pointers for the local five-point FV stencil in O(N)."""
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


def _build_augmented_csr_row_ptr_host(grid: Grid, ch: CHGeometry) -> tuple[np.ndarray, int]:
    """CSR row pointers for the exact auxiliary pitch-integral formulation.

    Physical kinetic rows retain only the local five-point FV stencil and, for
    an active CH target, one coupling to the auxiliary radial mass F_k.  The
    Np auxiliary constraint rows impose F_k - sum_j N[k,j] = 0.  Eliminating
    Eliminating the auxiliary mass therefore reproduces the direct discrete Chiu-Harvey map exactly while the
    graph contains O(Np*Nxi) rather than O(Np*Nxi^2) CH entries.
    """
    n = grid.size
    Np = grid.Np
    Nxi = grid.Nxi

    counts_phys = np.full(n, 5, dtype=np.int32)
    counts_phys[:Nxi] -= 1
    counts_phys[(Np-1)*Nxi:] -= 1
    counts_phys[0:n:Nxi] -= 1
    counts_phys[Nxi-1:n:Nxi] -= 1
    if ch.active_count:
        counts_phys[ch.active_rows.astype(np.int64, copy=False)] += 1

    counts_aux = np.full(Np, Nxi + 1, dtype=np.int32)
    counts = np.concatenate((counts_phys, counts_aux))
    if np.any(counts <= 0):
        raise RuntimeError("non-positive augmented CSR row length")

    row_ptr64 = np.empty(n + Np + 1, dtype=np.int64)
    row_ptr64[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_ptr64[1:])
    nnz = int(row_ptr64[-1])
    if nnz > np.iinfo(np.int32).max:
        raise ValueError("augmented CSR nnz exceeds int32 range required by the qualified cuDSS path")
    return np.ascontiguousarray(row_ptr64.astype(np.int32)), nnz








# =============================================================================
# Initial condition and diagnostics
# =============================================================================

def _cell_gauss_nodes(faces: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(q)
    lo = faces[:-1, None]
    hi = faces[1:, None]
    nodes = 0.5*(hi-lo)*x[None, :] + 0.5*(hi+lo)
    weights = 0.5*(hi-lo)*w[None, :]
    return nodes, weights


def project_initial(cfg: SolverConfig, phys: DerivedPhysics, grid: Grid) -> np.ndarray:
    # Baseline and optional seed shapes are separable, so exact cell-content
    # projection requires only independent 1-D Gauss-Legendre integrals.
    # The seed is mixed by particle content: seed_fraction of the requested
    # initial density is assigned to the seed and the remainder to the baseline.
    # Thus a seed does not change total represented density.
    pp, wpq = _cell_gauss_nodes(grid.p_faces, cfg.q_init)

    if cfg.init == "maxwell-juttner":
        ti = cfg.te_eV if cfg.init_te_eV is None else cfg.init_te_eV
        if not math.isfinite(ti) or ti <= 0.0:
            raise ValueError("init_te_eV must be finite and positive")
        theta = ti/MEC2_EV
        shape_p = np.exp(-(np.sqrt(1.0 + pp*pp)-1.0)/theta)
        radial = 2.0*math.pi*np.sum(wpq*pp*pp*shape_p, axis=1)
        pitch = np.full(grid.Nxi, grid.dxi, dtype=np.float64)
    elif cfg.init == "gaussian":
        if (not math.isfinite(cfg.gaussian_sigma_p) or cfg.gaussian_sigma_p <= 0.0 or
                not math.isfinite(cfg.gaussian_sigma_xi) or cfg.gaussian_sigma_xi <= 0.0):
            raise ValueError("Gaussian widths must be finite and positive")
        shape_p = np.exp(-0.5*((pp-cfg.gaussian_p0)/cfg.gaussian_sigma_p)**2)
        radial = 2.0*math.pi*np.sum(wpq*pp*pp*shape_p, axis=1)
        xx, wxq = _cell_gauss_nodes(grid.xi_faces, cfg.q_init)
        shape_x = np.exp(-0.5*((xx-cfg.gaussian_xi0)/cfg.gaussian_sigma_xi)**2)
        pitch = np.sum(wxq*shape_x, axis=1)
    else:
        raise ValueError(f"unsupported initial condition {cfg.init!r}")

    out = radial[:, None]*pitch[None, :]
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise FloatingPointError("initial projection produced invalid baseline cell contents")

    target = cfg.ne_m3 if cfg.init_density_m3 is None else cfg.init_density_m3
    target_ratio = target/phys.n_ref_m3
    base_total = float(out.sum())
    base_mass = (1.0-cfg.seed_fraction)*target_ratio
    if base_mass > 0.0 and base_total <= 0.0:
        raise ValueError("baseline initial shape has zero projected mass")
    if base_total > 0.0:
        out *= base_mass/base_total
    else:
        out.fill(0.0)

    if cfg.seed_fraction > 0.0 and target_ratio > 0.0:
        seed_shape_p = np.exp(-0.5*((pp-cfg.seed_p0)/cfg.seed_sigma_p)**2)
        seed_radial = 2.0*math.pi*np.sum(wpq*pp*pp*seed_shape_p, axis=1)
        xx, wxq = _cell_gauss_nodes(grid.xi_faces, cfg.q_init)
        seed_shape_x = np.exp(-0.5*((xx-cfg.seed_xi0)/cfg.seed_sigma_xi)**2)
        seed_pitch = np.sum(wxq*seed_shape_x, axis=1)
        seed = seed_radial[:, None]*seed_pitch[None, :]
        seed_total = float(seed.sum())
        if seed_total <= 0.0 or not np.all(np.isfinite(seed)) or np.any(seed < 0.0):
            raise FloatingPointError("runaway-seed projection produced invalid/zero cell contents")
        seed *= (cfg.seed_fraction*target_ratio)/seed_total
        out += seed

    total = float(out.sum())
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise FloatingPointError("combined initial projection produced invalid cell contents")
    if not math.isclose(total, target_ratio, rel_tol=5.0e-14, abs_tol=5.0e-15):
        raise RuntimeError(
            f"initial density bookkeeping failed: got {total:.17e}, expected {target_ratio:.17e}"
        )
    return np.ascontiguousarray(out.ravel())


def diagnostics(N: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics, grid: Grid) -> dict[str, float]:
    NN = np.asarray(N, dtype=np.float64).reshape(grid.Np, grid.Nxi)
    ff = NN/(grid.radial_volume[:, None]*grid.dxi)
    dens = float(NN.sum())
    current = -const.e*phys.n_ref_m3*const.c*float(np.sum(
        ff*grid.current_radial_weight[:, None]*grid.current_pitch_weight[None, :]
    ))
    return {
        "density_ratio": dens,
        "density_m3": dens*phys.n_ref_m3,
        "current_A_m2": current,
        "min_N": float(NN.min()),
        "max_f": float(ff.max()),
    }


def state_change_norms(N: np.ndarray, N0: np.ndarray) -> dict[str, float]:
    a = np.asarray(N, dtype=np.float64)
    b = np.asarray(N0, dtype=np.float64)
    d = a-b
    l1 = float(np.sum(np.abs(d))/max(np.sum(np.abs(b)), 1e-300))
    l2 = float(np.linalg.norm(d)/max(np.linalg.norm(b), 1e-300))
    linf = float(np.max(np.abs(d))/max(np.max(np.abs(b)), 1e-300))
    return {"rel_l1": l1, "rel_l2": l2, "rel_linf": linf}


def particle_balance_terms(
    N: np.ndarray, cfg: SolverConfig, phys: DerivedPhysics,
    grid: Grid, coll: CollisionData, ch: CHGeometry | None = None,
) -> dict[str, float]:
    NN = np.asarray(N, dtype=np.float64).reshape(grid.Np, grid.Nxi)
    xi = grid.xi_centers
    inner = 0.0
    if cfg.inner_boundary == "absorbing":
        pf0 = grid.p_faces[0]
        gam0 = math.sqrt(1.0 + pf0*pf0)
        A0 = -phys.Ebar*xi - phys.alpha*gam0*pf0*(1.0-xi*xi) - coll.cf_faces[0]
        D0 = coll.ca_faces[0]
        f_first = NN[0, :]/(grid.radial_volume[0]*grid.dxi)
        sink_speed = np.maximum(-A0, 0.0) + 2.0*D0/grid.dp
        inner = float(np.sum(2.0*math.pi*pf0*pf0*grid.dxi*sink_speed*f_first))

    pf = grid.p_faces[-1]
    gam = math.sqrt(1.0 + pf*pf)
    A = -phys.Ebar*xi - phys.alpha*gam*pf*(1.0-xi*xi) - coll.cf_faces[-1]
    f_last = NN[-1, :]/(grid.radial_volume[-1]*grid.dxi)
    outer = float(np.sum(2.0*math.pi*pf*pf*grid.dxi*np.maximum(A, 0.0)*f_last))

    if cfg.large_angle_model == "chiu-harvey" and ch is not None and ch.active_count:
        radial_mass = np.sum(NN, axis=1)
        prod = float(np.sum(ch.row_coefficient*radial_mass[ch.primary_radial]))
    else:
        prod = 0.0
    return {
        "inner_outflow_rate": inner,
        "outer_outflow_rate": outer,
        "large_angle_production_rate": prod,
        "ch_production_rate": prod if cfg.large_angle_model == "chiu-harvey" else 0.0,
        "net_particle_rate": -inner - outer + prod,
    }



def build_balance_test_state(grid: Grid) -> np.ndarray:
    p = grid.p_centers[:, None]
    xi = grid.xi_centers[None, :]
    f = 1.0 + 0.20*np.sin(0.7*p) + 0.15*xi + 0.05*xi*xi
    if np.any(f <= 0.0):
        raise RuntimeError("internal finite-volume balance test state is not positive")
    N = grid.radial_volume[:, None]*grid.dxi*f
    N /= np.sum(N)
    return np.ascontiguousarray(N.ravel())


def nsteps_from_cfg(cfg: SolverConfig) -> int:
    if cfg.adaptive:
        return int(math.ceil(cfg.tau_end/cfg.dtau)) if cfg.tau_end > 0.0 else 0
    n = int(round(cfg.tau_end/cfg.dtau))
    if not math.isclose(n*cfg.dtau, cfg.tau_end, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("tau_end must be an integer multiple of dtau when adaptive=false")
    return n


def should_record_intermediate(step: int, nsteps: int, every: int) -> bool:
    """True only for requested interior samples; endpoints are handled once."""
    return every > 0 and 0 < step < nsteps and step % every == 0



def _fmt_bytes(nbytes: int) -> str:
    x = float(nbytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.3f} {u}"
        x /= 1024.0
    return f"{x:.3f} TiB"


def cc_bernoulli(x: float) -> float:
    """Stable Chang--Cooper Bernoulli function B(x)=x/(exp(x)-1)."""
    if abs(x) < 1.0e-6:
        x2 = x*x
        return 1.0 - 0.5*x + x2/12.0 - x2*x2/720.0
    if x > 50.0:
        return 0.0
    if x < -50.0:
        return -x
    return x/math.expm1(x)


def explicit_storage_estimate(
    cfg: SolverConfig, grid: Grid, topo, ch: CHGeometry | None = None,
) -> dict[str, int]:
    """Known storage owned by this script, excluding opaque cuDSS internals."""
    nsys = int(topo.n)
    nphys = int(topo.physical_n)
    nnz = int(topo.nnz)
    na = int(ch.active_count) if ch is not None else 0

    topology = 4*((nsys + 1) + nnz + 5*nphys)
    if topo.formulation == "augmented-ch":
        topology += 4*(na + 2*grid.Np)
    values = 8*nnz
    # RHS/solution, three retained TR--BDF2 states, three CSR products, and
    # one residual vector, plus the host-projected physical initial state.
    vectors = 8*(9*nsys + nphys)
    coeff_elems = nphys + 6*grid.Np + 2*grid.Nxi + 4 + na
    coefficients = 8*coeff_elems
    explicit_gpu = topology + values + vectors + coefficients

    raw_csr = 4*(nsys + 1) + 12*nnz
    one_state = 8*nphys
    balance_gpu_transient = 16*nsys if cfg.balance_check else 0
    nsteps = nsteps_from_cfg(cfg)
    nsnap = ((nsteps - 1)//cfg.save_every
             if cfg.save_every > 0 and nsteps > 1 else 0)
    snapshot_host = nsnap*one_state
    endpoint_host = (2*one_state if nsteps > 0 else one_state) + one_state
    return {
        "raw_csr": raw_csr, "topology": topology, "values": values,
        "vectors": vectors, "coefficients": coefficients,
        "explicit_gpu": explicit_gpu, "balance_gpu_transient": balance_gpu_transient,
        "one_state": one_state, "endpoint_host": endpoint_host,
        "snapshot_host": snapshot_host, "snapshot_count": nsnap,
        "system_unknowns": nsys, "physical_unknowns": nphys,
    }



def print_storage_preflight(
    cfg: SolverConfig, grid: Grid, topo, ch: CHGeometry | None = None,
) -> dict[str, int]:
    m = explicit_storage_estimate(cfg, grid, topo, ch=ch)
    print("=== storage preflight (known allocations; cuDSS internals excluded) ===")
    if m["system_unknowns"] != m["physical_unknowns"]:
        print(f"algebraic unknowns={m['system_unknowns']:,}  physical={m['physical_unknowns']:,}  auxiliary={m['system_unknowns']-m['physical_unknowns']:,}")
    print(f"one physical FP64 state={_fmt_bytes(m['one_state'])}  raw CSR(row/col/values)={_fmt_bytes(m['raw_csr'])}")
    print(
        "explicit persistent GPU=" + _fmt_bytes(m["explicit_gpu"])
        + f"  [topology={_fmt_bytes(m['topology'])}, values={_fmt_bytes(m['values'])}, "
          f"state/work vectors={_fmt_bytes(m['vectors'])}, coeffs={_fmt_bytes(m['coefficients'])}]"
    )
    if cfg.balance_check:
        print(f"FV balance transient GPU≈{_fmt_bytes(m['balance_gpu_transient'])} (plus host test work arrays)")
    print(f"host initial/final + cell-volume≈{_fmt_bytes(m['endpoint_host'])}")
    if m["snapshot_count"]:
        print(f"requested interior snapshots={m['snapshot_count']:,} -> retained host state≈{_fmt_bytes(m['snapshot_host'])}")
    print("NOTE: cuDSS symbolic/factorization workspace and fill are not predictable from raw CSR size and can dominate for augmented Chiu-Harvey systems.")
    return m



# =============================================================================
# Warp kernels: finite-volume assembly, exact augmented CH coupling, time stepping, diagnostics
# =============================================================================

if wp is not None:

    @wp.func
    def cc_bernoulli_wp(x: wp.float64) -> wp.float64:
        """Stable GPU Bernoulli function for Chang--Cooper face fluxes."""
        ax = wp.abs(x)
        if ax < wp.float64(1.0e-6):
            x2 = x*x
            return wp.float64(1.0) - wp.float64(0.5)*x + x2/wp.float64(12.0) - x2*x2/wp.float64(720.0)
        if x > wp.float64(50.0):
            return wp.float64(0.0)
        if x < wp.float64(-50.0):
            return -x
        return x/(wp.exp(x)-wp.float64(1.0))

    @wp.kernel
    def mark_ch_rows_kernel(
        active_rows: wp.array(dtype=wp.int32),
        q_plus_one_by_row: wp.array(dtype=wp.int32),
    ):
        q = wp.tid()
        q_plus_one_by_row[active_rows[q]] = q + 1





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
    def fill_augmented_csr_topology_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        q_plus_one_by_row: wp.array(dtype=wp.int32),
        ch_primary_radial: wp.array(dtype=wp.int32),
        physical_n: int,
        Np: int,
        Nxi: int,
        col_ind: wp.array(dtype=wp.int32),
        diag_slot: wp.array(dtype=wp.int32),
        pm_slot: wp.array(dtype=wp.int32),
        pp_slot: wp.array(dtype=wp.int32),
        xm_slot: wp.array(dtype=wp.int32),
        xp_slot: wp.array(dtype=wp.int32),
        ch_aux_slot: wp.array(dtype=wp.int32),
        aux_n_start_slot: wp.array(dtype=wp.int32),
        aux_diag_slot: wp.array(dtype=wp.int32),
    ):
        row = wp.tid()
        if row < physical_n:
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
                pos += 1
            qtag = q_plus_one_by_row[row]
            if qtag != 0:
                q = qtag - 1
                col_ind[pos] = physical_n + ch_primary_radial[q]
                ch_aux_slot[q] = pos
        else:
            k = row - physical_n
            base = row_ptr[row]
            aux_n_start_slot[k] = base
            c0 = k*Nxi
            for j in range(Nxi):
                col_ind[base+j] = c0+j
            ds = base + Nxi
            col_ind[ds] = physical_n+k
            aux_diag_slot[k] = ds


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
        # Keep this validation kernel deliberately free of loop-carried mutable
        # integer state.  Warp 1.17 treats Python integer literals as compile-time
        # constants unless explicitly materialized, so constructs such as
        # ``bad = 0; bad = 1`` or ``prev = -1; prev = c`` inside a dynamic loop
        # trigger WarpCodegenError.  Compare adjacent CSR entries directly and
        # atomically raise the shared failure flag instead.
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
    def validate_csr_augmented_ch_kernel(
        active_rows: wp.array(dtype=wp.int32),
        ch_primary_radial: wp.array(dtype=wp.int32),
        ch_aux_slot: wp.array(dtype=wp.int32),
        row_ptr: wp.array(dtype=wp.int32),
        col_ind: wp.array(dtype=wp.int32),
        physical_n: int,
        flag: wp.array(dtype=wp.int32),
    ):
        q = wp.tid()
        row = active_rows[q]
        s = ch_aux_slot[q]
        if s < row_ptr[row] or s >= row_ptr[row+1]:
            wp.atomic_max(flag, 0, 1)
        elif col_ind[s] != physical_n + ch_primary_radial[q]:
            wp.atomic_max(flag, 0, 1)


    @wp.kernel
    def validate_csr_augmented_aux_kernel(
        row_ptr: wp.array(dtype=wp.int32),
        col_ind: wp.array(dtype=wp.int32),
        aux_n_start_slot: wp.array(dtype=wp.int32),
        aux_diag_slot: wp.array(dtype=wp.int32),
        physical_n: int,
        Nxi: int,
        flag: wp.array(dtype=wp.int32),
    ):
        k = wp.tid()
        row = physical_n+k
        base = row_ptr[row]
        hi = row_ptr[row+1]
        if hi-base != Nxi+1:
            wp.atomic_max(flag, 0, 1)
        if aux_n_start_slot[k] != base:
            wp.atomic_max(flag, 0, 1)
        for j in range(Nxi):
            if col_ind[base+j] != k*Nxi+j:
                wp.atomic_max(flag, 0, 1)
        ds = aux_diag_slot[k]
        if ds != base+Nxi or ds >= hi:
            wp.atomic_max(flag, 0, 1)
        elif col_ind[ds] != physical_n+k:
            wp.atomic_max(flag, 0, 1)





    @wp.kernel
    def assemble_trbdf2_matrix_kernel(
        p_faces: wp.array(dtype=wp.float64),
        p_centers: wp.array(dtype=wp.float64),
        xi_faces: wp.array(dtype=wp.float64),
        xi_centers: wp.array(dtype=wp.float64),
        radial_volume: wp.array(dtype=wp.float64),
        cell_volume: wp.array(dtype=wp.float64),
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
        dp: wp.float64,
        dxi: wp.float64,
        Ebar: wp.float64,
        alpha: wp.float64,
        dt: wp.float64,
        alpha0: wp.float64,
        inner_boundary_mode: int,
        values: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        i = row // Nxi
        j = row - i*Nxi
        xi = xi_centers[j]
        one_minus = wp.float64(1.0) - xi*xi
        self_L = wp.float64(0.0)

        # Left radial face: +F_left.  The default i=0 behavior remains exact
        # zero flux.  In opt-in absorbing mode the exterior distribution is
        # zero: positive A would be inflow and is suppressed, while negative A
        # removes particles.  If radial diffusion is enabled, f=0 is imposed at
        # the face with the center-to-face distance dp/2.
        if i > 0:
            pf = p_faces[i]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[i]
            D = ca_faces[i]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi
            cL = wp.float64(0.0)
            cR = wp.float64(0.0)
            if D > wp.float64(0.0):
                w = A*dp/D
                cL = fac*(D/dp)*cc_bernoulli_wp(-w)/cell_volume[row-Nxi]
                cR = -fac*(D/dp)*cc_bernoulli_wp(w)/cell_volume[row]
            elif A >= wp.float64(0.0):
                cL = fac*A/cell_volume[row-Nxi]
            else:
                cR = fac*A/cell_volume[row]
            values[pm_slot[row]] = -dt*cL
            self_L += cR
        elif inner_boundary_mode == 1:
            pf = p_faces[0]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[0]
            D = ca_faces[0]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi
            advR = wp.float64(0.0)
            if A < wp.float64(0.0):
                advR = A
            self_L += fac*(advR - wp.float64(2.0)*D/dp)/cell_volume[row]

        # Right radial face: -F_right.
        if i + 1 < Np:
            pf = p_faces[i+1]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[i+1]
            D = ca_faces[i+1]
            fac = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi
            cL = wp.float64(0.0)
            cR = wp.float64(0.0)
            if D > wp.float64(0.0):
                w = A*dp/D
                cL = fac*(D/dp)*cc_bernoulli_wp(-w)/cell_volume[row]
                cR = -fac*(D/dp)*cc_bernoulli_wp(w)/cell_volume[row+Nxi]
            elif A >= wp.float64(0.0):
                cL = fac*A/cell_volume[row]
            else:
                cR = fac*A/cell_volume[row+Nxi]
            self_L -= cL
            values[pp_slot[row]] = dt*cR
        else:
            pf = p_faces[Np]
            gam = wp.sqrt(wp.float64(1.0) + pf*pf)
            A = -Ebar*xi - alpha*gam*pf*one_minus - cf_faces[Np]
            if A > wp.float64(0.0):
                c = wp.float64(2.0)*wp.float64(3.14159265358979323846)*pf*pf*dxi*A/cell_volume[row]
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
            cL = wp.float64(0.0)
            cR = wp.float64(0.0)
            if D > wp.float64(0.0):
                w = A*dxi/D
                cL = rv*(D/dxi)*cc_bernoulli_wp(-w)/cell_volume[row-1]
                cR = -rv*(D/dxi)*cc_bernoulli_wp(w)/cell_volume[row]
            elif A >= wp.float64(0.0):
                cL = rv*A/cell_volume[row-1]
            else:
                cR = rv*A/cell_volume[row]
            values[xm_slot[row]] = -dt*cL
            self_L += cR

        # Upper pitch face: -F_upper.
        if j + 1 < Nxi:
            xf = xi_faces[j+1]
            om = wp.float64(1.0) - xf*xf
            A = om*(-Ebar/p + alpha*xf/gamc)
            D = wp.float64(0.5)*nud_centers[i]*om
            cL = wp.float64(0.0)
            cR = wp.float64(0.0)
            if D > wp.float64(0.0):
                w = A*dxi/D
                cL = rv*(D/dxi)*cc_bernoulli_wp(-w)/cell_volume[row]
                cR = -rv*(D/dxi)*cc_bernoulli_wp(w)/cell_volume[row+1]
            elif A >= wp.float64(0.0):
                cL = rv*A/cell_volume[row]
            else:
                cR = rv*A/cell_volume[row+1]
            self_L -= cL
            values[xp_slot[row]] = dt*cR

        values[diag_slot[row]] = alpha0 - dt*self_L














    @wp.kernel
    def add_ch_augmented_kernel(
        row_coefficient: wp.array(dtype=wp.float64),
        ch_aux_slot: wp.array(dtype=wp.int32),
        dt: wp.float64,
        values: wp.array(dtype=wp.float64),
    ):
        q = wp.tid()
        values[ch_aux_slot[q]] = -dt*row_coefficient[q]


    @wp.kernel
    def fill_aux_constraint_n_kernel(
        aux_n_start_slot: wp.array(dtype=wp.int32),
        Nxi: int,
        values: wp.array(dtype=wp.float64),
    ):
        q = wp.tid()
        k = q // Nxi
        j = q - k*Nxi
        values[aux_n_start_slot[k] + j] = wp.float64(-1.0)


    @wp.kernel
    def fill_aux_constraint_diag_kernel(
        aux_diag_slot: wp.array(dtype=wp.int32),
        values: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        values[aux_diag_slot[k]] = wp.float64(1.0)


    @wp.kernel
    def copy_prefix_kernel(
        src: wp.array(dtype=wp.float64),
        dst: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        dst[i] = src[i]


    @wp.kernel
    def fill_aux_state_kernel(
        physical: wp.array(dtype=wp.float64),
        physical_n: int,
        Nxi: int,
        full_state: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        acc = wp.float64(0.0)
        base = k*Nxi
        for j in range(Nxi):
            acc += physical[base+j]
        full_state[physical_n+k] = acc


    @wp.kernel
    def build_trbdf2_stage1_rhs_kernel(
        state: wp.array(dtype=wp.float64),
        applied: wp.array(dtype=wp.float64),
        physical_n: int,
        rhs: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        if i < physical_n:
            rhs[i] = wp.float64(2.0)*state[i] - applied[i]
        else:
            rhs[i] = wp.float64(0.0)


    @wp.kernel
    def build_trbdf2_stage2_rhs_kernel(
        state_n: wp.array(dtype=wp.float64),
        applied_n: wp.array(dtype=wp.float64),
        state_gamma: wp.array(dtype=wp.float64),
        applied_gamma: wp.array(dtype=wp.float64),
        physical_n: int,
        inv_d: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        if i < physical_n:
            rhs[i] = state_n[i] + inv_d*(
                wp.float64(0.35355339059327376220)*(state_n[i] - applied_n[i])
                + wp.float64(0.35355339059327376220)*(state_gamma[i] - applied_gamma[i])
            )
        else:
            rhs[i] = wp.float64(0.0)


    @wp.kernel
    def build_trbdf2_error_rhs_kernel(
        state_n: wp.array(dtype=wp.float64),
        applied_n: wp.array(dtype=wp.float64),
        state_gamma: wp.array(dtype=wp.float64),
        applied_gamma: wp.array(dtype=wp.float64),
        state_one: wp.array(dtype=wp.float64),
        applied_one: wp.array(dtype=wp.float64),
        physical_n: int,
        inv_d: wp.float64,
        rhs: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        if i < physical_n:
            rhs[i] = state_n[i] + inv_d*(
                wp.float64(0.21548220313557546505)*(state_n[i] - applied_n[i])
                + wp.float64(0.68688672392660700436)*(state_gamma[i] - applied_gamma[i])
                + wp.float64(0.09763107293781749796)*(state_one[i] - applied_one[i])
            ) - state_one[i]
        else:
            rhs[i] = wp.float64(0.0)


    @wp.kernel
    def copy_all_kernel(
        src: wp.array(dtype=wp.float64),
        dst: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        dst[i] = src[i]


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


    @wp.kernel
    def monitor_state_kernel(
        x: wp.array(dtype=wp.float64),
        flag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        v = x[i]
        # flag=1: at least one negative cell content occurred;
        # flag=2: NaN/Inf-like magnitude occurred (takes precedence).
        if v != v or wp.abs(v) > wp.float64(1.0e300):
            wp.atomic_max(flag, 0, 2)
        elif v < wp.float64(0.0):
            wp.atomic_max(flag, 0, 1)


def build_local_csr_topology_gpu(grid: Grid, device: str) -> tuple[CSRTopology, dict[str, float]]:
    """Construct and validate the local five-point FV CSR graph on CUDA."""
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
        fill_local_csr_topology_kernel, dim=grid.size,
        inputs=[row_ptr, grid.Np, grid.Nxi],
        outputs=[col_ind, diag, pm, pp, xm, xp], device=device,
    )
    wp.synchronize_stream(stream)
    fill_gpu_s = time.perf_counter() - t1

    flag = wp.zeros(1, dtype=wp.int32, device=device)
    t2 = time.perf_counter()
    wp.launch(
        validate_csr_local_kernel, dim=grid.size,
        inputs=[row_ptr, col_ind, diag, pm, pp, xm, xp,
                grid.size, grid.Np, grid.Nxi, flag], device=device,
    )
    wp.synchronize_stream(stream)
    validate_gpu_s = time.perf_counter() - t2
    if int(np.asarray(flag.numpy(), dtype=np.int32)[0]) != 0:
        raise RuntimeError("GPU local CSR topology validation failed")

    topo = CSRTopology(
        grid.size, nnz, row_ptr, col_ind, diag, pm, pp, xm, xp,
        grid.size, "local",
    )
    timing = {
        "rowptr_host_s": rowptr_host_s,
        "fill_gpu_s": fill_gpu_s,
        "validate_gpu_s": validate_gpu_s,
        "total_s": time.perf_counter() - t0,
    }
    return topo, timing


def build_augmented_ch_csr_topology_gpu(
    grid: Grid, ch: CHGeometry, device: str,
) -> tuple[CSRTopology, dict[str, float]]:
    """Construct the exact O(Np*Nxi) auxiliary Chiu-Harvey CSR graph on CUDA."""
    if wp is None:
        raise RuntimeError("GPU topology construction requires warp-lang")
    if ch.active_count == 0:
        return build_local_csr_topology_gpu(grid, device)

    t0 = time.perf_counter()
    row_ptr_host, nnz = _build_augmented_csr_row_ptr_host(grid, ch)
    rowptr_host_s = time.perf_counter() - t0
    physical_n = grid.size
    system_n = physical_n + grid.Np

    dev_obj = wp.get_device(device)
    stream = wp.get_stream(dev_obj)
    row_ptr = wp.array(row_ptr_host, dtype=wp.int32, device=device)
    col_ind = wp.zeros(nnz, dtype=wp.int32, device=device)
    diag = wp.zeros(physical_n, dtype=wp.int32, device=device)
    pm = wp.zeros(physical_n, dtype=wp.int32, device=device)
    pp = wp.zeros(physical_n, dtype=wp.int32, device=device)
    xm = wp.zeros(physical_n, dtype=wp.int32, device=device)
    xp = wp.zeros(physical_n, dtype=wp.int32, device=device)
    q_by_row = wp.zeros(physical_n, dtype=wp.int32, device=device)
    ch_aux = wp.zeros(ch.active_count, dtype=wp.int32, device=device)
    aux_n_start = wp.zeros(grid.Np, dtype=wp.int32, device=device)
    aux_diag = wp.zeros(grid.Np, dtype=wp.int32, device=device)
    active = wp.array(np.ascontiguousarray(ch.active_rows), dtype=wp.int32, device=device)
    primary = wp.array(np.ascontiguousarray(ch.primary_radial), dtype=wp.int32, device=device)

    t1 = time.perf_counter()
    wp.launch(mark_ch_rows_kernel, dim=ch.active_count, inputs=[active, q_by_row], device=device)
    wp.launch(
        fill_augmented_csr_topology_kernel, dim=system_n,
        inputs=[row_ptr, q_by_row, primary, physical_n, grid.Np, grid.Nxi],
        outputs=[col_ind, diag, pm, pp, xm, xp, ch_aux, aux_n_start, aux_diag],
        device=device,
    )
    wp.synchronize_stream(stream)
    fill_gpu_s = time.perf_counter() - t1

    flag = wp.zeros(1, dtype=wp.int32, device=device)
    t2 = time.perf_counter()
    wp.launch(
        validate_csr_local_kernel, dim=physical_n,
        inputs=[row_ptr, col_ind, diag, pm, pp, xm, xp,
                system_n, grid.Np, grid.Nxi, flag], device=device,
    )
    wp.launch(
        validate_csr_augmented_ch_kernel, dim=ch.active_count,
        inputs=[active, primary, ch_aux, row_ptr, col_ind, physical_n, flag], device=device,
    )
    wp.launch(
        validate_csr_augmented_aux_kernel, dim=grid.Np,
        inputs=[row_ptr, col_ind, aux_n_start, aux_diag, physical_n, grid.Nxi, flag],
        device=device,
    )
    wp.synchronize_stream(stream)
    validate_gpu_s = time.perf_counter() - t2
    if int(np.asarray(flag.numpy(), dtype=np.int32)[0]) != 0:
        raise RuntimeError("GPU augmented Chiu-Harvey CSR topology validation failed")

    topo = CSRTopology(
        system_n, nnz, row_ptr, col_ind, diag, pm, pp, xm, xp,
        physical_n, "augmented-ch", ch_aux, aux_n_start, aux_diag,
    )
    timing = {
        "rowptr_host_s": rowptr_host_s,
        "fill_gpu_s": fill_gpu_s,
        "validate_gpu_s": validate_gpu_s,
        "total_s": time.perf_counter() - t0,
    }
    return topo, timing






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
        # Surface asynchronous cuDSS errors when this binding exposes INFO.
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
                # Older bindings do not expose the same INFO query; cudss.execute
                # itself still raises synchronous API errors.
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

    def refactorize(self) -> float:
        """Refactor changed matrix values while reusing analyzed structure."""
        if not self.analyzed:
            raise RuntimeError("cuDSS analysis must precede refactorization")
        phase = getattr(self.cudss.Phase, "REFACTORIZATION", None)
        if phase is None:
            raise RuntimeError("installed cuDSS bindings do not expose refactorization")
        return self._execute(phase)

    def solve(self) -> float:
        if not self.analyzed:
            raise RuntimeError("cuDSS analysis must precede solve")
        return self._execute(self.cudss.Phase.SOLVE)

    def _destroy_resources(self, *, synchronize: bool) -> None:
        if self.closed:
            return
        if synchronize and self.stream is not None:
            try:
                wp.synchronize_stream(self.stream)
            except Exception:
                pass
        # Destroy in reverse construction order.  Cleanup should never mask the
        # original exception, so each call is individually guarded.
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
    nr = np.linalg.norm(rr); nb = np.linalg.norm(bb)
    return float(nr/nb if nb else nr)





def gpu_fv_balance_check(
    cfg: SolverConfig, phys: DerivedPhysics, grid: Grid, coll: CollisionData,
    topo, row_ptr, col_ind, values, device: str, matrix_dt: float,
    ch: CHGeometry | None = None,
) -> dict[str, float]:
    test = build_balance_test_state(grid)
    if topo.n > grid.size:
        z = np.zeros(topo.n, dtype=np.float64)
        z[:grid.size] = test
        z[grid.size:] = np.sum(test.reshape(grid.Np, grid.Nxi), axis=1)
    else:
        z = test
    x = wp.array(z, dtype=wp.float64, device=device)
    y = wp.zeros(topo.n, dtype=wp.float64, device=device)
    wp.launch(csr_matvec_kernel, dim=topo.n,
              inputs=[row_ptr, col_ind, values, x], outputs=[y], device=device)
    Ay_all = np.asarray(y.numpy(), dtype=np.float64)
    Ay = Ay_all[:grid.size]
    LN = (test-Ay)/matrix_dt
    lhs = float(np.sum(LN))
    terms = particle_balance_terms(test, cfg, phys, grid, coll, ch=ch)
    rhs = float(terms["net_particle_rate"])
    abs_defect = abs(lhs-rhs)
    scale = max(float(np.sum(np.abs(LN))),
                abs(float(terms["inner_outflow_rate"])) + abs(float(terms["outer_outflow_rate"]))
                + abs(float(terms["large_angle_production_rate"])),
                1e-300)
    rel_defect = abs_defect/scale
    aux_defect = (float(np.max(np.abs(Ay_all[grid.size:])))
                  if topo.n > grid.size else 0.0)
    return {
        "lhs_sum_LN": lhs, "rhs_boundary_plus_source": rhs,
        "abs_defect": abs_defect, "rel_defect": rel_defect,
        "rate_l1_scale": scale, "aux_constraint_abs_defect": aux_defect,
        **terms,
    }



def run_gpu(
    cfg: SolverConfig, phys: DerivedPhysics, grid: Grid, coll: CollisionData,
    topo: CSRTopology, N0: np.ndarray, gpu_context,
    ch: CHGeometry | None = None,
) -> dict:
    """Assemble and solve fixed/adaptive TR--BDF2 stages on the GPU."""
    if wp is None:
        raise RuntimeError("this production script requires warp-lang and a CUDA device")
    nvmath, cudss, dev_obj, runtime = gpu_context
    dev = cfg.device
    fixed_nsteps = nsteps_from_cfg(cfg) if not cfg.adaptive else None
    physical_n = grid.size
    trbdf2_d = 1.0 - 1.0/math.sqrt(2.0)

    def wa(x, dtype):
        return wp.array(np.ascontiguousarray(x), dtype=dtype, device=dev)

    row_ptr = topo.row_ptr
    col_ind = topo.col_ind
    values = wp.zeros(topo.nnz, dtype=wp.float64, device=dev)
    p_faces = wa(grid.p_faces, wp.float64)
    p_centers = wa(grid.p_centers, wp.float64)
    xi_faces = wa(grid.xi_faces, wp.float64)
    xi_centers = wa(grid.xi_centers, wp.float64)
    radial_volume = wa(grid.radial_volume, wp.float64)
    cell_volume = wa(grid.cell_volume, wp.float64)
    cf_faces = wa(coll.cf_faces, wp.float64)
    ca_faces = wa(coll.ca_faces, wp.float64)
    nud_centers = wa(coll.nud_centers, wp.float64)
    ch_active = ch is not None and ch.active_count > 0
    ch_coeff = wa(ch.row_coefficient, wp.float64) if ch_active else None

    # Algebraic RHS/solution span the full augmented system. Retained states
    # include auxiliary radial masses when Chiu--Harvey is active.
    rhs = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    solution = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    state_n = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    state_gamma = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    state_one = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    applied_n = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    applied_gamma = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    applied_one = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    residual = wp.zeros(topo.n, dtype=wp.float64, device=dev)
    state_flag = wp.zeros(1, dtype=wp.int32, device=dev)
    initial_physical = wa(N0, wp.float64)
    wp.launch(copy_prefix_kernel, dim=physical_n, inputs=[initial_physical], outputs=[state_n], device=dev)
    if ch_active:
        wp.launch(fill_aux_state_kernel, dim=grid.Np,
                  inputs=[initial_physical, physical_n, grid.Nxi],
                  outputs=[state_n], device=dev)

    # Physical block: A = I - d*dtau*L_local. If CH is active, each
    # target row couples to one auxiliary radial pitch sum M_k, and the Np
    # algebraic rows impose M_k - sum_j N[k,j] = 0 exactly.
    t0 = time.perf_counter()
    wp.launch(
        assemble_trbdf2_matrix_kernel, dim=physical_n,
        inputs=[p_faces, p_centers, xi_faces, xi_centers,
                radial_volume, cell_volume, cf_faces, ca_faces, nud_centers,
                topo.diag_slot, topo.pm_slot, topo.pp_slot, topo.xm_slot, topo.xp_slot,
                grid.Np, grid.Nxi, wp.float64(grid.dp), wp.float64(grid.dxi),
                wp.float64(phys.Ebar), wp.float64(phys.alpha),
                wp.float64(trbdf2_d*cfg.dtau), wp.float64(1.0),
                1 if cfg.inner_boundary == "absorbing" else 0],
        outputs=[values], device=dev,
    )
    if ch_active:
        if topo.formulation != "augmented-ch":
            raise RuntimeError("active Chiu-Harvey source requires augmented CH topology")
        wp.launch(
            add_ch_augmented_kernel, dim=ch.active_count,
            inputs=[ch_coeff, topo.ch_aux_slot, wp.float64(trbdf2_d*cfg.dtau)],
            outputs=[values], device=dev,
        )
        wp.launch(
            fill_aux_constraint_n_kernel, dim=physical_n,
            inputs=[topo.aux_n_start_slot, grid.Nxi], outputs=[values], device=dev,
        )
        wp.launch(
            fill_aux_constraint_diag_kernel, dim=grid.Np,
            inputs=[topo.aux_diag_slot], outputs=[values], device=dev,
        )
    wp.synchronize_stream(wp.get_stream(dev_obj))
    assembly_s = time.perf_counter() - t0

    balance = None
    if cfg.balance_check:
        balance = gpu_fv_balance_check(
            cfg, phys, grid, coll, topo, row_ptr, col_ind, values, dev,
            matrix_dt=trbdf2_d*cfg.dtau, ch=ch,
        )
        msg = (
            "finite-volume balance self-check: "
            f"abs={balance['abs_defect']:.3e} rel={balance['rel_defect']:.3e}"
        )
        if topo.n > physical_n:
            msg += f" aux={balance['aux_constraint_abs_defect']:.3e}"
        print(msg)
        if balance["rel_defect"] > 1.0e-9 or balance["aux_constraint_abs_defect"] > 1.0e-12:
            raise RuntimeError(
                "finite-volume/auxiliary balance self-check failed: "
                f"relative defect {balance['rel_defect']:.3e}, "
                f"aux defect {balance['aux_constraint_abs_defect']:.3e}"
            )

    times: list[float] = [0.0]
    dlist: list[dict[str, float]] = [diagnostics(N0, cfg, phys, grid)]
    snapshot_steps: list[int] = []
    snapshot_times: list[float] = []
    snaps: list[np.ndarray] = []

    def record_intermediate(step: int, t_now: float, arr) -> None:
        interior = t_now < cfg.tau_end - 1.0e-14*max(1.0, cfg.tau_end)
        need_diag = interior and cfg.diag_every > 0 and step % cfg.diag_every == 0
        need_snap = interior and cfg.save_every > 0 and step % cfg.save_every == 0
        if not (need_diag or need_snap):
            return
        host = np.asarray(arr.numpy(), dtype=np.float64)[:physical_n]
        if need_diag:
            times.append(t_now)
            dlist.append(diagnostics(host, cfg, phys, grid))
        if need_snap:
            snapshot_steps.append(step)
            snapshot_times.append(t_now)
            snaps.append(host.reshape(grid.Np, grid.Nxi))

    def finish(final: np.ndarray, accepted_steps: int, **timings) -> dict:
        flag = int(np.asarray(state_flag.numpy(), dtype=np.int32)[0])
        if flag >= 2:
            raise FloatingPointError("NaN/Inf occurred in the kinetic state during time integration")
        final_host = np.asarray(final, dtype=np.float64)[:physical_n]
        if cfg.tau_end > 0.0:
            times.append(cfg.tau_end)
            dlist.append(diagnostics(final_host, cfg, phys, grid))
        timings.setdefault("be_residual", timings.get("first_stage_residual", 0.0))
        timings.setdefault("be_factor_s", timings.get("factorization_s", 0.0))
        timings.setdefault("be_solve_s", timings.get("first_stage_solve_s", 0.0))
        timings.setdefault("bdf_factor_s", timings.get("refactorization_s", 0.0))
        timings.setdefault("mean_bdf_solve_s", timings.get("mean_stage_solve_s", 0.0))
        return {
            "backend": "gpu-warp-cudss",
            "runtime": runtime,
            "large_angle_formulation": topo.formulation,
            "final": final_host,
            "initial": N0,
            "history_t": times,
            "history_d": dlist,
            "snapshot_steps": snapshot_steps,
            "snapshot_t": snapshot_times,
            "snapshots": snaps,
            "balance_check": balance,
            "state_monitor_flag": flag,
            "accepted_steps": accepted_steps,
            "state_change": state_change_norms(final_host, N0),
            "final_particle_rates": particle_balance_terms(
                final_host, cfg, phys, grid, coll, ch=ch
            ),
            **timings,
        }

    if cfg.tau_end == 0.0:
        return finish(N0, 0, first_stage_residual=0.0, last_residual=0.0,
                      assembly_s=assembly_s, analysis_s=0.0,
                      factorization_s=0.0, first_stage_solve_s=0.0,
                      mean_stage_solve_s=0.0, refactorization_s=0.0,
                      rejected_steps=0)

    threading = find_cudss_threading_layer()
    with CudssDirectSystem(
        nvmath=nvmath, cudss=cudss,
        row_ptr=row_ptr, col_ind=col_ind, values=values,
        rhs=rhs, solution=solution, n=topo.n, nnz=topo.nnz,
        threading_layer=threading,
    ) as direct:
        analysis_s = direct.analyze()
        factorization_s = direct.factorize()
        matrix_dt = trbdf2_d*cfg.dtau
        current_t = 0.0
        trial_dt = min(cfg.dtau, cfg.tau_end)
        if cfg.adaptive and cfg.dtau_max is not None:
            trial_dt = min(trial_dt, cfg.dtau_max)
        accepted_steps = 0
        rejected_steps = 0
        assembly_total = assembly_s
        refactorization_total = 0.0
        stage_times: list[float] = []
        first_stage_residual = 0.0
        last_residual = 0.0

        while current_t < cfg.tau_end - 1.0e-14*max(1.0, cfg.tau_end):
            if cfg.adaptive:
                trial_dt = min(trial_dt, cfg.tau_end-current_t)
                if trial_dt < cfg.dtau_min:
                    raise RuntimeError("adaptive timestep fell below runtime.dtau_min")
                desired_matrix_dt = trbdf2_d*trial_dt
                if not math.isclose(desired_matrix_dt, matrix_dt, rel_tol=1.0e-14, abs_tol=0.0):
                    t_assemble = time.perf_counter()
                    wp.launch(
                        assemble_trbdf2_matrix_kernel, dim=physical_n,
                        inputs=[p_faces, p_centers, xi_faces, xi_centers,
                                radial_volume, cell_volume, cf_faces, ca_faces, nud_centers,
                                topo.diag_slot, topo.pm_slot, topo.pp_slot, topo.xm_slot, topo.xp_slot,
                                grid.Np, grid.Nxi, wp.float64(grid.dp), wp.float64(grid.dxi),
                                wp.float64(phys.Ebar), wp.float64(phys.alpha),
                                wp.float64(desired_matrix_dt), wp.float64(1.0),
                                1 if cfg.inner_boundary == "absorbing" else 0],
                        outputs=[values], device=dev,
                    )
                    if ch_active:
                        wp.launch(add_ch_augmented_kernel, dim=ch.active_count,
                                  inputs=[ch_coeff, topo.ch_aux_slot, wp.float64(desired_matrix_dt)],
                                  outputs=[values], device=dev)
                    wp.synchronize_stream(wp.get_stream(dev_obj))
                    assembly_total += time.perf_counter()-t_assemble
                    refactorization_total += direct.refactorize()
                    matrix_dt = desired_matrix_dt

            wp.launch(csr_matvec_kernel, dim=topo.n,
                      inputs=[row_ptr, col_ind, values, state_n], outputs=[applied_n], device=dev)
            wp.launch(build_trbdf2_stage1_rhs_kernel, dim=topo.n,
                      inputs=[state_n, applied_n, physical_n], outputs=[rhs], device=dev)
            stage_times.append(direct.solve())
            if accepted_steps == 0:
                first_stage_residual = gpu_relative_residual(
                    row_ptr, col_ind, values, solution, rhs, residual, topo.n, dev
                )
            wp.launch(copy_all_kernel, dim=topo.n,
                      inputs=[solution], outputs=[state_gamma], device=dev)

            wp.launch(csr_matvec_kernel, dim=topo.n,
                      inputs=[row_ptr, col_ind, values, state_gamma], outputs=[applied_gamma], device=dev)
            wp.launch(build_trbdf2_stage2_rhs_kernel, dim=topo.n,
                      inputs=[state_n, applied_n, state_gamma, applied_gamma,
                              physical_n, wp.float64(1.0/trbdf2_d)],
                      outputs=[rhs], device=dev)
            stage_times.append(direct.solve())
            last_residual = gpu_relative_residual(
                row_ptr, col_ind, values, solution, rhs, residual, topo.n, dev
            )
            wp.launch(copy_all_kernel, dim=topo.n,
                      inputs=[solution], outputs=[state_one], device=dev)
            wp.launch(monitor_state_kernel, dim=physical_n,
                      inputs=[state_one, state_flag], device=dev)

            error_norm = 0.0
            if cfg.adaptive:
                wp.launch(csr_matvec_kernel, dim=topo.n,
                          inputs=[row_ptr, col_ind, values, state_one], outputs=[applied_one], device=dev)
                wp.launch(build_trbdf2_error_rhs_kernel, dim=topo.n,
                          inputs=[state_n, applied_n, state_gamma, applied_gamma,
                                  state_one, applied_one, physical_n,
                                  wp.float64(1.0/trbdf2_d)], outputs=[rhs], device=dev)
                direct.solve()  # Hosea--Shampine stiff filter with the same factorization.
                err = np.asarray(solution.numpy(), dtype=np.float64)[:physical_n]
                endpoint = np.asarray(state_one.numpy(), dtype=np.float64)[:physical_n]
                previous = np.asarray(state_n.numpy(), dtype=np.float64)[:physical_n]
                scale = cfg.atol + cfg.rtol*np.maximum(np.abs(endpoint), np.abs(previous))
                error_norm = float(np.sqrt(np.mean((err/scale)**2)))
                if not math.isfinite(error_norm):
                    raise FloatingPointError("non-finite adaptive TR-BDF2 error estimate")

            if not cfg.adaptive or error_norm <= 1.0:
                accepted_steps += 1
                current_t += trial_dt
                wp.launch(copy_all_kernel, dim=topo.n,
                          inputs=[state_one], outputs=[state_n], device=dev)
                record_intermediate(accepted_steps, current_t, state_n)
                if cfg.adaptive:
                    factor = cfg.max_factor if error_norm == 0.0 else cfg.safety*error_norm**(-1.0/3.0)
                    factor = min(cfg.max_factor, max(cfg.min_factor, factor))
                    trial_dt *= factor
                    if cfg.dtau_max is not None:
                        trial_dt = min(trial_dt, cfg.dtau_max)
                else:
                    if accepted_steps >= fixed_nsteps:
                        break
            else:
                rejected_steps += 1
                factor = cfg.safety*error_norm**(-1.0/3.0)
                factor = min(1.0, max(cfg.min_factor, factor))
                trial_dt *= factor
                if trial_dt < cfg.dtau_min:
                    raise RuntimeError("adaptive timestep rejection reached runtime.dtau_min")
                if rejected_steps > cfg.max_steps:
                    raise RuntimeError("adaptive timestep exceeded runtime.max_steps")

            if accepted_steps + rejected_steps > cfg.max_steps:
                raise RuntimeError("adaptive timestep exceeded runtime.max_steps")

        final = np.asarray(state_n.numpy(), dtype=np.float64)

    return finish(
        final, accepted_steps, first_stage_residual=first_stage_residual,
        last_residual=last_residual, assembly_s=assembly_total,
        analysis_s=analysis_s, factorization_s=factorization_s,
        refactorization_s=refactorization_total,
        first_stage_solve_s=stage_times[0] if stage_times else 0.0,
        mean_stage_solve_s=float(np.mean(stage_times)) if stage_times else 0.0,
        rejected_steps=rejected_steps,
    )



# =============================================================================
# TOML configuration, reporting, and output
# =============================================================================




CONFIG_FILE = Path(__file__).resolve().with_suffix(".toml")


def load_config(config_path: Path | None = None) -> tuple[SolverConfig, str, Path]:
    """Read the case TOML and map it directly to the solver configuration."""
    path = (config_path or CONFIG_FILE).resolve()
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)

    plasma = raw["plasma"]
    collisions = raw["collisions"]
    avalanche = raw["avalanche"]
    grid = raw["grid"]
    time_cfg = raw["time"]
    initial = raw["initial"]
    seed = raw["seed"]
    runtime = raw["runtime"]
    output = raw["output"]

    ions = tuple(
        IonSpecies(
            name=str(ion["name"]),
            Z=int(ion["Z"]),
            Z0=int(ion["Z0"]),
            density_m3=float(ion["density_m3"]),
            I_eV=float(ion.get("I_eV", 0.0)),
            a_bar=float(ion.get("a_bar", 0.0)),
            charge_populations_m3=(
                tuple(float(value) for value in ion["charge_populations_m3"])
                if "charge_populations_m3" in ion else None
            ),
            I_eV_by_charge=(
                tuple(float(value) for value in ion["I_eV_by_charge"])
                if "I_eV_by_charge" in ion else None
            ),
            a_bar_by_charge=(
                tuple(float(value) for value in ion["a_bar_by_charge"])
                if "a_bar_by_charge" in ion else None
            ),
        )
        for ion in plasma["ions"]
    )

    output_path = Path(output["path"])
    if not output_path.is_absolute():
        output_path = path.parent / output_path

    cfg = SolverConfig(
        te_eV=float(plasma["te_eV"]),
        ne_m3=float(plasma["ne_m3"]),
        e_parallel_Vm=float(plasma["e_parallel_Vm"]),
        B_T=float(plasma["B_T"]),
        ions=ions,
        nt_m3=float(plasma["nt_m3"]) if "nt_m3" in plasma else None,
        n_ref_m3=float(plasma["n_ref_m3"]) if "n_ref_m3" in plasma else None,
        ln_lambda_ref=float(plasma["ln_lambda_ref"]) if "ln_lambda_ref" in plasma else None,
        small_angle_model=str(collisions["model"]),
        large_angle_model="chiu-harvey" if avalanche["enabled"] else "none",
        p_m=float(avalanche["p_m"]),
        Np=int(grid["Np"]),
        Nxi=int(grid["Nxi"]),
        pmin=float(grid["pmin"]),
        pmax=float(grid["pmax"]),
        inner_boundary=str(grid["inner_boundary"]),
        energy_diffusion=collisions["energy_diffusion"],
        q_coll=int(collisions["q_coll"]),
        q_init=int(initial["q_init"]),
        dtau=float(time_cfg["dtau"]),
        tau_end=float(time_cfg["tau_end"]),
        init=str(initial["kind"]),
        init_te_eV=float(initial["te_eV"]) if "te_eV" in initial else None,
        init_density_m3=float(initial["density_m3"]) if "density_m3" in initial else None,
        gaussian_p0=float(initial["gaussian_p0"]),
        gaussian_sigma_p=float(initial["gaussian_sigma_p"]),
        gaussian_xi0=float(initial["gaussian_xi0"]),
        gaussian_sigma_xi=float(initial["gaussian_sigma_xi"]),
        seed_fraction=float(seed["fraction"]),
        seed_p0=float(seed["p0"]),
        seed_sigma_p=float(seed["sigma_p"]),
        seed_xi0=float(seed["xi0"]),
        seed_sigma_xi=float(seed["sigma_xi"]),
        device=str(runtime["device"]),
        adaptive=bool(runtime.get("adaptive", False)),
        rtol=float(runtime.get("rtol", 1.0e-4)),
        atol=float(runtime.get("atol", 1.0e-12)),
        safety=float(runtime.get("safety", 0.9)),
        min_factor=float(runtime.get("min_factor", 0.2)),
        max_factor=float(runtime.get("max_factor", 5.0)),
        dtau_min=float(runtime.get("dtau_min", 1.0e-8)),
        dtau_max=(float(runtime["dtau_max"]) if "dtau_max" in runtime else None),
        max_steps=int(runtime.get("max_steps", 100000)),
        save_every=int(output["save_every"]),
        diag_every=int(output["diag_every"]),
        output=output_path,
        write_output=output["write"],
        balance_check=runtime["balance_check"],
    )
    validate_config(cfg)
    return cfg, text, path


def validate_config(cfg: SolverConfig) -> None:
    """Reject malformed cases before host preprocessing or GPU allocation."""
    if cfg.te_eV <= 0.0 or cfg.ne_m3 <= 0.0:
        raise ValueError("plasma te_eV and ne_m3 must be positive")
    if not cfg.ions:
        raise ValueError("plasma.ions must contain at least one species")
    for ion in cfg.ions:
        if ion.Z < 1 or not 0 <= ion.Z0 <= ion.Z or ion.density_m3 < 0.0:
            raise ValueError(f"invalid ion charge/density for {ion.name!r}")
        if ion.Z0 < ion.Z and (ion.I_eV <= 0.0 or ion.a_bar <= 0.0):
            raise ValueError(f"partially ionized ion {ion.name!r} needs I_eV and a_bar > 0")
        state_inputs = (
            ion.charge_populations_m3,
            ion.I_eV_by_charge,
            ion.a_bar_by_charge,
        )
        if any(value is not None for value in state_inputs):
            if any(value is None for value in state_inputs):
                raise ValueError(
                    f"state-resolved screening for {ion.name!r} needs populations, I_eV_by_charge, and a_bar_by_charge"
                )
            assert ion.charge_populations_m3 is not None
            assert ion.I_eV_by_charge is not None
            assert ion.a_bar_by_charge is not None
            expected = ion.Z + 1
            if any(len(value) != expected for value in state_inputs):
                raise ValueError(
                    f"state-resolved screening for {ion.name!r} needs Z+1 entries"
                )
            if any(value < 0.0 or not math.isfinite(value)
                   for value in ion.charge_populations_m3):
                raise ValueError(f"invalid charge populations for {ion.name!r}")
            if not math.isclose(
                sum(ion.charge_populations_m3), ion.density_m3,
                rel_tol=2.0e-12, abs_tol=1.0e6,
            ):
                raise ValueError(f"charge populations do not conserve density for {ion.name!r}")
            for charge, population in enumerate(ion.charge_populations_m3):
                if population > 0.0 and ion.Z > charge:
                    if ion.I_eV_by_charge[charge] <= 0.0 or ion.a_bar_by_charge[charge] <= 0.0:
                        raise ValueError(
                            f"bound state q={charge} for {ion.name!r} needs positive I_eV and a_bar"
                        )
    if cfg.small_angle_model not in {
        "finite-temperature-relativistic",
        "fully-relativistic-asymptotically-matched",
    }:
        raise ValueError(f"unsupported small_angle_model {cfg.small_angle_model!r}")
    if cfg.large_angle_model not in {"none", "chiu-harvey"}:
        raise ValueError(f"unsupported large_angle_model {cfg.large_angle_model!r}")
    if cfg.inner_boundary not in {"zero-flux", "absorbing"}:
        raise ValueError(f"inner_boundary must be 'zero-flux' or 'absorbing', got {cfg.inner_boundary!r}")
    if cfg.inner_boundary == "absorbing" and cfg.pmin <= 0.0:
        raise ValueError("absorbing inner_boundary requires pmin > 0")
    if cfg.Np < 1 or cfg.Nxi < 1 or cfg.pmin < 0.0 or cfg.pmax <= cfg.pmin:
        raise ValueError("grid requires Np,Nxi >= 1, pmin >= 0, and pmax > pmin")
    if cfg.large_angle_model == "chiu-harvey" and cfg.p_m <= 0.0:
        raise ValueError("Chiu-Harvey avalanche requires p_m > 0")
    if cfg.q_coll < 1 or cfg.q_init < 1:
        raise ValueError("q_coll and q_init must be positive")
    if cfg.dtau <= 0.0 or cfg.tau_end < 0.0:
        raise ValueError("dtau must be positive and tau_end must be nonnegative")
    if cfg.adaptive:
        if cfg.rtol <= 0.0 or cfg.atol <= 0.0:
            raise ValueError("adaptive rtol and atol must be positive")
        if not (0.0 < cfg.min_factor <= 1.0 <= cfg.max_factor):
            raise ValueError("adaptive step factors must satisfy 0 < min_factor <= 1 <= max_factor")
        if not (0.0 < cfg.safety < 1.0):
            raise ValueError("adaptive safety must lie in (0, 1)")
        if cfg.dtau_min <= 0.0 or (cfg.dtau_max is not None and cfg.dtau_max < cfg.dtau_min):
            raise ValueError("adaptive timestep bounds are invalid")
        if cfg.max_steps < 1:
            raise ValueError("adaptive max_steps must be positive")
    else:
        nsteps = round(cfg.tau_end / cfg.dtau)
        if not math.isclose(nsteps * cfg.dtau, cfg.tau_end, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("tau_end must be an integer multiple of dtau when adaptive=false")
    if cfg.init not in {"maxwell-juttner", "gaussian"}:
        raise ValueError(f"unsupported initial condition {cfg.init!r}")
    if cfg.gaussian_sigma_p <= 0.0 or cfg.gaussian_sigma_xi <= 0.0:
        raise ValueError("Gaussian initial-condition widths must be positive")
    if not 0.0 <= cfg.seed_fraction <= 1.0:
        raise ValueError("seed fraction must lie in [0, 1]")
    if cfg.save_every < 0 or cfg.diag_every < 0:
        raise ValueError("save_every and diag_every must be nonnegative")
    if not cfg.device:
        raise ValueError("runtime.device must be nonempty")



def print_problem(
    cfg: SolverConfig, phys: DerivedPhysics, grid: Grid, coll: CollisionData,
    topo, N0: np.ndarray, ch: CHGeometry | None = None,
):
    lhs = phys.tau_ref_s*2.0*math.pi*phys.nt_m3*const.c*R_E**2
    rhs = phys.nt_over_nref/(2.0*phys.ln_lambda_ref)
    normerr = abs(lhs-rhs)/max(abs(rhs), 1e-300)
    print("=== standalone 0D-2P forward FV solver ===")
    print(f"execution=gpu-warp-cudss  grid={grid.Np}x{grid.Nxi}  state={grid.size:,}  p=[{cfg.pmin:g},{cfg.pmax:g}]")
    print(f"dp={grid.dp:.6e}  dxi={grid.dxi:.6e}  dtau={cfg.dtau:.6e}  tau_end={cfg.tau_end:.6e}")
    print(f"inner boundary={cfg.inner_boundary}  energy diffusion={'on' if cfg.energy_diffusion else 'off'}")
    print(f"Te={cfg.te_eV:.6e} eV  ne={cfg.ne_m3:.6e} m^-3  Z_eff={phys.z_eff:.8g}")
    print(f"lnLambda0={phys.ln_lambda0:.8g}  nref={phys.n_ref_m3:.6e}  lnLambda_ref={phys.ln_lambda_ref:.8g}")
    print(f"tau_ref={phys.tau_ref_s:.6e} s  E_ref={phys.E_ref_Vm:.6e} V/m  Ebar={phys.Ebar:.6e}")
    print(f"B={cfg.B_T:.6e} T  alpha=tau_ref/tau_s={phys.alpha:.6e}")
    print(f"nt={phys.nt_m3:.6e} m^-3  nt/nref={phys.nt_over_nref:.6e}  knock-on normalization rel.err={normerr:.3e}")
    if phys.free_density_from_ions_m3 > 0.0:
        relqn = abs(phys.free_density_from_ions_m3-cfg.ne_m3)/cfg.ne_m3
        if relqn > 1e-10:
            print(f"WARNING: sum(n_i Z0_i) differs from prescribed ne by relative {relqn:.3e}")
    print(f"small-angle model={cfg.small_angle_model}")
    screening_active = any(ion.n_bound for ion in cfg.resolved_ions())
    print(f"partial screening={'on' if screening_active else 'off'}")
    if cfg.small_angle_model == "finite-temperature-relativistic":
        print(f"collision p_switch={coll.p_switch:.6e}  scaled K2={coll.k2_scaled:.6e}  q_coll={cfg.q_coll}")
        print(f"collision branch overlap max rel: Psi_s={coll.overlap_rel_psi_s:.3e}, Psi_D={coll.overlap_rel_psi_d:.3e}")
    else:
        print(f"fully-relativistic asymptotically matched collision p_switch={coll.p_switch:.6e}  Phi/Psi small-x overlap rel={coll.overlap_rel_psi_s:.3e}")
    if cfg.seed_fraction > 0.0:
        init_density = cfg.ne_m3 if cfg.init_density_m3 is None else cfg.init_density_m3
        print(
            "initial runaway seed: "
            f"fraction={cfg.seed_fraction:.6e}  n_seed={cfg.seed_fraction*init_density:.6e} m^-3  "
            f"p0={cfg.seed_p0:g} sigma_p={cfg.seed_sigma_p:g}  "
            f"xi0={cfg.seed_xi0:g} sigma_xi={cfg.seed_sigma_xi:g}"
        )
    print(f"large-angle model={cfg.large_angle_model}")
    if cfg.large_angle_model == "none":
        print("large-angle gain=off")
    else:
        print(f"Chiu-Harvey active targets={ch.active_count:,}")
        if topo.n > grid.size:
            print(f"CH implicit algebra=augmented-exact  auxiliary radial unknowns={grid.Np:,}  system={topo.n:,}")
        else:
            print("Chiu-Harvey source has no active target cells on this grid")
    print(f"CSR nnz={topo.nnz:,}  avg nnz/system-row={topo.nnz/topo.n:.2f}")
    print(f"initial diagnostics: {diagnostics(N0, cfg, phys, grid)}")



def report_result(result: dict, cfg: SolverConfig, phys: DerivedPhysics, grid: Grid):
    d = diagnostics(result["final"], cfg, phys, grid)
    print("=== result ===")
    print(f"backend={result['backend']}")
    print(f"small-angle model={cfg.small_angle_model}")
    print(f"partial screening={'on' if any(ion.n_bound for ion in cfg.resolved_ions()) else 'off'}")
    print(f"large-angle model={cfg.large_angle_model}  formulation={result.get('large_angle_formulation', 'local')}")
    if result.get("runtime"):
        print(f"runtime={result['runtime']}")
    print(f"final n/nref={d['density_ratio']:.12e}")
    print(f"final j_parallel={d['current_A_m2']:.12e} A/m^2")
    print(f"final min(N)={d['min_N']:.6e}  max(f)={d['max_f']:.6e}")
    print(f"TR-BDF2 stage-1 residual={result['first_stage_residual']:.3e}  final residual={result['last_residual']:.3e}")

    change = result["state_change"]
    eq_case = (
        cfg.init == "maxwell-juttner"
        and cfg.small_angle_model == "finite-temperature-relativistic"
        and cfg.large_angle_model == "none"
        and cfg.energy_diffusion
        and not any(ion.n_bound for ion in cfg.resolved_ions())
        and cfg.inner_boundary == "zero-flux" and cfg.pmin == 0.0
        and phys.Ebar == 0.0 and phys.alpha == 0.0
        and (cfg.init_te_eV is None or math.isclose(cfg.init_te_eV, cfg.te_eV, rel_tol=1e-14))
    )
    label = "Maxwell-Juttner equilibrium defect" if eq_case else "state change"
    print(f"{label}: relL1={change['rel_l1']:.6e} relL2={change['rel_l2']:.6e} relLinf={change['rel_linf']:.6e}")

    rates = result["final_particle_rates"]
    print("final particle rates: "
          f"inner_out={rates['inner_outflow_rate']:.6e} "
          f"outer_out={rates['outer_outflow_rate']:.6e} "
          f"LA_prod={rates['large_angle_production_rate']:.6e} "
          f"net={rates['net_particle_rate']:.6e} [1/tau]")
    if result.get("balance_check") is not None:
        b = result["balance_check"]
        extra = f" aux={b.get('aux_constraint_abs_defect', 0.0):.3e}" if result.get("large_angle_formulation") == "augmented-ch" else ""
        print(f"FV balance check: abs={b['abs_defect']:.3e} rel={b['rel_defect']:.3e}{extra}")
    if result.get("state_monitor_flag", 0) == 1:
        print("WARNING: at least one negative cell content occurred during the trajectory")

    if result.get("topology_timing") is not None:
        tt = result["topology_timing"]
        print(
            "topology_s="
            f"{tt['total_s']:.6e} "
            f"(rowptr_host={tt['rowptr_host_s']:.3e}, "
            f"fill_gpu={tt['fill_gpu_s']:.3e}, validate_gpu={tt['validate_gpu_s']:.3e})"
        )
    for k in ("assembly_s","analysis_s","factorization_s","refactorization_s","mean_stage_solve_s"):
        if k in result:
            print(f"{k}={result[k]:.6e} s")
    print(f"accepted_steps={result.get('accepted_steps', 0)} rejected_steps={result.get('rejected_steps', 0)}")



def config_json(cfg: SolverConfig) -> str:
    d = asdict(cfg)
    d["output"] = str(cfg.output)
    return json.dumps(d, sort_keys=True)


def save_result(
    path: Path, result: dict, cfg: SolverConfig, phys: DerivedPhysics,
    grid: Grid, coll: CollisionData, ch: CHGeometry | None = None,
    input_config_path: Path | None = None,
    input_config_toml: str | None = None,
):
    """Write one NPZ without constructing redundant full-state arrays.

    N_initial and N_final are stored once.  Intermediate snapshots, when
    requested, are written as independent NPZ members (N_snapshot_000000, ...),
    avoiding the former np.asarray(list_of_snapshots) stack/copy.  f_final is
    intentionally omitted because it is exactly reconstructed as
    N_final/(radial_volume[:,None]*dxi).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = result["history_d"]
    initial = np.asarray(result["initial"], dtype=np.float64).reshape(grid.Np, grid.Nxi)
    final = np.asarray(result["final"], dtype=np.float64).reshape(grid.Np, grid.Nxi)
    balance_json = json.dumps(result.get("balance_check"), sort_keys=True)
    rates_json = json.dumps(result.get("final_particle_rates"), sort_keys=True)
    change_json = json.dumps(result.get("state_change"), sort_keys=True)
    topology_json = json.dumps(result.get("topology_timing"), sort_keys=True)
    storage_json = json.dumps(result.get("storage_estimate"), sort_keys=True)

    payload = dict(
        output_schema=np.asarray("gpu-forward-fv-v1"),
        p_faces=grid.p_faces, p_centers=grid.p_centers,
        xi_faces=grid.xi_faces, xi_centers=grid.xi_centers,
        radial_volume=grid.radial_volume,
        N_initial=initial,
        N_final=final,
        diagnostic_tau=np.asarray(result["history_t"], dtype=np.float64),
        density_ratio=np.asarray([x["density_ratio"] for x in hist], dtype=np.float64),
        current_A_m2=np.asarray([x["current_A_m2"] for x in hist], dtype=np.float64),
        min_N=np.asarray([x["min_N"] for x in hist], dtype=np.float64),
        max_f=np.asarray([x["max_f"] for x in hist], dtype=np.float64),
        snapshot_step=np.asarray(result.get("snapshot_steps", []), dtype=np.int64),
        snapshot_tau=np.asarray(result.get("snapshot_t", []), dtype=np.float64),
        ch_active_rows=(ch.active_rows if ch is not None else np.empty(0, dtype=np.int32)),
        ch_primary_radial=(ch.primary_radial if ch is not None else np.empty(0, dtype=np.int32)),
        ch_root_p=(ch.root_p if ch is not None else np.empty(0, dtype=np.float64)),
        ch_row_coefficient=(ch.row_coefficient if ch is not None else np.empty(0, dtype=np.float64)),
        small_angle_model=np.asarray(cfg.small_angle_model),
        partial_screening=np.asarray(any(ion.n_bound for ion in cfg.resolved_ions())),
        large_angle_model=np.asarray(cfg.large_angle_model),
        psi_s_centers=coll.psi_s_centers,
        psi_d_centers=coll.psi_d_centers,
        config_json=np.asarray(config_json(cfg)),
        input_config_path=np.asarray(str(input_config_path) if input_config_path is not None else ""),
        input_config_toml=np.asarray(input_config_toml or ""),
        physics_json=np.asarray(json.dumps(asdict(phys), sort_keys=True)),
        balance_check_json=np.asarray(balance_json),
        final_particle_rates_json=np.asarray(rates_json),
        state_change_json=np.asarray(change_json),
        topology_timing_json=np.asarray(topology_json),
        storage_estimate_json=np.asarray(storage_json),
        backend=np.asarray(result["backend"]),
        large_angle_formulation=np.asarray(result.get("large_angle_formulation", "local")),
        state_monitor_flag=np.asarray(result.get("state_monitor_flag", 0), dtype=np.int32),
        accepted_steps=np.asarray(result.get("accepted_steps", 0), dtype=np.int64),
        rejected_steps=np.asarray(result.get("rejected_steps", 0), dtype=np.int64),
        adaptive=np.asarray(cfg.adaptive),
        first_stage_residual=np.asarray(result.get("first_stage_residual", 0.0)),
        be_residual=np.asarray(result["be_residual"]),
        last_residual=np.asarray(result["last_residual"]),
    )
    for k, snap in enumerate(result.get("snapshots", [])):
        payload[f"N_snapshot_{k:06d}"] = np.asarray(snap, dtype=np.float64)
    np.savez_compressed(path, **payload)



def main():
    parser = argparse.ArgumentParser(description="GPU forward finite-volume solver")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help="TOML case file (default: %(default)s)",
    )
    args = parser.parse_args()
    cfg, config_text, config_path = load_config(args.config)
    if wp is None:
        raise RuntimeError(
            "forward_fv_solver.py is GPU-only and requires warp-lang"
        )

    print(f"loaded case: {config_path.expanduser().resolve()}")
    t0 = time.perf_counter()
    phys = derive_physics(cfg)
    grid = build_grid(cfg)

    print(f"evaluating {cfg.small_angle_model} small-angle collision coefficients ...", flush=True)
    t = time.perf_counter()
    coll = build_collision_data(cfg, phys, grid)
    validate_collision_data(coll)
    print(f"collision coefficients: {time.perf_counter()-t:.3f} s")
    if max(coll.overlap_rel_psi_s, coll.overlap_rel_psi_d) > 1.0e-7:
        print("WARNING: small-argument/direct collision overlap exceeds 1e-7; inspect coefficient convergence")

    ch = None
    if cfg.include_ch:
        print("precomputing Chiu-Harvey geometry ...", flush=True)
        t = time.perf_counter()
        ch = build_ch_geometry(cfg, phys, grid)
        validate_ch_geometry(cfg, grid, ch)
        print(f"Chiu-Harvey geometry: {time.perf_counter()-t:.3f} s")

    print("initializing Warp/cuDSS runtime ...", flush=True)
    gpu_context = load_gpu_runtime(cfg.device)

    if ch is not None and ch.active_count:
        print("building and validating exact augmented CH CSR topology on GPU ...", flush=True)
        topo, topo_timing = build_augmented_ch_csr_topology_gpu(grid, ch, cfg.device)
    else:
        print("building and validating local FV CSR topology on GPU ...", flush=True)
        topo, topo_timing = build_local_csr_topology_gpu(grid, cfg.device)
    print(
        "CSR topology: "
        f"rowptr_host={topo_timing['rowptr_host_s']:.3e} s  "
        f"fill_gpu={topo_timing['fill_gpu_s']:.3e} s  "
        f"validate_gpu={topo_timing['validate_gpu_s']:.3e} s  "
        f"total={topo_timing['total_s']:.3e} s"
    )
    storage = print_storage_preflight(cfg, grid, topo, ch=ch)

    print("projecting initial distribution ...", flush=True)
    t = time.perf_counter()
    N0 = project_initial(cfg, phys, grid)
    if not np.all(np.isfinite(N0)) or np.any(N0 < 0.0):
        raise FloatingPointError("initial finite-volume state is invalid")
    print(f"initial projection: {time.perf_counter()-t:.3f} s")
    print_problem(cfg, phys, grid, coll, topo, N0, ch=ch)

    result = run_gpu(cfg, phys, grid, coll, topo, N0, gpu_context, ch=ch)
    result["topology_timing"] = topo_timing
    result["storage_estimate"] = storage
    report_result(result, cfg, phys, grid)
    if cfg.write_output:
        save_result(
            cfg.output, result, cfg, phys, grid, coll, ch=ch,
            input_config_path=config_path.expanduser().resolve(),
            input_config_toml=config_text,
        )
        print(f"wrote {cfg.output}")
    else:
        print("output writing disabled by [output].write=false")
    print(f"total wall={time.perf_counter()-t0:.3f} s")



if __name__ == "__main__":
    main()
