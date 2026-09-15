#!/usr/bin/env python3
"""MPI/PETSc forward relativistic Fokker--Planck solver.

This is the first PETSc replacement for the notebook prototype.  It keeps the
forward FP and auxiliary Chiu--Harvey source, uses the logarithmic momentum grid,
and advances constant-parameter cases with implicit TR--BDF2.  The primary kinetic
state is finite-volume cell content N; reconstructed f is used for local fluxes
and diagnostics.  PETSc assembles a distributed AIJ matrix and MUMPS performs the
distributed direct solves.

Run inside a Slurm allocation, for example::

    MPICH_GPU_SUPPORT_ENABLED=0 srun -n 4 \
        .petsc-cpu-venv/bin/python petsc_forward_solver.py

The fixed-plasma electric-field/induction path retains adaptive BDF2 with a
bordered Newton solve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import constants, special
from numba import njit
from mpi4py import MPI

from petsc4py import PETSc


MEC2_EV = constants.m_e * constants.c**2 / constants.e
NONTHERMAL_ENERGY_FACTOR = 10.0
FIXED_T_E_EV = 10.0
MOMENTUM_SWITCH = 1.0
R_E = constants.e**2 / (
    4.0*np.pi*constants.epsilon_0*constants.m_e*constants.c**2
)


@dataclass(frozen=True)
class GridConfig:
    # Runtime values come from the command-line configuration in main().
    p_min: float
    p_max: float
    N_p: int
    N_xi: int


@dataclass(frozen=True)
class IonSpecies:
    Z: int
    n: np.ndarray
    I_eV: np.ndarray
    a_bar: np.ndarray


@dataclass(frozen=True)
class PlasmaConfig:
    T_e_eV: float
    E_parallel: float
    B: float
    ions: tuple[IonSpecies, ...]


@dataclass(frozen=True)
class Grid:
    p_face: np.ndarray
    p_center: np.ndarray
    xi_face: np.ndarray
    xi_center: np.ndarray
    d_p: np.ndarray
    d_xi: np.ndarray
    d_xi_face: np.ndarray
    radial_volume: np.ndarray
    cell_volume: np.ndarray
    operator_row_ptr: np.ndarray
    operator_columns: np.ndarray
    operator_slots: np.ndarray

    @property
    def N_p(self) -> int:
        return self.p_center.size

    @property
    def N_xi(self) -> int:
        return self.xi_center.size

    @property
    def size(self) -> int:
        return self.N_p*self.N_xi


@dataclass(frozen=True)
class DerivedPlasma:
    n_e: float
    Z_eff: float
    Theta: float
    lnLambda0: float
    tau_c: float
    tau_syn: float
    alpha: float
    E_bar: float


@dataclass(frozen=True)
class CollisionData:
    C_F_face: np.ndarray
    C_A_face: np.ndarray
    nu_D_center: np.ndarray


@dataclass(frozen=True)
class CHGeometry:
    active_rows: np.ndarray
    primary_radial: np.ndarray
    row_coefficient: np.ndarray
    row_index: np.ndarray


@dataclass(frozen=True)
class RowRange:
    start: int
    end: int


@dataclass(frozen=True)
class AssemblyContext:
    physical: RowRange
    augmented: RowRange
    augmented_row_ptr: np.ndarray
    augmented_columns: np.ndarray


@dataclass(frozen=True)
class CoupledScales:
    area: float
    eta: float
    ebar_per_Vm: float
    kappa_I: float
    I_scale: float = 1.0e7
    E_scale: float = 30.0

    @property
    def bulk_current_coefficient(self) -> float:
        return self.eta*self.I_scale/(self.area*self.E_scale)

    @property
    def induction_coefficient(self) -> float:
        return self.kappa_I*self.E_scale/self.I_scale


@dataclass
class CoupledStageResidual:
    residual: PETSc.Vec
    current: float
    electric_field: float
    matrix: PETSc.Mat
    d_operator_dE: PETSc.Mat
    fp_operator: PETSc.Mat
    derivative_coefficient: float


@dataclass(frozen=True)
class CoupledErrorEstimate:
    total: float
    distribution: float
    current: float
    electric_field: float
    distribution_cell: int | None = None


def build_grid(cfg: GridConfig) -> Grid:
    if cfg.p_min <= 0.0 or cfg.p_max <= cfg.p_min:
        raise ValueError("logarithmic grid requires 0 < p_min < p_max")
    if cfg.N_p < 2 or cfg.N_xi < 2:
        raise ValueError("N_p and N_xi must exceed one")

    if cfg.p_min < MOMENTUM_SWITCH < cfg.p_max:
        log_total = math.log(cfg.p_max/cfg.p_min)
        N_low = max(1, min(cfg.N_p - 1, round(
            cfg.N_p*math.log(MOMENTUM_SWITCH/cfg.p_min)/log_total
        )))
        N_high = cfg.N_p - N_low
        p_face = np.concatenate((
            np.geomspace(cfg.p_min, MOMENTUM_SWITCH, N_low + 1),
            np.linspace(MOMENTUM_SWITCH, cfg.p_max, N_high + 1)[1:],
        ))
    else:
        p_face = np.geomspace(cfg.p_min, cfg.p_max, cfg.N_p + 1)

    # Uniform theta on the runaway half gives endpoint clustering in xi,
    # while the non-runaway half remains uniformly spaced in xi.
    N_xi_negative = cfg.N_xi//2
    N_xi_positive = cfg.N_xi - N_xi_negative
    theta_face = np.linspace(np.pi, np.pi/2.0, N_xi_negative + 1)
    xi_negative = np.cos(theta_face)
    xi_positive = np.linspace(0.0, 1.0, N_xi_positive + 1)
    xi_face = np.concatenate((xi_negative[:-1], xi_positive))
    p_center = np.sqrt(p_face[:-1]*p_face[1:])
    xi_center = 0.5*(xi_face[:-1] + xi_face[1:])

    # Distance across each radial face.  Interior distances join neighboring
    # cell centers; boundary distances join the cell center to the boundary.
    d_p = np.empty(cfg.N_p + 1, dtype=np.float64)
    d_p[0] = p_center[0] - p_face[0]
    d_p[1:-1] = p_center[1:] - p_center[:-1]
    d_p[-1] = p_face[-1] - p_center[-1]

    d_xi = np.diff(xi_face)
    d_xi_face = np.empty(cfg.N_xi + 1, dtype=np.float64)
    d_xi_face[0] = xi_center[0] - xi_face[0]
    d_xi_face[1:-1] = xi_center[1:] - xi_center[:-1]
    d_xi_face[-1] = xi_face[-1] - xi_center[-1]
    radial_volume = (2.0*np.pi/3.0)*(p_face[1:]**3 - p_face[:-1]**3)
    cell_volume = (radial_volume[:, None]*d_xi[None, :]).ravel()
    operator_row_ptr, operator_columns, operator_slots = build_operator_pattern(
        cfg.N_p, cfg.N_xi
    )
    return Grid(p_face, p_center, xi_face, xi_center, d_p, d_xi,
                d_xi_face, radial_volume, cell_volume,
                operator_row_ptr, operator_columns,
                operator_slots)


def build_operator_pattern(
    N_p: int, Nxi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed five-point CSR structure for FP transport."""
    row_ptr = [0]
    columns: list[int] = []
    slots = np.full((N_p*Nxi, 5), -1, dtype=np.int32)
    for r in range(N_p*Nxi):
        i, j = divmod(r, Nxi)
        row_columns: list[int] = []
        if i > 0:
            row_columns.append(r - Nxi)
        if i < N_p - 1:
            row_columns.append(r + Nxi)
        if j > 0:
            row_columns.append(r - 1)
        if j < Nxi - 1:
            row_columns.append(r + 1)
        row_columns.append(r)
        sorted_columns = sorted(row_columns)
        if i > 0:
            slots[r, 0] = sorted_columns.index(r - Nxi)
        if i < N_p - 1:
            slots[r, 1] = sorted_columns.index(r + Nxi)
        if j > 0:
            slots[r, 2] = sorted_columns.index(r - 1)
        if j < Nxi - 1:
            slots[r, 3] = sorted_columns.index(r + 1)
        slots[r, 4] = sorted_columns.index(r)
        columns.extend(sorted_columns)
        row_ptr.append(len(columns))
    return (
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        slots,
    )


def build_assembly_context(
    grid: Grid, ch: CHGeometry, comm: object,
) -> AssemblyContext:
    """Cache PETSc ownership ranges used by all matrices in one run."""
    physical_probe = PETSc.Mat().createAIJ(
        [grid.size, grid.size], nnz=5, comm=comm
    )
    physical_probe.setUp()
    physical_start, physical_end = physical_probe.getOwnershipRange()
    physical_probe.destroy()

    if ch.active_rows.size == 0:
        return AssemblyContext(
            RowRange(physical_start, physical_end),
            RowRange(physical_start, physical_end),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    augmented_size = grid.size + grid.N_p
    augmented_probe = PETSc.Mat().createAIJ(
        [augmented_size, augmented_size], nnz=grid.N_xi + 6, comm=comm
    )
    augmented_probe.setUp()
    augmented_start, augmented_end = augmented_probe.getOwnershipRange()
    augmented_probe.destroy()
    row_ptr = [0]
    columns: list[int] = []
    for row in range(augmented_start, augmented_end):
        if row < grid.size:
            start = grid.operator_row_ptr[row]
            end = grid.operator_row_ptr[row + 1]
            columns.extend(grid.operator_columns[start:end].tolist())
            ch_entry = ch.row_index[row]
            if ch_entry >= 0:
                columns.append(grid.size + int(ch.primary_radial[ch_entry]))
        else:
            k = row - grid.size
            columns.extend(range(k*grid.N_xi, (k + 1)*grid.N_xi))
            columns.append(grid.size + k)
        row_ptr.append(len(columns))
    return AssemblyContext(
        RowRange(physical_start, physical_end),
        RowRange(augmented_start, augmented_end),
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
    )


def derive_plasma(cfg: PlasmaConfig) -> DerivedPlasma:
    if cfg.T_e_eV <= 0.0:
        raise ValueError("electron temperature must be positive")
    n_e = sum(
        float(np.dot(np.arange(ion.Z + 1, dtype=np.float64), ion.n))
        for ion in cfg.ions
    )
    charge_square = sum(
        float(np.dot(np.arange(ion.Z + 1, dtype=np.float64)**2, ion.n))
        for ion in cfg.ions
    )
    if n_e <= 0.0:
        raise ValueError("electron density must be positive")
    Z_eff = charge_square/n_e
    Theta = cfg.T_e_eV/MEC2_EV
    lnLambda0 = 14.9 - 0.5*np.log(n_e/1.0e20) + np.log(cfg.T_e_eV/1.0e3)
    tau_c = (
        4.0*np.pi*constants.epsilon_0**2*constants.m_e**2*constants.c**3
        /(constants.e**4*n_e*lnLambda0)
    )
    tau_syn = (
        np.inf if cfg.B == 0.0 else
        6.0*np.pi*constants.epsilon_0*constants.m_e**3*constants.c**3
        /(constants.e**4*cfg.B**2)
    )
    alpha = 0.0 if cfg.B == 0.0 else tau_c/tau_syn
    E_bar = constants.e*cfg.E_parallel*tau_c/(constants.m_e*constants.c)
    return DerivedPlasma(n_e, Z_eff, Theta, lnLambda0, tau_c,
                         tau_syn, alpha, E_bar)


def nonthermal_cutoff_momentum(
    plasma: DerivedPlasma,
    energy_factor: float = NONTHERMAL_ENERGY_FACTOR,
) -> float:
    """Return p for the fixed nonthermal energy cutoff factor*T_e."""
    cutoff_energy_eV = energy_factor*plasma.Theta*MEC2_EV
    cutoff_gamma = 1.0 + cutoff_energy_eV/MEC2_EV
    return math.sqrt(max(0.0, cutoff_gamma**2 - 1.0))


def build_coupled_scales(
    plasma: DerivedPlasma, area: float, major_radius: float,
) -> CoupledScales:
    if area <= 0.0 or major_radius <= 0.0:
        raise ValueError("plasma area and major radius must be positive")
    # Spitzer resistivity fit, with temperature taken from the active plasma
    # configuration rather than silently assuming 10 eV.
    T_e_eV = plasma.Theta*MEC2_EV
    eta = 1.65e-9*plasma.Z_eff*plasma.lnLambda0/(T_e_eV/1.0e3)**1.5
    ebar_per_Vm = constants.e*plasma.tau_c/(constants.m_e*constants.c)
    kappa_I = plasma.tau_c*2.0*np.pi*major_radius/(constants.mu_0*major_radius)
    return CoupledScales(area, eta, ebar_per_Vm, kappa_I)


def collision_coefficients(
    p: np.ndarray, cfg: PlasmaConfig, plasma: DerivedPlasma,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    gamma = np.sqrt(1.0 + p**2)
    x = p/(gamma*np.sqrt(2.0*plasma.Theta))
    Phi = special.erf(x)
    Psi = (Phi - 2.0*x*np.exp(-x**2)/np.sqrt(np.pi))/(2.0*x**2)

    k = 5
    lnLambda_ee = plasma.lnLambda0 + np.log1p(
        ((gamma - 1.0)/plasma.Theta)**(k/2)
    )/k
    lnLambda_ei = plasma.lnLambda0 + np.log1p(
        (2.0*p/np.sqrt(2.0*plasma.Theta))**k
    )/k

    h = np.zeros_like(p)
    g = np.zeros_like(p)
    for ion in cfg.ions:
        for z in range(ion.Z + 1):
            N_bound = ion.Z - z
            if ion.n[z] == 0.0 or N_bound == 0:
                continue
            I_bar = ion.I_eV[z]/MEC2_EV
            q = p*np.sqrt(gamma - 1.0)/I_bar
            y = (ion.a_bar[z]*p)**1.5
            h += ion.n[z]/plasma.n_e*N_bound*(
                np.log1p(q**k)/k - p**2/gamma**2
            )
            g += ion.n[z]/plasma.n_e*(
                (2.0/3.0)*(ion.Z**2 - z**2)*np.log1p(y)
                - (2.0/3.0)*N_bound**2*y/(1.0 + y)
            )

    C_F = (lnLambda_ee*Psi/plasma.Theta + gamma**2*h/p**2)/plasma.lnLambda0
    C_A = gamma*Psi/p
    nu_D = gamma/(p**3*plasma.lnLambda0)*(
        plasma.Z_eff*lnLambda_ei
        + lnLambda_ee*(Phi - Psi + plasma.Theta*p**2/gamma**2)
        + g
    )
    return C_F, C_A, nu_D


def build_collision_data(
    grid: Grid, cfg: PlasmaConfig, plasma: DerivedPlasma,
) -> CollisionData:
    C_F_face, C_A_face, _ = collision_coefficients(grid.p_face, cfg, plasma)
    _, _, nu_D_center = collision_coefficients(grid.p_center, cfg, plasma)
    return CollisionData(C_F_face, C_A_face, nu_D_center)


@njit(cache=True)
def chang_cooper_face_terms(
    drift: float, diffusion: float, distance: float,
    drift_derivative: float,
) -> tuple[float, float, float, float]:
    """Return flux coefficients and their electric-field derivatives."""
    w = drift*distance/diffusion
    if abs(w) < 1.0e-4:
        delta = 0.5 - w/12.0 + w**3/720.0 - w**5/30240.0
        delta_prime = -1.0/12.0 + w*w/240.0 - w**4/6048.0
    elif w > 50.0:
        delta = 1.0/w
        delta_prime = -1.0/w**2
    elif w < -50.0:
        delta = 1.0 + 1.0/w
        delta_prime = -1.0/w**2
    else:
        expm1_w = math.expm1(w)
        delta = 1.0/w - 1.0/expm1_w
        delta_prime = -1.0/w**2 + math.exp(w)/expm1_w**2
    ddelta = delta_prime*distance/diffusion*drift_derivative
    left = drift*(1.0 - delta) + diffusion/distance
    right = drift*delta - diffusion/distance
    dleft = drift_derivative*(1.0 - delta) - drift*ddelta
    dright = drift_derivative*delta + drift*ddelta
    return left, right, dleft, dright


@njit(cache=True)
def fill_operator_values(
    row_ptr: np.ndarray, columns: np.ndarray,
    row_start: int, row_end: int, N_p: int, Nxi: int,
    p_face: np.ndarray, p_center: np.ndarray, xi_center: np.ndarray,
    xi_face: np.ndarray,
    operator_slots: np.ndarray,
    d_p: np.ndarray, d_xi: np.ndarray, d_xi_face: np.ndarray,
    radial_volume: np.ndarray, cell_volume: np.ndarray,
    C_F_face: np.ndarray, C_A_face: np.ndarray,
    nu_D_center: np.ndarray, E_bar: float, alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill local CSR values for FP transport and dL/dE_bar."""
    first = row_ptr[row_start]
    last = row_ptr[row_end]
    local_row_ptr = row_ptr[row_start:row_end + 1] - first
    local_columns = columns[first:last]
    values = np.empty(last - first, dtype=np.float64)
    derivatives = np.zeros(last - first, dtype=np.float64)

    for r in range(row_start, row_end):
        i = r // Nxi
        j = r - i*Nxi
        p = p_center[i]
        gamma = math.sqrt(1.0 + p*p)
        xi = xi_center[j]
        row_start_offset = row_ptr[r] - first
        diag = 0.0
        ddiag = 0.0

        if i > 0:
            pf = p_face[i]
            gammaf = math.sqrt(1.0 + pf*pf)
            drift = (-E_bar*xi - C_F_face[i]
                     - alpha*gammaf*pf*(1.0 - xi*xi))
            diffusion = C_A_face[i]
            left, right, dleft, dright = chang_cooper_face_terms(
                drift, diffusion, d_p[i], -xi
            )
            K = 2.0*math.pi*pf**2*d_xi[j]/cell_volume[r]
            pos = row_start_offset + operator_slots[r, 0]
            values[pos] = K*left
            derivatives[pos] = K*dleft
            diag += K*right
            ddiag += K*dright

        if i < N_p - 1:
            pf = p_face[i + 1]
            gammaf = math.sqrt(1.0 + pf*pf)
            drift = (-E_bar*xi - C_F_face[i + 1]
                     - alpha*gammaf*pf*(1.0 - xi*xi))
            diffusion = C_A_face[i + 1]
            left, right, dleft, dright = chang_cooper_face_terms(
                drift, diffusion, d_p[i + 1], -xi
            )
            K = 2.0*math.pi*pf**2*d_xi[j]/cell_volume[r]
            diag -= K*left
            ddiag -= K*dleft
            pos = row_start_offset + operator_slots[r, 1]
            values[pos] = -K*right
            derivatives[pos] = -K*dright
        else:
            # Open upper boundary: particles leave only through positive
            # radial drift.  The diffusive flux is set to zero at p_max;
            # making p_max large then gives the intended effectively open
            # domain without an artificial diffusive sink.
            pf = p_face[-1]
            gammaf = math.sqrt(1.0 + pf*pf)
            drift = (-E_bar*xi - C_F_face[-1]
                     - alpha*gammaf*pf*(1.0 - xi*xi))
            K = 2.0*math.pi*pf**2*d_xi[j]/cell_volume[r]
            if drift > 0.0:
                diag -= K*drift
                ddiag += K*xi

        if i == 0 and p_face[0] > 0.0:
            # A shifted grid represents a truncated domain.  Use an absorbing
            # lower boundary with f=0 outside the domain.  Apply the same
            # Chang--Cooper face flux as in the interior, with the exterior
            # (left) state set to zero; this consistently handles both drift
            # and diffusion at the cutoff.
            pf = p_face[0]
            gammaf = math.sqrt(1.0 + pf*pf)
            drift = (-E_bar*xi - C_F_face[0]
                     - alpha*gammaf*pf*(1.0 - xi*xi))
            diffusion = C_A_face[0]
            _, right, _, dright = chang_cooper_face_terms(
                drift, diffusion, d_p[0], -xi
            )
            K = 2.0*math.pi*pf**2*d_xi[j]/cell_volume[r]
            diag += K*right
            ddiag += K*dright

        if j > 0:
            xif = xi_face[j]
            drift = (1.0 - xif*xif)*(-E_bar/p + alpha*xif/gamma)
            diffusion = 0.5*nu_D_center[i]*(1.0 - xif*xif)
            left, right, dleft, dright = chang_cooper_face_terms(
                drift, diffusion, d_xi_face[j], -(1.0 - xif*xif)/p
            )
            K = radial_volume[i]/cell_volume[r]
            pos = row_start_offset + operator_slots[r, 2]
            values[pos] = K*left
            derivatives[pos] = K*dleft
            diag += K*right
            ddiag += K*dright

        if j < Nxi - 1:
            xif = xi_face[j + 1]
            drift = (1.0 - xif*xif)*(-E_bar/p + alpha*xif/gamma)
            diffusion = 0.5*nu_D_center[i]*(1.0 - xif*xif)
            left, right, dleft, dright = chang_cooper_face_terms(
                drift, diffusion, d_xi_face[j + 1], -(1.0 - xif*xif)/p
            )
            K = radial_volume[i]/cell_volume[r]
            diag -= K*left
            ddiag -= K*dleft
            pos = row_start_offset + operator_slots[r, 3]
            values[pos] = -K*right
            derivatives[pos] = -K*dright

        pos = row_start_offset + operator_slots[r, 4]
        values[pos] = diag
        derivatives[pos] = ddiag

    return local_row_ptr, local_columns, values, derivatives


def ch_knockon_dsigma_bar_dp(primary_p: np.ndarray, secondary_p: np.ndarray) -> np.ndarray:
    gamma1 = np.sqrt(1.0 + primary_p**2)
    gamma = np.sqrt(1.0 + secondary_p**2)
    beta = secondary_p/gamma
    eps1 = gamma1 - 1.0
    eps = gamma - 1.0
    x = eps1**2/(eps*(gamma1 - gamma))
    return 2.0*np.pi*beta*gamma1**2/(eps1**3*(gamma1 + 1.0))* (
        x**2 - 3.0*x + (eps1/gamma1)**2*(1.0 + x)
    )


def build_ch_geometry(
    grid: Grid, cfg: PlasmaConfig, plasma: DerivedPlasma,
) -> CHGeometry:
    P, XI = np.meshgrid(grid.p_center, grid.xi_center, indexing="ij")
    gamma = np.sqrt(1.0 + P**2)
    eps = gamma - 1.0
    # For the current nonthermal model, the CH cutoff is the kinetic-domain
    # cutoff.  A separate p_m would define a second, inconsistent threshold.
    p_m = grid.p_face[0]
    eps_m = np.sqrt(1.0 + p_m**2) - 1.0
    den = 1.0 + XI**2 - gamma*(1.0 - XI**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        primary = -2.0*P*XI/den
    mask = (
        (eps >= eps_m) & (XI < 0.0) & (den > 0.0)
        & np.isfinite(primary)
        & (primary >= grid.p_face[0]) & (primary < grid.p_face[-1])
    )
    i, j = np.nonzero(mask)
    primary_values = primary[i, j]
    eps1 = np.sqrt(1.0 + primary_values**2) - 1.0
    keep = eps[i, j] <= 0.5*eps1
    i, j = i[keep], j[keep]
    primary_values = primary_values[keep]
    if primary_values.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return CHGeometry(
            empty_i, empty_i.copy(), empty_f,
            np.full(grid.size, -1, dtype=np.int32),
        )

    target_p = grid.p_center[i]
    target_xi = grid.xi_center[j]
    primary_radial = np.searchsorted(
        grid.p_face, primary_values, side="right"
    ) - 1
    ds = ch_knockon_dsigma_bar_dp(primary_values, target_p)
    n_target = sum(ion.Z*float(np.sum(ion.n)) for ion in cfg.ions)
    pref = plasma.tau_c*n_target*constants.c*R_E**2
    coeff = pref*ds*primary_values**4/(target_p**2*np.abs(target_xi))
    rows = i*grid.N_xi + j
    good = np.isfinite(coeff) & (coeff > 0.0)
    active_rows = np.asarray(rows[good], dtype=np.int32)
    primary_radial = np.asarray(primary_radial[good], dtype=np.int32)
    row_coefficient = np.asarray(coeff[good], dtype=np.float64)
    row_index = np.full(grid.size, -1, dtype=np.int32)
    row_index[active_rows] = np.arange(active_rows.size, dtype=np.int32)
    return CHGeometry(active_rows, primary_radial, row_coefficient, row_index)


def empty_ch_geometry(grid: Grid) -> CHGeometry:
    """Return the exact zero large-angle operator geometry."""
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    return CHGeometry(
        empty_i, empty_i.copy(), empty_f,
        np.full(grid.size, -1, dtype=np.int32),
    )


def local_operator_rows(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    row_start: int, row_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local CSR rows for L and dL/dE_bar."""
    row_ptr, columns, values, derivatives = fill_operator_values(
        grid.operator_row_ptr, grid.operator_columns, row_start, row_end,
        grid.N_p, grid.N_xi, grid.p_face, grid.p_center, grid.xi_center,
        grid.xi_face, grid.operator_slots,
        grid.d_p, grid.d_xi, grid.d_xi_face, grid.radial_volume,
        grid.cell_volume,
        coll.C_F_face, coll.C_A_face, coll.nu_D_center,
        plasma.E_bar, plasma.alpha,
    )
    # The flux kernel is written in reconstructed density f.  Transform the
    # operator to the finite-volume content N = V f before handing it to
    # PETSc: L_N = D_V L_f D_V^{-1}.
    row_scale = np.repeat(grid.cell_volume[row_start:row_end], np.diff(row_ptr))
    values *= row_scale/grid.cell_volume[columns]
    derivatives *= row_scale/grid.cell_volume[columns]
    return row_ptr, columns, values, derivatives


def petsc_matrix_from_rows(
    grid: Grid, row_ptr: np.ndarray, columns: np.ndarray,
    values: np.ndarray, comm: object,
) -> PETSc.Mat:
    mat = PETSc.Mat().createAIJ(
        [grid.size, grid.size], nnz=5, comm=comm
    )
    mat.setUp()
    mat.setValuesCSR(row_ptr, columns, values)
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def create_distributed_operator(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    comm: object, assembly: AssemblyContext,
) -> tuple[PETSc.Mat, PETSc.Mat]:
    row_start = assembly.physical.start
    row_end = assembly.physical.end
    row_ptr, columns, values, derivatives = local_operator_rows(
        grid, plasma, coll, row_start, row_end
    )
    return (
        petsc_matrix_from_rows(grid, row_ptr, columns, values, comm),
        petsc_matrix_from_rows(grid, row_ptr, columns, derivatives, comm),
    )


def local_vector_from_array(values: np.ndarray, comm: object) -> PETSc.Vec:
    vec = PETSc.Vec().createMPI(values.size, comm=comm)
    lo, hi = vec.getOwnershipRange()
    local = vec.getArray()
    local[:] = np.asarray(values[lo:hi], dtype=np.float64)
    return vec


def destroy_petsc(*objects: object | None) -> None:
    """Destroy non-null PETSc objects on both normal and error paths."""
    for obj in objects:
        if obj is not None:
            obj.destroy()


def initial_distribution(grid: Grid) -> np.ndarray:
    P, XI = np.meshgrid(grid.p_center, grid.xi_center, indexing="ij")
    f = np.exp(-0.5*((P - 5.0)/0.5)**2 - 0.5*((XI + 1.0)/0.05)**2)
    f /= np.sum(grid.cell_volume.reshape(grid.N_p, grid.N_xi)*f)
    return f.ravel()


def make_mumps_solver(mat: PETSc.Mat, comm: object) -> PETSc.KSP:
    """Create PETSc direct solver using MUMPS."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp


def create_augmented_step_matrix(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    ch: CHGeometry, comm: object, assembly: AssemblyContext,
    alpha: float, operator_scale: float,
) -> PETSc.Mat:
    """Build exact CH auxiliary system for one implicit linear step.

    Physical rows use ``alpha*I - operator_scale*L_FP`` and one coupling to
    auxiliary radial content ``M_k = sum_j N[k,j]``.  Auxiliary rows impose
    ``M_k - sum_j N[k,j] = 0``.  The graph has O(Np*Nxi) entries.
    """
    physical_n = grid.size
    system_n = physical_n + grid.N_p
    mat = PETSc.Mat().createAIJ(
        [system_n, system_n], nnz=grid.N_xi + 6, comm=comm
    )
    mat.setUp()
    row_start = assembly.augmented.start
    row_end = assembly.augmented.end
    row_ptr = assembly.augmented_row_ptr
    columns = assembly.augmented_columns
    values = np.empty(columns.size, dtype=np.float64)
    physical_end = min(row_end, physical_n)
    if row_start < physical_end:
        base_row_ptr, _, base_values, _ = local_operator_rows(
            grid, plasma, coll, row_start, physical_end
        )
        for offset in range(physical_end-row_start):
            r = row_start + offset
            base_start = base_row_ptr[offset]
            base_end = base_row_ptr[offset+1]
            start = row_ptr[offset]
            base_columns = columns[start:start + base_end - base_start]
            values[start:start + base_end - base_start] = (
                -operator_scale*base_values[base_start:base_end]
            )
            diagonal = np.flatnonzero(base_columns == r)
            values[start + int(diagonal[0])] += alpha
            ch_entry = ch.row_index[r]
            if ch_entry >= 0:
                k = ch.primary_radial[ch_entry]
                coeff = ch.row_coefficient[ch_entry]
                values[start + base_end - base_start] = (
                    -operator_scale*grid.cell_volume[r]
                    *coeff/grid.radial_volume[k]
                )
    auxiliary_start = max(row_start, physical_n)
    for r in range(auxiliary_start, row_end):
        k = r - physical_n
        start = row_ptr[r-row_start]
        values[start:start + grid.N_xi] = -1.0
        values[start + grid.N_xi] = 1.0
    mat.setValuesCSR(row_ptr, columns, values)
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def augmented_initial_vector(grid: Grid, comm: object) -> PETSc.Vec:
    f = initial_distribution(grid)
    contents = grid.cell_volume*f
    masses = np.sum(contents.reshape(grid.N_p, grid.N_xi), axis=1)
    return local_vector_from_array(np.concatenate((contents, masses)), comm)


def zero_auxiliary_rhs(rhs: PETSc.Vec, physical_n: int) -> None:
    row_start, row_end = rhs.getOwnershipRange()
    values = rhs.getArray()
    for local, global_index in enumerate(range(row_start, row_end)):
        if global_index >= physical_n:
            values[local] = 0.0


def create_block_scatter(
    source: PETSc.Vec, target: PETSc.Vec, source_offset: int = 0,
    target_offset: int = 0, block_size: int | None = None,
) -> PETSc.Scatter:
    if block_size is None:
        block_size = min(source.getSize() - source_offset,
                         target.getSize() - target_offset)
    source_is = PETSc.IS().createStride(
        block_size, first=source_offset, step=1, comm=source.comm
    )
    target_is = PETSc.IS().createStride(
        block_size, first=target_offset, step=1, comm=target.comm
    )
    scatter = PETSc.Scatter().create(source, source_is, target, target_is)
    source_is.destroy()
    target_is.destroy()
    return scatter


def scatter_to_combined(
    source: PETSc.Vec, target: PETSc.Vec, scatter: PETSc.Scatter,
) -> None:
    target.set(0.0)
    scatter.scatter(source, target)


def bdf_coefficients(
    dtau: float, order: int, previous_dtau: float | None,
) -> tuple[float, float, float]:
    """Return derivative coefficient and right-hand-side weights."""
    if order == 1:
        return 1.0/dtau, 1.0/dtau, 0.0
    ratio = dtau/previous_dtau
    alpha = (1.0 + 2.0*ratio)/((1.0 + ratio)*dtau)
    return alpha, (1.0 + ratio)/dtau, -ratio**2/((1.0 + ratio)*dtau)


def augmented_stage_rhs(
    x_n: PETSc.Vec, x_nm1: PETSc.Vec, dtau: float,
    order: int, previous_dtau: float | None, physical_n: int,
) -> tuple[PETSc.Vec, float, np.ndarray, np.ndarray]:
    alpha, current_weight, previous_weight = bdf_coefficients(
        dtau, order, previous_dtau
    )
    rhs = x_n.copy()
    rhs.scale(current_weight)
    if previous_weight != 0.0:
        rhs.axpy(previous_weight, x_nm1)
    zero_auxiliary_rhs(rhs, physical_n)
    return rhs, alpha


def coupled_stage_residual(
    x: PETSc.Vec, u: float, v: float,
    x_n: PETSc.Vec, x_nm1: PETSc.Vec, u_n: float, u_nm1: float,
    dtau: float, order: int, previous_dtau: float | None,
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData, ch: CHGeometry,
    current_weight: PETSc.Vec, f_scatter: PETSc.Scatter,
    physical_work: PETSc.Vec,
    scales: CoupledScales, comm: object, assembly: AssemblyContext,
) -> CoupledStageResidual:
    """Assemble residual and Jacobian for augmented state ``[f, M]``.

    ``u`` stores scaled plasma current and ``v`` stores scaled parallel
    electric field.  Chiu--Harvey source is represented through ``M``.
    """
    rhs, alpha = augmented_stage_rhs(
        x_n, x_nm1, dtau, order, previous_dtau, grid.size
    )
    E = scales.E_scale*v
    L, dL_dEbar = create_distributed_operator(
        grid, replace(plasma, E_bar=scales.ebar_per_Vm*E), coll, comm, assembly
    )
    A = create_augmented_step_matrix(
        grid, replace(plasma, E_bar=scales.ebar_per_Vm*E), coll, ch,
        comm, assembly, alpha, 1.0
    )
    F = x.duplicate()
    A.mult(x, F)
    F.axpy(-1.0, rhs)
    scatter_to_combined(x, physical_work, f_scatter)
    j_RE = distributed_dot(current_weight, physical_work)
    F_u = alpha*u - (alpha*u_n if order == 1 else
                     ((1.0 + dtau/previous_dtau)*u_n
                      - (dtau/previous_dtau)**2/(1.0 + dtau/previous_dtau)*u_nm1)/dtau)
    F_u += scales.induction_coefficient*v
    F_v = v - scales.bulk_current_coefficient*u + scales.eta/scales.E_scale*j_RE
    rhs.destroy()
    return CoupledStageResidual(F, F_u, F_v, A, dL_dEbar, L, alpha)


def solve_coupled_stage(
    x_n: PETSc.Vec, x_nm1: PETSc.Vec, u_n: float, u_nm1: float,
    v_guess: float, dtau: float, order: int, previous_dtau: float | None,
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData, ch: CHGeometry,
    current_weight: PETSc.Vec, f_scatter: PETSc.Scatter,
    physical_work: PETSc.Vec,
    scales: CoupledScales, comm: object, assembly: AssemblyContext,
    max_iter: int = 12,
) -> tuple[PETSc.Vec, float, float, int]:
    """Solve one implicit BE/BDF2 stage with bordered Newton iteration."""
    x = x_n.copy()
    u = float(u_n)
    v = float(v_guess)
    c_weight = current_weight.copy()
    c_weight.scale(scales.eta/scales.E_scale)
    d_bulk = scales.bulk_current_coefficient

    for iteration in range(max_iter):
        try:
            stage = coupled_stage_residual(
                x, u, v, x_n, x_nm1, u_n, u_nm1, dtau, order, previous_dtau,
                grid, plasma, coll, ch, current_weight, f_scatter,
                physical_work, scales, comm, assembly,
            )
        except Exception:
            # The trial state is owned by this routine until convergence.
            x.destroy()
            c_weight.destroy()
            raise
        F = stage.residual
        Fu = stage.current
        Fv = stage.electric_field
        A = stage.matrix
        dL_dEbar = stage.d_operator_dE
        L = stage.fp_operator
        alpha = stage.derivative_coefficient
        norm = scaled_residual_norm(F, (Fu, Fv), alpha)
        if norm < 1.0e-9:
            for obj in (F, A, dL_dEbar, L):
                obj.destroy()
            c_weight.destroy()
            return x, u, v, iteration + 1

        ksp = minus_F = dx0 = qf = q = dxE = f_x = f_y = dx = None
        try:
            ksp = make_mumps_solver(A, comm)
            minus_F = F.copy()
            minus_F.scale(-1.0)
            dx0 = x.duplicate()
            ksp.solve(minus_F, dx0)

            scatter_to_combined(x, physical_work, f_scatter)
            qf = PETSc.Vec().createMPI(grid.size, comm=comm)
            dL_dEbar.mult(physical_work, qf)
            qf.scale(-scales.ebar_per_Vm*scales.E_scale)
            q = x.duplicate()
            q.set(0.0)
            reverse_scatter = create_block_scatter(qf, q, 0, 0, grid.size)
            reverse_scatter.scatter(qf, q)
            reverse_scatter.destroy()
            dxE = x.duplicate()
            ksp.solve(q, dxE)
            f_x = PETSc.Vec().createMPI(grid.size, comm=comm)
            f_y = PETSc.Vec().createMPI(grid.size, comm=comm)
            scatter_to_combined(dx0, f_x, f_scatter)
            scatter_to_combined(dxE, f_y, f_scatter)
            c_x = distributed_dot(c_weight, f_x)
            c_y = distributed_dot(c_weight, f_y)
            beta = scales.induction_coefficient
            rhs_small = np.asarray([-Fu, -Fv-c_x], dtype=np.float64)
            schur = np.asarray([[alpha, beta], [-d_bulk, 1.0-c_y]])
            try:
                du, dv = np.linalg.solve(schur, rhs_small)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "coupled scalar Schur complement is singular"
                ) from exc
            dx = dx0.copy()
            dx.axpy(-dv, dxE)
        except Exception:
            destroy_petsc(ksp, minus_F, dx0, qf, q, dxE, f_x, f_y, dx)
            destroy_petsc(F, A, dL_dEbar, L)
            x.destroy()
            c_weight.destroy()
            raise

        damping = 1.0
        accepted = False
        while damping >= 1.0e-5:
            trial = x.copy()
            trial.axpy(damping, dx)
            trial_u = u + damping*du
            trial_v = v + damping*dv
            local_trial = trial.getArray(readonly=True)
            local_f_end = min(trial.getOwnershipRange()[1], grid.size)
            local_f_start = min(max(trial.getOwnershipRange()[0], 0), grid.size)
            local_f = (
                local_trial[:max(0, local_f_end-local_f_start)]
                / grid.cell_volume[local_f_start:local_f_end]
            )
            local_min = float(np.min(local_f)) if local_f.size else np.inf
            global_min = comm.tompi4py().allreduce(local_min, op=MPI.MIN)
            if np.isfinite(trial_u) and np.isfinite(trial_v) and global_min >= -1.0e-12:
                try:
                    trial_stage = coupled_stage_residual(
                        trial, trial_u, trial_v, x_n, x_nm1, u_n, u_nm1,
                        dtau, order, previous_dtau, grid, plasma, coll, ch,
                        current_weight, f_scatter, physical_work, scales,
                        comm, assembly,
                    )
                except Exception:
                    trial.destroy()
                    destroy_petsc(F, A, dL_dEbar, L, ksp, minus_F, dx0,
                                  qf, q, dxE, f_x, f_y, dx)
                    x.destroy()
                    c_weight.destroy()
                    raise
                trial_F = trial_stage.residual
                trial_Fu = trial_stage.current
                trial_Fv = trial_stage.electric_field
                trial_norm = scaled_residual_norm(
                    trial_F, (trial_Fu, trial_Fv), alpha
                )
                for obj in (trial_F, trial_stage.matrix,
                            trial_stage.d_operator_dE, trial_stage.fp_operator):
                    obj.destroy()
                if trial_norm < norm:
                    x.destroy()
                    x = trial
                    u = trial_u
                    v = trial_v
                    accepted = True
                    break
            trial.destroy()
            damping *= 0.5

        destroy_petsc(F, A, dL_dEbar, L, ksp, minus_F, dx0, qf, q, dxE,
                      f_x, f_y, dx)
        if not accepted:
            x.destroy()
            c_weight.destroy()
            raise RuntimeError("coupled Newton damping failed")

    x.destroy()
    c_weight.destroy()
    raise RuntimeError("coupled Newton did not converge")


def trbdf2_stage_one_rhs(
    stage_mat: PETSc.Mat, state: PETSc.Vec, physical_n: int,
) -> PETSc.Vec:
    """Build (I + d h L) state as 2 state - (I - d h L) state."""
    stage_state = state.duplicate()
    stage_mat.mult(state, stage_state)
    rhs = state.copy()
    rhs.scale(2.0)
    rhs.axpy(-1.0, stage_state)
    zero_auxiliary_rhs(rhs, physical_n)
    stage_state.destroy()
    return rhs


def trbdf2_stage_two_rhs(
    stage_mat: PETSc.Mat, state_n: PETSc.Vec, state_gamma: PETSc.Vec,
    d: float, dt: float, physical_n: int,
) -> PETSc.Vec:
    """Build the TR--BDF2 endpoint right-hand side."""
    mat_state_n = state_n.duplicate()
    mat_state_gamma = state_gamma.duplicate()
    stage_mat.mult(state_n, mat_state_n)
    stage_mat.mult(state_gamma, mat_state_gamma)
    rhs = state_n.copy()
    row_start, row_end = rhs.getOwnershipRange()
    local_rhs = rhs.getArray()
    local_n = state_n.getArray(readonly=True)
    local_gamma = state_gamma.getArray(readonly=True)
    local_mat_n = mat_state_n.getArray(readonly=True)
    local_mat_gamma = mat_state_gamma.getArray(readonly=True)
    start = max(row_start, 0)
    end = min(row_end, physical_n)
    if start < end:
        sl = slice(0, end-start)
        L_n = (local_n[sl] - local_mat_n[sl])/(d*dt)
        L_gamma = (local_gamma[sl] - local_mat_gamma[sl])/(d*dt)
        local_rhs[sl] += dt/(2.0*math.sqrt(2.0))*(L_n + L_gamma)
    zero_auxiliary_rhs(rhs, physical_n)
    mat_state_n.destroy()
    mat_state_gamma.destroy()
    return rhs


def create_physical_step_matrix(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    comm: object, assembly: AssemblyContext, alpha: float,
    operator_scale: float,
) -> PETSc.Mat:
    """Build alpha*I - operator_scale*L for the CH-disabled physical state."""
    operator, _ = create_distributed_operator(
        grid, plasma, coll, comm, assembly
    )
    operator.scale(-operator_scale)
    operator.shift(alpha)
    return operator


def physical_initial_vector(grid: Grid, comm: object) -> PETSc.Vec:
    """Return the initial physical cell-content state without CH auxiliaries."""
    contents = grid.cell_volume*initial_distribution(grid)
    return local_vector_from_array(contents, comm)


def solve_constant_case_without_ch(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    t_end: float, steps: int, comm: object,
    assembly: AssemblyContext,
) -> tuple[PETSc.Vec, float, np.ndarray, np.ndarray]:
    """Advance the CH-disabled physical state with fixed-step TR--BDF2."""
    dt = t_end/steps
    d = 1.0 - 1.0/math.sqrt(2.0)
    stage_mat = create_physical_step_matrix(
        grid, plasma, coll, comm, assembly, 1.0, d*dt
    )
    stage_solver = make_mumps_solver(stage_mat, comm)
    x_n = physical_initial_vector(grid, comm)
    history_times = [0.0]
    history_mass = [distributed_content_sum(x_n, grid.size, comm)]
    for step in range(steps):
        rhs_gamma = trbdf2_stage_one_rhs(stage_mat, x_n, grid.size)
        x_gamma = x_n.duplicate()
        stage_solver.solve(rhs_gamma, x_gamma)
        rhs_endpoint = trbdf2_stage_two_rhs(
            stage_mat, x_n, x_gamma, d, dt, grid.size
        )
        x_new = x_n.duplicate()
        stage_solver.solve(rhs_endpoint, x_new)
        rhs_gamma.destroy()
        rhs_endpoint.destroy()
        x_gamma.destroy()
        x_n.destroy()
        x_n = x_new
        history_times.append((step + 1)*dt)
        history_mass.append(distributed_content_sum(x_n, grid.size, comm))
    stage_mat.destroy()
    stage_solver.destroy()
    return x_n, dt, np.asarray(history_times), np.asarray(history_mass)


def solve_constant_case(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData,
    ch: CHGeometry, t_end: float, steps: int, comm: object,
    assembly: AssemblyContext,
) -> tuple[PETSc.Vec, float, np.ndarray, np.ndarray]:
    """Advance constant forward FP system with fixed-step TR--BDF2."""
    dt = t_end/steps
    d = 1.0 - 1.0/math.sqrt(2.0)
    stage_mat = create_augmented_step_matrix(
        grid, plasma, coll, ch, comm, assembly, 1.0, d*dt
    )
    stage_solver = make_mumps_solver(stage_mat, comm)
    x_n = augmented_initial_vector(grid, comm)
    history_times = [0.0]
    history_mass = [distributed_content_sum(x_n, grid.size, comm)]
    for step in range(steps):
        rhs_gamma = trbdf2_stage_one_rhs(stage_mat, x_n, grid.size)
        x_gamma = x_n.duplicate()
        stage_solver.solve(rhs_gamma, x_gamma)
        rhs_endpoint = trbdf2_stage_two_rhs(
            stage_mat, x_n, x_gamma, d, dt, grid.size
        )
        x_new = x_n.duplicate()
        stage_solver.solve(rhs_endpoint, x_new)
        rhs_gamma.destroy()
        rhs_endpoint.destroy()
        x_gamma.destroy()
        x_n.destroy()
        x_n = x_new
        history_times.append((step + 1)*dt)
        history_mass.append(distributed_content_sum(x_n, grid.size, comm))
    stage_mat.destroy()
    stage_solver.destroy()
    return x_n, dt, np.asarray(history_times), np.asarray(history_mass)


def make_current_weights(
    grid: Grid, plasma: DerivedPlasma, p_threshold: float | None = None,
) -> np.ndarray:
    """Return current-density weights, optionally integrating above p_threshold."""
    p_left = grid.p_face[:-1].copy()
    p_right = grid.p_face[1:]
    if p_threshold is not None:
        p_left = np.maximum(p_left, p_threshold)
    gamma_left = np.sqrt(1.0 + p_left**2)
    gamma_right = np.sqrt(1.0 + p_right**2)
    primitive = gamma_right**3/3.0 - gamma_right
    radial = primitive - (gamma_left**3/3.0 - gamma_left)
    radial[p_right <= p_left] = 0.0
    pitch = 0.5*(grid.xi_face[1:]**2 - grid.xi_face[:-1]**2)
    weights = (
        -constants.e*plasma.n_e*2.0*np.pi*constants.c
        * radial[:, None]*pitch[None, :]
    ).ravel()
    # The moment integral above is written for reconstructed f.  The PETSc
    # state is N, so divide by the cell phase volume to obtain j = w_N dot N.
    return weights/grid.cell_volume


def current_weight_for_seed(
    grid: Grid, plasma: DerivedPlasma, area: float,
) -> np.ndarray:
    """Return current-density weights normalized to a one-ampere seed."""
    seed = (grid.cell_volume*initial_distribution(grid)).reshape(
        grid.N_p, grid.N_xi
    )
    p_threshold = nonthermal_cutoff_momentum(plasma)
    if not math.isclose(p_threshold, grid.p_face[0], rel_tol=1.0e-12,
                        abs_tol=1.0e-15):
        raise ValueError("grid p_min must equal the nonthermal energy cutoff")
    unit_weight = make_current_weights(
        grid, plasma, p_threshold=p_threshold
    ).reshape(grid.N_p, grid.N_xi)
    seed_current = area*np.sum(unit_weight*seed)
    if not np.isfinite(seed_current) or seed_current <= 0.0:
        raise ValueError("initial seed must carry positive conventional current")
    return unit_weight.ravel()/seed_current


def distributed_dot(weight: PETSc.Vec, state: PETSc.Vec) -> float:
    return float(weight.dot(state))


def distributed_content_sum(state: PETSc.Vec, physical_n: int, comm: object) -> float:
    """Sum physical cell contents in a distributed augmented vector."""
    row_start, row_end = state.getOwnershipRange()
    start = max(row_start, 0)
    end = min(row_end, physical_n)
    local = state.getArray(readonly=True)
    local_physical = local[:max(0, end-start)] if start < end else np.empty(0)
    return float(comm.tompi4py().allreduce(
        float(np.sum(local_physical)), op=MPI.SUM
    ))


def scaled_residual_norm(
    residual: PETSc.Vec, scalar_residuals: tuple[float, float],
    derivative_coefficient: float,
) -> float:
    """Return a dimensionless nonlinear residual norm.

    FP and current rows are derivative equations and therefore scale with
    the BDF derivative coefficient; the electric-field algebraic row does
    not.  Combining their raw infinity norms made Newton convergence depend
    on units and on the timestep.
    """
    derivative_scale = max(1.0, abs(derivative_coefficient))
    return max(
        float(residual.norm(PETSc.NormType.NORM_INFINITY))/derivative_scale,
        abs(float(scalar_residuals[0]))/derivative_scale,
        abs(float(scalar_residuals[1])),
    )


def coupled_error_components(
    f_bdf2: PETSc.Vec, u_bdf2: float, v_bdf2: float,
    f_be: PETSc.Vec, u_be: float, v_be: float,
    comm: object, rtol: float = 1.0e-4, atol: float = 1.0e-10,
) -> CoupledErrorEstimate:
    a = f_bdf2.getArray(readonly=True)
    b = f_be.getArray(readonly=True)
    scale = atol + rtol*np.maximum(np.abs(a), np.abs(b))
    local_error = np.abs(a-b)/scale
    if local_error.size:
        local_index = int(np.argmax(local_error))
        local_f = float(local_error[local_index])
        row_start = f_bdf2.getOwnershipRange()[0]
        global_index = row_start + local_index
    else:
        local_f = 0.0
        global_index = -1
    candidates = comm.tompi4py().allgather((local_f, global_index))
    error_f, error_cell = max(candidates, key=lambda item: (item[0], -item[1]))
    error_ip = abs(u_bdf2-u_be)/(atol + rtol*max(abs(u_bdf2), abs(u_be)))
    error_e = abs(v_bdf2-v_be)/(atol + rtol*max(abs(v_bdf2), abs(v_be)))
    error = max(float(error_f), float(error_ip), float(error_e))
    return CoupledErrorEstimate(
        float(error), float(error_f), float(error_ip), float(error_e),
        int(error_cell) if error_cell >= 0 else None,
    )


def coupled_error_components_from_auxiliary(
    x_bdf2: PETSc.Vec, u_bdf2: float, v_bdf2: float,
    x_be: PETSc.Vec, u_be: float, v_be: float,
    f_bdf2: PETSc.Vec, f_be: PETSc.Vec,
    f_scatter: PETSc.Scatter, cell_volume: np.ndarray, comm: object,
) -> CoupledErrorEstimate:
    scatter_to_combined(x_bdf2, f_bdf2, f_scatter)
    scatter_to_combined(x_be, f_be, f_scatter)
    for state in (f_bdf2, f_be):
        row_start, row_end = state.getOwnershipRange()
        state.getArray()[:] /= cell_volume[row_start:row_end]
    return coupled_error_components(
        f_bdf2, u_bdf2, v_bdf2, f_be, u_be, v_be, comm
    )


def describe_error_cell(grid: Grid, cell: int | None) -> str:
    """Format the global cell that maximizes the distribution error."""
    if cell is None:
        return "f_cell=none"
    i, j = divmod(cell, grid.N_xi)
    return (f"f_cell={cell} (i={i}, j={j}, p={grid.p_center[i]:.6e}, "
            f"xi={grid.xi_center[j]:+.6e})")


def solve_coupled_case(
    grid: Grid, plasma: DerivedPlasma, coll: CollisionData, ch: CHGeometry,
    t_end: float, dtau_initial: float, area: float, major_radius: float,
    initial_current: float, comm: object, assembly: AssemblyContext,
) -> None:
    scales = build_coupled_scales(plasma, area, major_radius)
    current_array = current_weight_for_seed(grid, plasma, area)
    current_weight = local_vector_from_array(current_array, comm)
    x_n = augmented_initial_vector(grid, comm)
    x_nm1 = x_n.copy()
    physical_work = PETSc.Vec().createMPI(grid.size, comm=comm)
    error_bdf2 = PETSc.Vec().createMPI(grid.size, comm=comm)
    error_be = PETSc.Vec().createMPI(grid.size, comm=comm)
    f_scatter = create_block_scatter(x_n, physical_work, block_size=grid.size)
    scatter_to_combined(x_n, physical_work, f_scatter)
    u_n = initial_current/scales.I_scale
    u_nm1 = u_n
    j_initial = distributed_dot(current_weight, physical_work)
    E_initial = scales.eta*(initial_current/area-j_initial)
    v_n = E_initial/scales.E_scale
    time = 0.0
    previous_dtau = None
    dtau = dtau_initial
    dtau_min = max(1.0e-9, dtau/128.0)
    dtau_max = min(t_end/50.0, 4.0*dtau)
    accepted_steps = 0
    rejected_steps = 0
    step_sizes = []
    if comm.getRank() == 0:
        print(f"adaptive coupled PETSc auxiliary run: ranks={comm.getSize()} "
              f"grid={grid.N_p}x{grid.N_xi} t_end={t_end:g}")
        print(f"initial Ip={initial_current:.6e} A I_RE={area*j_initial:.6e} A "
              f"E={E_initial:.6e} V/m")
        print(f"boundaries: p_min absorbing, p_max open-drift; "
              f"p_m=p_min; "
              f"j_RE cutoff={NONTHERMAL_ENERGY_FACTOR*plasma.Theta*MEC2_EV:.6e} eV")

    while time < t_end-1.0e-15:
        remaining = t_end-time
        dtau = min(dtau, remaining)
        # Permit a shortened final step so the integrator lands on t_end.
        if dtau < dtau_min and remaining > dtau_min:
            raise RuntimeError("adaptive PETSc timestep reached dtau_min")
        if accepted_steps == 0:
            x_new = None
            try:
                x_new, u_new, v_new, iterations = solve_coupled_stage(
                    x_n, x_nm1, u_n, u_nm1, v_n, dtau, 1, None,
                    grid, plasma, coll, ch, current_weight, f_scatter,
                    physical_work, scales, comm, assembly,
                )
            except RuntimeError as exc:
                if x_new is not None:
                    x_new.destroy()
                if dtau <= dtau_min*(1.0 + 1.0e-12):
                    raise RuntimeError(
                        f"coupled Newton failed at dtau_min={dtau_min:.6e}: {exc}"
                    ) from exc
                next_dtau = max(dtau_min, 0.5*dtau)
                if comm.getRank() == 0:
                    print(f"reject BE time={time:.6e} dtau={dtau:.3e} "
                          f"next={next_dtau:.3e}: {exc}")
                dtau = next_dtau; rejected_steps += 1; continue
            error = error_f = error_ip = error_e = 0.0
            error_cell = "f_cell=n/a"
        else:
            x_new = None
            x_be = None
            try:
                x_new, u_new, v_new, iterations = solve_coupled_stage(
                    x_n, x_nm1, u_n, u_nm1, v_n, dtau, 2, previous_dtau,
                    grid, plasma, coll, ch, current_weight, f_scatter,
                    physical_work, scales, comm, assembly,
                )
                x_be, u_be, v_be, _ = solve_coupled_stage(
                    x_n, x_nm1, u_n, u_nm1, v_n, dtau, 1, None,
                    grid, plasma, coll, ch, current_weight, f_scatter,
                    physical_work, scales, comm, assembly,
                )
            except RuntimeError as exc:
                if x_new is not None:
                    x_new.destroy()
                if x_be is not None:
                    x_be.destroy()
                if dtau <= dtau_min*(1.0 + 1.0e-12):
                    raise RuntimeError(
                        f"coupled Newton failed at dtau_min={dtau_min:.6e}: {exc}"
                    ) from exc
                next_dtau = max(dtau_min, 0.5*dtau)
                if comm.getRank() == 0:
                    print(f"reject Newton time={time:.6e} dtau={dtau:.3e} "
                          f"next={next_dtau:.3e}: {exc}")
                dtau = next_dtau; rejected_steps += 1; continue
            error_estimate = coupled_error_components_from_auxiliary(
                x_new, u_new, v_new, x_be, u_be, v_be,
                error_bdf2, error_be, f_scatter, grid.cell_volume, comm
            )
            error = error_estimate.total
            error_f = error_estimate.distribution
            error_ip = error_estimate.current
            error_e = error_estimate.electric_field
            error_cell = describe_error_cell(
                grid, error_estimate.distribution_cell
            )
            if error > 1.0:
                factor = max(0.1, min(0.5, 0.9*error**(-1.0/2.0)))
                if dtau <= dtau_min*(1.0 + 1.0e-12):
                    x_new.destroy()
                    x_be.destroy()
                    raise RuntimeError(
                        f"adaptive error remains {error:.3e} at "
                        f"dtau_min={dtau_min:.6e}"
                    )
                next_dtau = max(dtau_min, dtau*factor)
                if comm.getRank() == 0:
                    print(f"reject error time={time:.6e} dtau={dtau:.3e} "
                          f"err={error:.3e} err_f={error_f:.3e} "
                          f"err_Ip={error_ip:.3e} err_E={error_e:.3e} "
                          f"{error_cell} "
                          f"next={next_dtau:.3e}")
                x_new.destroy(); x_be.destroy()
                dtau = next_dtau; rejected_steps += 1; continue
            x_be.destroy()

        old_nm1 = x_nm1
        x_nm1 = x_n
        x_n = x_new
        old_nm1.destroy()
        u_nm1 = u_n; u_n = u_new
        v_n = v_new
        previous_dtau = dtau
        time += dtau
        accepted_steps += 1
        step_sizes.append(dtau)
        scatter_to_combined(x_n, physical_work, f_scatter)
        j_current = distributed_dot(current_weight, physical_work)
        E_current = scales.E_scale*v_new
        if comm.getRank() == 0:
            print(f"{accepted_steps:5d} accepted time={time:.6e} dtau={dtau:.3e} "
                  f"err={error:.3e} err_f={error_f:.3e} err_Ip={error_ip:.3e} "
                  f"err_E={error_e:.3e} {error_cell} "
                  f"Newton={iterations} "
                  f"Ip={scales.I_scale*u_n:.6e} I_RE={area*j_current:.6e} "
                  f"E={E_current:.6e}")
        # BE--BDF2 differences are O(dtau^2), so use a square-root
        # controller for this estimator.
        factor = 2.0 if error == 0.0 else min(2.0, max(0.5, 0.9*error**(-1.0/2.0)))
        dtau = min(dtau_max, max(dtau_min, dtau*factor))

    if comm.getRank() == 0:
        print(f"final accepted/rejected={accepted_steps}/{rejected_steps} "
              f"dtau=[{min(step_sizes) if step_sizes else 0.0:.3e},"
              f" {max(step_sizes) if step_sizes else 0.0:.3e}]")
    f_scatter.destroy()
    physical_work.destroy()
    error_bdf2.destroy()
    error_be.destroy()
    x_nm1.destroy(); x_n.destroy(); current_weight.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmax", type=float, default=100.0)
    parser.add_argument("--E-parallel", type=float, default=0.0,
                        help="fixed parallel electric field for constant mode [V/m]")
    parser.add_argument("--Np", type=int, default=128)
    parser.add_argument("--Nxi", type=int, default=64)
    parser.add_argument("--t-end", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=320,
                        help="fixed TR-BDF2 steps for constant-parameter mode")
    parser.add_argument("--dtau-initial", type=float, default=None,
                        help="initial timestep for adaptive coupled mode")
    parser.add_argument("--coupled", action="store_true",
                        help="run adaptive electric-field/induction Newton coupling")
    parser.add_argument("--disable-ch", action="store_true",
                        help="disable the Chiu--Harvey large-angle gain")
    parser.add_argument("--area", type=float, default=np.pi*2.0**2)
    parser.add_argument("--major-radius", type=float, default=6.0)
    parser.add_argument("--initial-current", type=float, default=15.0e6)
    parser.add_argument("--state-output", type=str, default=None,
                        help="absolute .npz path for the final constant-mode state")
    parser.add_argument("--history-output", type=str, default=None,
                        help="absolute .npz path for constant-mode mass history")
    args = parser.parse_args()

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    D = IonSpecies(1, np.array([0.0, 1.0e20]),
                   np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    ne_I = np.zeros(11); ne_I[3] = 97.08
    ne_a = np.zeros(11); ne_a[3] = 80.0
    Ne = IonSpecies(10, np.array([0.0, 0.0, 0.0, 1.0e19,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    ne_I, ne_a)
    cfg = PlasmaConfig(FIXED_T_E_EV, args.E_parallel, 5.0, (D, Ne))
    plasma = derive_plasma(cfg)
    p_min = nonthermal_cutoff_momentum(plasma)
    grid = build_grid(GridConfig(p_min, args.pmax, args.Np, args.Nxi))
    coll = build_collision_data(grid, cfg, plasma)
    # The large-angle cutoff is the same fixed nonthermal cutoff for now.
    ch = (empty_ch_geometry(grid) if args.disable_ch
          else build_ch_geometry(grid, cfg, plasma))
    assembly = build_assembly_context(grid, ch, comm)
    if args.coupled:
        if args.disable_ch:
            raise ValueError(
                "CH-disabled coupled mode is not implemented; use constant mode"
            )
        if args.dtau_initial is None or args.dtau_initial <= 0.0 or args.t_end <= 0.0:
            raise ValueError(
                "coupled run requires t_end > 0 and --dtau-initial > 0"
            )
        if rank == 0:
            print(f"PETSc coupled run: ranks={comm.getSize()} "
                  f"grid={args.Np}x{args.Nxi} "
                  f"p=[{p_min:.6e}, {args.pmax:g}]")
        solve_coupled_case(
            grid, plasma, coll, ch, args.t_end, args.dtau_initial,
            args.area, args.major_radius, args.initial_current, comm, assembly,
        )
        return
    if args.disable_ch:
        x_current, dt, history_times, history_mass = solve_constant_case_without_ch(
            grid, plasma, coll, args.t_end, args.steps, comm, assembly
        )
    else:
        x_current, dt, history_times, history_mass = solve_constant_case(
            grid, plasma, coll, ch, args.t_end, args.steps, comm, assembly
        )
    row_start, row_end = x_current.getOwnershipRange()
    local = x_current.getArray(readonly=True)
    physical_end = min(row_end, grid.size)
    local_physical = (local[:max(0, physical_end-row_start)]
                      if row_start < physical_end else np.empty(0))
    weight_start = max(row_start, 0)
    weight_end = min(row_end, grid.size)
    local_mass = float(np.sum(local_physical))
    mpi_comm = comm.tompi4py()
    mass = mpi_comm.allreduce(local_mass, op=MPI.SUM)
    local_f = (
        local_physical/grid.cell_volume[weight_start:weight_end]
        if local_physical.size else np.empty(0)
    )
    local_min = float(np.min(local_f)) if local_f.size else np.inf
    minimum = mpi_comm.allreduce(local_min, op=MPI.MIN)
    if args.state_output is not None:
        state_chunks = mpi_comm.gather(
            (weight_start, np.array(local_physical, copy=True)), root=0
        )
        if rank == 0:
            state = np.empty(grid.size, dtype=np.float64)
            for start, values in state_chunks:
                state[start:start + values.size] = values
            np.savez(
                args.state_output,
                N=state,
                p_face=grid.p_face,
                xi_center=grid.xi_center,
                E_parallel=cfg.E_parallel,
                t_end=args.t_end,
            )
    if args.history_output is not None and rank == 0:
        np.savez(
            args.history_output,
            time=history_times,
            mass=history_mass,
            E_parallel=cfg.E_parallel,
            Np=args.Np,
            Nxi=args.Nxi,
            t_end=args.t_end,
            steps=args.steps,
        )
    if rank == 0:
        mode = "CH disabled physical" if args.disable_ch else "auxiliary CH"
        print(f"========== PETSc Forward FP ({mode}) ==========")
        print(f"MPI ranks: {comm.getSize()}")
        state_size = grid.size if args.disable_ch else grid.size + grid.N_p
        print(f"PETSc state size: {state_size}")
        print(f"Grid: Np={grid.N_p}, Nxi={grid.N_xi}, p=[{grid.p_face[0]:.3e}, {grid.p_face[-1]:g}]")
        print(f"PETSc: {PETSc.Sys.getVersion()}")
        print(f"TR-BDF2: t_end={args.t_end:g}, steps={args.steps}, dtau={dt:.6e}")
        print(f"Fixed E_parallel: {cfg.E_parallel:.6e} V/m")
        print(f"CH: {'disabled' if args.disable_ch else 'enabled'}; "
              f"active targets: {ch.active_rows.size:,}")
        print(f"Final represented density: {mass:.12e}")
        print(f"Final minimum f: {minimum:.6e}")
        print("=====================================================")
    x_current.destroy()


if __name__ == "__main__":
    main()
