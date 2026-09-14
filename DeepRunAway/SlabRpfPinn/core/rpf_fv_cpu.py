"""CPU finite-volume reference operators for the relativistic FP equation.

Notebook and production GPU code should use these functions as a transparent
CPU reference. State vectors contain cell-average distribution values, ordered
by ``i * N_xi + j``. The physical adjoint uses the cell-volume inner product.
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants, special, sparse
from scipy.optimize import brentq

MEC2_EV = constants.m_e * constants.c**2 / constants.e
R_E = constants.e**2 / (4 * np.pi * constants.epsilon_0 * constants.m_e * constants.c**2)

# Hesslow screening data. Entries index partially stripped charge states
# Z0=0,...,Z-1; fully stripped state receives zero bound-electron data.
HESSLOW_ABAR_BY_Z = {
    1: (190.0,),
    10: (111.0, 100.0, 90.0, 80.0, 71.0, 62.0, 52.0, 40.0, 24.0, 23.0),
}
MEAN_EXCITATION_EV_BY_Z = {
    1: (14.99,),
    10: (137.2, 165.2, 196.9, 235.2, 282.8, 352.6, 475.0, 696.8, 1409.2, 1498.4),
}


@dataclass(frozen=True)
class GridConfig:
    p_min: float = 0.1
    p_max: float = 100.0
    N_p: int = 1024
    theta_min: float = 0.0
    theta_max: float = np.pi
    N_xi: int = 256


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
    p_face_spacing: np.ndarray
    xi_face_spacing: np.ndarray
    xi_cell_widths: np.ndarray
    radial_volume: np.ndarray
    V: np.ndarray


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


def ion_species_from_charge_state_densities(Z: int, densities) -> IonSpecies:
    """Build D or Ne species from densities indexed by charge state."""
    if Z not in HESSLOW_ABAR_BY_Z:
        raise ValueError("tabulated species supports Z=1 (D) and Z=10 (Ne)")
    n = np.asarray(densities, dtype=float)
    if n.shape != (Z + 1,):
        raise ValueError(f"densities for Z={Z} must have shape ({Z + 1},)")
    if np.any(n < 0.0):
        raise ValueError("charge-state densities must be nonnegative")
    I_eV = np.zeros(Z + 1)
    a_bar = np.zeros(Z + 1)
    I_eV[:Z] = MEAN_EXCITATION_EV_BY_Z[Z]
    a_bar[:Z] = HESSLOW_ABAR_BY_Z[Z]
    return IonSpecies(Z=Z, n=n, I_eV=I_eV, a_bar=a_bar)


def build_grid(cfg: GridConfig) -> Grid:
    if cfg.p_min <= 0 or cfg.p_max <= cfg.p_min:
        raise ValueError("logarithmic momentum grid requires 0 < p_min < p_max")
    p_face = np.geomspace(cfg.p_min, cfg.p_max, cfg.N_p + 1)
    if cfg.theta_min < 0 or cfg.theta_max <= cfg.theta_min or cfg.theta_max > np.pi:
        raise ValueError("theta grid must satisfy 0 <= theta_min < theta_max <= pi")
    theta_face = np.linspace(cfg.theta_min, cfg.theta_max, cfg.N_xi + 1)
    xi_face = np.cos(theta_face[::-1])
    # Geometric centers align with logarithmic momentum cells.
    p_center = np.sqrt(p_face[:-1] * p_face[1:])
    theta_center = 0.5 * (theta_face[:-1] + theta_face[1:])
    xi_center = np.cos(theta_center[::-1])
    p_face_spacing = np.empty(cfg.N_p + 1)
    p_face_spacing[0] = p_center[0] - p_face[0]
    p_face_spacing[-1] = p_face[-1] - p_center[-1]
    p_face_spacing[1:-1] = np.diff(p_center)
    xi_cell_widths = np.diff(xi_face)
    xi_face_spacing = np.empty(cfg.N_xi + 1)
    xi_face_spacing[0] = xi_center[0] - xi_face[0]
    xi_face_spacing[-1] = xi_face[-1] - xi_center[-1]
    xi_face_spacing[1:-1] = np.diff(xi_center)
    radial_volume = 2 * np.pi / 3 * (p_face[1:]**3 - p_face[:-1]**3)
    return Grid(p_face, p_center, xi_face, xi_center, p_face_spacing,
                xi_face_spacing, xi_cell_widths, radial_volume,
                radial_volume[:, None] * xi_cell_widths[None, :])


def derive_plasma(cfg: PlasmaConfig) -> DerivedPlasma:
    n_e = sum(np.sum(np.arange(ion.Z + 1) * ion.n) for ion in cfg.ions)
    charge_square = sum(np.sum(np.arange(ion.Z + 1)**2 * ion.n) for ion in cfg.ions)
    Z_eff = charge_square / n_e
    Theta = cfg.T_e_eV / MEC2_EV
    lnLambda0 = 14.9 - 0.5 * np.log(n_e / 1e20) + np.log(cfg.T_e_eV / 1e3)
    tau_c = (4 * np.pi * constants.epsilon_0**2 * constants.m_e**2 * constants.c**3
             / (constants.e**4 * n_e * lnLambda0))
    tau_syn = (np.inf if cfg.B == 0 else
               6 * np.pi * constants.epsilon_0 * constants.m_e**3 * constants.c**3
               / (constants.e**4 * cfg.B**2))
    alpha = 0.0 if cfg.B == 0 else tau_c / tau_syn
    E_bar = constants.e * cfg.E_parallel * tau_c / (constants.m_e * constants.c)
    return DerivedPlasma(n_e, Z_eff, Theta, lnLambda0, tau_c, tau_syn, alpha, E_bar)


def collision_coefficients(p, cfg: PlasmaConfig, plasma: DerivedPlasma):
    gamma = np.sqrt(1 + p**2)
    x = p / (gamma * np.sqrt(2 * plasma.Theta))
    Phi = special.erf(x)
    Psi = (Phi - 2 * x * np.exp(-x**2) / np.sqrt(np.pi)) / (2 * x**2)
    k = 5
    lnLambda_ee = plasma.lnLambda0 + np.log1p(((gamma - 1) / plasma.Theta)**(k / 2)) / k
    lnLambda_ei = plasma.lnLambda0 + np.log1p((2 * p / np.sqrt(2 * plasma.Theta))**k) / k
    h = np.zeros_like(p)
    g = np.zeros_like(p)
    for ion in cfg.ions:
        for z in range(ion.Z + 1):
            N = ion.Z - z
            if ion.n[z] == 0 or N == 0:
                continue
            I_bar = ion.I_eV[z] / MEC2_EV
            q = p * np.sqrt(gamma - 1) / I_bar
            y = (ion.a_bar[z] * p)**1.5
            h += ion.n[z] / plasma.n_e * N * (np.log1p(q**k) / k - p**2 / gamma**2)
            g += ion.n[z] / plasma.n_e * ((2 / 3) * (ion.Z**2 - z**2) * np.log1p(y)
                                           - (2 / 3) * N**2 * y / (1 + y))
    C_F = (lnLambda_ee * Psi / plasma.Theta + gamma**2 * h / p**2) / plasma.lnLambda0
    C_A = gamma * Psi / p
    nu_D = gamma / (p**3 * plasma.lnLambda0) * (
        plasma.Z_eff * lnLambda_ei + lnLambda_ee * (Phi - Psi + plasma.Theta * p**2 / gamma**2) + g)
    return C_F, C_A, nu_D


def build_collision_data(grid: Grid, cfg: PlasmaConfig, plasma: DerivedPlasma) -> CollisionData:
    C_F_face = np.zeros_like(grid.p_face)
    # This reference solver uses drift-only radial transport: C_A = 0.
    C_F_face, _, _ = collision_coefficients(grid.p_face, cfg, plasma)
    C_A_face = np.zeros_like(grid.p_face)
    _, _, nu_D_center = collision_coefficients(grid.p_center, cfg, plasma)
    return CollisionData(C_F_face, C_A_face, nu_D_center)


def find_up_zero_momentum(cfg: GridConfig, plasma_cfg: PlasmaConfig,
                          plasma: DerivedPlasma) -> float:
    """Find suprathermal U_p(p, xi=-1)=0 root, excluding thermal branch."""
    p_scan = np.geomspace(max(cfg.p_max * 1.0e-12, 1.0e-12), cfg.p_max, 1024)
    C_F, _, _ = collision_coefficients(p_scan, plasma_cfg, plasma)
    up = plasma.E_bar - C_F  # xi=-1; synchrotron term vanishes.
    changes = np.flatnonzero(up[:-1] * up[1:] <= 0.0)
    if changes.size == 0:
        raise ValueError("U_p(p, xi=-1) has no root in (0, p_max]")

    def up_minus_one(p):
        C_F_one, _, _ = collision_coefficients(np.asarray([p]), plasma_cfg, plasma)
        return float(plasma.E_bar - C_F_one[0])

    roots = [brentq(up_minus_one, float(p_scan[k]), float(p_scan[k + 1]))
             for k in changes]
    if not roots:
        raise ValueError("no U_p root found; increase p_max or check field")
    # Thermal drag can create a lower root. The largest root is the
    # suprathermal branch selected by the asymptotic runaway criterion.
    return float(max(roots))


def momentum_from_kinetic_energy(kinetic_energy: float) -> float:
    """Convert normalized kinetic energy gamma-1 to dimensionless momentum."""
    if kinetic_energy <= 0.0:
        raise ValueError("kinetic energy must be positive")
    return float(np.sqrt((1.0 + kinetic_energy)**2 - 1.0))


def chang_cooper_delta(w):
    w = np.asarray(w, dtype=float)
    result = np.empty_like(w)
    small = np.abs(w) < 1.0e-6
    result[small] = 0.5 - w[small] / 12.0 + w[small]**3 / 720.0
    result[~small] = 1.0 / w[~small] - 1.0 / np.expm1(w[~small])
    return result.item() if result.ndim == 0 else result


def chang_cooper_coefficients(A, D, dq):
    """Return left coefficient and face-state coefficient for one FV face."""
    A, D, dq = np.broadcast_arrays(np.asarray(A, dtype=float),
                                   np.asarray(D, dtype=float),
                                   np.asarray(dq, dtype=float))
    left = np.empty_like(A)
    right = np.empty_like(A)
    diffusive = D > 0.0
    w = np.zeros_like(A)
    w[diffusive] = A[diffusive] * dq[diffusive] / D[diffusive]
    delta = np.zeros_like(A)
    delta[diffusive] = chang_cooper_delta(w[diffusive])
    left[diffusive] = A[diffusive] * (1.0 - delta[diffusive]) + D[diffusive] / dq[diffusive]
    right[diffusive] = A[diffusive] * delta[diffusive] - D[diffusive] / dq[diffusive]
    left[~diffusive] = np.maximum(A[~diffusive], 0.0)
    right[~diffusive] = np.minimum(A[~diffusive], 0.0)
    return left, right


def radial_boundary_rates(grid: Grid, plasma: DerivedPlasma, coll: CollisionData):
    """Return absorbing p_min failure and p_max escape rates for C_A=0."""
    xi = grid.xi_center
    gamma_min = np.sqrt(1.0 + grid.p_face[0]**2)
    gamma_max = np.sqrt(1.0 + grid.p_face[-1]**2)
    A_min = (-plasma.E_bar * xi - coll.C_F_face[0]
             - plasma.alpha * gamma_min * grid.p_face[0] * (1.0 - xi**2))
    A_max = (-plasma.E_bar * xi - coll.C_F_face[-1]
             - plasma.alpha * gamma_max * grid.p_face[-1] * (1.0 - xi**2))
    K_min = 2.0 * np.pi * grid.p_face[0]**2 * grid.xi_cell_widths / grid.V[0, :]
    K_max = 2.0 * np.pi * grid.p_face[-1]**2 * grid.xi_cell_widths / grid.V[-1, :]
    failure = np.zeros(grid.V.size)
    escape = np.zeros(grid.V.size)
    failure[:len(xi)] = K_min * np.maximum(-A_min, 0.0)
    escape[-len(xi):] = K_max * np.maximum(A_max, 0.0)
    return failure, escape


def assemble_fp_operator(grid: Grid, plasma: DerivedPlasma, coll: CollisionData):
    Np, Nxi = len(grid.p_center), len(grid.xi_center)
    M = Np * Nxi
    row = np.arange(M)
    i = np.repeat(np.arange(Np), Nxi)
    j = np.tile(np.arange(Nxi), Np)
    p = grid.p_center[i]
    gamma = np.sqrt(1.0 + p**2)
    xi = grid.xi_center[j]
    volume = grid.V.ravel()
    xi_width = grid.xi_cell_widths[j]
    diag = np.zeros(M)
    p_lower = np.zeros(M)
    p_upper = np.zeros(M)
    xi_lower = np.zeros(M)
    xi_upper = np.zeros(M)

    lower = i > 0
    pf = grid.p_face[i[lower]]
    gammaf = np.sqrt(1.0 + pf**2)
    A = (-plasma.E_bar * xi[lower] - coll.C_F_face[i[lower]]
         - plasma.alpha * gammaf * pf * (1.0 - xi[lower]**2))
    D = coll.C_A_face[i[lower]]
    dq = grid.p_face_spacing[i[lower]]
    left_coeff, right_coeff = chang_cooper_coefficients(A, D, dq)
    K = 2.0 * np.pi * pf**2 * xi_width[lower] / volume[lower]
    p_lower[lower] = K * left_coeff
    diag[lower] += K * right_coeff

    upper = i < Np - 1
    pf = grid.p_face[i[upper] + 1]
    gammaf = np.sqrt(1.0 + pf**2)
    A = (-plasma.E_bar * xi[upper] - coll.C_F_face[i[upper] + 1]
         - plasma.alpha * gammaf * pf * (1.0 - xi[upper]**2))
    D = coll.C_A_face[i[upper] + 1]
    dq = grid.p_face_spacing[i[upper] + 1]
    left_coeff, right_coeff = chang_cooper_coefficients(A, D, dq)
    K = 2.0 * np.pi * pf**2 * xi_width[upper] / volume[upper]
    diag[upper] -= K * left_coeff
    p_upper[upper] = -K * right_coeff
    escape = ~upper
    failure_rate, escape_rate = radial_boundary_rates(grid, plasma, coll)
    diag[escape] -= escape_rate[escape]
    diag[~lower] -= failure_rate[~lower]

    lower = j > 0
    xif = grid.xi_face[j[lower]]
    A = (1.0 - xif**2) * (-plasma.E_bar / p[lower] + plasma.alpha * xif / gamma[lower])
    D = 0.5 * coll.nu_D_center[i[lower]] * (1.0 - xif**2)
    dq = grid.xi_face_spacing[j[lower]]
    left_coeff, right_coeff = chang_cooper_coefficients(A, D, dq)
    K = grid.radial_volume[i[lower]] / volume[lower]
    xi_lower[lower] = K * left_coeff
    diag[lower] += K * right_coeff

    upper = j < Nxi - 1
    xif = grid.xi_face[j[upper] + 1]
    A = (1.0 - xif**2) * (-plasma.E_bar / p[upper] + plasma.alpha * xif / gamma[upper])
    D = 0.5 * coll.nu_D_center[i[upper]] * (1.0 - xif**2)
    dq = grid.xi_face_spacing[j[upper] + 1]
    left_coeff, right_coeff = chang_cooper_coefficients(A, D, dq)
    K = grid.radial_volume[i[upper]] / volume[upper]
    diag[upper] -= K * left_coeff
    xi_upper[upper] = -K * right_coeff

    data = np.concatenate((p_lower, xi_lower, diag, xi_upper, p_upper))
    cols = np.concatenate((np.where(i > 0, row - Nxi, row),
                           np.where(j > 0, row - 1, row), row,
                           np.where(j < Nxi - 1, row + 1, row),
                           np.where(i < Np - 1, row + Nxi, row)))
    rows = np.tile(row, 5)
    return sparse.coo_matrix((data, (rows, cols)), shape=(M, M)).tocsr()


def ch_knockon_dsigma_bar_dp(p1, p):
    gamma1 = np.sqrt(1 + p1**2)
    gamma = np.sqrt(1 + p**2)
    beta = p / gamma
    eps1 = gamma1 - 1
    eps = gamma - 1
    x = eps1**2 / (eps * (gamma1 - gamma))
    return 2 * np.pi * beta * gamma1**2 / (eps1**3 * (gamma1 + 1)) * (
        x**2 - 3 * x + (eps1 / gamma1)**2 * (1 + x))


def assemble_ch_operator(grid: Grid, cfg: PlasmaConfig, plasma: DerivedPlasma, p_m=0.1):
    Np, Nxi = len(grid.p_center), len(grid.xi_center)
    P, XI = np.meshgrid(grid.p_center, grid.xi_center, indexing='ij')
    gamma = np.sqrt(1 + P**2)
    eps = gamma - 1
    eps_m = np.sqrt(1 + p_m**2) - 1
    den = 1 + XI**2 - gamma * (1 - XI**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = -2 * P * XI / den
    mask = ((eps >= eps_m) & (XI < 0) & (den > 0) & np.isfinite(p1)
            & (p1 >= grid.p_face[0]) & (p1 < grid.p_face[-1]))
    i, j = np.nonzero(mask)
    p1v = p1[i, j]
    eps1 = np.sqrt(1 + p1v**2) - 1
    keep = eps[i, j] <= 0.5 * eps1
    i, j, p1v = i[keep], j[keep], p1v[keep]
    p = grid.p_center[i]
    xi = grid.xi_center[j]
    k = np.searchsorted(grid.p_face, p1v, side='right') - 1
    ds = ch_knockon_dsigma_bar_dp(p1v, p)
    n_t = sum(ion.Z * np.sum(ion.n) for ion in cfg.ions)
    pref = plasma.tau_c * n_t * constants.c * R_E**2
    coeff = pref * ds * p1v**4 / (p**2 * np.abs(xi))
    rows = np.repeat(i * Nxi + j, Nxi)
    cols = (k[:, None] * Nxi + np.arange(Nxi)).ravel()
    vals = np.repeat(coeff * grid.xi_cell_widths[j], Nxi)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(Np * Nxi, Np * Nxi)).tocsr()


def physical_adjoint_operator(L, grid: Grid):
    """Return V^-1 L.T V for cell-average states."""
    volumes = grid.V.ravel()
    return sparse.diags(1.0 / volumes) @ L.T @ sparse.diags(volumes)
