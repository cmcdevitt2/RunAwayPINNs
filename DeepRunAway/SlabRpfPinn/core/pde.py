"""PDE coefficients, residuals, boundaries, and collocation sampling.

Model inputs stay normalized to ``[0, 1]^9``. The first two coordinates are
momentum and ``xi``; the remaining seven are ``E/Ec, Te_eV, nD_m3, nNe_m3,
zD, zNe, B_T``. Physical derivatives are converted through the selected
momentum mapping before residual evaluation.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import constants
from scipy.stats import qmc

from core.rpf_fv_cpu import (
    HESSLOW_ABAR_BY_Z, MEAN_EXCITATION_EV_BY_Z, MEC2_EV,
)
from core.training_config import PinnDomain


def _table(values, fully_stripped):
    # Keep constants on host until distributed JAX initialization completes.
    return np.asarray(list(values) + [fully_stripped], dtype=np.float64)


D_I_EV = _table(MEAN_EXCITATION_EV_BY_Z[1], 1.0)
D_ABAR = _table(HESSLOW_ABAR_BY_Z[1], 0.0)
NE_I_EV = _table(MEAN_EXCITATION_EV_BY_Z[10], 1.0)
NE_ABAR = _table(HESSLOW_ABAR_BY_Z[10], 0.0)


def _transform_angular_sampling(points: np.ndarray, angular_sampling: str):
    """Map uniform-theta Sobol samples to normalized ``xi`` coordinates in [0, 1]."""
    if angular_sampling == "xi":
        return points
    if angular_sampling == "theta":
        points = points.copy()
        points[:, 1] = 0.5 * (1.0 + np.cos(np.pi * points[:, 1]))
        return points
    raise ValueError("angular_sampling must be 'xi' or 'theta'")


def normalize_momentum(p, p_min: float, p_max: float,
                       momentum_sampling: str = "log"):
    """Map positive physical momentum to normalized linear or logarithmic space."""
    if momentum_sampling == "log":
        return (np.log(p) - np.log(p_min)) / np.log(p_max / p_min)
    if momentum_sampling == "linear":
        return (np.asarray(p) - p_min) / (p_max - p_min)
    raise ValueError("momentum_sampling must be 'log' or 'linear'")


def sobol_9d(n_points: int, seed: int = 0, *, angular_sampling: str = "xi",
             momentum_sampling: str = "log") -> np.ndarray:
    """Generate reproducible normalized nine-coordinate Sobol samples."""
    if n_points <= 0:
        return np.empty((0, 9), dtype=np.float64)
    if momentum_sampling not in ("log", "linear"):
        raise ValueError("momentum_sampling must be 'log' or 'linear'")
    points = np.asarray(
        qmc.Sobol(d=9, scramble=True, seed=seed).random(n_points),
        dtype=np.float64)
    return _transform_angular_sampling(points, angular_sampling)


def _map_unit(u, lo, hi, scale):
    return (jnp.exp(jnp.log(lo) + u * (jnp.log(hi) - jnp.log(lo)))
            if scale == "log" else lo + u * (hi - lo))


def split_inputs(z, domain: PinnDomain):
    """Split normalized inputs into momentum, pitch cosine, plasma values, and B."""
    p = (jnp.exp(jnp.log(domain.p_min)
         + z[..., 0] * jnp.log(domain.p_max / domain.p_min))
         if domain.momentum_sampling == "log"
         else domain.p_min + z[..., 0] * (domain.p_max - domain.p_min))
    xi = -1.0 + 2.0 * z[..., 1]
    ebar = _map_unit(z[..., 2], domain.ebar_min, domain.ebar_max, "log")
    te = _map_unit(z[..., 3], domain.te_min_eV, domain.te_max_eV, "log")
    nD = _map_unit(z[..., 4], domain.nD_min_m3, domain.nD_max_m3, "log")
    nNe = _map_unit(z[..., 5], domain.nNe_min_m3, domain.nNe_max_m3, "log")
    zD = jnp.exp(jnp.log(domain.zD_min)
                 + z[..., 6] * jnp.log(domain.zD_max / domain.zD_min))
    zNe = jnp.exp(jnp.log(domain.zNe_min)
                  + z[..., 7] * jnp.log(domain.zNe_max / domain.zNe_min))
    B_T = domain.B_T_min + z[..., 8] * (domain.B_T_max - domain.B_T_min)
    return p, xi, ebar, te, nD, nNe, zD, zNe, B_T


def charge_weights(zavg, Z):
    """Return piecewise-linear weights for neighboring integer charge states."""
    q = jnp.arange(Z + 1, dtype=jnp.float64)
    return jnp.maximum(0.0, 1.0 - jnp.abs(jnp.asarray(zavg)[..., None] - q))


def collision_coefficients(p, te_eV, nD, nNe, zD, zNe, B_T):
    """Compute normalized friction, angular diffusion, and synchrotron scale."""
    qD = jnp.arange(2, dtype=jnp.float64)
    qNe = jnp.arange(11, dtype=jnp.float64)
    wD = charge_weights(zD, 1)
    wNe = charge_weights(zNe, 10)
    nD = jnp.asarray(nD)
    nNe = jnp.asarray(nNe)
    # State order must match the CPU reference: D charge states, then Ne
    # charge states. Charge-state weights preserve each element's total ion
    # density while allowing continuous mean-charge parameters.
    n_state = jnp.concatenate([nD[..., None] * wD, nNe[..., None] * wNe], axis=-1)
    q_state = jnp.concatenate([qD, qNe])
    Z_nuc = jnp.concatenate([jnp.ones(2), jnp.full(11, 10.0)])
    I_ev = jnp.concatenate([D_I_EV, NE_I_EV])
    abar = jnp.concatenate([D_ABAR, NE_ABAR])
    n_bound = Z_nuc - q_state
    ne = jnp.sum(n_state * q_state, axis=-1)
    zeff = jnp.sum(n_state * q_state * q_state, axis=-1) / ne
    ln0 = 14.9 - 0.5 * jnp.log(ne / 1.0e20) + jnp.log(te_eV / 1.0e3)
    theta = te_eV / MEC2_EV
    delta = jnp.sqrt(2.0 * theta)
    gamma = jnp.sqrt(1.0 + p * p)
    gamma_minus_one = p * p / (gamma + 1.0)
    x = p / (delta * gamma)
    phi = jax.scipy.special.erf(x)
    psi_direct = ((phi - 2.0 * x * jnp.exp(-x * x) / jnp.sqrt(jnp.pi))
                  / (2.0 * jnp.maximum(x * x, 1.0e-30)))
    psi_series = ((2.0 / 3.0) * x - (2.0 / 5.0) * x**3
                  + (1.0 / 7.0) * x**5 - (1.0 / 27.0) * x**7) / jnp.sqrt(jnp.pi)
    psi = jnp.where(jnp.abs(x) < 1.0e-3, psi_series, psi_direct)
    q_ee = 2.0 * gamma_minus_one / (delta * delta)
    q_ei = 2.0 * p / delta
    ln_ee = ln0 + jnp.log1p(q_ee ** 2.5) / 5.0
    ln_ei = ln0 + jnp.log1p(q_ei ** 5.0) / 5.0
    r_ee = ln_ee / ln0
    r_ei = ln_ei / ln0
    beta2 = p * p / (gamma * gamma)
    Ibar = I_ev / MEC2_EV
    p_species = p[..., None]
    harg = p_species * jnp.sqrt(jnp.maximum(gamma - 1.0, 0.0))[..., None] / Ibar
    h = jnp.sum((n_state / ne[..., None]) * n_bound * (
        0.2 * jnp.log1p(harg ** 5) - beta2[..., None]), axis=-1)
    y = jnp.power(jnp.maximum(p_species * abar, 0.0), 1.5)
    g = jnp.sum((n_state / ne[..., None]) * (2.0 / 3.0) * (
        (Z_nuc * Z_nuc - q_state * q_state) * jnp.log1p(y)
        - n_bound * n_bound * y / (1.0 + y)), axis=-1)
    ee_deflection = phi - psi + delta * delta * p * p / (2.0 * gamma * gamma)
    cf = r_ee * 2.0 * psi / (delta * delta) + (gamma * gamma / (p * p)) * (h / ln0)
    nud = gamma / (p ** 3) * (zeff * r_ei + r_ee * ee_deflection + g / ln0)
    tau_c = 4.0 * jnp.pi * constants.epsilon_0**2 * constants.m_e**2 * constants.c**3 / (
        constants.e**4 * ne * ln0)
    tau_syn = (6.0 * jnp.pi * constants.epsilon_0 * constants.m_e ** 3
               * constants.c ** 3 /
               (constants.e ** 4 * B_T ** 2))
    alpha = tau_c / tau_syn
    return cf, nud, alpha


def sobol_9d_nontrivial(n_points: int, domain: PinnDomain, seed: int = 0, *,
                         angular_sampling: str = "xi",
                         momentum_sampling: str = "log",
                         batch_size: int = 32768) -> np.ndarray:
    """Sample nine-coordinate cases with positive upper-boundary drift."""
    if n_points <= 0:
        return np.empty((0, 9), dtype=np.float64)
    engine = qmc.Sobol(d=10, scramble=True, seed=seed)
    accepted, count = [], 0
    for _ in range(128):
        z = _transform_angular_sampling(
            np.asarray(engine.random(batch_size), dtype=np.float64),
            angular_sampling)
        z_boundary = z.copy()
        z_boundary[:, 0] = 1.0
        z_boundary[:, 1] = 0.0
        p, xi, ebar, te, nD, nNe, zD, zNe, B_T = split_inputs(
            jnp.asarray(z_boundary), domain)
        cf, _, _ = collision_coefficients(p, te, nD, nNe, zD, zNe, B_T)
        keep = np.asarray(jax.device_get(ebar - cf)) > 0.0
        if np.any(keep):
            accepted.append(z[keep, :9])
            count += int(np.count_nonzero(keep))
            if count >= n_points:
                break
    if count < n_points:
        raise RuntimeError(
            f"only found {count} nontrivial parameter points; "
            "increase batch_size or restrict the parameter domain")
    return np.concatenate(accepted, axis=0)[:n_points]


def sobol_7d(n_points: int, seed: int = 0) -> np.ndarray:
    """Generate reproducible normalized seven-parameter Sobol samples."""
    if n_points <= 0:
        return np.empty((0, 7), dtype=np.float64)
    return np.asarray(qmc.Sobol(d=7, scramble=True, seed=seed).random(n_points),
                      dtype=np.float64)


def sobol_pmax_boundary_nontrivial(n_points: int, domain: PinnDomain,
                                   seed: int = 0, *,
                                   angular_sampling: str = "xi") -> np.ndarray:
    """Return normalized samples fixed to the upper-momentum boundary."""
    z = sobol_9d_nontrivial(
        n_points, domain, seed, angular_sampling=angular_sampling,
        momentum_sampling=domain.momentum_sampling)
    z[:, 0] = 1.0
    return z


def sobol_plow_boundary_nontrivial(n_points: int, domain: PinnDomain,
                                   seed: int = 0, *,
                                   angular_sampling: str = "xi") -> np.ndarray:
    """Return normalized samples fixed to the lower-momentum boundary."""
    z = sobol_9d_nontrivial(
        n_points, domain, seed, angular_sampling=angular_sampling,
        momentum_sampling=domain.momentum_sampling)
    z[:, 0] = 0.0
    return z


def analytic_threshold_collocation(n_points: int, domain: PinnDomain, *,
                                   seed: int = 0, band_width: float = 0.02,
                                   batch_size: int = 32768) -> np.ndarray:
    """Sample a band around the analytic physical ``U_p=0`` threshold curve."""
    if n_points <= 0:
        return np.empty((0, 9), dtype=np.float64)
    if not 0.0 < band_width < 1.0:
        raise ValueError("band_width must be in (0, 1)")
    engine = qmc.Sobol(d=10, scramble=True, seed=seed)
    accepted, count = [], 0
    for _ in range(32):
        u = np.asarray(engine.random(batch_size), dtype=np.float64)
        z = np.empty((batch_size, 9), dtype=np.float64)
        z[:, 0] = u[:, 0]
        z[:, 2:] = u[:, 1:8]
        z[:, 1] = 0.5
        p, _, ebar, te, nD, nNe, zD, zNe, B_T = split_inputs(jnp.asarray(z), domain)
        cf, _, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, B_T)
        z_boundary = z.copy()
        z_boundary[:, 0] = 1.0
        z_boundary[:, 1] = 0.0
        pp, _, ee, tt, dd, nn, zd, zn, BB = split_inputs(jnp.asarray(z_boundary), domain)
        cf_max, _, _ = collision_coefficients(pp, tt, dd, nn, zd, zn, BB)
        gamma = jnp.sqrt(1.0 + p * p)
        A = alpha * gamma * p
        discriminant = ebar * ebar + 4.0 * A * (cf + A)
        xi_root = jnp.where(
            A > 1.0e-14,
            (ebar - jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (2.0 * A),
            -cf / ebar)
        xi_root = np.asarray(jax.device_get(xi_root))
        nontrivial = np.asarray(jax.device_get(ebar - cf_max)) > 0.0
        physical = (np.isfinite(xi_root) & nontrivial
                    & (xi_root >= -1.0) & (xi_root <= 1.0))
        delta = band_width * (0.25 + 0.75 * u[:, 8])
        delta *= np.where(u[:, 9] < 0.5, -1.0, 1.0)
        xi_sample = xi_root + delta
        physical &= (xi_sample >= -1.0) & (xi_sample <= 1.0)
        if np.any(physical):
            z[physical, 1] = 0.5 * (xi_sample[physical] + 1.0)
            accepted.append(z[physical])
            count += int(np.count_nonzero(physical))
            if count >= n_points:
                break
    if count < n_points:
        raise RuntimeError(
            f"only found {count} physical U_p=0-band points; "
            "increase batch_size or check the parameter domain")
    return np.concatenate(accepted, axis=0)[:n_points]


def drift_up(z, domain: PinnDomain):
    """Return normalized radial drift ``U_p``; positive means outward motion."""
    p, xi, ebar, te, nD, nNe, zD, zNe, B_T = split_inputs(z, domain)
    cf, _, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    return -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)


def rescale_coefficients(coefficients, cf, ebar, coeff_norm="cf_ebar"):
    """Apply the named heuristic scale-balancing scheme for PDE coefficients.

    This is distinct from the change-of-variables Jacobian already folded
    into `a_p` inside `pde_coefficients`, which is a correctness factor and
    must not be made configurable.
    """
    if coeff_norm == "cf_ebar":
        scale = jnp.abs(cf)[..., None] * jnp.sqrt(ebar)[..., None]
        return coefficients / scale
    if coeff_norm == "coeff_l2":
        norm = jnp.sqrt(jnp.sum(coefficients ** 2, axis=-1, keepdims=True))
        return coefficients / jnp.maximum(norm, 1.0e-30)
    if coeff_norm == "none":
        return coefficients
    raise ValueError("coeff_norm must be 'cf_ebar', 'coeff_l2', or 'none'")


def pde_coefficients(z, domain: PinnDomain, coeff_norm: str = "cf_ebar"):
    """Return normalized coefficients for p, xi, and second-xi derivatives."""
    p, xi, ebar, te, nD, nNe, zD, zNe, B_T = split_inputs(z, domain)
    cf, nud, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    up = -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)
    # Chain rule converts derivatives in normalized momentum to derivatives
    # in physical momentum. This factor is required, not a tunable loss scale.
    dp_dpnorm = (p * jnp.log(domain.p_max / domain.p_min)
                 if domain.momentum_sampling == "log"
                 else jnp.asarray(domain.p_max - domain.p_min, dtype=jnp.float64))
    a_p = -up / dp_dpnorm
    a_xi = 0.5 * ((1.0 - xi * xi) * (ebar / p - alpha * xi / gamma) + nud * xi)
    a_xixi = -0.25 * 0.5 * nud * (1.0 - xi * xi)
    coefficients = jnp.stack([a_p, a_xi, a_xixi], axis=-1)
    return rescale_coefficients(coefficients, cf, ebar, coeff_norm)


def residual_single(params, z, coefficients, probability_fn, residual_floor=0.1):
    """Evaluate stabilized relative PDE residual using phase-space autodiff."""
    phase = z[:2]
    def probability_phase(q):
        zz = z.at[0].set(q[0]).at[1].set(q[1])
        return probability_fn(params, zz)
    grad_phase = jax.grad(probability_phase)(phase)
    _, second_xi = jax.jvp(
        lambda q: jax.grad(probability_phase)(q)[1],
        (phase,), (jnp.array([0.0, 1.0]),))
    P = probability_fn(params, z)
    denom = jnp.maximum(P + residual_floor, 1.0e-12)
    return (coefficients[0] * grad_phase[0]
            + coefficients[1] * grad_phase[1]
            + coefficients[2] * second_xi) / denom


@lru_cache(maxsize=16)
def _make_pde_functions_cached(domain, probability_fn, coeff_norm, residual_floor):
    """Build and cache JIT/VMap functions for one immutable PDE setup."""
    coefficients = jax.jit(jax.vmap(
        lambda z: pde_coefficients(z, domain, coeff_norm)))
    residual = jax.jit(jax.vmap(
        lambda params, z, c: residual_single(
            params, z, c, probability_fn, residual_floor),
        in_axes=(None, 0, 0)))
    return coefficients, residual


def make_pde_functions(domain: PinnDomain, *, probability_fn=None,
                       coeff_norm: str = "cf_ebar", residual_floor: float = 0.1):
    """Build or reuse JIT/VMap coefficient and residual functions."""
    if probability_fn is None:
        from core.model import probability as probability_fn
    return _make_pde_functions_cached(
        domain, probability_fn, coeff_norm, float(residual_floor))


def evaluate_pde_residuals(params, z, domain: PinnDomain, *, chunk_size=65536,
                           probability_fn=None, coeff_norm: str = "cf_ebar",
                           residual_floor: float = 0.1):
    """Evaluate residuals in bounded host/device chunks, preserving input order."""
    z = np.asarray(z, dtype=np.float64)
    coeff_fn, residual_fn = make_pde_functions(
        domain, probability_fn=probability_fn, coeff_norm=coeff_norm,
        residual_floor=residual_floor)
    values = []
    for start in range(0, len(z), chunk_size):
        batch = jnp.asarray(z[start:start + chunk_size])
        values.append(np.asarray(jax.device_get(
            residual_fn(params, batch, coeff_fn(batch)))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def success_boundary_target(z, domain: PinnDomain):
    """Return binary success targets where upper-boundary drift points outward."""
    p = jnp.asarray(domain.p_max, dtype=jnp.float64)
    xi = -1.0 + 2.0 * z[..., 1]
    ebar = _map_unit(z[..., 2], domain.ebar_min, domain.ebar_max, "log")
    te = _map_unit(z[..., 3], domain.te_min_eV, domain.te_max_eV, "log")
    nD = _map_unit(z[..., 4], domain.nD_min_m3, domain.nD_max_m3, "log")
    nNe = _map_unit(z[..., 5], domain.nNe_min_m3, domain.nNe_max_m3, "log")
    zD = jnp.exp(jnp.log(domain.zD_min)
                 + z[..., 6] * jnp.log(domain.zD_max / domain.zD_min))
    zNe = jnp.exp(jnp.log(domain.zNe_min)
                  + z[..., 7] * jnp.log(domain.zNe_max / domain.zNe_min))
    B_T = domain.B_T_min + z[..., 8] * (domain.B_T_max - domain.B_T_min)
    cf, _, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    up = -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)
    return (up > 0.0).astype(jnp.float64)
