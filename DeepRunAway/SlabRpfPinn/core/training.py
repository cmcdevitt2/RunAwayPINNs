"""Shared data, physics, SOAP, SSBroyden, and active training loops."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import constants
from scipy.stats import qmc
from tqdm.auto import tqdm
from jax.flatten_util import ravel_pytree

from core.rpf_fv_cpu import HESSLOW_ABAR_BY_Z, MEAN_EXCITATION_EV_BY_Z, MEC2_EV


@dataclass(frozen=True)
class PinnDomain:
    p_min: float
    p_max: float
    B_T: float
    momentum_sampling: str = "log"
    ebar_min: float = 1.0
    ebar_max: float = 1000.0
    te_min_eV: float = 1.0
    te_max_eV: float = 100.0
    nD_min_m3: float = 1.0e20
    nD_max_m3: float = 1.0e22
    nNe_min_m3: float = 1.0e16
    nNe_max_m3: float = 1.0e22
    zD_min: float = 0.01
    zD_max: float = 1.0
    zNe_min: float = 0.01
    zNe_max: float = 10.0


@dataclass(frozen=True)
class PinnConfig:
    """All training, architecture, and loss-term settings for the PINN."""
    training_mode: str = "physics_informed"
    steps: int = 1000
    # Global points per optimizer step. Zero keeps full-batch training.
    batch_size: int = 0
    # DeepONet cases per optimizer step. Zero derives this from batch_size.
    case_batch_size: int = 0
    # Number of test points/cells evaluated when recording the test loss.
    # Zero evaluates the complete held-out set.
    test_batch_size: int = 262144
    # Host-to-device loss readback interval. One records every step.
    log_every: int = 1
    # Model snapshot interval. Zero disables periodic checkpoints.
    checkpoint_every: int = 0
    learning_rate: float = 3.0e-3
    learning_rate_schedule: str = "constant"
    learning_rate_final_fraction: float = 0.1
    learning_rate_decay_steps: int = 0
    seed: int = 2026
    width: int = 32
    depth: int = 3
    model_type: str = "mlp"
    latent_width: int = 64
    branch_width: int = 32
    branch_depth: int = 3
    trunk_width: int = 32
    trunk_depth: int = 3
    momentum_sampling: str = "log"
    angular_sampling: str = "xi"
    enable_data: bool = True
    enable_pde: bool = True
    enable_threshold_pde: bool = True
    enable_low_p_bc: bool = True
    enable_pmax_bc: bool = True
    data_weight: float = 1.0
    pde_weight: float = 1.0
    threshold_weight: float = 1.0
    low_p_weight: float = 1.0
    bc_weight: float = 1.0
    soap_b1: float = 0.95
    soap_b2: float = 0.95
    soap_shampoo_beta: float = -1.0
    soap_eps: float = 1.0e-8
    soap_weight_decay: float = 0.0
    soap_correct_bias: bool = True
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 10000
    soap_precondition_1d: bool = False
    # Number of local GPUs for data-parallel JAX training. Zero means all
    # visible local GPUs; one preserves the single-device path.
    n_devices: int = 1
    ssbroyden_rtol: float = 1.0e-12
    ssbroyden_atol: float = 1.0e-12
    ssbroyden_blocks: int = 5
    ssbroyden_block_iters: int = 100
    train_case_fraction: float = 0.8

    def __post_init__(self):
        if not any((self.enable_data, self.enable_pde,
                    self.enable_low_p_bc, self.enable_pmax_bc)):
            raise ValueError("at least one PINN loss term must be enabled")
        if self.enable_threshold_pde and not self.enable_pde:
            raise ValueError("enable_pde must be True when enable_threshold_pde is True")
        if self.steps < 0 or self.width <= 0 or self.depth < 0:
            raise ValueError("invalid training or architecture setting")
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if (self.case_batch_size < 0 or self.test_batch_size < 0
                or self.log_every <= 0 or self.checkpoint_every < 0):
            raise ValueError("invalid batch or logging setting")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError("learning_rate_schedule must be 'constant' or 'cosine'")
        if not 0.0 <= self.learning_rate_final_fraction <= 1.0:
            raise ValueError("learning_rate_final_fraction must be in [0, 1]")
        if self.learning_rate_decay_steps < 0:
            raise ValueError("learning_rate_decay_steps must be non-negative")
        if self.training_mode not in ("data", "physics_informed"):
            raise ValueError("training_mode must be 'data' or 'physics_informed'")
        if self.model_type not in ("mlp", "deeponet"):
            raise ValueError("model_type must be 'mlp' or 'deeponet'")
        if (self.latent_width <= 0 or self.branch_width <= 0
                or self.branch_depth < 0 or self.trunk_width <= 0
                or self.trunk_depth < 0):
            raise ValueError("invalid DeepONet architecture setting")
        if self.angular_sampling not in ("xi", "theta"):
            raise ValueError("angular_sampling must be 'xi' or 'theta'")
        if self.momentum_sampling not in ("log", "linear"):
            raise ValueError("momentum_sampling must be 'log' or 'linear'")
        if self.ssbroyden_blocks < 0 or self.ssbroyden_block_iters <= 0:
            raise ValueError("invalid SSBroyden setting")
        if self.soap_precondition_frequency <= 0 or self.soap_max_precond_dim <= 0:
            raise ValueError("invalid SOAP preconditioner setting")
        if self.n_devices < 0:
            raise ValueError("n_devices must be non-negative")
        if self.ssbroyden_rtol < 0.0 or self.ssbroyden_atol < 0.0:
            raise ValueError("SSBroyden tolerances must be non-negative")

        if not 0.0 < self.train_case_fraction < 1.0:
            raise ValueError("train_case_fraction must be between 0 and 1")
        if any(weight < 0.0 for weight in
               (self.data_weight, self.pde_weight, self.threshold_weight,
                self.low_p_weight, self.bc_weight)):
            raise ValueError("loss weights must be non-negative")


def make_pinn_config(run_config):
    """Build flattened training config from one run config and canonical mode."""
    values = dict(run_config["pinn_config"])
    split_config = run_config.get("data", {})
    if "seed" in split_config:
        values["seed"] = split_config["seed"]
    if "train_case_fraction" in split_config:
        values["train_case_fraction"] = split_config["train_case_fraction"]
    mode = run_config.get("mode", "data")
    if mode == "data":
        values["training_mode"] = "data"
    elif mode == "physics":
        values["training_mode"] = "physics_informed"
    else:
        raise ValueError("run config mode must be 'data' or 'physics'")
    return PinnConfig(**values)


def _table(values, fully_stripped):
    return jnp.asarray(list(values) + [fully_stripped], dtype=jnp.float64)


D_I_EV = _table(MEAN_EXCITATION_EV_BY_Z[1], 1.0)
D_ABAR = _table(HESSLOW_ABAR_BY_Z[1], 0.0)
NE_I_EV = _table(MEAN_EXCITATION_EV_BY_Z[10], 1.0)
NE_ABAR = _table(HESSLOW_ABAR_BY_Z[10], 0.0)


def _transform_angular_sampling(points: np.ndarray, angular_sampling: str) -> np.ndarray:
    if angular_sampling == "xi":
        return points
    if angular_sampling == "theta":
        points = points.copy()
        points[:, 1] = 0.5 * (1.0 + np.cos(np.pi * points[:, 1]))
        return points
    raise ValueError("angular_sampling must be 'xi' or 'theta'")


def normalize_momentum(p, p_min: float, p_max: float,
                       momentum_sampling: str = "log"):
    """Map physical momentum to normalized log- or linearly-spaced coordinate."""
    if momentum_sampling == "log":
        return (np.log(p) - np.log(p_min)) / np.log(p_max / p_min)
    if momentum_sampling == "linear":
        return (np.asarray(p) - p_min) / (p_max - p_min)
    raise ValueError("momentum_sampling must be 'log' or 'linear'")


def sobol_8d(n_points: int, seed: int = 0, *, angular_sampling: str = "xi",
             momentum_sampling: str = "log") -> np.ndarray:
    """Return Sobol points with selectable momentum and angular sampling."""
    if n_points <= 0:
        return np.empty((0, 8), dtype=np.float64)
    engine = qmc.Sobol(d=8, scramble=True, seed=seed)
    if momentum_sampling not in ("log", "linear"):
        raise ValueError("momentum_sampling must be 'log' or 'linear'")
    points = np.asarray(engine.random(n_points), dtype=np.float64)
    return _transform_angular_sampling(points, angular_sampling)


def sobol_8d_nontrivial(n_points: int, domain: PinnDomain, seed: int = 0, *,
                         angular_sampling: str = "xi",
                         momentum_sampling: str = "log",
                         batch_size: int = 32768) -> np.ndarray:
    """Sample full 8D Sobol points, rejecting trivial parameter cases."""
    if n_points <= 0:
        return np.empty((0, 8), dtype=np.float64)
    engine = qmc.Sobol(d=8, scramble=True, seed=seed)
    accepted = []
    n_accepted = 0
    for _ in range(128):
        z = np.asarray(engine.random(batch_size), dtype=np.float64)
        z = _transform_angular_sampling(z, angular_sampling)
        if momentum_sampling not in ("log", "linear"):
            raise ValueError("momentum_sampling must be 'log' or 'linear'")
        z_boundary = z.copy()
        z_boundary[:, 0] = 1.0
        z_boundary[:, 1] = 0.0  # xi=-1
        p, xi, ebar, te, nD, nNe, zD, zNe = split_inputs(
            jnp.asarray(z_boundary), domain)
        cf, _nud, _alpha = collision_coefficients(
            p, te, nD, nNe, zD, zNe, domain.B_T)
        keep = np.asarray(jax.device_get(ebar - cf)) > 0.0
        if np.any(keep):
            accepted.append(z[keep])
            n_accepted += int(np.count_nonzero(keep))
            if n_accepted >= n_points:
                break
    if n_accepted < n_points:
        raise RuntimeError(
            f"only found {n_accepted} nontrivial parameter points; "
            "increase batch_size or restrict the parameter domain"
        )
    return np.concatenate(accepted, axis=0)[:n_points]


def sobol_6d(n_points: int, seed: int = 0) -> np.ndarray:
    """Return scrambled Sobol points in normalized six-parameter domain."""
    if n_points <= 0:
        return np.empty((0, 6), dtype=np.float64)
    engine = qmc.Sobol(d=6, scramble=True, seed=seed)
    return np.asarray(engine.random(n_points), dtype=np.float64)


def sobol_pmax_boundary(n_points: int, seed: int = 0, *,
                        angular_sampling: str = "xi") -> np.ndarray:
    """Return Sobol points on p_norm=1 with uniform xi or theta sampling."""
    if n_points <= 0:
        return np.empty((0, 8), dtype=np.float64)
    engine = qmc.Sobol(d=7, scramble=True, seed=seed)
    sampled = np.asarray(engine.random(n_points), dtype=np.float64)
    if angular_sampling == "theta":
        sampled[:, 0] = 0.5 * (1.0 + np.cos(np.pi * sampled[:, 0]))
    elif angular_sampling != "xi":
        raise ValueError("angular_sampling must be 'xi' or 'theta'")
    return np.column_stack((np.ones(n_points), sampled))


def sobol_pmax_boundary_nontrivial(n_points: int, domain: PinnDomain, seed: int = 0, *,
                                   angular_sampling: str = "xi") -> np.ndarray:
    """Sample p_norm=1 boundary points only for nontrivial cases."""
    z = sobol_8d_nontrivial(
        n_points, domain, seed, angular_sampling=angular_sampling,
        momentum_sampling=domain.momentum_sampling)
    z[:, 0] = 1.0
    return z


def sobol_plow_boundary(n_points: int, seed: int = 0, *,
                        angular_sampling: str = "xi") -> np.ndarray:
    """Return Sobol points on p_norm=0 with uniform xi or theta sampling."""
    if n_points <= 0:
        return np.empty((0, 8), dtype=np.float64)
    engine = qmc.Sobol(d=7, scramble=True, seed=seed)
    sampled = np.asarray(engine.random(n_points), dtype=np.float64)
    if angular_sampling == "theta":
        sampled[:, 0] = 0.5 * (1.0 + np.cos(np.pi * sampled[:, 0]))
    elif angular_sampling != "xi":
        raise ValueError("angular_sampling must be 'xi' or 'theta'")
    return np.column_stack((np.zeros(n_points), sampled))


def sobol_plow_boundary_nontrivial(n_points: int, domain: PinnDomain, seed: int = 0, *,
                                   angular_sampling: str = "xi") -> np.ndarray:
    """Sample p_norm=0 boundary points only for nontrivial cases."""
    z = sobol_8d_nontrivial(
        n_points, domain, seed, angular_sampling=angular_sampling,
        momentum_sampling=domain.momentum_sampling)
    z[:, 0] = 0.0
    return z


def analytic_threshold_collocation(n_points: int, domain: PinnDomain, *,
                                   seed: int = 0, band_width: float = 0.02,
                                   batch_size: int = 32768) -> np.ndarray:
    """Sample a thin normalized band around the physical ``U_p=0`` curve.

    With ``A = alpha * gamma * p``, the exact drift-zero condition is
    ``A*xi**2 - Ebar*xi - (C_F + A) = 0``.  The lower quadratic branch is
    the physical root for the positive electric field convention used here.
    Sampling is outside autodiff; returned points remain ordinary PDE points.
    """
    if n_points <= 0:
        return np.empty((0, 8), dtype=np.float64)
    if band_width <= 0.0 or band_width >= 1.0:
        raise ValueError("band_width must be in (0, 1)")

    engine = qmc.Sobol(d=9, scramble=True, seed=seed)
    accepted = []
    n_accepted = 0
    for _ in range(32):
        u = np.asarray(engine.random(batch_size), dtype=np.float64)
        z = np.empty((batch_size, 8), dtype=np.float64)
        z[:, 0] = u[:, 0]
        z[:, 2:] = u[:, 1:7]
        z[:, 1] = 0.5  # xi is replaced by the analytic threshold branch below.

        p, _xi, ebar, te, nD, nNe, zD, zNe = split_inputs(jnp.asarray(z), domain)
        cf, _nud, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, domain.B_T)
        z_boundary = z.copy()
        z_boundary[:, 0] = 1.0
        z_boundary[:, 1] = 0.0
        _pmax, _xmax, ebar_max, te_max, nD_max, nNe_max, zD_max, zNe_max = split_inputs(
            jnp.asarray(z_boundary), domain)
        cf_max, _nud_max, _alpha_max = collision_coefficients(
            _pmax, te_max, nD_max, nNe_max, zD_max, zNe_max, domain.B_T)
        gamma = jnp.sqrt(1.0 + p * p)
        A = alpha * gamma * p
        discriminant = ebar * ebar + 4.0 * A * (cf + A)
        sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        xi_root = jnp.where(
            A > 1.0e-14,
            (ebar - sqrt_discriminant) / (2.0 * A),
            -cf / ebar,
        )
        xi_root = np.asarray(jax.device_get(xi_root))
        finite = np.isfinite(xi_root)
        nontrivial = np.asarray(jax.device_get(ebar_max - cf_max)) > 0.0
        physical = finite & nontrivial & (xi_root >= -1.0) & (xi_root <= 1.0)

        delta = band_width * (0.25 + 0.75 * u[:, 7])
        delta *= np.where(u[:, 8] < 0.5, -1.0, 1.0)
        xi_sample = xi_root + delta
        physical &= (xi_sample >= -1.0) & (xi_sample <= 1.0)
        if np.any(physical):
            z[physical, 1] = 0.5 * (xi_sample[physical] + 1.0)
            accepted.append(z[physical])
            n_accepted += int(np.count_nonzero(physical))
            if n_accepted >= n_points:
                break

    if n_accepted < n_points:
        raise RuntimeError(
            f"only found {n_accepted} physical U_p=0-band points; "
            "increase batch_size or check the parameter domain"
        )
    return np.concatenate(accepted, axis=0)[:n_points]


def _map_unit(u, lo, hi, scale):
    return lo + u * (hi - lo) if scale == "linear" else jnp.exp(jnp.log(lo) + u * (jnp.log(hi) - jnp.log(lo)))


def split_inputs(z, domain: PinnDomain):
    p = (jnp.exp(jnp.log(domain.p_min) + z[..., 0] *
                 (jnp.log(domain.p_max) - jnp.log(domain.p_min)))
         if domain.momentum_sampling == "log" else
         domain.p_min + z[..., 0] * (domain.p_max - domain.p_min))
    xi = -1.0 + 2.0 * z[..., 1]
    ebar = _map_unit(z[..., 2], domain.ebar_min, domain.ebar_max, "log")
    te = _map_unit(z[..., 3], domain.te_min_eV, domain.te_max_eV, "log")
    nD = _map_unit(z[..., 4], domain.nD_min_m3, domain.nD_max_m3, "log")
    nNe = _map_unit(z[..., 5], domain.nNe_min_m3, domain.nNe_max_m3, "log")
    zD = domain.zD_min + z[..., 6] * (domain.zD_max - domain.zD_min)
    zNe = domain.zNe_min + z[..., 7] * (domain.zNe_max - domain.zNe_min)
    return p, xi, ebar, te, nD, nNe, zD, zNe


def charge_weights(zavg, Z):
    q = jnp.arange(Z + 1, dtype=jnp.float64)
    return jnp.maximum(0.0, 1.0 - jnp.abs(jnp.asarray(zavg)[..., None] - q))


def collision_coefficients(p, te_eV, nD, nNe, zD, zNe, B_T):
    qD = jnp.arange(2, dtype=jnp.float64)
    qNe = jnp.arange(11, dtype=jnp.float64)
    wD = charge_weights(zD, 1)
    wNe = charge_weights(zNe, 10)
    nD = jnp.asarray(nD)
    nNe = jnp.asarray(nNe)
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
    psi = (phi - 2.0 * x * jnp.exp(-x * x) / jnp.sqrt(jnp.pi)) / (2.0 * x * x)
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
    h = jnp.sum((n_state / ne[..., None]) * n_bound *
                (0.2 * jnp.log1p(harg ** 5) - beta2[..., None]), axis=-1)
    y = jnp.power(jnp.maximum(p_species * abar, 0.0), 1.5)
    g = jnp.sum((n_state / ne[..., None]) * (2.0 / 3.0) * (
        (Z_nuc * Z_nuc - q_state * q_state) * jnp.log1p(y)
        - n_bound * n_bound * y / (1.0 + y)), axis=-1)
    ee_deflection = phi - psi + delta * delta * p * p / (2.0 * gamma * gamma)
    cf = r_ee * 2.0 * psi / (delta * delta) + (gamma * gamma / (p * p)) * (h / ln0)
    nud = gamma / (p ** 3) * (zeff * r_ei + r_ee * ee_deflection + g / ln0)
    tau_c = 4.0 * jnp.pi * constants.epsilon_0**2 * constants.m_e**2 * constants.c**3 / (
        constants.e**4 * ne * ln0)
    alpha = (jnp.asarray(0.0, dtype=jnp.float64) if B_T == 0.0 else
             tau_c / (6.0 * jnp.pi * constants.epsilon_0 * constants.m_e**3 * constants.c**3 /
                      (constants.e**4 * B_T**2)))
    return cf, nud, alpha


def mlp_raw(params, z):
    if isinstance(params, dict):
        branch = _apply_mlp(params["branch"], z[..., 2:])
        trunk = _apply_mlp(params["trunk"], z[..., :2])
        return jnp.sum(branch * trunk, axis=-1) + params["bias"]
    return _apply_mlp(params, z)[..., 0]


def _apply_mlp(params, z):
    h = z
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def probability_transform(raw):
    """Map unconstrained network output to probability."""
    return jax.nn.sigmoid(raw)


def probability(params, z):
    """Return pointwise PINN or DeepONet probability."""
    return probability_transform(mlp_raw(params, z))


predict = jax.jit(probability)


def init_mlp(key, width=32, depth=4):
    return _init_dense_mlp(key, 8, 1, width, depth)


def _init_dense_mlp(key, input_dim, output_dim, width, depth):
    dims = [input_dim] + [width] * depth + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    return [{"W": math.sqrt(2.0 / dims[i]) * jax.random.normal(keys[i], (dims[i], dims[i + 1]), dtype=jnp.float64),
             "b": jnp.zeros((dims[i + 1],), dtype=jnp.float64)} for i in range(len(keys))]


def init_deeponet(key, latent_width=64, branch_width=32, branch_depth=3,
                  trunk_width=32, trunk_depth=3):
    branch_key, trunk_key = jax.random.split(key)
    return {
        "branch": _init_dense_mlp(
            branch_key, 6, latent_width, branch_width, branch_depth),
        "trunk": _init_dense_mlp(
            trunk_key, 2, latent_width, trunk_width, trunk_depth),
        "bias": jnp.asarray(0.0, dtype=jnp.float64),
    }


def init_model(key, config: PinnConfig):
    if config.model_type == "deeponet":
        return init_deeponet(
            key, latent_width=config.latent_width,
            branch_width=config.branch_width, branch_depth=config.branch_depth,
            trunk_width=config.trunk_width, trunk_depth=config.trunk_depth)
    return init_mlp(key, width=config.width, depth=config.depth)


def save_model(path, params):
    """Save flattened model parameters."""
    weights, _ = ravel_pytree(params)
    np.savez(path, weights=np.asarray(jax.device_get(weights), dtype=np.float64))


def load_model(path, *, model_type="mlp", width=32, depth=4,
               latent_width=64, branch_width=32, branch_depth=3,
               trunk_width=32, trunk_depth=3):
    """Restore flattened model parameters using recorded architecture."""
    if model_type == "deeponet":
        template = init_deeponet(
            jax.random.PRNGKey(0), latent_width=latent_width,
            branch_width=branch_width, branch_depth=branch_depth,
            trunk_width=trunk_width, trunk_depth=trunk_depth)
    else:
        template = init_mlp(jax.random.PRNGKey(0), width=width, depth=depth)
    _, unravel = ravel_pytree(template)
    with np.load(path, allow_pickle=False) as data:
        weights = jnp.asarray(data["weights"], dtype=jnp.float64)
    return unravel(weights)


def drift_up(z, domain: PinnDomain):
    p, xi, ebar, te, nD, nNe, zD, zNe = split_inputs(z, domain)
    cf, nud, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, domain.B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    return -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)


def pde_coefficients(z, domain: PinnDomain):
    p, xi, ebar, te, nD, nNe, zD, zNe = split_inputs(z, domain)
    cf, nud, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, domain.B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    up = -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)
    dp_dpnorm = (p * jnp.log(domain.p_max / domain.p_min)
                 if domain.momentum_sampling == "log" else
                 jnp.asarray(domain.p_max - domain.p_min, dtype=jnp.float64))
    a_p = -up / dp_dpnorm
    a_xi = 0.5 * ((1.0 - xi * xi) * (ebar / p - alpha * xi / gamma) + nud * xi)
    a_xixi = -0.25 * 0.5 * nud * (1.0 - xi * xi)
    coefficients = jnp.stack([a_p, a_xi, a_xixi], axis=-1)
    return coefficients / jnp.abs(cf) / jnp.sqrt(ebar)


def residual_single(params, z, coefficients):
    phase = z[:2]
    def f_phase(q):
        zz = z.at[0].set(q[0]).at[1].set(q[1])
        return probability(params, zz)
    def probability_phase(q):
        zz = z.at[0].set(q[0]).at[1].set(q[1])
        return probability(params, zz)
    grad_phase = jax.grad(probability_phase)(phase)
    _, second_xi = jax.jvp(
        lambda q: jax.grad(probability_phase)(q)[1],
        (phase,), (jnp.array([0.0, 1.0]),))
    P = probability(params, z)
    return (coefficients[0] * grad_phase[0]
            + coefficients[1] * grad_phase[1]
            + coefficients[2] * second_xi) / (P + 0.1)


def make_pde_functions(domain: PinnDomain):
    coefficients = jax.jit(jax.vmap(lambda z: pde_coefficients(z, domain)))
    residual = jax.jit(jax.vmap(residual_single, in_axes=(None, 0, 0)))
    return coefficients, residual


def evaluate_pde_residuals(params, z, domain: PinnDomain, *, chunk_size=65536):
    """Evaluate direct PDE residuals in bounded host-to-device chunks."""
    z = np.asarray(z, dtype=np.float64)
    coeff_fn, residual_fn = make_pde_functions(domain)
    values = []
    for start in range(0, len(z), chunk_size):
        batch = jnp.asarray(z[start:start + chunk_size])
        values.append(np.asarray(jax.device_get(
            residual_fn(params, batch, coeff_fn(batch)))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _training_devices(requested):
    """Resolve the local GPU set used by data-parallel training."""
    devices = tuple(jax.local_devices(backend="gpu"))
    if not devices:
        raise RuntimeError("multi-device PINN training requires visible GPU devices")
    if requested == 0:
        return devices
    if requested > len(devices):
        raise ValueError(
            f"n_devices={requested} requested, but only {len(devices)} local GPUs are visible")
    return devices[:requested]


def _shard_with_mask(values, devices, valid_count=None):
    """Pad a leading-axis array and place equal-sized shards on local GPUs."""
    values = np.asarray(values)
    n_devices = len(devices)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("cannot shard an empty or scalar array")
    valid_count = values.shape[0] if valid_count is None else int(valid_count)
    if not 0 <= valid_count <= values.shape[0]:
        raise ValueError("valid_count must be within the leading-axis length")
    shard_size = (values.shape[0] + n_devices - 1) // n_devices
    padded_size = shard_size * n_devices
    padded = np.zeros((padded_size,) + values.shape[1:], dtype=values.dtype)
    padded[:values.shape[0]] = values
    mask = np.zeros(padded_size, dtype=np.float64)
    mask[:valid_count] = 1.0
    shards = [padded[start:start + shard_size]
              for start in range(0, padded_size, shard_size)]
    masks = [mask[start:start + shard_size]
             for start in range(0, padded_size, shard_size)]
    return (jax.device_put_sharded(shards, devices),
            jax.device_put_sharded(masks, devices),
            int(values.shape[0]))


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _soap_optimizer(config):
    import optax
    from soap_jax import soap
    learning_rate = config.learning_rate
    if config.learning_rate_schedule == "cosine":
        learning_rate = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=max(1, config.learning_rate_decay_steps or config.steps),
            alpha=config.learning_rate_final_fraction,
        )
    return soap(
        learning_rate=learning_rate, b1=config.soap_b1, b2=config.soap_b2,
        shampoo_beta=config.soap_shampoo_beta, eps=config.soap_eps,
        weight_decay=config.soap_weight_decay, correct_bias=config.soap_correct_bias,
        precondition_frequency=config.soap_precondition_frequency,
        max_precond_dim=config.soap_max_precond_dim,
        precondition_1d=config.soap_precondition_1d,
    )


def _predict_host_chunks(params, z, *, chunk_size=65536):
    """Evaluate predictions without materializing a large device batch."""
    z = np.asarray(z, dtype=np.float64)
    values = []
    for start in range(0, len(z), chunk_size):
        values.append(np.asarray(jax.device_get(
            predict(params, jnp.asarray(z[start:start + chunk_size])))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


class _GroupedDeepONetBatchSampler:
    """Shuffle pre-grouped case tensors without scanning pointwise inputs."""

    def __init__(self, data, case_batch_size, seed):
        self.data = data
        self.size = int(data["branch"].shape[0])
        self.case_batch_size = min(int(case_batch_size), self.size)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.size)
        self.cursor = 0

    def next(self):
        if self.cursor + self.case_batch_size <= self.size:
            ids = self.order[self.cursor:self.cursor + self.case_batch_size]
            self.cursor += self.case_batch_size
        else:
            first = self.order[self.cursor:]
            self.order = self.rng.permutation(self.size)
            needed = self.case_batch_size - len(first)
            ids = np.concatenate((first, self.order[:needed]))
            self.cursor = needed
        return {name: values[ids] for name, values in self.data.items()}


def deeponet_probability(params, branch_z, trunk_z):
    """Evaluate DeepONet output from case parameters and phase-space points."""
    branch = _apply_mlp(params["branch"], branch_z)
    shape = trunk_z.shape
    trunk = _apply_mlp(params["trunk"], trunk_z.reshape((-1, 2)))
    trunk = trunk.reshape(shape[:-1] + (trunk.shape[-1],))
    raw = jnp.sum(branch[..., None, :] * trunk, axis=-1) + params["bias"]
    return probability_transform(raw)


def _shard_leading(values, devices):
    """Place a fixed leading-axis case batch on local devices."""
    values = np.asarray(values)
    n_devices = len(devices)
    shard_size = (values.shape[0] + n_devices - 1) // n_devices
    padded_size = shard_size * n_devices
    padded = np.zeros((padded_size,) + values.shape[1:], dtype=values.dtype)
    padded[:values.shape[0]] = values
    shards = [padded[start:start + shard_size]
              for start in range(0, padded_size, shard_size)]
    return jax.device_put_sharded(shards, devices)


def _grouped_deeponet_mse(params, data, *, max_cases=None):
    if max_cases is not None:
        data = {name: values[:max_cases] for name, values in data.items()}
    numerator = 0.0
    denominator = 0.0
    for start in range(0, data["branch"].shape[0], 32):
        stop = start + 32
        prediction = np.asarray(jax.device_get(deeponet_probability(
            params,
            jnp.asarray(data["branch"][start:stop]),
            jnp.asarray(data["trunk"][start:stop]))))
        error = prediction - data["target"][start:stop]
        mask = data["mask"][start:stop]
        numerator += float(np.sum(mask * error * error))
        denominator += float(np.sum(mask))
    return numerator / max(denominator, 1.0)


def _grouped_deeponet_metrics(params, train_data, test_data):
    metrics = {}
    metrics["train_mse"] = _grouped_deeponet_mse(params, train_data)
    metrics["test_mse"] = _grouped_deeponet_mse(params, test_data)
    return metrics


def train_supervised_deeponet(train_data, test_data, *, config, distributed=False,
                              checkpoint_callback=None, global_case_count=None):
    """Train DeepONet directly from grouped case tensors."""
    devices = _training_devices(config.n_devices)
    local_devices = len(devices)
    process_count = jax.process_count() if distributed else 1
    n_devices = jax.device_count() if distributed else local_devices
    n_cases = int(train_data["branch"].shape[0])
    max_points = int(train_data["trunk"].shape[1])
    if config.case_batch_size > 0:
        global_case_batch_size = config.case_batch_size
    elif config.batch_size > 0:
        global_case_batch_size = max(1, config.batch_size // max_points)
    else:
        total_cases = n_cases * process_count if global_case_count is None else global_case_count
        global_case_batch_size = total_cases
        if distributed:
            global_case_batch_size -= global_case_batch_size % n_devices
            global_case_batch_size = max(n_devices, global_case_batch_size)
    if distributed:
        if global_case_batch_size % n_devices != 0:
            raise ValueError(
                "DeepONet case_batch_size must be divisible by global device count")
        case_batch_size = global_case_batch_size // process_count
    else:
        case_batch_size = global_case_batch_size
    if case_batch_size > n_cases:
        raise ValueError(
            "DeepONet local case batch exceeds local case shard; "
            "reduce case_batch_size or add training cases")
    sampler = _GroupedDeepONetBatchSampler(
        train_data, case_batch_size,
        config.seed + (jax.process_index() if distributed else 0))
    test_case_limit = (test_data["branch"].shape[0]
                       if (config.test_batch_size == 0
                           and (not distributed or jax.process_index() == 0)) else
                       max(1, config.test_batch_size // max_points))
    if distributed and jax.process_index() != 0:
        test_case_limit = 0
    params = init_model(jax.random.PRNGKey(config.seed), config)
    optimizer = _soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    pmap_kwargs = {"axis_name": "data"}
    if not distributed:
        pmap_kwargs["devices"] = devices

    @partial(jax.pmap, **pmap_kwargs)
    def step(params, opt_state, branch_z, trunk_z, target, mask):
        def loss_fn(params):
            error = deeponet_probability(params, branch_z, trunk_z) - target
            count = jax.lax.psum(jnp.sum(mask), axis_name="data")
            return jnp.sum(mask * error * error) * n_devices / jnp.maximum(count, 1.0)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        gradients = jax.lax.pmean(gradients, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    def prepare_host_batch():
        return sampler.next()

    history = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prepare_host_batch)
        for step_number in tqdm(
                range(config.steps), desc="supervised SOAP", unit="step",
                disable=distributed and jax.process_index() != 0):
            host_batch = future.result()
            future = executor.submit(prepare_host_batch)
            batch = tuple(_shard_leading(host_batch[name], devices)
                          for name in ("branch", "trunk", "target", "mask"))
            params, opt_state, loss = step(params, opt_state, *batch)
            if ((step_number + 1) % config.log_every == 0
                    or step_number + 1 == config.steps):
                train_loss = float(jax.device_get(loss[0]))
                test_loss = _grouped_deeponet_mse(
                    _unreplicate(params), test_data, max_cases=test_case_limit)
                if not distributed or jax.process_index() == 0:
                    history.append([train_loss, test_loss])
            checkpoint_due = (
                checkpoint_callback is not None
                and config.checkpoint_every > 0
                and ((step_number + 1) % config.checkpoint_every == 0
                     or step_number + 1 == config.steps))
            if checkpoint_due and distributed and process_count > 1:
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices(
                    f"deeponet_checkpoint_before_{step_number + 1}")
            if checkpoint_due and (not distributed or jax.process_index() == 0):
                checkpoint_callback(step_number + 1, _unreplicate(params))
            if checkpoint_due and distributed and process_count > 1:
                multihost_utils.sync_global_devices(
                    f"deeponet_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("deeponet_training_complete")
    params = _unreplicate(params)
    metrics = (_grouped_deeponet_metrics(params, train_data, test_data)
               if not distributed or jax.process_index() == 0 else {})
    return params, np.asarray(history), metrics



class _GroupedPointSampler:
    """Sample pointwise MLP batches from grouped FV tensors."""

    def __init__(self, data, seed):
        self.data = data
        self.n_cases = int(data["branch"].shape[0])
        self.counts = np.asarray(np.sum(data["mask"], axis=1), dtype=np.int64)
        if self.n_cases == 0 or np.any(self.counts <= 0):
            raise ValueError("grouped point sampler requires nonempty cases")
        self.rng = np.random.default_rng(seed)

    def next(self, size):
        case_ids = self.rng.integers(0, self.n_cases, size=size)
        point_ids = (self.rng.random(size) * self.counts[case_ids]).astype(np.int64)
        trunk = self.data["trunk"][case_ids, point_ids]
        branch = self.data["branch"][case_ids]
        z = np.concatenate((trunk, branch), axis=1)
        y = self.data["target"][case_ids, point_ids]
        return z, np.asarray(y, dtype=np.float64)


def train_supervised_mlp_grouped(train_data, test_data, *, config,
                                 distributed=False, checkpoint_callback=None):
    """Train pointwise MLP from grouped FV data without flattening all cells."""
    devices = _training_devices(config.n_devices)
    process_count = jax.process_count() if distributed else 1
    n_devices = jax.device_count() if distributed else len(devices)
    global_batch_size = config.batch_size if config.batch_size > 0 else 262144
    if distributed:
        if global_batch_size % process_count != 0:
            raise ValueError("MLP batch_size must be divisible by process count")
        local_batch_size = global_batch_size // process_count
    else:
        local_batch_size = global_batch_size
    sampler = _GroupedPointSampler(
        train_data, config.seed + (jax.process_index() if distributed else 0))
    test_sampler = (_GroupedPointSampler(test_data, config.seed + 100003)
                    if (not distributed or jax.process_index() == 0)
                    and test_data["branch"].shape[0] else None)
    params = init_model(jax.random.PRNGKey(config.seed), config)
    optimizer = _soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    pmap_kwargs = {"axis_name": "data"}
    if not distributed:
        pmap_kwargs["devices"] = devices

    @partial(jax.pmap, **pmap_kwargs)
    def step(params, opt_state, z_batch, y_batch, mask):
        def loss_fn(params):
            error = probability(params, z_batch) - y_batch
            count = jax.lax.psum(jnp.sum(mask), axis_name="data")
            return jnp.sum(mask * error * error) * n_devices / jnp.maximum(count, 1.0)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        gradients = jax.lax.pmean(gradients, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    test_size = config.test_batch_size if config.test_batch_size > 0 else global_batch_size
    history = []
    for step_number in tqdm(
            range(config.steps), desc="supervised SOAP", unit="step",
            disable=distributed and jax.process_index() != 0):
        z_host, y_host = sampler.next(local_batch_size)
        z_batch, mask, _ = _shard_with_mask(z_host, devices)
        y_batch, _, _ = _shard_with_mask(y_host, devices)
        params, opt_state, loss = step(params, opt_state, z_batch, y_batch, mask)
        if ((step_number + 1) % config.log_every == 0
                or step_number + 1 == config.steps):
            train_loss = float(jax.device_get(loss[0]))
            if test_sampler is not None:
                z_test, y_test = test_sampler.next(test_size)
                test_prediction = _predict_host_chunks(
                    _unreplicate(params), z_test)
                test_loss = float(np.mean((test_prediction - y_test) ** 2))
            else:
                test_loss = 0.0
            if not distributed or jax.process_index() == 0:
                history.append([train_loss, test_loss])
        checkpoint_due = (
            checkpoint_callback is not None
            and config.checkpoint_every > 0
            and ((step_number + 1) % config.checkpoint_every == 0
                 or step_number + 1 == config.steps))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"mlp_grouped_checkpoint_before_{step_number + 1}")
        if checkpoint_due and (not distributed or jax.process_index() == 0):
            checkpoint_callback(step_number + 1, _unreplicate(params))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"mlp_grouped_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mlp_grouped_training_complete")
    params = _unreplicate(params)
    if not distributed or jax.process_index() == 0:
        z_train, y_train = sampler.next(min(global_batch_size, test_size))
        train_prediction = _predict_host_chunks(params, z_train)
        if test_sampler is not None:
            z_test, y_test = test_sampler.next(test_size)
            test_prediction = _predict_host_chunks(params, z_test)
        else:
            y_test = np.empty(0, dtype=np.float64)
            test_prediction = y_test
        metrics = {
            "train_mse": float(np.mean((train_prediction - y_train) ** 2)),
            "test_mse": (float(np.mean((test_prediction - y_test) ** 2))
                         if len(y_test) else None),
        }
    else:
        metrics = {}
    return params, np.asarray(history), metrics


def _train_physics_multi(z_data, y_data, z_pde, z_bc, *, domain, config,
                         z_pde_threshold=None, z_low=None,
                         initial_params=None, checkpoint_callback=None,
                         distributed=False):
    devices = _training_devices(config.n_devices)
    local_device_count = len(devices)
    process_count = jax.process_count() if distributed else 1
    process_index = jax.process_index() if distributed else 0
    n_devices = jax.device_count() if distributed else local_device_count
    z_data = np.asarray(z_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    z_pde = np.asarray(z_pde, dtype=np.float64)
    z_bc = np.asarray(z_bc, dtype=np.float64)
    if config.enable_data and len(z_data) == 0:
        raise ValueError("enable_data=True requires data points")
    if config.enable_pde and len(z_pde) == 0:
        raise ValueError("enable_pde=True requires PDE points")
    if config.enable_pmax_bc and len(z_bc) == 0:
        raise ValueError("enable_pmax_bc=True requires boundary points")
    if len(z_data) == 0:
        z_data, y_data = np.zeros((1, 8)), np.zeros(1)
    if len(z_pde) == 0:
        z_pde = np.zeros((1, 8))
    if len(z_bc) == 0:
        z_bc = np.zeros((1, 8))
    if z_pde_threshold is None or len(z_pde_threshold) == 0:
        z_pde_threshold = np.asarray(z_pde[:1])
    if z_low is None or len(z_low) == 0:
        z_low = np.asarray(z_bc[:1])
    z_pde_threshold = np.asarray(z_pde_threshold, dtype=np.float64)
    z_low = np.asarray(z_low, dtype=np.float64)

    def local_values(values):
        values = np.asarray(values)
        if distributed:
            local_size = max(1, math.ceil(len(values) / process_count))
            padded = np.zeros(
                (local_size * process_count,) + values.shape[1:],
                dtype=values.dtype)
            padded[:len(values)] = values
            start = process_index * local_size
            local = padded[start:start + local_size]
            valid_count = max(0, min(local_size, len(values) - start))
            return local, valid_count
        if len(values) == 0:
            values = np.zeros((1,) + values.shape[1:], dtype=np.float64)
            return values, 0
        return values, len(values)

    z_data, data_count = local_values(z_data)
    y_data, _ = local_values(y_data)
    z_pde, pde_count = local_values(z_pde)
    z_pde_threshold, threshold_count = local_values(z_pde_threshold)
    z_low, low_count = local_values(z_low)
    z_bc, bc_count = local_values(z_bc)

    if initial_params is None:
        params = init_model(jax.random.PRNGKey(config.seed), config)
    else:
        params = initial_params
    optimizer = _soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    coeff_fn, residual_fn = make_pde_functions(domain)
    pmap_kwargs = {} if distributed else {"devices": devices}
    coeff_pmap = jax.pmap(coeff_fn, **pmap_kwargs)
    boundary_pmap = jax.pmap(
        lambda z: success_boundary_target(z, domain), **pmap_kwargs)

    step_kwargs = {"axis_name": "data"}
    if not distributed:
        step_kwargs["devices"] = devices

    @partial(jax.pmap, **step_kwargs)
    def step(params, opt_state, zd, yd, md, zp, cp, mp, zt, ct, mt,
             zl, ml, zb, mb, active):
        def scaled_mse(values, mask):
            count = jax.lax.psum(jnp.sum(mask), axis_name="data")
            return jnp.sum(mask * values * values) * n_devices / jnp.maximum(count, 1.0)

        def loss_fn(params):
            data_loss = (scaled_mse(probability(params, zd) - yd, md)
                         if config.enable_data else jnp.asarray(0.0, jnp.float64))
            pde_loss = (scaled_mse(residual_fn(params, zp, cp), mp)
                        if config.enable_pde else jnp.asarray(0.0, jnp.float64))
            threshold_loss = (scaled_mse(residual_fn(params, zt, ct), mt)
                if config.enable_pde and config.enable_threshold_pde
                else jnp.asarray(0.0, jnp.float64))
            low_p_loss = (scaled_mse(probability(params, zl), ml)
                          if config.enable_low_p_bc
                          else jnp.asarray(0.0, jnp.float64))
            bc_loss = (scaled_mse(probability(params, zb) - 1.0, mb * active)
                       if config.enable_pmax_bc
                       else jnp.asarray(0.0, jnp.float64))
            total = (config.data_weight * data_loss + config.pde_weight * pde_loss
                     + config.pde_weight * config.threshold_weight * threshold_loss
                     + config.low_p_weight * low_p_loss + config.bc_weight * bc_loss)
            return total, jnp.stack([data_loss, pde_loss, threshold_loss,
                                     low_p_loss, bc_loss])
        (loss, components), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        gradients = jax.lax.pmean(gradients, axis_name="data")
        values = jax.lax.pmean(
            jnp.concatenate((jnp.asarray([loss]), components)), axis_name="data")
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, values

    def sample_shard(values, valid_count, rng):
        batch_size = (max(1, math.ceil(config.batch_size / process_count))
                      if config.batch_size > 0 else 0)
        if batch_size and valid_count > batch_size:
            indices = rng.integers(0, valid_count, size=batch_size)
            values = values[indices]
            valid_count = batch_size
        else:
            values = values[:valid_count] if valid_count else values[:1]
        return _shard_with_mask(values, devices, valid_count)

    def sample_pair_shard(values, targets, valid_count, rng):
        batch_size = (max(1, math.ceil(config.batch_size / process_count))
                      if config.batch_size > 0 else 0)
        if batch_size and valid_count > batch_size:
            indices = rng.integers(0, valid_count, size=batch_size)
            values, targets = values[indices], targets[indices]
            valid_count = batch_size
        else:
            values = values[:valid_count] if valid_count else values[:1]
            targets = targets[:valid_count] if valid_count else targets[:1]
        return (_shard_with_mask(values, devices, valid_count),
                _shard_with_mask(targets, devices, valid_count))

    batch_rng = np.random.default_rng(config.seed + 1)
    history = []
    for step_number in tqdm(
            range(config.steps), desc="physics-informed SOAP", unit="step",
            disable=distributed and process_index != 0):
        data, data_y = sample_pair_shard(
            z_data, y_data, data_count, batch_rng)
        pde = sample_shard(z_pde, pde_count, batch_rng)
        threshold = sample_shard(z_pde_threshold, threshold_count, batch_rng)
        low = sample_shard(z_low, low_count, batch_rng)
        bc = sample_shard(z_bc, bc_count, batch_rng)
        coeff_pde = coeff_pmap(pde[0])
        coeff_threshold = coeff_pmap(threshold[0])
        bc_active = boundary_pmap(bc[0])
        params, opt_state, values = step(
            params, opt_state, data[0], data_y[0], data[1],
            pde[0], coeff_pde, pde[1],
            threshold[0], coeff_threshold, threshold[1],
            low[0], low[1], bc[0], bc[1], bc_active)
        if process_index == 0:
            history.append(np.asarray(jax.device_get(values[0])))
        checkpoint_due = (checkpoint_callback is not None
                and config.checkpoint_every > 0
                and ((step_number + 1) % config.checkpoint_every == 0
                     or step_number + 1 == config.steps))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"physics_checkpoint_before_{step_number + 1}")
        if checkpoint_due and process_index == 0:
            checkpoint_callback(step_number + 1, _unreplicate(params))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"physics_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("physics_training_complete")
    return _unreplicate(params), np.asarray(history), {
        "pde_coefficients": None, "n_devices": n_devices}


def success_boundary_target(z, domain: PinnDomain):
    """Return active p_max mask where outward drift makes escape successful."""
    p = jnp.asarray(domain.p_max, dtype=jnp.float64)
    xi = -1.0 + 2.0 * z[..., 1]
    ebar = _map_unit(z[..., 2], domain.ebar_min, domain.ebar_max, "log")
    te = _map_unit(z[..., 3], domain.te_min_eV, domain.te_max_eV, "log")
    nD = _map_unit(z[..., 4], domain.nD_min_m3, domain.nD_max_m3, "log")
    nNe = _map_unit(z[..., 5], domain.nNe_min_m3, domain.nNe_max_m3, "log")
    zD = domain.zD_min + z[..., 6] * (domain.zD_max - domain.zD_min)
    zNe = domain.zNe_min + z[..., 7] * (domain.zNe_max - domain.zNe_min)
    cf, _nud, alpha = collision_coefficients(p, te, nD, nNe, zD, zNe, domain.B_T)
    gamma = jnp.sqrt(1.0 + p * p)
    up = -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)
    return (up > 0.0).astype(jnp.float64)


def train_physics_informed(z_data, y_data, z_pde, z_bc, *, domain,
                           config=None, z_pde_threshold=None, z_low=None,
                           initial_params=None, checkpoint_callback=None,
                           distributed=False):
    """Train FC PINN with FV data, PDE residual, and successful-pmax BC."""
    config = PinnConfig() if config is None else config
    if distributed or config.n_devices != 1 or config.batch_size > 0:
        return _train_physics_multi(
            z_data, y_data, z_pde, z_bc, domain=domain, config=config,
            z_pde_threshold=z_pde_threshold, z_low=z_low,
            initial_params=initial_params,
            checkpoint_callback=checkpoint_callback, distributed=distributed)
    if initial_params is None:
        key = jax.random.PRNGKey(config.seed)
        params = init_model(key, config)
    else:
        params = initial_params
    optimizer = _soap_optimizer(config)
    opt_state = optimizer.init(params)
    z_data, y_data = jnp.asarray(z_data), jnp.asarray(y_data)
    z_pde, z_bc = jnp.asarray(z_pde), jnp.asarray(z_bc)
    if config.enable_data and len(z_data) == 0:
        raise ValueError("enable_data=True requires data points")
    if config.enable_pde and len(z_pde) == 0:
        raise ValueError("enable_pde=True requires PDE points")
    if config.enable_pmax_bc and len(z_bc) == 0:
        raise ValueError("enable_pmax_bc=True requires boundary points")
    use_low = z_low is not None and len(z_low) > 0
    if config.enable_low_p_bc and not use_low:
        raise ValueError("enable_low_p_bc=True requires low-p boundary points")
    z_data = z_data if len(z_data) else jnp.zeros((1, 8), dtype=jnp.float64)
    y_data = y_data if len(y_data) else jnp.zeros((1,), dtype=jnp.float64)
    z_pde = z_pde if len(z_pde) else jnp.zeros((1, 8), dtype=jnp.float64)
    z_bc = z_bc if len(z_bc) else jnp.zeros((1, 8), dtype=jnp.float64)
    use_threshold = z_pde_threshold is not None and len(z_pde_threshold) > 0
    threshold_weight = config.threshold_weight if use_threshold and config.enable_threshold_pde else 0.0
    z_pde_threshold = (jnp.asarray(z_pde_threshold) if use_threshold else z_pde[:1])
    z_low = jnp.asarray(z_low) if use_low else z_bc[:1]
    coeff_fn, residual_fn = make_pde_functions(domain)
    coeff_pde = coeff_fn(z_pde)
    coeff_pde_threshold = coeff_fn(z_pde_threshold)
    bc_active = (success_boundary_target(z_bc, domain)
                 if config.enable_pmax_bc else jnp.zeros(len(z_bc), dtype=jnp.float64))
    @jax.jit
    def step(params, opt_state, zd, yd, zp, cb, zt, ct, zl, zb, yb):
        def loss_fn(params):
            data_loss = (jnp.mean((probability(params, zd) - yd) ** 2)
                         if config.enable_data else jnp.asarray(0.0, dtype=jnp.float64))
            pde_loss = (jnp.mean(residual_fn(params, zp, cb) ** 2)
                        if config.enable_pde else jnp.asarray(0.0, dtype=jnp.float64))
            threshold_loss = (jnp.mean(residual_fn(params, zt, ct) ** 2)
                              if config.enable_pde and config.enable_threshold_pde
                              else jnp.asarray(0.0, dtype=jnp.float64))
            low_p_loss = (jnp.mean(probability(params, zl) ** 2)
                          if config.enable_low_p_bc
                          else jnp.asarray(0.0, dtype=jnp.float64))
            bc_loss = jnp.sum(yb * (probability(params, zb) - 1.0) ** 2) / jnp.maximum(jnp.sum(yb), 1.0)
            if not config.enable_pmax_bc:
                bc_loss = jnp.asarray(0.0, dtype=jnp.float64)
            total = (config.data_weight * data_loss + config.pde_weight * pde_loss
                     + config.pde_weight * threshold_weight * threshold_loss
                     + config.low_p_weight * low_p_loss
                     + config.bc_weight * bc_loss)
            return total, \
                   jnp.stack([data_loss, pde_loss, threshold_loss, low_p_loss, bc_loss])
        (loss, components), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss, jnp.concatenate((jnp.asarray([loss]), components))

    history = []
    for step_number in tqdm(range(config.steps), desc="physics-informed SOAP", unit="step"):
        params, opt_state, loss, components = step(
            params, opt_state, z_data, y_data, z_pde, coeff_pde,
            z_pde_threshold, coeff_pde_threshold, z_low, z_bc, bc_active)
        history.append(np.asarray(components))
        if (checkpoint_callback is not None
                and config.checkpoint_every > 0
                and ((step_number + 1) % config.checkpoint_every == 0
                     or step_number + 1 == config.steps)):
            checkpoint_callback(step_number + 1, params)
    return params, np.asarray(history), {"pde_coefficients": coeff_pde}


def train_physics_active(z_data, y_data, z_pde, z_bc, *, domain,
                         config=None, z_pde_threshold=None, z_low=None,
                         active_config=None, acquire_data=None,
                         checkpoint_callback=None, distributed=False,
                         initial_params=None):
    """Train physics-informed cycles with residual-guided FV acquisition.

    ``acquire_data`` receives selected normalized eight-dimensional points and
    returns additional pointwise ``(z, y)`` FV labels. The callback stays
    outside JAX so CPU FV generation never enters autodiff.
    """
    config = PinnConfig() if config is None else config
    active_config = {} if active_config is None else dict(active_config)
    cycles = int(active_config.get("cycles", 1))
    dense_points = int(active_config.get("dense_points", 131072))
    threshold_points = int(active_config.get("threshold_points", 0))
    acquire_points = int(active_config.get("acquire_points", 32))
    seed = int(active_config.get("seed", config.seed + 10000))
    if cycles <= 0 or dense_points <= 0 or acquire_points <= 0:
        raise ValueError("invalid active-training setting")
    if not config.enable_pde:
        raise ValueError("active training requires enable_pde=True")
    if acquire_data is None:
        raise ValueError("active training requires an acquire_data callback")

    z_data = np.asarray(z_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    z_pde = np.asarray(z_pde, dtype=np.float64)
    params = initial_params
    histories = []
    records = []
    for cycle in range(cycles):
        def cycle_checkpoint(step, cycle_params, *, cycle_number=cycle):
            if checkpoint_callback is not None:
                checkpoint_callback(cycle_number * config.steps + step,
                                    cycle_params)

        params, history, _ = train_physics_informed(
            z_data, y_data, z_pde, z_bc, domain=domain,
            config=config, z_pde_threshold=z_pde_threshold, z_low=z_low,
            initial_params=params, checkpoint_callback=cycle_checkpoint,
            distributed=distributed)
        if len(history):
            histories.append(np.asarray(history))
        if cycle + 1 == cycles:
            break

        dense = sobol_8d_nontrivial(
            dense_points, domain, seed + cycle,
            angular_sampling=config.angular_sampling,
            momentum_sampling=config.momentum_sampling)
        candidate_parts = [dense]
        if threshold_points > 0:
            candidate_parts.append(analytic_threshold_collocation(
                threshold_points, domain, seed=seed + 1000 + cycle,
                band_width=float(active_config.get(
                    "threshold_band_width", 0.02))))
        candidates = np.concatenate(candidate_parts, axis=0)
        residual = evaluate_pde_residuals(params, candidates, domain)
        score = np.abs(np.asarray(residual, dtype=np.float64))
        score[~np.isfinite(score)] = -np.inf
        count = min(acquire_points, len(candidates))
        selected = np.argpartition(score, -count)[-count:]
        selected = selected[np.argsort(score[selected])[::-1]]
        z_selected = candidates[selected]
        z_new, y_new = acquire_data(z_selected)
        z_new = np.asarray(z_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if len(z_new):
            if z_new.ndim != 2 or z_new.shape[1] != 8:
                raise ValueError("acquired z data must have shape (N, 8)")
            if len(z_new) != len(y_new):
                raise ValueError("acquired z and y data lengths must match")
            z_data = np.concatenate((z_data, z_new), axis=0)
            y_data = np.concatenate((y_data, y_new), axis=0)
        z_pde = np.concatenate((z_pde, z_selected), axis=0)
        records.append({
            "cycle": cycle + 1,
            "dense_points": int(len(candidates)),
            "threshold_points": int(threshold_points),
            "selected_points": int(len(z_selected)),
            "acquired_points": int(len(z_new)),
            "max_residual": float(np.max(score[selected])),
            "rms_residual": float(np.sqrt(np.mean(residual * residual))),
            "data_points_after": int(len(z_data)),
            "pde_points_after": int(len(z_pde)),
        })
    combined_history = (np.concatenate(histories, axis=0)
                        if histories else np.empty((0, 6), dtype=np.float64))
    return params, combined_history, records


def _normalize_parameter_cases(cases, parameter_domain):
    cases = np.asarray(cases, dtype=np.float64)
    normalized = np.empty_like(cases)
    for k, (lo, hi, scale) in enumerate(parameter_domain.values()):
        normalized[:, k] = (
            (np.log(cases[:, k]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if scale == "log" else (cases[:, k] - lo) / (hi - lo))
    return normalized


def exact_deeponet_inputs(cases, results, parameter_domain, domain, low_p_Np=0,
                          max_points=None):
    """Build grouped normalized DeepONet tensors directly from FV cases."""
    cases = np.asarray(cases, dtype=np.float64)
    if len(cases) != len(results) or len(cases) == 0:
        raise ValueError("cases and FV results must have equal nonzero length")
    if low_p_Np < 0:
        raise ValueError("low-p Np must be non-negative")
    case_norm = _normalize_parameter_cases(cases, parameter_domain)
    trunks, targets = [], []
    for result in results:
        p = np.asarray(result["p"], dtype=np.float64)
        xi = np.asarray(result["xi"], dtype=np.float64)
        field = np.asarray(result["P"], dtype=np.float64)
        if field.shape != (len(p), len(xi)):
            raise ValueError("FV field shape does not match its grid")
        pp, xx = np.meshgrid(p, xi, indexing="ij")
        p_norm = normalize_momentum(
            pp.ravel(), domain.p_min, domain.p_max, domain.momentum_sampling)
        xi_norm = 0.5 * (xx.ravel() + 1.0)
        trunk = np.column_stack((p_norm, xi_norm))
        target = field.ravel()
        if low_p_Np:
            p_upper = float(result["p_min"])
            if p_upper > domain.p_min:
                p_low = np.geomspace(domain.p_min, p_upper, low_p_Np + 2)[1:-1]
                pp_low, xx_low = np.meshgrid(p_low, xi, indexing="ij")
                trunk = np.vstack((
                    trunk,
                    np.column_stack((
                        normalize_momentum(pp_low.ravel(), domain.p_min,
                                           domain.p_max, domain.momentum_sampling),
                        0.5 * (xx_low.ravel() + 1.0),
                    )),
                ))
                target = np.concatenate((target, np.zeros(pp_low.size)))
        trunks.append(trunk)
        targets.append(target)
    local_max_points = max(len(trunk) for trunk in trunks)
    if max_points is None:
        max_points = local_max_points
    if max_points < local_max_points:
        raise ValueError("max_points is smaller than a grouped FV case")
    data = {
        "branch": case_norm,
        "trunk": np.zeros((len(cases), max_points, 2), dtype=np.float64),
        "target": np.zeros((len(cases), max_points), dtype=np.float64),
        "mask": np.zeros((len(cases), max_points), dtype=np.float64),
    }
    for index, (trunk, target) in enumerate(zip(trunks, targets)):
        count = len(trunk)
        data["trunk"][index, :count] = trunk
        data["target"][index, :count] = target
        data["mask"][index, :count] = 1.0
    return data


def grouped_to_pointwise(data):
    """Flatten grouped case tensors into exact pointwise model inputs."""
    z_parts, y_parts = [], []
    for index in range(data["branch"].shape[0]):
        valid = np.asarray(data["mask"][index]) > 0.0
        trunk = np.asarray(data["trunk"][index])[valid]
        branch = np.repeat(
            np.asarray(data["branch"][index])[None, :], len(trunk), axis=0)
        z_parts.append(np.concatenate((trunk, branch), axis=1))
        y_parts.append(np.asarray(data["target"][index])[valid])
    if not z_parts:
        return np.empty((0, 8), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(z_parts), np.concatenate(y_parts)


def train_ssbroyden(params, z_data, y_data, z_pde, z_bc, *, domain,
                    config=None, z_pde_threshold=None, z_low=None,
                    checkpoint_callback=None, step_offset=0):
    """Refine PINN with blockwise full-batch SSBroyden."""
    import equinox as eqx
    import optimistix as optx

    config = PinnConfig() if config is None else config
    z_data, y_data = jnp.asarray(z_data), jnp.asarray(y_data)
    z_pde, z_bc = jnp.asarray(z_pde), jnp.asarray(z_bc)
    if config.enable_data and len(z_data) == 0:
        raise ValueError("enable_data=True requires data points")
    if config.enable_pde and len(z_pde) == 0:
        raise ValueError("enable_pde=True requires PDE points")
    if config.enable_pmax_bc and len(z_bc) == 0:
        raise ValueError("enable_pmax_bc=True requires boundary points")
    use_low = z_low is not None and len(z_low) > 0
    if config.enable_low_p_bc and not use_low:
        raise ValueError("enable_low_p_bc=True requires low-p boundary points")
    z_data = z_data if len(z_data) else jnp.zeros((1, 8), dtype=jnp.float64)
    y_data = y_data if len(y_data) else jnp.zeros((1,), dtype=jnp.float64)
    z_pde = z_pde if len(z_pde) else jnp.zeros((1, 8), dtype=jnp.float64)
    z_bc = z_bc if len(z_bc) else jnp.zeros((1, 8), dtype=jnp.float64)
    use_threshold = z_pde_threshold is not None and len(z_pde_threshold) > 0
    threshold_weight = config.threshold_weight if use_threshold and config.enable_threshold_pde else 0.0
    z_pde_threshold = (jnp.asarray(z_pde_threshold) if use_threshold else z_pde[:1])
    z_low = jnp.asarray(z_low) if use_low else z_bc[:1]
    coeff_fn, residual_fn = make_pde_functions(domain)
    coeff_pde = coeff_fn(z_pde)
    coeff_pde_threshold = coeff_fn(z_pde_threshold)
    bc_active = success_boundary_target(z_bc, domain)
    weights, unravel = ravel_pytree(params)

    def loss_components(p):
        data_loss = (jnp.mean((probability(p, z_data) - y_data) ** 2)
                     if config.enable_data else jnp.asarray(0.0, dtype=jnp.float64))
        pde_loss = (jnp.mean(residual_fn(p, z_pde, coeff_pde) ** 2)
                    if config.enable_pde else jnp.asarray(0.0, dtype=jnp.float64))
        threshold_loss = (jnp.mean(residual_fn(p, z_pde_threshold, coeff_pde_threshold) ** 2)
                          if config.enable_pde and config.enable_threshold_pde
                          else jnp.asarray(0.0, dtype=jnp.float64))
        low_p_loss = (jnp.mean(probability(p, z_low) ** 2)
                      if config.enable_low_p_bc
                      else jnp.asarray(0.0, dtype=jnp.float64))
        bc_loss = jnp.sum(bc_active * (probability(p, z_bc) - 1.0) ** 2) / jnp.maximum(jnp.sum(bc_active), 1.0)
        if not config.enable_pmax_bc:
            bc_loss = jnp.asarray(0.0, dtype=jnp.float64)
        total = (config.data_weight * data_loss + config.pde_weight * pde_loss
                 + config.pde_weight * threshold_weight * threshold_loss
                 + config.low_p_weight * low_p_loss
                 + config.bc_weight * bc_loss)
        return total, jnp.stack([data_loss, pde_loss, threshold_loss, low_p_loss, bc_loss])

    def scalar_loss(w, _args=None):
        return loss_components(unravel(w))[0]

    class SSBroyden(optx.AbstractSSBroyden):
        rtol: float = config.ssbroyden_rtol
        atol: float = config.ssbroyden_atol
        norm: callable = optx.max_norm
        use_inverse: bool = True
        search: optx.AbstractSearch = eqx.field(default_factory=optx.BacktrackingStrongWolfe)
        descent: optx.AbstractDescent = eqx.field(default_factory=optx.NewtonDescent)
        verbose: frozenset[str] = frozenset()

    solver = optx.BestSoFarMinimiser(SSBroyden())

    @eqx.filter_jit
    def solve_block(w0):
        before = scalar_loss(w0)
        solution = optx.minimise(scalar_loss, solver, w0,
                                 max_steps=config.ssbroyden_block_iters, throw=False)
        after = scalar_loss(solution.value)
        finite = jnp.all(jnp.isfinite(solution.value)) & jnp.isfinite(after)
        return solution.value, before, after, solution.stats["num_steps"], finite, solution.result

    history = []
    for block in tqdm(range(config.ssbroyden_blocks), desc="SSBroyden refinement", unit="block"):
        candidate, before, after, n_steps, finite, solver_result = solve_block(weights)
        if not bool(jax.device_get(finite)):
            raise FloatingPointError("SSBroyden produced non-finite model parameters")
        before_value = float(jax.device_get(before))
        after_value = float(jax.device_get(after))
        accepted = after_value <= before_value * (1.0 + 1.0e-10)
        if accepted:
            weights = candidate
        params = unravel(weights)
        total, components = loss_components(params)
        record = {
            "block": block + 1,
            "steps": int(jax.device_get(n_steps)),
            "solver_result": str(solver_result),
            "before": float(jax.device_get(before)),
            "after": float(jax.device_get(after)),
            "accepted": accepted,
            "total": float(jax.device_get(total)),
            "data": float(jax.device_get(components[0])),
            "pde": float(jax.device_get(components[1])),
            "pde_threshold": float(jax.device_get(components[2])),
            "low_p": float(jax.device_get(components[3])),
            "bc": float(jax.device_get(components[4])),
        }
        history.append(record)
        checkpoint_due = (
            checkpoint_callback is not None
            and config.checkpoint_every > 0
            and ((step_offset + block + 1) % config.checkpoint_every == 0
                 or block + 1 == config.ssbroyden_blocks))
        if checkpoint_due:
            checkpoint_callback(step_offset + block + 1, params)
    return unravel(weights), history


def detect_local_gpu_count():
    """Return the number of GPUs visible to this process."""
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS"):
        visible = os.environ.get(name)
        if visible and visible not in ("NoDevFiles", "-1"):
            return len([item for item in visible.split(",") if item.strip()])
    slurm_count = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if slurm_count:
        try:
            return int(slurm_count.rsplit(":", 1)[-1])
        except ValueError:
            pass
    visible = os.environ.get("SLURM_JOB_GPUS")
    if visible and visible not in ("NoDevFiles", "-1"):
        return len([item for item in visible.split(",") if item.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        return sum(line.startswith("GPU ") for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def initialize_slurm_jax(local_gpu_count):
    """Initialize multi-host JAX collectives for an srun step."""
    if "SLURM_PROCID" not in os.environ:
        return
    process_count = int(os.environ.get("SLURM_NTASKS", "1"))
    process_id = int(os.environ.get("SLURM_PROCID", "0"))
    if process_count <= 1:
        return
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        text=True).splitlines()
    port = 29500 + int(os.environ.get("SLURM_JOB_ID", "0")) % 1000
    coordinator_host = socket.gethostbyname(hosts[0])
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_host}:{port}",
        num_processes=process_count,
        process_id=process_id,
        local_device_ids=list(range(local_gpu_count)),
        coordinator_bind_address=f"0.0.0.0:{port}",
    )


# Canonical ownership after the split: the names above remain only as a
# compatibility scaffold while all active calls resolve to the new modules.
from core.training_config import PinnConfig, PinnDomain, make_pinn_config
from core.model import (
    deeponet_probability, init_model, load_model, predict, probability,
    save_model,
)
from core.pde import (
    analytic_threshold_collocation, evaluate_pde_residuals,
    make_pde_functions, normalize_momentum, sobol_8d_nontrivial,
    sobol_plow_boundary_nontrivial, sobol_pmax_boundary_nontrivial,
    success_boundary_target,
)


def _split_cases(cases, results, train_fraction, seed):
    permutation = np.random.default_rng(seed).permutation(len(cases))
    n_train = min(max(1, int(round(train_fraction * len(cases)))), len(cases) - 1)
    train_ids = np.sort(permutation[:n_train])
    test_ids = np.sort(permutation[n_train:])
    return (cases[train_ids], [results[i] for i in train_ids],
            cases[test_ids], [results[i] for i in test_ids])


def train_data_from_config(config_path=Path("run_configs/train.json"),
                           run_config=None):
    """Config-driven data/DeepONet training orchestration."""
    from core.fv_dataset import load_fv_cases
    from core.training_artifacts import (
        atomic_write_json, begin_run, distributed_file_barrier,
        ensure_training_output_dir, finish_run, make_checkpoint_callback,
        validate_dataset_manifest, write_training_outputs,
    )
    config_path = Path(config_path)
    if run_config is None:
        run_config = json.loads(config_path.read_text())
    local_gpus = detect_local_gpu_count()
    if local_gpus <= 0:
        raise RuntimeError("no visible GPUs found")
    launched = "SLURM_PROCID" in os.environ
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1")) if launched else 1
    run_dir = Path(run_config.get("run_dir", run_config["output_dir"]))
    launch_id = os.environ.get("SLURM_STEP_ID") or os.environ.get(
        "SLURM_JOB_ID", str(os.getpid()))
    if not launched:
        launch_id = f"{launch_id}.{os.getpid()}"
    launch_dir = run_dir.parent / f".{run_dir.name}.launch.{launch_id}"
    manifest = None
    try:
        if rank == 0:
            launch_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(launch_dir / "launch.json", {
                "config": str(config_path.resolve()), "world": world,
            })
        distributed_file_barrier(launch_dir, "launch", rank, world)
        initialize_slurm_jax(local_gpus)
        if len(jax.local_devices()) != local_gpus:
            raise RuntimeError("local GPU count changed during JAX initialization")
        config = make_pinn_config(run_config)
        if config.training_mode != "data":
            raise ValueError("data training requires mode='data'")
        validate_dataset_manifest(
            run_config["dataset_path"], run_config["dataset_manifest"],
            expected_config=run_config,
            expected_dataset_config={
                "p_max": run_config["domain"]["p_max"],
                "B_T": run_config["domain"]["B_T"],
            })
        output_dir = Path(run_config["output_dir"])
        ensure_training_output_dir(output_dir)
        distributed_file_barrier(launch_dir, "preflight", rank, world)
        if jax.process_index() == 0:
            manifest = begin_run(run_dir, "pinn_training", run_config)
        distributed_file_barrier(launch_dir, "run_started", rank, world)
        checkpoint_dir = Path(run_config.get("checkpoint_dir", output_dir / "checkpoints"))
        checkpoint = (make_checkpoint_callback(
            checkpoint_dir, save_model, enabled=(jax.process_index() == 0))
            if config.checkpoint_every > 0 else None)
        cases, results = load_fv_cases(run_config["dataset_path"])
        train_cases, train_results, test_cases, test_results = _split_cases(
            cases, results, config.train_case_fraction, config.seed + 1)
        process = jax.process_index()
        processes = jax.process_count()
        local_cases = train_cases[process::processes]
        local_results = train_results[process::processes]
        def points(result):
            count = len(result["p"]) * len(result["xi"])
            if run_config.get("data", {}).get("low_p_Np", 0) and result["p_min"] > config.domain.p_min:
                count += int(run_config["data"]["low_p_Np"]) * len(result["xi"])
            return count
        max_points = max(points(result) for result in train_results + test_results)
        grouped_train = exact_deeponet_inputs(
            local_cases, local_results, run_config["parameter_domain"],
            config.domain, int(run_config.get("data", {}).get("low_p_Np", 0)),
            max_points=max_points)
        if process == 0:
            grouped_test = exact_deeponet_inputs(
                test_cases, test_results, run_config["parameter_domain"],
                config.domain, int(run_config.get("data", {}).get("low_p_Np", 0)),
                max_points=max_points)
        else:
            grouped_test = {
                "branch": np.empty((0, 6)),
                "trunk": np.empty((0, max_points, 2)),
                "target": np.empty((0, max_points)),
                "mask": np.empty((0, max_points)),
            }
        distributed_file_barrier(launch_dir, "data_ready", rank, world)
        if config.model_type == "deeponet":
            params, history, metrics = train_supervised_deeponet(
                grouped_train, grouped_test, config=config, distributed=True,
                checkpoint_callback=checkpoint,
                global_case_count=min(len(train_cases[i::processes])
                                      for i in range(processes)) * processes)
        elif config.model_type == "mlp":
            params, history, metrics = train_supervised_mlp_grouped(
                grouped_train, grouped_test, config=config, distributed=True,
                checkpoint_callback=checkpoint)
        else:
            raise ValueError(f"unsupported model_type: {config.model_type}")
        if jax.process_index() == 0:
            history = np.asarray(history)
            metadata = {
                "architecture": {
                    "input_dim": 8, "model_type": config.model_type,
                    "width": config.width, "depth": config.depth,
                    "latent_width": config.latent_width,
                    "branch_width": config.branch_width,
                    "branch_depth": config.branch_depth,
                    "trunk_width": config.trunk_width,
                    "trunk_depth": config.trunk_depth,
                },
                "domain": run_config["domain"],
                "normalization": {"p_floor": config.domain.p_min,
                                  "p_max": config.domain.p_max},
                "training_config": {
                    "model": run_config.get("model", run_config.get("pinn_config", {})),
                    "loss": run_config.get("loss", {}),
                    "optimizer": run_config.get("optimizer", {}),
                    "data": run_config.get("data", {}),
                },
                "parameter_domain": run_config["parameter_domain"],
                "dataset": run_config["dataset_path"],
            }
            summary = {
                "dataset_path": run_config["dataset_path"],
                "output_dir": str(output_dir),
                "train_case_count": int(len(train_cases)),
                "test_case_count": int(len(test_cases)),
                "model_type": config.model_type, "metrics": metrics,
                "history": history.tolist(),
                "history_columns": (["train_loss", "test_loss"]
                                    if history.ndim == 2 and history.shape[1] == 2 else None),
                "history_steps": ((np.arange(len(history)) + 1) * config.log_every).tolist(),
                "log_every": config.log_every,
                "checkpoint_every": config.checkpoint_every,
                "global_device_count": int(jax.device_count()),
                "process_count": int(jax.process_count()),
            }
            artifacts = write_training_outputs(
                output_dir, params, save_model, run_config=run_config,
                metadata=metadata, summary=summary,
                loss_history={"steps": summary["history_steps"],
                              "columns": summary["history_columns"],
                              "values": summary["history"]})
            finish_run(manifest, status="complete", artifacts=artifacts, metrics=metrics)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("training_artifacts_complete")
    except Exception as exc:
        try:
            atomic_write_json(launch_dir / f"failure.rank{rank}.json", {
                "rank": rank, "error": repr(exc),
            })
        except OSError:
            pass
        if manifest is not None:
            finish_run(manifest, status="failed", error=repr(exc))
        raise


def _normalize_parameter_cases(cases, parameter_domain):
    cases = np.asarray(cases, dtype=np.float64)
    normalized = np.empty_like(cases)
    for column, (lo, hi, scale) in enumerate(parameter_domain.values()):
        normalized[:, column] = (
            (np.log(cases[:, column]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if scale == "log" else (cases[:, column] - lo) / (hi - lo))
    return normalized


def _build_physics_inputs(dataset, cases, parameter_domain, domain, low_p_Np):
    flat = flatten_fv_dataset(dataset)
    case_index = np.asarray(flat["case_index"], dtype=np.int64)
    p = np.asarray(flat["p"], dtype=np.float64)
    z = np.empty((len(p), 8), dtype=np.float64)
    z[:, 0] = normalize_momentum(
        p, domain.p_min, domain.p_max, domain.momentum_sampling)
    z[:, 1] = 0.5 * (np.asarray(flat["xi"]) + 1.0)
    case_norm = _normalize_parameter_cases(cases, parameter_domain)
    z[:, 2:] = case_norm[case_index]
    y = np.asarray(flat["P"], dtype=np.float64)
    metadata = json.loads(str(np.asarray(dataset["case_metadata_json"]).item()))
    low_z, low_y, low_indices = [], [], []
    xi_grid = np.asarray(dataset["xi_grid"], dtype=np.float64)
    for index, item in enumerate(metadata):
        p_min_case = float(item["p_min"])
        if low_p_Np <= 0 or p_min_case <= domain.p_min:
            continue
        p_low = np.geomspace(domain.p_min, p_min_case, low_p_Np + 2)[1:-1]
        pp, xx = np.meshgrid(p_low, xi_grid[index], indexing="ij")
        z_low = np.empty((pp.size, 8), dtype=np.float64)
        z_low[:, 0] = normalize_momentum(
            pp.ravel(), domain.p_min, domain.p_max, domain.momentum_sampling)
        z_low[:, 1] = 0.5 * (xx.ravel() + 1.0)
        z_low[:, 2:] = case_norm[index]
        low_z.append(z_low)
        low_y.append(np.zeros(pp.size, dtype=np.float64))
        low_indices.append(np.full(pp.size, index, dtype=np.int64))
    if low_z:
        z = np.concatenate((z, np.concatenate(low_z)))
        y = np.concatenate((y, np.concatenate(low_y)))
        case_index = np.concatenate((case_index, np.concatenate(low_indices)))
    return z, y, case_index


def train_physics_from_config(config_path=Path("run_configs/train.json"),
                              config=None):
    """Config-driven physics-informed training orchestration."""
    from core.fv_dataset import (
        coarsen_fv_cases, denormalize_parameter_cases, load_fv_dataset,
        generate_cpu_cases,
    )
    from core.training_artifacts import (
        atomic_write_json, begin_run, distributed_file_barrier,
        ensure_training_output_dir, finish_run, make_checkpoint_callback,
        validate_dataset_manifest, write_training_outputs,
    )
    config_path = Path(config_path)
    if config is None:
        config = json.loads(config_path.read_text())
    if config.get("mode") not in (None, "physics"):
        raise ValueError("set mode='physics' in the training config")
    local_gpus = detect_local_gpu_count()
    if local_gpus <= 0:
        raise RuntimeError("no visible GPUs found")
    launched = "SLURM_PROCID" in os.environ
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1")) if launched else 1
    run_dir = Path(config.get("run_dir", config["output_dir"]))
    launch_id = os.environ.get("SLURM_STEP_ID") or os.environ.get(
        "SLURM_JOB_ID", str(os.getpid()))
    if not launched:
        launch_id = f"{launch_id}.{os.getpid()}"
    launch_dir = run_dir.parent / f".{run_dir.name}.launch.{launch_id}"
    manifest = None
    try:
        if rank == 0:
            launch_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(launch_dir / "launch.json", {
                "config": str(config_path.resolve()), "world": world,
            })
        distributed_file_barrier(launch_dir, "launch", rank, world)
        initialize_slurm_jax(local_gpus)
        if len(jax.local_devices()) != local_gpus:
            raise RuntimeError("local GPU count changed during JAX initialization")
        if jax.default_backend() != "gpu":
            raise RuntimeError("GPU backend required for physics training")
        training_config = make_pinn_config(config)
        if training_config.training_mode != "physics_informed":
            raise ValueError("physics training requires mode='physics'")
        if training_config.model_type != "mlp":
            raise ValueError("physics training requires model.model_type='mlp'")
        initial_model = config.get("initial_model")
        initial_params = None
        if initial_model:
            initial_params = load_model(
                initial_model, model_type=training_config.model_type,
                width=training_config.width, depth=training_config.depth,
                latent_width=training_config.latent_width,
                branch_width=training_config.branch_width,
                branch_depth=training_config.branch_depth,
                trunk_width=training_config.trunk_width,
                trunk_depth=training_config.trunk_depth)
        dataset_path = Path(config["dataset_path"])
        validate_dataset_manifest(
            dataset_path, config["dataset_manifest"], expected_config=config,
            expected_dataset_config={"p_max": config["domain"]["p_max"],
                                     "B_T": config["domain"]["B_T"]})
        output_dir = Path(config["output_dir"])
        ensure_training_output_dir(output_dir)
        distributed_file_barrier(launch_dir, "preflight", rank, world)
        if jax.process_index() == 0:
            manifest = begin_run(run_dir, "physics_pinn_training", config)
        distributed_file_barrier(launch_dir, "run_started", rank, world)
        domain = PinnDomain(**config["domain"])
        checkpoint_dir = Path(config.get("checkpoint_dir", output_dir / "checkpoints"))
        checkpoint = (make_checkpoint_callback(
            checkpoint_dir, save_model, enabled=(jax.process_index() == 0))
            if training_config.checkpoint_every > 0 else None)
        dataset = load_fv_dataset(dataset_path)
        cases = np.asarray(dataset["cases"], dtype=np.float64)
        low_p_Np = int(config.get("data", {}).get("low_p_Np", 0))
        z_all, y_all, case_index = _build_physics_inputs(
            dataset, cases, config["parameter_domain"], domain, low_p_Np)
        rng = np.random.default_rng(training_config.seed)
        permutation = rng.permutation(len(cases))
        n_train = min(
            max(1, int(round(training_config.train_case_fraction * len(cases)))),
            len(cases) - 1)
        train_cases = np.sort(permutation[:n_train])
        test_cases = np.sort(permutation[n_train:])
        train_pool = np.flatnonzero(np.isin(case_index, train_cases))
        test_pool = np.flatnonzero(np.isin(case_index, test_cases))
        train_count = min(int(config.get("data", {}).get("train_points", len(train_pool))),
                          len(train_pool))
        test_count = min(int(config.get("data", {}).get("test_points", len(test_pool))),
                         len(test_pool))
        train_idx = rng.choice(train_pool, size=train_count, replace=False)
        test_idx = rng.choice(test_pool, size=test_count, replace=False)
        collocation = config["collocation"]
        z_pde = sobol_8d_nontrivial(
            int(collocation["pde_points"]), domain, training_config.seed + 1,
            angular_sampling=training_config.angular_sampling,
            momentum_sampling=training_config.momentum_sampling)
        z_threshold = analytic_threshold_collocation(
            int(collocation["threshold_points"]), domain,
            seed=training_config.seed + 2,
            band_width=float(collocation.get("threshold_band_width", 0.02)))
        z_low = sobol_plow_boundary_nontrivial(
            int(collocation["low_p_points"]), domain, training_config.seed + 3,
            angular_sampling=training_config.angular_sampling)
        z_bc = sobol_pmax_boundary_nontrivial(
            int(collocation["boundary_points"]), domain, training_config.seed + 4,
            angular_sampling=training_config.angular_sampling)
        start = time.perf_counter()
        active = config.get("active", {})
        active_records = []
        active_cycles = int(active.get("cycles", 1))
        if active.get("enabled", False):
            if jax.process_count() > 1:
                raise RuntimeError("active FV acquisition requires one JAX process")
            def acquire_data(z_selected):
                case_norm = np.unique(np.asarray(z_selected)[:, 2:], axis=0)
                selected_cases = denormalize_parameter_cases(
                    case_norm, config["parameter_domain"])
                results = generate_cpu_cases(
                    selected_cases, p_max=domain.p_max,
                    Np=int(active.get("fv_Np", 256)), Nxi=int(active.get("fv_Nxi", 64)),
                    B_T=domain.B_T, n_jobs=int(active.get("n_jobs", -1)))
                keep = np.asarray([
                    not result["trivial_zero"] and result.get("valid", True)
                    for result in results])
                if not np.any(keep):
                    return np.empty((0, 8)), np.empty(0)
                kept_cases = selected_cases[keep]
                kept_results = [r for r, k in zip(results, keep) if k]
                kept_results = coarsen_fv_cases(
                    kept_results, p_stride=int(active.get("fv_p_stride", 1)),
                    xi_stride=int(active.get("fv_xi_stride", 1)))
                grouped = exact_deeponet_inputs(
                    kept_cases, kept_results, config["parameter_domain"], domain)
                return grouped_to_pointwise(grouped)
            params, soap_history, active_records = train_physics_active(
                z_all[train_idx], y_all[train_idx], z_pde, z_bc, domain=domain,
                z_pde_threshold=z_threshold, z_low=z_low,
                config=training_config, active_config=active,
                acquire_data=acquire_data, checkpoint_callback=checkpoint,
                distributed=False, initial_params=initial_params)
            active_cycles = max(1, active_cycles)
        else:
            params, soap_history, _ = train_physics_informed(
                z_all[train_idx], y_all[train_idx], z_pde, z_bc, domain=domain,
                z_pde_threshold=z_threshold, z_low=z_low,
                config=training_config, checkpoint_callback=checkpoint,
                distributed=(jax.process_count() > 1), initial_params=initial_params)
            active_cycles = 1
        ssb_history = []
        if config.get("ssbroyden", {}).get("enabled", False):
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices("ssbroyden_start")
                if jax.process_index() == 0:
                    params, ssb_history = train_ssbroyden(
                        params, z_all[train_idx], y_all[train_idx], z_pde, z_bc,
                        domain=domain, z_pde_threshold=z_threshold, z_low=z_low,
                        config=training_config, checkpoint_callback=checkpoint,
                        step_offset=training_config.steps * active_cycles)
                params = multihost_utils.broadcast_one_to_all(
                    params, is_source=(jax.process_index() == 0))
                multihost_utils.sync_global_devices("ssbroyden_complete")
            elif jax.process_index() == 0:
                params, ssb_history = train_ssbroyden(
                    params, z_all[train_idx], y_all[train_idx], z_pde, z_bc,
                    domain=domain, z_pde_threshold=z_threshold, z_low=z_low,
                    config=training_config, checkpoint_callback=checkpoint,
                    step_offset=training_config.steps * active_cycles)
        if jax.process_index() == 0:
            prediction_train = np.asarray(jax.device_get(
                predict(params, jnp.asarray(z_all[train_idx]))))
            prediction_test = np.asarray(jax.device_get(
                predict(params, jnp.asarray(z_all[test_idx]))))
            metadata = {"architecture": {
                "input_dim": 8, "model_type": training_config.model_type,
                "width": training_config.width, "depth": training_config.depth,
                "latent_width": training_config.latent_width,
                "branch_width": training_config.branch_width,
                "branch_depth": training_config.branch_depth,
                "trunk_width": training_config.trunk_width,
                "trunk_depth": training_config.trunk_depth,
            }, "domain": config["domain"], "normalization": {
                "p_floor": domain.p_min, "p_max": domain.p_max,
            }, "training_config": {"model": config.get("model", {}),
                                    "loss": config.get("loss", {}),
                                    "optimizer": config.get("optimizer", {}),
                                    "data": config.get("data", {})},
            "parameter_domain": config["parameter_domain"],
            "dataset": str(dataset_path),
            "initial_model": initial_model}
            summary = {"dataset": str(dataset_path),
                       "train_cases": int(len(train_cases)),
                       "test_cases": int(len(test_cases)),
                       "train_mse": float(np.mean((prediction_train - y_all[train_idx]) ** 2)),
                       "test_mse": float(np.mean((prediction_test - y_all[test_idx]) ** 2)),
                       "soap_loss_history": np.asarray(soap_history).tolist(),
                       "ssbroyden_history": ssb_history,
                       "active_history": active_records,
                       "elapsed_seconds": time.perf_counter() - start,
                       "checkpoint_every": training_config.checkpoint_every,
                       "global_device_count": int(jax.device_count()),
                       "process_count": int(jax.process_count())}
            loss_history = {"soap": {"steps": list(range(1, len(soap_history) + 1)),
                                      "columns": ["total", "data", "PDE", "threshold PDE", "low-p", "p_max BC"],
                                      "values": np.asarray(soap_history).tolist()},
                            "ssbroyden": ssb_history, "active": active_records}
            artifacts = write_training_outputs(
                output_dir, params, save_model, run_config=config,
                metadata=metadata, summary=summary, loss_history=loss_history)
            finish_run(manifest, status="complete", artifacts=artifacts,
                       metrics={"train_mse": summary["train_mse"],
                                "test_mse": summary["test_mse"]})
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("training_artifacts_complete")
    except Exception as exc:
        try:
            atomic_write_json(launch_dir / f"failure.rank{rank}.json", {
                "rank": rank, "error": repr(exc),
            })
        except OSError:
            pass
        if manifest is not None:
            finish_run(manifest, status="failed", error=repr(exc))
        raise
