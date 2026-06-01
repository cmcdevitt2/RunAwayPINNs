"""Shared JAX PINN utilities for the runaway-probability-function demo.

The learned scalar is P(p, xi; |E|, Zeff, alpha), where p is normalized
relativistic momentum and xi is pitch-angle cosine.  The low-energy boundary
P(p_min, xi) = 0 is hard-enforced by an output transform.  The high-energy
runaway boundary P(p_max, xi) = 1 is imposed as a soft boundary loss on
pitch angles for which U_p > 0, optionally gated by a reduced O-X threshold
criterion E > E_OX(Z_eff, alpha).
"""

from __future__ import annotations
import os

# Must be set before importing JAX.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from functools import partial

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import optax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicsConfig:
    mec_sq_ev: float = 511.0e3
    energy_min_ev: float = 1.0e4
    energy_max_ev: float = 5.0e6
    xi_min: float = -1.0
    xi_max: float = 1.0

    evert_min: float = 1.0
    evert_max: float = 20.0
    zeff_min: float = 1.0
    zeff_max: float = 5.0
    alpha_min: float = 0.0
    alpha_max: float = 0.1

    # Output transform. Choices: "pnorm_sigmoid", "tanh_square".
    output_transform: str = "tanh_square"

    # Residual normalization. Use "none" to match the reference fixed-parameter
    # JAX script. Use "cf_e" only for the older preconditioned RPF_test form.
    residual_scaling: str = "cf_e"

    @property
    def gamma_min(self) -> float:
        return 1.0 + self.energy_min_ev / self.mec_sq_ev

    @property
    def gamma_max(self) -> float:
        return 1.0 + self.energy_max_ev / self.mec_sq_ev

    @property
    def p_min(self) -> float:
        return math.sqrt(self.gamma_min**2 - 1.0)

    @property
    def p_max(self) -> float:
        return math.sqrt(self.gamma_max**2 - 1.0)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1234
    width: int = 64
    depth: int = 4

    n_pde: int = 2**14
    n_bc_high: int = 2**12
    n_test: int = 2**14

    adam_steps: int = 5000
    adam_lr: float = 1.0e-3
    resample_every: int = 500

    # Independent Adam logging controls.
    # Full training loss is recorded every Adam iteration from returned scan losses.
    # test_loss_every controls expensive test-loss evaluations.
    # print_every controls terminal progress printing only.
    test_loss_every: int = 250
    print_every: int = 250

    high_bc_weight: float = 10.0

    # Optional reduced O-X threshold gate for the high-energy BC. If enabled,
    # the high-energy BC contribution is multiplied by a stopped-gradient gate
    # based on E_abs - E_OX(Zeff, alpha). The network input remains 5D.
    use_eox_bc_gate: bool = False
    eox_gate_mode: str = "smooth"  # "smooth" or "hard"
    eox_gate_width: float = 0.05
    eox_e_scan_min: float = 1.0
    eox_e_scan_max: float = 10.0
    eox_n_e_scan: int = 256
    eox_n_e_refine: int = 32
    eox_n_p_scan: int = 2**12
    eox_gamma_scan_min: float = 1.02
    eox_p_scan_max: float = 300.0
    eox_xi_eps: float = 1.0e-10
    eox_min_roots_for_vortex: int = 2
    eox_batch_size: int = 2**13

    # Optional residual-adaptive resampling.
    adaptive_resampling: bool = False
    candidate_pool_mult: int = 4
    adaptive_resampling_k: float = 1.0
    adaptive_resampling_c: float = 1.0
    candidate_eval_chunk: int = 2**14

    # Optional Optimistix SSBroyden refinement. Set ssb_blocks = 0 to skip.
    ssb_blocks: int = 0
    ssb_block_iters: int = 500
    ssb_resample_every_blocks: int = 1
    ssb_rtol: float = 1.0e-14
    ssb_atol: float = 1.0e-14
    ssb_search: str = "wolfe"  # "wolfe", "zoom", or "trust_region"
    ssb_verbose: bool = False

    # Outputs.
    outdir: str = "models"
    copy_models_for_web: bool = True
    export_onnx: bool = True


# -----------------------------------------------------------------------------
# Normalization / parameter utilities
# -----------------------------------------------------------------------------

def denorm01(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    return lo + x * (hi - lo)


def norm01(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    return (x - lo) / (hi - lo)


def split_inputs(z: jnp.ndarray, phys: PhysicsConfig):
    """Return physical p, xi, |E|, Zeff, alpha from normalized inputs.

    z has columns [p_norm, xi_norm, evert_norm, zeff_norm, alpha_norm].
    """
    p_norm, xi_norm, e_norm, z_norm, a_norm = z
    p = denorm01(p_norm, phys.p_min, phys.p_max)
    xi = denorm01(xi_norm, phys.xi_min, phys.xi_max)
    evert = denorm01(e_norm, phys.evert_min, phys.evert_max)
    zeff = denorm01(z_norm, phys.zeff_min, phys.zeff_max)
    alpha = denorm01(a_norm, phys.alpha_min, phys.alpha_max)
    return p, xi, evert, zeff, alpha


def gamma_from_p(p: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(1.0 + p * p)


def high_energy_xi_crit(evert: jnp.ndarray, alpha: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    """Critical xi at p=p_max where U_p = 0.

    U_p = -xi |E| - C_F - alpha gamma p (1 - xi^2)
        = A xi^2 - E xi - (C_F + A),  A = alpha gamma p.

    The runaway/inflow boundary satisfying U_p > 0 is xi in [-1, xi_crit].
    """
    pmax = jnp.asarray(phys.p_max, dtype=jnp.result_type(evert, alpha))
    gam = gamma_from_p(pmax)
    c_f = gam * gam / (pmax * pmax)
    a = alpha * gam * pmax
    eps = jnp.asarray(1.0e-14, dtype=a.dtype)
    xi_alpha0 = -c_f / evert
    disc = evert * evert + 4.0 * a * (c_f + a)
    xi_alpha_pos = (evert - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * jnp.maximum(a, eps))
    xi_crit = jnp.where(a > eps, xi_alpha_pos, xi_alpha0)
    return jnp.clip(xi_crit, phys.xi_min, phys.xi_max)


# -----------------------------------------------------------------------------
# Reduced O-X threshold model used for training-time high-BC gating
# -----------------------------------------------------------------------------

def eox_c_f_simple(p: jnp.ndarray) -> jnp.ndarray:
    gam = gamma_from_p(p)
    return gam * gam / (p * p)


def eox_nu_d_simple(p: jnp.ndarray, zeff: jnp.ndarray) -> jnp.ndarray:
    gam = gamma_from_p(p)
    return (zeff + 1.0) * gam / (p * p * p)


def eox_pitch_width_model(p: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * (1.0 + xi) / p


def eox_xi_from_gamma_p_zero(p: jnp.ndarray, evert: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    gam = gamma_from_p(p)
    c_f = eox_c_f_simple(p)
    a = alpha * gam * p

    xi_alpha0 = -c_f / evert
    disc = evert * evert + 4.0 * a * (c_f + a)
    xi_alpha_pos = (evert - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * jnp.maximum(a, 1.0e-300))
    return jnp.where(a > 1.0e-14, xi_alpha_pos, xi_alpha0)


def eox_gamma_xi_reduced_condition(
    p: jnp.ndarray,
    xi: jnp.ndarray,
    evert: jnp.ndarray,
    zeff: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    gam = gamma_from_p(p)
    nud = eox_nu_d_simple(p, zeff)
    one_minus_xi2 = jnp.maximum(1.0 - xi * xi, 0.0)
    sqrt_one_minus_xi2 = jnp.sqrt(one_minus_xi2)
    delta_xi = jnp.maximum(eox_pitch_width_model(p, xi), 1.0e-300)
    pitch_focusing = sqrt_one_minus_xi2 * (evert / p - alpha * p * xi / gam)
    pitch_scattering = 0.5 * nud / delta_xi
    return pitch_focusing - pitch_scattering


def eox_count_sign_change_roots(f: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    f0 = f[:-1]
    f1 = f[1:]
    valid_pair = valid[:-1] & valid[1:]
    finite_pair = jnp.isfinite(f0) & jnp.isfinite(f1)
    sign_change = f0 * f1 <= 0.0
    return jnp.sum(valid_pair & finite_pair & sign_change)


def eox_vortex_exists(evert: jnp.ndarray, zeff: jnp.ndarray, alpha: jnp.ndarray, phys: PhysicsConfig, run: TrainConfig) -> jnp.ndarray:
    gamma_scan_min = jnp.asarray(run.eox_gamma_scan_min, dtype=jnp.float64)
    p_scan_min = jnp.sqrt(gamma_scan_min * gamma_scan_min - 1.0)
    p_grid = jnp.geomspace(p_scan_min, run.eox_p_scan_max, run.eox_n_p_scan, dtype=jnp.float64)

    xi_gp0 = eox_xi_from_gamma_p_zero(p_grid, evert, alpha)
    valid_xi = (xi_gp0 > phys.xi_min + run.eox_xi_eps) & (xi_gp0 < phys.xi_max - run.eox_xi_eps)
    f = eox_gamma_xi_reduced_condition(p_grid, xi_gp0, evert, zeff, alpha)
    n_roots = eox_count_sign_change_roots(f, valid_xi)
    return n_roots >= run.eox_min_roots_for_vortex


def eox_single(zeff: jnp.ndarray, alpha: jnp.ndarray, phys: PhysicsConfig, run: TrainConfig) -> jnp.ndarray:
    e_grid = jnp.linspace(run.eox_e_scan_min, run.eox_e_scan_max, run.eox_n_e_scan, dtype=jnp.float64)
    has = jax.vmap(lambda ee: eox_vortex_exists(ee, zeff, alpha, phys, run))(e_grid)
    has_any = jnp.any(has)
    first_idx = jnp.argmax(has)
    lo_idx = jnp.maximum(first_idx - 1, 0)
    e_lo = e_grid[lo_idx]
    e_hi = e_grid[first_idx]

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        exists = eox_vortex_exists(mid, zeff, alpha, phys, run)
        hi_new = jnp.where(exists, mid, hi)
        lo_new = jnp.where(exists, lo, mid)
        return lo_new, hi_new

    _, e_refined = jax.lax.fori_loop(0, run.eox_n_e_refine, body, (e_lo, e_hi))
    return jnp.where(has_any, e_refined, jnp.nan)


@partial(jax.jit, static_argnames=("phys", "run"))
def eox_batch(zeff: jnp.ndarray, alpha: jnp.ndarray, phys: PhysicsConfig, run: TrainConfig) -> jnp.ndarray:
    return jax.vmap(lambda zz, aa: eox_single(zz, aa, phys, run))(zeff, alpha)


def high_bc_eox_gate(evert: jnp.ndarray, zeff: jnp.ndarray, alpha: jnp.ndarray, phys: PhysicsConfig, run: TrainConfig) -> jnp.ndarray:
    if not run.use_eox_bc_gate:
        return jnp.ones_like(evert)

    n = evert.shape[0]
    bs = run.eox_batch_size

    gates = []
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        e_b = evert[i0:i1]
        z_b = zeff[i0:i1]
        a_b = alpha[i0:i1]

        e_ox_b = eox_batch(z_b, a_b, phys, run)
        finite = jnp.isfinite(e_ox_b)

        if run.eox_gate_mode == "hard":
            gate_b = (finite & (e_b > e_ox_b)).astype(jnp.float64)
        elif run.eox_gate_mode == "smooth":
            gate_b = jax.nn.sigmoid((e_b - e_ox_b) / run.eox_gate_width)
            gate_b = jnp.where(finite, gate_b, 0.0)
        else:
            raise ValueError("eox_gate_mode must be 'smooth' or 'hard'")

        gates.append(gate_b)

    gate = jnp.concatenate(gates, axis=0)
    return jax.lax.stop_gradient(gate)


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

def init_mlp(key: jax.Array, in_dim: int, width: int, depth: int, out_dim: int = 1):
    sizes = [in_dim] + [width] * depth + [out_dim]
    keys = random.split(key, len(sizes) - 1)
    params = []
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        w = random.normal(k, (din, dout), dtype=jnp.float64) * jnp.sqrt(2.0 / (din + dout))
        b = jnp.zeros((dout,), dtype=jnp.float64)
        params.append({"w": w, "b": b})
    return params


def mlp_apply(params, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]


def raw_apply_single(params, z: jnp.ndarray) -> jnp.ndarray:
    return mlp_apply(params, z[None, :])[0, 0]


def output_transform(p_norm: jnp.ndarray, raw: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    """Hard low-energy boundary transform P(p_norm=0) = 0."""
    if phys.output_transform == "pnorm_sigmoid":
        return p_norm * jax.nn.sigmoid(raw)
    if phys.output_transform == "tanh_square":
        return jnp.tanh((p_norm * raw) ** 2)
    raise ValueError(f"unknown output_transform={phys.output_transform!r}")


def prob_apply_single(params, z: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    raw = raw_apply_single(params, z)
    return output_transform(z[0], raw, phys)


def prob_apply(params, z: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    return jax.vmap(lambda zz: prob_apply_single(params, zz, phys))(z)[:, None]


# -----------------------------------------------------------------------------
# PDE residual
# -----------------------------------------------------------------------------

def residual_single(params, z: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    """Steady 2D adjoint RPF residual for one normalized input.

    The default form matches the reference fixed-parameter JAX script:

        -U_p P_p
        + (1-xi^2) (|E|/p - alpha xi/gamma) P_xi
        - (C_B/(2 p^2)) d_xi[(1-xi^2) P_xi] = 0.

    Here ``c_b_half = 0.5 * (Zeff + 1) * gamma / p`` is the complete-screening
    pitch-angle diffusion coefficient including the 1/2 factor, so the final
    diffusion term is ``-(c_b_half/p^2) * dflux_dxi``.
    """
    p, xi, evert, zeff, alpha = split_inputs(z, phys)
    dp_dpbar = phys.p_max - phys.p_min
    dxi_dxibar = phys.xi_max - phys.xi_min

    def f(zz):
        return prob_apply_single(params, zz, phys)

    grad_z = jax.grad(f)(z)
    xi_bar = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=z.dtype)
    _, d2_xibar2 = jax.jvp(lambda zz: jax.grad(f)(zz)[1], (z,), (xi_bar,))

    p_p = grad_z[0] / dp_dpbar
    p_xi = grad_z[1] / dxi_dxibar
    p_xixi = d2_xibar2 / (dxi_dxibar**2)

    gam = gamma_from_p(p)
    c_f = gam * gam / (p * p)
    c_b_half = ((zeff + 1.0) / 2.0) * (gam / p)
    u_p = -xi * evert - c_f - alpha * gam * p * (1.0 - xi * xi)

    momentum_advection = -u_p * p_p
    pitch_advection = (1.0 - xi * xi) * (evert / p - alpha * xi / gam) * p_xi
    dflux_dxi = (1.0 - xi * xi) * p_xixi - 2.0 * xi * p_xi
    pitch_diffusion = -(c_b_half / (p * p)) * dflux_dxi
    residual = momentum_advection + pitch_advection + pitch_diffusion

    if phys.residual_scaling == "none":
        return residual
    if phys.residual_scaling == "cf_e":
        return residual# / c_f#jnp.sqrt(evert)#(c_f * evert)
    raise ValueError(f"unknown residual_scaling={phys.residual_scaling!r}")


def residual_apply(params, z: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    return jax.vmap(lambda zz: residual_single(params, zz, phys))(z)[:, None]


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def sample_params(key: jax.Array, n: int, fixed: tuple[float, float, float] | None, phys: PhysicsConfig):
    if fixed is not None:
        evert, zeff, alpha = fixed
        en = jnp.full((n, 1), norm01(evert, phys.evert_min, phys.evert_max), dtype=jnp.float64)
        zn = jnp.full((n, 1), norm01(zeff, phys.zeff_min, phys.zeff_max), dtype=jnp.float64)
        an = jnp.full((n, 1), norm01(alpha, phys.alpha_min, phys.alpha_max), dtype=jnp.float64)
        return en, zn, an
    u = random.uniform(key, (n, 3), dtype=jnp.float64)
    return u[:, 0:1], u[:, 1:2], u[:, 2:3]


def sample_pde_points(key: jax.Array, n: int, phys: PhysicsConfig, fixed_params=None):
    k1, k2 = random.split(key)
    px = random.uniform(k1, (n, 2), dtype=jnp.float64)
    en, zn, an = sample_params(k2, n, fixed_params, phys)
    return jnp.concatenate([px[:, 0:1], px[:, 1:2], en, zn, an], axis=1)


def sample_high_boundary(key: jax.Array, n: int, phys: PhysicsConfig, run: TrainConfig, fixed_params=None):
    """Sample p_norm=1 and xi in [-1, xi_crit) for the U_p>0 boundary.

    The returned array has six columns:

        [p_norm, xi_norm, e_norm, z_norm, alpha_norm, bc_gate]

    The first five columns are the network input. The sixth column is a
    stopped-gradient high-energy BC gate. If ``run.use_eox_bc_gate`` is False,
    the gate is one. If it is True, the gate decreases as E_abs crosses below
    the reduced O-X threshold E_OX(Zeff, alpha).
    """
    k_s, k_p = random.split(key)
    en, zn, an = sample_params(k_p, n, fixed_params, phys)
    evert = denorm01(en[:, 0], phys.evert_min, phys.evert_max)
    zeff = denorm01(zn[:, 0], phys.zeff_min, phys.zeff_max)
    alpha = denorm01(an[:, 0], phys.alpha_min, phys.alpha_max)

    xi_crit = high_energy_xi_crit(evert, alpha, phys)
    eps_xi = jnp.asarray(1.0e-10, dtype=jnp.float64)
    xi_hi = jnp.clip(xi_crit - eps_xi, phys.xi_min, phys.xi_max)
    s = random.uniform(k_s, (n,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    xi = phys.xi_min + s * (xi_hi - phys.xi_min)
    xi_norm = norm01(xi, phys.xi_min, phys.xi_max)
    p_norm = jnp.ones((n, 1), dtype=jnp.float64)

    gate = high_bc_eox_gate(evert, zeff, alpha, phys, run)
    z_high = jnp.concatenate([p_norm, xi_norm[:, None], en, zn, an], axis=1)
    return jnp.concatenate([z_high, gate[:, None]], axis=1)


def high_boundary_mask(z: jnp.ndarray, phys: PhysicsConfig) -> jnp.ndarray:
    """Return 1 where p=p_max boundary points satisfy U_p > 0, else 0."""
    p_norm = z[:, 0]
    xi_norm = z[:, 1]
    e_norm = z[:, 2]
    a_norm = z[:, 4]
    p = denorm01(p_norm, phys.p_min, phys.p_max)
    xi = denorm01(xi_norm, phys.xi_min, phys.xi_max)
    evert = denorm01(e_norm, phys.evert_min, phys.evert_max)
    alpha = denorm01(a_norm, phys.alpha_min, phys.alpha_max)
    gam = gamma_from_p(p)
    c_f = gam * gam / (p * p)
    u_p = -xi * evert - c_f - alpha * gam * p * (1.0 - xi * xi)
    return (u_p > 0.0).astype(jnp.float64)


def residual_adaptive_resample(params, key, n: int, phys: PhysicsConfig, fixed_params, run: TrainConfig):
    pool_n = int(run.candidate_pool_mult * n)
    k_pool, k_choose = random.split(key)
    pool = sample_pde_points(k_pool, pool_n, phys, fixed_params)
    r = jnp.abs(residual_apply(params, pool, phys)[:, 0])
    score = jnp.power(jnp.maximum(r, 0.0), run.adaptive_resampling_k)
    score = score / jnp.maximum(jnp.mean(score), jnp.finfo(score.dtype).tiny) + run.adaptive_resampling_c
    prob = score / jnp.sum(score)
    idx = random.choice(k_choose, pool_n, shape=(n,), replace=False, p=prob)
    return pool[idx]


def sample_or_adaptive(params, key, n: int, phys: PhysicsConfig, fixed_params, run: TrainConfig):
    if run.adaptive_resampling and params is not None:
        return residual_adaptive_resample(params, key, n, phys, fixed_params, run)
    return sample_pde_points(key, n, phys, fixed_params)


# -----------------------------------------------------------------------------
# Losses and training
# -----------------------------------------------------------------------------

def loss_terms(params, x_pde, x_high, phys: PhysicsConfig, run: TrainConfig):
    r = residual_apply(params, x_pde, phys)[:, 0]
    loss_pde = jnp.mean(r * r)

    z_high = x_high[:, :5]
    bc_gate = x_high[:, 5] if x_high.shape[1] > 5 else jnp.ones((x_high.shape[0],), dtype=x_high.dtype)
    p_high = prob_apply(params, z_high, phys)[:, 0]
    mask = high_boundary_mask(z_high, phys)
    weight = mask * bc_gate
    # Average over the full high-boundary batch. This lets the high-BC loss
    # naturally shrink when the U_p > 0 interval is small and/or the E_OX
    # gate is below one. Do not renormalize by active points.
    loss_high = jnp.mean(weight * (p_high - 1.0) ** 2)
    total = loss_pde + run.high_bc_weight * loss_high
    return total, {"total": total, "pde": loss_pde, "high_bc": loss_high}


def train_adam(params, key, phys: PhysicsConfig, run: TrainConfig, fixed_params=None, history_prefix="adam"):
    """Train with Optax Adam using blocked lax.scan.

    This follows the fixed-parameter script bookkeeping:
      - the actual Adam training loss is returned for every optimizer update
        from the compiled scan and recorded after each block;
      - the test loss is evaluated only every ``run.test_loss_every`` steps;
      - progress printing is controlled independently by ``run.print_every``.

    The training loss is the scalar objective used to form the Adam gradient on
    the current collocation batch. If resampling is enabled, it naturally
    switches to the newly sampled batch after each resampling event.
    """
    optimizer = optax.adam(run.adam_lr)
    opt_state = optimizer.init(params)
    records_by_iter = {}

    key, k_pde, k_high, k_test, k_test_high = random.split(key, 5)
    x_pde = sample_or_adaptive(params, k_pde, run.n_pde, phys, fixed_params, run)
    x_high = sample_high_boundary(k_high, run.n_bc_high, phys, run, fixed_params)
    x_test = sample_pde_points(k_test, run.n_test, phys, fixed_params)
    x_test_high = sample_high_boundary(k_test_high, run.n_bc_high, phys, run, fixed_params)

    def loss_only(pp, xpde, xhi):
        return loss_terms(pp, xpde, xhi, phys, run)[0]

    @jax.jit
    def adam_step_jit(pp, state, xpde, xhi):
        loss_val, grads = jax.value_and_grad(loss_only)(pp, xpde, xhi)
        updates, state = optimizer.update(grads, state, pp)
        pp = optax.apply_updates(pp, updates)
        return pp, state, loss_val

    @partial(jax.jit, static_argnames=("n_steps",))
    def adam_steps_jit(pp, state, xpde, xhi, n_steps: int):
        def body(carry, _):
            pp_i, state_i = carry
            pp_i, state_i, loss_i = adam_step_jit(pp_i, state_i, xpde, xhi)
            return (pp_i, state_i), loss_i

        (pp, state), losses = jax.lax.scan(
            body,
            (pp, state),
            xs=None,
            length=n_steps,
        )
        return pp, state, losses

    @jax.jit
    def eval_total_jit(pp, xpde, xhi):
        return loss_terms(pp, xpde, xhi, phys, run)[0]

    def get_record(iter_i: int):
        if iter_i not in records_by_iter:
            records_by_iter[iter_i] = {
                "phase": history_prefix,
                "iter": int(iter_i),
                "step": int(iter_i),
                "train_total": float("nan"),
                "test_total": float("nan"),
            }
        return records_by_iter[iter_i]

    def next_boundary(done: int, every: int) -> int:
        if every <= 0:
            return run.adam_steps
        if done < 1:
            return 1
        return min(((done // every) + 1) * every, run.adam_steps)

    step = 0
    while step < run.adam_steps:
        if step > 0 and run.resample_every > 0 and step % run.resample_every == 0:
            key, kp, kh = random.split(key, 3)
            x_pde = sample_or_adaptive(params, kp, run.n_pde, phys, fixed_params, run)
            x_high = sample_high_boundary(kh, run.n_bc_high, phys, run, fixed_params)

        # Block boundaries are only events that require host-side work.
        # Full per-step training losses are returned by the compiled scan, so
        # loss-history bookkeeping does not force one-step dispatches.
        target = min(
            next_boundary(step, run.test_loss_every),
            next_boundary(step, run.print_every),
            next_boundary(step, run.resample_every) if run.resample_every > 0 else run.adam_steps,
            run.adam_steps,
        )
        n_steps = max(target - step, 1)
        start_step = step

        params, opt_state, losses = adam_steps_jit(params, opt_state, x_pde, x_high, int(n_steps))
        jax.block_until_ready(params)
        losses_np = np.asarray(jax.device_get(losses))
        step = target

        # Always keep the full per-iteration Adam training-loss history.
        for j, loss_j in enumerate(losses_np):
            iter_j = start_step + j + 1
            get_record(iter_j)["train_total"] = float(loss_j)

        do_test = (
            step == 1
            or step == run.adam_steps
            or (run.test_loss_every > 0 and step % run.test_loss_every == 0)
        )
        if do_test:
            test_total = eval_total_jit(params, x_test, x_test_high)
            get_record(step)["test_total"] = float(jax.device_get(test_total))

        do_print = (
            step == 1
            or step == run.adam_steps
            or (run.print_every > 0 and step % run.print_every == 0)
        )
        if do_print:
            rec = records_by_iter.get(step, {})
            train_val = rec.get("train_total", float("nan"))
            test_val = rec.get("test_total", float("nan"))
            if np.isfinite(test_val):
                print(f"adam step {step:7d} train={train_val:.3e} test={test_val:.3e}")
            else:
                print(f"adam step {step:7d} train={train_val:.3e}")

    history = [records_by_iter[k] for k in sorted(records_by_iter)]
    return params, history


# Optional SSBroyden support ---------------------------------------------------

def _make_ssb_solver(run: TrainConfig):
    try:
        import optimistix as optx
        import equinox as eqx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Optimistix/equinox unavailable; set ssb_blocks=0 or install the SSB-enabled Optimistix fork") from exc

    class SSBroyden(optx.AbstractSSBroyden):
        rtol: float
        atol: float
        norm: Callable = optx.max_norm
        use_inverse: bool = True
        search: optx.AbstractSearch = eqx.field(default_factory=optx.BacktrackingStrongWolfe)
        descent: optx.AbstractDescent = eqx.field(default_factory=optx.NewtonDescent)
        verbose: frozenset[str] = frozenset()

    # Match the lid-driven-cavity implementation that already worked in this
    # demo workflow: keep the user-facing "wolfe" option mapped to
    # BacktrackingStrongWolfe. Zoom and trust-region remain available for tests.
    if run.ssb_search == "wolfe":
        search = optx.BacktrackingStrongWolfe()
    elif run.ssb_search == "zoom":
        search = optx.Zoom()
    elif run.ssb_search == "trust_region":
        search = optx.LinearTrustRegion()
    else:
        raise ValueError("ssb_search must be 'wolfe', 'zoom', or 'trust_region'")

    solver = SSBroyden(
        rtol=run.ssb_rtol,
        atol=run.ssb_atol,
        search=search,
        descent=optx.NewtonDescent(),
        verbose=frozenset(["loss"]) if run.ssb_verbose else frozenset(),
    )
    # Match the CrunchOptimizer/PINNs usage pattern: wrap the SSBroyden solver
    # with BestSoFarMinimiser and read the updated parameter pytree from
    # sol.value.
    solver = optx.BestSoFarMinimiser(solver)
    return solver, optx


def train_ssbroyden(params, key, phys: PhysicsConfig, run: TrainConfig, fixed_params=None):
    """Run blockwise Optimistix SSBroyden on a single flat weight vector.

    This mirrors the lid-driven-cavity scripts in this repository, which are the
    working Optimistix SSBroyden reference for this demo workflow:

        params pytree -> ravel_pytree(params) -> optx.minimise(flat_loss, ...)
        -> sol.value -> unflatten(sol.value)

    The Optimistix solve itself is JIT-compiled as one block, so the inner
    quasi-Newton iterations and line search run as compiled XLA work for the
    current batch shapes. Python only handles resampling and logging between
    blocks.
    """
    if run.ssb_blocks <= 0:
        return params, []

    solver, optx = _make_ssb_solver(run)
    weights, unflatten = ravel_pytree(params)
    n_params = int(weights.shape[0])
    print("SSBroyden backend: optimistix")
    print(f"SSBroyden parameter count: {n_params}")
    print(f"Dense inverse-H memory estimate: {(n_params * n_params * 8) / 1.0e9:.3f} GB")

    history = []
    ssb_iter = 0

    @jax.jit
    def optimistix_loss(weights_flat, batch):
        # Optimistix calls scalar objectives as fn(y, args), i.e. it passes the
        # whole args object as one positional argument. Keep xpde/xhi bundled.
        xpde, xhi = batch
        pp = unflatten(weights_flat)
        return loss_terms(pp, xpde, xhi, phys, run)[0]

    @jax.jit
    def ssb_step_jit(weights_in, xpde, xhi):
        batch = (xpde, xhi)
        loss_before = optimistix_loss(weights_in, batch)
        sol = optx.minimise(
            optimistix_loss,
            solver,
            weights_in,
            args=batch,
            max_steps=run.ssb_block_iters,
            throw=False,
        )
        candidate_weights = sol.value
        loss_after = optimistix_loss(candidate_weights, batch)
        delta = jnp.linalg.norm(candidate_weights - weights_in)
        finite = jnp.all(jnp.isfinite(candidate_weights)) & jnp.isfinite(loss_after)
        num_steps = sol.stats["num_steps"]
        return candidate_weights, loss_before, loss_after, delta, finite, num_steps

    @jax.jit
    def eval_terms_jit(weights_flat, xpde, xhi):
        pp = unflatten(weights_flat)
        total, terms = loss_terms(pp, xpde, xhi, phys, run)
        return total, terms["pde"], terms["high_bc"]

    x_pde = None
    x_high = None
    x_test = None
    x_test_high = None

    for block in range(1, run.ssb_blocks + 1):
        if x_pde is None or (run.ssb_resample_every_blocks > 0 and (block - 1) % run.ssb_resample_every_blocks == 0):
            key, kp, kh, kt, kth = random.split(key, 5)
            current_params = unflatten(weights)
            x_pde = sample_or_adaptive(current_params, kp, run.n_pde, phys, fixed_params, run)
            x_high = sample_high_boundary(kh, run.n_bc_high, phys, run, fixed_params)
            x_test = sample_pde_points(kt, run.n_test, phys, fixed_params)
            x_test_high = sample_high_boundary(kth, run.n_bc_high, phys, run, fixed_params)

        candidate_weights, block_loss_before, block_loss_after, block_delta, block_finite, block_num_steps = ssb_step_jit(weights, x_pde, x_high)
        jax.block_until_ready(candidate_weights)
        weights = candidate_weights

        train_total, train_pde, train_high_bc = eval_terms_jit(weights, x_pde, x_high)
        test_total, test_pde, test_high_bc = eval_terms_jit(weights, x_test, x_test_high)
        (
            train_total,
            train_pde,
            train_high_bc,
            test_total,
            test_pde,
            test_high_bc,
            block_loss_before,
            block_loss_after,
            block_delta,
            block_finite,
            block_num_steps,
        ) = jax.device_get(
            (
                train_total,
                train_pde,
                train_high_bc,
                test_total,
                test_pde,
                test_high_bc,
                block_loss_before,
                block_loss_after,
                block_delta,
                block_finite,
                block_num_steps,
            )
        )
        ssb_iter += int(block_num_steps)

        rec = {
            "phase": "ssbroyden",
            "iter": int(run.adam_steps + ssb_iter),
            "step": int(block),
            "train_total": float(train_total),
            "train_pde": float(train_pde),
            "train_high_bc": float(train_high_bc),
            "test_total": float(test_total),
            "test_pde": float(test_pde),
            "test_high_bc": float(test_high_bc),
            "block_loss_before": float(block_loss_before),
            "block_loss_after": float(block_loss_after),
            "block_delta_weights": float(block_delta),
            "block_finite": bool(block_finite),
            "block_num_steps": int(block_num_steps),
        }
        history.append(rec)
        print(
            f"ssb block {block:4d}/{run.ssb_blocks} "
            f"block={rec['block_loss_before']:.3e}->{rec['block_loss_after']:.3e} "
            f"steps={rec['block_num_steps']} finite={rec['block_finite']} "
            f"|dw|={rec['block_delta_weights']:.3e} "
            f"train={rec['train_total']:.3e} test={rec['test_total']:.3e} "
            f"pde={rec['train_pde']:.3e} bc={rec['train_high_bc']:.3e}"
        )

    return unflatten(weights), history


# -----------------------------------------------------------------------------
# Plotting and persistence
# -----------------------------------------------------------------------------

def save_history(history, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "loss_history.json").write_text(json.dumps(history, indent=2))
    import csv
    keys = [
        "phase",
        "iter",
        "step",
        "train_total",
        "train_pde",
        "train_high_bc",
        "test_total",
        "test_pde",
        "test_high_bc",
        "block_loss_before",
        "block_loss_after",
        "block_delta_weights",
        "block_finite",
        "block_num_steps",
    ]
    with (outdir / "loss_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})


def plot_loss_history(history, outdir: Path):
    if plt is None or not history:
        return

    x = np.asarray([h.get("iter", h.get("step", i)) for i, h in enumerate(history)], dtype=float)
    train_total = np.asarray([h.get("train_total", np.nan) for h in history], dtype=float)
    test_total = np.asarray([h.get("test_total", np.nan) for h in history], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)

    m_train = np.isfinite(x) & np.isfinite(train_total) & (train_total > 0.0)
    m_test = np.isfinite(x) & np.isfinite(test_total) & (test_total > 0.0)

    if np.any(m_train):
        ax.semilogy(x[m_train], train_total[m_train], label="train loss")
    if np.any(m_test):
        ax.semilogy(x[m_test], test_total[m_test], "o-", markersize=3, linewidth=1.2, label="test loss")

    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "loss_history.png")
    plt.close(fig)


def make_eval_grid(phys: PhysicsConfig, n_p=180, n_xi=120, fixed_params=(10.0, 1.0, 0.0)):
    energy = np.logspace(np.log10(phys.energy_min_ev), np.log10(phys.energy_max_ev), n_p)
    xi = np.linspace(phys.xi_min, phys.xi_max, n_xi)
    ee, xx = np.meshgrid(energy, xi, indexing="ij")
    gamma = 1.0 + ee / phys.mec_sq_ev
    p = np.sqrt(gamma**2 - 1.0)
    p_norm = (p - phys.p_min) / (phys.p_max - phys.p_min)
    xi_norm = (xx - phys.xi_min) / (phys.xi_max - phys.xi_min)
    evert, zeff, alpha = fixed_params
    z = np.stack(
        [
            p_norm.ravel(),
            xi_norm.ravel(),
            np.full(p_norm.size, norm01(evert, phys.evert_min, phys.evert_max)),
            np.full(p_norm.size, norm01(zeff, phys.zeff_min, phys.zeff_max)),
            np.full(p_norm.size, norm01(alpha, phys.alpha_min, phys.alpha_max)),
        ],
        axis=1,
    )
    return energy, xi, ee, xx, jnp.asarray(z, dtype=jnp.float64)


def plot_training_results(params, phys: PhysicsConfig, outdir: Path, fixed_params=(10.0, 1.0, 0.0)):
    if plt is None:
        return
    energy, xi, ee, xx, z = make_eval_grid(phys, fixed_params=fixed_params)
    p = np.asarray(prob_apply(params, z, phys)[:, 0]).reshape(ee.shape)
    r = np.asarray(jnp.abs(residual_apply(params, z, phys)[:, 0])).reshape(ee.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, constrained_layout=True)
    c0 = axes[0].contourf(ee, xx, p, levels=80, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel(r"$\xi$")
    axes[0].set_title("Runaway probability P")
    fig.colorbar(c0, ax=axes[0])

    vmax = np.nanpercentile(r, 99.0)
    c1 = axes[1].contourf(ee, xx, r, levels=80, cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-16))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel(r"$\xi$")
    axes[1].set_title("PDE residual magnitude")
    fig.colorbar(c1, ax=axes[1])
    fig.savefig(outdir / "training_results.png")
    plt.close(fig)


def write_run_config(outdir: Path, phys: PhysicsConfig, run: TrainConfig, fixed_params=None):
    payload = {"physics": asdict(phys), "run": asdict(run), "fixed_params": fixed_params}
    (outdir / "run_config.json").write_text(json.dumps(payload, indent=2))


def write_metadata(outdir: Path, phys: PhysicsConfig):
    metadata = {
        "model": "parametric runaway probability function PINN",
        "inputs": ["p_norm", "xi_norm", "evert_norm", "zeff_norm", "alpha_norm"],
        "outputs": {"forward": ["P"], "residual": ["R"]},
        "parameter_ranges": {
            "Evert": [phys.evert_min, phys.evert_max],
            "Zeff": [phys.zeff_min, phys.zeff_max],
            "alpha": [phys.alpha_min, phys.alpha_max],
            "energy_eV": [phys.energy_min_ev, phys.energy_max_ev],
            "xi": [phys.xi_min, phys.xi_max],
        },
        "models": {"forward": "rpf_forward.onnx", "residual": "rpf_residual.onnx"},
        "onnx": {
            "forward_batch": "dynamic",
            "residual_batch_size": RESIDUAL_ONNX_BATCH,
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))


# -----------------------------------------------------------------------------
# ONNX export
# -----------------------------------------------------------------------------

# The web app evaluates the forward model on the full 100x100 grid, so the
# forward ONNX model must have a symbolic batch dimension. The residual graph
# contains nested autodiff/JVP operations and is less robust with symbolic shape
# export, so it is exported with this fixed chunk size. main.js reads the same
# chunk size from metadata.json and pads residual calls to exactly this shape.
RESIDUAL_ONNX_BATCH = 4096


def export_onnx(params, outdir: Path, phys: PhysicsConfig):
    try:
        import onnx
        from jax2onnx import to_onnx
    except Exception as exc:
        print(f"[warn] ONNX export unavailable: {exc}")
        return False

    outdir.mkdir(parents=True, exist_ok=True)

    export_dtype = jnp.float32
    params32 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=export_dtype), params)

    forward_input_spec = jax.ShapeDtypeStruct(("B", 5), export_dtype)
    residual_input_spec = jax.ShapeDtypeStruct((RESIDUAL_ONNX_BATCH, 5), export_dtype)

    def fwd(x):
        x = jnp.asarray(x, dtype=export_dtype)
        return prob_apply(params32, x, phys).astype(export_dtype)

    def res(x):
        x = jnp.asarray(x, dtype=export_dtype)
        return residual_apply(params32, x, phys).astype(export_dtype)

    def export_one(fn, filename, input_spec):
        out_path = outdir / filename
        model = to_onnx(
            fn,
            inputs=[input_spec],
            enable_double_precision=False,
        )
        onnx.save(model, out_path)
        print(f"wrote {out_path}")

    export_one(fwd, "rpf_forward.onnx", forward_input_spec)
    export_one(res, "rpf_residual.onnx", residual_input_spec)
    write_metadata(outdir, phys)
    return True


def copy_models_for_web(outdir: Path, web_public_models: Path):
    web_public_models.mkdir(parents=True, exist_ok=True)
    for name in ["rpf_forward.onnx", "rpf_residual.onnx", "metadata.json"]:
        src = outdir / name
        if src.exists():
            dst = web_public_models / name
            dst.write_bytes(src.read_bytes())
            print(f"copied {src} -> {dst}")


# =============================================================================
# Parametric training entry point
# =============================================================================

PHYS = PhysicsConfig(
    evert_min=1.0,
    evert_max=10.0,
    zeff_min=1.0,
    zeff_max=10.0,
    alpha_min=0.0,
    alpha_max=0.2,
    output_transform="tanh_square",
    residual_scaling="none",
)

RUN = TrainConfig(
    seed=1234,
    width=48,
    depth=6,
    n_pde=2**20,
    n_bc_high=2**18,
    n_test=2**20,
    adam_steps=5000,
    adam_lr=1.0e-3,
    resample_every=1000,
    test_loss_every=100,
    print_every=1000,
    high_bc_weight=1.0,
    use_eox_bc_gate=True,
    eox_gate_mode="hard",
    eox_gate_width=0.05,
    eox_e_scan_min=1.0,
    eox_e_scan_max=10.0,
    eox_n_e_scan=256,
    eox_n_e_refine=32,
    eox_n_p_scan=2**12,
    eox_gamma_scan_min=1.02,
    eox_p_scan_max=300.0,
    eox_batch_size=2**12,
    adaptive_resampling=True,
    candidate_pool_mult=4,
    adaptive_resampling_k=1.0,
    adaptive_resampling_c=1.0,
    ssb_blocks=50,
    ssb_block_iters=1000,
    ssb_resample_every_blocks=1,
    ssb_rtol=1.0e-14,
    ssb_atol=1.0e-14,
    ssb_search="wolfe",
    ssb_verbose=False,
    outdir="models",
    copy_models_for_web=True,
    export_onnx=True,
)

# Representative parameter values used only for training_results.png.
PLOT_PARAMS = (10.0, 1.0, 0.0)


def main():
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    outdir = Path(RUN.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(RUN.seed)
    key, k_init, k_adam, k_ssb = random.split(key, 4)

    params = init_mlp(k_init, in_dim=5, width=RUN.width, depth=RUN.depth, out_dim=1)
    print(f"initialized parametric RPF MLP: input=5 width={RUN.width} depth={RUN.depth} output=1")
    print("parameters: |Ephi|, Zeff, alpha")
    print("output transform: tanh(p_norm**2 * raw**2)")
    print("high-energy BC: U_p>0 samples with E_OX gate; full-batch BC averaging")
    print("Adam train loss recorded every iteration")
    print(f"Adam test loss evaluated every {RUN.test_loss_every} iteration(s)")
    print(f"Adam progress printed every {RUN.print_every} iteration(s)")

    params, hist_adam = train_adam(params, k_adam, PHYS, RUN, fixed_params=None, history_prefix="adam")
    params, hist_ssb = train_ssbroyden(params, k_ssb, PHYS, RUN, fixed_params=None)
    history = hist_adam + hist_ssb

    save_history(history, outdir)
    plot_loss_history(history, outdir)
    plot_training_results(params, PHYS, outdir, fixed_params=PLOT_PARAMS)
    write_run_config(outdir, PHYS, RUN, fixed_params=None)

    if RUN.export_onnx:
        ok = export_onnx(params, outdir, PHYS)
        if ok and RUN.copy_models_for_web:
            copy_models_for_web(outdir, Path("web/public/models"))

    print(f"wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
