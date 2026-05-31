"""Fixed-parameter JAX PINN for the runaway probability function.

This standalone diagnostic script trains one fixed RPF case and writes:
  models/fixed/loss_history.png
  models/fixed/training_results.png

Features:
  - optional E_OX high-energy BC gate using OXmerger.py if available
  - batched residual-adaptive resampling to avoid evaluating huge candidate
    pools in one device call
  - blocked Adam with lax.scan for GPU efficiency
  - full training-loss bookkeeping from returned per-step scan losses
  - test loss evaluated/plotted only every TEST_LOSS_EVERY iterations
  - terminal printing controlled independently by PRINT_EVERY
  - optional localized residual downweighting near the high-energy
    endpoint (p=p_max, xi=xi_crit where U_p=0)
  - reduced O/X null-curve plotting with U_p=0 and scattering-inclusive
    Gamma_xi=0

Run:
  python train_rpf_fixed_jax.py
"""
import os

# Must be set before importing JAX.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
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

import equinox as eqx
import optimistix as optx


# =============================================================================
# Configuration
# =============================================================================

# Fixed physical parameters.
E_ABS = 2.55
ZEFF = 2.0
ALPHA = 0.1

# Physical domain.
MEC2_EV = 511.0e3
ENERGY_MIN_EV = 1.0e4
ENERGY_MAX_EV = 5.0e6
XI_MIN = -1.0
XI_MAX = 1.0

# Parameter ranges are retained only to normalize the fixed parameters in the
# same 5-input layout used by the parametric script: [p_norm, xi_norm, E_norm,
# Z_norm, alpha_norm].
E_ABS_MIN = 1.0
E_ABS_MAX = 20.0
ZEFF_MIN = 1.0
ZEFF_MAX = 5.0
ALPHA_MIN = 0.0
ALPHA_MAX = 0.1

# Model.
SEED = 1234
WIDTH = 32
DEPTH = 4

# Training points and Adam.
N_PDE = 2**18
N_BC_HIGH = 2**16
N_TEST = 2**18
ADAM_STEPS = 0
ADAM_LR = 1.0e-3
RESAMPLE_EVERY = 100
HIGH_BC_WEIGHT = 1.0

# Loss-history and printing controls.
# Full training loss is recorded every Adam iteration from returned scan losses.
# Test loss and terminal printing are independent host-side events.
TEST_LOSS_EVERY = 100
PRINT_EVERY = 1000

# Residual-adaptive resampling.
ADAPTIVE_RESAMPLING = True
CANDIDATE_POOL_MULT = 24
ADAPTIVE_K = 1.0
ADAPTIVE_C = 1.0
# Residuals for candidate pools are evaluated in chunks of this size.
# Increase CANDIDATE_POOL_MULT freely; lower this if GPU memory is tight.
CANDIDATE_RESIDUAL_BATCH = 2**15

# High-energy BC sampling margin away from the U_p=0 endpoint.
HIGH_BC_XI_MARGIN = 1.0E-10

# SSBroyden refinement. Set SSB_BLOCKS = 0 to skip.
SSB_BLOCKS = 50
SSB_BLOCK_ITERS = 500
SSB_RESAMPLE_EVERY_BLOCKS = 1
SSB_RTOL = 1.0e-14
SSB_ATOL = 1.0e-14
SSB_SEARCH = "trust_region"  # "wolfe", "zoom", or "trust_region"
SSB_VERBOSE = False

# Reduced O-X threshold gate for the high-energy BC.
# This calls OXmerger.py if available; otherwise a local fallback reduced model
# with the same closure is used.
USE_EOX_BC_GATE = True
EOX_GATE_MODE = "hard"  # "smooth" or "hard"
EOX_GATE_WIDTH = 0.5

# Fallback O-X calculation settings, used only if OXmerger.py is unavailable
# or its API is not recognized.
EOX_E_SCAN_MIN = 1.0
EOX_E_SCAN_MAX = 10.0
EOX_N_E_SCAN = 256
EOX_N_E_REFINE = 32
EOX_N_P_SCAN = 2**12
EOX_GAMMA_SCAN_MIN = 1.02
EOX_P_SCAN_MAX = 300.0
EOX_XI_EPS = 1.0e-10
EOX_MIN_ROOTS_FOR_VORTEX = 2

# Output.
OUTDIR = Path("models/fixed")

# Output transform and residual scaling.
RESIDUAL_SCALING = "none"  # "cf_e" or "none"

# Optional PDE residual weight localized at the high-energy U_p=0 endpoint.
# This regularizes the strong-form PDE loss only near
#     p_norm = 1, xi = xi_crit(E_ABS, ALPHA),
# without suppressing the entire U_p=0 curve. The diagnostic residual plot
# remains unweighted.
USE_ENDPOINT_RESIDUAL_WEIGHT = False
ENDPOINT_P_WIDTH = 1.0e-4
ENDPOINT_XI_WIDTH = 5.0e-1

# If True, adaptive resampling also ranks candidates by the weighted residual.
# If False, adaptive resampling still hunts the raw/unweighted residual.
ADAPTIVE_USES_ENDPOINT_WEIGHT = True


# =============================================================================
# Derived constants
# =============================================================================

GAMMA_MIN = 1.0 + ENERGY_MIN_EV / MEC2_EV
GAMMA_MAX = 1.0 + ENERGY_MAX_EV / MEC2_EV
P_MIN = math.sqrt(GAMMA_MIN**2 - 1.0)
P_MAX = math.sqrt(GAMMA_MAX**2 - 1.0)
E_NORM = (E_ABS - E_ABS_MIN) / (E_ABS_MAX - E_ABS_MIN)
Z_NORM = (ZEFF - ZEFF_MIN) / (ZEFF_MAX - ZEFF_MIN)
A_NORM = (ALPHA - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)

# Filled in at startup before JIT compilation of the loss.
EOX_VALUE = float("nan")
HIGH_BC_EOX_GATE = 1.0


# =============================================================================
# Utilities
# =============================================================================

def denorm01(x, lo, hi):
    return lo + x * (hi - lo)


def norm01(x, lo, hi):
    return (x - lo) / (hi - lo)


def gamma_from_p(p):
    return jnp.sqrt(1.0 + p * p)


def split_inputs(z):
    p_norm, xi_norm, e_norm, z_norm, a_norm = z
    p = denorm01(p_norm, P_MIN, P_MAX)
    xi = denorm01(xi_norm, XI_MIN, XI_MAX)
    e_abs = denorm01(e_norm, E_ABS_MIN, E_ABS_MAX)
    zeff = denorm01(z_norm, ZEFF_MIN, ZEFF_MAX)
    alpha = denorm01(a_norm, ALPHA_MIN, ALPHA_MAX)
    return p, xi, e_abs, zeff, alpha


def high_energy_xi_crit(e_abs, alpha):
    pmax = jnp.asarray(P_MAX, dtype=jnp.result_type(e_abs, alpha))
    gamma = gamma_from_p(pmax)
    c_f = gamma * gamma / (pmax * pmax)
    a = alpha * gamma * pmax
    eps = jnp.asarray(1.0e-14, dtype=a.dtype)
    xi_alpha0 = -c_f / e_abs
    disc = e_abs * e_abs + 4.0 * a * (c_f + a)
    xi_alpha_pos = (e_abs - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * jnp.maximum(a, eps))
    return jnp.clip(jnp.where(a > eps, xi_alpha_pos, xi_alpha0), XI_MIN, XI_MAX)


# =============================================================================
# Reduced O-X threshold calculation
# =============================================================================

def _eox_c_f_simple(p):
    gamma = gamma_from_p(p)
    return gamma * gamma / (p * p)


def _eox_nu_d_simple(p, zeff):
    gamma = gamma_from_p(p)
    return (zeff + 1.0) * gamma / (p * p * p)


def _eox_pitch_width_model(p, xi):
    return 2.0 * (1.0 + xi) / p


def _eox_xi_from_gamma_p_zero(p, e_abs, alpha):
    gamma = gamma_from_p(p)
    c_f = _eox_c_f_simple(p)
    a = alpha * gamma * p
    xi_alpha0 = -c_f / e_abs
    disc = e_abs * e_abs + 4.0 * a * (c_f + a)
    xi_alpha_pos = (e_abs - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * jnp.maximum(a, 1.0e-300))
    return jnp.where(a > 1.0e-14, xi_alpha_pos, xi_alpha0)


def _eox_gamma_xi_reduced_condition(p, xi, e_abs, zeff, alpha):
    gamma = gamma_from_p(p)
    nu_d = _eox_nu_d_simple(p, zeff)
    one_minus_xi2 = jnp.maximum(1.0 - xi * xi, 0.0)
    sqrt_one_minus_xi2 = jnp.sqrt(one_minus_xi2)
    delta_xi = jnp.maximum(_eox_pitch_width_model(p, xi), 1.0e-300)
    pitch_focusing = sqrt_one_minus_xi2 * (e_abs / p - alpha * p * xi / gamma)
    pitch_scattering = 0.5 * nu_d / delta_xi
    return pitch_focusing - pitch_scattering


def _eox_count_sign_change_roots(f, valid):
    f0 = f[:-1]
    f1 = f[1:]
    valid_pair = valid[:-1] & valid[1:]
    finite_pair = jnp.isfinite(f0) & jnp.isfinite(f1)
    sign_change = f0 * f1 <= 0.0
    return jnp.sum(valid_pair & finite_pair & sign_change)


def _eox_vortex_exists(e_abs, zeff, alpha):
    gamma_scan_min = jnp.asarray(EOX_GAMMA_SCAN_MIN, dtype=jnp.float64)
    p_scan_min = jnp.sqrt(gamma_scan_min * gamma_scan_min - 1.0)
    p_grid = jnp.geomspace(p_scan_min, EOX_P_SCAN_MAX, EOX_N_P_SCAN, dtype=jnp.float64)
    xi_gp0 = _eox_xi_from_gamma_p_zero(p_grid, e_abs, alpha)
    valid_xi = (xi_gp0 > XI_MIN + EOX_XI_EPS) & (xi_gp0 < XI_MAX - EOX_XI_EPS)
    f = _eox_gamma_xi_reduced_condition(p_grid, xi_gp0, e_abs, zeff, alpha)
    n_roots = _eox_count_sign_change_roots(f, valid_xi)
    return n_roots >= EOX_MIN_ROOTS_FOR_VORTEX


@jax.jit
def _eox_single_fallback(zeff, alpha):
    e_grid = jnp.linspace(EOX_E_SCAN_MIN, EOX_E_SCAN_MAX, EOX_N_E_SCAN, dtype=jnp.float64)
    has = jax.vmap(lambda ee: _eox_vortex_exists(ee, zeff, alpha))(e_grid)
    has_any = jnp.any(has)
    first_idx = jnp.argmax(has)
    lo_idx = jnp.maximum(first_idx - 1, 0)
    e_lo = e_grid[lo_idx]
    e_hi = e_grid[first_idx]

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        exists = _eox_vortex_exists(mid, zeff, alpha)
        hi_new = jnp.where(exists, mid, hi)
        lo_new = jnp.where(exists, lo, mid)
        return lo_new, hi_new

    _, e_refined = jax.lax.fori_loop(0, EOX_N_E_REFINE, body, (e_lo, e_hi))
    return jnp.where(has_any, e_refined, jnp.nan)


def compute_eox_from_oxmerger_or_fallback():
    """Compute E_OX(ZEFF, ALPHA), preferring OXmerger.py when available."""
    zeff_j = jnp.asarray(ZEFF, dtype=jnp.float64)
    alpha_j = jnp.asarray(ALPHA, dtype=jnp.float64)

    try:
        import OXmerger as ox

        if hasattr(ox, "e_ox_single"):
            val = ox.e_ox_single(zeff_j, alpha_j)
            return float(jax.device_get(val)), "OXmerger.e_ox_single"
        if hasattr(ox, "compute_eox"):
            return float(ox.compute_eox(ZEFF, ALPHA)), "OXmerger.compute_eox"
        if hasattr(ox, "compute_e_ox"):
            return float(ox.compute_e_ox(ZEFF, ALPHA)), "OXmerger.compute_e_ox"
        if hasattr(ox, "E_OX"):
            return float(ox.E_OX(ZEFF, ALPHA)), "OXmerger.E_OX"

        print("[warn] Imported OXmerger.py, but did not find a recognized E_OX function. Using fallback.")
    except Exception as exc:
        print(f"[warn] Could not use OXmerger.py for E_OX: {exc}. Using fallback.")

    val = _eox_single_fallback(zeff_j, alpha_j)
    return float(jax.device_get(val)), "local fallback"


def compute_eox_gate(e_abs, e_ox):
    if not USE_EOX_BC_GATE:
        return 1.0
    if not np.isfinite(e_ox):
        return 0.0
    if EOX_GATE_MODE == "hard":
        return float(e_abs > e_ox)
    if EOX_GATE_MODE == "smooth":
        return float(1.0 / (1.0 + np.exp(-(e_abs - e_ox) / EOX_GATE_WIDTH)))
    raise ValueError("EOX_GATE_MODE must be 'smooth' or 'hard'")


# =============================================================================
# Network and output transform
# =============================================================================

def init_mlp(key, in_dim=5, width=WIDTH, depth=DEPTH, out_dim=1):
    sizes = [in_dim] + [width] * depth + [out_dim]
    keys = random.split(key, len(sizes) - 1)
    params = []
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        w = random.normal(k, (din, dout), dtype=jnp.float64) * jnp.sqrt(2.0 / (din + dout))
        b = jnp.zeros((dout,), dtype=jnp.float64)
        params.append({"w": w, "b": b})
    return params


def mlp_apply(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]


def raw_apply_single(params, z):
    return mlp_apply(params, z[None, :])[0, 0]


def output_transform(p_norm, raw):
    return jnp.tanh(p_norm**2 * raw**2)


def prob_single(params, z):
    return output_transform(z[0], raw_apply_single(params, z))


def prob_apply(params, z):
    return jax.vmap(lambda zz: prob_single(params, zz))(z)[:, None]


# =============================================================================
# PDE residual
# =============================================================================

def residual_single(params, z):
    p, xi, e_abs, zeff, alpha = split_inputs(z)
    dp_dpbar = P_MAX - P_MIN
    dxi_dxibar = XI_MAX - XI_MIN

    def f(zz):
        return prob_single(params, zz)

    grad_z = jax.grad(f)(z)
    xi_bar_direction = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=z.dtype)
    _, d2_xibar2 = jax.jvp(lambda zz: jax.grad(f)(zz)[1], (z,), (xi_bar_direction,))

    p_p = grad_z[0] / dp_dpbar
    p_xi = grad_z[1] / dxi_dxibar
    p_xixi = d2_xibar2 / (dxi_dxibar * dxi_dxibar)

    gamma = gamma_from_p(p)
    c_f = gamma * gamma / (p * p)
    c_b_half = 0.5 * (zeff + 1.0) * gamma / p
    u_p = -xi * e_abs - c_f - alpha * gamma * p * (1.0 - xi * xi)

    momentum_advection = -u_p * p_p
    pitch_advection = (1.0 - xi * xi) * (e_abs / p - alpha * xi / gamma) * p_xi
    dflux_dxi = (1.0 - xi * xi) * p_xixi - 2.0 * xi * p_xi
    pitch_diffusion = -(c_b_half / (p * p)) * dflux_dxi
    residual = momentum_advection + pitch_advection + pitch_diffusion

    if RESIDUAL_SCALING == "cf_e":
        residual = residual / (c_f * e_abs)
    elif RESIDUAL_SCALING != "none":
        raise ValueError("RESIDUAL_SCALING must be 'cf_e' or 'none'")
    return residual


def residual_apply(params, z):
    return jax.vmap(lambda zz: residual_single(params, zz))(z)[:, None]


# =============================================================================
# Sampling
# =============================================================================

def fixed_param_columns(n):
    en = jnp.full((n, 1), E_NORM, dtype=jnp.float64)
    zn = jnp.full((n, 1), Z_NORM, dtype=jnp.float64)
    an = jnp.full((n, 1), A_NORM, dtype=jnp.float64)
    return en, zn, an


def sample_pde_points(key, n):
    px = random.uniform(key, (n, 2), dtype=jnp.float64)
    en, zn, an = fixed_param_columns(n)
    return jnp.concatenate([px[:, 0:1], px[:, 1:2], en, zn, an], axis=1)


def sample_high_boundary(key, n):
    xi_crit = high_energy_xi_crit(jnp.asarray(E_ABS, dtype=jnp.float64), jnp.asarray(ALPHA, dtype=jnp.float64))
    xi_hi = jnp.clip(xi_crit - jnp.asarray(HIGH_BC_XI_MARGIN, dtype=jnp.float64), XI_MIN, XI_MAX)
    s = random.uniform(key, (n,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    xi = XI_MIN + s * (xi_hi - XI_MIN)
    xi_norm = norm01(xi, XI_MIN, XI_MAX)
    p_norm = jnp.ones((n, 1), dtype=jnp.float64)
    en, zn, an = fixed_param_columns(n)
    return jnp.concatenate([p_norm, xi_norm[:, None], en, zn, an], axis=1)


def high_boundary_mask(z):
    p = denorm01(z[:, 0], P_MIN, P_MAX)
    xi = denorm01(z[:, 1], XI_MIN, XI_MAX)
    e_abs = denorm01(z[:, 2], E_ABS_MIN, E_ABS_MAX)
    alpha = denorm01(z[:, 4], ALPHA_MIN, ALPHA_MAX)
    gamma = gamma_from_p(p)
    c_f = gamma * gamma / (p * p)
    u_p = -xi * e_abs - c_f - alpha * gamma * p * (1.0 - xi * xi)
    return (u_p > 0.0).astype(jnp.float64)


def endpoint_residual_weight(z):
    """Smooth PDE-loss weight that vanishes only near (pmax, xi_crit).

    The target point is the high-energy endpoint where the p=pmax U_p>0
    boundary segment terminates. This avoids downweighting the whole U_p=0
    curve while relaxing the strong-form residual at the localized endpoint.
    """
    if not USE_ENDPOINT_RESIDUAL_WEIGHT:
        return jnp.ones((z.shape[0],), dtype=z.dtype)

    p_norm = z[:, 0]
    xi = denorm01(z[:, 1], XI_MIN, XI_MAX)
    xi_crit = high_energy_xi_crit(
        jnp.asarray(E_ABS, dtype=z.dtype),
        jnp.asarray(ALPHA, dtype=z.dtype),
    )

    dp = (1.0 - p_norm) / jnp.asarray(ENDPOINT_P_WIDTH, dtype=z.dtype)
    dxi = (xi - xi_crit) / jnp.asarray(ENDPOINT_XI_WIDTH, dtype=z.dtype)
    d2 = dp * dp# + dxi * dxi
    return 1.0 - jnp.exp(-d2)


@jax.jit
def _residual_abs_chunk(params, x_chunk):
    r = residual_apply(params, x_chunk)[:, 0]
    if ADAPTIVE_USES_ENDPOINT_WEIGHT:
        r = endpoint_residual_weight(x_chunk) * r
    return jnp.abs(r)


def residual_abs_batched(params, pool, chunk_size):
    """Evaluate |residual| over pool in chunks to limit peak memory use."""
    n = int(pool.shape[0])
    chunks = []
    for i0 in range(0, n, int(chunk_size)):
        i1 = min(i0 + int(chunk_size), n)
        r_i = _residual_abs_chunk(params, pool[i0:i1])
        chunks.append(r_i)
    return jnp.concatenate(chunks, axis=0)


def residual_adaptive_resample(params, key, n):
    pool_n = int(CANDIDATE_POOL_MULT * n)
    k_pool, k_choose = random.split(key)
    pool = sample_pde_points(k_pool, pool_n)

    r = residual_abs_batched(params, pool, CANDIDATE_RESIDUAL_BATCH)
    score = jnp.power(jnp.maximum(r, 0.0), ADAPTIVE_K)
    score = score / jnp.maximum(jnp.mean(score), jnp.finfo(score.dtype).tiny) + ADAPTIVE_C
    prob = score / jnp.sum(score)
    idx = random.choice(k_choose, pool_n, shape=(n,), replace=False, p=prob)
    return pool[idx]


def sample_or_adaptive(params, key, n):
    if ADAPTIVE_RESAMPLING and params is not None:
        return residual_adaptive_resample(params, key, n)
    return sample_pde_points(key, n)


# =============================================================================
# Loss and optimizers
# =============================================================================

def loss_terms(params, x_pde, x_high):
    r = residual_apply(params, x_pde)[:, 0]
    if USE_ENDPOINT_RESIDUAL_WEIGHT:
        r = endpoint_residual_weight(x_pde) * r
    loss_pde = jnp.mean(r * r)

    p_high = prob_apply(params, x_high)[:, 0]
    mask = high_boundary_mask(x_high)
    loss_high_raw = jnp.mean(mask * (p_high - 1.0) ** 2)
    loss_high = jnp.asarray(HIGH_BC_EOX_GATE, dtype=jnp.float64) * loss_high_raw
    total = loss_pde + HIGH_BC_WEIGHT * loss_high
    return total, loss_pde, loss_high


def train_adam(params, key):
    optimizer = optax.adam(ADAM_LR)
    state = optimizer.init(params)
    records_by_iter = {}

    key, k_pde, k_high, k_test, k_test_high = random.split(key, 5)
    x_pde = sample_or_adaptive(params, k_pde, N_PDE)
    x_high = sample_high_boundary(k_high, N_BC_HIGH)
    x_test = sample_pde_points(k_test, N_TEST)
    x_test_high = sample_high_boundary(k_test_high, N_BC_HIGH)

    def loss_only(pp, xpde, xhi):
        return loss_terms(pp, xpde, xhi)[0]

    @jax.jit
    def adam_step(pp, st, xpde, xhi):
        loss, grads = jax.value_and_grad(loss_only)(pp, xpde, xhi)
        updates, st = optimizer.update(grads, st, pp)
        pp = optax.apply_updates(pp, updates)
        return pp, st, loss

    @jax.jit
    def eval_total(pp, xpde, xhi):
        return loss_terms(pp, xpde, xhi)[0]

    @partial(jax.jit, static_argnames=("n_steps",))
    def adam_steps(pp, st, xpde, xhi, n_steps: int):
        def body(carry, _):
            pp_i, st_i = carry
            pp_i, st_i, loss_i = adam_step(pp_i, st_i, xpde, xhi)
            return (pp_i, st_i), loss_i

        (pp, st), losses = jax.lax.scan(body, (pp, st), xs=None, length=n_steps)
        return pp, st, losses

    def get_record(iter_i):
        if iter_i not in records_by_iter:
            records_by_iter[iter_i] = {
                "phase": "adam",
                "iter": int(iter_i),
                "step": int(iter_i),
                "train_total": float("nan"),
                "test_total": float("nan"),
            }
        return records_by_iter[iter_i]

    def next_boundary(done, every):
        if every <= 0:
            return ADAM_STEPS
        if done < 1:
            return 1
        return min(((done // every) + 1) * every, ADAM_STEPS)

    step = 0
    while step < ADAM_STEPS:
        if step > 0 and RESAMPLE_EVERY > 0 and step % RESAMPLE_EVERY == 0:
            key, k_pde, k_high = random.split(key, 3)
            x_pde = sample_or_adaptive(params, k_pde, N_PDE)
            x_high = sample_high_boundary(k_high, N_BC_HIGH)

        # Block boundaries are only events that require host-side work.
        # Training losses are returned for every Adam update from the compiled
        # scan and recorded afterward, so loss-history bookkeeping does not
        # force one-step dispatches.
        target = min(
            next_boundary(step, TEST_LOSS_EVERY),
            next_boundary(step, PRINT_EVERY),
            next_boundary(step, RESAMPLE_EVERY) if RESAMPLE_EVERY > 0 else ADAM_STEPS,
            ADAM_STEPS,
        )
        n_steps = max(target - step, 1)
        start_step = step

        params, state, losses = adam_steps(params, state, x_pde, x_high, int(n_steps))
        jax.block_until_ready(params)
        losses_np = np.asarray(jax.device_get(losses))
        step = target

        # Always keep the full per-iteration training-loss history.
        for j, loss_j in enumerate(losses_np):
            iter_j = start_step + j + 1
            get_record(iter_j)["train_total"] = float(loss_j)

        do_test = step == 1 or step == ADAM_STEPS or (TEST_LOSS_EVERY > 0 and step % TEST_LOSS_EVERY == 0)
        if do_test:
            test_total = eval_total(params, x_test, x_test_high)
            get_record(step)["test_total"] = float(jax.device_get(test_total))

        do_print = step == 1 or step == ADAM_STEPS or (PRINT_EVERY > 0 and step % PRINT_EVERY == 0)
        if do_print:
            rec = records_by_iter.get(step, {})
            train_val = rec.get("train_total", float("nan"))
            test_val = rec.get("test_total", float("nan"))
            if np.isfinite(test_val):
                print(f"adam step {step:7d} train={train_val:.3e} test={test_val:.3e}")
            else:
                print(f"adam step {step:7d} train={train_val:.3e}")

    history = [records_by_iter[k] for k in sorted(records_by_iter)]
    return params, history, x_pde

def make_ssb_solver():
    class SSBroyden(optx.AbstractSSBroyden):
        rtol: float
        atol: float
        norm: Callable = optx.max_norm
        use_inverse: bool = True
        search: optx.AbstractSearch = eqx.field(default_factory=optx.BacktrackingStrongWolfe)
        descent: optx.AbstractDescent = eqx.field(default_factory=optx.NewtonDescent)
        verbose: frozenset[str] = frozenset()

    if SSB_SEARCH == "wolfe":
        search = optx.BacktrackingStrongWolfe()
    elif SSB_SEARCH == "zoom":
        search = optx.Zoom()
    elif SSB_SEARCH == "trust_region":
        search = optx.LinearTrustRegion()
    else:
        raise ValueError("SSB_SEARCH must be 'wolfe', 'zoom', or 'trust_region'")

    solver = SSBroyden(
        rtol=SSB_RTOL,
        atol=SSB_ATOL,
        search=search,
        descent=optx.NewtonDescent(),
        verbose=frozenset(["loss"]) if SSB_VERBOSE else frozenset(),
    )
    return optx.BestSoFarMinimiser(solver)


def train_ssbroyden(params, key):
    if SSB_BLOCKS <= 0:
        return params, [], None

    solver = make_ssb_solver()
    weights, unflatten = ravel_pytree(params)
    n_params = int(weights.shape[0])
    print("SSBroyden backend: optimistix")
    print(f"SSBroyden parameter count: {n_params}")
    print(f"Dense inverse-H memory estimate: {(n_params * n_params * 8) / 1.0e9:.3f} GB")

    history = []

    @jax.jit
    def optimistix_loss(w, batch):
        xpde, xhi = batch
        return loss_terms(unflatten(w), xpde, xhi)[0]

    @jax.jit
    def ssb_step(w, xpde, xhi):
        batch = (xpde, xhi)
        loss_before = optimistix_loss(w, batch)
        sol = optx.minimise(
            optimistix_loss,
            solver,
            w,
            args=batch,
            max_steps=SSB_BLOCK_ITERS,
            throw=False,
        )
        w_new = sol.value
        loss_after = optimistix_loss(w_new, batch)
        delta = jnp.linalg.norm(w_new - w)
        finite = jnp.all(jnp.isfinite(w_new)) & jnp.isfinite(loss_after)
        num_steps = sol.stats["num_steps"]
        return w_new, loss_before, loss_after, delta, finite, num_steps

    @jax.jit
    def eval_total_flat(w, xpde, xhi):
        return loss_terms(unflatten(w), xpde, xhi)[0]

    x_pde = x_high = x_test = x_test_high = None
    for block in range(1, SSB_BLOCKS + 1):
        if x_pde is None or (SSB_RESAMPLE_EVERY_BLOCKS > 0 and (block - 1) % SSB_RESAMPLE_EVERY_BLOCKS == 0):
            key, k_pde, k_high, k_test, k_test_high = random.split(key, 5)
            x_pde = sample_or_adaptive(unflatten(weights), k_pde, N_PDE)
            x_high = sample_high_boundary(k_high, N_BC_HIGH)
            x_test = sample_pde_points(k_test, N_TEST)
            x_test_high = sample_high_boundary(k_test_high, N_BC_HIGH)

        weights, loss_before, loss_after, delta, finite, num_steps = ssb_step(weights, x_pde, x_high)
        jax.block_until_ready(weights)
        train_total = eval_total_flat(weights, x_pde, x_high)
        test_total = eval_total_flat(weights, x_test, x_test_high)
        train_total, test_total, loss_before, loss_after, delta, finite, num_steps = jax.device_get(
            (train_total, test_total, loss_before, loss_after, delta, finite, num_steps)
        )
        rec = {
            "phase": "ssbroyden",
            "iter": int(ADAM_STEPS + block),
            "step": int(block),
            "train_total": float(train_total),
            "test_total": float(test_total),
            "block_loss_before": float(loss_before),
            "block_loss_after": float(loss_after),
            "block_delta_weights": float(delta),
        }
        history.append(rec)
        print(
            f"ssb block {block:4d}/{SSB_BLOCKS} "
            f"block={rec['block_loss_before']:.3e}->{rec['block_loss_after']:.3e} "
            f"steps={int(num_steps)} finite={bool(finite)} |dw|={rec['block_delta_weights']:.3e} "
            f"train={rec['train_total']:.3e} test={rec['test_total']:.3e}"
        )
    return unflatten(weights), history, x_pde


# =============================================================================
# Plots
# =============================================================================

def training_points_to_energy_xi(x_train, max_points=8000):
    x_np = np.asarray(jax.device_get(x_train))
    if x_np.shape[0] > max_points:
        idx = np.linspace(0, x_np.shape[0] - 1, max_points).astype(int)
        x_np = x_np[idx]
    p_norm = x_np[:, 0]
    xi_norm = x_np[:, 1]
    p = P_MIN + p_norm * (P_MAX - P_MIN)
    gamma = np.sqrt(1.0 + p**2)
    energy = (gamma - 1.0) * MEC2_EV
    xi = XI_MIN + xi_norm * (XI_MAX - XI_MIN)
    return energy, xi


def plot_loss_history(history):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    x = np.asarray([h.get("iter", h.get("step", i)) for i, h in enumerate(history)], dtype=float)
    train = np.asarray([h.get("train_total", np.nan) for h in history], dtype=float)
    test = np.asarray([h.get("test_total", np.nan) for h in history], dtype=float)

    m_train = np.isfinite(train) & (train > 0.0)
    m_test = np.isfinite(test) & (test > 0.0)

    if np.any(m_train):
        ax.semilogy(x[m_train], train[m_train], label="train loss")
    if np.any(m_test):
        ax.semilogy(x[m_test], test[m_test], "o-", markersize=3, linewidth=1.2, label="test loss")

    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "loss_history.png")
    plt.close(fig)


def eval_grid(n_p=512, n_xi=512):
    energy = np.logspace(np.log10(ENERGY_MIN_EV), np.log10(ENERGY_MAX_EV), n_p)
    xi = np.linspace(XI_MIN, XI_MAX, n_xi)
    ee, xx = np.meshgrid(energy, xi, indexing="ij")
    gamma = 1.0 + ee / MEC2_EV
    p = np.sqrt(gamma**2 - 1.0)
    p_norm = (p - P_MIN) / (P_MAX - P_MIN)
    xi_norm = (xx - XI_MIN) / (XI_MAX - XI_MIN)
    z = np.stack(
        [
            p_norm.ravel(),
            xi_norm.ravel(),
            np.full(p_norm.size, E_NORM),
            np.full(p_norm.size, Z_NORM),
            np.full(p_norm.size, A_NORM),
        ],
        axis=1,
    )
    return ee, xx, jnp.asarray(z, dtype=jnp.float64)


def reduced_ox_null_fields(ee, xx):
    """Return U_p and reduced Gamma_xi fields on the plotting grid."""
    gamma = 1.0 + ee / MEC2_EV
    p = np.sqrt(gamma**2 - 1.0)
    xi = xx

    c_f = gamma * gamma / (p * p)
    u_p = -xi * E_ABS - c_f - ALPHA * gamma * p * (1.0 - xi * xi)

    one_minus_xi2 = np.maximum(1.0 - xi * xi, 0.0)
    sqrt_one_minus_xi2 = np.sqrt(one_minus_xi2)
    nu_d = (ZEFF + 1.0) * gamma / (p * p * p)
    delta_xi = 2.0 * (1.0 + xi) / p
    delta_xi = np.maximum(delta_xi, 1.0e-300)

    pitch_focusing = sqrt_one_minus_xi2 * (E_ABS / p - ALPHA * p * xi / gamma)
    pitch_scattering = 0.5 * nu_d / delta_xi
    gamma_xi = pitch_focusing - pitch_scattering
    return u_p, gamma_xi


def _safe_zero_contour(ax, ee, xx, field, **kwargs):
    fmin = np.nanmin(field)
    fmax = np.nanmax(field)
    if np.isfinite(fmin) and np.isfinite(fmax) and fmin <= 0.0 <= fmax and fmin < fmax:
        return ax.contour(ee, xx, field, levels=[0.0], **kwargs)
    return None


def plot_training_results(params, train_points=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ee, xx, z = eval_grid()
    p_pred = np.asarray(prob_apply(params, z)[:, 0]).reshape(ee.shape)
    r = np.asarray(jnp.abs(residual_apply(params, z)[:, 0])).reshape(ee.shape)
    u_p, gamma_xi = reduced_ox_null_fields(ee, xx)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=150, constrained_layout=True)

    c0 = axes[0, 0].contourf(ee, xx, p_pred, levels=50, cmap="turbo")
    if train_points is not None:
        e_pts, xi_pts = training_points_to_energy_xi(train_points)
        axes[0, 0].scatter(
            e_pts,
            xi_pts,
            s=1.0,
            c="white",
            alpha=0.25,
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
            label="final PDE points",
        )
        axes[0, 0].legend(loc="lower right", fontsize=7, framealpha=0.6)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Energy [eV]")
    axes[0, 0].set_ylabel(r"$\xi$")
    axes[0, 0].set_title("Runaway probability P")
    fig.colorbar(c0, ax=axes[0, 0])

    c1 = axes[0, 1].contourf(ee, xx, r, levels=50, cmap="inferno")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Energy [eV]")
    axes[0, 1].set_ylabel(r"$\xi$")
    axes[0, 1].set_title("PDE residual magnitude")
    fig.colorbar(c1, ax=axes[0, 1])

    bg = np.tanh(u_p / max(np.nanpercentile(np.abs(u_p), 95.0), 1.0e-16))
    c2 = axes[1, 0].contourf(ee, xx, bg, levels=50, cmap="coolwarm", vmin=-1.0, vmax=1.0, alpha=0.75)
    _safe_zero_contour(axes[1, 0], ee, xx, u_p, colors="tab:blue", linewidths=2.0)
    _safe_zero_contour(axes[1, 0], ee, xx, gamma_xi, colors="tab:orange", linewidths=2.0)
    axes[1, 0].plot([], [], color="tab:blue", linewidth=2.0, label=r"$U_p=0$")
    axes[1, 0].plot([], [], color="tab:orange", linewidth=2.0, label=r"$\Gamma_\xi=0$")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Energy [eV]")
    axes[1, 0].set_ylabel(r"$\xi$")
    axes[1, 0].set_title(r"Reduced O/X null curves")
    axes[1, 0].legend(fontsize=8, loc="best")
    fig.colorbar(c2, ax=axes[1, 0], label=r"signed $U_p$ background")

    c3 = axes[1, 1].contourf(ee, xx, p_pred, levels=50, cmap="turbo", vmin=0.0, vmax=1.0)
    _safe_zero_contour(axes[1, 1], ee, xx, u_p, colors="white", linewidths=2.0)
    _safe_zero_contour(axes[1, 1], ee, xx, gamma_xi, colors="black", linewidths=2.0)
    axes[1, 1].plot([], [], color="white", linewidth=2.0, label=r"$U_p=0$")
    axes[1, 1].plot([], [], color="black", linewidth=2.0, label=r"$\Gamma_\xi=0$")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Energy [eV]")
    axes[1, 1].set_ylabel(r"$\xi$")
    axes[1, 1].set_title(r"$P$ with reduced O/X null curves")
    axes[1, 1].legend(fontsize=8, loc="best")
    fig.colorbar(c3, ax=axes[1, 1])

    fig.savefig(OUTDIR / "training_results.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    global EOX_VALUE, HIGH_BC_EOX_GATE

    print("XLA_PYTHON_CLIENT_PREALLOCATE:", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print(f"initialized fixed RPF MLP: input=5 width={WIDTH} depth={DEPTH} output=1")
    print(f"fixed parameters: |Ephi|={E_ABS}, Zeff={ZEFF}, alpha={ALPHA}")
    print(f"adaptive candidate pool: {CANDIDATE_POOL_MULT} * {N_PDE} = {CANDIDATE_POOL_MULT * N_PDE}")
    print(f"candidate residual batch size: {CANDIDATE_RESIDUAL_BATCH}")
    print("Adam optimizer: optax.adam with blocked lax.scan")

    gamma_max = jnp.sqrt(1.0 + P_MAX**2)
    cf_max = gamma_max**2 / P_MAX**2
    A = ALPHA * gamma_max * P_MAX
    if float(abs(A)) > 0.0:
        xi_crit = (E_ABS - jnp.sqrt(E_ABS**2 + 4.0 * A * (cf_max + A))) / (2.0 * A)
    else:
        xi_crit = -cf_max / E_ABS
    xi_crit = jnp.clip(xi_crit, XI_MIN, XI_MAX)
    print(f"high-energy BC interval: xi in [{XI_MIN:.6f}, {float(xi_crit):.6f})")
    print(f"high-energy BC sampling margin from U_p=0: {HIGH_BC_XI_MARGIN:.3e}")
    print(
        "endpoint residual weight: "
        f"enabled={USE_ENDPOINT_RESIDUAL_WEIGHT}, "
        f"p_width={ENDPOINT_P_WIDTH:.3e}, xi_width={ENDPOINT_XI_WIDTH:.3e}, "
        f"adaptive_uses_weight={ADAPTIVE_USES_ENDPOINT_WEIGHT}"
    )

    EOX_VALUE, eox_source = compute_eox_from_oxmerger_or_fallback()
    HIGH_BC_EOX_GATE = compute_eox_gate(E_ABS, EOX_VALUE)
    print(f"E_OX source: {eox_source}")
    print(f"E_OX(Zeff={ZEFF}, alpha={ALPHA}) = {EOX_VALUE:.8g}")
    print(f"E_OX BC gate mode={EOX_GATE_MODE}, width={EOX_GATE_WIDTH}, gate={HIGH_BC_EOX_GATE:.8g}")

    key = random.PRNGKey(SEED)
    key, k_init, k_adam, k_ssb = random.split(key, 4)
    params = init_mlp(k_init)

    params, hist_adam, adam_points = train_adam(params, k_adam)
    params, hist_ssb, ssb_points = train_ssbroyden(params, k_ssb)
    history = hist_adam + hist_ssb

    final_train_points = ssb_points if ssb_points is not None else adam_points
    plot_loss_history(history)
    plot_training_results(params, final_train_points)
    print(f"wrote diagnostics to {OUTDIR}")


if __name__ == "__main__":
    main()
