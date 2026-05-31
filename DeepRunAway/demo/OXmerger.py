"""JAX/JIT reduced O-X threshold random-sample scan.

This standalone script estimates E_ox(Zeff, alpha) at random 2D samples.

Outputs:
  models/eox/eox_random_contour.png
  models/eox/eox_random_samples.npz

Run:
  python compute_eox_random_jax.py
"""

import os

# Must be set before importing JAX.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

OUTDIR = Path("models/eox")

SEED = 1234

# Random parameter samples.
N_SAMPLES_TOTAL = 2**15
BATCH_SIZE = 2**13

ZEFF_MIN = 1.0
ZEFF_MAX = 2.0

ALPHA_MIN = 0.0
ALPHA_MAX = 0.5

# Electric-field scan.
E_SCAN_MIN = 1.0
E_SCAN_MAX = 10.0
N_E_SCAN = 256
N_E_REFINE = 32

# Momentum scan.
N_P_SCAN = 2**12
GAMMA_SCAN_MIN = 1.02
P_SCAN_MIN = float(np.sqrt(GAMMA_SCAN_MIN**2 - 1.0))
P_SCAN_MAX = 300.0

# Pitch limits.
XI_MIN = -1.0
XI_MAX = 1.0
XI_EPS = 1.0e-10

# Above threshold we expect two roots corresponding to O/X pair.
MIN_ROOTS_FOR_VORTEX = 2

# Plotting.
N_CONTOUR_LEVELS = 80
PLOT_POINT_OVERLAY = True
POINT_SIZE = 1.0
POINT_ALPHA = 0.20


# =============================================================================
# Static grids
# =============================================================================

P_GRID = jnp.asarray(
    np.geomspace(P_SCAN_MIN, P_SCAN_MAX, N_P_SCAN),
    dtype=jnp.float64,
)

E_GRID = jnp.asarray(
    np.linspace(E_SCAN_MIN, E_SCAN_MAX, N_E_SCAN),
    dtype=jnp.float64,
)


# =============================================================================
# Reduced O-X model
# =============================================================================

def gamma_from_p(p):
    return jnp.sqrt(1.0 + p * p)


def c_f_simple(p):
    gamma = gamma_from_p(p)
    return gamma * gamma / (p * p)


def nu_d_simple(p, zeff):
    gamma = gamma_from_p(p)
    return (zeff + 1.0) * gamma / (p * p * p)


def pitch_width_model(p, xi):
    return 2.0 * (1.0 + xi) / p


def xi_from_gamma_p_zero(p, e_abs, alpha):
    gamma = gamma_from_p(p)
    cf = c_f_simple(p)
    a = alpha * gamma * p

    xi_alpha0 = -cf / e_abs

    disc = e_abs * e_abs + 4.0 * a * (cf + a)
    xi_alpha_pos = (e_abs - jnp.sqrt(jnp.maximum(disc, 0.0))) / (
        2.0 * jnp.maximum(a, 1.0e-300)
    )

    return jnp.where(a > 1.0e-14, xi_alpha_pos, xi_alpha0)


def gamma_xi_reduced_condition(p, xi, e_abs, zeff, alpha):
    gamma = gamma_from_p(p)
    nud = nu_d_simple(p, zeff)

    one_minus_xi2 = jnp.maximum(1.0 - xi * xi, 0.0)
    sqrt_one_minus_xi2 = jnp.sqrt(one_minus_xi2)

    delta_xi = jnp.maximum(pitch_width_model(p, xi), 1.0e-300)

    pitch_focusing = sqrt_one_minus_xi2 * (e_abs / p - alpha * p * xi / gamma)
    pitch_scattering = 0.5 * nud / delta_xi

    return pitch_focusing - pitch_scattering


def count_sign_change_roots(f, valid):
    f0 = f[:-1]
    f1 = f[1:]

    valid_pair = valid[:-1] & valid[1:]
    finite_pair = jnp.isfinite(f0) & jnp.isfinite(f1)
    sign_change = f0 * f1 <= 0.0

    return jnp.sum(valid_pair & finite_pair & sign_change)


def vortex_exists(e_abs, zeff, alpha):
    p = P_GRID

    xi_gp0 = xi_from_gamma_p_zero(p, e_abs, alpha)
    valid_xi = (xi_gp0 > XI_MIN + XI_EPS) & (xi_gp0 < XI_MAX - XI_EPS)

    f = gamma_xi_reduced_condition(p, xi_gp0, e_abs, zeff, alpha)
    n_roots = count_sign_change_roots(f, valid_xi)

    return n_roots >= MIN_ROOTS_FOR_VORTEX


def vortex_exists_on_e_grid(zeff, alpha):
    return jax.vmap(lambda e: vortex_exists(e, zeff, alpha))(E_GRID)


def refine_threshold_bisection(e_lo, e_hi, zeff, alpha):
    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        exists = vortex_exists(mid, zeff, alpha)

        hi_new = jnp.where(exists, mid, hi)
        lo_new = jnp.where(exists, lo, mid)

        return lo_new, hi_new

    _, hi = jax.lax.fori_loop(0, N_E_REFINE, body, (e_lo, e_hi))
    return hi


def e_ox_single(zeff, alpha):
    has = vortex_exists_on_e_grid(zeff, alpha)
    has_any = jnp.any(has)

    first_idx = jnp.argmax(has)
    lo_idx = jnp.maximum(first_idx - 1, 0)

    e_lo = E_GRID[lo_idx]
    e_hi = E_GRID[first_idx]

    e_refined = refine_threshold_bisection(e_lo, e_hi, zeff, alpha)
    return jnp.where(has_any, e_refined, jnp.nan)


@jax.jit
def e_ox_batch(zeff_batch, alpha_batch):
    return jax.vmap(e_ox_single)(zeff_batch, alpha_batch)


# =============================================================================
# Sampling and plotting
# =============================================================================

def sample_random_parameters(seed, n):
    rng = np.random.default_rng(seed)
    zeff = rng.uniform(ZEFF_MIN, ZEFF_MAX, size=n)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n)
    return zeff.astype(np.float64), alpha.astype(np.float64)


def compute_in_batches(zeff, alpha):
    n = zeff.shape[0]
    eox = np.empty(n, dtype=np.float64)

    n_batches = int(np.ceil(n / BATCH_SIZE))

    print(f"total samples: {n}")
    print(f"batch size:    {BATCH_SIZE}")
    print(f"num batches:   {n_batches}")

    # Compile on first batch.
    compile_t0 = time.perf_counter()

    for b in range(n_batches):
        i0 = b * BATCH_SIZE
        i1 = min((b + 1) * BATCH_SIZE, n)

        z_b = jnp.asarray(zeff[i0:i1], dtype=jnp.float64)
        a_b = jnp.asarray(alpha[i0:i1], dtype=jnp.float64)

        t0 = time.perf_counter()
        e_b = e_ox_batch(z_b, a_b)
        e_b.block_until_ready()
        t1 = time.perf_counter()

        eox[i0:i1] = np.asarray(e_b)

        if b == 0:
            compile_t1 = time.perf_counter()
            print(f"batch {b + 1:4d}/{n_batches}: {i1 - i0:6d} samples, {t1 - t0:.3f} s  includes compile")
            print(f"compile + first batch: {compile_t1 - compile_t0:.3f} s")
        else:
            print(f"batch {b + 1:4d}/{n_batches}: {i1 - i0:6d} samples, {t1 - t0:.3f} s")

    return eox


def plot_random_eox_contour(zeff, alpha, eox):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    finite = np.isfinite(eox)
    if np.count_nonzero(finite) < 3:
        print("Not enough finite E_ox samples to make a contour plot.")
        return

    z = zeff[finite]
    a = alpha[finite]
    e = eox[finite]

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=170)

    c = ax.tricontourf(
        z,
        a,
        e,
        levels=N_CONTOUR_LEVELS,
        cmap="turbo",
    )

    if PLOT_POINT_OVERLAY:
        ax.scatter(
            z,
            a,
            s=POINT_SIZE,
            c="black",
            alpha=POINT_ALPHA,
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
        )

    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"$E_{\mathrm{OX}}$")

    ax.set_xlim(ZEFF_MIN, ZEFF_MAX)
    ax.set_ylim(ALPHA_MIN, ALPHA_MAX)
    ax.set_xlabel(r"$Z_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(
        rf"Reduced O-X threshold from random samples "
        rf"$(N={len(zeff):,})$"
    )

    fig.tight_layout()
    fig.savefig(OUTDIR / "eox_random_contour.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("XLA_PYTHON_CLIENT_PREALLOCATE:", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print(
        f"N_SAMPLES_TOTAL={N_SAMPLES_TOTAL}, BATCH_SIZE={BATCH_SIZE}, "
        f"N_E_SCAN={N_E_SCAN}, N_E_REFINE={N_E_REFINE}, N_P_SCAN={N_P_SCAN}"
    )
    print(f"total reduced checks: {N_SAMPLES_TOTAL * N_E_SCAN:,}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    zeff, alpha = sample_random_parameters(SEED, N_SAMPLES_TOTAL)

    t0 = time.perf_counter()
    eox = compute_in_batches(zeff, alpha)
    t1 = time.perf_counter()

    plot_random_eox_contour(zeff, alpha, eox)

    finite = np.isfinite(eox)
    print(f"finite fraction: {np.mean(finite):.3f}")
    if np.any(finite):
        print(f"E_ox finite range: {np.nanmin(eox):.6g} to {np.nanmax(eox):.6g}")

    print(f"total runtime: {t1 - t0:.3f} s")
    print(f"wrote {OUTDIR / 'eox_random_samples.npz'}")
    print(f"wrote {OUTDIR / 'eox_random_contour.png'}")


if __name__ == "__main__":
    main()