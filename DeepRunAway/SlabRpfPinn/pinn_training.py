#!/usr/bin/env python3
"""FV-assisted parametric steady-RPF PINN trainer.

Purpose
-------
Train one fully parametric steady runaway-probability function

    P(p, xi; E/Ec, Te, nD, nImp, <ZD>, <ZImp>)

on one GPU using

    L = w_pde L_pde + w_bc L_bc + w_fv L_fv,

with Adam warmup followed by configurable blockwise full-batch SSBroyden.
Each SSBroyden block starts a fresh inverse-Hessian approximation; PDE/BC
collocation points may optionally be resampled between blocks while the FV
supervised data remain fixed.

The script can generate its own steady FV labels by importing the unified
Warp/cuDSS solver, or load a previously generated dataset.  FV generation is
outside JAX/XLA.  Training arrays are fixed, transferred to the GPU once, and
the repeated training math is JIT compiled.  The fixed PDE operator
coefficients are precomputed once per collocation set. Full-batch losses may
be evaluated either in fixed-size XLA chunks or in one whole-batch call.

The PINN physics matches the FV first-passage model used here:
  * quasineutral ne from atomic D + impurity densities and mean charge states;
  * linear neighboring-charge-state shape functions;
  * matched relativistic small-angle collisions;
  * Hesslow energy-dependent Coulomb logarithms and partial screening;
  * synchrotron reaction;
  * no radial energy diffusion;
  * pmin absorbing failure is hard-enforced by the network;
  * pmax P=1 only where the forward radial drift is outward (successful escape).

No OXmerger, ONNX, web export, residual-adaptive sampling, or plotting is included.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import tomllib
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

# JAX and Warp share one GPU.  Do not let JAX reserve the whole device up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import scipy.constants as const
from scipy.stats import qmc

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import optax
import equinox as eqx
import optimistix as optx


CONFIG_PATH = Path(__file__).with_suffix(".toml")
EPS = 1.0e-300
MEC2_EV = const.m_e * const.c**2 / const.e
E_CHARGE = const.e
EPS0 = const.epsilon_0
M_E = const.m_e
C_LIGHT = const.c
SIGMA = 5.0

CFG: dict = {}
IMPURITY_Z: int = 10
D_I_EV: jnp.ndarray | None = None
D_ABAR: jnp.ndarray | None = None
IMP_I_EV: jnp.ndarray | None = None
IMP_ABAR: jnp.ndarray | None = None


# =============================================================================
# Configuration and fixed-domain mappings
# =============================================================================

def load_config(config_path: Path | None = None) -> tuple[dict, Path]:
    path = (config_path or CONFIG_PATH).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    for section in ("fv", "domain", "training", "loss", "output"):
        if section not in raw:
            raise ValueError(f"missing [{section}] section in {path}")

    fv = raw["fv"]
    dom = raw["domain"]
    tr = raw["training"]

    if int(fv["n_train_cases"]) <= 0 or int(fv["n_validation_cases"]) <= 0:
        raise ValueError("FV train and validation case counts must be positive")
    if int(fv["Np"]) <= 1 or int(fv["Nxi"]) <= 1:
        raise ValueError("FV grid must have Np,Nxi > 1")
    if not (0.0 < float(fv["pmin"]) < float(fv["pmax"])):
        raise ValueError("require 0 < fv.pmin < fv.pmax")
    if bool(fv.get("energy_diffusion", False)):
        raise ValueError("this PINN residual currently requires fv.energy_diffusion=false")

    zimp = int(dom.get("impurity_Z", 10))
    if not (0.0 <= float(dom["zDavg_min"]) <= float(dom["zDavg_max"]) <= 1.0):
        raise ValueError("require 0 <= <ZD> <= 1")
    if not (0.0 <= float(dom["zImpavg_min"]) <= float(dom["zImpavg_max"]) <= zimp):
        raise ValueError("require 0 <= <ZImp> <= impurity_Z")

    ranged_parameters = (
        ("e_parallel", "e_parallel_min", "e_parallel_max", "e_parallel_sampling"),
        ("te", "te_min_ev", "te_max_ev", "te_sampling"),
        ("nD", "nD_min_m3", "nD_max_m3", "nD_sampling"),
        ("nImp", "nImp_min_m3", "nImp_max_m3", "nImp_sampling"),
    )
    for label, lo_key, hi_key, scale_key in ranged_parameters:
        lo = float(dom[lo_key])
        hi = float(dom[hi_key])
        scale = str(dom[scale_key]).lower()
        if not (hi > lo > 0.0):
            raise ValueError(f"require positive increasing range for {label}")
        if scale not in ("linear", "log"):
            raise ValueError(f"{scale_key} must be linear or log")

    for name in ("n_pde", "n_bc"):
        if int(tr[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    for name in ("pde_chunk", "bc_chunk", "fv_chunk"):
        if int(tr[name]) < 0:
            raise ValueError(f"training.{name} must be >= 0 (0 means whole-batch evaluation)")
    if int(tr["adam_steps"]) < 0 or int(tr["adam_block_steps"]) <= 0:
        raise ValueError("invalid Adam step configuration")
    if int(tr["adam_steps"]) % int(tr["adam_block_steps"]) != 0:
        raise ValueError("adam_steps must be divisible by adam_block_steps")
    if int(tr["ssb_blocks"]) <= 0 or int(tr["ssb_block_iters"]) <= 0:
        raise ValueError("training.ssb_blocks and training.ssb_block_iters must be positive")
    if int(tr["ssb_resample_every_blocks"]) < 0:
        raise ValueError("training.ssb_resample_every_blocks must be >= 0")
    if str(tr["ssb_search"]).lower() not in ("wolfe", "zoom", "trust_region"):
        raise ValueError("training.ssb_search must be wolfe, zoom, or trust_region")

    return raw, path


def _map_unit_np(u: np.ndarray, lo: float, hi: float, scale: str) -> np.ndarray:
    if scale == "linear":
        return lo + u * (hi - lo)
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def _map_unit_jax(u, lo: float, hi: float, scale: str):
    if scale == "linear":
        return lo + u * (hi - lo)
    return jnp.exp(jnp.log(lo) + u * (jnp.log(hi) - jnp.log(lo)))


def sobol_unit(n: int, dim: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, dim), dtype=np.float64)
    m = int(math.ceil(math.log2(max(n, 1))))
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return np.asarray(eng.random_base2(m)[:n], dtype=np.float64)


def theta_phys_from_unit_np(u: np.ndarray) -> np.ndarray:
    d = CFG["domain"]
    out = np.empty_like(u, dtype=np.float64)
    out[:, 0] = _map_unit_np(u[:, 0], float(d["e_parallel_min"]), float(d["e_parallel_max"]), str(d["e_parallel_sampling"]).lower())
    out[:, 1] = _map_unit_np(u[:, 1], float(d["te_min_ev"]), float(d["te_max_ev"]), str(d["te_sampling"]).lower())
    out[:, 2] = _map_unit_np(u[:, 2], float(d["nD_min_m3"]), float(d["nD_max_m3"]), str(d["nD_sampling"]).lower())
    out[:, 3] = _map_unit_np(u[:, 3], float(d["nImp_min_m3"]), float(d["nImp_max_m3"]), str(d["nImp_sampling"]).lower())
    out[:, 4] = float(d["zDavg_min"]) + u[:, 4] * (float(d["zDavg_max"]) - float(d["zDavg_min"]))
    out[:, 5] = float(d["zImpavg_min"]) + u[:, 5] * (float(d["zImpavg_max"]) - float(d["zImpavg_min"]))
    return out


def p_from_pnorm(pn):
    fv = CFG["fv"]
    l0 = math.log(float(fv["pmin"]))
    l1 = math.log(float(fv["pmax"]))
    return jnp.exp(l0 + pn * (l1 - l0))


def pnorm_from_p_np(p: np.ndarray) -> np.ndarray:
    fv = CFG["fv"]
    l0 = math.log(float(fv["pmin"]))
    l1 = math.log(float(fv["pmax"]))
    return (np.log(p) - l0) / (l1 - l0)


def pnorm_from_p_jax(p):
    fv = CFG["fv"]
    l0 = math.log(float(fv["pmin"]))
    l1 = math.log(float(fv["pmax"]))
    return (jnp.log(p) - l0) / (l1 - l0)


def xi_from_xinorm(xn):
    return -1.0 + 2.0 * xn


def xinorm_from_xi_np(xi: np.ndarray) -> np.ndarray:
    return 0.5 * (xi + 1.0)


def xinorm_from_xi_jax(xi):
    return 0.5 * (xi + 1.0)


def split_inputs(z):
    d = CFG["domain"]
    p = p_from_pnorm(z[..., 0])
    xi = xi_from_xinorm(z[..., 1])
    ebar = _map_unit_jax(z[..., 2], float(d["e_parallel_min"]), float(d["e_parallel_max"]), str(d["e_parallel_sampling"]).lower())
    te = _map_unit_jax(z[..., 3], float(d["te_min_ev"]), float(d["te_max_ev"]), str(d["te_sampling"]).lower())
    nD = _map_unit_jax(z[..., 4], float(d["nD_min_m3"]), float(d["nD_max_m3"]), str(d["nD_sampling"]).lower())
    nImp = _map_unit_jax(z[..., 5], float(d["nImp_min_m3"]), float(d["nImp_max_m3"]), str(d["nImp_sampling"]).lower())
    zD = float(d["zDavg_min"]) + z[..., 6] * (float(d["zDavg_max"]) - float(d["zDavg_min"]))
    zImp = float(d["zImpavg_min"]) + z[..., 7] * (float(d["zImpavg_max"]) - float(d["zImpavg_min"]))
    return p, xi, ebar, te, nD, nImp, zD, zImp


# =============================================================================
# Import FV solver and prepare shared atomic data
# =============================================================================

def load_fv_module(config_path: Path):
    solver_path = Path(str(CFG["fv"]["solver_script"]))
    if not solver_path.is_absolute():
        solver_path = config_path.parent / solver_path
    solver_path = solver_path.resolve()
    if not solver_path.exists():
        raise FileNotFoundError(f"FV solver not found: {solver_path}")

    name = "deeprunaway_fv_rpf_solver"
    spec = importlib.util.spec_from_file_location(name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import FV solver: {solver_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, solver_path


def prepare_atomic_tables(fv_module) -> None:
    global IMPURITY_Z, D_I_EV, D_ABAR, IMP_I_EV, IMP_ABAR

    IMPURITY_Z = int(CFG["domain"].get("impurity_Z", 10))
    if IMPURITY_Z not in fv_module.MEAN_EXCITATION_EV_BY_Z:
        raise ValueError(f"FV solver has no screening table for impurity Z={IMPURITY_Z}")

    d_i = list(fv_module.MEAN_EXCITATION_EV_BY_Z[1]) + [1.0]
    d_a = list(fv_module.HESSLOW_ABAR_BY_Z[1]) + [0.0]
    i_i = list(fv_module.MEAN_EXCITATION_EV_BY_Z[IMPURITY_Z]) + [1.0]
    i_a = list(fv_module.HESSLOW_ABAR_BY_Z[IMPURITY_Z]) + [0.0]

    D_I_EV = jnp.asarray(d_i, dtype=jnp.float64)
    D_ABAR = jnp.asarray(d_a, dtype=jnp.float64)
    IMP_I_EV = jnp.asarray(i_i, dtype=jnp.float64)
    IMP_ABAR = jnp.asarray(i_a, dtype=jnp.float64)


# =============================================================================
# JAX physics: same steady adjoint coefficients as the FV solver
# =============================================================================

def charge_weights(zavg, Z: int):
    q = jnp.arange(Z + 1, dtype=jnp.float64)
    return jnp.maximum(0.0, 1.0 - jnp.abs(zavg - q))


def chandrasekhar_phi_psi(x):
    phi = jax.scipy.special.erf(x)
    psi_dir = (phi - 2.0 * x * jnp.exp(-x * x) / jnp.sqrt(jnp.pi)) / (2.0 * x * x)
    z = x * x
    poly = (((((-1.0 / 1560.0 * z + 1.0 / 264.0) * z - 1.0 / 54.0) * z
              + 1.0 / 14.0) * z - 1.0 / 5.0) * z + 1.0 / 3.0)
    psi_ser = (2.0 * x / jnp.sqrt(jnp.pi)) * poly
    psi = jnp.where(jnp.abs(x) <= 0.08, psi_ser, psi_dir)
    return phi, psi


def collision_coefficients_scalar(p, te_ev, nD, nImp, zD, zImp):
    assert D_I_EV is not None and D_ABAR is not None
    assert IMP_I_EV is not None and IMP_ABAR is not None

    qD = jnp.arange(2, dtype=jnp.float64)
    qI = jnp.arange(IMPURITY_Z + 1, dtype=jnp.float64)
    wD = charge_weights(zD, 1)
    wI = charge_weights(zImp, IMPURITY_Z)

    n_state = jnp.concatenate([nD * wD, nImp * wI])
    q_state = jnp.concatenate([qD, qI])
    Z_nuc = jnp.concatenate([
        jnp.ones((2,), dtype=jnp.float64),
        jnp.full((IMPURITY_Z + 1,), float(IMPURITY_Z), dtype=jnp.float64),
    ])
    I_ev = jnp.concatenate([D_I_EV, IMP_I_EV])
    abar = jnp.concatenate([D_ABAR, IMP_ABAR])
    n_bound = Z_nuc - q_state

    ne = jnp.sum(n_state * q_state)
    zeff = jnp.sum(n_state * q_state * q_state) / ne
    ln0 = 14.9 - 0.5 * jnp.log(ne / 1.0e20) + jnp.log(te_ev / 1.0e3)

    theta = te_ev / MEC2_EV
    delta = jnp.sqrt(2.0 * theta)
    gamma = jnp.sqrt(1.0 + p * p)
    gamma_minus_one = p * p / (gamma + 1.0)
    x = p / (delta * gamma)
    phi, psi = chandrasekhar_phi_psi(x)

    q_ee = 2.0 * gamma_minus_one / (delta * delta)
    q_ei = 2.0 * p / delta
    ln_ee = ln0 + jnp.log1p(q_ee ** (0.5 * SIGMA)) / SIGMA
    ln_ei = ln0 + jnp.log1p(q_ei ** SIGMA) / SIGMA
    r_ee = ln_ee / ln0
    r_ei = ln_ei / ln0

    beta2 = p * p / (gamma * gamma)
    Ibar = I_ev / MEC2_EV
    harg = p * jnp.sqrt(jnp.maximum(gamma - 1.0, 0.0)) / Ibar
    h_term = n_bound * (0.2 * jnp.log1p(harg ** 5) - beta2)
    h = jnp.sum((n_state / ne) * h_term)

    y = jnp.power(jnp.maximum(p * abar, 0.0), 1.5)
    g_term = (2.0 / 3.0) * (
        (Z_nuc * Z_nuc - q_state * q_state) * jnp.log1p(y)
        - n_bound * n_bound * y / (1.0 + y)
    )
    g = jnp.sum((n_state / ne) * g_term)

    ee_deflection = phi - psi + delta * delta * p * p / (2.0 * gamma * gamma)
    cf = r_ee * 2.0 * psi / (delta * delta) + (gamma * gamma / (p * p)) * (h / ln0)
    nud = (gamma / (p ** 3)) * (zeff * r_ei + r_ee * ee_deflection + g / ln0)

    B = float(CFG["fv"]["B_T"])
    tau_c = 4.0 * jnp.pi * EPS0**2 * M_E**2 * C_LIGHT**3 / (E_CHARGE**4 * ne * ln0)
    if B == 0.0:
        alpha = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        tau_s = 6.0 * jnp.pi * EPS0 * M_E**3 * C_LIGHT**3 / (E_CHARGE**4 * B * B)
        alpha = tau_c / tau_s

    return cf, nud, alpha, ne, zeff, ln0


# =============================================================================
# Network and steady adjoint residual
# =============================================================================

def init_mlp(key):
    tr = CFG["training"]
    width = int(tr["width"])
    depth = int(tr["depth"])
    dims = [8] + [width] * depth + [1]
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for i, k in enumerate(keys):
        scale = math.sqrt(2.0 / dims[i])
        params.append({
            "W": scale * jax.random.normal(k, (dims[i], dims[i + 1]), dtype=jnp.float64),
            "b": jnp.zeros((dims[i + 1],), dtype=jnp.float64),
        })
    return params


def mlp_raw(params, z):
    h = z
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return (h @ params[-1]["W"] + params[-1]["b"])[..., 0]


def prob_from_norm(params, z):
    raw = mlp_raw(params, z)
    pn = z[..., 0]
    # Hard P=0 at p=pmin and 0 <= P < 1 everywhere.
    return jnp.tanh((pn * pn) * (raw * raw))


prob_apply = jax.vmap(lambda params, z: prob_from_norm(params, z), in_axes=(None, 0))


def pde_operator_coeff_single(z):
    """Precompute coefficients multiplying normalized NN derivatives.

    For fixed collocation points the plasma state and collision coefficients do
    not depend on the network parameters, so this work is done once per sampled
    PDE batch instead of once per optimizer iteration.
    """
    p, xi, ebar, te, nD, nImp, zD, zImp = split_inputs(z)
    cf, nud, alpha, _ne, _zeff, _ln0 = collision_coefficients_scalar(
        p, te, nD, nImp, zD, zImp
    )
    gamma = jnp.sqrt(1.0 + p * p)
    U_p = -ebar * xi - cf - alpha * gamma * p * (1.0 - xi * xi)

    # Physical-coordinate residual:
    #   a_p P_p + a_xi P_xi + a_xixi P_xixi.
    a_p = -U_p
    a_xi = (1.0 - xi * xi) * (ebar / p - alpha * xi / gamma) + nud * xi
    a_xixi = -0.5 * nud * (1.0 - xi * xi)

    # Convert coefficients to derivatives with respect to normalized network
    # coordinates p_norm and xi_norm. p_norm is logarithmic in p and
    # xi_norm=(xi+1)/2.
    log_prange = math.log(float(CFG["fv"]["pmax"]) / float(CFG["fv"]["pmin"]))
    scale = float(CFG["loss"].get("residual_scale", 1.0))
    return jnp.stack([
        a_p / (p * log_prange),
        0.5 * a_xi,
        0.25 * a_xixi,
    ]) / scale


pde_operator_coeff_apply = jax.jit(jax.vmap(pde_operator_coeff_single))


def residual_single(params, z, coeff):
    """Steady residual using precomputed operator coefficients.

    One gradient supplies P_pnorm and P_xinorm; one JVP supplies only the
    required second xi-normalized derivative.
    """
    phase = z[:2]

    def f_phase(q):
        zz = z.at[0].set(q[0]).at[1].set(q[1])
        return prob_from_norm(params, zz)

    grad_phase = jax.grad(f_phase)(phase)
    xi_dir = jnp.asarray([0.0, 1.0], dtype=z.dtype)
    _, P_xinxin = jax.jvp(
        lambda q: jax.grad(f_phase)(q)[1],
        (phase,),
        (xi_dir,),
    )

    return (
        coeff[0] * grad_phase[0]
        + coeff[1] * grad_phase[1]
        + coeff[2] * P_xinxin
    )


residual_apply = jax.vmap(residual_single, in_axes=(None, 0, 0))


def success_mask_single(z):
    _p, xi, ebar, te, nD, nImp, zD, zImp = split_inputs(z)
    pmax = jnp.asarray(float(CFG["fv"]["pmax"]), dtype=jnp.float64)
    cf, _nud, alpha, _ne, _zeff, _ln0 = collision_coefficients_scalar(
        pmax, te, nD, nImp, zD, zImp
    )
    gamma = jnp.sqrt(1.0 + pmax * pmax)
    U_p = -ebar * xi - cf - alpha * gamma * pmax * (1.0 - xi * xi)
    return U_p > 0.0


success_mask_apply = jax.jit(jax.vmap(success_mask_single))


# =============================================================================
# FV dataset generation
# =============================================================================

def generate_fv_dataset(fv, dataset_path: Path):
    fvc = CFG["fv"]
    dom = CFG["domain"]
    n_train = int(fvc["n_train_cases"])
    n_val = int(fvc["n_validation_cases"])
    n_cases = n_train + n_val

    theta_norm = sobol_unit(n_cases, 6, int(fvc["parameter_seed"]))
    theta_phys = theta_phys_from_unit_np(theta_norm)

    base_cfg = fv.SolverConfig(
        te_eV=float(theta_phys[0, 1]),
        E_over_Ec=float(theta_phys[0, 0]),
        B_T=float(fvc["B_T"]),
        species=(
            fv.AtomicSpecies("D", 1, float(theta_phys[0, 4]), float(theta_phys[0, 2])),
            fv.AtomicSpecies(str(dom.get("impurity_name", "Ne")), int(dom.get("impurity_Z", 10)), float(theta_phys[0, 5]), float(theta_phys[0, 3])),
        ),
        energy_diffusion=False,
        solve_mode="steady",
        Np=int(fvc["Np"]),
        Nxi=int(fvc["Nxi"]),
        pmin=float(fvc["pmin"]),
        pmax=float(fvc["pmax"]),
        device=str(fvc["device"]),
        write_output=False,
        save_every=0,
    )
    fv.validate_rpf_config(base_cfg)
    grid = fv.build_grid(base_cfg, fv.derive_physics(base_cfg))

    print("initializing FV Warp/cuDSS runtime ...", flush=True)
    gpu_context = fv.load_gpu_runtime(base_cfg.device)
    print("building reusable FV CSR topology ...", flush=True)
    topo, topo_timing = fv.build_local_csr_topology_gpu(grid, base_cfg.device)
    print(f"FV topology total={topo_timing['total_s']:.3e} s")

    P = np.empty((n_cases, grid.Np, grid.Nxi), dtype=np.float64)
    case_info = []

    t0 = time.perf_counter()
    for i in range(n_cases):
        ebar, te, nD, nImp, zD, zImp = theta_phys[i]
        cfg = replace(
            base_cfg,
            te_eV=float(te),
            E_over_Ec=float(ebar),
            species=(
                fv.AtomicSpecies("D", 1, float(zD), float(nD)),
                fv.AtomicSpecies(str(dom.get("impurity_name", "Ne")), int(dom.get("impurity_Z", 10)), float(zImp), float(nImp)),
            ),
        )
        fv.validate_rpf_config(cfg)
        phys = fv.derive_physics(cfg)
        coll = fv.build_collision_data(cfg, phys, grid)
        fv.validate_collision_data(coll)
        result = fv.solve_adjoint_rpf(cfg, phys, grid, coll, topo, gpu_context)
        P[i] = np.asarray(result["P_steady_direct"], dtype=np.float64)
        case_info.append({
            "index": i,
            "split": "train" if i < n_train else "validation",
            "ne_m3": float(phys.free_density_from_ions_m3),
            "zeff": float(phys.z_eff),
            "steady_residual": float(result["steady_linear_residual"]),
            "success_pitch_cells": int(result["steady_target"]["success_pitch_cells"]),
        })
        print(
            f"FV {i+1:4d}/{n_cases}: split={case_info[-1]['split']:10s} "
            f"E/Ec={ebar:.5g} Te={te:.5g} nD={nD:.3e} nImp={nImp:.3e} "
            f"<ZD>={zD:.4g} <ZImp>={zImp:.4g} "
            f"res={case_info[-1]['steady_residual']:.2e}"
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_path,
        theta_norm=theta_norm,
        theta_phys=theta_phys,
        p=np.asarray(grid.p_centers, dtype=np.float64),
        xi=np.asarray(grid.xi_centers, dtype=np.float64),
        P=P,
        n_train_cases=np.asarray(n_train, dtype=np.int64),
        case_info_json=np.asarray(json.dumps(case_info)),
    )
    print(f"wrote FV dataset: {dataset_path}")
    print(f"FV data generation wall={time.perf_counter()-t0:.3f} s")


def load_fv_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as f:
        out = {k: np.asarray(f[k]) for k in f.files}
    return out


# =============================================================================
# Fixed training batches
# =============================================================================

class TrainBatch(NamedTuple):
    pde_z: jnp.ndarray
    pde_c: jnp.ndarray
    pde_w: jnp.ndarray
    bc_z: jnp.ndarray
    bc_w: jnp.ndarray
    fv_z: jnp.ndarray
    fv_y: jnp.ndarray
    fv_w: jnp.ndarray


def _pad_to_chunk(z: np.ndarray, chunk: int, y: np.ndarray | None = None, w: np.ndarray | None = None):
    n = z.shape[0]
    if n <= 0:
        raise ValueError("cannot pad an empty training set")
    if w is None:
        w = np.ones(n, dtype=np.float64)
    if chunk <= 0:
        if y is None:
            return z, w
        return z, y, w
    n_pad = (-n) % chunk
    if n_pad:
        z = np.concatenate([z, np.repeat(z[-1:, :], n_pad, axis=0)], axis=0)
        w = np.concatenate([w, np.zeros(n_pad, dtype=np.float64)])
        if y is not None:
            y = np.concatenate([y, np.repeat(y[-1:], n_pad)], axis=0)
    if y is None:
        return z, w
    return z, y, w


def build_fixed_training_batch(dataset: dict[str, np.ndarray]) -> TrainBatch:
    tr = CFG["training"]
    n_train_cases = int(np.asarray(dataset["n_train_cases"]).item())
    theta_norm = np.asarray(dataset["theta_norm"], dtype=np.float64)
    p = np.asarray(dataset["p"], dtype=np.float64)
    xi = np.asarray(dataset["xi"], dtype=np.float64)
    P = np.asarray(dataset["P"], dtype=np.float64)

    # Fixed continuous 8D PDE collocation set.
    z_pde = sobol_unit(int(tr["n_pde"]), 8, int(tr["sampling_seed"]))
    eps = float(tr.get("interior_eps", 1.0e-6))
    z_pde[:, 0] = eps + (1.0 - 2.0 * eps) * z_pde[:, 0]
    z_pde[:, 1] = eps + (1.0 - 2.0 * eps) * z_pde[:, 1]

    # Fixed p=pmax boundary candidates across full parameter space and pitch.
    z_bc = sobol_unit(int(tr["n_bc"]), 8, int(tr["sampling_seed"]) + 1)
    z_bc[:, 0] = 1.0
    bc_active = np.asarray(jax.device_get(success_mask_apply(jnp.asarray(z_bc))), dtype=np.float64)
    print(f"high-p successful BC fraction={float(np.mean(bc_active)):.6f}")

    # Fixed FV labels from training parameter cases only.
    rng = np.random.default_rng(int(tr["sampling_seed"]) + 2)
    pp, xx = np.meshgrid(p, xi, indexing="ij")
    phase_pn = pnorm_from_p_np(pp.ravel())
    phase_xn = xinorm_from_xi_np(xx.ravel())
    ncell = phase_pn.size
    per_case = int(tr["fv_points_per_case"])
    if per_case <= 0 or per_case >= ncell:
        indices = np.arange(ncell, dtype=np.int64)
    else:
        indices = None

    z_fv_parts = []
    y_fv_parts = []
    for i in range(n_train_cases):
        idx = indices if indices is not None else rng.choice(ncell, size=per_case, replace=False)
        nsel = idx.size
        zc = np.empty((nsel, 8), dtype=np.float64)
        zc[:, 0] = phase_pn[idx]
        zc[:, 1] = phase_xn[idx]
        zc[:, 2:] = theta_norm[i][None, :]
        z_fv_parts.append(zc)
        y_fv_parts.append(P[i].ravel()[idx])

    z_fv = np.concatenate(z_fv_parts, axis=0)
    y_fv = np.concatenate(y_fv_parts, axis=0)

    z_pde, w_pde = _pad_to_chunk(z_pde, int(tr["pde_chunk"]))
    z_bc, w_bc = _pad_to_chunk(z_bc, int(tr["bc_chunk"]), w=bc_active)
    z_fv, y_fv, w_fv = _pad_to_chunk(z_fv, int(tr["fv_chunk"]), y=y_fv)

    pde_z = jnp.asarray(z_pde)
    t_coeff = time.perf_counter()
    pde_c = pde_operator_coeff_apply(pde_z)
    pde_c.block_until_ready()
    print(f"precomputed PDE operator coefficients in {time.perf_counter()-t_coeff:.3f} s")

    print(
        f"fixed training sets: PDE={int(np.sum(w_pde)):,}, "
        f"BC active={int(np.sum(w_bc)):,}/{int(tr['n_bc']):,}, "
        f"FV={int(np.sum(w_fv)):,}"
    )

    return TrainBatch(
        pde_z=pde_z, pde_c=pde_c, pde_w=jnp.asarray(w_pde),
        bc_z=jnp.asarray(z_bc), bc_w=jnp.asarray(w_bc),
        fv_z=jnp.asarray(z_fv), fv_y=jnp.asarray(y_fv), fv_w=jnp.asarray(w_fv),
    )


def resample_pde_bc(batch: TrainBatch, seed: int) -> TrainBatch:
    """Replace only PDE/BC collocation points; keep FV labels fixed.

    Shapes are unchanged, so the compiled loss/optimizer kernels are reused.
    """
    tr = CFG["training"]
    eps = float(tr.get("interior_eps", 1.0e-6))

    z_pde = sobol_unit(int(tr["n_pde"]), 8, int(seed))
    z_pde[:, 0] = eps + (1.0 - 2.0 * eps) * z_pde[:, 0]
    z_pde[:, 1] = eps + (1.0 - 2.0 * eps) * z_pde[:, 1]

    z_bc = sobol_unit(int(tr["n_bc"]), 8, int(seed) + 1)
    z_bc[:, 0] = 1.0
    bc_active = np.asarray(
        jax.device_get(success_mask_apply(jnp.asarray(z_bc))), dtype=np.float64
    )

    z_pde, w_pde = _pad_to_chunk(z_pde, int(tr["pde_chunk"]))
    z_bc, w_bc = _pad_to_chunk(z_bc, int(tr["bc_chunk"]), w=bc_active)

    pde_z = jnp.asarray(z_pde)
    t_coeff = time.perf_counter()
    pde_c = pde_operator_coeff_apply(pde_z)
    pde_c.block_until_ready()

    print(
        f"resampled PDE/BC: PDE={int(np.sum(w_pde)):,}, "
        f"BC active={int(np.sum(w_bc)):,}/{int(tr['n_bc']):,}, "
        f"coeff_s={time.perf_counter()-t_coeff:.3f}"
    )

    return TrainBatch(
        pde_z=pde_z, pde_c=pde_c, pde_w=jnp.asarray(w_pde),
        bc_z=jnp.asarray(z_bc), bc_w=jnp.asarray(w_bc),
        fv_z=batch.fv_z, fv_y=batch.fv_y, fv_w=batch.fv_w,
    )


# =============================================================================
# XLA-chunked full-batch objective
# =============================================================================

def _pde_mse(params, z, c, w):
    chunk = int(CFG["training"]["pde_chunk"])
    if chunk <= 0:
        r = residual_apply(params, z, c)
        return jnp.sum(w * r * r) / jnp.maximum(jnp.sum(w), 1.0)

    zc = z.reshape((-1, chunk, 8))
    cc = c.reshape((-1, chunk, 3))
    wc = w.reshape((-1, chunk))

    def body(total, x):
        zz, coeff, ww = x
        r = residual_apply(params, zz, coeff)
        return total + jnp.sum(ww * r * r), None

    total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), (zc, cc, wc))
    return total / jnp.maximum(jnp.sum(w), 1.0)


def _bc_mse(params, z, w):
    chunk = int(CFG["training"]["bc_chunk"])
    if chunk <= 0:
        pred = prob_apply(params, z)
        return jnp.sum(w * (pred - 1.0) ** 2) / jnp.maximum(jnp.sum(w), 1.0)

    zc = z.reshape((-1, chunk, 8))
    wc = w.reshape((-1, chunk))

    def body(total, x):
        zz, ww = x
        pred = prob_apply(params, zz)
        return total + jnp.sum(ww * (pred - 1.0) ** 2), None

    total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), (zc, wc))
    return total / jnp.maximum(jnp.sum(w), 1.0)


def _fv_mse(params, z, y, w):
    chunk = int(CFG["training"]["fv_chunk"])
    if chunk <= 0:
        pred = prob_apply(params, z)
        return jnp.sum(w * (pred - y) ** 2) / jnp.maximum(jnp.sum(w), 1.0)

    zc = z.reshape((-1, chunk, 8))
    yc = y.reshape((-1, chunk))
    wc = w.reshape((-1, chunk))

    def body(total, x):
        zz, yy, ww = x
        pred = prob_apply(params, zz)
        return total + jnp.sum(ww * (pred - yy) ** 2), None

    total, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), (zc, yc, wc))
    return total / jnp.maximum(jnp.sum(w), 1.0)


def loss_terms(params, batch: TrainBatch):
    pde = _pde_mse(params, batch.pde_z, batch.pde_c, batch.pde_w)
    bc = _bc_mse(params, batch.bc_z, batch.bc_w)
    fv_loss = _fv_mse(params, batch.fv_z, batch.fv_y, batch.fv_w)
    lc = CFG["loss"]
    total = float(lc["w_pde"]) * pde + float(lc["w_bc"]) * bc + float(lc["w_fv"]) * fv_loss
    return total, jnp.stack([total, pde, bc, fv_loss])


@jax.jit
def eval_terms(params, batch):
    return loss_terms(params, batch)[1]


def make_adam_kernels():
    tr = CFG["training"]
    optimizer = optax.adam(float(tr["adam_lr"]))

    @jax.jit
    def step(params, state, batch):
        def objective(p):
            total, _ = loss_terms(p, batch)
            return total
        loss, grads = jax.value_and_grad(objective)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    @partial(jax.jit, static_argnames=("n_steps",))
    def block(params, state, batch, n_steps: int):
        def body(carry, _):
            p, s = carry
            p, s, _loss = step(p, s, batch)
            return (p, s), None
        (params, state), _ = jax.lax.scan(body, (params, state), xs=None, length=n_steps)
        return params, state

    return optimizer, block


# =============================================================================
# Adam -> full-batch SSBroyden
# =============================================================================

def train_adam(params, batch):
    tr = CFG["training"]
    steps = int(tr["adam_steps"])
    if steps == 0:
        return params, []

    optimizer, adam_block = make_adam_kernels()
    state = optimizer.init(params)
    block_steps = int(tr["adam_block_steps"])
    history = []

    print("compiling/running Adam full-batch blocks ...", flush=True)
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        params, state = adam_block(params, state, batch, block_steps)
        jax.block_until_ready(params)
        done += block_steps
        terms = np.asarray(jax.device_get(eval_terms(params, batch)), dtype=np.float64)
        history.append({"phase": "adam", "step": done, "total": float(terms[0]), "pde": float(terms[1]), "bc": float(terms[2]), "fv": float(terms[3])})
        print(
            f"Adam {done:7d}/{steps}: total={terms[0]:.4e} "
            f"pde={terms[1]:.4e} bc={terms[2]:.4e} fv={terms[3]:.4e}"
        )
    print(f"Adam wall={time.perf_counter()-t0:.3f} s")
    return params, history


def make_ssb_solver():
    tr = CFG["training"]

    class SSBroyden(optx.AbstractSSBroyden):
        rtol: float
        atol: float
        norm: Callable = optx.max_norm
        use_inverse: bool = True
        search: optx.AbstractSearch = eqx.field(default_factory=optx.BacktrackingStrongWolfe)
        descent: optx.AbstractDescent = eqx.field(default_factory=optx.NewtonDescent)
        verbose: frozenset[str] = frozenset()

    search_name = str(tr["ssb_search"]).lower()
    if search_name == "wolfe":
        search = optx.BacktrackingStrongWolfe()
    elif search_name == "zoom":
        search = optx.Zoom()
    elif search_name == "trust_region":
        search = optx.LinearTrustRegion()
    else:  # guarded by load_config
        raise ValueError(f"unknown SSBroyden search: {search_name}")

    solver = SSBroyden(
        rtol=float(tr["ssb_rtol"]),
        atol=float(tr["ssb_atol"]),
        search=search,
        descent=optx.NewtonDescent(),
        verbose=frozenset(),
    )
    return optx.BestSoFarMinimiser(solver)


def train_ssbroyden(params, batch):
    tr = CFG["training"]
    solver = make_ssb_solver()
    weights, unflatten = ravel_pytree(params)
    n_params = int(weights.size)
    n_blocks = int(tr["ssb_blocks"])
    block_iters = int(tr["ssb_block_iters"])
    resample_every = int(tr["ssb_resample_every_blocks"])
    search_name = str(tr["ssb_search"]).lower()

    print(
        f"SSBroyden: parameters={n_params:,}, dense inverse-H estimate="
        f"{n_params*n_params*8/1e9:.3f} GB, blocks={n_blocks}, "
        f"block_iters={block_iters}, search={search_name}, "
        f"resample_every_blocks={resample_every}"
    )

    def scalar_loss(w, b):
        return loss_terms(unflatten(w), b)[0]

    @eqx.filter_jit
    def solve_block(w0, b):
        before = scalar_loss(w0, b)
        sol = optx.minimise(
            scalar_loss,
            solver,
            w0,
            args=b,
            max_steps=block_iters,
            throw=False,
        )
        after = scalar_loss(sol.value, b)
        delta = jnp.linalg.norm(sol.value - w0)
        finite = jnp.all(jnp.isfinite(sol.value)) & jnp.isfinite(after)
        return sol.value, before, after, delta, finite, sol.stats["num_steps"]

    print("compiling/running blockwise full-batch SSBroyden ...", flush=True)
    history = []
    total_steps = 0
    total_wall = 0.0
    base_seed = int(tr["sampling_seed"]) + 100000
    current_batch = batch

    for block in range(1, n_blocks + 1):
        # A new optx.minimise call starts a fresh SSBroyden inverse-Hessian
        # approximation. Optional resampling changes only the PDE/BC points;
        # the supervised FV data remain fixed throughout training.
        if block > 1 and resample_every > 0 and (block - 1) % resample_every == 0:
            current_batch = resample_pde_bc(
                current_batch,
                seed=base_seed + 2 * (block - 1),
            )

        t0 = time.perf_counter()
        w_new, before, after, delta, finite, nsteps = solve_block(weights, current_batch)
        w_new.block_until_ready()
        wall = time.perf_counter() - t0
        total_wall += wall

        before, after, delta, finite, nsteps = jax.device_get(
            (before, after, delta, finite, nsteps)
        )
        weights = w_new
        total_steps += int(nsteps)
        params = unflatten(weights)
        terms = np.asarray(jax.device_get(eval_terms(params, current_batch)), dtype=np.float64)

        rec = {
            "phase": "ssbroyden",
            "block": block,
            "step": int(tr["adam_steps"]) + total_steps,
            "ssb_step": total_steps,
            "block_steps": int(nsteps),
            "resampled": bool(block > 1 and resample_every > 0 and (block - 1) % resample_every == 0),
            "total": float(terms[0]),
            "pde": float(terms[1]),
            "bc": float(terms[2]),
            "fv": float(terms[3]),
            "block_loss_before": float(before),
            "block_loss_after": float(after),
            "block_delta_weights": float(delta),
            "block_finite": bool(finite),
            "wall_s": wall,
        }
        history.append(rec)
        print(
            f"SSB block {block:3d}/{n_blocks}: steps={int(nsteps):5d} "
            f"loss={float(before):.4e}->{float(after):.4e} "
            f"final[pde,bc,fv]=[{terms[1]:.4e},{terms[2]:.4e},{terms[3]:.4e}] "
            f"resampled={int(rec['resampled'])} wall={wall:.3f} s"
        )

    print(f"SSBroyden total steps={total_steps} total wall={total_wall:.3f} s")
    return unflatten(weights), history


# =============================================================================
# Validation and persistence
# =============================================================================

def predict_array(params, z_np: np.ndarray, chunk: int) -> np.ndarray:
    @jax.jit
    def pred(z):
        return prob_apply(params, z)

    out = np.empty(z_np.shape[0], dtype=np.float64)
    for i in range(0, z_np.shape[0], chunk):
        j = min(i + chunk, z_np.shape[0])
        out[i:j] = np.asarray(jax.device_get(pred(jnp.asarray(z_np[i:j]))), dtype=np.float64)
    return out


def validate_heldout(params, dataset: dict[str, np.ndarray]):
    tr = CFG["training"]
    n_train = int(np.asarray(dataset["n_train_cases"]).item())
    theta_norm = np.asarray(dataset["theta_norm"], dtype=np.float64)
    p = np.asarray(dataset["p"], dtype=np.float64)
    xi = np.asarray(dataset["xi"], dtype=np.float64)
    P = np.asarray(dataset["P"], dtype=np.float64)

    pp, xx = np.meshgrid(p, xi, indexing="ij")
    pn = pnorm_from_p_np(pp.ravel())
    xn = xinorm_from_xi_np(xx.ravel())
    metrics = []

    for i in range(n_train, P.shape[0]):
        z = np.empty((pn.size, 8), dtype=np.float64)
        z[:, 0] = pn
        z[:, 1] = xn
        z[:, 2:] = theta_norm[i][None, :]
        pred = predict_array(params, z, int(tr["validation_chunk"]))
        ref = P[i].ravel()
        diff = pred - ref
        rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-300))
        max_abs = float(np.max(np.abs(diff)))
        metrics.append({"case": i, "rel_l2": rel_l2, "max_abs": max_abs})
        print(f"validation case {i}: rel_l2={rel_l2:.4e} max_abs={max_abs:.4e}")
    return metrics


def save_params(params, path: Path):
    payload = {}
    for i, layer in enumerate(params):
        payload[f"W_{i}"] = np.asarray(jax.device_get(layer["W"]))
        payload[f"b_{i}"] = np.asarray(jax.device_get(layer["b"]))
    np.savez(path, **payload)


def physics_parity_check(fv) -> None:
    """One small startup check that JAX PINN coefficients match the FV model."""
    d = CFG["domain"]
    fvc = CFG["fv"]
    vals = np.array([
        0.5 * (float(d["e_parallel_min"]) + float(d["e_parallel_max"])),
        math.sqrt(float(d["te_min_ev"]) * float(d["te_max_ev"])),
        math.sqrt(float(d["nD_min_m3"]) * float(d["nD_max_m3"])),
        math.sqrt(float(d["nImp_min_m3"]) * float(d["nImp_max_m3"])),
        0.5 * (float(d["zDavg_min"]) + float(d["zDavg_max"])),
        0.5 * (float(d["zImpavg_min"]) + float(d["zImpavg_max"])),
    ], dtype=np.float64)
    ebar, te, nD, nImp, zD, zImp = vals
    cfg = fv.SolverConfig(
        te_eV=float(te), E_over_Ec=float(ebar), B_T=float(fvc["B_T"]),
        species=(
            fv.AtomicSpecies("D", 1, float(zD), float(nD)),
            fv.AtomicSpecies(str(d.get("impurity_name", "Ne")), int(d.get("impurity_Z", 10)), float(zImp), float(nImp)),
        ),
        energy_diffusion=False, solve_mode="steady",
        Np=8, Nxi=8, pmin=float(fvc["pmin"]), pmax=float(fvc["pmax"]), write_output=False,
    )
    phys = fv.derive_physics(cfg)
    p = np.geomspace(float(fvc["pmin"]) * 1.1, float(fvc["pmax"]) * 0.9, 5)
    fv_cf, _fv_ca, fv_nud, *_ = fv.fully_relativistic_asymptotically_matched_collision_coefficients(p, cfg, phys)

    jax_out = jax.vmap(lambda pp: collision_coefficients_scalar(pp, te, nD, nImp, zD, zImp))(jnp.asarray(p))
    j_cf, j_nud, j_alpha = [np.asarray(jax.device_get(x)) for x in jax_out[:3]]
    rel_cf = np.linalg.norm(j_cf - fv_cf) / max(np.linalg.norm(fv_cf), 1e-300)
    rel_nud = np.linalg.norm(j_nud - fv_nud) / max(np.linalg.norm(fv_nud), 1e-300)
    rel_alpha = abs(float(j_alpha[0]) - phys.alpha) / max(abs(phys.alpha), 1.0) if np.ndim(j_alpha) else abs(float(j_alpha) - phys.alpha) / max(abs(phys.alpha), 1.0)
    print(f"PINN/FV coefficient parity: C_F={rel_cf:.3e} nu_D={rel_nud:.3e} alpha={rel_alpha:.3e}")
    if rel_cf > 5e-11 or rel_nud > 5e-11 or rel_alpha > 5e-11:
        raise RuntimeError("PINN/FV coefficient parity check failed")


# =============================================================================
# Main
# =============================================================================

def main():
    global CFG
    parser = argparse.ArgumentParser(description="GPU FV-assisted PINN trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="TOML training file (default: %(default)s)",
    )
    args = parser.parse_args()
    CFG, config_path = load_config(args.config)
    fv, fv_path = load_fv_module(config_path)
    prepare_atomic_tables(fv)

    outdir = Path(str(CFG["output"]["directory"]))
    if not outdir.is_absolute():
        outdir = config_path.parent / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(str(CFG["fv"]["dataset_file"]))
    if not dataset_path.is_absolute():
        dataset_path = config_path.parent / dataset_path

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print(f"FV solver: {fv_path}")
    print(f"dataset: {dataset_path}")
    print("training objective: PDE + successful-pmax BC + FV labels")
    tr = CFG["training"]
    def _chunk_desc(name):
        n = int(tr[name])
        return "whole batch" if n <= 0 else f"chunks of {n:,}"
    print(
        "loss evaluation: "
        f"PDE={_chunk_desc('pde_chunk')}, "
        f"BC={_chunk_desc('bc_chunk')}, "
        f"FV={_chunk_desc('fv_chunk')}"
    )
    print(
        "optimizer: Adam -> blockwise full-batch SSBroyden "
        f"(search={CFG['training']['ssb_search']}, "
        f"blocks={CFG['training']['ssb_blocks']}x{CFG['training']['ssb_block_iters']}, "
        f"resample_every={CFG['training']['ssb_resample_every_blocks']})"
    )

    physics_parity_check(fv)

    if bool(CFG["fv"]["generate"]):
        generate_fv_dataset(fv, dataset_path)
    elif not dataset_path.exists():
        raise FileNotFoundError(f"FV dataset does not exist: {dataset_path}")

    dataset = load_fv_dataset(dataset_path)
    batch = build_fixed_training_batch(dataset)

    key = jax.random.PRNGKey(int(CFG["training"]["seed"]))
    params = init_mlp(key)

    # First full objective evaluation triggers the main XLA compilation.
    print("compiling full-batch loss ...", flush=True)
    t0 = time.perf_counter()
    initial = np.asarray(jax.device_get(eval_terms(params, batch)), dtype=np.float64)
    print(
        f"initial loss: total={initial[0]:.4e} pde={initial[1]:.4e} "
        f"bc={initial[2]:.4e} fv={initial[3]:.4e} compile/eval={time.perf_counter()-t0:.3f} s"
    )

    params, hist_adam = train_adam(params, batch)
    params, hist_ssb = train_ssbroyden(params, batch)
    validation = validate_heldout(params, dataset)

    params_path = outdir / "params_parametric_rpf.npz"
    history_path = outdir / "training_history.json"
    summary_path = outdir / "training_summary.json"
    save_params(params, params_path)

    history = hist_adam + hist_ssb
    history_path.write_text(json.dumps(history, indent=2))
    summary = {
        "fv_solver": str(fv_path),
        "dataset": str(dataset_path),
        "initial_loss": {"total": float(initial[0]), "pde": float(initial[1]), "bc": float(initial[2]), "fv": float(initial[3])},
        "validation": validation,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    shutil.copyfile(config_path, outdir / config_path.name)

    print(f"wrote {params_path}")
    print(f"wrote {history_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
