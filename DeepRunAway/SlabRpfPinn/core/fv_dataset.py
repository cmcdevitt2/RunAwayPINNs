"""Parallel CPU FV label generation for adaptive-p_min PINN datasets.

Case columns always follow ``E/Ec, Te_eV, nD_m3, nNe_m3, zD, zNe``.
Saved fields use ``(case, p, xi)`` order and remain outside JAX autodiff.
"""

from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from joblib import Parallel, delayed
from scipy import constants, sparse
from scipy.sparse.linalg import spsolve

from core.rpf_fv_cpu import (
    GridConfig, PlasmaConfig, build_collision_data, build_grid, derive_plasma,
    find_up_zero_momentum, ion_species_from_charge_state_densities,
    assemble_fp_operator, physical_adjoint_operator, momentum_from_kinetic_energy,
    radial_boundary_rates, collision_coefficients,
)


PROBABILITY_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class FvDatasetConfig:
    """Settings that determine generated CPU-FV training data."""
    n_cases: int = 64
    p_max: float = 15.0
    B_T: float = 5.0
    fv_Np: int = 512
    fv_Nxi: int = 128
    fv_p_stride: int = 2
    fv_xi_stride: int = 4
    fv_p_min: float = 0.002
    fv_p_coarse_N: int = 32
    n_jobs: int = 1
    seed: int = 2026
    candidate_oversample: float = 1.25

    def __post_init__(self):
        if (self.n_cases <= 0 or self.p_max <= 0.0 or self.B_T < 0.0
                or self.fv_Np <= 0 or self.fv_Nxi <= 0
                or self.fv_p_stride <= 0 or self.fv_xi_stride <= 0
                or self.fv_p_min <= 0.0 or self.fv_p_min >= self.p_max
                or self.fv_p_coarse_N < 1
                or self.n_jobs < -1 or self.n_jobs == 0
                or self.candidate_oversample < 1.0):
            raise ValueError("invalid FV dataset setting")


def denormalize_parameter_cases(values, parameter_domain):
    """Map unit Sobol samples to physical six-dimensional cases."""
    values = np.asarray(values, dtype=np.float64)
    cases = np.empty_like(values)
    for column, (lo, hi, scale) in enumerate(parameter_domain.values()):
        cases[:, column] = (
            np.exp(np.log(lo) + values[:, column] * (np.log(hi) - np.log(lo)))
            if scale == "log" else lo + values[:, column] * (hi - lo))
    return cases


def _charge_state_densities(total, mean_charge, Z):
    """Interpolate total density between neighboring integer charge states."""
    q = np.arange(Z + 1, dtype=float)
    return total * np.maximum(0.0, 1.0 - np.abs(mean_charge - q))


def _prepend_zero_tail(p_center, P, p_min_global, N_p_coarse):
    """Prepend a coarse zero-P tail so every case spans [p_min_global, p_max].

    RPF is guaranteed zero below the dense grid's adaptive floor, so the tail
    needs no solve: it is filled analytically.
    """
    p_coarse = np.geomspace(p_min_global, p_center[0], N_p_coarse + 1)[:-1]
    p_full = np.concatenate((p_coarse, p_center))
    P_full = np.vstack((np.zeros((N_p_coarse, P.shape[1]), dtype=np.float64), P))
    return p_full, P_full


def solve_cpu_case(case, *, p_max, Np, Nxi, B_T, p_min_global, N_p_coarse,
                    p_scan_min=1.0e-12):
    """Solve one normalized-E D/Ne case on its adaptive dense log-p grid.

    The returned grid always spans the prescribed [p_min_global, p_max]: cells
    below the adaptive dense floor are filled with the analytically known
    zero probability rather than solved.
    """
    # The solver receives physical case values, not normalized Sobol values.
    case_start = time.perf_counter()
    ebar, te_eV, nD, nNe, zD, zNe = map(float, case)
    D = ion_species_from_charge_state_densities(1, _charge_state_densities(nD, zD, 1))
    Ne = ion_species_from_charge_state_densities(10, _charge_state_densities(nNe, zNe, 10))
    plasma_base_cfg = PlasmaConfig(te_eV, 0.0, B_T, (D, Ne))
    plasma_base = derive_plasma(plasma_base_cfg)
    E_parallel = ebar * constants.m_e * constants.c / (constants.e * plasma_base.tau_c)
    plasma_cfg = PlasmaConfig(te_eV, E_parallel, B_T, (D, Ne))
    plasma = derive_plasma(plasma_cfg)
    cf_max, _, _ = collision_coefficients(np.asarray([p_max]), plasma_cfg, plasma)
    up_max_minus_one = float(plasma.E_bar - cf_max[0])
    template = GridConfig(p_min=p_scan_min, p_max=p_max, N_p=Np, N_xi=Nxi)
    p_cap = 0.75 * p_max
    # Rejecting this branch later keeps the training set focused on nonzero
    # runaway probabilities while still returning a shape-compatible result.
    if up_max_minus_one <= 0.0:
        # No outward drift exists at the successful boundary: RPF is zero.
        p_dense_min = min(p_cap, max(p_scan_min, p_max * 1.0e-6))
        grid = build_grid(GridConfig(p_min=p_dense_min, p_max=p_max, N_p=Np, N_xi=Nxi))
        P = np.zeros((Np, Nxi), dtype=np.float64)
        p_full, P_full = _prepend_zero_tail(
            grid.p_center, P, p_min_global, N_p_coarse)
        return {
            "case": np.asarray(case, dtype=np.float64), "p": p_full,
            "xi": grid.xi_center, "P": P_full, "p_dense_min": p_dense_min,
            "p_zero": np.nan, "trivial_zero": True,
            "valid": True, "probability_min": 0.0, "probability_max": 0.0,
            "up_max_minus_one": up_max_minus_one, "linear_residual": 0.0,
            "runtime_seconds": time.perf_counter() - case_start,
        }
    # Resolve the suprathermal U_p=0 root, then solve only above half its
    # kinetic-energy threshold. Lower momenta receive an analytic zero tail.
    p_zero = find_up_zero_momentum(template, plasma_cfg, plasma)
    energy_zero = np.sqrt(1.0 + p_zero**2) - 1.0
    p_dense_min_uncapped = momentum_from_kinetic_energy(0.5 * energy_zero)
    p_dense_min = min(p_dense_min_uncapped, p_cap)
    grid = build_grid(GridConfig(p_min=p_dense_min, p_max=p_max, N_p=Np, N_xi=Nxi))
    coll = build_collision_data(grid, plasma_cfg, plasma)
    L = assemble_fp_operator(grid, plasma, coll)
    failure, escape = radial_boundary_rates(grid, plasma, coll)
    A = -physical_adjoint_operator(L, grid)
    solve_start = time.perf_counter()
    P = spsolve(A, escape).reshape(Np, Nxi)
    solve_seconds = time.perf_counter() - solve_start
    residual = A @ P.ravel() - escape
    probability_min = float(np.min(P))
    probability_max = float(np.max(P))
    valid = bool(
        np.all(np.isfinite(P))
        and np.isfinite(residual).all()
        and probability_min >= -PROBABILITY_TOLERANCE
        and probability_max <= 1.0 + PROBABILITY_TOLERANCE)
    p_full, P_full = _prepend_zero_tail(
        grid.p_center, P, p_min_global, N_p_coarse)
    return {
        "case": np.asarray(case, dtype=np.float64),
        "p": p_full,
        "xi": grid.xi_center,
        "P": P_full,
        "p_dense_min": p_dense_min,
        "p_zero": p_zero,
        "trivial_zero": False,
        "p_min_capped": p_dense_min < p_dense_min_uncapped,
        "valid": valid,
        "probability_min": probability_min,
        "probability_max": probability_max,
        "up_max_minus_one": up_max_minus_one,
        "solve_seconds": solve_seconds,
        "runtime_seconds": time.perf_counter() - case_start,
        "linear_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(escape), 1.0e-300)),
    }


def generate_cpu_cases(cases, *, p_max, Np, Nxi, B_T, p_min_global, N_p_coarse,
                        n_jobs=-1, p_scan_min=1.0e-12):
    """Generate adaptive-grid FV cases in parallel with joblib."""
    worker = partial(solve_cpu_case, p_max=p_max, Np=Np, Nxi=Nxi, B_T=B_T,
                     p_min_global=p_min_global, N_p_coarse=N_p_coarse,
                     p_scan_min=p_scan_min)
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(worker)(case) for case in np.asarray(cases, dtype=np.float64)
    )


def flatten_fv_dataset(dataset):
    """Flatten exact ``(case, p, xi)`` arrays without interpolation."""
    p_grid = np.asarray(dataset["p_grid"], dtype=np.float64)
    xi_grid = np.asarray(dataset["xi_grid"], dtype=np.float64)
    P_grid = np.asarray(dataset["P_grid"], dtype=np.float64)
    n_cases, Np, Nxi = P_grid.shape
    return {
        "p": np.broadcast_to(p_grid[:, :, None], (n_cases, Np, Nxi)).reshape(-1),
        "xi": np.broadcast_to(xi_grid[:, None, :], (n_cases, Np, Nxi)).reshape(-1),
        "case_index": np.repeat(np.arange(n_cases), Np * Nxi),
        "P": P_grid.reshape(-1),
    }


def coarsen_fv_cases(results, *, p_stride=1, xi_stride=1):
    """Return exact strided cell centers and fields; no interpolation occurs."""
    if p_stride < 1 or xi_stride < 1:
        raise ValueError("FV strides must be positive integers")
    if not results:
        return []
    return [{
        **result,
        "p": result["p"][::p_stride],
        "xi": result["xi"][::xi_stride],
        "P": result["P"][::p_stride, ::xi_stride],
    } for result in results]


def save_fv_dataset_npz(path, cases, results, *, p_floor=None, p_max=None):
    """Save FP64 cases, grids, fields, floors, and JSON metadata in one archive."""
    path = str(path)
    if not results:
        raise ValueError("cannot save empty FV dataset")
    p_grid = np.stack([result["p"] for result in results])
    xi_grid = np.stack([result["xi"] for result in results])
    P_grid = np.stack([result["P"] for result in results])
    metadata = [{
        "p_dense_min": float(result["p_dense_min"]),
        "p_zero": float(result["p_zero"]) if np.isfinite(result["p_zero"]) else None,
            "trivial_zero": bool(result["trivial_zero"]),
            "valid": bool(result.get("valid", True)),
            "probability_min": float(result.get("probability_min", np.min(result["P"]))),
            "probability_max": float(result.get("probability_max", np.max(result["P"]))),
            "p_min_capped": bool(result.get("p_min_capped", False)),
        "up_max_minus_one": float(result["up_max_minus_one"]),
        "linear_residual": float(result["linear_residual"]),
        "runtime_seconds": float(result["runtime_seconds"]),
    } for result in results]
    np.savez_compressed(
        path,
        cases=np.asarray(cases, dtype=np.float64),
        p_grid=p_grid,
        xi_grid=xi_grid,
        P_grid=P_grid,
        p_floor=np.asarray(np.nan if p_floor is None else p_floor, dtype=np.float64),
        p_max=np.asarray(np.nan if p_max is None else p_max, dtype=np.float64),
        case_metadata_json=np.asarray(json.dumps(metadata)),
    )


def save_fv_dataset_directory(path, cases, results, *, p_floor=None, p_max=None):
    """Save FP64 FV arrays as memory maps, then publish by directory rename."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"dataset path already exists: {path}")
    if not results:
        raise ValueError("cannot save empty FV dataset")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        # Build all files in a temporary sibling directory so readers never
        # observe a partially written dataset.
        n_cases = len(results)
        n_p, n_xi = np.asarray(results[0]["P"]).shape
        np.save(temporary / "cases.npy", np.asarray(cases, dtype=np.float64))
        p_grid = np.lib.format.open_memmap(
            temporary / "p_grid.npy", mode="w+", dtype=np.float64,
            shape=(n_cases, n_p))
        xi_grid = np.lib.format.open_memmap(
            temporary / "xi_grid.npy", mode="w+", dtype=np.float64,
            shape=(n_cases, n_xi))
        P_grid = np.lib.format.open_memmap(
            temporary / "P_grid.npy", mode="w+", dtype=np.float64,
            shape=(n_cases, n_p, n_xi))
        metadata = []
        for index, result in enumerate(results):
            field = np.asarray(result["P"], dtype=np.float64)
            if field.shape != (n_p, n_xi):
                raise ValueError("FV cases have inconsistent grid shapes")
            p_grid[index] = result["p"]
            xi_grid[index] = result["xi"]
            P_grid[index] = field
            metadata.append({
                "p_dense_min": float(result["p_dense_min"]),
                "p_zero": float(result["p_zero"]) if np.isfinite(result["p_zero"]) else None,
                "trivial_zero": bool(result["trivial_zero"]),
                "valid": bool(result.get("valid", True)),
                "probability_min": float(result.get("probability_min", np.min(field))),
                "probability_max": float(result.get("probability_max", np.max(field))),
                "p_min_capped": bool(result.get("p_min_capped", False)),
                "up_max_minus_one": float(result["up_max_minus_one"]),
                "linear_residual": float(result["linear_residual"]),
                "runtime_seconds": float(result["runtime_seconds"]),
            })
        for array in (p_grid, xi_grid, P_grid):
            array.flush()
        del p_grid, xi_grid, P_grid
        (temporary / "case_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n")
        np.save(temporary / "p_floor.npy", np.asarray(
            np.nan if p_floor is None else p_floor, dtype=np.float64))
        np.save(temporary / "p_max.npy", np.asarray(
            np.nan if p_max is None else p_max, dtype=np.float64))
        os.replace(temporary, path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_fv_dataset(path):
    """Load NPZ arrays or read-only directory memory maps by path type."""
    path = Path(path)
    if path.is_dir():
        return {
            "cases": np.load(path / "cases.npy", mmap_mode="r"),
            "p_grid": np.load(path / "p_grid.npy", mmap_mode="r"),
            "xi_grid": np.load(path / "xi_grid.npy", mmap_mode="r"),
            "P_grid": np.load(path / "P_grid.npy", mmap_mode="r"),
            "p_floor": np.load(path / "p_floor.npy", mmap_mode="r"),
            "p_max": np.load(path / "p_max.npy", mmap_mode="r"),
            "case_metadata_json": np.asarray(
                (path / "case_metadata.json").read_text()),
        }
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_fv_cases(path):
    """Load saved FV grids into the case/result format used by training."""
    dataset = load_fv_dataset(path)
    cases = np.asarray(dataset["cases"], dtype=np.float64)
    p_grid = np.asarray(dataset["p_grid"], dtype=np.float64)
    xi_grid = np.asarray(dataset["xi_grid"], dtype=np.float64)
    P_grid = np.asarray(dataset["P_grid"], dtype=np.float64)
    metadata = json.loads(str(np.asarray(dataset["case_metadata_json"]).item()))
    if not (len(cases) == len(p_grid) == len(xi_grid) == len(P_grid) == len(metadata)):
        raise ValueError("saved FV dataset case arrays have inconsistent lengths")
    results = []
    for index in range(len(cases)):
        result = {
            "case": cases[index], "p": p_grid[index], "xi": xi_grid[index],
            "P": P_grid[index],
        }
        result.update(metadata[index])
        results.append(result)
    return cases, results


def summarize_fv_dataset(dataset_path):
    """Return reproducibility and coverage metrics for a saved dataset."""
    from core.training_artifacts import sha256_path
    dataset = load_fv_dataset(dataset_path)
    cases = np.asarray(dataset["cases"], dtype=np.float64)
    p_grid = np.asarray(dataset["p_grid"], dtype=np.float64)
    xi_grid = np.asarray(dataset["xi_grid"], dtype=np.float64)
    fields = np.asarray(dataset["P_grid"], dtype=np.float64)
    metadata = json.loads(str(np.asarray(dataset["case_metadata_json"]).item()))
    p_dense_min = np.asarray([item["p_dense_min"] for item in metadata], dtype=np.float64)
    runtime = np.asarray(
        [item["runtime_seconds"] for item in metadata], dtype=np.float64)
    valid = np.asarray([item.get("valid", True) for item in metadata], dtype=bool)
    return {
        "dataset": str(dataset_path),
        "sha256": sha256_path(dataset_path),
        "cases": int(len(cases)),
        "grid_shape": [int(value) for value in fields.shape[1:]],
        "points": int(fields.size),
        "p_dense_min_range": [float(p_dense_min.min()), float(p_dense_min.max())],
        "p_range": [float(p_grid.min()), float(p_grid.max())],
        "xi_range": [float(xi_grid.min()), float(xi_grid.max())],
        "probability_range": [float(fields.min()), float(fields.max())],
        "zero_fraction": float(np.mean(fields == 0.0)),
        "invalid_cases": int(np.count_nonzero(~valid)),
        "average_case_runtime_seconds": float(runtime.mean()),
        "parameter_ranges": {
            "E/Ec": [float(cases[:, 0].min()), float(cases[:, 0].max())],
            "Te_eV": [float(cases[:, 1].min()), float(cases[:, 1].max())],
            "nD_m3": [float(cases[:, 2].min()), float(cases[:, 2].max())],
            "nNe_m3": [float(cases[:, 3].min()), float(cases[:, 3].max())],
            "zD": [float(cases[:, 4].min()), float(cases[:, 4].max())],
            "zNe": [float(cases[:, 5].min()), float(cases[:, 5].max())],
        },
    }


def plot_fv_coverage(dataset_path, output_path):
    """Save parameter-coverage and adaptive-grid summary plots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt
    dataset = load_fv_dataset(dataset_path)
    cases = np.asarray(dataset["cases"], dtype=np.float64)
    fields = np.asarray(dataset["P_grid"], dtype=np.float64)
    metadata = json.loads(str(np.asarray(dataset["case_metadata_json"]).item()))
    p_dense_min = np.asarray([item["p_dense_min"] for item in metadata], dtype=np.float64)
    runtime = np.asarray(
        [item["runtime_seconds"] for item in metadata], dtype=np.float64)
    zero_fraction = np.mean(fields == 0.0, axis=(1, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].hist(p_dense_min, bins=40)
    axes[0, 0].set(xscale="log", xlabel="adaptive dense-zone front", ylabel="cases")
    axes[0, 1].hist(runtime, bins=40)
    axes[0, 1].set(xlabel="runtime [s/case]", ylabel="cases")
    axes[0, 2].scatter(cases[:, 0], cases[:, 1], s=4, alpha=0.35)
    axes[0, 2].set(xscale="log", yscale="log", xlabel="E/Ec", ylabel="Te [eV]")
    axes[1, 0].scatter(cases[:, 2], cases[:, 3], s=4, alpha=0.35)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="nD", ylabel="nNe")
    axes[1, 1].scatter(cases[:, 4], cases[:, 5], s=4, alpha=0.35)
    axes[1, 1].set(xlabel="zD", ylabel="zNe")
    axes[1, 2].scatter(p_dense_min, zero_fraction, s=4, alpha=0.35)
    axes[1, 2].set(xscale="log", xlabel="adaptive dense-zone front", ylabel="zero-P fraction")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_fv_dataset(config_path="run_configs/fv_dataset.json"):
    config = json.loads(Path(config_path).read_text())
    dataset = Path(config.get("dataset", config["dataset_path"]))
    run_dir = Path(config.get("run_dir", dataset.parent))
    output = config.get("analytics_output", run_dir / "analytics.json")
    plot = config.get("analytics_plot", run_dir / "coverage.png")
    summary = summarize_fv_dataset(dataset)
    for key in ("cases", "grid_shape", "points", "p_dense_min_range",
                "probability_range", "average_case_runtime_seconds"):
        print(f"{key}: {summary[key]}")
    if output:
        from core.training_artifacts import atomic_write_json
        atomic_write_json(output, summary)
        print(f"saved: {output}")
    if plot:
        plot_fv_coverage(dataset, plot)
        print(f"saved: {plot}")
