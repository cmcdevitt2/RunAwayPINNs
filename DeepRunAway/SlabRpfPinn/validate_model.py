#!/usr/bin/env python3
"""Validate saved PINN against exact CPU-FV cases and PDE residuals.

The model metadata supplies architecture, domain, and residual settings. A
saved dataset must pass its manifest checks; fresh cases use the same physical
parameter domain and global momentum floor as training.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np
from core.fv_dataset import flatten_fv_dataset, generate_cpu_cases, load_fv_cases
from core.training_config import PinnDomain
from core.model import load_model, make_probability
from core.pde import drift_up, make_pde_functions, normalize_momentum, sobol_6d
from core.training_artifacts import (
    atomic_write_json, sha256_path, validate_dataset_manifest,
    validate_model_manifest,
)


PARAMETER_DOMAIN = {
    "E/Ec": (1.0, 1000.0, "log"),
    "Te_eV": (0.1, 100.0, "log"),
    "nD_m3": (1.0e20, 1.0e22, "log"),
    "nNe_m3": (1.0e16, 1.0e22, "log"),
    "zD": (0.01, 1.0, "linear"),
    "zNe": (0.01, 10.0, "linear"),
}


def regional_metrics(prediction, target):
    """Report MSE and maximum error for zero, transition, and saturated targets."""
    target = np.asarray(target)
    error = np.asarray(prediction) - target
    masks = {
        "zero": target <= 1.0e-12,
        "transition": (target > 1.0e-12) & (target < 1.0 - 1.0e-12),
        "saturated": target >= 1.0 - 1.0e-12,
    }
    return {
        name: {
            "count": int(np.sum(mask)),
            "mse": float(np.mean(error[mask] ** 2)) if np.any(mask) else None,
            "max_abs_error": float(np.max(np.abs(error[mask])))
            if np.any(mask) else None,
        }
        for name, mask in masks.items()
    }


def sample_cases(n_cases, seed, parameter_domain):
    """Generate reproducible six-parameter physical cases from scrambled Sobol points."""
    unit = sobol_6d(n_cases, seed)
    cases = np.empty_like(unit)
    for k, (lo, hi, scale) in enumerate(parameter_domain.values()):
        cases[:, k] = (np.exp(np.log(lo) + unit[:, k] * (np.log(hi) - np.log(lo)))
                       if scale == "log" else lo + unit[:, k] * (hi - lo))
    return cases


def build_validation_inputs(dataset, cases, p_floor, p_max,
                            momentum_sampling, parameter_domain):
    """Build normalized eight-coordinate inputs aligned with flattened FV values."""
    case_index = np.asarray(dataset["case_index"], dtype=np.int64)
    p = np.asarray(dataset["p"], dtype=np.float64)
    normalized = np.empty_like(cases, dtype=np.float64)
    for column, (lo, hi, scale) in enumerate(parameter_domain.values()):
        normalized[:, column] = (
            (np.log(cases[:, column]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if scale == "log" else (cases[:, column] - lo) / (hi - lo))
    z = np.empty((len(p), 8), dtype=np.float64)
    z[:, 0] = normalize_momentum(p, p_floor, p_max, momentum_sampling)
    z[:, 1] = 0.5 * (np.asarray(dataset["xi"]) + 1.0)
    z[:, 2:] = normalized[case_index]
    return z


def evaluate_pde_chunked(params, z, domain, chunk_size, *, probability_fn=None,
                         coeff_norm="cf_ebar", residual_floor=0.1):
    """Evaluate literal PINN residual in bounded JAX batches."""
    coeff_fn, residual_fn = make_pde_functions(
        domain, probability_fn=probability_fn, coeff_norm=coeff_norm,
        residual_floor=residual_floor)
    values = []
    for start in range(0, len(z), chunk_size):
        batch = jnp.asarray(z[start:start + chunk_size])
        values.append(np.asarray(jax.device_get(
            residual_fn(params, batch, coeff_fn(batch)))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def save_validation_plots(output_base, target, prediction, error, pde,
                          case_index, results, per_case, z, domain):
    """Save correlation and worst-case field plots with the analytic threshold overlay."""
    import matplotlib.pyplot as plt

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_base.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes[0, 0].scatter(target, prediction, c=case_index, s=2, alpha=0.25,
                       cmap="turbo")
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set(xlabel="FV P", ylabel="PINN P", title="Prediction correlation")
    axes[0, 1].scatter(target, np.abs(error), c=case_index, s=2, alpha=0.25,
                       cmap="turbo")
    axes[0, 1].set(xlabel="FV P", ylabel="|PINN − FV|", title="Absolute error")
    axes[1, 0].scatter(target, np.abs(pde), c=case_index, s=2, alpha=0.25,
                       cmap="turbo")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="FV P", ylabel="|PDE residual|",
                   title="PDE residual correlation")
    axes[1, 1].scatter(np.maximum(np.abs(error), 1.0e-16),
                       np.maximum(np.abs(pde), 1.0e-16),
                       c=case_index, s=2, alpha=0.25, cmap="turbo")
    axes[1, 1].set(xscale="log", yscale="log",
                   xlabel="|PINN − FV|", ylabel="|PDE residual|",
                   title="Error/residual correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    correlation_path = output_base.with_name(output_base.name + "_correlation.png")
    fig.savefig(correlation_path, dpi=150)
    plt.close(fig)

    worst = sorted(per_case, key=lambda item: item["mse"], reverse=True)
    selected = [item["case"] for item in worst[:min(3, len(worst))]]
    fig, axes = plt.subplots(len(selected), 4,
                             figsize=(18, 4.5 * len(selected)),
                             squeeze=False, constrained_layout=True)
    for row, case_number in enumerate(selected):
        result = results[case_number]
        mask = case_index == case_number
        p = np.asarray(result["p"])
        xi = np.asarray(result["xi"])
        shape = (len(p), len(xi))
        fields = [
            np.asarray(result["P"]),
            prediction[mask].reshape(shape),
            np.abs(error[mask]).reshape(shape),
            np.abs(pde[mask]).reshape(shape),
        ]
        z_case = z[mask].reshape(shape + (8,))
        up = np.asarray(jax.device_get(
            drift_up(jnp.asarray(z_case.reshape(-1, 8)), domain)
        )).reshape(shape)
        titles = ["FV RPF", "PINN RPF", "|PINN − FV|", "|PDE residual|"]
        for column, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row, column]
            energy = 511.0e3 * (np.sqrt(1.0 + p * p) - 1.0)
            values = np.clip(field.T, 0.0, 1.0) if column < 2 else np.log10(
                np.maximum(field.T, 1.0e-16))
            mesh = axis.contourf(
                energy, xi, values, levels=50, cmap="turbo" if column < 2 else "magma",
                vmin=0.0 if column < 2 else None,
                vmax=1.0 if column < 2 else None)
            axis.set(xscale="log", xlabel="Energy [eV]", ylabel=r"$\xi$",
                     title=f"case {case_number}: {title}")
            axis.grid(alpha=0.25)
            axis.contour(
                energy, xi, up.T, levels=[0.0], colors="white",
                linewidths=1.0)
            if column == 0:
                axis.plot([], [], color="white", label=r"$U_p=0$")
                axis.legend(loc="best", fontsize="small")
            fig.colorbar(mesh, ax=axis, pad=0.02)
    cases_path = output_base.with_name(output_base.name + "_cases.png")
    fig.savefig(cases_path, dpi=150)
    plt.close(fig)
    return correlation_path, cases_path


CONFIG_PATH = Path("run_configs/validate_model.json")


def main(config_path=CONFIG_PATH):
    """Load model metadata, validate or generate FV data, and write metrics/plots."""
    config = json.loads(Path(config_path).read_text())
    args = SimpleNamespace(
        model_dir=Path(config["model_dir"]),
        model_manifest=config.get("model_manifest"),
        output_dir=Path(config["output_dir"]),
        plot=Path(config["plot"]) if config.get("plot") else None,
        cases=int(config.get("cases", 32)),
        fv_Np=int(config.get("fv_Np", 512)),
        fv_Nxi=int(config.get("fv_Nxi", 128)),
        fv_p_coarse_N=int(config.get("fv_p_coarse_N", 32)),
        pde_chunk=int(config.get("pde_chunk", 8192)),
        n_jobs=int(config.get("n_jobs", -1)),
        seed=int(config.get("seed", 9090)),
        dataset_path=config.get("dataset_path"),
        dataset_manifest=config.get("dataset_manifest"),
    )
    if args.pde_chunk <= 0:
        raise ValueError("--pde-chunk must be positive")

    # Validation uses the same JAX execution path as training and currently
    # requires visible GPUs, even though FV generation itself remains on CPU.
    if jax.default_backend() != "gpu" or not any(
        getattr(device, "platform", "") == "gpu" for device in jax.devices()
    ):
        raise RuntimeError("GPU backend required; refusing PINN validation")

    timing_start = time.perf_counter()
    print("loading model", flush=True)
    model_path = args.model_dir / "pinn_params.npz"
    if args.model_manifest:
        validate_model_manifest(model_path, args.model_manifest)
    metadata = json.loads((args.model_dir / "model_metadata.json").read_text())
    parameter_domain = metadata.get("parameter_domain", PARAMETER_DOMAIN)
    p_floor = float(metadata["normalization"]["p_floor"])
    p_max = float(metadata["normalization"]["p_max"])
    B_T = float(metadata["domain"]["B_T"])
    momentum_sampling = metadata["domain"].get("momentum_sampling", "log")
    architecture = metadata.get("architecture", {})
    training_config = metadata.get(
        "training_config", metadata.get("pinn_config", {}))
    model_type = architecture.get(
        "model_type", training_config.get("model_type", "mlp"))
    output_transform = architecture.get(
        "output_transform",
        training_config.get("model", {}).get("output_transform", "sigmoid"))
    regularization = metadata.get("residual_regularization", {})
    loss_config = training_config.get("loss", {})
    coeff_norm = regularization.get(
        "coeff_norm", loss_config.get("residual_coeff_norm", "cf_ebar"))
    residual_floor = float(regularization.get(
        "floor", loss_config.get("residual_floor", 0.1)))
    probability_fn, predict_fn = make_probability(output_transform)
    width = int(metadata["architecture"]["width"])
    depth = int(metadata["architecture"]["depth"])
    params = load_model(
        model_path, model_type=model_type,
        width=width, depth=depth,
        latent_width=int(architecture.get("latent_width", training_config.get("latent_width", 64))),
        branch_width=int(architecture.get("branch_width", training_config.get("branch_width", 32))),
        branch_depth=int(architecture.get("branch_depth", training_config.get("branch_depth", 3))),
        trunk_width=int(architecture.get("trunk_width", training_config.get("trunk_width", 32))),
        trunk_depth=int(architecture.get("trunk_depth", training_config.get("trunk_depth", 3))),
    )
    domain = PinnDomain(**metadata["domain"])
    print(f"model loaded: {time.perf_counter() - timing_start:.3f} s", flush=True)

    # Reuse a saved dataset only after path, compatibility, and checksum checks;
    # otherwise generate fresh CPU-FV cases and discard trivial-zero results.
    if args.dataset_path:
        if not args.dataset_manifest:
            raise ValueError(
                "dataset_manifest is required when dataset_path is supplied")
        validate_dataset_manifest(
            args.dataset_path, args.dataset_manifest,
            expected_config={"parameter_domain": parameter_domain},
            expected_dataset_config={"p_max": p_max, "B_T": B_T})
        cases, results = load_fv_cases(args.dataset_path)
        print(f"using validated FV dataset: {args.dataset_path}", flush=True)
    else:
        cases = sample_cases(args.cases, args.seed, parameter_domain)
        stage_start = time.perf_counter()
        results = generate_cpu_cases(
            cases, p_max=p_max, Np=args.fv_Np, Nxi=args.fv_Nxi,
            B_T=B_T, p_min_global=p_floor, N_p_coarse=args.fv_p_coarse_N,
            n_jobs=args.n_jobs)
        print(f"FV generation: {time.perf_counter() - stage_start:.3f} s", flush=True)
        keep = np.array([
            not result["trivial_zero"] and result.get("valid", True)
            for result in results
        ])
        cases = cases[keep]
        results = [result for result, retain in zip(results, keep) if retain]
        if not results:
            raise RuntimeError("all validation cases are trivial zero-RPF cases")
    stage_start = time.perf_counter()
    # Flatten exact FV arrays once. Case indices keep per-case metrics aligned
    # after predictions and residuals are computed in chunks.
    data = flatten_fv_dataset({
        "p_grid": np.stack([result["p"] for result in results]),
        "xi_grid": np.stack([result["xi"] for result in results]),
        "P_grid": np.stack([result["P"] for result in results]),
    })
    print(f"exact FV cells: {time.perf_counter() - stage_start:.3f} s", flush=True)
    z = build_validation_inputs(
        data, cases, p_floor, p_max, momentum_sampling, parameter_domain)
    target = np.asarray(data["P"], dtype=np.float64)
    stage_start = time.perf_counter()
    prediction = np.asarray(jax.device_get(predict_fn(params, jnp.asarray(z))))
    print(f"PINN prediction: {time.perf_counter() - stage_start:.3f} s", flush=True)
    error = prediction - target

    case_index = np.asarray(data["case_index"], dtype=np.int64)
    stage_start = time.perf_counter()
    pde = evaluate_pde_chunked(
        params, z, domain, args.pde_chunk, probability_fn=probability_fn,
        coeff_norm=coeff_norm, residual_floor=residual_floor)
    print(f"PDE residual: {time.perf_counter() - stage_start:.3f} s", flush=True)
    pde_rms = float(np.sqrt(np.mean(pde * pde)))

    per_case = []
    parameter_names = list(parameter_domain)
    for case_number in range(len(results)):
        mask = case_index == case_number
        per_case.append({
            "case": case_number,
            "parameters": {
                name: float(value)
                for name, value in zip(parameter_names, cases[case_number])
            },
            "p_dense_min": float(results[case_number]["p_dense_min"]),
            "trivial_zero": bool(results[case_number]["trivial_zero"]),
            "mse": float(np.mean(error[mask] ** 2)),
            "max_abs_error": float(np.max(np.abs(error[mask]))),
        })

    summary = {
        "model_dir": str(args.model_dir),
        "model_sha256": sha256_path(model_path),
        "cases": len(results),
        "pde_points": int(len(pde)),
        "mse": float(np.mean(error * error)),
        "max_abs_error": float(np.max(np.abs(error))),
        "regional_metrics": regional_metrics(prediction, target),
        "pde_residual_rms": pde_rms,
        "per_case": per_case,
    }
    if args.dataset_path:
        summary["dataset_path"] = str(args.dataset_path)
        summary["dataset_sha256"] = sha256_path(args.dataset_path)
    if args.model_manifest:
        summary["model_manifest"] = str(args.model_manifest)
    output_dir = args.output_dir or args.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot is not None:
        correlation_path, cases_path = save_validation_plots(
            args.plot, target, prediction, error, pde, case_index, results,
            per_case, z, domain)
        summary["plots"] = {
            "correlation": str(correlation_path),
            "cases": str(cases_path),
        }
    output = output_dir / "fresh_validation.json"
    atomic_write_json(output, summary)
    print(f"fresh nontrivial cases: {len(results)}")
    print(f"MSE: {summary['mse']:.6e}")
    print(f"max absolute error: {summary['max_abs_error']:.6e}")
    print(f"PDE residual RMS: {pde_rms if pde_rms is not None else 'n/a'}")
    print(f"saved: {output}")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1])) if len(sys.argv) > 1 else main()
