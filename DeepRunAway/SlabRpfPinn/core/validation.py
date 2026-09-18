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
import numpy as np
from core.fv_dataset import flatten_fv_dataset, generate_cpu_cases, load_fv_cases
from core.training_config import (
    DEFAULT_PARAMETER_DOMAIN, PARAMETER_NAMES, PinnDomain,
    validate_parameter_domain,
)
from core.model import load_model, make_probability
from core.evaluation import (
    build_validation_inputs, evaluate_pde_chunked,
    evaluate_predictions_chunked, regional_metrics, sample_cases,
)
from core.validation_plots import save_validation_plots
from core.training_artifacts import (
    atomic_write_json, sha256_path, validate_dataset_manifest,
    validate_model_manifest,
)


PARAMETER_DOMAIN = DEFAULT_PARAMETER_DOMAIN


CONFIG_PATH = Path("configs/validate_model.json")


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
    if not args.model_manifest:
        raise ValueError("model_manifest is required for validation")
    validate_model_manifest(model_path, args.model_manifest)
    metadata = json.loads((args.model_dir / "model_metadata.json").read_text())
    parameter_domain = validate_parameter_domain(
        metadata.get("parameter_domain", PARAMETER_DOMAIN))
    p_floor = float(metadata["normalization"]["p_floor"])
    p_max = float(metadata["normalization"]["p_max"])
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
            expected_dataset_config={"p_max": p_max, "fv_p_min": p_floor})
        cases, results = load_fv_cases(args.dataset_path)
        print(f"using validated FV dataset: {args.dataset_path}", flush=True)
    else:
        cases = sample_cases(args.cases, args.seed, parameter_domain)
        stage_start = time.perf_counter()
        results = generate_cpu_cases(
            cases, p_max=p_max, Np=args.fv_Np, Nxi=args.fv_Nxi,
            p_min_global=p_floor, N_p_coarse=args.fv_p_coarse_N,
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
    prediction = evaluate_predictions_chunked(
        params, z, predict_fn, chunk_size=max(args.pde_chunk, 65536))
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
    parameter_names = PARAMETER_NAMES
    if len(target) % len(results) != 0:
        raise ValueError("validation cells are not evenly grouped by case")
    points_per_case = len(target) // len(results)
    error_by_case = error.reshape(len(results), points_per_case)
    for case_number in range(len(results)):
        per_case.append({
            "case": case_number,
            "parameters": {
                name: float(value)
                for name, value in zip(parameter_names, cases[case_number])
            },
            "p_dense_min": float(results[case_number]["p_dense_min"]),
            "trivial_zero": bool(results[case_number]["trivial_zero"]),
            "mse": float(np.mean(error_by_case[case_number] ** 2)),
            "max_abs_error": float(np.max(np.abs(error_by_case[case_number]))),
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"validation output directory is not empty: {output_dir}")
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
