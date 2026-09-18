"""Model evaluation, validation-input construction, and validation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from core.pde import make_pde_functions, normalize_momentum, sobol_7d
from core.training_config import (
    PARAMETER_NAMES, validate_parameter_domain,
)


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
    """Generate reproducible seven-parameter physical cases from Sobol points."""
    parameter_domain = validate_parameter_domain(parameter_domain)
    unit = sobol_7d(n_cases, seed)
    cases = np.empty_like(unit)
    for k, name in enumerate(PARAMETER_NAMES):
        lo, hi, scale = parameter_domain[name]
        cases[:, k] = (np.exp(np.log(lo) + unit[:, k] * (np.log(hi) - np.log(lo)))
                       if scale == "log" else lo + unit[:, k] * (hi - lo))
    return cases


def build_validation_inputs(dataset, cases, p_floor, p_max,
                            momentum_sampling, parameter_domain):
    """Build normalized nine-coordinate inputs aligned with flattened FV values."""
    case_index = np.asarray(dataset["case_index"], dtype=np.int64)
    p = np.asarray(dataset["p"], dtype=np.float64)
    normalized = np.empty_like(cases, dtype=np.float64)
    parameter_domain = validate_parameter_domain(parameter_domain)
    for column, name in enumerate(PARAMETER_NAMES):
        lo, hi, scale = parameter_domain[name]
        normalized[:, column] = (
            (np.log(cases[:, column]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if scale == "log" else (cases[:, column] - lo) / (hi - lo))
    z = np.empty((len(p), 9), dtype=np.float64)
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


def evaluate_predictions_chunked(params, z, predict_fn, chunk_size=65536):
    """Evaluate model predictions in bounded JAX batches."""
    values = []
    for start in range(0, len(z), chunk_size):
        batch = jnp.asarray(z[start:start + chunk_size])
        values.append(np.asarray(jax.device_get(predict_fn(params, batch))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)
