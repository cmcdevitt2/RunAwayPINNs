"""Physics-informed and SSBroyden training loops.

Physics training uses FP64 JAX arrays. FV labels stay on the host and outside
automatic differentiation. Distributed paths use one JAX process per node.
"""

import math
import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from jax.flatten_util import ravel_pytree

from core.training_config import PinnConfig
from core.model import init_model, make_probability
from core.pde import (
    analytic_threshold_collocation, evaluate_pde_residuals,
    make_pde_functions, pde_coefficients, sobol_9d_nontrivial,
    success_boundary_target,
)
from core.optimizers import make_soap_optimizer
from core.training_runtime import (
    replicate, shard_with_mask, training_devices, unreplicate,
)

def _train_physics_multi(z_data, y_data, z_pde, z_bc, *, domain, config,
                         z_pde_threshold=None, z_low=None,
                         initial_params=None, checkpoint_callback=None,
                         distributed=False):
    """Run masked physics loss across local devices and optional JAX hosts."""
    devices = training_devices(config.n_devices)
    if (distributed
            and config.n_devices not in (0, len(jax.local_devices(backend="gpu")))):
        raise ValueError("distributed training requires all local GPUs")
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
    if (config.enable_pde and config.enable_threshold_pde
            and (z_pde_threshold is None or len(z_pde_threshold) == 0)):
        raise ValueError("threshold PDE loss requires threshold points")
    if config.enable_low_p_bc and (z_low is None or len(z_low) == 0):
        raise ValueError("low-p boundary loss requires low-p points")
    if len(z_data) == 0:
            z_data, y_data = np.zeros((1, 9)), np.zeros(1)
    if len(z_pde) == 0:
        z_pde = np.zeros((1, 9))
    if len(z_bc) == 0:
        z_bc = np.zeros((1, 9))
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
    optimizer = make_soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = replicate(params, devices)
    opt_state = replicate(opt_state, devices)
    probability_fn, _ = make_probability(config.output_transform)
    _, residual_fn = make_pde_functions(
        domain, probability_fn=probability_fn,
        coeff_norm=config.residual_coeff_norm,
        residual_floor=config.residual_floor)
    pmap_kwargs = {} if distributed else {"devices": devices}
    @partial(jax.pmap, **pmap_kwargs)
    def preprocess(zp, zt, zb):
        return (
            pde_coefficients(zp, domain, config.residual_coeff_norm),
            pde_coefficients(zt, domain, config.residual_coeff_norm),
            success_boundary_target(zb, domain),
        )

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
            # Masks remove padded samples. The threshold term intentionally
            # uses the PDE weight because it is a PDE residual subset.
            data_loss = (scaled_mse(probability_fn(params, zd) - yd, md)
                         if config.enable_data else jnp.asarray(0.0, jnp.float64))
            pde_loss = (scaled_mse(residual_fn(params, zp, cp), mp)
                        if config.enable_pde else jnp.asarray(0.0, jnp.float64))
            threshold_loss = (scaled_mse(residual_fn(params, zt, ct), mt)
                if config.enable_pde and config.enable_threshold_pde
                else jnp.asarray(0.0, jnp.float64))
            low_p_loss = (scaled_mse(probability_fn(params, zl), ml)
                          if config.enable_low_p_bc
                          else jnp.asarray(0.0, jnp.float64))
            bc_loss = (scaled_mse(probability_fn(params, zb) - 1.0, mb * active)
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
        return shard_with_mask(values, devices, valid_count)

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
        return (shard_with_mask(values, devices, valid_count),
                shard_with_mask(targets, devices, valid_count))

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
        coeff_pde, coeff_threshold, bc_active = preprocess(
            pde[0], threshold[0], bc[0])
        params, opt_state, values = step(
            params, opt_state, data[0], data_y[0], data[1],
            pde[0], coeff_pde, pde[1],
            threshold[0], coeff_threshold, threshold[1],
            low[0], low[1], bc[0], bc[1], bc_active)
        if process_index == 0:
            history.append(values[0])
        checkpoint_due = (checkpoint_callback is not None
                and config.checkpoint_every > 0
                and ((step_number + 1) % config.checkpoint_every == 0
                     or step_number + 1 == config.steps))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"physics_checkpoint_before_{step_number + 1}")
        if checkpoint_due and process_index == 0:
            checkpoint_callback(step_number + 1, unreplicate(params))
        if checkpoint_due and distributed and process_count > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(
                f"physics_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("physics_training_complete")
    history_array = (np.asarray(jax.device_get(jnp.stack(history)))
                     if history else np.empty((0, 6), dtype=np.float64))
    return unreplicate(params), history_array, {
        "pde_coefficients": None, "n_devices": n_devices}


def train_physics_informed(z_data, y_data, z_pde, z_bc, *, domain,
                           config=None, z_pde_threshold=None, z_low=None,
                           initial_params=None, checkpoint_callback=None,
                           distributed=False):
    """Train PINN with data, PDE, threshold, low-p, and high-p boundary terms."""
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
    optimizer = make_soap_optimizer(config)
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
    z_data = z_data if len(z_data) else jnp.zeros((1, 9), dtype=jnp.float64)
    y_data = y_data if len(y_data) else jnp.zeros((1,), dtype=jnp.float64)
    z_pde = z_pde if len(z_pde) else jnp.zeros((1, 9), dtype=jnp.float64)
    z_bc = z_bc if len(z_bc) else jnp.zeros((1, 9), dtype=jnp.float64)
    use_threshold = z_pde_threshold is not None and len(z_pde_threshold) > 0
    if config.enable_pde and config.enable_threshold_pde and not use_threshold:
        raise ValueError("threshold PDE loss requires threshold points")
    threshold_weight = config.threshold_weight if use_threshold and config.enable_threshold_pde else 0.0
    z_pde_threshold = (jnp.asarray(z_pde_threshold) if use_threshold else z_pde[:1])
    z_low = jnp.asarray(z_low) if use_low else z_bc[:1]
    probability_fn, _ = make_probability(config.output_transform)
    coeff_fn, residual_fn = make_pde_functions(
        domain, probability_fn=probability_fn,
        coeff_norm=config.residual_coeff_norm,
        residual_floor=config.residual_floor)
    coeff_pde = coeff_fn(z_pde)
    coeff_pde_threshold = coeff_fn(z_pde_threshold)
    bc_active = (success_boundary_target(z_bc, domain)
                 if config.enable_pmax_bc else jnp.zeros(len(z_bc), dtype=jnp.float64))
    @jax.jit
    def step(params, opt_state, zd, yd, zp, cb, zt, ct, zl, zb, yb):
        def loss_fn(params):
            # ``yb`` is a binary high-momentum success target. Low-p uses a
            # zero target, while PDE terms use the same relative residual.
            data_loss = (jnp.mean((probability_fn(params, zd) - yd) ** 2)
                         if config.enable_data else jnp.asarray(0.0, dtype=jnp.float64))
            pde_loss = (jnp.mean(residual_fn(params, zp, cb) ** 2)
                        if config.enable_pde else jnp.asarray(0.0, dtype=jnp.float64))
            threshold_loss = (jnp.mean(residual_fn(params, zt, ct) ** 2)
                              if config.enable_pde and config.enable_threshold_pde
                              else jnp.asarray(0.0, dtype=jnp.float64))
            low_p_loss = (jnp.mean(probability_fn(params, zl) ** 2)
                          if config.enable_low_p_bc
                          else jnp.asarray(0.0, dtype=jnp.float64))
            bc_loss = jnp.sum(yb * (probability_fn(params, zb) - 1.0) ** 2) / jnp.maximum(jnp.sum(yb), 1.0)
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
        history.append(components)
        if (checkpoint_callback is not None
                and config.checkpoint_every > 0
                and ((step_number + 1) % config.checkpoint_every == 0
                     or step_number + 1 == config.steps)):
            checkpoint_callback(step_number + 1, params)
    history_array = (np.asarray(jax.device_get(jnp.stack(history)))
                     if history else np.empty((0, 6), dtype=np.float64))
    return params, history_array, {"pde_coefficients": coeff_pde}


def train_physics_active(z_data, y_data, z_pde, z_bc, *, domain,
                         config=None, z_pde_threshold=None, z_low=None,
                         active_config=None, acquire_data=None,
                         checkpoint_callback=None, distributed=False,
                         initial_params=None):
    """Train physics-informed cycles with residual-guided FV acquisition.

    ``acquire_data`` receives selected normalized nine-dimensional points and
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
    probability_fn, _ = make_probability(config.output_transform)
    histories = []
    records = []
    for cycle in range(cycles):
        # Checkpoint steps include prior cycles so filenames remain unique.
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

        dense = sobol_9d_nontrivial(
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
        residual = evaluate_pde_residuals(
            params, candidates, domain, probability_fn=probability_fn,
            coeff_norm=config.residual_coeff_norm,
            residual_floor=config.residual_floor)
        score = np.abs(np.asarray(residual, dtype=np.float64))
        # Rank absolute residuals on host. FV acquisition remains a normal
        # Python callback and therefore cannot enter JAX's differentiation.
        score[~np.isfinite(score)] = -np.inf
        count = min(acquire_points, len(candidates))
        selected = np.argpartition(score, -count)[-count:]
        selected = selected[np.argsort(score[selected])[::-1]]
        z_selected = candidates[selected]
        z_new, y_new = acquire_data(z_selected)
        z_new = np.asarray(z_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if len(z_new):
            if z_new.ndim != 2 or z_new.shape[1] != 9:
                raise ValueError("acquired z data must have shape (N, 9)")
            if (y_new.ndim != 1 or len(z_new) != len(y_new)
                    or not np.isfinite(y_new).all()
                    or np.any(y_new < 0.0) or np.any(y_new > 1.0)):
                raise ValueError("acquired y data must be finite shape (N,) in [0, 1]")
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


def train_ssbroyden(params, z_data, y_data, z_pde, z_bc, *, domain,
                    config=None, z_pde_threshold=None, z_low=None,
                    checkpoint_callback=None, step_offset=0):
    """Refine PINN with blockwise full-batch SSBroyden and monotonic acceptance."""
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
    z_data = z_data if len(z_data) else jnp.zeros((1, 9), dtype=jnp.float64)
    y_data = y_data if len(y_data) else jnp.zeros((1,), dtype=jnp.float64)
    z_pde = z_pde if len(z_pde) else jnp.zeros((1, 9), dtype=jnp.float64)
    z_bc = z_bc if len(z_bc) else jnp.zeros((1, 9), dtype=jnp.float64)
    use_threshold = z_pde_threshold is not None and len(z_pde_threshold) > 0
    if config.enable_pde and config.enable_threshold_pde and not use_threshold:
        raise ValueError("threshold PDE loss requires threshold points")
    threshold_weight = config.threshold_weight if use_threshold and config.enable_threshold_pde else 0.0
    z_pde_threshold = (jnp.asarray(z_pde_threshold) if use_threshold else z_pde[:1])
    z_low = jnp.asarray(z_low) if use_low else z_bc[:1]
    probability_fn, _ = make_probability(config.output_transform)
    coeff_fn, residual_fn = make_pde_functions(
        domain, probability_fn=probability_fn,
        coeff_norm=config.residual_coeff_norm,
        residual_floor=config.residual_floor)
    coeff_pde = coeff_fn(z_pde)
    coeff_pde_threshold = coeff_fn(z_pde_threshold)
    bc_active = success_boundary_target(z_bc, domain)
    weights, unravel = ravel_pytree(params)

    def loss_components(p):
        data_loss = (jnp.mean((probability_fn(p, z_data) - y_data) ** 2)
                     if config.enable_data else jnp.asarray(0.0, dtype=jnp.float64))
        pde_loss = (jnp.mean(residual_fn(p, z_pde, coeff_pde) ** 2)
                    if config.enable_pde else jnp.asarray(0.0, dtype=jnp.float64))
        threshold_loss = (jnp.mean(residual_fn(p, z_pde_threshold, coeff_pde_threshold) ** 2)
                          if config.enable_pde and config.enable_threshold_pde
                          else jnp.asarray(0.0, dtype=jnp.float64))
        low_p_loss = (jnp.mean(probability_fn(p, z_low) ** 2)
                      if config.enable_low_p_bc
                      else jnp.asarray(0.0, dtype=jnp.float64))
        bc_loss = jnp.sum(bc_active * (probability_fn(p, z_bc) - 1.0) ** 2) / jnp.maximum(jnp.sum(bc_active), 1.0)
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
