"""Supervised training and FV data preparation.

This module owns grouped FV data conversion plus supervised MLP and DeepONet
training. Physics-informed, active, and SSBroyden training remains in
``core.training``.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from core.model import OUTPUT_TRANSFORMS, deeponet_probability, init_model, make_probability
from core.optimizers import make_soap_optimizer
from core.pde import normalize_momentum
from core.sampling import GroupedDeepONetBatchSampler, GroupedPointSampler
from core.training_config import PARAMETER_NAMES, validate_parameter_domain
from core.training_runtime import (
    predict_host_chunks, replicate, shard_leading, shard_with_mask,
    training_devices, unreplicate,
)


def _grouped_deeponet_mse(params, data, *, max_cases=None, transform=None):
    """Compute MSE over valid cells, excluding padded grouped-case entries."""
    if transform is None:
        transform = OUTPUT_TRANSFORMS["sigmoid"]
    if max_cases is not None:
        data = {name: values[:max_cases] for name, values in data.items()}
    numerator = 0.0
    denominator = 0.0
    for start in range(0, data["branch"].shape[0], 32):
        stop = start + 32
        prediction = np.asarray(jax.device_get(deeponet_probability(
            params,
            jnp.asarray(data["branch"][start:stop]),
            jnp.asarray(data["trunk"][start:stop]), transform=transform)))
        error = prediction - data["target"][start:stop]
        mask = data["mask"][start:stop]
        numerator += float(np.sum(mask * error * error))
        denominator += float(np.sum(mask))
    return numerator / max(denominator, 1.0)


def _grouped_deeponet_metrics(params, train_data, test_data,
                              *, transform=None):
    if transform is None:
        transform = OUTPUT_TRANSFORMS["sigmoid"]
    metrics = {}
    metrics["train_mse"] = _grouped_deeponet_mse(params, train_data, transform=transform)
    metrics["test_mse"] = _grouped_deeponet_mse(params, test_data, transform=transform)
    return metrics


def train_supervised_deeponet(train_data, test_data, *, config, distributed=False,
                              checkpoint_callback=None, global_case_count=None):
    """Train DeepONet from grouped cases with global batch semantics."""
    devices = training_devices(config.n_devices)
    if (distributed
            and config.n_devices not in (0, len(jax.local_devices(backend="gpu")))):
        raise ValueError("distributed training requires all local GPUs")
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
    sampler = GroupedDeepONetBatchSampler(
        train_data, case_batch_size,
        config.seed + (jax.process_index() if distributed else 0))
    test_case_limit = (test_data["branch"].shape[0]
                       if (config.test_batch_size == 0
                           and (not distributed or jax.process_index() == 0)) else
                       max(1, config.test_batch_size // max_points))
    if distributed and jax.process_index() != 0:
        test_case_limit = 0
    params = init_model(jax.random.PRNGKey(config.seed), config)
    optimizer = make_soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = replicate(params, devices)
    opt_state = replicate(opt_state, devices)
    transform = OUTPUT_TRANSFORMS[config.output_transform]

    pmap_kwargs = {"axis_name": "data"}
    if not distributed:
        pmap_kwargs["devices"] = devices

    @partial(jax.pmap, **pmap_kwargs)
    def step(params, opt_state, branch_z, trunk_z, target, mask):
        def loss_fn(params):
            error = deeponet_probability(
                params, branch_z, trunk_z, transform=transform) - target
            count = jax.lax.psum(jnp.sum(mask), axis_name="data")
            return jnp.sum(mask * error * error) * n_devices / jnp.maximum(count, 1.0)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        gradients = jax.lax.pmean(gradients, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    def prepare_host_batch():
        host_batch = sampler.next()
        return host_batch, tuple(shard_leading(host_batch[name], devices)
                                for name in ("branch", "trunk", "target", "mask"))

    history = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prepare_host_batch)
        initial_host_batch, batch = future.result()
        initial_train_loss = _grouped_deeponet_mse(
            unreplicate(params), initial_host_batch,
            transform=transform)
        initial_test_loss = _grouped_deeponet_mse(
            unreplicate(params), test_data, max_cases=test_case_limit,
            transform=transform)
        if not distributed or jax.process_index() == 0:
            history.append([initial_train_loss, initial_test_loss])
        for step_number in tqdm(
                range(config.steps), desc="supervised SOAP", unit="step",
            disable=distributed and jax.process_index() != 0):
            future = executor.submit(prepare_host_batch)
            params, opt_state, loss = step(params, opt_state, *batch)
            if ((step_number + 1) % config.log_every == 0
                    or step_number + 1 == config.steps):
                train_loss = float(jax.device_get(loss[0]))
                test_loss = _grouped_deeponet_mse(
                    unreplicate(params), test_data, max_cases=test_case_limit,
                    transform=transform)
                if not distributed or jax.process_index() == 0:
                    history.append([train_loss, test_loss])
            initial_host_batch, batch = future.result()
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
                checkpoint_callback(step_number + 1, unreplicate(params))
            if checkpoint_due and distributed and process_count > 1:
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices(
                    f"deeponet_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("deeponet_training_complete")
    params = unreplicate(params)
    metrics = (_grouped_deeponet_metrics(params, train_data, test_data, transform=transform)
               if not distributed or jax.process_index() == 0 else {})
    return params, np.asarray(history), metrics


def train_supervised_mlp_grouped(train_data, test_data, *, config,
                                 distributed=False, checkpoint_callback=None):
    """Train pointwise MLP without flattening all grouped FV cells."""
    devices = training_devices(config.n_devices)
    process_count = jax.process_count() if distributed else 1
    n_devices = jax.device_count() if distributed else len(devices)
    global_batch_size = config.batch_size if config.batch_size > 0 else 262144
    if distributed:
        if global_batch_size % process_count != 0:
            raise ValueError("MLP batch_size must be divisible by process count")
        local_batch_size = global_batch_size // process_count
    else:
        local_batch_size = global_batch_size
    sampler = GroupedPointSampler(
        train_data, config.seed + (jax.process_index() if distributed else 0))
    test_sampler = (GroupedPointSampler(test_data, config.seed + 100003)
                    if (not distributed or jax.process_index() == 0)
                    and test_data["branch"].shape[0] else None)
    params = init_model(jax.random.PRNGKey(config.seed), config)
    optimizer = make_soap_optimizer(config)
    opt_state = optimizer.init(params)
    params = replicate(params, devices)
    opt_state = replicate(opt_state, devices)
    probability_fn, predict_fn = make_probability(config.output_transform)
    pmap_kwargs = {"axis_name": "data"}
    if not distributed:
        pmap_kwargs["devices"] = devices

    @partial(jax.pmap, **pmap_kwargs)
    def step(params, opt_state, z_batch, y_batch, mask):
        def loss_fn(params):
            error = probability_fn(params, z_batch) - y_batch
            count = jax.lax.psum(jnp.sum(mask), axis_name="data")
            return jnp.sum(mask * error * error) * n_devices / jnp.maximum(count, 1.0)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        gradients = jax.lax.pmean(gradients, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    test_size = config.test_batch_size if config.test_batch_size > 0 else global_batch_size

    def prepare_batch():
        z_host, y_host = sampler.next(local_batch_size)
        z_batch, mask, _ = shard_with_mask(z_host, devices)
        y_batch = shard_leading(y_host, devices)
        return z_host, y_host, z_batch, y_batch, mask

    history = []
    initial_test = (test_sampler.next(test_size)
                    if test_sampler is not None else None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prepare_batch)
        for step_number in tqdm(
                range(config.steps), desc="supervised SOAP", unit="step",
                disable=distributed and jax.process_index() != 0):
            z_host, y_host, z_batch, y_batch, mask = future.result()
            future = executor.submit(prepare_batch)
            if step_number == 0 and (not distributed or jax.process_index() == 0):
                initial_prediction = predict_host_chunks(
                    unreplicate(params), z_host, predict_fn=predict_fn)
                initial_train_loss = float(np.mean((initial_prediction - y_host) ** 2))
                initial_test_loss = 0.0
                if initial_test is not None:
                    initial_test_prediction = predict_host_chunks(
                        unreplicate(params), initial_test[0], predict_fn=predict_fn)
                    initial_test_loss = float(
                        np.mean((initial_test_prediction - initial_test[1]) ** 2))
                history.append([initial_train_loss, initial_test_loss])
            params, opt_state, loss = step(
                params, opt_state, z_batch, y_batch, mask)
            if ((step_number + 1) % config.log_every == 0
                    or step_number + 1 == config.steps):
                train_loss = float(jax.device_get(loss[0]))
                if test_sampler is not None:
                    z_test, y_test = test_sampler.next(test_size)
                    test_prediction = predict_host_chunks(
                        unreplicate(params), z_test, predict_fn=predict_fn)
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
                checkpoint_callback(step_number + 1, unreplicate(params))
            if checkpoint_due and distributed and process_count > 1:
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices(
                    f"mlp_grouped_checkpoint_after_{step_number + 1}")
    if distributed and process_count > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mlp_grouped_training_complete")
    params = unreplicate(params)
    if not distributed or jax.process_index() == 0:
        z_train, y_train = sampler.next(min(global_batch_size, test_size))
        train_prediction = predict_host_chunks(params, z_train, predict_fn=predict_fn)
        if test_sampler is not None:
            z_test, y_test = test_sampler.next(test_size)
            test_prediction = predict_host_chunks(params, z_test, predict_fn=predict_fn)
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


def _normalize_parameter_cases(cases, parameter_domain):
    """Map physical seven-parameter cases into normalized branch coordinates."""
    cases = np.asarray(cases, dtype=np.float64)
    parameter_domain = validate_parameter_domain(parameter_domain)
    if cases.ndim != 2 or cases.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"parameter cases must have shape (N, {len(PARAMETER_NAMES)})")
    if not np.isfinite(cases).all():
        raise ValueError("parameter cases must be finite")
    normalized = np.empty_like(cases)
    for column, name in enumerate(PARAMETER_NAMES):
        lo, hi, scale = parameter_domain[name]
        if np.any(cases[:, column] < lo) or np.any(cases[:, column] > hi):
            raise ValueError(f"parameter cases outside {name} domain")
        normalized[:, column] = (
            (np.log(cases[:, column]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if scale == "log" else (cases[:, column] - lo) / (hi - lo))
    return normalized


def exact_deeponet_inputs(cases, results, parameter_domain, domain,
                          max_points=None):
    """Build padded groups with seven-parameter branch and two-coordinate trunk."""
    cases = np.asarray(cases, dtype=np.float64)
    if len(cases) != len(results) or len(cases) == 0:
        raise ValueError("cases and FV results must have equal nonzero length")
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
    """Flatten groups into ``[p, xi, seven parameters]`` pointwise inputs."""
    z_parts, y_parts = [], []
    for index in range(data["branch"].shape[0]):
        valid = np.asarray(data["mask"][index]) > 0.0
        trunk = np.asarray(data["trunk"][index])[valid]
        branch = np.repeat(
            np.asarray(data["branch"][index])[None, :], len(trunk), axis=0)
        z_parts.append(np.concatenate((trunk, branch), axis=1))
        y_parts.append(np.asarray(data["target"][index])[valid])
    if not z_parts:
        return np.empty((0, 9), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(z_parts), np.concatenate(y_parts)
