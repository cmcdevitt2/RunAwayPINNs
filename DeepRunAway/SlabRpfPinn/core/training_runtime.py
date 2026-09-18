"""Shared JAX device and host-sharding utilities for training loops."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from core.model import predict


def training_devices(requested):
    """Resolve the local GPU set used by data-parallel training."""
    devices = tuple(jax.local_devices(backend="gpu"))
    if not devices:
        raise RuntimeError("multi-device PINN training requires visible GPU devices")
    if requested == 0:
        return devices
    if requested > len(devices):
        raise ValueError(
            f"n_devices={requested} requested, but only {len(devices)} local GPUs are visible")
    return devices[:requested]


def shard_with_mask(values, devices, valid_count=None):
    """Pad a leading-axis array and place equal-sized shards on local GPUs."""
    values = np.asarray(values)
    n_devices = len(devices)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("cannot shard an empty or scalar array")
    valid_count = values.shape[0] if valid_count is None else int(valid_count)
    if not 0 <= valid_count <= values.shape[0]:
        raise ValueError("valid_count must be within the leading-axis length")
    shard_size = (values.shape[0] + n_devices - 1) // n_devices
    padded_size = shard_size * n_devices
    padded = np.zeros((padded_size,) + values.shape[1:], dtype=values.dtype)
    padded[:values.shape[0]] = values
    mask = np.zeros(padded_size, dtype=np.float64)
    mask[:valid_count] = 1.0
    shards = [padded[start:start + shard_size]
              for start in range(0, padded_size, shard_size)]
    masks = [mask[start:start + shard_size]
             for start in range(0, padded_size, shard_size)]
    return np.stack(shards), np.stack(masks), int(values.shape[0])


def unreplicate(tree):
    """Take the first value from each single-device replica."""
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def replicate(tree, devices):
    """Replicate pytree leaves as host-local leading-axis arrays for pmap."""
    return jax.tree_util.tree_map(
        lambda value: np.stack([np.asarray(value) for _ in devices]), tree)


def predict_host_chunks(params, z, *, chunk_size=65536, predict_fn=None):
    """Evaluate predictions without materializing a large device batch."""
    if predict_fn is None:
        predict_fn = predict
    z = np.asarray(z, dtype=np.float64)
    values = []
    for start in range(0, len(z), chunk_size):
        values.append(np.asarray(jax.device_get(
            predict_fn(params, jnp.asarray(z[start:start + chunk_size])))))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def shard_leading(values, devices):
    """Place a fixed leading-axis case batch on local devices."""
    values = np.asarray(values)
    n_devices = len(devices)
    shard_size = (values.shape[0] + n_devices - 1) // n_devices
    padded_size = shard_size * n_devices
    padded = np.zeros((padded_size,) + values.shape[1:], dtype=values.dtype)
    padded[:values.shape[0]] = values
    shards = [padded[start:start + shard_size]
              for start in range(0, padded_size, shard_size)]
    return np.stack(shards)
