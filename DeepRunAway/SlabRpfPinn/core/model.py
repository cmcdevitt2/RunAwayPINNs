"""MLP and DeepONet architectures plus probability-constrained prediction."""

from __future__ import annotations

import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from core.training_config import TrainingConfig


def _apply_mlp(params, z):
    h = z
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def mlp_raw(params, z):
    if isinstance(params, dict):
        branch = _apply_mlp(params["branch"], z[..., 2:])
        trunk = _apply_mlp(params["trunk"], z[..., :2])
        return jnp.sum(branch * trunk, axis=-1) + params["bias"]
    return _apply_mlp(params, z)[..., 0]


def probability_transform(raw):
    """Map unconstrained output to the open probability interval (0, 1)."""
    return jax.nn.sigmoid(raw)


def probability(params, z):
    return probability_transform(mlp_raw(params, z))


predict = jax.jit(probability)


def _init_dense_mlp(key, input_dim, output_dim, width, depth):
    dims = [input_dim] + [width] * depth + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    return [{
        "W": math.sqrt(2.0 / dims[i]) * jax.random.normal(
            keys[i], (dims[i], dims[i + 1]), dtype=jnp.float64),
        "b": jnp.zeros((dims[i + 1],), dtype=jnp.float64),
    } for i in range(len(keys))]


def init_mlp(key, width=32, depth=4):
    return _init_dense_mlp(key, 8, 1, width, depth)


def init_deeponet(key, latent_width=64, branch_width=32, branch_depth=3,
                  trunk_width=32, trunk_depth=3):
    branch_key, trunk_key = jax.random.split(key)
    return {
        "branch": _init_dense_mlp(
            branch_key, 6, latent_width, branch_width, branch_depth),
        "trunk": _init_dense_mlp(
            trunk_key, 2, latent_width, trunk_width, trunk_depth),
        "bias": jnp.asarray(0.0, dtype=jnp.float64),
    }


def init_model(key, config: TrainingConfig):
    if config.model_type == "deeponet":
        return init_deeponet(
            key, latent_width=config.latent_width,
            branch_width=config.branch_width, branch_depth=config.branch_depth,
            trunk_width=config.trunk_width, trunk_depth=config.trunk_depth)
    return init_mlp(key, width=config.width, depth=config.depth)


def deeponet_probability(params, branch_z, trunk_z):
    """Evaluate DeepONet output for grouped cases and phase-space points."""
    branch = _apply_mlp(params["branch"], branch_z)
    shape = trunk_z.shape
    trunk = _apply_mlp(params["trunk"], trunk_z.reshape((-1, 2)))
    trunk = trunk.reshape(shape[:-1] + (trunk.shape[-1],))
    raw = jnp.sum(branch[..., None, :] * trunk, axis=-1) + params["bias"]
    return probability_transform(raw)


def save_model(path, params):
    """Save flattened model parameters in a portable NumPy archive."""
    weights, _ = ravel_pytree(params)
    np.savez(path, weights=np.asarray(jax.device_get(weights), dtype=np.float64))


def load_model(path, *, model_type="mlp", width=32, depth=4,
               latent_width=64, branch_width=32, branch_depth=3,
               trunk_width=32, trunk_depth=3):
    """Restore flattened parameters using recorded architecture settings."""
    if model_type == "deeponet":
        template = init_deeponet(
            jax.random.PRNGKey(0), latent_width=latent_width,
            branch_width=branch_width, branch_depth=branch_depth,
            trunk_width=trunk_width, trunk_depth=trunk_depth)
    else:
        template = init_mlp(jax.random.PRNGKey(0), width=width, depth=depth)
    _, unravel = ravel_pytree(template)
    with np.load(path, allow_pickle=False) as data:
        weights = jnp.asarray(data["weights"], dtype=jnp.float64)
    return unravel(weights)
