"""Optimizers used by model-training workflows."""

from __future__ import annotations


def make_soap_optimizer(config):
    """Build SOAP with configured learning-rate schedule and preconditioning."""
    import optax
    from soap_jax import soap

    learning_rate = config.learning_rate
    if config.learning_rate_schedule == "cosine":
        learning_rate = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=max(1, config.learning_rate_decay_steps or config.steps),
            alpha=config.learning_rate_final_fraction,
        )
    return soap(
        learning_rate=learning_rate, b1=config.soap_b1, b2=config.soap_b2,
        shampoo_beta=config.soap_shampoo_beta, eps=config.soap_eps,
        weight_decay=config.soap_weight_decay, correct_bias=config.soap_correct_bias,
        precondition_frequency=config.soap_precondition_frequency,
        max_precond_dim=config.soap_max_precond_dim,
        precondition_1d=config.soap_precondition_1d,
    )
