"""Core numerical utilities for ofdtd."""

import jax
import jax.numpy as jnp

from ofdtd.config import SimulationConfig


def make_grid(config: SimulationConfig) -> jax.Array:
    """Create a zero-valued scalar grid.

    Parameters
    ----------
    config:
        Simulation configuration.

    Returns
    -------
    jax.Array
        A JAX array with shape ``(nx, ny, nz)``.
    """
    return jnp.zeros((config.nx, config.ny, config.nz))
