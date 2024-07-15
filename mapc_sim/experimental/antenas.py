import chex
import jax.numpy as jnp


def isotropic(pos: chex.Array) -> chex.Array:
    """Calculate gain matrix for each point to point communication.

    Parameters
    ----------
    pos: Array
        An array of 2d positions for `n` points

    Returns
    -------
    Array
        Antena gain for each point in pos.

    """
    npoints = pos.shape[0]

    return jnp.zeros((npoints, npoints))
