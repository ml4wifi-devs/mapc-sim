import chex
import jax.numpy as jnp


def free_space(pos: chex.Array) -> chex.Array:
    """Calculate free space loss

    Parameters
    ----------
    pos: Array
        An array of 2d positions for `n` points

    Returns
    -------
        Transmission loss for every pair of points in `pos`.

    """
    npoints = pos.shape[0]

    return jnp.zeros((npoints, npoints))
