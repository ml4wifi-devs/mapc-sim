import jax
import jax.numpy as jnp
from chex import Array

from mapc_sim.constants import *


def tgax_path_loss(distance: Array, walls: Array) -> Array:
    r"""
    Calculates the path loss according to the TGax channel model [1]_.

    Parameters
    ----------
    distance: Array
        Distance between nodes
    walls: Array
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).

    Returns
    -------
    Array
        Path loss in dB

    References
    ----------
    .. [1] https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
    """

    return (40.05 + 20 * jnp.log10((jnp.minimum(distance, BREAKING_POINT) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > BREAKING_POINT) * 35 * jnp.log10(distance / BREAKING_POINT) + WALL_LOSS * walls)


def logsumexp_db(a: Array, b: Array) -> Array:
    r"""
    Computes :ref:`jax.nn.logsumexp` for dB i.e. :math:`10 * log_10(\sum_i b_i 10^{a_i/10})`

    This function is equivalent to

    .. code-block:: python

        interference_lin = jnp.power(10, a / 10)
        interference_lin = (b * interference_lin).sum()
        interference = 10 * jnp.log10(interference_lin)


    Parameters
    ----------
    a: Array
        Parameters are the same as for :ref:`jax.nn.logsumexp`
    b: Array
        Parameters are the same as for :ref:`jax.nn.logsumexp`

    Returns
    -------
    Array
        `logsumexp` for dB
    """

    LOG10DIV10 = jnp.log(10.) / 10.
    return jax.nn.logsumexp(a=LOG10DIV10 * a, b=b) / LOG10DIV10
