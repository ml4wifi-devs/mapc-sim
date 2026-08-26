from functools import partial

import jax
import jax.numpy as jnp

from mapc_sim.constants import *


def tgax_path_loss(distance: jax.Array, loss_gain: jax.Array, breaking_point: jax.Array) -> jax.Array:
    r"""
    Calculates the path loss according to the TGax channel model [1]_.

    Parameters
    ----------
    distance: Array
        Distance between nodes
    loss_gain: Array
        Matrix of the additional loss (positive) and gain (negative) for each pair of nodes, in dB.
        It accounts for the attenuation caused by walls and for the antenna gains, see
        :mod:`mapc_sim.experimental.walls` and :mod:`mapc_sim.experimental.antenas`.
        A binary adjacency matrix of walls ``walls`` used in the previous versions of the simulator
        corresponds to ``loss_gain = wall_loss * walls``, cf. :func:`mapc_sim.utils.binary_walls`.
    breaking_point: Array
        Breaking point of the path loss model

    Returns
    -------
    Array
        Path loss in dB

    References
    ----------
    .. [1] https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
    """

    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)
    return (40.05 + 20 * jnp.log10((jnp.minimum(distance, breaking_point) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > breaking_point) * 35 * jnp.log10(distance / breaking_point) + loss_gain)


residential_tgax_path_loss = partial(tgax_path_loss, breaking_point=RESIDENTIAL_BREAKING_POINT)
enterprise_tgax_path_loss = partial(tgax_path_loss, breaking_point=ENTERPRISE_BREAKING_POINT)
default_path_loss = enterprise_tgax_path_loss


def binary_walls(walls: jax.Array, wall_loss: jax.Array = ENTERPRISE_WALL_LOSS) -> jax.Array:
    r"""
    Converts a binary adjacency matrix of walls into a ``loss_gain`` matrix (dB).

    This is the model used by the previous versions of the simulator, where each pair of
    nodes is either separated by a single wall of a fixed loss or not separated at all.
    For a geometric description of the environment see :mod:`mapc_sim.experimental.walls`.

    Parameters
    ----------
    walls: Array
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).
    wall_loss: Array
        Loss of a single wall (dB), e.g. :const:`mapc_sim.constants.ENTERPRISE_WALL_LOSS`.

    Returns
    -------
    Array
        Matrix of the loss caused by walls (dB).
    """

    return wall_loss * walls


def logsumexp_db(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""
    Computes :func:`jax.nn.logsumexp` for dB i.e. :math:`10 * \log_{10}(\sum_i b_i 10^{a_i/10})`

    This function is equivalent to

    .. code-block:: python

        interference_lin = jnp.power(10, a / 10)
        interference_lin = (b * interference_lin).sum()
        interference = 10 * jnp.log10(interference_lin)


    Parameters
    ----------
    a: Array
        Parameters are the same as for :func:`jax.nn.logsumexp`
    b: Array
        Parameters are the same as for :func:`jax.nn.logsumexp`

    Returns
    -------
    Array
        ``logsumexp`` for dB
    """

    LOG10DIV10 = jnp.log(10.) / 10.
    return jax.nn.logsumexp(a=LOG10DIV10 * a, b=b) / LOG10DIV10
