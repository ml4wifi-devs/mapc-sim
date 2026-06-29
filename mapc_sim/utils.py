from functools import partial

import jax
import jax.numpy as jnp

from mapc_sim.constants import *


def tgax_path_loss(distance: jax.Array, walls: jax.Array, breaking_point: jax.Array, wall_loss: jax.Array) -> jax.Array:
    r"""
    Calculates the path loss according to the TGax channel model [1]_.

    Parameters
    ----------
    distance: Array
        Distance between nodes
    walls: Array
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).
    breaking_point: Array
        Breaking point of the path loss model
    wall_loss: Array
        Wall loss factor

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
            (distance > breaking_point) * 35 * jnp.log10(distance / breaking_point) + wall_loss * walls)


residential_tgax_path_loss = partial(tgax_path_loss, breaking_point=RESIDENTIAL_BREAKING_POINT, wall_loss=RESIDENTIAL_WALL_LOSS)
enterprise_tgax_path_loss = partial(tgax_path_loss, breaking_point=ENTERPRISE_BREAKING_POINT, wall_loss=ENTERPRISE_WALL_LOSS)
default_path_loss = enterprise_tgax_path_loss


def nakagami_fading_db(key: jax.random.PRNGKey, m: float, shape: tuple) -> jax.Array:
    r"""
    Samples Nakagami-m fading loss in dB for a matrix of wireless links.

    The fading factor :math:`g \sim \text{Gamma}(m, 1/m)` has mean 1 and variance :math:`1/m`,
    matching ns-3's ``NakagamiPropagationLossModel`` parameterization. Mean received power is
    preserved in linear scale; in dB the distribution is left-skewed (negative mean).

    Special cases: :math:`m = 1` reduces to Rayleigh fading (exponential power distribution);
    large :math:`m` approaches no fading (:math:`g \to 1`).

    Parameters
    ----------
    key: PRNGKey
        JAX random key.
    m: float
        Nakagami shape parameter (:math:`m \geq 0.5`). Higher values mean less fading depth.
    shape: tuple
        Output shape, typically ``signal_power.shape`` i.e. ``(n_tx, n_rx)``.

    Returns
    -------
    Array
        Fading loss in dB (negative values attenuate the signal).
    """
    g = jax.random.gamma(key, a=m, shape=shape) / m
    return 10.0 * jnp.log10(g)


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
