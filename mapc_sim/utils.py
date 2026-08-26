import inspect
import warnings
from functools import partial, wraps
from typing import Callable

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


def accepts_walls(fn: Callable) -> Callable:
    r"""
    Decorator adding a legacy ``walls`` keyword argument to a function taking a
    ``loss_gain`` matrix.

    The simulator used to describe the environment with a binary adjacency matrix of
    walls, which it scaled internally by a fixed per-wall loss. It now takes a single
    additive ``loss_gain`` matrix in dB (losses positive, gains negative), which is more
    general -- it also accommodates the geometric wall model of
    :mod:`mapc_sim.experimental.walls` and the antenna gains of
    :mod:`mapc_sim.experimental.antenas`. This decorator keeps the old call style
    working:

    .. code-block:: python

        # legacy call, equivalent to loss_gain=binary_walls(walls, ENTERPRISE_WALL_LOSS)
        network_data_rate(key, tx, pos, mcs, tx_power, sigma, walls=walls)

        # ... and with a different wall loss
        network_data_rate(key, tx, pos, mcs, tx_power, sigma, walls=walls, wall_loss=RESIDENTIAL_WALL_LOSS)

    .. important::

        The legacy argument must be passed **by keyword**. A matrix passed positionally
        is always the ``loss_gain`` matrix, since the two cannot be told apart -- under
        ``jit`` the values are tracers, and a binary matrix is a valid ``loss_gain``
        matrix in its own right.

    Passing both ``walls`` and ``loss_gain`` raises a ``TypeError``. Using ``walls``
    raises a ``DeprecationWarning``.

    Parameters
    ----------
    fn: Callable
        A function with a ``loss_gain`` parameter, e.g. :func:`mapc_sim.sim.network_data_rate`.

    Returns
    -------
    Callable
        The same function, additionally accepting ``walls`` and ``wall_loss`` keywords.
    """

    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    loss_gain_pos = [p.name for p in parameters].index('loss_gain')

    @wraps(fn)
    def wrapper(*args, walls: jax.Array = None, wall_loss: jax.Array = ENTERPRISE_WALL_LOSS, **kwargs):
        if walls is None:
            return fn(*args, **kwargs)

        if len(args) > loss_gain_pos or 'loss_gain' in kwargs:
            raise TypeError(f'{fn.__name__}() got both `walls` and `loss_gain`, pass only one of them')

        warnings.warn(
            '`walls` is deprecated, pass `loss_gain=binary_walls(walls, wall_loss)` instead',
            DeprecationWarning, stacklevel=2
        )
        return fn(*args, loss_gain=binary_walls(walls, wall_loss), **kwargs)

    legacy = [
        inspect.Parameter('walls', inspect.Parameter.KEYWORD_ONLY, default=None, annotation=jax.Array),
        inspect.Parameter('wall_loss', inspect.Parameter.KEYWORD_ONLY, default=ENTERPRISE_WALL_LOSS, annotation=jax.Array)
    ]
    wrapper.__signature__ = signature.replace(parameters=parameters + legacy)

    return wrapper


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
