r"""
Antenna gain models.

**Convention.** Gains are *classical*, i.e. absolute and expressed in dBi --
relative to an isotropic radiator, not relative to the maximum of the pattern.
An isotropic antenna therefore has a gain of 0 dBi everywhere
(:func:`isotropic`), and a directional antenna has a positive gain around its
boresight and a negative one in the nulls. The alternative convention (gain
normalized to the maximum of the pattern, i.e. always non-positive) was
rejected because:

* the link budget stays additive and physically interpretable -- the received
  power is ``tx_power - path_loss + G_tx + G_rx`` with the same numbers that a
  data sheet reports, so ``tx_power`` keeps meaning conducted power and not EIRP;
* directivity is what makes beamforming useful: with the relative convention the
  gain of the main lobe is 0 dB and the benefit of narrowing the beam silently
  moves into ``tx_power``, which has to be re-normalized per antenna;
* comparing two antennas with different beamwidths requires absolute gains.

The simulator accepts a single additive ``loss_gain`` matrix (dB), where losses
are positive and gains are negative. Antenna gains therefore enter with a minus
sign, and the gain of a link is the sum of the transmitter and the receiver
gains:

.. math::

    \mathrm{loss\_gain}_{ij} = L_{ij} - \left( G_{ij} + G_{ji} \right) ,

where :math:`L_{ij}` is the wall attenuation (see
:mod:`mapc_sim.experimental.walls`) and :math:`G_{ij}` is the gain of the
antenna of node :math:`i` in the direction of node :math:`j`. The second term is
computed by :func:`link_gain`.
"""

import jax
import jax.numpy as jnp

__all__ = ['bearing', 'isotropic', 'sector_max_gain', 'sector', 'link_gain']


def bearing(pos: jax.Array) -> jax.Array:
    r"""
    Direction from each node to each other node.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.

    Returns
    -------
    Array
        Matrix of shape ``(n, n)`` where the entry ``[i, j]`` is the angle (degrees,
        counterclockwise from the x axis) of the direction from node ``i`` to node ``j``.
    """

    delta = pos[None, ...] - pos[:, None, :]
    return jnp.rad2deg(jnp.arctan2(delta[..., 1], delta[..., 0]))


def isotropic(pos: jax.Array, max_gain: float | jax.Array = 0.) -> jax.Array:
    r"""
    Gain matrix of an isotropic antenna (dBi).

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.
    max_gain: Numeric
        Gain of the antenna (dBi). Zero for an ideal isotropic radiator.

    Returns
    -------
    Array
        Constant matrix of shape ``(n, n)``.
    """

    n = pos.shape[0]
    return jnp.full((n, n), max_gain, dtype=float)


def sector_max_gain(beamwidth: float | jax.Array) -> jax.Array:
    r"""
    Peak gain (dBi) of an ideal antenna radiating uniformly within ``beamwidth`` in
    the azimuth plane, :math:`G_{\max} = 10 \log_{10} (360 / \theta_{3\mathrm{dB}})`.

    This is a planar approximation which ignores the elevation pattern, so it
    underestimates the gain of a real antenna. Pass ``max_gain`` explicitly to
    :func:`sector` to use a value from a data sheet instead.

    Parameters
    ----------
    beamwidth: Numeric
        Half power (3 dB) beamwidth in the azimuth plane (degrees).

    Returns
    -------
    Array
        Peak gain (dBi).
    """

    return 10. * jnp.log10(360. / jnp.asarray(beamwidth, dtype=float))


def sector(
        pos: jax.Array,
        boresight: jax.Array,
        beamwidth: float | jax.Array = 65.,
        max_gain: float | jax.Array | None = None,
        front_back_ratio: float | jax.Array = 20.
) -> jax.Array:
    r"""
    Gain matrix (dBi) of a sectored antenna with the parabolic (in dB) pattern of the
    3GPP/TGax models:

    .. math::

        G(\theta) = G_{\max} - \min \left( 12 \left(
        \frac{\theta - \theta_0}{\theta_{3\mathrm{dB}}} \right)^2, A_m \right) ,

    where :math:`\theta_0` is the boresight of the antenna, :math:`\theta_{3\mathrm{dB}}`
    is the half power beamwidth, and :math:`A_m` is the front-to-back ratio. The angle
    difference is wrapped to :math:`[-180^\circ, 180^\circ)`.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.
    boresight: Array
        Direction in which the antenna of each node points (degrees, counterclockwise
        from the x axis), shape ``(n,)``.
    beamwidth: Numeric
        Half power (3 dB) beamwidth (degrees).
    max_gain: Numeric, optional
        Peak gain of the antenna (dBi). Defaults to :func:`sector_max_gain` of the beamwidth.
    front_back_ratio: Numeric
        Maximum attenuation of the pattern relative to its peak (dB).

    Returns
    -------
    Array
        Matrix of shape ``(n, n)`` where the entry ``[i, j]`` is the gain of the antenna
        of node ``i`` in the direction of node ``j``.
    """

    max_gain = sector_max_gain(beamwidth) if max_gain is None else jnp.asarray(max_gain, dtype=float)

    angle = bearing(pos) - jnp.asarray(boresight, dtype=float)[:, None]
    angle = (angle + 180.) % 360. - 180.

    return max_gain - jnp.minimum(12. * (angle / beamwidth) ** 2, front_back_ratio)


def link_gain(gain: jax.Array) -> jax.Array:
    r"""
    Total gain of each link, i.e. the sum of the gains of the transmitting and the
    receiving antenna, :math:`G_{ij} + G_{ji}`.

    The result is symmetric (the channel is reciprocal) and should be *subtracted*
    from the ``loss_gain`` matrix of :func:`mapc_sim.sim.network_data_rate`:

    .. code-block:: python

        loss_gain = wall_attenuation(key, pos, walls) - link_gain(sector(pos, boresight))

    Parameters
    ----------
    gain: Array
        Matrix of antenna gains (dBi), e.g. built with :func:`sector`.

    Returns
    -------
    Array
        Symmetric matrix of the total link gains (dB).
    """

    return gain + gain.T
