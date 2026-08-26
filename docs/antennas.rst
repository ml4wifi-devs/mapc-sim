Antenna gains and beamforming
=============================

The simulator takes a single additive ``loss_gain`` matrix expressed in dB, which
is added to the path loss. Everything that attenuates a link enters with a plus
sign (walls, see :doc:`walls`) and everything that strengthens it with a minus
sign (antenna gains):

.. math::

    \mathrm{loss\_gain}_{ij} = L_{ij} - \left( G_{ij} + G_{ji} \right) ,

where :math:`G_{ij}` is the gain of the antenna of node :math:`i` in the direction
of node :math:`j`. The link gain is the sum of the transmitter and the receiver
gain, which is what :func:`~mapc_sim.experimental.antenas.link_gain` computes; the
result is symmetric, so the channel stays reciprocal.

Which gain convention?
----------------------

A pattern can be described either by its **absolute** gain in dBi (relative to an
isotropic radiator, the convention of every data sheet) or **relative to the
maximum** of the pattern (always non-positive, the convention of a normalized
radiation pattern). This module uses the **absolute, classical dBi convention**:

* The link budget stays additive and directly interpretable:
  :math:`P_{rx} = P_{tx} - PL + G_{tx} + G_{rx}`, with the same numbers a data
  sheet reports. ``tx_power`` keeps its meaning of conducted power rather than
  silently becoming EIRP.
* Directivity is the entire point of beamforming. With the relative convention the
  main lobe has a gain of 0 dB, so narrowing the beam brings no gain at all unless
  the missing directivity is folded into ``tx_power`` -- i.e. re-normalized for
  every antenna and every beamwidth.
* Comparing antennas with different beamwidths, or a directional antenna against an
  omnidirectional one, requires absolute gains. The relative convention can only
  compare directions of one fixed pattern.

The price is that the modeller must supply a peak gain. When it is not known,
:func:`~mapc_sim.experimental.antenas.sector_max_gain` provides the planar
estimate :math:`G_{\max} = 10 \log_{10}(360 / \theta_{3\mathrm{dB}})`, which
conserves radiated power in the azimuth plane; an ideal isotropic antenna
(:math:`\theta_{3\mathrm{dB}} = 360^\circ`) then correctly gets 0 dBi. Since the
elevation pattern is ignored, this underestimates the gain of a real antenna --
pass ``max_gain`` explicitly when a measured value is available.

Sectored pattern
----------------

:func:`~mapc_sim.experimental.antenas.sector` implements the parabolic (in dB)
pattern used by the 3GPP and TGax models:

.. math::

    G(\theta) = G_{\max} - \min \left( 12 \left(
    \frac{\theta - \theta_0}{\theta_{3\mathrm{dB}}} \right)^2, A_m \right) ,

with boresight :math:`\theta_0`, half power beamwidth
:math:`\theta_{3\mathrm{dB}}` (the pattern is 3 dB below its peak at
:math:`\pm \theta_{3\mathrm{dB}} / 2`), and front-to-back ratio :math:`A_m`, which
caps the attenuation of the back lobe.

Usage
-----

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from mapc_sim.experimental.antenas import isotropic, link_gain, sector
    from mapc_sim.experimental.walls import Wall, stack, wall_attenuation
    from mapc_sim.sim import network_data_rate

    walls_key, sim_key = jax.random.split(jax.random.PRNGKey(42))
    pos = jnp.array([[0., 0.], [30., 0.]])

    walls = stack([Wall.create(xy=jnp.array([14., -10.]), wh=jnp.array([0.2, 20.]), attenuation=25.)])
    attenuation = wall_attenuation(walls_key, pos, walls, n_samples=1024)

    # each node points its antenna at the other one
    boresight = jnp.array([0., 180.])
    loss_gain = attenuation - link_gain(sector(pos, boresight, beamwidth=30.))

    # ... or uses an omnidirectional antenna, which contributes nothing
    loss_gain_omni = attenuation - link_gain(isotropic(pos))

    data_rate = network_data_rate(sim_key, tx, pos, None, tx_power, sigma, loss_gain)

Steering the beam is a matter of changing ``boresight``, which is a plain array:
the gain matrix is differentiable with respect to it (away from the front-to-back
cap, where the gradient is zero), so beam directions can be optimized with the
usual JAX machinery.
