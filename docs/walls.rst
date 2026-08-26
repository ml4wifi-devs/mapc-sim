Monte Carlo walls
=================

Obstacles attenuate the radio signal, and the simulator accounts for them through
the additive (dB) wall matrix passed to :func:`mapc_sim.sim.network_data_rate`.
Historically that matrix had to be prepared by hand -- for every pair of nodes one
had to decide whether a wall separates them. The
:mod:`mapc_sim.experimental.walls` module replaces this bookkeeping with a
*geometric* description of the environment: a list of rectangles, each with its own
attenuation coefficient.

The model
---------

An obstacle is a rectangle :math:`W_w` (position of a corner, width, height,
rotation angle) made of a material of attenuation :math:`\alpha_w` expressed in
dB per meter. A link between nodes :math:`i` and :math:`j` is the segment
:math:`\overline{p_i p_j}`, and its attenuation is the sum of the contributions of
all obstacles it traverses:

.. math::

    L_{ij} = \sum_w \alpha_w \, \left| \overline{p_i p_j} \cap W_w \right| ,

where :math:`|\cdot|` is the length of the traversed part. Note that the loss
depends on *how* the link crosses the wall: a grazing link is attenuated more than
a perpendicular one, and a link that only clips the corner of a wall is barely
attenuated at all. This is the main modelling gain over the binary
"is there a wall in between?" description.

Monte Carlo estimation
----------------------

Computing :math:`\left| \overline{p_i p_j} \cap W_w \right|` analytically means
clipping a segment against a rotated rectangle: branches, sorted intersection
points, and special cases (parallel edges, degenerate segments). None of this is
friendly to ``jit``, ``vmap``, or automatic differentiation.

Instead we integrate by Monte Carlo. Write the intersection length as an integral
along the link, parametrized by :math:`t \in [0, 1]`:

.. math::

    \left| \overline{p_i p_j} \cap W_w \right| = d_{ij} \int_0^1
    \mathbb{1}\left[ p_i + t (p_j - p_i) \in W_w \right] \, \mathrm{d}t ,
    \qquad d_{ij} = \| p_j - p_i \| ,

and estimate the integral by sampling :math:`N` points uniformly along the segment
and counting how many of them land inside the obstacle:

.. math::

    \hat{L}_{ij} = d_{ij} \sum_w \alpha_w \frac{K_w}{N} , \qquad
    K_w = \sum_{n=1}^{N} \mathbb{1}\left[ p_i + t_n (p_j - p_i) \in W_w \right] ,
    \qquad t_n \sim \mathcal{U}(0, 1) .

Testing whether a point lies inside a rotated rectangle is a rotation followed by
two comparisons, so the whole estimator is a single branch-free expression which
maps over pairs of nodes and over obstacles. Arbitrarily many overlapping walls are
supported for free, and the estimator is unbiased,
:math:`\mathbb{E}\hat{L}_{ij} = L_{ij}`.

The same samples are used for both directions of a link (the estimate is computed
for unordered pairs), so the resulting matrix is symmetric -- the channel stays
reciprocal.

Usage
-----

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from mapc_sim.experimental.walls import Wall, stack, wall_attenuation
    from mapc_sim.sim import network_data_rate

    key = jax.random.PRNGKey(42)
    pos = jnp.array([[0., 0.], [10., 0.]])

    walls = stack([
        # a 0.2 m thick concrete wall of 25 dB/m, rotated by 30 degrees
        Wall.create(xy=jnp.array([4., -10.]), wh=jnp.array([0.2, 20.]), attenuation=25., angle=30.),
        # a thin partition wall
        Wall.create(xy=jnp.array([7., -10.]), wh=jnp.array([0.1, 20.]), attenuation=30.),
    ])

    walls_key, sim_key = jax.random.split(key)
    loss_gain = wall_attenuation(walls_key, pos, walls, n_samples=256)

    data_rate = network_data_rate(sim_key, tx, pos, mcs, tx_power, sigma, loss_gain)

The ``loss_gain`` matrix is additive, so antenna gains are simply subtracted from
the wall attenuation, see :doc:`antennas`:

.. code-block:: python

    loss_gain = wall_attenuation(walls_key, pos, walls) - link_gain(sector(pos, boresight))

Obstacles can be drawn with :func:`~mapc_sim.experimental.walls.plot_walls`:

.. code-block:: python

    from matplotlib import pyplot as plt
    from mapc_sim.experimental.walls import plot_walls

    ax = plot_walls(walls)
    ax.scatter(pos[:, 0], pos[:, 1])
    plt.show()

:class:`~mapc_sim.experimental.walls.Wall` is a standard dataclass registered as a
JAX pytree (:func:`jax.tree_util.register_dataclass`), so obstacles are modified
with :func:`dataclasses.replace`:

.. code-block:: python

    from dataclasses import replace

    thicker = replace(wall, wh=wall.wh.at[0].set(0.5))

A stacked environment is a pytree too, so it can be passed through ``jit`` and
``vmap``, and the attenuation matrix is differentiable with respect to the
attenuation coefficients (the geometry enters through a hard indicator, so the
gradient with respect to the *size and position* of a wall is zero almost
everywhere).

.. _sampling-noise-fading:

Sampling noise as a fading model
--------------------------------

The Monte Carlo error is usually something to be minimized. Here it can be put to
work. The number of samples inside obstacle :math:`w` is binomial,
:math:`K_w \sim \mathrm{Binomial}(N, p_w)` with
:math:`p_w = \left| \overline{p_i p_j} \cap W_w \right| / d_{ij}`, so the estimator
is unbiased with variance

.. math::

    \operatorname{Var} \hat{L}_{ij} = d_{ij}^2 \sum_w \alpha_w^2
    \frac{p_w (1 - p_w)}{N} .

In other words, drawing a fresh key at every simulation step perturbs the link
budget by a zero-mean, approximately Gaussian (by the central limit theorem, for
moderate :math:`N`) term whose standard deviation is controlled by :math:`N`. This
is exactly the shape of the log-normal shadowing term that the simulator otherwise
adds explicitly as a normal random variable of standard deviation ``sigma``
(the value derived from ns-3 simulations with Nakagami fading). The sampling noise
can therefore *replace* the explicit random normal: pick :math:`N` such that
:func:`~mapc_sim.experimental.walls.attenuation_std` matches the desired
``sigma``, and set the simulator's ``sigma`` to a smaller residual value (or to
zero) to avoid counting the same randomness twice.

The estimator standard deviation is available directly:

.. code-block:: python

    from mapc_sim.experimental.walls import attenuation_std

    attenuation_std(key, pos, walls, n_samples=256)  # dB, per pair of nodes

There is a physical argument for this substitution. Shadow fading is, to a large
extent, *caused* by the uncertainty about the environment: the exact position and
thickness of the obstacles a link crosses is never known. The Monte Carlo estimator
makes this uncertainty explicit and propagates it into the link budget with the
right magnitude, instead of adding an independent normal variable on top of a
deterministic geometry.

Caveats
^^^^^^^

The correspondence is approximate and should be used with the following in mind:

* The noise is **zero for unobstructed links** and for links entirely immersed in a
  single obstacle (:math:`p_w \in \{0, 1\}`). This is arguably a feature -- a clean
  line-of-sight link really is subject to much weaker shadowing -- but it means the
  sampling noise cannot model the fading of an open-space link. Keep a small
  ``sigma`` for that component.
* The noise is **discrete**: :math:`\hat{L}_{ij}` is a multiple of
  :math:`\alpha_w d_{ij} / N`. The Gaussian approximation requires
  :math:`N p_w (1 - p_w) \gtrsim 10`.
* The noise is **bounded and skewed** for small :math:`p_w`, unlike a normal
  variable.
* The noise is **spatially correlated**: links that share geometry see similar
  perturbations, which mirrors the correlation of real shadow fading, but it also
  means the perturbations of different links are not independent.
* The magnitude is **not freely tunable per link** -- it is dictated by the geometry
  and by :math:`N`, which is shared by all links. Matching a target ``sigma``
  exactly for every link is not possible; matching it on average is.

Choosing ``n_samples``
----------------------

Two regimes are useful:

* **Accurate geometry** (:math:`N` large, e.g. :math:`N \geq 4096`): the estimator
  is effectively deterministic and the fading is left to the simulator's ``sigma``.
  A wall of thickness :math:`\delta` is resolved with a relative error of about
  :math:`\sqrt{d_{ij} / (N \delta)}`, so thin walls in large scenarios need more
  samples.
* **Noise as fading** (:math:`N` moderate): the estimator doubles as the
  shadowing model, as described above. The right value depends on the geometry --
  pick it so that :func:`~mapc_sim.experimental.walls.attenuation_std` is of the
  order of the desired ``sigma`` for typical links. Since the standard deviation
  scales as :math:`N^{-1/2}`, a single evaluation is enough to extrapolate:
  for a 10 m link crossing 0.35 m of a 25 dB/m wall and 0.1 m of a 30 dB/m one,
  :math:`N = 32` yields 7.7 dB and :math:`N \approx 470` is needed for 2 dB.

The cost is :math:`O(n^2 N |W|)` point-in-rectangle tests per call, all vectorized,
so the second regime is essentially free.
