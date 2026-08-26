r"""
Monte Carlo model of walls (obstacles) in the environment.

The simulator (:func:`mapc_sim.sim.network_data_rate`) accepts an additive
``loss_gain`` matrix expressed in dB. This module builds the wall part of that
matrix from a *geometric* description of the environment instead of from a
hand-crafted adjacency matrix of "there is a wall between node ``i`` and node
``j``".

**Model.** An obstacle is a (possibly rotated) rectangle with a specific
attenuation coefficient :math:`\alpha` (dB/m of traversed material). A radio
link between nodes :math:`i` and :math:`j` is the segment
:math:`\overline{p_i p_j}`, and the attenuation of that link is the sum, over
all walls, of the attenuation coefficient times the length of the part of the
segment which lies inside the wall:

.. math::

    L_{ij} = \sum_w \alpha_w \, \left| \overline{p_i p_j} \cap W_w \right| .

**Estimation.** Instead of computing the segment-rectangle intersection
analytically (which involves branching and is not differentiable), the length is
estimated by Monte Carlo integration. We draw :math:`N` points uniformly along
the segment and count how many of them fall inside each wall:

.. math::

    \hat{L}_{ij} = d_{ij} \sum_w \alpha_w \frac{K_w}{N}, \qquad
    K_w = \sum_{n=1}^N \mathbb{1}\left[ p_i + t_n (p_j - p_i) \in W_w \right],
    \quad t_n \sim \mathcal{U}(0, 1),

where :math:`d_{ij} = \| p_j - p_i \|`. The estimator is unbiased,
:math:`\mathbb{E} \hat{L}_{ij} = L_{ij}`, it is a single ``vmap``-ed expression
(no branching, no sorting of intersection points), and it works for any number
of arbitrarily overlapping obstacles.

**Sampling noise as fading.** :math:`K_w \sim \mathrm{Binomial}(N, p_w)` with
:math:`p_w = |\overline{p_i p_j} \cap W_w| / d_{ij}`, hence

.. math::

    \operatorname{Var} \hat{L}_{ij} = d_{ij}^2 \sum_w \alpha_w^2
    \frac{p_w (1 - p_w)}{N} .

The Monte Carlo error is therefore *not* a nuisance: it is a zero-mean,
approximately Gaussian (for moderate :math:`N`) perturbation of the link budget,
i.e. exactly the shape of the log-normal shadowing term that the simulator adds
as ``sigma``. Drawing a fresh ``key`` per simulation step and picking :math:`N`
so that :func:`attenuation_std` matches the desired :math:`\sigma` lets the
sampling noise *replace* the explicit random normal in the fading model, at no
extra cost. See :func:`attenuation_std` for the caveats (the noise vanishes for
links that miss every wall or are fully immersed in one, and it is spatially
correlated in the way real shadowing is: two nearby links see nearly the same
obstacle geometry).
"""

from typing import Optional, Sequence, TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import matplotlib

__all__ = ['Wall', 'rotation_matrix_2d', 'stack', 'free_space', 'wall_attenuation', 'attenuation_std', 'plot_walls']


@chex.dataclass
class Wall:
    r"""
    A rectangular obstacle. Registered as a JAX pytree, so it can be passed
    through ``jit``, ``vmap``, and differentiated with respect to its geometry
    and attenuation.

    Attributes
    ----------
    xy: Array
        Position of the reference (lower-left) corner of the rectangle, shape ``(2,)``.
    wh: Array
        Width and height of the rectangle (m), shape ``(2,)``.
    rot: Array
        Rotation matrix of the rectangle, shape ``(2, 2)``. Use :func:`rotation_matrix_2d`
        or the :meth:`create` constructor.
    attenuation: Array
        Attenuation coefficient of the material (dB/m).
    """

    xy: chex.Array
    wh: chex.Array
    rot: chex.Array
    attenuation: chex.Array

    @classmethod
    def create(cls, xy: chex.Array, wh: chex.Array, attenuation: chex.Numeric, angle: chex.Numeric = 0.) -> 'Wall':
        r"""
        Creates a wall from its corner, size, attenuation, and rotation angle.

        Parameters
        ----------
        xy: Array
            Position of the lower-left corner of the rectangle (before rotation).
        wh: Array
            Width and height of the rectangle (m).
        attenuation: Numeric
            Attenuation coefficient of the material (dB/m).
        angle: Numeric
            Counterclockwise rotation angle around ``xy`` (degrees).

        Returns
        -------
        Wall
            The obstacle.
        """

        return cls(
            xy=jnp.asarray(xy, dtype=float),
            wh=jnp.asarray(wh, dtype=float),
            rot=rotation_matrix_2d(angle),
            attenuation=jnp.asarray(attenuation, dtype=float)
        )

    def contains(self, p: chex.Array) -> chex.Array:
        r"""
        Checks whether a point lies inside the rectangle.

        Parameters
        ----------
        p: Array
            A 2d point.

        Returns
        -------
        Array
            A boolean scalar.
        """

        local = self.rot.T @ (p - self.xy)
        return jnp.logical_and(jnp.all(local >= 0.), jnp.all(local <= self.wh))


def rotation_matrix_2d(angle: chex.Numeric) -> chex.Array:
    r"""
    Counterclockwise rotation matrix.

    Parameters
    ----------
    angle: Numeric
        Rotation angle (degrees).

    Returns
    -------
    Array
        Rotation matrix of shape ``(2, 2)``.
    """

    angle = jnp.deg2rad(jnp.asarray(angle, dtype=float))
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def stack(walls: Sequence[Wall]) -> Wall:
    r"""
    Stacks a sequence of walls into a single batched :class:`Wall` pytree,
    i.e. a ``Wall`` whose leaves have a leading axis of size ``len(walls)``.
    This is the representation expected by :func:`wall_attenuation`.

    Parameters
    ----------
    walls: Sequence[Wall]
        Obstacles present in the environment.

    Returns
    -------
    Wall
        Batched obstacles.
    """

    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *walls)


def _pair_keys(key: chex.PRNGKey, n: int) -> chex.PRNGKey:
    r"""Builds a symmetric ``(n, n)`` matrix of keys, one per unordered pair of nodes."""

    idx = jnp.arange(n)
    lo = jnp.minimum(idx[:, None], idx[None, :])
    hi = jnp.maximum(idx[:, None], idx[None, :])
    return jax.vmap(jax.vmap(jax.random.fold_in, in_axes=(None, 0)), in_axes=(None, 0))(key, lo * n + hi)


def _pair_hits(key: chex.PRNGKey, a: chex.Array, b: chex.Array, walls: Wall, n_samples: int) -> chex.Array:
    r"""Fraction of the uniformly sampled points of the segment ``ab`` inside each wall."""

    t = jax.random.uniform(key, shape=(n_samples, 1))
    points = a + t * (b - a)
    inside = jax.vmap(jax.vmap(Wall.contains, in_axes=(0, None)), in_axes=(None, 0))(walls, points)
    return inside.mean(axis=0)


def _hit_matrix(key: chex.PRNGKey, pos: chex.Array, walls: Wall, n_samples: int) -> chex.Array:
    r"""``(n, n, n_walls)`` matrix of the estimated fractions of the links inside each wall."""

    n = pos.shape[0]
    idx = jnp.arange(n)
    lo = jnp.minimum(idx[:, None], idx[None, :])
    hi = jnp.maximum(idx[:, None], idx[None, :])

    fn = jax.vmap(jax.vmap(_pair_hits, in_axes=(0, 0, 0, None, None)), in_axes=(0, 0, 0, None, None))
    return fn(_pair_keys(key, n), pos[lo], pos[hi], walls, n_samples)


def free_space(pos: chex.Array) -> chex.Array:
    r"""
    Attenuation matrix of an environment without obstacles.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.

    Returns
    -------
    Array
        A zero matrix of shape ``(n, n)``.
    """

    n = pos.shape[0]
    return jnp.zeros((n, n))


def wall_attenuation(key: chex.PRNGKey, pos: chex.Array, walls: Wall, n_samples: int = 256) -> chex.Array:
    r"""
    Monte Carlo estimate of the attenuation (dB) caused by walls for every pair of nodes.

    Points are sampled uniformly along the segment joining each pair of nodes and the
    fraction of them falling inside a wall estimates the fraction of the link immersed
    in that wall (see the :mod:`~mapc_sim.experimental.walls` module description).
    The resulting matrix is symmetric (the same samples are used for both directions
    of a link) and can be passed directly as the ``loss_gain`` argument of
    :func:`mapc_sim.sim.network_data_rate`.

    Parameters
    ----------
    key: PRNGKey
        JAX random number generator key.
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.
    walls: Wall
        Batched obstacles, e.g. built with :func:`stack`.
    n_samples: int
        Number of points sampled along each link. Controls both the accuracy and
        the variance of the estimator, cf. :func:`attenuation_std`.

    Returns
    -------
    Array
        Symmetric attenuation matrix of shape ``(n, n)`` in dB.
    """

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    hits = _hit_matrix(key, pos, walls, n_samples)
    return distance * (hits * walls.attenuation).sum(axis=-1)


def attenuation_std(key: chex.PRNGKey, pos: chex.Array, walls: Wall, n_samples: int = 256) -> chex.Array:
    r"""
    Standard deviation of the :func:`wall_attenuation` estimator (dB).

    Because the number of sampled points inside wall :math:`w` is
    :math:`\mathrm{Binomial}(N, p_w)`, the estimator error is a zero-mean random
    variable of standard deviation

    .. math::

        \sqrt{\operatorname{Var} \hat{L}_{ij}} = d_{ij}
        \sqrt{\sum_w \alpha_w^2 \frac{p_w (1 - p_w)}{N}} ,

    which for a fresh ``key`` per simulation step acts as a shadowing (fading)
    term added to the link budget -- see :ref:`sampling-noise-fading`. This function
    plugs the sample estimates :math:`\hat{p}_w` into the formula above, so it
    can be used to calibrate ``n_samples`` against the desired ``sigma`` of
    :func:`mapc_sim.sim.network_data_rate`.

    .. warning::

        The correspondence with a normal shadowing term is only approximate.
        The noise is zero for links that miss every obstacle or lie entirely
        inside one (:math:`p_w \in \{0, 1\}`), it is discrete (a multiple of
        :math:`\alpha_w d_{ij} / N`), and it is spatially correlated, as nearby
        links share the same geometry. The first property is a modelling
        decision rather than a bug: an unobstructed line-of-sight link is indeed
        subject to much weaker shadowing than an obstructed one.

    Parameters
    ----------
    key: PRNGKey
        JAX random number generator key.
    pos: Array
        Two dimensional array of node positions, shape ``(n, 2)``.
    walls: Wall
        Batched obstacles, e.g. built with :func:`stack`.
    n_samples: int
        Number of points sampled along each link.

    Returns
    -------
    Array
        Symmetric matrix of shape ``(n, n)`` with the standard deviation of the estimator in dB.
    """

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    p = _hit_matrix(key, pos, walls, n_samples)
    var = ((walls.attenuation ** 2) * p * (1. - p) / n_samples).sum(axis=-1)
    return distance * jnp.sqrt(var)


def plot_walls(walls: Wall, ax: Optional['matplotlib.axes.Axes'] = None, **kwargs) -> 'matplotlib.axes.Axes':
    r"""
    Draws the obstacles. Requires ``matplotlib``.

    Parameters
    ----------
    walls: Wall
        Batched obstacles, e.g. built with :func:`stack`.
    ax: matplotlib.axes.Axes, optional
        Axes to draw on. If not given, the current axes are used.
    **kwargs
        Additional keyword arguments passed to :class:`matplotlib.patches.Rectangle`,
        e.g. ``color`` or ``alpha``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the obstacles drawn.
    """

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    ax = ax if ax is not None else plt.gca()
    kwargs = {'color': 'gray', 'alpha': 0.5, **kwargs}

    xy, wh, rot = np.asarray(walls.xy), np.asarray(walls.wh), np.asarray(walls.rot)
    xy, wh, rot = np.atleast_2d(xy), np.atleast_2d(wh), rot.reshape(-1, 2, 2)

    for (x, y), (w, h), r in zip(xy, wh, rot):
        angle = np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))
        ax.add_patch(Rectangle(xy=(x, y), width=w, height=h, angle=angle, rotation_point='xy', **kwargs))

    ax.set_aspect('equal')
    return ax
