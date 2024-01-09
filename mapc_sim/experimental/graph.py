from functools import reduce, partial
from typing import Optional, Protocol

import chex
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from chex import Array
from jraph import segment_max, segment_sum, segment_softmax


@chex.dataclass
class Node:
    is_ap: Array
    mcs: Array
    pos: Array
    tx_power: Array


@chex.dataclass
class Edge:
    wall: Array
    transmission: Array


class Scenario(Protocol):
    id: int
    associations: dict[int, [list[int]]]
    pos: Array
    mcs: Array
    tx_power: Array
    walls: Array


def scenario_to_graph_tupple(scenario: Scenario) -> jraph.GraphsTuple:
    """Converts a Scenario object to a :ref:`GraphsTuple` object

    Parameters
    ----------
    scenario
        an instance of :ref:`ml4wifi.envs.scenarios.Scenario`

    Returns
    -------
    GraphsTuple
        Graph representation of a scenario

    """
    nodes = reduce(set.union, map(set, scenario.associations.values()),
                   set(scenario.associations))
    id = jnp.asarray(list(nodes))

    g = jraph.get_fully_connected_graph(len(nodes), 1, Node(
        is_ap=id < len(scenario.associations),
        pos=scenario.pos,
        mcs=scenario.mcs, tx_power=scenario.tx_power), global_features=None,
                                        add_self_edges=False)

    A = np.zeros((len(id), len(id)))
    for k, lv in scenario.associations.items():
        for v in lv:
            A[k, v] = 1

    graph = g._replace(edges=Edge(wall=scenario.walls[g.senders, g.receivers],
                                  transmission=A[g.senders, g.receivers]))

    return graph


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4))
def segment_logsumexp(x: Array, segment_ids: Array,
                      num_segments: Optional[int] = None,
                      indices_are_sorted: bool = False,
                      unique_indices: bool = False) -> Array:
    """Computes a segment-wise logsumexp.

    Parameters
    ----------
    x: Array
        An array of to be segment logsumexped.

    segment_ids: Array
        an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: Array
        an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.

    indices_are_sorted: Array
        whether ``segment_ids`` is known to be sorted

    unique_indices: Array
        whether ``segment_ids`` is known to be free of duplicates


    Returns
    -------
    Array

        The segment logsumexp-ed ``x``.

    """
    maxs = segment_max(x, segment_ids, num_segments, indices_are_sorted,
                       unique_indices)
    x = x - maxs[segment_ids]

    ex = jnp.exp(x)
    sums = segment_sum(ex, segment_ids, num_segments, indices_are_sorted,
                       unique_indices)
    lse = maxs + jnp.log(sums)
    return lse


@segment_logsumexp.defjvp
def segment_logsumexp_jvp(segment_ids: Array, num_segments: Optional[int],
                          indices_are_sorted: bool, unique_indices: bool,
                          primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = segment_logsumexp(x, segment_ids, num_segments,
                                   indices_are_sorted, unique_indices)
    tangent_out = segment_softmax(x, segment_ids, num_segments,
                                  indices_are_sorted, unique_indices) * x_dot
    tangent_out = segment_sum(tangent_out, segment_ids, num_segments,
                              indices_are_sorted, unique_indices)

    return primal_out, tangent_out


def segment_logsumexp_db(x: Array, segment_ids: Array,
                         num_segments: Optional[int] = None,
                         indices_are_sorted: bool = False,
                         unique_indices: bool = False) -> Array:
    """Computes :ref:`segment_logsumexp` for dB i.e. :math:`10log_10(\sum_i 10^{a_i/10})`

    Notes
    -----

        Parameters are the same as for :ref:`segment_logsumexp`



    Returns
    -------
    Array
        The segment `logsumexp` in dB

    """
    LOG10DIV10 = jnp.log(10.) / 10.
    return segment_logsumexp(LOG10DIV10 * x, segment_ids, num_segments,
                             indices_are_sorted, unique_indices) / LOG10DIV10
