from chex import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from mapc_sim.constants import *
from mapc_sim.utils import logsumexp_db, default_path_loss

tfd = tfp.distributions


@dataclass
class Internals:
    ampdu_size: jax.Array
    average_data_rate: jax.Array
    frames_transmitted: jax.Array
    mcs: jax.Array
    signal_power: jax.Array
    sinr: jax.Array


def network_data_rate(
        key: jax.random.PRNGKey,
        tx: jax.Array,
        pos: jax.Array,
        mcs: jax.Array | None,
        tx_power: jax.Array,
        sigma: float,
        walls: jax.Array,
        channel_width: int = 20,
        return_internals: bool = False,
        path_loss_fn: Callable = default_path_loss
) -> float | tuple[float, Internals]:
    r"""
    Calculates the aggregated effective data rate based on the nodes' positions, MCS, and tx power.
    Channel is modeled using TGax channel model with additive white Gaussian noise. Effective
    data rate is calculated as the sum of data rates of all successful transmissions. Success of
    a transmission is a Binomial random variable with success probability depending on the SINR and
    number of trials equal to the number of frames in the slot. SINR is calculated as the difference
    between the signal power and the interference level. Interference level is calculated as the sum
    of the signal powers of all interfering nodes and the noise floor in the linear scale.

    .. important::

        This simulation does not support multiple simultaneous transmissions to the same node.

    Parameters
    ----------
    key: PRNGKey
        JAX random number generator key.
    tx: Array
        Two dimensional array of booleans indicating whether a node is transmitting to another node.
        If node i is transmitting to node j, then `tx[i, j] = 1`, otherwise `tx[i, j] = 0`.
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: Array | None
        Modulation and coding scheme of the nodes. Each entry corresponds to MCS of the transmitting node.
        If MCS is set to None, the simulator will select the best MCS greedily.
    tx_power: Array
        Transmission power of the nodes. Each entry corresponds to the transmission power of the transmitting node.
    sigma: float
        Standard deviation of the additive white Gaussian noise.
    walls: Array
        Adjacency matrix of walls. If node i is separated from node j by a wall,
        then `walls[i, j] = 1`, otherwise `walls[i, j] = 0`.
    channel_width: int
        Channel width in MHz.
    return_internals: bool
        A flag indicating whether the simulator returns additional information about the simulation results.
    path_loss_fn: Callable
        A function that calculates the path loss between two nodes. The function signature should be
        `path_loss_fn(distance: Array, walls: Array) -> Array`, where `distance` is the matrix of distances
        between nodes and `walls` is the adjacency matrix of walls. By default, the simulator uses the
        residential TGax path loss model.

    Returns
    -------
    float | tuple[float, Internals]
        Aggregated effective data rate in Mb/s if ``return_sample`` is ``False``.
        Otherwise, a pair of data rate and the number of transmitted frames.
    """

    normal_key, binomial_key = jax.random.split(key)

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

    signal_power = tx_power[:, None] - path_loss_fn(distance, walls)

    interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
    a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
    b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
    interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

    sinr = signal_power - interference
    sinr = sinr + tfd.Normal(jnp.zeros_like(signal_power), sigma).sample(seed=normal_key)
    sinr = (sinr * tx).sum(axis=1)

    if mcs is None:
        expected_data_rate = DATA_RATES[channel_width][:, None] * tfd.Normal(MEAN_SNRS[channel_width][:, None], STD_SNR).cdf(sinr)
        mcs = jnp.argmax(expected_data_rate, axis=0)

    sdist = tfd.Normal(MEAN_SNRS[channel_width][mcs], STD_SNR)
    logit_success_prob = sdist.log_cdf(sinr) - sdist.log_survival_function(sinr)
    logit_success_prob = jnp.where(sinr > 0, logit_success_prob, -jnp.inf)

    n = jnp.round(DATA_RATES[channel_width][mcs] * 1e6 * TAU / FRAME_LEN)
    frames_transmitted = tfd.Binomial(n, logit_success_prob).sample(seed=binomial_key)
    average_data_rate = FRAME_LEN * (frames_transmitted / TAU)
    total_data_rate = average_data_rate.sum() / float(1e6)

    if return_internals:
        return total_data_rate, Internals(
            ampdu_size=jnp.where(sinr > 0, n, 0),
            average_data_rate=average_data_rate,
            frames_transmitted=frames_transmitted,
            mcs=mcs,
            signal_power=signal_power,
            sinr=sinr
        )

    return total_data_rate
