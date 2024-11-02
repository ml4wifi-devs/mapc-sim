from typing import Union, Optional

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from mapc_sim.constants import *
from mapc_sim.utils import logsumexp_db, tgax_path_loss

tfd = tfp.distributions


def network_data_rate(key: jax.random.PRNGKey, tx: jax.Array, pos: jax.Array, mcs: Optional[jax.Array], tx_power: jax.Array,
                      sigma: float, walls: jax.Array, return_sample: bool = False) -> Union[float, tuple]:
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
    mcs: Array
        Modulation and coding scheme of the nodes. Each entry corresponds to MCS of the transmitting node.
        If MCS is set to None, the simulator will select the best MCS greedily.
    tx_power: Array
        Transmission power of the nodes. Each entry corresponds to the transmission power of the transmitting node.
    sigma: float
        Standard deviation of the additive white Gaussian noise.
    walls: Array
        Adjacency matrix of walls. If node i is separated from node j by a wall,
        then `walls[i, j] = 1`, otherwise `walls[i, j] = 0`.
    return_sample: bool
        A flag indicating whether the simulator returns raw number of transmitted frames.

    Returns
    -------
    float | tuple
        Aggregated effective data rate in Mb/s if ``return_sample`` is ``False``.
        Otherwise, a pair of data rate and the number of transmitted frames.
    """

    normal_key, binomial_key = jax.random.split(key)

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

    signal_power = tx_power[:, None] - tgax_path_loss(distance, walls)

    interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
    a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
    b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
    interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

    sinr = signal_power - interference
    sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=sigma).sample(seed=normal_key)
    sinr = (sinr * tx).sum(axis=1)

    if mcs is None:
        expected_data_rate = DATA_RATES[:, None] * tfd.Normal(loc=MEAN_SNRS[:, None], scale=2.).cdf(sinr)
        mcs = jnp.argmax(expected_data_rate, axis=0)

    sdist = tfd.Normal(loc=MEAN_SNRS[mcs], scale=2.)
    logit_success_prob = sdist.log_cdf(sinr) - sdist.log_survival_function(sinr)
    logit_success_prob = jnp.where(sinr > 0, logit_success_prob, -jnp.inf)

    n = jnp.round(DATA_RATES[mcs] * 1e6 * TAU / FRAME_LEN)
    frames_transmitted = tfd.Binomial(total_count=n, logits=logit_success_prob).sample(seed=binomial_key)
    average_data_rate = FRAME_LEN * (frames_transmitted / TAU)

    if return_sample:
        return average_data_rate.sum() / float(1e6), frames_transmitted
    else:
        return average_data_rate.sum() / float(1e6)
