import timeit

import jax
import jax.numpy as jnp
import numpy as np
from mapc_sim.constants import DEFAULT_SIGMA, DEFAULT_TX_POWER
from mapc_sim.sim import network_data_rate


def benchmark():
    # Position of the nodes given by X and Y coordinates
    pos = jnp.array([[10., 10.],  # AP A
                     [40., 10.],  # AP B
                     [10., 20.],  # STA 1
                     [5., 10.],  # STA 2
                     [25., 10.],  # STA 3
                     [45., 10.]  # STA 4
                     ])

    n_nodes = pos.shape[0]

    # Transmission matrices indicating which node is transmitting to which node:
    # - in this example, STA 1 is transmitting to AP A
    tx1 = jnp.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]])

    # Modulation and coding scheme of the nodes (here, all nodes use MCS 4)
    mcs = jnp.ones(n_nodes, dtype=jnp.int32) * 4

    # Transmission power of the nodes (all nodes use the default transmission power)
    tx_power = jnp.ones(n_nodes) * DEFAULT_TX_POWER

    # Standard deviation of the additive white Gaussian noise
    sigma = DEFAULT_SIGMA

    # JAX random number generator key
    key = jax.random.PRNGKey(42)

    # Walls
    walls = jnp.zeros((pos.shape[0], pos.shape[0]))

    @jax.jit
    def run():
        return jax.block_until_ready(network_data_rate(key, tx1, pos, mcs, tx_power, sigma, walls))

    run()
    r = 80
    n = 10000
    t = timeit.repeat(run, repeat=r, number=n, globals=locals())

    print(jax.devices())
    print(f'network_throughput: {np.mean(t) / n} s ± {2 * np.std(t) / np.sqrt(r * n)} s')


if __name__ == '__main__':
    benchmark()
