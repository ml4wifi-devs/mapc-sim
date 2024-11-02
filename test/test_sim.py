import unittest

import jax
import jax.numpy as jnp

from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA
from mapc_sim.sim import network_data_rate


class SimTestCase(unittest.TestCase):
    def test_simple_network(self):
        # Position of the nodes given by X and Y coordinates
        pos = jnp.array([
            [10., 10.],  # AP A
            [40., 10.],  # AP B
            [10., 20.],  # STA 1
            [ 5., 10.],  # STA 2
            [25., 10.],  # STA 3
            [45., 10.]   # STA 4
        ])

        n_nodes = pos.shape[0]

        # Transmission matrices indicating which node is transmitting to which node:
        # - in this example, STA 1 is transmitting to AP A
        tx1 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, STA 2 is transmitting to AP A and STA 3 is transmitting to AP B
        tx2 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, STA 1 is transmitting to AP A and STA 4 is transmitting to AP B
        tx3 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Modulation and coding scheme of the nodes (here, all nodes use MCS 4)
        mcs = jnp.ones(n_nodes, dtype=jnp.int32) * 4

        # Transmission power of the nodes (all nodes use the default transmission power)
        tx_power = jnp.ones(n_nodes) * DEFAULT_TX_POWER

        # Standard deviation of the additive white Gaussian noise
        sigma = DEFAULT_SIGMA

        # Set walls to zero
        walls = jnp.zeros((pos.shape[0], pos.shape[0]))

        # JAX random number generator key
        key = jax.random.PRNGKey(42)

        # Simulate the network for 150 steps
        data_rate_1, data_rate_2, data_rate_3 = [], [], []

        for _ in range(150):
            key, k1, k2, k3 = jax.random.split(key, 4)
            data_rate_1.append(jax.jit(network_data_rate)(k1, tx1, pos, mcs, tx_power, sigma, walls))
            data_rate_2.append(jax.jit(network_data_rate)(k2, tx2, pos, mcs, tx_power, sigma, walls))
            data_rate_3.append(jax.jit(network_data_rate)(k3, tx3, pos, mcs, tx_power, sigma, walls))

        print('STA 1 -> AP A')
        print(data_rate_1)

        print('STA 2 -> AP A and STA 3 -> AP B')
        print(data_rate_2)

        print('STA 1 -> AP A and STA 4 -> AP B')
        print(data_rate_3)
