import unittest

import jax
import jax.numpy as jnp

from mapc_sim.constants import NOISE_FLOOR
from mapc_sim.utils import logsumexp_db


class UtilsTestCase(unittest.TestCase):
    def test_logsumexp(self):
        NOISE_FLOOR_LIN = jnp.power(10, NOISE_FLOOR / 10)

        key = jax.random.key(42)
        signal_power = 15 * jax.random.normal(key, (10, 10)) - 70.

        tx = jnp.zeros((10, 10))
        tx = tx.at[3, 5].set(1)
        tx = tx.at[4, 1].set(1)
        tx = tx.at[0, 2].set(1)

        interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=-1, keepdims=True) * (1 - tx)
        interference_lin = jnp.power(10, signal_power / 10)
        interference_lin = (interference_matrix * interference_lin).sum(axis=0)
        interference_original = 10 * jnp.log10(interference_lin + NOISE_FLOOR_LIN)

        a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
        b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
        interference_new = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

        self.assertTrue(jnp.allclose(interference_original, interference_new))
