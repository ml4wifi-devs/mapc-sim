import unittest

import jax
import jax.numpy as jnp

from mapc_sim.utils import nakagami_fading_db
from mapc_sim.sim import network_data_rate


N_SAMPLES = 100_000
SHAPE = (N_SAMPLES,)
RTOL = 0.02


class TestNakagamiFading(unittest.TestCase):

    def test_fading_mean_linear(self):
        key = jax.random.PRNGKey(42)
        for m in [0.5, 1.0, 1.5, 3.0]:
            fading_db = nakagami_fading_db(key, m, SHAPE)
            g = jnp.power(10.0, fading_db / 10.0)
            self.assertLess(float(jnp.abs(jnp.mean(g) - 1.0)), RTOL, msg=f"m={m}: mean={float(jnp.mean(g)):.4f}")

    def test_fading_variance(self):
        key = jax.random.PRNGKey(7)
        for m in [0.5, 1.0, 2.0, 4.0]:
            fading_db = nakagami_fading_db(key, m, SHAPE)
            g = jnp.power(10.0, fading_db / 10.0)
            self.assertLess(float(jnp.abs(jnp.var(g) - 1.0 / m)), 0.05, msg=f"m={m}: var={float(jnp.var(g)):.4f}, expected {1/m:.4f}")

    def test_m1_rayleigh(self):
        key = jax.random.PRNGKey(99)
        fading_db = nakagami_fading_db(key, 1.0, SHAPE)
        g = jnp.power(10.0, fading_db / 10.0)
        self.assertLess(float(jnp.abs(jnp.mean(g) - 1.0)), RTOL)
        self.assertLess(float(jnp.abs(jnp.var(g) - 1.0)), 0.05)

    def test_large_m_no_fading(self):
        key = jax.random.PRNGKey(1)
        fading_db = nakagami_fading_db(key, 1000.0, (500,))
        self.assertLess(float(jnp.abs(fading_db).max()), 1.0)

    def _make_simple_network(self):
        key = jax.random.PRNGKey(0)
        tx = jnp.array([[0, 1], [0, 0]], dtype=jnp.float32)
        pos = jnp.array([[0.0, 0.0], [5.0, 0.0]])
        mcs = jnp.array([5, 0])
        tx_power = jnp.array([20.0, 20.0])
        sigma = jnp.array(0.0)
        walls = jnp.zeros((2, 2))
        return key, tx, pos, mcs, tx_power, sigma, walls

    def test_nakagami_affects_signal(self):
        key, tx, pos, mcs, tx_power, sigma, walls = self._make_simple_network()
        keys = jax.random.split(key, 200)

        def get_signal_power(k, nakagami_m=None):
            _, internals = network_data_rate(k, tx, pos, mcs, tx_power, sigma, walls,
                                             return_internals=True, nakagami_m=nakagami_m)
            return internals.signal_power

        sp_no_fading = jax.vmap(get_signal_power)(keys)
        sp_nakagami = jax.vmap(lambda k: get_signal_power(k, nakagami_m=1.0))(keys)
        self.assertGreater(float(jnp.std(sp_nakagami)), float(jnp.std(sp_no_fading)) + 1.0)


if __name__ == '__main__':
    unittest.main()
