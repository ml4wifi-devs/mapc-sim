import unittest
import warnings

import jax
import jax.numpy as jnp

from mapc_sim.constants import DEFAULT_TX_POWER, ENTERPRISE_WALL_LOSS, RESIDENTIAL_WALL_LOSS
from mapc_sim.sim import network_data_rate
from mapc_sim.utils import accepts_walls, binary_walls


class BinaryWallsTestCase(unittest.TestCase):
    def test_binary_walls(self):
        walls = jnp.array([[0., 1.], [1., 0.]])

        self.assertTrue(jnp.array_equal(binary_walls(walls), ENTERPRISE_WALL_LOSS * walls))
        self.assertTrue(jnp.array_equal(binary_walls(walls, RESIDENTIAL_WALL_LOSS), RESIDENTIAL_WALL_LOSS * walls))


class AcceptsWallsTestCase(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.pos = jnp.array([[0., 0.], [20., 0.]])
        self.walls = jnp.array([[0., 1.], [1., 0.]])
        self.args = (jnp.array([[0, 1], [0, 0]]), self.pos, None, jnp.full(2, DEFAULT_TX_POWER), 2.)

    def rate(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return float(network_data_rate(self.key, *self.args, **kwargs))

    def test_legacy_call_matches_loss_gain(self):
        self.assertEqual(self.rate(walls=self.walls), self.rate(loss_gain=binary_walls(self.walls)))
        self.assertEqual(
            self.rate(walls=self.walls, wall_loss=RESIDENTIAL_WALL_LOSS),
            self.rate(loss_gain=binary_walls(self.walls, RESIDENTIAL_WALL_LOSS))
        )

    def test_walls_attenuate(self):
        self.assertLess(self.rate(walls=self.walls), self.rate(walls=jnp.zeros((2, 2))))

    def test_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning):
            network_data_rate(self.key, *self.args, walls=self.walls)

        # the modern call must not warn (other libraries may, so filter by message)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            network_data_rate(self.key, *self.args, loss_gain=binary_walls(self.walls))

        self.assertEqual([w for w in caught if '`walls` is deprecated' in str(w.message)], [])

    def test_both_arguments(self):
        with self.assertRaises(TypeError):
            network_data_rate(self.key, *self.args, self.walls, walls=self.walls)

        with self.assertRaises(TypeError):
            network_data_rate(self.key, *self.args, loss_gain=self.walls, walls=self.walls)

    def test_no_arguments(self):
        with self.assertRaises(TypeError):
            network_data_rate(self.key, *self.args)

    def test_positional_is_loss_gain(self):
        # a matrix passed positionally is always the loss_gain matrix
        self.assertEqual(self.rate_positional(), self.rate(loss_gain=self.walls))

    def rate_positional(self):
        return float(network_data_rate(self.key, *self.args, self.walls))

    def test_jit(self):
        fn = jax.jit(network_data_rate, static_argnames=('return_internals',))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            legacy = fn(self.key, *self.args, walls=self.walls)

        self.assertAlmostEqual(float(legacy), self.rate(loss_gain=binary_walls(self.walls)), places=3)

    def test_decorator_is_reusable(self):
        @accepts_walls
        def loss(loss_gain):
            return loss_gain

        self.assertTrue(jnp.array_equal(loss(loss_gain=self.walls), self.walls))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertTrue(jnp.array_equal(loss(walls=self.walls), ENTERPRISE_WALL_LOSS * self.walls))


if __name__ == '__main__':
    unittest.main()
