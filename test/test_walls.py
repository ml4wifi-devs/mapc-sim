import unittest

import jax
import jax.numpy as jnp

from mapc_sim.constants import DATA_RATES, DEFAULT_TX_POWER
from mapc_sim.experimental.walls import Wall, attenuation_std, free_space, stack, wall_attenuation
from mapc_sim.sim import network_data_rate


class WallGeometryTestCase(unittest.TestCase):
    def test_contains(self):
        wall = Wall.create(xy=jnp.array([1., 1.]), wh=jnp.array([2., 4.]), attenuation=5.)

        self.assertTrue(wall.contains(jnp.array([2., 3.])))
        self.assertTrue(wall.contains(jnp.array([1., 1.])))
        self.assertFalse(wall.contains(jnp.array([0., 3.])))
        self.assertFalse(wall.contains(jnp.array([2., 6.])))

    def test_contains_rotated(self):
        wall = Wall.create(xy=jnp.array([0., 0.]), wh=jnp.array([2., 2.]), attenuation=5., angle=45.)

        self.assertTrue(wall.contains(jnp.array([0., 1.])))
        self.assertFalse(wall.contains(jnp.array([1., 0.])))


class WallAttenuationTestCase(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.pos = jnp.array([[0., 0.], [10., 0.]])

        # a 1 m thick wall of 5 dB/m spanning the whole scenario at x in [4, 5]
        self.walls = stack([Wall.create(xy=jnp.array([4., -10.]), wh=jnp.array([1., 20.]), attenuation=5.)])

    def test_free_space(self):
        self.assertTrue(jnp.array_equal(free_space(self.pos), jnp.zeros((2, 2))))

    def test_single_wall(self):
        loss = wall_attenuation(self.key, self.pos, self.walls, n_samples=4096)

        # the link crosses 1 m of a 5 dB/m material
        self.assertAlmostEqual(float(loss[0, 1]), 5., delta=0.5)

    def test_symmetry_and_diagonal(self):
        loss = wall_attenuation(self.key, self.pos, self.walls, n_samples=64)

        self.assertTrue(jnp.array_equal(loss, loss.T))
        self.assertTrue(jnp.array_equal(jnp.diag(loss), jnp.zeros(2)))

    def test_no_obstruction(self):
        pos = jnp.array([[0., 0.], [3., 0.]])
        loss = wall_attenuation(self.key, pos, self.walls, n_samples=64)

        self.assertEqual(float(loss[0, 1]), 0.)

    def test_unbiased(self):
        keys = jax.random.split(self.key, 200)
        loss = jax.vmap(lambda k: wall_attenuation(k, self.pos, self.walls, n_samples=16))(keys)[:, 0, 1]

        # the estimator is unbiased and its variance follows the binomial formula
        self.assertAlmostEqual(float(loss.mean()), 5., delta=0.5)
        self.assertAlmostEqual(float(loss.std()), 50. * jnp.sqrt(0.1 * 0.9 / 16), delta=1.)

    def test_attenuation_std(self):
        std = attenuation_std(self.key, self.pos, self.walls, n_samples=4096)

        # d * alpha * sqrt(p (1 - p) / N) for p = 0.1
        self.assertAlmostEqual(float(std[0, 1]), 50. * jnp.sqrt(0.1 * 0.9 / 4096), delta=0.05)
        self.assertTrue(jnp.array_equal(std, std.T))

    def test_multiple_walls(self):
        walls = stack([
            Wall.create(xy=jnp.array([4., -10.]), wh=jnp.array([1., 20.]), attenuation=5.),
            Wall.create(xy=jnp.array([7., -10.]), wh=jnp.array([2., 20.]), attenuation=1.)
        ])
        loss = wall_attenuation(self.key, self.pos, walls, n_samples=4096)

        self.assertAlmostEqual(float(loss[0, 1]), 5. + 2., delta=0.5)

    def test_jit_and_grad(self):
        fn = jax.jit(lambda k, p, w: wall_attenuation(k, p, w, n_samples=64)[0, 1], static_argnums=())
        self.assertGreater(float(fn(self.key, self.pos, self.walls)), 0.)

        # differentiable w.r.t. the attenuation of the material
        grad = jax.grad(lambda a: wall_attenuation(
            self.key, self.pos, self.walls.replace(attenuation=a), n_samples=4096
        )[0, 1])(self.walls.attenuation)
        self.assertAlmostEqual(float(grad[0]), 1., delta=0.1)

    def test_simulation(self):
        key, sim_key = jax.random.split(self.key)
        loss_gain = wall_attenuation(key, self.pos, self.walls, n_samples=256)

        tx = jnp.array([[0, 1], [0, 0]])
        mcs = jnp.zeros(2, dtype=int)
        tx_power = jnp.full(2, DEFAULT_TX_POWER)

        rate = network_data_rate(sim_key, tx, self.pos, mcs, tx_power, sigma=2., loss_gain=loss_gain)
        self.assertGreater(float(rate), 0.)
        self.assertLess(float(rate), float(DATA_RATES[20].max()))


if __name__ == '__main__':
    unittest.main()
