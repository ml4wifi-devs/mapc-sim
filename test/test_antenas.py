import unittest

import jax
import jax.numpy as jnp

from mapc_sim.constants import DEFAULT_TX_POWER
from mapc_sim.experimental.antenas import bearing, isotropic, link_gain, sector, sector_max_gain
from mapc_sim.experimental.walls import Wall, stack, wall_attenuation
from mapc_sim.sim import network_data_rate


class AntenaTestCase(unittest.TestCase):
    def setUp(self):
        self.pos = jnp.array([[0., 0.], [10., 0.], [0., 10.]])

    def test_bearing(self):
        angles = bearing(self.pos)

        self.assertAlmostEqual(float(angles[0, 1]), 0.)
        self.assertAlmostEqual(float(angles[0, 2]), 90.)
        self.assertAlmostEqual(float(angles[1, 0]), 180.)
        self.assertAlmostEqual(float(angles[2, 0]), -90.)

    def test_isotropic(self):
        # an isotropic antena has a gain of 0 dBi in every direction
        self.assertTrue(jnp.array_equal(isotropic(self.pos), jnp.zeros((3, 3))))
        self.assertTrue(jnp.array_equal(link_gain(isotropic(self.pos)), jnp.zeros((3, 3))))

    def test_sector_boresight(self):
        boresight = jnp.array([0., 180., 270.])
        gain = sector(self.pos, boresight, beamwidth=65.)
        peak = sector_max_gain(65.)

        # every antena points at node 0 or is pointed at by it
        self.assertAlmostEqual(float(gain[0, 1]), float(peak), places=5)
        self.assertAlmostEqual(float(gain[1, 0]), float(peak), places=5)
        self.assertAlmostEqual(float(gain[2, 0]), float(peak), places=5)

    def test_sector_half_power(self):
        # the pattern drops by 3 dB at half of the beamwidth from the boresight
        gain = sector(jnp.array([[0., 0.], [jnp.cos(jnp.pi / 8), jnp.sin(jnp.pi / 8)]]), jnp.array([0., 0.]), beamwidth=45.)
        self.assertAlmostEqual(float(gain[0, 1]), float(sector_max_gain(45.) - 3.), places=4)

    def test_sector_front_back_ratio(self):
        gain = sector(self.pos, jnp.zeros(3), beamwidth=65., max_gain=10., front_back_ratio=20.)

        # the attenuation of the pattern is capped at the front-to-back ratio
        self.assertAlmostEqual(float(gain[1, 0]), -10.)
        self.assertGreaterEqual(float(gain.min()), -10.)

    def test_sector_max_gain(self):
        # a narrower beam concentrates the same power in a smaller angle
        self.assertGreater(float(sector_max_gain(30.)), float(sector_max_gain(120.)))
        self.assertAlmostEqual(float(sector_max_gain(360.)), 0.)

    def test_link_gain_symmetry(self):
        gain = sector(self.pos, jnp.array([0., 180., 270.]), beamwidth=65.)
        total = link_gain(gain)

        self.assertTrue(jnp.array_equal(total, total.T))
        self.assertAlmostEqual(float(total[0, 1]), float(gain[0, 1] + gain[1, 0]), places=5)

    def test_beamforming_improves_data_rate(self):
        key, sim_key = jax.random.split(jax.random.PRNGKey(42))
        pos = jnp.array([[0., 0.], [30., 0.]])
        walls = stack([Wall.create(xy=jnp.array([14., -10.]), wh=jnp.array([0.2, 20.]), attenuation=25.)])

        attenuation = wall_attenuation(key, pos, walls, n_samples=1024)
        directional = attenuation - link_gain(sector(pos, jnp.array([0., 180.]), beamwidth=30.))
        omni = attenuation - link_gain(isotropic(pos))

        tx = jnp.array([[0, 1], [0, 0]])
        tx_power = jnp.full(2, DEFAULT_TX_POWER)
        args = (tx, pos, None, tx_power, 2.)  # greedy MCS selection

        rates = [network_data_rate(sim_key, *args, loss_gain) for loss_gain in (omni, directional)]
        self.assertLess(float(rates[0]), float(rates[1]))


if __name__ == '__main__':
    unittest.main()
