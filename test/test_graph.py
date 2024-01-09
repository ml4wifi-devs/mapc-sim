import unittest

import jax
import jax.numpy as jnp

from mapc_sim.experimental.graph import segment_logsumexp


class GraphTest(unittest.TestCase):
    def test_segment_logsumexp(self):
        x = jnp.asarray([1, 2, 3, 4, -5.])
        id = jnp.asarray([0, 0, 0, 1, 1])

        si = segment_logsumexp(x, id, 2)

        def manual(x):
            return jnp.stack([
                jax.nn.logsumexp(x[:3]),
                jax.nn.logsumexp(x[3:])
            ])

        self.assertTrue(jnp.allclose(si, manual(x)))

        gman = jax.grad(lambda x: manual(x).sum())(x)
        ga = jax.grad(lambda x: segment_logsumexp(x, id, 2).sum())(x)

        self.assertTrue(jnp.allclose(gman, ga))
        ...
