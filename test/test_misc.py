import unittest
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class MiscTestCase(unittest.TestCase):
    def test_logit_success_prob(self):
        d = tfd.Normal(10., 2.)
        logits = d.log_cdf(14.) - d.log_survival_function(14.)
        d1 = tfd.Binomial(40, logits=logits)
        d2 = tfd.Binomial(40, probs=d.cdf(14))
        self.assertAlmostEqual(d1.mean(), d2.mean())


if __name__ == '__main__':
    unittest.main()
