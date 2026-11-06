"""Phase 1 parity + gradient tests. Torch dependency is optional until ML phase.

Run: python -m unittest discover -s tests -v
"""
import itertools
import unittest

import numpy as np
from lotto_probability import FixedCardinalityDistribution, fair_log_prob

try:
    import torch
    from lotto_torch_decoder import FixedCardinalityDecoder
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed: run pip install torch for Phase 1")
class TorchDecoderTests(unittest.TestCase):
    def test_uniform_49_choose_6(self):
        logits = torch.zeros(49, dtype=torch.float64, requires_grad=True)
        chosen = torch.tensor([3, 11, 18, 27, 35, 46])
        logp = FixedCardinalityDecoder()(logits, chosen)
        self.assertAlmostEqual(logp.item(), fair_log_prob(), places=11)
        logp.backward()
        expected = torch.full((49,), -6 / 49, dtype=torch.float64)
        expected[chosen - 1] += 1
        torch.testing.assert_close(logits.grad, expected, atol=1e-11, rtol=0)

    def test_numpy_parity_batched(self):
        rng = np.random.default_rng(1008)
        weights = rng.normal(0, 1.5, size=(4, 49))
        chosen = np.stack([rng.choice(np.arange(1, 50), 6, replace=False) for _ in range(4)])
        actual = FixedCardinalityDecoder()(torch.from_numpy(weights), torch.from_numpy(chosen)).detach().numpy()
        expected = [FixedCardinalityDistribution(w).log_prob(s) for w, s in zip(weights, chosen)]
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)

    def test_exact_small_space_normalization(self):
        logits = torch.tensor([1.7, -0.4, 0.2, -1.0, 2.0, 0.3, -0.5, 1.4], dtype=torch.float64)
        combinations = torch.tensor(list(itertools.combinations(range(1, 9), 3)))
        batch = logits.expand(len(combinations), -1)
        logp = FixedCardinalityDecoder(8, 3)(batch, combinations)
        self.assertAlmostEqual(torch.exp(logp).sum().item(), 1.0, places=12)
        reference = FixedCardinalityDistribution(logits.numpy(), 3)
        np.testing.assert_allclose(logp.detach().numpy(),
                                   [reference.log_prob(s) for s in combinations.tolist()],
                                   atol=1e-11, rtol=0)

    def test_gradient_matches_marginal_identity(self):
        rng = np.random.default_rng(649)
        logits = torch.tensor(rng.normal(size=9), dtype=torch.float64, requires_grad=True)
        chosen = torch.tensor([2, 5, 9])
        logp = FixedCardinalityDecoder(9, 3)(logits, chosen)
        grad, = torch.autograd.grad(logp, logits)
        expected = -FixedCardinalityDistribution(logits.detach().numpy(), 3).marginals()
        expected[chosen.numpy() - 1] += 1
        np.testing.assert_allclose(grad.numpy(), expected, atol=1e-11, rtol=0)

    def test_gradcheck(self):
        logits = torch.tensor(np.linspace(-2, 2, 8), dtype=torch.float64, requires_grad=True)
        target = torch.tensor([2, 3, 6])
        decoder = FixedCardinalityDecoder(8, 3)
        self.assertTrue(torch.autograd.gradcheck(lambda x: decoder(x, target),
                                                 (logits,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_large_logits_and_shift_invariance(self):
        logits = torch.tensor([800., -900., 700., 750., -600., 100., 42., -80.], dtype=torch.float64)
        chosen = torch.tensor([1, 3, 4])
        decoder = FixedCardinalityDecoder(8, 3)
        a = decoder(logits, chosen)
        b = decoder(logits + 10000, chosen)
        self.assertTrue(torch.isfinite(a))
        self.assertAlmostEqual(a.item(), b.item(), places=9)

    def test_invalid_numbers_rejected(self):
        decoder = FixedCardinalityDecoder(8, 3)
        logits = torch.zeros(8)
        for numbers in ([1, 1, 2], [1, 2, 9], [0, 2, 3], [1, 2], [1.0, 2.0, 3.0]):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                decoder(logits, torch.tensor(numbers))

    def test_gradients_batch_finite(self):
        logits = torch.randn(5, 49, dtype=torch.float64, requires_grad=True)
        numbers = torch.tensor([[1, 7, 13, 22, 35, 49]] * 5)
        logps = FixedCardinalityDecoder()(logits, numbers)
        self.assertEqual(tuple(logps.shape), (5,))
        (-logps.mean()).backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        torch.testing.assert_close(logits.grad.sum(dim=1), torch.zeros(5, dtype=torch.float64),
                                   atol=1e-11, rtol=0)


if __name__ == "__main__":
    unittest.main()
