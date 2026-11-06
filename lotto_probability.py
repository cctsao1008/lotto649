"""Lotto649 v0.2 reference: exact weighted fixed-size subset distribution.

P(S) = exp(sum(logits[i] for i in S)) / e_k(exp(logits))
for unordered subsets of k distinct numbers. Log-space O(n*k) DP.
Neural decoders must be benchmarked against this NumPy reference.
"""
from __future__ import annotations

import argparse
from math import comb, exp, log
import numpy as np

VERSION = "0.1.0"


class FixedCardinalityDistribution:
    def __init__(self, logits, k=6):
        w = np.asarray(logits, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("logits must be a nonempty finite 1D array")
        if not isinstance(k, int) or not 0 <= k <= len(w):
            raise ValueError("invalid k")
        self.logits = w.copy()
        self.n, self.k = len(w), k
        b = np.full((self.n + 1, k + 1), -np.inf)
        b[:, 0] = 0.0
        for i in range(self.n - 1, -1, -1):
            for j in range(1, min(k, self.n - i) + 1):
                b[i, j] = np.logaddexp(b[i + 1, j], w[i] + b[i + 1, j - 1])
        self._suffix = b
        self.log_partition = float(b[0, k])

    def log_prob(self, subset):
        values = list(subset)
        if (len(values) != self.k or len(set(values)) != self.k or
                any(not isinstance(v, (int, np.integer)) or v < 1 or v > self.n for v in values)):
            raise ValueError(f"expected {self.k} unique integers within 1..{self.n}")
        return float(np.sum(self.logits[np.asarray(values, dtype=int) - 1]) - self.log_partition)

    def prob(self, subset):
        return exp(self.log_prob(subset))

    def sample(self, rng):
        result = []
        remain = self.k
        for i in range(self.n):
            if remain == 0:
                break
            if self.n - i == remain:
                result.extend(range(i + 1, self.n + 1))
                break
            log_p = self.logits[i] + self._suffix[i + 1, remain - 1] - self._suffix[i, remain]
            if rng.random() < float(np.clip(exp(float(log_p)), 0, 1)):
                result.append(i + 1)
                remain -= 1
        assert len(result) == self.k
        return tuple(result)

    def marginals(self):
        """Exact inclusion probabilities; should sum to k."""
        a = np.full((self.n + 1, self.k + 1), -np.inf)
        a[:, 0] = 0.0
        for i in range(self.n):
            for j in range(1, min(self.k, i + 1) + 1):
                a[i + 1, j] = np.logaddexp(a[i, j], self.logits[i] + a[i, j - 1])
        result = np.zeros(self.n)
        for i in range(self.n):
            if not self.k:
                break
            terms = [a[i, j] + self._suffix[i + 1, self.k - 1 - j]
                     for j in range(self.k)
                     if j <= i and self.k - 1 - j <= self.n - i - 1]
            result[i] = exp(float(self.logits[i] + np.logaddexp.reduce(terms) - self.log_partition))
        return result


def fair_log_prob(n=49, k=6):
    if not (isinstance(n, int) and isinstance(k, int) and 0 <= k <= n):
        raise ValueError("invalid n, k")
    return -log(comb(n, k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=649)
    args = parser.parse_args()
    d = FixedCardinalityDistribution(np.zeros(49))
    assert abs(d.log_prob([1, 2, 3, 4, 5, 6]) - fair_log_prob()) < 1e-11
    assert np.allclose(d.marginals(), 6 / 49, atol=1e-12)
    print(f"Lotto649 Probability Reference v{VERSION}")
    print(f"[PASS] C(49,6)={comb(49,6):,}")
    print(f"[PASS] fair log P={fair_log_prob():.10f}")
    print(f"[PASS] sum of marginal probabilities={d.marginals().sum():.10f}")
    print("[PASS] legal sample:", d.sample(np.random.default_rng(args.seed)))
    print("[INFO] No final ticket selection")


if __name__ == "__main__":
    main()
