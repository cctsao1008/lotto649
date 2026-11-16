"""Common six-number validation, uniform sampling and sum counts."""
from math import comb

import numpy as np


BALLS = 49
PICKS = 6
TOTAL = comb(BALLS, PICKS)


def validate_ticket(ticket):
    if len(ticket) != 6 or any(not isinstance(v, int) or isinstance(v, bool)
                              or not 1 <= v <= 49 for v in ticket):
        raise ValueError("each ticket needs six integers in 1..49")
    if len(set(ticket)) != 6:
        raise ValueError("duplicate number in ticket")
    return tuple(sorted(ticket))


def uniform_candidates(size: int, seed: int) -> np.ndarray:
    """IID six-of-49 *tickets*, rejecting duplicates until `size` distinct.

    The resulting unordered pool is an exactly uniform sample *without*
    replacement from all C(49,6) six-sets; no historical shapes or class
    probabilities are used in pool generation.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or not 5 <= size <= 5000
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('candidate pool must be 5..5000 and seed a nonnegative integer')
    rng = np.random.default_rng(seed)
    seen: set[tuple[int,...]] = set()
    out: list[tuple[int,...]] = []
    while len(out) < size:
        # Keys: 49 iid continuous uniforms, select six smallest without replacement.
        keys = rng.random((max(32, size-len(out)), 49))
        selected = np.argpartition(keys, kth=5, axis=1)[:, :6] + 1
        for row in selected:
            ticket = tuple(sorted(int(v) for v in row))
            if ticket not in seen:
                seen.add(ticket)
                out.append(ticket)
                if len(out) == size:
                    break
    return np.asarray(out, dtype=np.uint8)


def exact_sum_counts(n: int = BALLS, k: int = PICKS) -> tuple[int, ...]:
    """Count all k-subsets of {1,...,n} by sum using exact integer DP.

    The returned vector includes zero-count sums below the possible minimum.
    Time: O(n*k^2*n); memory: O(k^2*n), no enumeration or Monte Carlo.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("n and k must be integers")
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    maximum = k * (2 * n - k + 1) // 2
    dp = [[0] * (maximum + 1) for _ in range(k + 1)]
    dp[0][0] = 1
    for ball in range(1, n + 1):
        for chosen in range(min(ball, k), 0, -1):
            dst, src = dp[chosen], dp[chosen - 1]
            for subtotal in range(maximum, ball - 1, -1):
                dst[subtotal] += src[subtotal - ball]
    answer = tuple(dp[k])
    if sum(answer) != comb(n, k):
        raise AssertionError("exact theory does not normalize")
    return answer
