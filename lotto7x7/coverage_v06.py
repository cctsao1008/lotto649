"""Exact fair 6/49 coverage of up to five fixed distinct six-number bets.

Dynamic programming partitions all 49 labels by their membership masks across
bets, then counts combinations in each class. No 13.98m-draw enumeration,
Monte Carlo approximation, target draw, or historical frequency is involved.
"""
from __future__ import annotations

from collections import Counter
from math import comb

import numpy as np

UNIVERSE = comb(49, 6)
SCHEMA = "lotto7x7.coverage_audit.v0.6"


def normalized_bets(bets, *, maximum=5):
    """Reject repeated/invalid bets instead of silently inflating coverage."""
    if not isinstance(bets, (list, tuple)) or not 1 <= len(bets) <= maximum:
        raise ValueError(f"expected 1..{maximum} bets")
    result = []
    for row in bets:
        if (not isinstance(row, (list, tuple)) or len(row) != 6 or
                any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                    or not 1 <= int(n) <= 49 for n in row) or
                len(set(row)) != 6):
            raise ValueError("each bet requires six distinct integers 1..49")
        result.append(tuple(sorted(map(int, row))))
    if len(set(result)) != len(result):
        raise ValueError("duplicate bets are not distinct tickets")
    return tuple(result)


def coverage_counts(bets):
    """Exact distribution of maximum hits among all C(49,6) possible draws.

    Group each label by the binary vector of ticket membership; picking k from
    a group of m contributes comb(m,k) draws and k hits to its member tickets.
    Track exactly six chosen labels, without enumerating all draws.
    """
    rows = normalized_bets(bets)
    n = len(rows)
    groups = Counter()
    for number in range(1, 50):
        mask = sum(1 << i for i, row in enumerate(rows) if number in row)
        groups[mask] += 1
    dims = (7,) * (n + 1)  # n hit counts + number of labels picked
    dp = np.zeros(dims, dtype=np.int64)
    dp[(0,) * (n + 1)] = 1
    for mask, group_size in sorted(groups.items()):
        next_dp = np.zeros_like(dp)
        for k in range(min(6, group_size) + 1):
            src = tuple(slice(0, 7-k) if mask & (1 << i) else slice(None)
                        for i in range(n)) + (slice(0, 7-k),)
            dst = tuple(slice(k, 7) if mask & (1 << i) else slice(None)
                        for i in range(n)) + (slice(k, 7),)
            next_dp[dst] += dp[src] * comb(group_size, k)
        dp = next_dp
    table = dp[..., 6]
    maximum = np.maximum.reduce(np.indices((7,) * n))
    counts = [int(table[maximum == hits].sum(dtype=np.int64)) for hits in range(7)]
    if sum(counts) != UNIVERSE or counts[6] != n:
        raise AssertionError("6/49 exact coverage identity failed")
    return counts


def exact_coverage(bets):
    """Exact probability distribution and nested >=3/4/5/6 coverages."""
    rows = normalized_bets(bets)
    counts = coverage_counts(rows)
    return {
        "method": "exact_membership_class_combinatorial_dp",
        "tickets": len(rows),
        "universe": UNIVERSE,
        "max_hits_counts_0_to_6": counts,
        "max_hits_probabilities_0_to_6": [x / UNIVERSE for x in counts],
        "at_least": {str(h): sum(counts[h:]) / UNIVERSE for h in (3, 4, 5, 6)},
        "at_least_counts": {str(h): sum(counts[h:]) for h in (3, 4, 5, 6)},
    }


def single_bet_anchor():
    """Hypergeometric reference used to validate exact DP implementation."""
    return [comb(6, hits) * comb(43, 6-hits) for hits in range(7)]