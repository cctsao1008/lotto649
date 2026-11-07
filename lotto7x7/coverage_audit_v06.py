"""Audit fixed v0.6 ticket portfolios using exact fair-draw probabilities.

Up to five purchased tickets are evaluated with exact combinatorial DP.
The entire 100-candidate research pool is separately estimated via IID
Monte Carlo for partial hits, with Wilson intervals, and never counted as bets.
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from .bet_selection_v06 import SCHEMA as SELECTION_SCHEMA
from .candidate_pool_v06 import pool_digest
from .coverage_v06 import (SCHEMA, UNIVERSE, exact_coverage, normalized_bets,
                           single_bet_anchor)


def _masks(number_sets):
    return np.array([sum(1 << (n-1) for n in row) for row in number_sets],
                    dtype=np.uint64)


def _popcount64(x):
    """Portable, vectorized SWAR uint64 bit counts (no np.bit_count required)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def _wilson(successes, trials):
    p = successes / trials
    z = 1.959963984540054
    denom = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denom
    span = z*sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denom
    return [max(0.,center-span), min(1.,center+span)]


def pool_coverage_estimate(pool_numbers, *, samples=50000, seed=20261108):
    """Uniform IID samples, independent of the fixed selection stage."""
    if (not isinstance(samples,int) or isinstance(samples,bool) or samples < 100 or
            not isinstance(seed,int) or seed < 0):
        raise ValueError("samples must >=100, seed must be nonnegative")
    numbers = normalized_bets(pool_numbers, maximum=len(pool_numbers))
    rng = np.random.default_rng(seed)
    ticket_masks = _masks(numbers)
    hist = np.zeros(7, dtype=np.int64)
    remaining = samples
    while remaining:
        batch = min(4096, remaining)
        winning = np.stack([rng.choice(49, 6, replace=False) for _ in range(batch)])
        draw_masks = np.bitwise_or.reduce(np.left_shift(np.uint64(1), winning.astype(np.uint64)), axis=1)
        maxhits = _popcount64(draw_masks[:,None] & ticket_masks[None,:]).max(axis=1)
        hist += np.bincount(maxhits, minlength=7)
        remaining -= batch
    estimates = {}
    for h in (3,4,5):
        count = int(hist[h:].sum())
        p = count / samples
        estimates[str(h)] = {"p_hat": p, "successes": count,
                              "standard_error": sqrt(p*(1-p)/samples),
                              "wilson_95": _wilson(count, samples)}
    return {"method": "iid_uniform_monte_carlo", "draw_samples": samples,
            "seed": seed, "max_hits_counts_0_to_6": hist.tolist(),
            "at_least_estimated": estimates,
            "exact_six_coverage": len(numbers) / UNIVERSE,
            "warning": "Pool-wide partial-hit probabilities are estimates; the 6/6 probability is exact. The research pool is not purchased."}


def _dominates(a, b):
    aa = a["at_least_counts"]
    bb = b["at_least_counts"]
    return (all(aa[str(h)] >= bb[str(h)] for h in (3,4,5)) and
            any(aa[str(h)] > bb[str(h)] for h in (3,4,5)))


def audit_selection(report, *, samples=50000, seed=20261108, actual=None):
    if not isinstance(report, dict) or report.get("schema") != SELECTION_SCHEMA:
        raise ValueError("expected v0.6 bet-selection JSON")
    pool = report.get("pool_numbers")
    if not isinstance(pool,list) or len(pool) != report.get("pool_candidate_count"):
        raise ValueError("missing complete research candidate pool")
    pool_rows = normalized_bets(pool, maximum=len(pool))
    if pool_digest([{"numbers": list(row)} for row in pool_rows]) != report.get("pool_sha256"):
        raise ValueError("candidate pool hash mismatch")
    configured_count = report.get("config", {}).get("count")
    strategies = report.get("strategies")
    if (not isinstance(configured_count,int) or not 1 <= configured_count <= 5 or
            not isinstance(strategies,dict) or not strategies):
        raise ValueError("invalid ticket count or strategies")
    actual_numbers = None if actual is None else normalized_bets([actual])[0]
    checked = {}
    for name, item in strategies.items():
        if name not in ("random", "shape-diverse", "coverage-optimized"):
            raise ValueError("unrecognized selection strategy")
        bets = item.get("bets", [])
        if len(bets) != configured_count:
            raise ValueError("the selected ticket count does not match")
        ids = [row["pool_index"] for row in bets]
        if (len(set(ids)) != configured_count or
                any(not isinstance(i, int) or isinstance(i,bool) or
                    not 0 <= i < len(pool_rows) for i in ids) or
                any(pool_rows[i] != tuple(row["numbers"]) for i, row in zip(ids,bets))):
            raise ValueError("ticket not in committed candidate pool")
        result = exact_coverage([pool_rows[i] for i in ids])
        previous = item.get("exact_fair_coverage")
        if previous is not None and previous != result:
            raise ValueError("stored coverage differs from recomputed exact coverage")
        checked[name] = {"tickets": configured_count,
                         "pool_indices": ids, "exact_fair_coverage": result}
        if actual_numbers is not None:
            actual_set = set(actual_numbers)
            hits = [len(actual_set.intersection(pool_rows[i])) for i in ids]
            checked[name]["realized_draw"] = {
                "per_ticket_hits": hits, "max_hits": max(hits),
                "post_draw_only": True,
            }
    nondominated = [name for name, value in checked.items() if not any(
        _dominates(checked[other]["exact_fair_coverage"], value["exact_fair_coverage"])
        for other in checked if other != name)]
    audit = {"schema": SCHEMA, "research_only": True,
             "pool_sha256": report["pool_sha256"],
             "source": report.get("source"),
             "ticket_count_per_strategy": configured_count,
             "research_pool_count": len(pool_rows),
             "population": "uniform independent 6 of 49; bonus excluded",
             "individual_ticket_exact_hit_counts_0_to_6": single_bet_anchor(),
             "strategies": checked,
             "pareto_nondominated": nondominated,
             "research_pool_coverage": pool_coverage_estimate(pool_rows,samples=samples,seed=seed),
             "scope_note": "Exact 1..5 ticket coverages are independent of historical results. Pool-wide partial-hit metrics are MC estimates, not purchased-ticket probabilities."}
    if actual_numbers is not None:
        audit["realized_draw"] = {"numbers": list(actual_numbers),
                                  "pool_max_hits": max(len(set(actual_numbers).intersection(row)) for row in pool_rows),
                                  "post_draw_only": True}
    return audit