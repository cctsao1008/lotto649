"""Exact paired award discordance and fixed-horizon power for portfolio design."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations, islice
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import binom

from lotto7x7.portfolio.coverage_audit_v06 import _masks, _popcount64
from lotto7x7.coverage_v06 import normalized_bets
from lotto7x7.portfolio.official_prize_v026 import official_prize_coverage

DEFAULT_REPORT = Path("research/artifacts/v025_exact_covariance/v025_portfolio_pareto_20261116.json")


def paired_award_counts(repair, control, *, universe_size=49):
    """Count repair-only, control-only, joint and no-award outcomes exactly."""
    repair, control = normalized_bets(repair), normalized_bets(control)
    if (len(repair) != 4 or len(control) != 4 or not 7 <= universe_size <= 49
            or max(map(max, repair + control)) > universe_size):
        raise ValueError("need two four-ticket portfolios in the declared universe")
    masks = _masks(repair + control)
    universe_mask = np.uint64((1 << universe_size) - 1)
    denominator = comb(universe_size, 6) * (universe_size - 6)
    repair_wins = control_wins = both = 0
    combinations_iter = combinations(range(1, universe_size + 1), 6)
    while block := list(islice(combinations_iter, 100_000)):
        values = np.asarray(block, dtype=np.uint64)
        draws = np.bitwise_or.reduce(np.uint64(1) << (values - 1), axis=1)
        available = universe_mask ^ draws
        hits = _popcount64(draws[:, None] & masks[None, :])
        # Any-prize means at least three mains or two mains plus the bonus.
        candidate_masks = np.where(hits == 2, masks[None, :], np.uint64(0))
        bonus_masks = []
        for start in (0, 4):
            end = start + 4
            two_main = np.bitwise_or.reduce(candidate_masks[:, start:end], axis=1)
            bonus_masks.append(np.where(
                (hits[:, start:end] >= 3).any(axis=1), available, two_main & available,
            ))
        r, c = bonus_masks
        repair_wins += int(_popcount64(r).sum(dtype=np.int64))
        control_wins += int(_popcount64(c).sum(dtype=np.int64))
        both += int(_popcount64(r & c).sum(dtype=np.int64))
    for portfolio, wins in ((repair, repair_wins), (control, control_wins)):
        expected = official_prize_coverage(portfolio, universe_size=universe_size)
        if wins != expected["any_prize_outcomes"]:
            raise AssertionError("paired census disagrees with the exact single-arm DP")
    return {
        "repair_only": repair_wins - both, "control_only": control_wins - both,
        "both": both, "neither": denominator - repair_wins - control_wins + both,
        "total": denominator,
    }


def sign_test_power(draws, discordance, delta, *, alpha=0.05):
    """Exact finite-horizon power averaged over random discordance counts."""
    if (not isinstance(draws, int) or isinstance(draws, bool) or draws < 1
            or not 0 < alpha < 1):
        raise ValueError("positive integer draw horizon and alpha in (0, 1) required")
    if not 0 < discordance <= 1 or not 0 <= delta <= discordance:
        raise ValueError("positive incidence effect cannot exceed the discordance rate")
    m = np.arange(draws + 1)
    reject_from = binom.isf(alpha, m, 0.5).astype(int) + 1
    # Account for roundoff on the exact significance boundary.
    reject_from -= binom.sf(reject_from - 2, m, 0.5) <= alpha
    rho = (1 + delta / discordance) / 2
    power = float(binom.pmf(m, draws, discordance) @ binom.sf(reject_from - 1, m, rho))
    return min(1.0, power)


def design(artifact, *, pareto_report_sha256):
    best = artifact["pareto"]["best_repair_at_hybrid_floors_among_evaluated"]
    if best is None:
        raise ValueError("the Pareto archive has no qualifying repair portfolio")
    repair = best["tickets"]
    control = artifact["strategies"]["coverage_v021"]["tickets"]
    if any(len({number for row in arm for number in row}) != 24 for arm in (repair, control)):
        raise ValueError("both design arms must be pairwise disjoint")
    counts = paired_award_counts(repair, control)
    if counts["repair_only"] != counts["control_only"]:
        raise AssertionError("candidate arms do not have equal exact fair any-award coverage")
    discordance = (counts["repair_only"] + counts["control_only"]) / counts["total"]
    horizons = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
    return {
        "schema": "lotto649.v025.portfolio_validation_power.v1",
        "status": "DESIGN_ONLY",
        "candidate_pool_sha256": artifact["config"]["pool_sha256"],
        "pareto_report_sha256": pareto_report_sha256,
        "repair_pool_indices": best["pool_indices"],
        "control_pool_indices": artifact["strategies"]["coverage_v021"]["pool_indices"],
        "comparison": "best evaluated repair at coverage floors vs history-blind v0.21 coverage",
        "paired_award_counts": counts,
        "iid_discordance_probability": discordance,
        "power_assumption": "Independent paired outcomes, discordance rate fixed at fair IID",
        "checkpoints": [
            {"draws": n, "power_by_absolute_incidence_gain": {
                str(delta): sign_test_power(n, discordance, delta)
                for delta in (0.01, 0.02, 0.03, 0.05) if 0 < delta <= discordance
            }} for n in horizons
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=Path("results/v025_portfolio_validation_power.json"))
    args = parser.parse_args(argv)
    raw = args.pareto_report.read_bytes()
    report = design(json.loads(raw), pareto_report_sha256=sha256(raw).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("[DESIGN] Fair paired award discordance:", report["iid_discordance_probability"])
    print("[REPORT]", args.out)


if __name__ == "__main__":
    main()
