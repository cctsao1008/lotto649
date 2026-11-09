"""v0.26 P2: Coverage Strategy A vs HDDC Strategy B, parity-locked.

Both strategies always buy exactly four DISTINCT pairwise-disjoint six-number
tickets, hence have identical theoretical official eight-tier joint
probabilities under uniform 6/49. A is seed-based, independent of historical
draws; B is the pre-existing v0.25 multivariate historical optimizer.

Optional strict chronological rolling evaluation uses only preceding training
draws to select tickets; each fixed portfolio is scored against a disjoint
future holdout block with the REAL special ball from independently checked CSV.
No target draw, outcome, or future label enters selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path
from random import Random

from .coverage_v06 import exact_coverage
from .deficit_compensation_v025 import REFERENCE, load_csv_history
from .multivariate_audit_v025 import (
    fair_covariance, portfolio_audit, select_multivariate_portfolio,
)
from .official_prize_v026 import PRIZE_ORDER, prize_tier, official_prize_coverage
from .shape_deficit_v025 import load_exact_baseline

SCHEMA = "lotto7x7.coverage_locked_hddc.v0.26.p2"


def disjoint_portfolio(seed):
    """A valid history-blind Coverage Strategy A with 24 unique labels."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    labels = list(range(1, 50))
    Random(seed).shuffle(labels)
    return tuple(tuple(sorted(labels[6*i:6*(i+1)])) for i in range(4))


def assert_official_coverage_parity(a, b):
    """Stronger than equal union probability: exact full prize-event profile."""
    results = [official_prize_coverage(x) for x in (a, b)]
    for label, tickets in zip(("A", "B"), (a, b)):
        if len(tickets) != 4 or len({n for t in tickets for n in t}) != 24:
            raise AssertionError(f"{label} must contain four disjoint tickets")
    keys = ("any_prize_outcomes", "awarded_ticket_count_outcomes",
            "any_ticket_by_tier", "expected_tickets_by_tier")
    for key in keys:
        if results[0][key] != results[1][key]:
            raise AssertionError(f"official coverage differs: {key}")
    if results[0]["any_prize_outcomes"] != official_prize_coverage(REFERENCE)["any_prize_outcomes"]:
        raise AssertionError("four-ticket exact global coverage is not reference maximum")
    mains = [exact_coverage(tickets) for tickets in (a, b)]
    if mains[0]["max_hits_counts_0_to_6"] != mains[1]["max_hits_counts_0_to_6"]:
        raise AssertionError("main-ball exact coverage differs")
    return {
        "any_official_prize_probability": results[0]["any_prize_probability"],
        "exact_official_prize_event_counts_equal": True,
        "exact_main_ball_coverage_equal": True,
        "expected_awarded_tickets": results[0]["expected_awarded_tickets"],
        "per_tier_probability": {
            tier: results[0]["any_ticket_by_tier"][tier]["probability"]
            for tier in PRIZE_ORDER
        },
    }


def checked_bonus_by_draw(csv_dir, records):
    """Read real bonus values and recheck key/balls versus validated history.

    The old historical feature loader intentionally excludes the bonus.
    Without this step, evaluating *official* historical award tiers would
    silently be impossible (main-only coverage is not enough).
    """
    indexed = {(date, ident): tuple(numbers) for date, ident, numbers in records}
    out = {}
    for path in sorted(Path(csv_dir).glob("lotto649_????.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["date","draw_id","n1","n2","n3","n4","n5","n6","bonus"]:
                raise ValueError("CSV bonus schema unexpected")
            for row in reader:
                key = (row["date"].replace("/", "-"), int(row["draw_id"]))
                if key not in indexed:
                    continue
                numbers = tuple(sorted(int(row[f"n{i}"]) for i in range(1, 7)))
                bonus = int(row["bonus"])
                if numbers != indexed[key] or not 1 <= bonus <= 49 or bonus in numbers:
                    raise ValueError("bonus/main verification mismatch")
                if key in out:
                    raise ValueError("duplicate bonus observation")
                out[key] = bonus
    if len(out) != len(records):
        raise ValueError("missing bonus in validated historical records")
    return out


def draw_awards(tickets, main, bonus):
    """Return single-draw total awarded tickets and highest official tier."""
    main_set = set(main)
    found = [prize_tier(len(main_set.intersection(ticket)), bonus in ticket)
             for ticket in tickets]
    winners = [x for x in found if x is not None]
    return {
        "any_award": bool(winners),
        "awarded_tickets": len(winners),
        "tiers": winners,
    }


def paired_exact_p(a_only, b_only):
    """Two-sided exact conditional binomial discordance p, no normal approx."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def compare(records, model, *, selector_seed=649, pool_size=300, a_seed=649,
            windows=()):
    """Compare same-coverage portfolios without confusing score and prediction."""
    strategy_a = disjoint_portfolio(a_seed)
    selected = select_multivariate_portfolio(
        records, model, pool_size=pool_size, seed=selector_seed,
        coverage_constraint="reference-3plus")
    strategy_b = tuple(tuple(t) for t in selected["tickets"])
    parity = assert_official_coverage_parity(strategy_a, strategy_b)
    desc_a = portfolio_audit(records, model, strategy_a)
    desc_b = portfolio_audit(records, model, strategy_b)
    return {
        "strategy_a": {"tickets": [list(t) for t in strategy_a],
                       "historical_distance": desc_a["after"],
                       "uses_history": False},
        "strategy_b": {"tickets": [list(t) for t in strategy_b],
                       "historical_distance": desc_b["after"],
                       "uses_history": True},
        "difference_a_minus_b": desc_a["after"] - desc_b["after"],
        "parity": parity,
        "selection_is_global_optimum": False,
    }


def chronological_walkforward(records, bonus_by_key, model, *,
                              train_min=1800, horizon=25, stride=25,
                              max_folds=8, selector_seed=649,
                              pool_size=300, a_seed=649):
    """Strictly forward, NON-overlapping realized official tier evaluation.

    No future draw is included in the training slice. A is randomly relabeled
    independently each fold. IID fair probability is the same for both.
    This is historical exploratory walk-forward, not a prospective test.
    """
    if (not 256 <= train_min < len(records) or horizon < 1 or
            stride < horizon or max_folds < 1):
        raise ValueError("invalid walk-forward folds: need nonoverlap and training >=256")
    folds = []
    stats = {"a_wins": 0, "b_wins": 0, "a_only": 0, "b_only": 0,
             "both": 0, "neither": 0, "a_awarded_tickets": 0,
             "b_awarded_tickets": 0, "draws": 0}
    for fi, start in enumerate(range(train_min, len(records), stride)):
        if fi >= max_folds:
            break
        train = records[:start]
        test = records[start:min(len(records), start + horizon)]
        if not test:
            break
        if train[-1][0] >= test[0][0] or train[-1][1] >= test[0][1]:
            raise AssertionError("future leakage in chronological split")
        strategy_a = disjoint_portfolio(a_seed + fi)
        strategy_b_raw = select_multivariate_portfolio(
            train, model, pool_size=pool_size, seed=selector_seed,
            coverage_constraint="reference-3plus")
        strategy_b = tuple(tuple(t) for t in strategy_b_raw["tickets"])
        assert_official_coverage_parity(strategy_a, strategy_b)
        this = {"a_wins": 0, "b_wins": 0, "a_awarded_tickets": 0,
                "b_awarded_tickets": 0}
        for date, draw_id, main in test:
            bonus = bonus_by_key[(date, draw_id)]
            av = draw_awards(strategy_a, main, bonus)
            bv = draw_awards(strategy_b, main, bonus)
            a_hit, b_hit = av["any_award"], bv["any_award"]
            this["a_wins"] += int(a_hit)
            this["b_wins"] += int(b_hit)
            this["a_awarded_tickets"] += av["awarded_tickets"]
            this["b_awarded_tickets"] += bv["awarded_tickets"]
            stats["a_only" if a_hit and not b_hit else
                  "b_only" if b_hit and not a_hit else
                  "both" if a_hit else "neither"] += 1
        for name in this:
            stats[name] += this[name]
        stats["draws"] += len(test)
        folds.append({
            "fold": fi, "train_draws": len(train),
            "train_end_id": train[-1][1], "train_end_date": train[-1][0],
            "test_begin_id": test[0][1], "test_end_id": test[-1][1],
            "test_begin_date": test[0][0], "test_end_date": test[-1][0],
            "test_draws": len(test), **this,
            "a_tickets": [list(t) for t in strategy_a],
            "b_tickets": [list(t) for t in strategy_b],
        })
    if not folds:
        raise ValueError("no walk-forward folds")
    if stats["draws"] != sum(stats[k] for k in ("a_only","b_only","both","neither")):
        raise AssertionError("paired win table does not close")
    p0 = official_prize_coverage(REFERENCE)["any_prize_probability"]
    return {
        "training_policy": "expanding window strictly before each holdout",
        "holdout_policy": "non-overlapping forward blocks; official bonus included",
        "iid_theoretical_any_prize_probability_both": p0,
        "covariance_source": "independent IID simulation only",
        "exploratory_historical_not_prospective": True,
        "folds": folds,
        "aggregate": {
            **stats,
            "a_win_rate": stats["a_wins"]/stats["draws"],
            "b_win_rate": stats["b_wins"]/stats["draws"],
            "b_minus_a_win_rate": (stats["b_wins"]-stats["a_wins"])/stats["draws"],
            "paired_two_sided_exact_p_exploratory": paired_exact_p(
                stats["a_only"], stats["b_only"]),
            "caution": "Selection rules/parameters inspected retrospectively; no predictive edge asserted.",
        },
    }


def report(csv_dir, universe, *, cutoff="2026-11-06", iid_samples=8192,
           covariance_seed=649, ridge=0.25, pool_size=300,
           selector_seed=649, a_seed=649, train_min=1800,
           horizon=25, stride=25, max_folds=8):
    if cutoff is None:
        raise ValueError("explicit historical cutoff required")
    records, csv_hash = load_csv_history(Path(csv_dir), cutoff)
    bonus = checked_bonus_by_draw(Path(csv_dir), records)
    baseline, universe_hash = load_exact_baseline(Path(universe))
    model = fair_covariance(baseline, samples=iid_samples,
                            seed=covariance_seed, ridge=ridge)
    comparison = compare(records, model, selector_seed=selector_seed,
                         pool_size=pool_size, a_seed=a_seed)
    backtest = chronological_walkforward(
        records, bonus, model, train_min=train_min, horizon=horizon,
        stride=stride, max_folds=max_folds, selector_seed=selector_seed,
        pool_size=pool_size, a_seed=a_seed)
    return {
        "schema": SCHEMA,
        "history_cutoff": cutoff, "history_draws": len(records),
        "history_last_draw_id": records[-1][1],
        "csv_sha256": csv_hash, "exact_universe_sha256": universe_hash,
        "model_covariance": model["meta"],
        "selection": {"pool_size": pool_size, "selector_seed": selector_seed,
                      "a_seed": a_seed},
        "comparison": comparison,
        "walkforward": backtest,
        "caution": ("Identical theoretical win-event distributions under IID. "
                    "Retrospective test is exploratory, not an edge proof. "
                    "Original paid A-D untouched."),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    p.add_argument("--universe", type=Path,
                   default=Path("results/v015_v016_p1_shape_catalog_universe.json"))
    p.add_argument("--cutoff", default="2026-11-06")
    p.add_argument("--iid-samples", type=int, default=8192)
    p.add_argument("--covariance-seed", type=int, default=649)
    p.add_argument("--ridge", type=float, default=0.25)
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--selector-seed", type=int, default=649)
    p.add_argument("--a-seed", type=int, default=649)
    p.add_argument("--train-min", type=int, default=1800)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--max-folds", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/v026_strategy_ab_walkforward.json"))
    a = p.parse_args(argv)
    result = report(a.csv_dir, a.universe, cutoff=a.cutoff,
                    iid_samples=a.iid_samples, covariance_seed=a.covariance_seed,
                    ridge=a.ridge, pool_size=a.pool_size,
                    selector_seed=a.selector_seed, a_seed=a.a_seed,
                    train_min=a.train_min, horizon=a.horizon,
                    stride=a.stride, max_folds=a.max_folds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    cmp = result["comparison"]
    agg = result["walkforward"]["aggregate"]
    print(f"[P2] historical={result['history_draws']}; "
          f"cutoff={result['history_cutoff']}; folds={len(result['walkforward']['folds'])}")
    print("[COVERAGE] exact eight-tier and main-ball equality: PASS")
    print(f"[DESCRIPTIVE] A={cmp['strategy_a']['historical_distance']:.10f} "
          f"B={cmp['strategy_b']['historical_distance']:.10f}")
    print(f"[WALKFORWARD] draws={agg['draws']} A={agg['a_wins']} "
          f"B={agg['b_wins']} discordant_A={agg['a_only']} "
          f"discordant_B={agg['b_only']} exploratory_p={agg['paired_two_sided_exact_p_exploratory']:.6f}")
    print("[CAUTION] retrospective exploratory; no future predictive edge established")
    print(f"[REPORT] {a.out}")


if __name__ == "__main__":
    main()
