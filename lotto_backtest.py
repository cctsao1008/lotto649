"""Lotto649 walk-forward strategy backtest v0.1.0.

All five strategies use only draws *before* the prediction date.
This is exploratory research, not a method to improve fair-draw odds.
Each strategy selects two distinct six-number tickets with no overlap.
Special ball and bonus activities are excluded from scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
SOURCE = Path("data/normalized/lotto649_base.npy")
OUTPUT = Path("data/backtest")
STRATEGIES = ("random", "hot", "cold", "gap", "pair")
NUMBERS = np.arange(1, 50, dtype=np.int16)
THEORETICAL_MEAN_HITS_PER_TICKET = 36 / 49


def load_draws():
    draws = np.load(SOURCE, allow_pickle=False)
    if draws.dtype.names is None or not {"date", "draw_id", "numbers"}.issubset(draws.dtype.names):
        raise ValueError("Expected structured NPY with date, draw_id and numbers")
    numbers = draws["numbers"].astype(np.int16)
    if numbers.ndim != 2 or numbers.shape[1] != 6:
        raise ValueError("Expected (N, 6) lotto numbers")
    if np.any((numbers < 1) | (numbers > 49)):
        raise ValueError("Numbers must be within 1..49")
    if np.any(np.diff(np.sort(numbers, axis=1), axis=1) == 0):
        raise ValueError("Duplicate numbers within a draw")
    dates = draws["date"].astype(np.int64)
    ids = draws["draw_id"].astype(np.int64)
    if np.any(np.diff(dates) <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("Draw dates must be ascending and IDs unique")
    return dates, ids, numbers


def choose_by_order(order):
    chosen = list(map(int, order[:12]))
    if len(chosen) != 12 or len(set(chosen)) != 12:
        raise ValueError("Ticket selection must cover twelve unique numbers")
    return tuple(sorted(chosen[:6])), tuple(sorted(chosen[6:]))


def recent_gaps(history):
    # Number of previous draws since last appearance; never-seen gets len(history).
    last = np.full(50, -1, dtype=np.int32)
    for i, draw in enumerate(history):
        last[draw] = i
    return len(history) - 1 - last[1:]


def pair_order(history, freq):
    """Greedy maximum pair affinity on training window; no future data."""
    present = np.zeros((len(history), 49), dtype=np.int32)
    present[np.arange(len(history))[:, None], history - 1] = 1
    co = present.T @ present
    # Pair strength is an exploratory feature; no claim of predictive value.
    chosen = []
    available = set(range(1, 50))
    for _ticket in range(2):
        ticket = []
        while len(ticket) < 6:
            def score(n):
                affinity = sum(int(co[n - 1, m - 1]) for m in ticket)
                return (affinity, int(freq[n - 1]), -n)
            nxt = max(available, key=score)
            ticket.append(nxt)
            chosen.append(nxt)
            available.remove(nxt)
    return chosen


def pick(strategy, history, rng):
    frequency = np.bincount(history.ravel(), minlength=50)[1:]
    # Number tie breaks are deterministic; prevents look-ahead via global data.
    if strategy == "random":
        order = rng.permutation(NUMBERS)
    elif strategy == "hot":
        order = sorted(NUMBERS.tolist(), key=lambda n: (-int(frequency[n - 1]), n))
    elif strategy == "cold":
        order = sorted(NUMBERS.tolist(), key=lambda n: (int(frequency[n - 1]), n))
    elif strategy == "gap":
        gaps = recent_gaps(history)
        order = sorted(NUMBERS.tolist(), key=lambda n: (-int(gaps[n - 1]), n))
    elif strategy == "pair":
        order = pair_order(history, frequency)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return choose_by_order(order)


def score(ticket, actual):
    return len(set(ticket).intersection(actual))


def evaluate(numbers, dates, ids, window, seed):
    rng = np.random.default_rng(seed)
    detail = []
    totals = {s: Counter() for s in STRATEGIES}
    picks = {s: Counter() for s in STRATEGIES}
    for idx in range(window, len(numbers)):
        history = numbers[idx - window:idx]
        actual = set(map(int, numbers[idx]))
        for strategy in STRATEGIES:
            t1, t2 = pick(strategy, history, rng)
            h1, h2 = score(t1, actual), score(t2, actual)
            best = max(h1, h2)
            totals[strategy]["draws"] += 1
            totals[strategy]["total_hits"] += h1 + h2
            totals[strategy]["tickets_ge3"] += (h1 >= 3) + (h2 >= 3)
            totals[strategy]["draws_ge3"] += (best >= 3)
            totals[strategy]["draws_ge4"] += (best >= 4)
            totals[strategy]["draws_ge5"] += (best >= 5)
            totals[strategy]["draws_ge6"] += (best >= 6)
            picks[strategy][best] += 1
            detail.append({
                "date": int(dates[idx]), "draw_id": int(ids[idx]),
                "strategy": strategy,
                "ticket_a": "-".join(f"{v:02d}" for v in t1),
                "ticket_b": "-".join(f"{v:02d}" for v in t2),
                "hit_a": h1, "hit_b": h2, "best_hits": best
            })
    return totals, picks, detail


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward 2-ticket Lotto649 strategy comparison")
    ap.add_argument("--window", type=int, default=100, help="Past draws used for each decision (default: 100)")
    ap.add_argument("--seed", type=int, default=649)
    args = ap.parse_args()
    dates, ids, numbers = load_draws()
    if args.window < 20 or args.window >= len(numbers):
        ap.error("--window must be >=20 and less than the number of draws")
    print(f"Lotto649 Walk-Forward Backtest v{VERSION}")
    print(f"[INFO] {len(numbers):,} base draws | window={args.window} | seed={args.seed}")
    totals, distributions, detail = evaluate(numbers, dates, ids, args.window, args.seed)
    summary = []
    for strategy in STRATEGIES:
        t = totals[strategy]
        n = t["draws"]
        summary.append({
            "strategy": strategy, "test_draws": n, "tickets": 2 * n,
            "avg_hits_per_ticket": round(t["total_hits"] / (2 * n), 6),
            "theoretical_avg_hits_per_ticket": round(THEORETICAL_MEAN_HITS_PER_TICKET, 6),
            "tickets_ge3": t["tickets_ge3"],
            "draws_at_least_one_ge3": t["draws_ge3"],
            "draws_at_least_one_ge4": t["draws_ge4"],
            "draws_at_least_one_ge5": t["draws_ge5"],
            "draws_at_least_one_ge6": t["draws_ge6"],
            "best_hit_distribution": {str(i): distributions[strategy][i] for i in range(7)}
        })
        print(f"[PASS] {strategy:6s}: avg_hits/ticket={summary[-1]['avg_hits_per_ticket']:.4f}, "
              f"draws>=3={t['draws_ge3']}/{n}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "detail.csv", detail)
    flat_summary = [{k: v for k, v in row.items() if k != "best_hit_distribution"} for row in summary]
    write_csv(OUTPUT / "summary.csv", flat_summary)
    report = {
        "version": VERSION, "train_window": args.window, "seed": args.seed,
        "first_test_date": int(dates[args.window]), "last_test_date": int(dates[-1]),
        "ticket_policy": "Two distinct nonoverlapping 6-number tickets per draw.",
        "scoring": "Main six only; 3+ hits is not a payout guarantee without the special number.",
        "strategies": summary,
        "limitations": [
            "Strategies are selected using only the preceding fixed-size training window.",
            "No transaction payouts, jackpot sharing, or bonus prize rules modeled.",
            "These are correlated, repeated historical trials; differences do not establish predictive advantage.",
            "Random strategy uses deterministic RNG stream; one seed is not a confidence interval.",
            "No strategy is used to nominate final fixed ticket numbers."
        ]
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"Lotto649 Walk-Forward Backtest v{VERSION}",
        f"Window: {args.window} | Testing: {int(dates[args.window])} - {int(dates[-1])}",
        f"Test draws: {len(numbers) - args.window} | 2 tickets each",
        "", "Strategy         Mean hits/ticket       Draws with >=3      Draws with >=4"
    ]
    for row in summary:
        lines.append(
            f"{row['strategy']:<16} {row['avg_hits_per_ticket']:>8.4f}"
            f"               {row['draws_at_least_one_ge3']:>5}"
            f"               {row['draws_at_least_one_ge4']:>5}"
        )
    lines += ["", "No final ticket recommendations are made by this module.",
              "Observed backtest performance does not change theoretical fair-draw odds."]
    (OUTPUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {OUTPUT.resolve()}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
