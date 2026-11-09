"""v0.25 P0/P1 foundation: audited number-frequency deficit compensation.

This is a deterministic *selection heuristic*, not a next-draw probability
model. Shape/multivariate and historical backtest are tracked separately in #8/#9.
"""
from __future__ import annotations

import csv
from datetime import date
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path
import random

UNIVERSE = comb(49, 6)
P = 6 / 49
P2 = 6 * 5 / (49 * 48)
# Compound-symmetry covariance has this eigenvalue on the sum-zero subspace.
LAMBDA_NUMBER = P - P2
REFERENCE = (
    (4, 9, 17, 24, 29, 42),
    (10, 13, 14, 27, 28, 34),
    (6, 12, 19, 20, 44, 48),
    (3, 7, 18, 33, 40, 47),
)


def validate_ticket(ticket):
    if len(ticket) != 6 or any(not isinstance(v, int) or isinstance(v, bool)
                              or not 1 <= v <= 49 for v in ticket):
        raise ValueError("each ticket needs six integers in 1..49")
    if len(set(ticket)) != 6:
        raise ValueError("duplicate number in ticket")
    return tuple(sorted(ticket))


def load_csv_history(folder: Path, cutoff: str | None = None):
    """Validate all rows and chronological ordering; hash *input CSV bytes*."""
    files = sorted(folder.glob("lotto649_????.csv"))
    if not files:
        raise ValueError("no annual lotto649 CSV files")
    records = []
    fingerprint = sha256()
    seen_ids, seen_dates = set(), set()
    last = None
    for path in files:
        raw = path.read_bytes()
        fingerprint.update(path.name.encode("ascii") + b"\0" + raw)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            expected = ["date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
            if reader.fieldnames != expected:
                raise ValueError(f"{path}: unexpected CSV schema")
            for row in reader:
                try:
                    current = date.fromisoformat(row["date"].replace("/", "-"))
                    ident = int(row["draw_id"])
                    numbers = validate_ticket(tuple(int(row[f"n{i}"]) for i in range(1, 7)))
                    bonus = int(row["bonus"])
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}: invalid row {row}") from exc
                if not 1 <= bonus <= 49 or bonus in numbers:
                    raise ValueError(f"{path}: invalid bonus")
                if ident in seen_ids or current in seen_dates:
                    raise ValueError(f"{path}: repeated draw/date")
                if last is not None and current <= last:
                    raise ValueError(f"{path}: non-chronological date")
                last = current
                seen_ids.add(ident)
                seen_dates.add(current)
                if cutoff is None or current <= date.fromisoformat(cutoff):
                    records.append((current.isoformat(), ident, numbers))
    if not records:
        raise ValueError("no historical draws before cutoff")
    return records, fingerprint.hexdigest()


def frequency_audit(records):
    counts = [0] * 49
    for _, _, ticket in records:
        for value in ticket:
            counts[value - 1] += 1
    n = len(records)
    expected = n * P
    # Across independent draws, each individual indicator is Bernoulli(p).
    sd = (n * P * (1 - P)) ** 0.5
    return [{"number": i + 1, "observed": value, "expected": expected,
             "deficit": expected - value,
             "z_marginal": (value - expected) / sd}
            for i, value in enumerate(counts)]


def _mean_after(records, tickets):
    n = len(records)
    counts = [0] * 49
    for _, _, ticket in records:
        for v in ticket:
            counts[v - 1] += 1
    for ticket in tickets:
        for v in validate_ticket(ticket):
            counts[v - 1] += 1
    return [v / (n + len(tickets)) for v in counts]


def discrepancy(records, tickets=(), ridge=0.0):
    """Exact indicator covariance inverse on the 48-D sum-zero eigenspace.

    Equivalent to r^T (Sigma_0 + ridge*I)^-1 r with r on that
    subspace. Deliberately NO N factor: before/after comparability.
    """
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if len(set(map(validate_ticket, tickets))) != len(tickets):
        raise ValueError("ticket combinations must be distinct")
    mu = _mean_after(records, tickets)
    return sum((v - P) ** 2 for v in mu) / (LAMBDA_NUMBER + ridge)


def deficit_select(records, count=4, pool_size=1000, seed=649, ridge=0.0):
    """Greedy + deterministic improving one-swap, bounded candidate pool.

    Uses cached indicator counts rather than recomputing all history in each
    trial. This is a local heuristic, NOT proof of a global optimum.
    """
    if not 1 <= count <= 49 or pool_size < count or pool_size > 20000:
        raise ValueError("invalid count / candidate pool size")
    if seed < 0 or ridge < 0:
        raise ValueError("seed and ridge must be nonnegative")
    rng = random.Random(seed)
    pool = []
    seen = set()
    while len(pool) < pool_size:
        ticket = tuple(sorted(rng.sample(range(1, 50), 6)))
        if ticket not in seen:
            seen.add(ticket)
            pool.append(ticket)
    historical_counts = [0] * 49
    for _, _, ticket in records:
        for value in ticket:
            historical_counts[value - 1] += 1
    n = len(records)

    def score(selected):
        counts = historical_counts.copy()
        for ticket in selected:
            for v in ticket:
                counts[v - 1] += 1
        target = (n + len(selected)) * P
        return sum((c - target) ** 2 for c in counts) / ((n + len(selected)) ** 2
                                                            * (LAMBDA_NUMBER + ridge))

    selected = []
    for _ in range(count):
        winner = min((t for t in pool if t not in selected),
                     key=lambda t: (score(selected + [t]), t))
        selected.append(winner)
    for _ in range(3):
        improved = False
        for i in range(count):
            current = score(selected)
            candidate = min((t for t in pool if t not in selected or t == selected[i]),
                            key=lambda t: (score(selected[:i] + [t] + selected[i + 1:]), t))
            trial = selected[:i] + [candidate] + selected[i + 1:]
            if score(trial) < current - 1e-15:
                selected = trial
                improved = True
        if not improved:
            break
    selected.sort()
    return {"tickets": [list(t) for t in selected],
            "baseline_discrepancy": discrepancy(records, ridge=ridge),
            "after_discrepancy": discrepancy(records, selected, ridge),
            "pool_size": pool_size, "seed": seed, "ridge": ridge,
            "method": "number-only greedy plus 3 one-swap passes",
            "globally_optimal": False}


def report(folder: Path, cutoff=None, pool_size=1000, seed=649):
    records, digest = load_csv_history(folder, cutoff)
    selected = deficit_select(records, pool_size=pool_size, seed=seed)
    before = selected["baseline_discrepancy"]
    return {"schema": "lotto7x7.deficit_compensation.v0.25.p0p1",
            "scope": "number-only; multivariate/shape and coverage validation pending",
            "historical_draws": len(records),
            "first_draw": records[0][1], "last_draw": records[-1][1],
            "last_date": records[-1][0], "cutoff": cutoff,
            "csv_sha256": digest, "universe": UNIVERSE,
            "theoretical_marginal_probability": P,
            "numbers": frequency_audit(records),
            "selected": {**selected,
                         "improvement": before - selected["after_discrepancy"]},
            "reference": {"tickets": [list(t) for t in REFERENCE],
                          "before": before,
                          "after": discrepancy(records, REFERENCE)},
            "interpretation": "Historical compensation is not next-draw predictability."}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path, default=Path("results/v025_deficit_foundation.json"))
    args = parser.parse_args(argv)
    result = report(args.csv_dir, args.cutoff, args.pool_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[v0.25 P0/P1] {result['historical_draws']} draws; last={result['last_draw']}")
    print(f"[SELECT] {result['selected']['tickets']}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
