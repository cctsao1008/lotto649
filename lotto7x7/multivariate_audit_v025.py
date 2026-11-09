"""v0.25 P3: multivariate fair-IID feature audit, *not* a predictor.

Exact fair-universe means: number indicators, sum, odd count, and span,
plus L1/L2/L3 class probabilities read from the exhaustive shape catalog.
Full cross-feature covariance is estimated from independent IID legal tickets,
never fitted from historical outcomes. Ridge-whitening is exploratory; the
rank-deficient nested feature vector requires regularization.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import comb, isfinite
from pathlib import Path

import numpy as np

from .coverage_v06 import exact_coverage
from .deficit_compensation_v025 import REFERENCE, load_csv_history, validate_ticket
from .shape_deficit_v025 import load_exact_baseline
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import LEVELS, UNIVERSE, classification_keys

SCHEMA = "lotto7x7.multivariate_compensation.v0.25.p3"
DIM_NAMES = ("sum", "odd_count", "span")


def registry(baseline):
    """Stable sparse category registry, with exact fair means."""
    names = [f"number_{n:02d}" for n in range(1, 50)]
    means = [6 / 49] * 49
    names.extend(DIM_NAMES)
    # Random sample without replacement from population 1..49.
    # Expected range = (sample_size-1)*(population_size+1)/(sample_size+1).
    means.extend((150.0, 6 * 25 / 49, 250 / 7))
    catalog = {}
    for level in LEVELS:
        weights = {int(k): int(v) for k, v in baseline["counts"][level].items()}
        classes = tuple(sorted(weights))
        catalog[level] = classes
        for code in classes:
            names.append(f"{level}_{code}")
            means.append(weights[code] / UNIVERSE)
    return tuple(names), np.asarray(means, dtype=float), catalog


def encode_features(tickets, catalog):
    """Each row describes ONE legal ticket, never a future label."""
    tickets = [validate_ticket(tuple(int(v) for v in row)) for row in tickets]
    if not tickets:
        raise ValueError("feature matrix requires at least one ticket")
    arr = np.asarray(tickets, dtype=np.int16)
    n = len(arr)
    number = np.zeros((n, 49), dtype=float)
    number[np.arange(n)[:, None], arr - 1] = 1.0
    fields = [
        arr.sum(axis=1), (arr % 2).sum(axis=1),
        arr[:, -1] - arr[:, 0],
    ]
    shapes = classification_keys(classify_numbers(arr))
    blocks = [number, np.column_stack(fields)]
    for level in LEVELS:
        classes = np.asarray(catalog[level], dtype=np.uint32)
        idx = np.searchsorted(classes, shapes[level])
        if np.any(idx >= len(classes)) or np.any(classes[idx] != shapes[level]):
            raise ValueError("ticket shape missing from exact baseline")
        block = np.zeros((n, len(classes)), dtype=float)
        block[np.arange(n), idx] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def iid_tickets(count, seed):
    if count < 256:
        raise ValueError("at least 256 IID draws needed for covariance estimate")
    rng = np.random.default_rng(seed)
    # Continuous-key order statistics yield independent fair legal 6/49 draws.
    out = np.empty((count, 6), dtype=np.int16)
    for start in range(0, count, 1024):
        end = min(start + 1024, count)
        u = rng.random((end - start, 49))
        out[start:end] = np.sort(np.argpartition(u, 5, axis=1)[:, :6] + 1, axis=1)
    return out


def fair_covariance(baseline, *, samples=8192, seed=649, ridge=0.25):
    """IID-estimated correlation, regularized in standardized coordinates.

    Theory *means* are exact. Covariance is not exhaustively enumerated:
    sample count, seed, rank, and ridge are recorded. No training history
    enters covariance fitting and no inverse singular matrix is computed.
    """
    if not isinstance(samples, int) or samples < 256 or samples > 200000:
        raise ValueError("IID sample count outside 256..200000")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid seed")
    if not isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive finite")
    names, theory, catalog = registry(baseline)
    x = encode_features(iid_tickets(samples, seed), catalog)
    covariance = np.cov(x, rowvar=False, ddof=1)
    variance = np.diag(covariance).copy()
    # Tiny theoretical categories may have zero simulated occurrences.
    # Stabilize with exact Bernoulli variance for the shape indicators.
    for idx in range(52, len(names)):
        p = theory[idx]
        variance[idx] = max(variance[idx], p * (1 - p), 1e-9)
    # Number indicators and physical features must have nonzero variance.
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError("nonpositive simulated variance")
    sd = np.sqrt(variance)
    corr = covariance / np.outer(sd, sd)
    # diag(corr) may be zero for rare classes absent from MC; identity
    # regularization still guarantees positive definiteness.
    regularized = (corr + corr.T) / 2 + ridge * np.eye(len(names))
    eig, basis = np.linalg.eigh(regularized)
    if eig.min() <= 0:
        raise AssertionError("regularization failed")
    return {
        "names": names, "theory": theory, "sd": sd,
        "basis": basis, "eigenvalues": eig,
        "meta": {
            "source": "independent uniform legal 6/49 Monte Carlo",
            "moments": "exact marginal means; MC estimated cross-covariance",
            "samples": samples, "seed": seed, "ridge": ridge,
            "dimension": len(names),
            "smallest_eigenvalue": float(eig.min()),
            "largest_eigenvalue": float(eig.max()),
            "condition_number": float(eig.max() / eig.min()),
        },
        "catalog": catalog,
    }


def distance(average, model):
    residual = (average - model["theory"]) / model["sd"]
    transformed = model["basis"].T @ residual
    return float(np.dot(transformed / model["eigenvalues"], transformed))


def portfolio_audit(records, model, tickets):
    tickets = tuple(validate_ticket(t) for t in tickets)
    if len(set(tickets)) != len(tickets):
        raise ValueError("portfolio contains repeated tickets")
    history = encode_features([r[2] for r in records], model["catalog"])
    n = len(history)
    historical_sum = history.sum(axis=0)
    before = distance(historical_sum / n, model)
    if tickets:
        chosen = encode_features(tickets, model["catalog"])
        after = distance((historical_sum + chosen.sum(axis=0)) /
                         (n + len(tickets)), model)
    else:
        after = before
    return {"before": before, "after": after, "improvement": before - after,
            "tickets": [list(t) for t in tickets]}



def select_multivariate_portfolio(records, model, *, pool_size=300,
                                  seed=649, coverage_constraint="reference-3plus"):
    """Bounded covariance-aware four-ticket joint search.

    Disjointness is a sufficient but not necessary condition for the A-D
    exact 3+ main-ball coverage floor.
    """
    if coverage_constraint not in ("none", "reference-3plus"):
        raise ValueError("unknown coverage constraint")
    if not isinstance(pool_size, int) or not 4 <= pool_size <= 5000:
        raise ValueError("pool size must be in 4..5000")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be nonnegative")
    from random import Random
    rng = Random(seed)
    tickets = []
    seen = set()
    while len(tickets) < pool_size:
        row = tuple(sorted(rng.sample(range(1, 50), 6)))
        if row not in seen:
            seen.add(row)
            tickets.append(row)
    history = encode_features([r[2] for r in records], model["catalog"])
    candidates = encode_features(tickets, model["catalog"])
    n = len(history)
    transform = (model["basis"] / np.sqrt(model["eigenvalues"])[None, :]) / model["sd"][:, None]
    residual = (history.sum(axis=0) - n * model["theory"]) @ transform
    increments = (candidates - model["theory"]) @ transform

    def score(indices):
        total = residual + increments[list(indices)].sum(axis=0) if indices else residual
        return float(total @ total / (n + len(indices)) ** 2)

    def feasible(indices):
        if len(set(indices)) != len(indices):
            return False
        if coverage_constraint == "none":
            return True
        used = set()
        for index in indices:
            values = set(tickets[index])
            if used & values:
                return False
            used.update(values)
        return True

    chosen = []
    for _ in range(4):
        options = [i for i in range(pool_size) if feasible(chosen + [i])]
        if not options:
            raise ValueError("candidate pool cannot form four pairwise-disjoint tickets")
        chosen.append(min(options, key=lambda i: (score(chosen + [i]), tickets[i])))
    for _ in range(3):
        changed = False
        for position in range(4):
            options = [i for i in range(pool_size)
                       if feasible(chosen[:position] + [i] + chosen[position + 1:])]
            replacement = min(options, key=lambda i:
                              (score(chosen[:position] + [i] + chosen[position + 1:]),
                               tickets[i]))
            trial = chosen[:position] + [replacement] + chosen[position + 1:]
            if score(trial) < score(chosen) - 1e-15:
                chosen = trial
                changed = True
        if not changed:
            break
    selected = tuple(sorted(tickets[i] for i in chosen))
    coverage = exact_coverage(selected)
    reference = exact_coverage(REFERENCE)
    meets = (coverage["at_least_counts"]["3"] >= reference["at_least_counts"]["3"])
    if coverage_constraint == "reference-3plus" and not meets:
        raise AssertionError("exact 3+ reference floor violated")
    return {
        "tickets": [list(t) for t in selected],
        "objective": score(chosen),
        "coverage_constraint": coverage_constraint,
        "meets_reference_3plus_floor": meets,
        "reference_3plus_count_floor": reference["at_least_counts"]["3"],
        "exact_main_ball_coverage": coverage,
        "pool_size": pool_size, "seed": seed,
        "algorithm": "greedy and improving one-swap local search",
        "globally_optimal": False,
    }


def report(csv_dir: Path, universe: Path, *, cutoff=None,
           samples=8192, seed=649, ridge=0.25,
           pool_size=300, coverage_constraint="reference-3plus"):
    records, history_hash = load_csv_history(csv_dir, cutoff)
    baseline, universe_hash = load_exact_baseline(universe)
    model = fair_covariance(baseline, samples=samples, seed=seed, ridge=ridge)
    names, theory = model["names"], model["theory"]
    # Exact fair mean assertions prevent silent formula drift.
    assert len(names) == len(theory)
    np.testing.assert_allclose(theory[:49].sum(), 6.0, atol=1e-12)
    audit = portfolio_audit(records, model, REFERENCE)
    selected = select_multivariate_portfolio(records, model, pool_size=pool_size,
                                             seed=seed, coverage_constraint=coverage_constraint)
    return {
        "schema": SCHEMA, "history_draws": len(records),
        "first_draw_id": records[0][1], "last_draw_id": records[-1][1],
        "last_date": records[-1][0],
        "csv_sha256": history_hash, "universe_sha256": universe_hash,
        "feature_registry": list(names),
        "theory_mean": theory.tolist(),
        "covariance": model["meta"],
        "reference": {**audit, "exact_main_ball_coverage": exact_coverage(REFERENCE)},
        "selected": selected,
        "scope": "multivariate audit only; no hybrid optimizer or prospective validation",
        "caution": "Optimization of descriptive historical deficit does not predict next draw.",
    }


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    p.add_argument("--universe", type=Path,
                   default=Path("results/v015_v016_p1_shape_catalog_universe.json"))
    p.add_argument("--cutoff", default=None)
    p.add_argument("--iid-samples", type=int, default=8192)
    p.add_argument("--seed", type=int, default=649)
    p.add_argument("--ridge", type=float, default=0.25)
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--coverage-constraint", choices=("none", "reference-3plus"), default="reference-3plus")
    p.add_argument("--out", type=Path, default=Path("results/v025_multivariate_audit.json"))
    args = p.parse_args(argv)
    result = report(args.csv_dir, args.universe, cutoff=args.cutoff,
                    samples=args.iid_samples, seed=args.seed, ridge=args.ridge,
                    pool_size=args.pool_size, coverage_constraint=args.coverage_constraint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[MULTIVARIATE] {result['history_draws']} draws; "
          f"{result['covariance']['dimension']} features")
    print(f"[IID] {result['covariance']['samples']} fair tickets; ridge={args.ridge}")
    print(f"[REFERENCE] distance={result['reference']['after']:.8f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
