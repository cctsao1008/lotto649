"""v0.7.1 statistical robustness audits of matched pre-target portfolios.

Exploratory retrospective inference only. The draw, rather than a random seed,
is the independent scoring unit. This module never constructs tickets using a
future target result. Exact McNemar tests address paired 0/1 hit outcomes;
non-parametric intervals summarize paired draw-level differences.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from functools import lru_cache
from typing import Callable

import numpy as np

from .baseline_v07 import (ARMS, SCHEMA as BASELINE_SCHEMA,
                           STRATEGIES, rolling_origin)
from .coverage_v06 import exact_coverage, normalized_bets

AUDIT_SCHEMA = "lotto7x7.robustness_audit.v0.7.1"
SWEEP_SCHEMA = "lotto7x7.robustness_sweep.v0.7.1"
PRIMARY = ("random", "shape-diverse", "coverage-optimized")


def mcnemar_exact(s1_only: int, b1_only: int) -> float:
    """Two-sided exact conditional test; discordant outcome under H0 is Bin(n,.5)."""
    if (type(s1_only) is not int or type(b1_only) is not int or
            s1_only < 0 or b1_only < 0):
        raise ValueError("discordant cell counts must be nonnegative integers")
    n = s1_only + b1_only
    if n == 0:
        return 1.0
    k = min(s1_only, b1_only)
    # scipy is already a repository dependency; avoids overflow for large n.
    from scipy.stats import binom
    return min(1.0, float(2.0 * binom.cdf(k, n, 0.5)))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error adjusted p-values."""
    if not pvalues:
        raise ValueError("p-values cannot be empty")
    for p in pvalues.values():
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p-values must be probabilities")
    ranks = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted = {}
    high = 0.0
    m = len(ranks)
    for i, key in enumerate(ranks):
        high = max(high, (m-i) * pvalues[key])
        adjusted[key] = min(1.0, float(high))
    return {key: adjusted[key] for key in pvalues}


def _mean_interval(values: np.ndarray, *, seed: int, replicates: int = 4000) -> list[float]:
    """Resample *draws* as the units, not ticket rows or seed realizations."""
    x = np.asarray(values, dtype=float)
    if (x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or
            type(seed) is not int or seed < 0 or replicates < 100):
        raise ValueError("invalid bootstrap observations")
    rng = np.random.default_rng(seed)
    # Bounded chunk allocation, even for longer evaluations.
    samples = np.empty(replicates)
    for start in range(0, replicates, 256):
        end = min(replicates, start + 256)
        index = rng.integers(0, len(x), size=(end-start, len(x)))
        samples[start:end] = np.mean(x[index], axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


@lru_cache(maxsize=8192)
def _fair_thresholds(bets: tuple[tuple[int, ...], ...]) -> dict:
    """Independently recompute exact fair coverage for a sealed portfolio."""
    return exact_coverage(bets)["at_least"]


def _validate_rolling(report: dict) -> tuple[list[dict], int]:
    if not isinstance(report, dict) or report.get("schema") != BASELINE_SCHEMA:
        raise ValueError("expected a v0.7 matched-baselines JSON report")
    params = report.get("parameters", {})
    n = params.get("count")
    if type(n) is not int or not 1 <= n <= 5:
        raise ValueError("invalid per-portfolio ticket count")
    targets = report.get("targets")
    if (not isinstance(targets, list) or not targets or
            report.get("target_count") != len(targets)):
        raise ValueError("missing or inconsistent chronological targets")
    seen = set()
    previous_id = 0
    for target in targets:
        draw_id = target.get("target_draw_id")
        cutoff = target.get("cutoff_draw_id")
        if (type(draw_id) is not int or type(cutoff) is not int or
                cutoff >= draw_id or draw_id <= previous_id or draw_id in seen or
                not isinstance(target.get("history_sha256"), str)):
            raise ValueError("invalid chronological target or source commitment")
        seen.add(draw_id)
        previous_id = draw_id
        actual = normalized_bets([target["actual"]])[0]
        scores = target["scores"]
        selections = target["selection_commitments"]
        if set(scores) != set(ARMS) or set(selections) != set(ARMS):
            raise ValueError("incomplete arm records")
        for arm in ARMS:
            expected = ("random",) if arm == ARMS[0] else (
                ("disjoint",) if arm == ARMS[3] else STRATEGIES)
            if (set(scores[arm]) != set(expected) or
                    set(selections[arm]["portfolios"]) != set(expected)):
                raise ValueError("incomplete strategy records")
            for strategy in expected:
                tickets = normalized_bets(selections[arm]["portfolios"][strategy])
                if len(tickets) != n:
                    raise ValueError("portfolio is not the configured N")
                hits = [len(set(actual).intersection(ticket)) for ticket in tickets]
                result = scores[arm][strategy]
                if (hits != result.get("ticket_hits") or
                        max(hits) != result.get("max_hits") or
                        any(result.get(f"any_ge{k}") != int(max(hits) >= k)
                            for k in range(3, 7))):
                    raise ValueError("scored outcome does not match sealed selection")
                expected_cov = result.get("exact_fair_coverage", {})
                computed_cov = _fair_thresholds(tickets)
                if (any(str(k) not in expected_cov for k in range(3, 7)) or
                        any(not np.isclose(expected_cov[str(k)], computed_cov[str(k)],
                                               rtol=0, atol=1e-15)
                                for k in range(3, 7))):
                    raise ValueError("exact fair coverage does not match sealed tickets")
        if type(draw_id) is int and draw_id // 1000000 < 1:
            raise ValueError("invalid ROC-year draw ID")
    return targets, n


def _year(draw_id: int) -> str:
    return str(draw_id // 1_000_000 + 1911)


def _extract_pairs(targets: list[dict], method: str):
    s1 = np.asarray([row["scores"][ARMS[2]][method]["any_ge3"]
                     for row in targets], dtype=float)
    b1 = np.asarray([row["scores"][ARMS[1]][method]["any_ge3"]
                     for row in targets], dtype=float)
    ps1 = np.asarray([row["scores"][ARMS[2]][method]["exact_fair_coverage"]["3"]
                      for row in targets], dtype=float)
    pb1 = np.asarray([row["scores"][ARMS[1]][method]["exact_fair_coverage"]["3"]
                      for row in targets], dtype=float)
    return s1, b1, ps1, pb1


def _by_year(targets: list[dict], s1, b1, ps1, pb1):
    groups = defaultdict(list)
    for i, row in enumerate(targets):
        groups[_year(row["target_draw_id"])].append(i)
    out = {}
    for year, positions in sorted(groups.items()):
        indices = np.asarray(positions, dtype=int)
        out[year] = {
            "draws": len(indices),
            "s1_ge3_count": int(s1[indices].sum()),
            "b1_ge3_count": int(b1[indices].sum()),
            "observed_difference": float(np.mean(s1[indices]-b1[indices])),
            "fair_coverage_difference": float(np.mean(ps1[indices]-pb1[indices])),
        }
    return out


def audit_rolling(report: dict, *, seed: int = 649) -> dict:
    """Validate a frozen v0.7 report and compute direct S1-vs-B1 comparisons."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    targets, n = _validate_rolling(report)
    contrasts, pvalues = {}, {}
    for i, method in enumerate(PRIMARY):
        s1, b1, ps1, pb1 = _extract_pairs(targets, method)
        s1_only = int(np.count_nonzero((s1 == 1) & (b1 == 0)))
        b1_only = int(np.count_nonzero((s1 == 0) & (b1 == 1)))
        p = mcnemar_exact(s1_only, b1_only)
        pvalues[method] = p
        observed = s1 - b1
        excess = observed - (ps1-pb1)
        contrasts[method] = {
            "draws": len(targets),
            "both_ge3": int(np.count_nonzero((s1 == 1) & (b1 == 1))),
            "s1_only_ge3": s1_only, "b1_only_ge3": b1_only,
            "neither_ge3": int(np.count_nonzero((s1 == 0) & (b1 == 0))),
            "s1_ge3_rate": float(s1.mean()),
            "b1_ge3_rate": float(b1.mean()),
            "observed_difference": float(observed.mean()),
            "fair_coverage_difference": float((ps1-pb1).mean()),
            "excess_over_fair_difference": float(excess.mean()),
            "observed_paired_bootstrap_95": _mean_interval(observed, seed=seed + 199*i),
            "excess_paired_bootstrap_95": _mean_interval(excess, seed=seed + 199*i + 1),
            "mcnemar_two_sided_exact_p": p,
            "year": _by_year(targets, s1, b1, ps1, pb1),
        }
    adjusted = holm(pvalues)
    for method in PRIMARY:
        contrasts[method]["holm_familywise_adjusted_p"] = adjusted[method]
    source_hash = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    return {
        "schema": AUDIT_SCHEMA, "research_only": True,
        "source_schema": BASELINE_SCHEMA,
        "source_canonical_json_sha256": source_hash,
        "target_count": len(targets), "ticket_count_per_portfolio": n,
        "input_seed": report["parameters"].get("seed"),
        "audit_seed": seed,
        "comparisons": contrasts,
        "multiple_comparisons": {
            "family": "S1 vs B1 >=3 across three fixed selectors",
            "method": "Holm FWER (two-sided exact paired McNemar)",
            "tests": len(PRIMARY),
        },
        "limitations": [
            "Retrospective and multiply inspected historical draws: exploratory only.",
            "McNemar tests equality of paired marginal >=3 probabilities; does not prove predictive information.",
            "McNemar comparison does not itself adjust for the small difference in exact fair coverages.",
            "Draws are resampled as units for paired bootstrap; repeated random seeds are not new draws.",
            "No correction covers choices of historical periods, seeds, methods or metrics tried before this audit.",
        ],
    }


def _positive_tuple(spec: str | tuple[int, ...], *, low: int, high: int | None = None):
    """Strict de-duplicated comma-separated integer list."""
    try:
        values = tuple(int(s.strip()) for s in spec.split(",")) if isinstance(spec, str) else tuple(spec)
    except (ValueError, TypeError) as exc:
        raise ValueError("expected comma-separated integer values") from exc
    if (not values or any(type(x) is not int or x < low or
                          (high is not None and x > high) for x in values) or
            len(set(values)) != len(values)):
        raise ValueError("values must be unique integers in requested bounds")
    return values


def sweep_rolling(images: np.ndarray, draw_ids: np.ndarray, *,
                  seeds=(649, 650, 651), counts=(1, 3, 5),
                  warmup=200, eval_draws=12, stride=5,
                  pool_size=100, family_count=20,
                  on_progress: Callable[[str], None] | None = None) -> dict:
    """Same chronological targets for every (N,seed); no pseudo-replication."""
    seed_values = _positive_tuple(seeds, low=0)
    count_values = _positive_tuple(counts, low=1, high=5)
    records = {}
    reference_ids = None
    for count in count_values:
        for seed in seed_values:
            if on_progress:
                on_progress(f"[SWEEP] N={count} seed={seed} targets={eval_draws}")
            study = rolling_origin(
                images, draw_ids, count=count, seed=seed,
                warmup=warmup, eval_draws=eval_draws, stride=stride,
                pool_size=pool_size, family_count=family_count)
            targets = study["targets"]
            target_ids = [row["target_draw_id"] for row in targets]
            if reference_ids is None:
                reference_ids = target_ids
            elif target_ids != reference_ids:
                raise AssertionError("sweep target IDs differ across N or seed")
            audit = audit_rolling(study, seed=seed)
            records[f"N{count}-seed{seed}"] = {
                "count": count, "seed": seed,
                "comparisons": {k: {key: val for key, val in v.items()
                                    if key != "year"}
                                for k, v in audit["comparisons"].items()},
                "series": {
                    method: {
                        "s1": [int(row["scores"][ARMS[2]][method]["any_ge3"]) for row in targets],
                        "b1": [int(row["scores"][ARMS[1]][method]["any_ge3"]) for row in targets],
                        "s1_fair": [float(row["scores"][ARMS[2]][method]["exact_fair_coverage"]["3"]) for row in targets],
                        "b1_fair": [float(row["scores"][ARMS[1]][method]["exact_fair_coverage"]["3"]) for row in targets],
                    } for method in PRIMARY},
            }
    all_p = {f"{name}/{method}": entry["comparisons"][method]["mcnemar_two_sided_exact_p"]
             for name, entry in records.items() for method in PRIMARY}
    adjusted = holm(all_p)
    for name, entry in records.items():
        for method in PRIMARY:
            entry["comparisons"][method]["holm_sweep_adjusted_p"] = adjusted[f"{name}/{method}"]
    aggregate = {}
    for count in count_values:
        aggregate[f"N{count}"] = {}
        for j, method in enumerate(PRIMARY):
            # One averaged S1-B1 difference per *draw* across seeds.
            values = np.stack([
                np.asarray(records[f"N{count}-seed{seed}"]["series"][method]["s1"], dtype=float)-
                np.asarray(records[f"N{count}-seed{seed}"]["series"][method]["b1"], dtype=float)
                for seed in seed_values], axis=0)
            expected = np.stack([
                np.asarray(records[f"N{count}-seed{seed}"]["series"][method]["s1_fair"], dtype=float)-
                np.asarray(records[f"N{count}-seed{seed}"]["series"][method]["b1_fair"], dtype=float)
                for seed in seed_values], axis=0)
            per_draw = values.mean(axis=0)
            fair = expected.mean(axis=0)
            excess = per_draw - fair
            per_seed = [float(v) for v in values.mean(axis=1)]
            aggregate[f"N{count}"][method] = {
                "independent_target_draws": len(per_draw),
                "seed_replicates_per_draw": len(seed_values),
                "per_seed_observed_differences": dict(zip(map(str, seed_values), per_seed)),
                "mean_observed_difference": float(per_draw.mean()),
                "mean_fair_coverage_difference": float(fair.mean()),
                "mean_excess_difference": float(excess.mean()),
                "draw_level_paired_bootstrap_95": _mean_interval(per_draw, seed=8281 + count*53 + j),
                "draw_level_excess_bootstrap_95": _mean_interval(excess, seed=9071 + count*53 + j),
                "year": {year: {
                    "draws": len(pos),
                    "observed_difference": float(per_draw[pos].mean()),
                    "excess_difference": float(excess[pos].mean()),
                } for year, pos in _year_positions(reference_ids).items()},
            }
    return {
        "schema": SWEEP_SCHEMA, "research_only": True,
        "target_draw_ids": reference_ids,
        "independent_target_draws": len(reference_ids),
        "seeds": list(seed_values), "counts": list(count_values),
        "parameters": {"warmup": warmup, "eval_draws": eval_draws,
                       "stride": stride, "pool_size": pool_size,
                       "family_count": family_count},
        "runs": records, "draw_clustered_aggregate": aggregate,
        "multiple_comparisons": {
            "method": "Holm FWER across all seed x N x strategy exact McNemar tests",
            "tests": len(all_p), "hypothesis_family_predeclared": False,
        },
        "limitations": [
            "Random seed replicates reuse the same winning draws; no seed is treated as a new independent target.",
            "Bootstrap averages observations across seeds for each draw, then resamples draws only.",
            "Historical replay is exploratory; many configurations increase the risk of data snooping.",
            "The Holm correction covers tests in this sweep, not any prior experiments.",
        ],
    }


def _year_positions(target_ids):
    d = defaultdict(list)
    for i, draw_id in enumerate(target_ids):
        d[_year(draw_id)].append(i)
    return dict(sorted(d.items()))
