"""Lotto7x7 v0.7: matched 6/49 baselines and chronological evaluation.

B0: uniformly random tickets from the complete universe.
B1: uniform 100-combination pool, followed by v0.6 strategies.
S1: historical shape pool, followed by the *same* v0.6 strategies.
B2: random 6*N disjoint labels partitioned into N tickets.

All generation and selection are functions of history through the cutoff only.
The target is observed solely in the scoring path. Retrospective evidence is
exploratory; this module does not establish a predictive lottery advantage.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from typing import Callable

import numpy as np

from .bet_selection_v06 import select_bets
from .candidate_pool_v06 import (SCHEMA as POOL_SCHEMA, create_shape_pool,
                                 pool_digest)
from .coverage_v06 import UNIVERSE, exact_coverage, normalized_bets
from .shape_engine_v06 import (bitmap, shape_id, shape_key, validate_images)

SCHEMA = "lotto7x7.matched_baselines.v0.7"
SNAPSHOT_SCHEMA = "lotto7x7.shadow_snapshot.v0.7"
SCORE_SCHEMA = "lotto7x7.shadow_score.v0.7"
ARMS = ("B0-global-random", "B1-random-pool", "S1-shape-pool", "B2-disjoint")
STRATEGIES = ("random", "shape-diverse", "coverage-optimized")


def _int_seed(seed: int, cutoff: int, label: str) -> int:
    raw = f"lotto7x7:v0.7:{seed}:{cutoff}:{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _checked_history(images, draw_ids):
    x = validate_images(np.asarray(images))
    ids = np.asarray(draw_ids)
    if (ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("history IDs must be positive and strictly increasing")
    return x, np.asarray(ids, dtype=np.int64)


def _settings(count, pool_size, family_count, seed):
    if any(type(v) is not int for v in (count, pool_size, family_count, seed)):
        raise ValueError("count, pool size, families, and seed must be integers")
    if not (1 <= count <= 5 and count <= pool_size and
            1 <= family_count <= pool_size and seed >= 0):
        raise ValueError("N=1..5, M>=N, families=1..M, seed>=0 required")


def _draw_numbers(image):
    numbers = tuple(int(v) for v in np.flatnonzero(image) + 1)
    return normalized_bets([numbers])[0]


def random_sets(count: int, seed: int, *, disjoint: bool = False):
    """Sample fixed, unique six-number tickets; B2 enforces zero pair overlap."""
    if type(count) is not int or not 1 <= count <= 5 or type(seed) is not int or seed < 0:
        raise ValueError("random ticket count must be 1..5 and seed >=0")
    rng = np.random.default_rng(seed)
    if disjoint:
        shuffled = rng.permutation(49)[:6 * count] + 1
        return [tuple(sorted(map(int, shuffled[6*i:6*i+6]))) for i in range(count)]
    sets = set()
    ordered = []
    while len(ordered) < count:
        row = tuple(sorted(int(v) for v in rng.choice(49, 6, replace=False) + 1))
        if row not in sets:
            sets.add(row)
            ordered.append(row)
    return ordered


def random_pool(size: int, seed: int, cutoff_draw_id: int):
    """Sample B1 uniformly from C(49,6), then present v0.6-compatible rows.

    Sampling without replacement by duplicate rejection is uniform over ordered
    distinct sequences; shape metadata is solely to reuse identical selectors.
    """
    if (type(size) is not int or not 1 <= size <= UNIVERSE or
            type(seed) is not int or seed < 0 or
            type(cutoff_draw_id) is not int or cutoff_draw_id <= 0):
        raise ValueError("invalid random candidate pool parameters")
    rng = np.random.default_rng(seed)
    rows, used, families = [], set(), {}
    while len(rows) < size:
        nums = tuple(sorted(int(v) for v in rng.choice(49, 6, replace=False) + 1))
        if nums in used:
            continue
        used.add(nums)
        key = shape_key(bitmap(nums))
        family_id = shape_id(key)
        if family_id in families and families[family_id]["shape_key"] != key:
            raise AssertionError("truncated shape-family ID collision")
        families.setdefault(family_id, {"shape_id": family_id, "shape_key": key})
        rows.append({"numbers": list(nums), "shape_id": family_id,
                     "shape_key": key, "bitmap": bitmap(nums).tolist()})
    return {"schema": POOL_SCHEMA,
            "candidate_count": size,
            "candidate_sha256": pool_digest(rows),
            "selected_families": list(families.values()),
            "candidates": rows,
            "source": {"type": "iid_uniform_six_number_pool",
                       "cutoff_draw_id": cutoff_draw_id,
                       "seed": seed,
                       "historical_draw_count_used": 0}}


def _portfolio(numbers, *, pool_indices=None):
    checked = normalized_bets(numbers)
    bets = []
    for i, nums in enumerate(checked):
        row = {"numbers": list(nums)}
        if pool_indices is not None:
            row["pool_index"] = pool_indices[i]
        bets.append(row)
    return {"bets": bets, "exact_fair_coverage": exact_coverage(checked)}


def make_arms(images, draw_ids, *, count=5, pool_size=100,
              family_count=20, seed=649):
    """Build all pre-draw arms using only the supplied history prefix."""
    _settings(count, pool_size, family_count, seed)
    x, ids = _checked_history(images, draw_ids)
    if len(x) < 50:
        raise ValueError("at least 50 past draws required")
    cutoff = int(ids[-1])
    b0 = random_sets(count, _int_seed(seed, cutoff, "B0"))
    b2 = random_sets(count, _int_seed(seed, cutoff, "B2"), disjoint=True)
    b1_pool = random_pool(pool_size, _int_seed(seed, cutoff, "B1-pool"), cutoff)
    s1_pool = create_shape_pool(x, ids, pool_size=pool_size,
                                family_count=family_count,
                                family_policy="uniform-family",
                                sample_count=min(5, count),
                                seed=_int_seed(seed, cutoff, "S1-pool"))
    records = {"B0-global-random": {
                   "generator": "uniform C(49,6) direct; independent of history",
                   "strategies": {"random": _portfolio(b0)}},
               "B2-disjoint": {
                   "generator": "random disjoint N tickets; no global optimum claimed",
                   "strategies": {"disjoint": _portfolio(b2)}}}
    for arm, pool in (("B1-random-pool", b1_pool), ("S1-shape-pool", s1_pool)):
        selection = select_bets(pool, count=count, strategy="all",
                                seed=_int_seed(seed, cutoff, "common-pool-selector"))
        records[arm] = {
            "generator": "iid uniform 6/49 pool" if arm.startswith("B1") else
                         "history-derived shape D4 placements",
            "pool_sha256": pool["candidate_sha256"],
            "pool_numbers": [r["numbers"] for r in pool["candidates"]],
            "pool_size": len(pool["candidates"]),
            "strategies": {name: _portfolio(
                [row["numbers"] for row in entry["bets"]],
                pool_indices=[row["pool_index"] for row in entry["bets"]])
                for name, entry in selection["strategies"].items()},
        }
    return {"count": count, "pool_size": pool_size, "family_count": family_count,
            "seed": seed, "history_cutoff_draw_id": cutoff,
            "history_draw_count": len(x),
            "history_sha256": hashlib.sha256(x.tobytes() + ids.tobytes()).hexdigest(),
            "arms": {name: records[name] for name in ARMS}}


def _score_portfolios(arms, actual):
    actual_set = set(normalized_bets([actual])[0])
    result = {}
    for name in ARMS:
        if name not in arms:
            raise ValueError("missing baseline arm")
        arm = arms[name]
        strategies = arm.get("strategies", {})
        expected = ("random",) if name == ARMS[0] else (
            ("disjoint",) if name == ARMS[3] else STRATEGIES)
        if set(strategies) != set(expected):
            raise ValueError("missing or unexpected strategy")
        result[name] = {}
        for method, payload in strategies.items():
            bets = normalized_bets([v["numbers"] for v in payload["bets"]])
            computed = exact_coverage(bets)
            if computed != payload["exact_fair_coverage"]:
                raise ValueError("coverage commitment mismatch")
            indices = [v.get("pool_index") for v in payload["bets"]]
            if "pool_numbers" in arm:
                pool = arm["pool_numbers"]
                if (pool_digest([{"numbers": row} for row in pool]) != arm["pool_sha256"]
                        or len(pool) != arm["pool_size"]):
                    raise ValueError("tampered baseline candidate pool")
                for idx, bet in zip(indices, bets):
                    if type(idx) is not int or not 0 <= idx < len(pool) or list(bet) != pool[idx]:
                        raise ValueError("selected ticket not in sealed pool")
            elif any(v is not None for v in indices):
                raise ValueError("direct baseline must not use pool indices")
            hits = [len(actual_set.intersection(b)) for b in bets]
            result[name][method] = {
                "ticket_hits": hits, "max_hits": max(hits),
                "any_ge3": int(max(hits) >= 3),
                "any_ge4": int(max(hits) >= 4),
                "any_ge5": int(max(hits) >= 5),
                "any_ge6": int(max(hits) == 6),
                "exact_fair_coverage": computed["at_least"],
            }
    return result


def _bootstrap_diff(series_a, series_b, *, seed, reps=1000):
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(reps, len(delta)))
    estimates = delta[draws].mean(axis=1)
    return {"paired_mean_difference": float(delta.mean()),
            "bootstrap_interval_95": [float(v) for v in np.quantile(estimates, [0.025, 0.975])],
            "method": "paired_bootstrap_1000_exploratory_not_multiplicity_adjusted"}


def rolling_origin(images, draw_ids, *, warmup=200, eval_draws=12,
                   stride=1, count=5, pool_size=100, family_count=20,
                   seed=649, on_progress: Callable[[str], None] | None = None):
    """Each target draw is scored only after pre-target arms were constructed.

    Uses the last eval_draws eligible chronological targets at requested stride.
    This is an exploratory retrospective comparison (NOT truly blind evidence).
    """
    _settings(count, pool_size, family_count, seed)
    x, ids = _checked_history(images, draw_ids)
    if (any(type(z) is not int for z in (warmup, eval_draws, stride))
            or warmup < 50 or warmup >= len(x) or eval_draws < 1 or stride < 1):
        raise ValueError("invalid warmup, evaluation draw count or stride")
    eligible = list(range(warmup, len(x), stride))
    indices = eligible[-eval_draws:]
    if len(indices) != eval_draws:
        raise ValueError("insufficient chronological targets")
    rows = []
    for pos, target in enumerate(indices, 1):
        before = make_arms(x[:target], ids[:target], count=count,
                           pool_size=pool_size, family_count=family_count,
                           seed=seed)
        actual = _draw_numbers(x[target])  # strictly after generating/locking bets
        scores = _score_portfolios(before["arms"], actual)
        rows.append({"target_draw_id": int(ids[target]),
                     "cutoff_draw_id": int(ids[target-1]),
                     "history_sha256": before["history_sha256"],
                     "actual": list(actual),
                     "selection_commitments": {
                         name: {"pool_sha256": arm.get("pool_sha256"),
                                "portfolios": {k: [b["numbers"] for b in v["bets"]]
                                               for k, v in arm["strategies"].items()}}
                         for name, arm in before["arms"].items()},
                     "scores": scores})
        if on_progress is not None:
            on_progress(f"[ROLLING {pos}/{len(indices)}] "
                        f"cutoff={ids[target-1]} -> target={ids[target]}")
    summary = {}
    for arm, method in [(a, m) for a in ARMS for m in
                        (("random",) if a == ARMS[0] else
                         ("disjoint",) if a == ARMS[3] else STRATEGIES)]:
        values = [row["scores"][arm][method] for row in rows]
        hist = [sum(v["max_hits"] == k for v in values) for k in range(7)]
        summary[f"{arm}/{method}"] = {
            "draws": len(rows), "max_hits_counts_0_to_6": hist,
            "mean_max_hits": float(np.mean([v["max_hits"] for v in values])),
            "realized_any_ge3_rate": float(np.mean([v["any_ge3"] for v in values])),
            "realized_any_ge4_rate": float(np.mean([v["any_ge4"] for v in values])),
            "realized_any_ge5_rate": float(np.mean([v["any_ge5"] for v in values])),
            "realized_any_ge6_rate": float(np.mean([v["any_ge6"] for v in values])),
            "mean_theoretical_ge3": float(np.mean([
                v["exact_fair_coverage"]["3"] for v in values])),
        }
    reference = [r["scores"][ARMS[0]]["random"]["any_ge3"] for r in rows]
    paired = {}
    for arm, method in [(a, m) for a in ARMS for m in
                        (("random",) if a == ARMS[0] else
                         ("disjoint",) if a == ARMS[3] else STRATEGIES)]:
        if arm == ARMS[0]:
            continue
        values = [r["scores"][arm][method]["any_ge3"] for r in rows]
        paired[f"{arm}/{method} vs B0/random"] = _bootstrap_diff(
            values, reference, seed=_int_seed(seed, int(ids[-1]), arm+method))
    return {"schema": SCHEMA, "research_only": True,
            "evaluation": "historical_rolling_origin_exploratory_not_prospective",
            "parameters": {"warmup": warmup, "eval_draws": eval_draws,
                           "stride": stride, "count": count, "pool_size": pool_size,
                           "family_count": family_count, "seed": seed},
            "target_count": len(rows), "targets": rows,
            "summary": summary, "paired_ge3_vs_B0": paired,
            "limitations": [
                "Target draw excluded from pool generation and selection in every step.",
                "Retrospective history has already been inspected; not blind validation.",
                "Paired bootstrap intervals are exploratory and not multiplicity adjusted.",
                "All baselines have identical N and actual target; B0/B2 lack 100-pool conditioning.",
                "Max-six probability for any N distinct precommitted bets remains N/C(49,6).",
                "B2 is a disjoint-portfolio baseline, not a proved global optimum.",
                "Bonus numbers and official payouts excluded.",
            ]}


def _json_sha(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def shadow_snapshot(images, draw_ids, *, cutoff_draw_id=None, count=5,
                    pool_size=100, family_count=20, seed=649,
                    code_commit="unknown", created_at_utc=None):
    """Pre-draw commitment; snapshot never accepts a target winning result."""
    x, ids = _checked_history(images, draw_ids)
    if cutoff_draw_id is not None:
        if type(cutoff_draw_id) is not int or cutoff_draw_id not in set(map(int, ids)):
            raise ValueError("cutoff draw ID must exist in the input history")
        end = int(np.searchsorted(ids, cutoff_draw_id, side="right"))
        x, ids = x[:end], ids[:end]
    pre = make_arms(x, ids, count=count, pool_size=pool_size,
                    family_count=family_count, seed=seed)
    timestamp = created_at_utc if created_at_utc is not None else (
        datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("timestamp must be nonempty")
    doc = {"schema": SNAPSHOT_SCHEMA,
           "research_only": True,
           "created_at_utc": timestamp,
           "code_commit": code_commit,
           "pre_draw": pre,
           "provenance_note": "Local timestamp / SHA256 alone do not prove pre-draw publication. Publish the digest externally before the next draw.",
           "limitations": ["Cutoff and source hash are bound to the supplied historical archive.",
                           "No unknown result is read; score with shadow-score only after the draw.",
                           "Save before the upcoming draw, preferably with trusted external timestamp."]}
    doc["snapshot_sha256"] = _json_sha(doc)
    return doc


def score_shadow(snapshot, *, actual_draw_id: int, actual_numbers):
    """Read-only post-draw scoring, validating the immutable snapshot digest."""
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported shadow snapshot")
    committed = dict(snapshot)
    digest = committed.pop("snapshot_sha256", None)
    if not isinstance(digest, str) or _json_sha(committed) != digest:
        raise ValueError("invalid shadow snapshot SHA256 commitment")
    if type(actual_draw_id) is not int or actual_draw_id <= snapshot["pre_draw"]["history_cutoff_draw_id"]:
        raise ValueError("scored draw must be strictly later than snapshot cutoff")
    actual = normalized_bets([actual_numbers])[0]
    scores = _score_portfolios(snapshot["pre_draw"]["arms"], actual)
    return {"schema": SCORE_SCHEMA, "research_only": True,
            "snapshot_sha256": digest, "cutoff_draw_id": snapshot["pre_draw"]["history_cutoff_draw_id"],
            "scored_draw_id": actual_draw_id, "actual_numbers": list(actual),
            "score": scores,
            "note": "Ex-post realized outcomes; snapshot was not optimized using these numbers."}