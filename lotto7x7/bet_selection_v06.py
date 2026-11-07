"""Choose 1..5 distinct tickets from one frozen v0.6 shape-derived pool.

Strategies share the same candidates; coverage optimization is exact greedy
lexicographic >=3, >=4, >=5 with a one-swap improvement pass. These are
mathematical coverage objectives, not future-draw predictions.
"""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np

from .candidate_pool_v06 import SCHEMA as POOL_SCHEMA, pool_digest
from .shape_engine_v06 import bitmap, orientations, parse_shape_key, shape_key, shape_id
from .coverage_v06 import exact_coverage, normalized_bets

SCHEMA = "lotto7x7.bet_selection.v0.6"
STRATEGIES = ("random", "shape-diverse", "coverage-optimized")


def validate_pool(pool):
    """Verify pool commitment, shape identity, candidate uniqueness and metadata."""
    if not isinstance(pool, dict) or pool.get("schema") != POOL_SCHEMA:
        raise ValueError("expected v0.6 Phase 1 shape-pool JSON")
    rows = pool.get("candidates")
    if (not isinstance(rows, list) or not rows or
            pool.get("candidate_count") != len(rows) or
            not isinstance(pool.get("candidate_sha256"), str) or
            pool_digest(rows) != pool["candidate_sha256"]):
        raise ValueError("inconsistent shape-pool candidate digest or count")
    numbers = normalized_bets([row["numbers"] for row in rows], maximum=len(rows))
    family_keys = {fam["shape_id"]: fam["shape_key"] for fam in pool["selected_families"]}
    for row, nums in zip(rows, numbers):
        if (list(nums) != row["numbers"] or
                row.get("shape_id") not in family_keys or
                row.get("shape_key") != family_keys[row["shape_id"]] or
                shape_key(bitmap(nums)) != row["shape_key"] or
                shape_id(row["shape_key"]) != row["shape_id"]):
            raise ValueError("candidate numbers/shape family identity mismatch")
    return rows, numbers


@lru_cache(maxsize=4096)
def _shape_distance(key_a: str, key_b: str) -> float:
    """Best D4+translation point overlap, independent of exact number hits."""
    if key_a == key_b:
        return 0.0
    aa = set(parse_shape_key(key_a))
    bb = parse_shape_key(key_b)
    best = 0
    for oriented, _, _ in orientations(bb):
        for ar, ac in aa:
            for br, bc in oriented:
                dr, dc = ar-br, ac-bc
                match = sum((r+dr, c+dc) in aa for r, c in oriented)
                if match > best:
                    best = match
                if best == 6:
                    return 0.0
    return (6-best) / 6.0


def _random_indices(m: int, count: int, seed: int) -> list[int]:
    return [int(i) for i in np.random.default_rng(seed).permutation(m)[:count]]


def _shape_diverse(rows, numbers, *, count, seed):
    ranking = _random_indices(len(rows), len(rows), seed)
    chosen = [ranking[0]]
    while len(chosen) < count:
        best_idx = None
        best_score = None
        for idx in ranking:
            if idx in chosen:
                continue
            num = set(numbers[idx])
            similarities = [_shape_distance(rows[idx]["shape_key"], rows[j]["shape_key"])
                            for j in chosen]
            overlaps = [len(num.intersection(numbers[j])) for j in chosen]
            score = (min(similarities), -max(overlaps), -sum(overlaps),
                     int(rows[idx]["shape_id"] not in {rows[j]["shape_id"] for j in chosen}))
            if best_score is None or score > best_score:
                best_idx, best_score = idx, score
        chosen.append(best_idx)
    return chosen


def _coverage_key(numbers, indices):
    metric = exact_coverage([numbers[j] for j in indices])["at_least_counts"]
    return (metric["3"], metric["4"], metric["5"])


def _coverage_optimized(numbers, *, count, seed):
    """Exact greedy, followed by deterministic first-improvement 1-swap search.

    Uses lexicographic threshold priority: >=3, then >=4, then >=5. No
    arbitrary probability weights and no use of a future winning draw.
    """
    order = _random_indices(len(numbers), len(numbers), seed)
    chosen = []
    for _ in range(count):
        best_idx, best_key = None, None
        for j in order:
            if j in chosen:
                continue
            key = _coverage_key(numbers, chosen+[j])
            if best_key is None or key > best_key:
                best_idx, best_key = j, key
        chosen.append(best_idx)
    for _ in range(1):  # one full 1-swap improvement pass; deterministic order
        improved = False
        for slot in range(count):
            old_key = _coverage_key(numbers, chosen)
            for j in order:
                if j in chosen:
                    continue
                trial = chosen.copy()
                trial[slot] = j
                key = _coverage_key(numbers, trial)
                if key > old_key:
                    chosen = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return chosen


def select_bets(pool: dict, *, count: int = 5, strategy: str = "all",
                seed: int = 1658) -> dict:
    """Select ticket sets without conditioning on an unseen target draw."""
    if (not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 5 or
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or
            strategy not in ("all",) + STRATEGIES):
        raise ValueError("count must be 1..5, seed nonnegative and strategy supported")
    rows, numbers = validate_pool(pool)
    if count > len(rows):
        raise ValueError("insufficient pool candidates for requested ticket count")
    names = STRATEGIES if strategy == "all" else (strategy,)
    selections = {}
    for name in names:
        if name == "random":
            indices = _random_indices(len(rows), count, seed)
        elif name == "shape-diverse":
            indices = _shape_diverse(rows, numbers, count=count, seed=seed)
        else:
            indices = _coverage_optimized(numbers, count=count, seed=seed)
        picks = [{"pool_index": j, "numbers": list(numbers[j]),
                  "shape_id": rows[j]["shape_id"]} for j in indices]
        selections[name] = {
            "bets": picks,
            "exact_fair_coverage": exact_coverage([numbers[j] for j in indices]),
        }
    return {
        "schema": SCHEMA,
        "research_only": True,
        "source": pool.get("source"),
        "pool_schema": pool["schema"],
        "pool_sha256": pool["candidate_sha256"],
        "pool_candidate_count": len(rows),
        "pool_numbers": [list(x) for x in numbers],
        "config": {"count": count, "strategy": strategy, "seed": seed,
                   "shape_distance": "1 - maximal D4+translation six-point intersection/6",
                   "coverage_method": "exact count; greedy >=3, >=4, >=5 lexicographic then one 1-swap pass",
                   "coverage_scope": "fair independent 6/49 combinations, not predicted winning numbers"},
        "strategies": selections,
        "limitations": [
            "The pool size is not the number of purchased tickets.",
            "Coverage-optimized uses lexicographic >=3/4/5 and is not a proven global optimum.",
            "At 6/6 every portfolio of N distinct tickets has probability N/13983816.",
            "No target winning draw is used in selection; shape features do not create an edge.",
        ],
    }