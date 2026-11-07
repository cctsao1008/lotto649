"""Lotto7x7 v0.6 Phase 1: build reproducible image-derived six-number pools.

Pool size and physical ticket count are separate: candidates are research
objects; the optional 1..5 preview is an *unoptimized* random sample.
No claim of above-chance forecasting or lottery return is made.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import hashlib
import json
from math import comb

import numpy as np

from .shape_engine_v06 import (
    SIDE, PIXELS, UNIVERSE_SIZE, bitmap, enumerate_placements,
    history_shape_families, parse_shape_key, validate_images,
)

SCHEMA = "lotto7x7.shape_pool.v0.6.phase1"
POLICIES = ("uniform-family", "orbit-weighted", "historical-frequency")


def _validate_ids(images: np.ndarray, draw_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(draw_ids)
    if (ids.shape != (len(images),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("draw ids must be positive and strictly increasing")
    return np.asarray(ids, dtype=np.int64)


def _source_hash(x: np.ndarray, ids: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes(order="C") + ids.tobytes(order="C")).hexdigest()


def _candidate_entry(placement, family: dict) -> dict:
    numbers = placement.numbers
    image = bitmap(numbers)
    return {
        "numbers": list(numbers),
        "bitmap": image.tolist(),
        "shape_id": family["shape_id"],
        "shape_key": family["shape_key"],
        "transform": {
            "mirrored": placement.mirrored,
            "rotations_ccw": placement.rotations_ccw,
            "row_shift": placement.row_shift,
            "col_shift": placement.col_shift,
        },
    }


def pool_digest(candidates: list[dict]) -> str:
    """Commit to the ordered candidate set (not to eventual draw outcomes)."""
    six_sets = [tuple(row["numbers"]) for row in candidates]
    if (not six_sets or any(len(x) != 6 or len(set(x)) != 6 or
                           tuple(sorted(x)) != x or
                           any(n < 1 or n > 49 for n in x)
                           for x in six_sets)
            or len(set(six_sets)) != len(six_sets)):
        raise ValueError("candidate sets must be unique sorted six-number sets")
    data = json.dumps(six_sets, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def random_sample(pool: dict, *, count: int = 5, seed: int = 1658) -> list[dict]:
    """Uniform selection of at most five distinct candidates; no optimization."""
    if (not isinstance(count, int) or not 1 <= count <= 5 or
            not isinstance(seed, int) or seed < 0):
        raise ValueError("sample count must be 1..5 and seed >= 0")
    candidates = pool.get("candidates")
    if (pool.get("schema") != SCHEMA or not isinstance(candidates, list)
            or len(candidates) < count or len(candidates) != pool.get("candidate_count")):
        raise ValueError("invalid or undersized v0.6 candidate pool")
    pool_digest(candidates)
    indices = np.random.default_rng(seed).permutation(len(candidates))[:count]
    return [{"pool_index": int(i), "numbers": candidates[int(i)]["numbers"],
             "shape_id": candidates[int(i)]["shape_id"]} for i in indices]


def create_shape_pool(images: np.ndarray, draw_ids: np.ndarray, *,
                      pool_size: int = 100, family_count: int = 20,
                      family_policy: str = "uniform-family", seed: int = 649,
                      sample_count: int = 5) -> dict:
    """Select historical shape families, then sample their legal D4 placements.

    Family/placement sampling uses only supplied historical images. Selection
    is seeded and deterministic; it is NOT trained against future draws.
    Selected family orbits partition the generated pool, so cross-family
    duplicate full-number sets are mathematically impossible.
    """
    if (not isinstance(pool_size, int) or pool_size < 1
            or not isinstance(family_count, int) or family_count < 1
            or family_count > pool_size or family_policy not in POLICIES
            or not isinstance(seed, int) or seed < 0
            or not isinstance(sample_count, int) or not 1 <= sample_count <= 5
            or sample_count > pool_size):
        raise ValueError("invalid pool size, family count, policy, seed or sample count")
    x = validate_images(images)
    ids = _validate_ids(x, draw_ids)
    families = history_shape_families(x, ids)
    if len(families) < family_count:
        raise ValueError("not enough distinct historical image families")
    rng = np.random.default_rng(seed)
    weights = np.array([
        1 if family_policy == "uniform-family" else
        f["orbit_size"] if family_policy == "orbit-weighted"
        else f["historical_count"]
        for f in families], dtype=np.float64)
    weights /= weights.sum()
    chosen_ids = rng.choice(len(families), size=family_count,
                            replace=False, p=weights)
    selected = [dict(families[int(i)]) for i in chosen_ids]
    placements = []
    capacity = 0
    for family in selected:
        all_positions = enumerate_placements(family["shape_key"])
        order = rng.permutation(len(all_positions))
        placements.append([all_positions[int(i)] for i in order])
        capacity += len(all_positions)
    if capacity < pool_size:
        raise ValueError("selected shape families cannot supply the requested pool size")
    rows: list[dict] = []
    cursor = [0] * family_count
    while len(rows) < pool_size:
        progressed = False
        for i, family in enumerate(selected):
            if len(rows) == pool_size:
                break
            if cursor[i] >= len(placements[i]):
                continue
            rows.append(_candidate_entry(placements[i][cursor[i]], family))
            cursor[i] += 1
            progressed = True
        if not progressed:
            raise AssertionError("shape pool could not reach requested size")
    digest = pool_digest(rows)
    stats = Counter(item["shape_id"] for item in rows)
    for family in selected:
        family["generated_count"] = stats[family["shape_id"]]
        keypoints = parse_shape_key(family["shape_key"])
        canonical_bitmap = np.zeros((SIDE, SIDE), dtype=np.uint8)
        for row, col in keypoints:
            canonical_bitmap[row, col] = 1
        family["canonical_bitmap"] = canonical_bitmap.tolist()
    report = {
        "schema": SCHEMA,
        "research_only": True,
        "source": {
            "draw_count": len(x),
            "first_draw_id": int(ids[0]),
            "cutoff_draw_id": int(ids[-1]),
            "history_sha256": _source_hash(x, ids),
        },
        "configuration": {
            "pool_size": pool_size, "family_count": family_count,
            "family_policy": family_policy, "seed": seed,
            "sample_count": sample_count, "preview_seed": seed + 1009,
            "spatial_layout": "sequential",
            "spatial_transform": "D4 + nonwrapping translation",
            "selection_method": "seeded family draw then round-robin placement draw",
        },
        "shape_library_family_count": len(families),
        "candidate_count": len(rows),
        "candidate_sha256": digest,
        "selected_families": selected,
        "candidates": rows,
        "fair_independent_reference": {
            "total_six_number_sets": UNIVERSE_SIZE,
            "candidate_pool_exact_six_coverage": pool_size / UNIVERSE_SIZE,
            "preview_exact_six_coverage": sample_count / UNIVERSE_SIZE,
        },
        "limitations": [
            "Image topology is imposed by the sequential 1..49 layout, not lottery physics.",
            "This version only selects shapes observed in historical data; no future draw enters the generator.",
            "Family policies induce different distributions over combinations; do not interpret as forecasts.",
            "Historical shape frequency is descriptive and does not imply increased future draw probability.",
            "The random preview is an unoptimized 1..5 ticket research sample, not a purchase instruction.",
            "For fair independent draws, each distinct six-number set has equal exact-hit probability.",
        ],
    }
    report["random_preview"] = random_sample(report, count=sample_count,
                                               seed=seed + 1009)
    return report


def _svg(image: list[list[int]], *, dim: int = 16) -> str:
    x = np.asarray(image)
    if (x.shape != (7, 7) or int(x.sum()) != 6
            or not np.all((x == 0) | (x == 1))):
        raise ValueError("invalid bitmap in shape pool report")
    parts = [f'<svg viewBox="0 0 {7*dim} {7*dim}" role="img" aria-label="7 by 7 six point bitmap">']
    for r in range(7):
        for c in range(7):
            color = '#347db7' if x[r, c] else '#e8eef4'
            parts.append(f'<rect x="{c*dim+1}" y="{r*dim+1}" width="{dim-2}" '
                         f'height="{dim-2}" rx="2" fill="{color}"/>')
    parts.append('</svg>')
    return ''.join(parts)


def shape_pool_gallery(report: dict) -> str:
    """Single-file offline HTML gallery of canonical forms and generated tickets."""
    if (report.get("schema") != SCHEMA
            or len(report.get("candidates", [])) != report.get("candidate_count")
            or pool_digest(report["candidates"]) != report.get("candidate_sha256")):
        raise ValueError("invalid or inconsistent v0.6 pool report")
    families = []
    for item in report["selected_families"]:
        families.append('<div class="family">' +
                        _svg(item["canonical_bitmap"]) +
                        f'<div><b>{escape(item["shape_id"])}</b>' +
                        f'<small>Observed {item["historical_count"]} · orbit '
                        f'{item["orbit_size"]} · generated {item["generated_count"]}</small>' +
                        '</div></div>')
    candidates = []
    for i, entry in enumerate(report["candidates"]):
        nums = ' '.join(f'{int(n):02d}' for n in entry["numbers"])
        candidates.append(f'<div class="candidate"><strong>#{i+1:03d}</strong>' +
                          _svg(entry["bitmap"]) +
                          f'<div class="nums">{nums}</div>' +
                          f'<small>{escape(entry["shape_id"])}</small></div>')
    selected = []
    for row in report["random_preview"]:
        numbers = ' '.join(f'{int(n):02d}' for n in row["numbers"])
        selected.append(f'<div class="bet"><b>Pool #{row["pool_index"]+1:03d}</b>'
                        f'<span>{numbers}</span></div>')
    cfg = report["configuration"]
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Lotto7x7 v0.6 Shape Pool</title><style>'
            'body{font:14px system-ui,sans-serif;max-width:1180px;'
            'margin:22px auto;padding:14px;color:#25384c;background:#f4f7fb}'
            'section{background:white;border-radius:13px;padding:20px;margin:18px 0}'
            'h1,h2{margin:4px 0 12px}h2{font-size:19px}'
            '.muted,small{color:#60758c}small{display:block;font-size:11px}'
            '.stats{display:flex;flex-wrap:wrap;gap:20px;margin:18px 0}'
            '.stat{background:#ebf2fa;padding:12px;border-radius:9px}'
            '.stat b{font-size:23px;display:block}.families{display:grid;'
            'grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}'
            '.family{border:1px solid #e5edf6;border-radius:9px;padding:8px}'
            '.family svg{width:110px;display:block;margin:auto}'
            '.family b{font-size:11px}.family small{line-height:1.5}'
            '.candidates{display:grid;grid-template-columns:repeat(auto-fill,'
            'minmax(140px,1fr));gap:12px}.candidate{border:1px solid #e5edf6;'
            'padding:12px;border-radius:9px;display:grid;justify-items:center;gap:6px}'
            '.candidate svg{width:125px}.candidate strong{justify-self:start}'
            '.nums{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12px}'
            '.bet{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;'
            'border-bottom:1px solid #e7edf3;padding:10px 0;font-variant-numeric:tabular-nums}'
            '@media print{.candidate{break-inside:avoid}.family{break-inside:avoid}}'
            '</style><h1>Lotto7x7 · Shape-Based Candidate Pool</h1>'
            '<p class="muted">Research-only image-driven selection, not a prediction guarantee.</p>'
            f'<div class="stats"><div class="stat"><b>{report["candidate_count"]}</b>distinct candidates</div>'
            f'<div class="stat"><b>{len(report["selected_families"])}</b>shape families</div>'
            f'<div class="stat"><b>{len(report["random_preview"])}</b>random preview only</div></div>'
            f'<p>Source cutoff draw: {report["source"]["cutoff_draw_id"]} · '
            f'Policy: {escape(cfg["family_policy"])} · Seed: {cfg["seed"]}</p>'
            '<section><h2>Preview — randomly sampled from the pool</h2><div>'+
            ''.join(selected)+'</div><p class="muted">Preview has no advantage over fair '
            'random six-number sets for a complete 6/6 hit.</p></section>'
            '<section><h2>Selected canonical shapes</h2><div class="families">'+
            ''.join(families)+'</div></section>'
            '<section><h2>All candidate images</h2><div class="candidates">'+
            ''.join(candidates)+'</div></section>'
            '<p class="muted">Each bitmap is exact-six. A shape may appear at different '
            'positions and orientations. Selection and bets are distinct research stages.</p>'
            '</html>')
