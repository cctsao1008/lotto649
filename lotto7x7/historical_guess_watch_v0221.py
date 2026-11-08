"""v0.22.1: paced, quiet, 10,000-TOTAL historical exact-six retrieval.

One unique, fixed candidate pool per execution. Only matching candidates
produce terminal event messages. Compute/sort upfront and reveal events at
paced guess boundaries; the 20-minute duration is a presentation/throttle
budget, not additional learning or computation.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .historical_guess_v022 import (
    build_reference, make_pool, trial, _paths, _write_summary,
)
from .shape_prospective_v019 import read_sources, file_sha256
from .shape_theory_v016 import UNIVERSE

SCHEMA = 'lotto7x7.historical_guess_watch.v0.22.1'
STRATEGIES = ('shape_score', 'uniform')
HIT_COLUMNS = (
    'timestamp_utc', 'elapsed_seconds', 'hit_number', 'guess_number',
    'strategy', 'uniform_rank', 'shape_rank', 'numbers',
    'historical_draw_ids', 'shape_ratio', 'edge_class',
)
ROUND_COLUMNS = (
    'elapsed_seconds', 'strategy', 'max_guesses', 'guesses_completed',
    'hit_events', 'unique_historical_draw_ids', 'first_hit_guess', 'status',
)


def run_paced_experiment(*, catalog: Path, universe: Path, out: Path,
                         minutes: float = 20.0, max_guesses: int = 10000,
                         seed: int = 649, target: str = 'all',
                         holdout_count: int = 500, tau: float = 384.0,
                         strategy: str = 'shape_score',
                         clock: Callable[[], float] = time.perf_counter,
                         sleep: Callable[[float], None] = time.sleep,
                         emit: Callable[[str], None] = print) -> dict:
    """Pace at most ``max_guesses`` unique tickets across ``minutes``.

    Output is intentionally event-only: HIT lines for exact matches, plus
    a final DONE line and paths. No per-round repeats and no progress spam.
    ``clock`` and ``sleep`` are injectable for deterministic, fast tests.
    """
    if not np.isfinite(minutes) or minutes <= 0:
        raise ValueError('minutes must be finite and >0')
    if not isinstance(max_guesses, int) or isinstance(max_guesses, bool) or not 1 <= max_guesses <= 10000:
        raise ValueError('max_guesses must be 1..10000 for the ENTIRE experiment')
    if not isinstance(seed, int) or seed < 0 or strategy not in STRATEGIES:
        raise ValueError('invalid seed or strategy')
    if target not in ('all', 'holdout'):
        raise ValueError('target must be all or holdout')
    paths = _paths(Path(out))
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('output already exists: supply a new --out filename')
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    ids, nums, tables, _ = read_sources(Path(catalog), Path(universe))
    ref = build_reference(ids, nums, tables, target=target,
                          holdout_count=holdout_count, tau=tau)
    candidates = make_pool(np.random.default_rng(seed), max_guesses)
    evaluation = trial(candidates, ref)
    if evaluation['unique_candidates'] != max_guesses:
        raise AssertionError('candidate pool contains duplicates')

    # The same candidate set can be searched in both orders. The active
    # order is frozen BEFORE looking up outcomes for terminal presentation.
    if strategy == 'shape_score':
        search_rank = 'shape_rank'
    else:
        search_rank = 'uniform_rank'
    hit_by_rank = {int(hit[search_rank]): hit for hit in evaluation['hits']}

    start = clock()
    duration = float(minutes) * 60.0
    # First guess at t=0; last guess at t=duration. Even a 10,000-ticket
    # search will therefore not spin millions of trials per second.
    gap = duration / (max_guesses - 1) if max_guesses > 1 else 0.0
    target_probability = ref['unique_target_sets'] / UNIVERSE
    expected = max_guesses * target_probability
    p_any = -np.expm1(np.log1p(-ref['unique_target_sets'] /
                             (UNIVERSE - np.arange(max_guesses, dtype=np.float64))).sum())
    summary = {
        'schema': SCHEMA,
        'settings': {
            'minutes': minutes, 'max_guesses_total': max_guesses,
            'seed': seed, 'target': target, 'holdout_count': holdout_count,
            'tau': tau, 'strategy': strategy,
            'pacing_seconds_between_guesses': gap,
            'sample_mode': 'exactly one uniform distinct six-number pool',
            'console_mode': 'match-only; final summary',
        },
        'sources': {
            'catalog_sha256': file_sha256(Path(catalog)),
            'universe_sha256': file_sha256(Path(universe)),
            'historical_draw_count': len(ids),
            'training_draws': ref['training_draws'],
            'target_draw_count': ref['target_count'],
            'unique_target_sets': ref['unique_target_sets'],
            'all_history_leakage': ref['all_history_leakage'],
        },
        'fair_reference': {
            'universe': UNIVERSE,
            'expected_hits_total': expected,
            'probability_at_least_one_exact': p_any,
        },
        'stats': {
            'status': 'RUNNING', 'stop_reason': None,
            'elapsed_seconds': 0.0, 'guesses_completed': 0,
            'total_guesses_limit': max_guesses, 'hit_events': 0,
            'first_hit_guess': None, 'unique_historical_draw_ids': 0,
        },
        'cautions': [
            'Full-history target=all uses training data containing the target; historical retrieval, not future prediction.',
            'The CPU calculates the pool and ranks upfront. Time pacing does not perform extra learning.',
            'Shape-score and uniform rankings use the exact same unique candidate pool and find exactly the same hits if fully searched.',
            'The number of hits in 10,000 guesses does not increase by waiting 20 minutes.',
            'Under fair draws, any individual legal six-number combination has identical full-hit odds.',
        ],
    }
    stats = summary['stats']
    seen_ids: set[int] = set()
    events = 0
    first_hit_guess = None
    stop_reason = 'guess_limit'
    with paths['hits'].open('x', encoding='utf-8', newline='') as hf, \
         paths['rounds'].open('x', encoding='utf-8', newline='') as rf:
        hw = csv.DictWriter(hf, fieldnames=HIT_COLUMNS)
        hw.writeheader()
        hf.flush()
        rw = csv.DictWriter(rf, fieldnames=ROUND_COLUMNS)
        rw.writeheader()
        rf.flush()
        _write_summary(paths['summary'], summary)
        try:
            for position in range(1, max_guesses + 1):
                # Sleep until the next assigned guess time, then reveal.
                due = start + (position - 1) * gap
                remaining = due - clock()
                if remaining > 0:
                    sleep(remaining)
                stats['guesses_completed'] = position
                hit = hit_by_rank.get(position)
                if hit is None:
                    continue
                events += 1
                if first_hit_guess is None:
                    first_hit_guess = position
                seen_ids.update(int(x) for x in hit['historical_draw_ids'])
                elapsed = round(clock() - start, 3)
                numbers = ' '.join(f'{number:02d}' for number in hit['numbers'])
                record = {
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'elapsed_seconds': elapsed,
                    'hit_number': events,
                    'guess_number': position,
                    'strategy': strategy,
                    'uniform_rank': hit['uniform_rank'],
                    'shape_rank': hit['shape_rank'],
                    'numbers': numbers,
                    'historical_draw_ids': ';'.join(map(str, hit['historical_draw_ids'])),
                    'shape_ratio': hit['shape_ratio'],
                    'edge_class': hit['edge_class'],
                }
                hw.writerow(record)
                hf.flush()
                stats['hit_events'] = events
                stats['first_hit_guess'] = first_hit_guess
                stats['unique_historical_draw_ids'] = len(seen_ids)
                stats['elapsed_seconds'] = elapsed
                _write_summary(paths['summary'], summary)
                emit(f'[HIT #{events}] guess #{position:,}/{max_guesses:,} '
                     f'numbers={numbers} historic={record["historical_draw_ids"]} '
                     f'(uniform=#{hit["uniform_rank"]:,}; shape=#{hit["shape_rank"]:,})')
        except KeyboardInterrupt:
            stop_reason = 'interrupted'
        stats['elapsed_seconds'] = round(clock() - start, 3)
        stats['status'] = 'COMPLETED' if stop_reason == 'guess_limit' else 'INTERRUPTED'
        stats['stop_reason'] = stop_reason
        rw.writerow({
            'elapsed_seconds': stats['elapsed_seconds'],
            'strategy': strategy, 'max_guesses': max_guesses,
            'guesses_completed': stats['guesses_completed'],
            'hit_events': events,
            'unique_historical_draw_ids': len(seen_ids),
            'first_hit_guess': first_hit_guess or '',
            'status': stats['status'],
        })
        rf.flush()
        _write_summary(paths['summary'], summary)
    emit(f'[DONE] {stats["status"]}: guesses={stats["guesses_completed"]:,}/'
         f'{max_guesses:,}, hits={events}, elapsed={stats["elapsed_seconds"]:.1f}s')
    return summary
