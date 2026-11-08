"""v0.22.2: unpaced historical exact-hit stream; runs until Ctrl+C.

A guess is an independently sampled legal 6/49 six-set. Duplicates may occur
across and inside batches and count as separate guesses; exact hit EVENTS may
therefore repeat. The concise console reports only first 10 unique historical
hits and each 100th unique hit by default; CSV records every *new* historical
six-set. All events (including repeats) count toward the requested hit rate.

The optional shape_score order only reorders each pre-generated IID candidate
batch according to frozen historical L3 q/p. It does not create a predictive
advantage or change a completed batch's number of exact historical hits.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import time
from typing import Callable

import numpy as np

from .historical_guess_v022 import build_reference, encode_tickets, _paths, _write_summary
from .shape_prospective_v019 import read_sources, file_sha256
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import UNIVERSE, classification_keys

SCHEMA = 'lotto7x7.historical_hit_stream.v0.22.2'
HIT_COLUMNS = ('timestamp_utc', 'guess_number', 'hit_number', 'unique_hit_number',
               'numbers', 'historical_draw_ids', 'strategy', 'batch_number')
ROUND_COLUMNS = ('batch_number', 'guesses_in_batch', 'exact_hits_in_batch',
                 'total_guesses', 'total_hits', 'unique_historical_six_sets', 'elapsed_seconds')


def make_fast_batch(rng: np.random.Generator, n: int) -> np.ndarray:
    """Independent uniform legal six-sets; repeat tickets allowed, for speed."""
    if not isinstance(n, int) or not 1 <= n <= 65536:
        raise ValueError('batch size must be 1..65536')
    keys = rng.random((n, 49), dtype=np.float64)
    return np.sort(np.argpartition(keys, kth=5, axis=1)[:, :6] + 1, axis=1).astype(np.uint8)


def ordered_batch(pool: np.ndarray, strategy: str, reference: dict) -> np.ndarray:
    """Frozen L3 ranking; never looks at target membership while ordering."""
    if strategy == 'uniform':
        return pool
    if strategy != 'shape_score':
        raise ValueError('strategy must be uniform or shape_score')
    classes = classification_keys(classify_numbers(pool))['edge'].astype(np.int64)
    codes = np.asarray(reference['class_codes'], dtype=np.int64)
    indices = np.searchsorted(codes, classes)
    if (np.any(indices >= len(codes))
            or not np.array_equal(codes[indices], classes)):
        raise ValueError('candidate topology class missing from theory')
    ratios = np.asarray(reference['class_ratio'], dtype=np.float64)[indices]
    return pool[np.argsort(-ratios, kind='stable')]


def historical_matches(pool: np.ndarray, target_codes: np.ndarray) -> np.ndarray:
    """Vectorized exact matching with pre-sorted base-50 history codes."""
    keys = encode_tickets(pool)
    positions = np.searchsorted(target_codes, keys)
    within = positions < len(target_codes)
    result = np.zeros(len(keys), dtype=bool)
    if np.any(within):
        result[within] = (target_codes[positions[within]] == keys[within])
    return np.flatnonzero(result)


def _ratio_fields(hits: int, guesses: int) -> dict:
    p = hits / guesses if guesses else 0.0
    return {
        'hits_per_guess': p,
        'hit_rate_percent': 100.0 * p,
        'guesses_per_hit': guesses / hits if hits else None,
        'hit_rate_fraction': f'{hits}/{guesses}',
    }


def run_stream(*, catalog: Path, universe: Path, out: Path,
               strategy: str = 'shape_score', target: str = 'all',
               holdout_count: int = 500, tau: float = 384.0,
               batch_size: int = 10000, seed: int | None = None,
               checkpoint_seconds: float = 10.0, show_all_hits: bool = False,
               max_batches: int = 0,
               clock: Callable[[], float] = time.perf_counter,
               emit: Callable[[str], None] = print) -> dict:
    """Process independent legal six-sets as fast as possible until Ctrl+C.

    ``max_batches`` defaults to 0 (unlimited) and is intended for reproducible
    diagnostics. Ctrl+C stops, closes files and writes a final summary.
    Each completed batch is counted exactly once; an interrupted in-flight
    batch is intentionally discarded rather than misreported as completed.
    """
    if strategy not in ('uniform', 'shape_score') or target not in ('all', 'holdout'):
        raise ValueError('unsupported strategy or target')
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 65536:
        raise ValueError('batch_size must be 1..65536')
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('seed must be nonnegative int, or omitted for OS random')
    if not np.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0 or max_batches < 0:
        raise ValueError('checkpoint interval must be positive; max_batches >= 0')
    paths = _paths(Path(out))
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('output file already exists: choose a different --out')
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    # OS cryptographic randomness for seed selection; NumPy PCG64 does the
    # bulk high-speed sampling. Saved seed makes an independent run repeatable.
    actual_seed = secrets.randbits(128) if seed is None else seed
    rng = np.random.default_rng(actual_seed)
    ids, nums, tables, _ = read_sources(Path(catalog), Path(universe))
    ref = build_reference(ids, nums, tables, target=target,
                          holdout_count=holdout_count, tau=tau)
    lookup = ref['lookup']
    target_codes = np.asarray(sorted(lookup), dtype=np.uint64)
    if not len(target_codes):
        raise ValueError('no historical target numbers')
    summary = {
        'schema': SCHEMA,
        'settings': {
            'strategy': strategy, 'target': target,
            'holdout_count': holdout_count, 'tau': tau,
            'batch_size': batch_size, 'seed': actual_seed,
            'seed_source': 'OS_CSPRNG' if seed is None else 'explicit',
            'checkpoint_seconds': checkpoint_seconds,
            'show_all_hits': show_all_hits,
            'max_batches': max_batches,
            'sampling': 'IID uniformly sampled legal six-sets with replacement',
            'stop_policy': 'Ctrl+C; no time or total-guesses limit by default',
        },
        'source': {
            'catalog_sha256': file_sha256(Path(catalog)),
            'universe_sha256': file_sha256(Path(universe)),
            'historical_draw_count': len(ids),
            'target_count': ref['target_count'],
            'unique_target_six_sets': len(target_codes),
            'training_draws': ref['training_draws'],
            'history_leakage': bool(ref['all_history_leakage']),
        },
        'fair_baseline': {
            'probability_of_historical_match_per_guess': len(target_codes) / UNIVERSE,
            'note': 'Benchmark probability for independent uniform six-sets against a fixed historical archive',
        },
        'stats': {
            'status': 'RUNNING', 'stop_reason': None,
            'elapsed_seconds': 0.0, 'batches': 0,
            'guesses_completed': 0, 'hit_events': 0,
            'unique_historical_six_sets_hit': 0,
            'unique_historical_draw_ids_hit': 0,
            'first_hit_guess': None,
            **_ratio_fields(0, 0),
        },
        'cautions': [
            'Targets are PREVIOUSLY KNOWN historical results. This is a retrieval experiment, NOT future prediction.',
            'In target=all, the shape scorer is fitted to the target history (data leakage).',
            'Repeated six-sets across batches (or within one batch) count as separate guesses and can produce repeat hit events.',
            'Hit rate is hits/guesses, not an estimate of next-draw jackpot probability.',
            'Shape ordering within each batch does not change full-batch exact historical hit counts.',
            'Only completed batches are included in the accumulated numerator and denominator.',
        ],
    }
    st = summary['stats']
    seen_six: set[int] = set()
    seen_ids: set[int] = set()
    start = clock()
    last_checkpoint = start
    reason = 'interrupt'
    with paths['hits'].open('x', encoding='utf-8', newline='') as hit_file, \
         paths['rounds'].open('x', encoding='utf-8', newline='') as round_file:
        hits_writer = csv.DictWriter(hit_file, fieldnames=HIT_COLUMNS)
        hits_writer.writeheader()
        rounds_writer = csv.DictWriter(round_file, fieldnames=ROUND_COLUMNS)
        rounds_writer.writeheader()
        hit_file.flush()
        round_file.flush()
        _write_summary(paths['summary'], summary)
        try:
            while True:
                if max_batches and st['batches'] >= max_batches:
                    reason = 'batch_limit'
                    break
                # Build and rank the entire batch before considering targets.
                pool = make_fast_batch(rng, batch_size)
                ordered = ordered_batch(pool, strategy, ref)
                idx = historical_matches(ordered, target_codes)
                old_guesses = st['guesses_completed']
                old_hits = st['hit_events']
                st['batches'] += 1
                st['guesses_completed'] += batch_size
                st['hit_events'] += int(len(idx))
                if st['first_hit_guess'] is None and len(idx):
                    st['first_hit_guess'] = old_guesses + int(idx[0]) + 1
                # Batch size is bounded: looping over only the rare matches
                # keeps the hot path fast. Repeated events still count.
                for event_i, local_ix in enumerate(idx, 1):
                    row = ordered[int(local_ix)]
                    key = int(encode_tickets(row.reshape(1, 6))[0])
                    if key in seen_six:
                        continue
                    seen_six.add(key)
                    historic_ids = lookup[key]
                    seen_ids.update(int(x) for x in historic_ids)
                    nth_unique = len(seen_six)
                    nth_hit = old_hits + event_i
                    position = old_guesses + int(local_ix) + 1
                    numbers = ' '.join(f'{int(x):02d}' for x in row)
                    hits_writer.writerow({
                        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        'guess_number': position,
                        'hit_number': nth_hit,
                        'unique_hit_number': nth_unique,
                        'numbers': numbers,
                        'historical_draw_ids': ';'.join(str(x) for x in historic_ids),
                        'strategy': strategy,
                        'batch_number': st['batches'],
                    })
                    if show_all_hits or nth_unique <= 10 or nth_unique % 100 == 0:
                        emit(f'[NEW HIT #{nth_unique}] guess #{position:,} '
                             f'(total hits={nth_hit:,}) numbers={numbers} '
                             f'historic={";".join(str(x) for x in historic_ids)}')
                    if nth_unique == 10 and not show_all_hits:
                        emit('[INFO] Further distinct historical hits are saved to CSV; '
                             'terminal only shows each 100th new hit (faster and quieter).')
                st['unique_historical_six_sets_hit'] = len(seen_six)
                st['unique_historical_draw_ids_hit'] = len(seen_ids)
                elapsed = clock() - start
                st['elapsed_seconds'] = round(elapsed, 3)
                rounds_writer.writerow({
                    'batch_number': st['batches'],
                    'guesses_in_batch': batch_size,
                    'exact_hits_in_batch': len(idx),
                    'total_guesses': st['guesses_completed'],
                    'total_hits': st['hit_events'],
                    'unique_historical_six_sets': len(seen_six),
                    'elapsed_seconds': st['elapsed_seconds'],
                })
                if elapsed - last_checkpoint >= checkpoint_seconds:
                    st.update(_ratio_fields(st['hit_events'], st['guesses_completed']))
                    hit_file.flush()
                    round_file.flush()
                    _write_summary(paths['summary'], summary)
                    last_checkpoint = elapsed + start
        except KeyboardInterrupt:
            reason = 'ctrl_c'
        finally:
            st['elapsed_seconds'] = round(clock() - start, 3)
            st['status'] = 'INTERRUPTED' if reason == 'ctrl_c' else 'COMPLETED'
            st['stop_reason'] = reason
            st.update(_ratio_fields(st['hit_events'], st['guesses_completed']))
            hit_file.flush()
            round_file.flush()
            _write_summary(paths['summary'], summary)
    rate = st['hit_rate_percent']
    avg = st['guesses_per_hit']
    emit(f'[DONE] {st["status"]} | guesses={st["guesses_completed"]:,} '
         f'| hits={st["hit_events"]:,} '
         f'| hit_rate={rate:.6f}% '
         f'({st["hit_rate_fraction"]}) '
         f'| 1 hit / {avg:,.1f} guesses' if avg is not None else
         f'[DONE] {st["status"]} | guesses={st["guesses_completed"]:,} '
         f'| hits=0 | hit_rate=0% (0/{st["guesses_completed"]})')
    emit(f'[SAVED] {paths["summary"]} | {paths["hits"]} | {paths["rounds"]}')
    return summary
