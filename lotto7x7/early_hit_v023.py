"""v0.23: paired, chronologically held-out early-hit efficiency study.

For every independent random candidate pool, compare the first K guesses in
uniform order versus frozen L3 shape-score order. Training uses ONLY earlier
historical draw records, and all candidate generation is outcome-blind.
The test is retrospective, not evidence of a future lottery edge.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import secrets
import time
from typing import Callable

import numpy as np
from scipy.stats import binomtest, ttest_1samp

from .historical_guess_v022 import build_reference, encode_tickets, _write_summary
from .historical_guess_stream_v0222 import make_fast_batch
from .shape_prospective_v019 import read_sources, file_sha256
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import UNIVERSE, classification_keys

SCHEMA = 'lotto7x7.early_hit_efficiency.v0.23'
DEFAULT_CUTOFFS = (100, 500, 1000, 5000)


def lookup_matches(tickets: np.ndarray, sorted_target_codes: np.ndarray) -> np.ndarray:
    """One boolean match flag per six-number ticket; no target peeking in scorer."""
    keys = encode_tickets(tickets)
    positions = np.searchsorted(sorted_target_codes, keys)
    valid = positions < len(sorted_target_codes)
    matches = np.zeros(len(keys), dtype=np.bool_)
    if np.any(valid):
        matches[valid] = sorted_target_codes[positions[valid]] == keys[valid]
    return matches


def ranking_from_training(tickets: np.ndarray, reference: dict) -> np.ndarray:
    """Return candidate index order sorted by TRAIN-only L3 posterior / fair ratio."""
    codes = np.asarray(reference['class_codes'], dtype=np.int64)
    labels = classification_keys(classify_numbers(tickets))['edge'].astype(np.int64)
    index = np.searchsorted(codes, labels)
    if (np.any(index >= len(codes))
            or not np.array_equal(codes[index], labels)):
        raise ValueError('an L3 candidate category is absent from exact fair table')
    ratios = np.asarray(reference['class_ratio'], dtype=np.float64)[index]
    return np.argsort(-ratios, kind='stable')


def evaluate_paired_pool(tickets: np.ndarray, ref: dict,
                         target_codes: np.ndarray, cutoffs: tuple[int, ...]) -> dict:
    """Guarantee identical total historical hits; only ordering is different."""
    match = lookup_matches(tickets, target_codes)
    shape_index = ranking_from_training(tickets, ref)
    shape_match = match[shape_index]
    uniform_idx = np.flatnonzero(match)
    shape_idx = np.flatnonzero(shape_match)
    counts = {}
    for k in cutoffs:
        if k < 1 or k > len(match):
            raise ValueError('cutoff outside pool')
        counts[k] = (int(np.searchsorted(uniform_idx, k, side='left')),
                     int(np.searchsorted(shape_idx, k, side='left')))
    if len(uniform_idx) != len(shape_idx):
        raise AssertionError('same-pool full-match invariant failed')
    return {
        'counts': counts,
        'full_pool_hits': len(uniform_idx),
        'uniform_first': int(uniform_idx[0]) + 1 if len(uniform_idx) else None,
        'shape_first': int(shape_idx[0]) + 1 if len(shape_idx) else None,
    }


def _holm_pvalues(pvalues: list[float | None]) -> list[float | None]:
    """Holm familywise p adjustment across predeclared cutoffs."""
    finite = sorted((float(p), i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * len(pvalues)
    last = 0.0
    m = len(finite)
    for j, (p, i) in enumerate(finite):
        last = max(last, min(1.0, (m - j) * p))
        out[i] = last
    return out


def _bootstrap_ci(d: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    """Bootstrap paired pool differences in bounded memory (≤64×N indices)."""
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 64):
        n = min(64, reps - start)
        index = rng.integers(0, len(d), size=(n, len(d)), dtype=np.int32)
        means[start:start + n] = d[index].mean(axis=1)
    lo, hi = np.percentile(means, (2.5, 97.5))
    return [float(lo), float(hi)]


def _summarize_cutoff(k: int, trials: int, fair_target_count: int,
                      uniform: np.ndarray, shape: np.ndarray,
                      boot_rng: np.random.Generator, bootstrap_reps: int) -> dict:
    """Paired seed-level CIs: simulation uncertainty conditional on one archive."""
    u = uniform.astype(np.int64)
    s = shape.astype(np.int64)
    d = s - u
    wins_s = int(np.count_nonzero(d > 0))
    wins_u = int(np.count_nonzero(d < 0))
    ties = trials - wins_s - wins_u
    ci = _bootstrap_ci(d, boot_rng, bootstrap_reps)
    # A paired t-test tests the average per-replicate *early-event* delta.
    # If all differences are identically zero, no evidence of a difference.
    if np.all(d == 0):
        pvalue = 1.0
    elif np.all(d == d[0]):
        pvalue = 0.0  # Constant nonzero paired effect has zero estimated SE.
    else:
        pvalue = float(ttest_1samp(d, 0.0).pvalue)
    p_sign = float(binomtest(wins_s, wins_s + wins_u, 0.5).pvalue) if wins_s + wins_u else 1.0
    return {
        'cutoff': k, 'trials': trials,
        'total_guesses_per_strategy': trials * k,
        'iid_expected_historical_hit_events': trials * k * fair_target_count / UNIVERSE,
        'uniform_hit_events': int(u.sum()),
        'shape_hit_events': int(s.sum()),
        'event_delta_shape_minus_uniform': int(d.sum()),
        'mean_event_delta_per_pool': float(d.mean()),
        'paired_bootstrap_95ci_delta_per_pool': ci,
        'paired_ttest_p_exploratory': pvalue,
        'paired_sign_test_p_exploratory': p_sign,
        'shape_better_pools': wins_s,
        'uniform_better_pools': wins_u,
        'equal_pools': ties,
        'uniform_any_hit_pools': int(np.count_nonzero(u)),
        'shape_any_hit_pools': int(np.count_nonzero(s)),
        'uniform_hits_per_guess': float(u.sum() / (trials * k)),
        'shape_hits_per_guess': float(s.sum() / (trials * k)),
        'limited_power': trials * k * fair_target_count / UNIVERSE < 30,
    }


def _paths(out: Path) -> dict[str, Path]:
    p = Path(out)
    if p.suffix.lower() != '.json':
        raise ValueError('--out must end with .json')
    base = p.with_suffix('')
    return {'summary': p,
            'trials': Path(str(base) + '_trials.csv'),
            'cutoffs': Path(str(base) + '_cutoffs.csv')}


def run_backtest(*, catalog: Path, universe: Path, out: Path,
                 trials: int = 1000, pool_size: int = 10000,
                 cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
                 primary_cutoff: int = 5000, holdout_count: int = 500,
                 tau: float = 384.0, seed: int | None = None,
                 bootstrap_reps: int = 2000,
                 progress_every: int = 0,
                 emit: Callable[[str], None] = print) -> dict:
    if (not isinstance(trials, int) or isinstance(trials, bool)
            or not 2 <= trials <= 100000):
        raise ValueError('trials must be in 2..100000')
    if (not isinstance(pool_size, int) or isinstance(pool_size, bool)
            or not 10 <= pool_size <= 65536):
        raise ValueError('pool_size must be in 10..65536')
    ks = tuple(int(k) for k in cutoffs)
    if (not ks or any(k < 1 or k > pool_size for k in ks)
            or len(set(ks)) != len(ks) or list(ks) != sorted(ks)):
        raise ValueError('cutoffs must be distinct increasing integers within pool')
    if primary_cutoff not in ks:
        raise ValueError('primary_cutoff must be among cutoffs')
    if (not isinstance(bootstrap_reps, int) or not 100 <= bootstrap_reps <= 20000
            or not isinstance(progress_every, int) or progress_every < 0
            or seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0)):
        raise ValueError('invalid bootstrap count, progress interval or root seed')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite positive')
    paths = _paths(out)
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('output exists; choose a fresh --out filename')
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    ids, numbers, tables, _ = read_sources(catalog, universe)
    ref = build_reference(ids, numbers, tables, target='holdout',
                          holdout_count=holdout_count, tau=tau)
    target_codes = np.asarray(sorted(ref['lookup']), dtype=np.uint64)
    if not len(target_codes) or ref['all_history_leakage']:
        raise AssertionError('strict early-train / later-target split failed')
    actual_seed = secrets.randbits(128) if seed is None else seed
    seed_root = np.random.SeedSequence(actual_seed)
    child_seeds = seed_root.spawn(trials)
    ccounts = np.zeros((trials, len(ks), 2), dtype=np.int32)
    uniform_first: list[int | None] = []
    shape_first: list[int | None] = []
    total_full_hits = 0
    start = time.perf_counter()
    headers = ['trial', 'seed_spawn_index', 'full_pool_hits', 'uniform_first', 'shape_first']
    for k in ks:
        headers.extend((f'uniform_at_{k}', f'shape_at_{k}'))

    # Each seed creates a new independent pool. The paired strategies share
    # precisely those same candidates (but inspect them in different orders).
    with paths['trials'].open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i, ss in enumerate(child_seeds):
            pool = make_fast_batch(np.random.default_rng(ss), pool_size)
            paired = evaluate_paired_pool(pool, ref, target_codes, ks)
            total_full_hits += paired['full_pool_hits']
            uniform_first.append(paired['uniform_first'])
            shape_first.append(paired['shape_first'])
            rec = {'trial': i + 1, 'seed_spawn_index': i,
                   'full_pool_hits': paired['full_pool_hits'],
                   'uniform_first': paired['uniform_first'] or '',
                   'shape_first': paired['shape_first'] or ''}
            for j, k in enumerate(ks):
                a, b = paired['counts'][k]
                ccounts[i, j, :] = (a, b)
                rec[f'uniform_at_{k}'] = a
                rec[f'shape_at_{k}'] = b
            writer.writerow(rec)
            if progress_every and (i + 1) % progress_every == 0:
                emit(f'[PROGRESS] seeds={i+1:,}/{trials:,} '
                     f'elapsed={time.perf_counter()-start:.1f}s')
        f.flush()

    # Separate PRNG: resampling must never influence candidate generation.
    bootstrap_seed = np.random.SeedSequence(actual_seed, spawn_key=(2**31 - 1,))
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    stats = []
    for j, k in enumerate(ks):
        stats.append(_summarize_cutoff(k, trials, len(target_codes),
                                     ccounts[:, j, 0], ccounts[:, j, 1],
                                     bootstrap_rng, bootstrap_reps))
    holm = _holm_pvalues([r['paired_ttest_p_exploratory'] for r in stats])
    for row, p in zip(stats, holm):
        row['holm_adjusted_p_all_cutoffs_exploratory'] = p
    cut_headers = list(stats[0])
    with paths['cutoffs'].open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cut_headers)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

    successful_pools = sum(x is not None for x in uniform_first)
    compare_first = {
        'nonempty_pool_count': successful_pools,
        'uniform_first_earlier': int(sum(a < b for a, b in zip(uniform_first, shape_first)
                                          if a is not None and b is not None)),
        'shape_first_earlier': int(sum(b < a for a, b in zip(uniform_first, shape_first)
                                        if a is not None and b is not None)),
        'first_equal': int(sum(a == b for a, b in zip(uniform_first, shape_first)
                               if a is not None and b is not None)),
        'mean_uniform_first_rank_given_any_full_pool_match':
            (float(np.mean([v for v in uniform_first if v is not None])) if successful_pools else None),
        'mean_shape_first_rank_given_any_full_pool_match':
            (float(np.mean([v for v in shape_first if v is not None])) if successful_pools else None),
    }
    elapsed = time.perf_counter() - start
    report = {
        'schema': SCHEMA,
        'settings': {
            'trials': trials, 'pool_size': pool_size, 'cutoffs': list(ks),
            'primary_cutoff': primary_cutoff, 'holdout_count': holdout_count,
            'tau': tau, 'seed': actual_seed,
            'seed_source': 'OS_CSPRNG' if seed is None else 'explicit',
            'seed_construction': 'NumPy SeedSequence(root).spawn(trials), spawn index in trials.csv',
            'bootstrap_replicates': bootstrap_reps,
            'candidate_sampling': 'IID uniform 6-of-49 with replacement; repeat tickets permitted',
            'shape_ranking': 'train-only L3 Dirichlet q/p; stable descending order',
        },
        'data': {
            'catalog_sha256': file_sha256(catalog),
            'universe_sha256': file_sha256(universe),
            'historical_draw_count': len(ids),
            'training_count': ref['training_draws'],
            'training_last_draw_id': int(ids[-holdout_count - 1]),
            'target_count': ref['target_count'],
            'target_unique_six_sets': len(target_codes),
            'target_first_draw_id': int(ids[-holdout_count]),
            'target_last_draw_id': int(ids[-1]),
            'target_was_used_in_model_fit': False,
        },
        'results': {
            'elapsed_seconds': round(elapsed, 3),
            'total_candidate_guesses': trials * pool_size,
            'total_full_pool_historical_matches_per_strategy': total_full_hits,
            'full_pool_match_count_invariant': True,
            'comparisons_by_cutoff': stats,
            'primary_comparison': next(row for row in stats if row['cutoff'] == primary_cutoff),
            'first_hit_full_pool': compare_first,
        },
        'cautions': [
            'Only the first 1997 draws are used to fit shape classes when holdout_count=500.',
            'The final 500 draws are a retrospective holdout and have been inspected in earlier project research; this is not a new blind prospective evaluation.',
            'Paired bootstrap intervals cover Monte Carlo seed/pool variation CONDITIONAL on this fixed holdout, not uncertainty about genuinely new lottery draws.',
            'Exploratory paired t-tests and Holm adjustments do not establish causality or a fair-lottery prediction advantage.',
            'Early smaller cutoffs are underpowered when IID expected matches <30; treat any single early hit cautiously.',
            'Shape scorer ranks candidates; it does not change the complete candidate pool, and the full-pool hit counts are identical by construction.',
            'Fair 6/49 jackpot probability remains 1 / 13,983,816 for every legal six-number combination.',
        ],
    }
    _write_summary(paths['summary'], report)
    primary = report['results']['primary_comparison']
    emit(f'[DONE] v0.23: {trials:,} seeds × {pool_size:,} candidates = '
         f'{trials*pool_size:,} guesses per strategy; '
         f'holdout={ref["target_count"]:,}; elapsed={elapsed:.1f}s')
    emit(f'[PRIMARY K={primary_cutoff:,}] uniform={primary["uniform_hit_events"]:,}, '
         f'shape={primary["shape_hit_events"]:,}, '
         f'delta={primary["event_delta_shape_minus_uniform"]:+,}, '
         f'Holm p={primary["holm_adjusted_p_all_cutoffs_exploratory"]:.4g}')
    emit(f'[SAVED] {paths["summary"]} | {paths["trials"]} | {paths["cutoffs"]}')
    return report
