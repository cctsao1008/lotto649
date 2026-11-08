"""v0.23 tests: no leakage, paired ordering, correct statistical denominators."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lotto7x7.early_hit_v023 import (
    _bootstrap_ci, _holm_pvalues, _paths, _summarize_cutoff,
    evaluate_paired_pool, lookup_matches, ranking_from_training, run_backtest,
)
from lotto7x7.historical_guess_v022 import build_reference, encode_tickets
from lotto7x7.historical_guess_stream_v0222 import make_fast_batch


def fixture_data(monkeypatch):
    import lotto7x7.early_hit_v023 as mod
    from lotto7x7.shape_predictive_v017 import _baseline_tables
    baseline = json.loads((Path(__file__).parent / 'fixtures' / 'p1_exact_universe_v1.json').read_text())
    tables = _baseline_tables(baseline)
    draws = make_fast_batch(np.random.default_rng(17), 32)
    ids = np.arange(93000001, 93000033, dtype=np.int64)
    monkeypatch.setattr(mod, 'read_sources', lambda a, b: (ids, draws, tables, {}))
    monkeypatch.setattr(mod, 'file_sha256', lambda _: 'b' * 64)
    return mod, ids, draws, tables


def test_exact_six_match_boolean_and_duplicate_trials():
    pool = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7],
                     [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
    codes = np.asarray([encode_tickets(pool[:1])[0]], dtype=np.uint64)
    assert lookup_matches(pool, codes).tolist() == [True, False, True]


def test_shape_order_uses_training_only_and_full_pool_invariant(monkeypatch):
    mod, ids, draws, tables = fixture_data(monkeypatch)
    ref = build_reference(ids, draws, tables, target='holdout', holdout_count=8)
    assert ref['training_draws'] == 24
    assert not ref['all_history_leakage']
    pool = np.vstack((draws[-8:], draws[:16]))
    result = evaluate_paired_pool(pool, ref, np.array(sorted(ref['lookup']), np.uint64),
                                  (1, 5, 10, 24))
    assert result['full_pool_hits'] == 8
    assert result['counts'][24] == (8, 8)
    assert result['uniform_first'] == 1
    assert 1 <= result['shape_first'] <= 24
    order = ranking_from_training(pool, ref)
    assert sorted(order.tolist()) == list(range(24))


def test_holm_multicomparison_adjustment():
    a = _holm_pvalues([.04, .03, .9, None])
    assert a[0] == pytest.approx(.09) and a[1] == pytest.approx(.09)
    assert a[2] == pytest.approx(.9) and a[3] is None


def test_paired_statistics_zero_difference():
    x = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    result = _summarize_cutoff(5, 5, 8, x, x, np.random.default_rng(123), 100)
    assert result['event_delta_shape_minus_uniform'] == 0
    assert result['paired_ttest_p_exploratory'] == 1
    assert result['paired_sign_test_p_exploratory'] == 1
    assert result['paired_bootstrap_95ci_delta_per_pool'] == [0.0, 0.0]


def test_bootstrap_is_bounded_and_stable():
    x = np.array([0, 1, -1, 2, -2] * 50, dtype=np.int32)
    a = _bootstrap_ci(x, np.random.default_rng(42), 129)
    b = _bootstrap_ci(x, np.random.default_rng(42), 129)
    assert a == b and a[0] <= 0 <= a[1]


def test_full_backtest_outputs_and_train_split(monkeypatch, tmp_path):
    mod, ids, draws, _ = fixture_data(monkeypatch)
    # Monkeypatch candidate draws to force a visible difference and ensure
    # we can test the full report even with a 32-row synthetic archive.
    monkeypatch.setattr(mod, 'make_fast_batch', lambda rng, n: draws[:n].copy())
    out = tmp_path / 'trial.json'
    messages = []
    a = mod.run_backtest(catalog=tmp_path/'input.npz', universe=tmp_path/'univ.json',
                         out=out, trials=4, pool_size=12, cutoffs=(2, 5, 12),
                         primary_cutoff=5, holdout_count=8, seed=123,
                         bootstrap_reps=100, emit=messages.append)
    assert a['settings']['seed'] == 123
    assert a['settings']['seed_source'] == 'explicit'
    assert a['data']['training_count'] == 24
    assert a['data']['target_count'] == 8
    assert a['data']['target_was_used_in_model_fit'] is False
    assert a['results']['total_candidate_guesses'] == 48
    assert a['results']['full_pool_match_count_invariant']
    assert _paths(out)['summary'].is_file()
    assert _paths(out)['cutoffs'].is_file()
    assert _paths(out)['trials'].is_file()
    with _paths(out)['trials'].open(encoding='utf8', newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == 4 and records[0]['seed_spawn_index'] == '0'
    assert all(int(x['full_pool_hits']) == 0 for x in records)
    assert a['results']['primary_comparison']['uniform_hit_events'] == 0
    assert messages[-1].startswith('[SAVED]')
    with pytest.raises(FileExistsError):
        mod.run_backtest(catalog=tmp_path/'input.npz', universe=tmp_path/'univ.json',
                         out=out, trials=4, pool_size=12, cutoffs=(5,),
                         primary_cutoff=5, holdout_count=8, bootstrap_reps=100)


def test_forced_matching_full_backtest(monkeypatch, tmp_path):
    mod, ids, draws, _ = fixture_data(monkeypatch)
    monkeypatch.setattr(mod, 'make_fast_batch', lambda rng, n: draws[-n:].copy())
    rep = mod.run_backtest(catalog=tmp_path/'input.npz', universe=tmp_path/'univ.json',
                           out=tmp_path/'hits.json', trials=5, pool_size=10,
                           cutoffs=(2, 5, 10), primary_cutoff=5,
                           holdout_count=8, seed=42, bootstrap_reps=100, emit=lambda _: None)
    assert rep['results']['total_full_pool_historical_matches_per_strategy'] == 40
    assert rep['results']['comparisons_by_cutoff'][-1]['uniform_hit_events'] == 40
    assert rep['results']['comparisons_by_cutoff'][-1]['shape_hit_events'] == 40


@pytest.mark.parametrize('kw', [
    {'trials': 1},
    {'pool_size': 9},
    {'cutoffs': (10, 5)},
    {'cutoffs': (5, 5)},
    {'cutoffs': (100,)},
    {'cutoffs': (2,), 'primary_cutoff': 5},
    {'bootstrap_reps': 99},
    {'seed': -1},
    {'seed': True},
    {'progress_every': -1},
    {'tau': -1.},
])
def test_invalid_input_fails_before_read_sources(monkeypatch, tmp_path, kw):
    mod, *_ = fixture_data(monkeypatch)
    args = dict(catalog=tmp_path/'cat.npz', universe=tmp_path/'u.json',
                out=tmp_path/'trial.json', trials=2, pool_size=10,
                cutoffs=(2, 5), primary_cutoff=5, holdout_count=8,
                bootstrap_reps=100, seed=1)
    args.update(kw)
    with pytest.raises(ValueError):
        mod.run_backtest(**args)


def test_random_seed_generation_is_recorded(monkeypatch, tmp_path):
    mod, *_ = fixture_data(monkeypatch)
    monkeypatch.setattr(mod.secrets, 'randbits', lambda _: 9876543212345)
    rep = mod.run_backtest(catalog=tmp_path/'cat.npz', universe=tmp_path/'u.json',
                           out=tmp_path/'random.json', trials=2, pool_size=10,
                           cutoffs=(2, 5), primary_cutoff=5, holdout_count=8,
                           bootstrap_reps=100, emit=lambda _: None)
    assert rep['settings']['seed'] == 9876543212345
    assert rep['settings']['seed_source'] == 'OS_CSPRNG'
