"""v0.8.1 frozen next-number forecasts: scored JSON-only paired audit.

Never fits a model or modifies the source report. The main analysis unit is
one chronological target draw, even where training blocks are reused.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from math import comb

import numpy as np
from scipy.stats import binomtest

SCHEMA = 'lotto7x7.probability_audit.v0.8.1'
SOURCE_SCHEMA = 'lotto7x7.probability_replay.v0.8'
BASE = 'iid-uniform'
METHODS = ('iid-uniform', 'prefix-frequency', 'recent-ewma', 'image-cnn-gru')
P3 = sum(comb(6, k)*comb(43, 6-k) for k in range(3, 7)) / comb(49, 6)


def _holm(raw: dict[str, float]):
    ranked = sorted(raw, key=lambda name: (raw[name], name))
    out = {}
    running = 0.0
    for i, method in enumerate(ranked):
        running = max(running, (len(ranked)-i)*raw[method])
        out[method] = min(1.0, running)
    return out


def _ci(draw_values, *, seed, reps=2000, unit='draw', block=100):
    """Bootstrap draw units or chronological contiguous blocks of targets.

    Block resampling is a sensitivity interval, not independent confirmation:
    the expanding historical fits and lottery target sets still overlap.
    """
    arr = np.asarray(draw_values, dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.all(np.isfinite(arr)):
        raise ValueError('invalid confidence interval series')
    rng = np.random.default_rng(seed)
    if unit == 'draw':
        items = [arr[i:i+1] for i in range(len(arr))]
    elif unit == 'block':
        items = [arr[i:i+block] for i in range(0, len(arr), block)]
    else:
        raise ValueError('unknown resampling unit')
    mean = np.empty(reps)
    for k in range(reps):
        idx = rng.integers(0, len(items), size=len(items))
        sample = np.concatenate([items[int(i)] for i in idx])
        mean[k] = sample.mean()
    return [float(q) for q in np.quantile(mean, (0.025, 0.975))]


def _as_draw(row):
    if type(row) is not dict:
        raise ValueError('invalid draw record')
    draw = row.get('actual_main_numbers')
    if (not isinstance(draw, list) or len(draw) != 6 or
            any(type(v) is not int or not 1 <= v <= 49 for v in draw) or
            len(set(draw)) != 6):
        raise ValueError('invalid six-number actual draw')
    label = np.zeros(49, dtype=np.float64)
    label[np.asarray(draw)-1] = 1
    pred = row.get('predictions')
    if not isinstance(pred, dict) or set(pred) != set(METHODS):
        raise ValueError('incomplete or unexpected model predictions')
    return label, pred


def audit(report: dict, *, seed=649, allow_partial=False) -> dict:
    """Recompute 49-number scores, all 4 Top-6 hit series and paired inference.

    By design a standalone JSON audit cannot prove raw-BIN history integrity
    or retraining causality; these are already separately tested by the repo.
    """
    if (type(seed) is not int or seed < 0 or
            not isinstance(report, dict) or report.get('schema') != SOURCE_SCHEMA):
        raise ValueError('expected a v0.8 probability report and nonnegative seed')
    rows = report.get('targets')
    config = report.get('config')
    if not isinstance(rows, list) or not rows or not isinstance(config, dict):
        raise ValueError('missing targets/config')
    if (report.get('evaluated_draws') != len(rows) or
            (not allow_partial and (config.get('tail_targets') is not None or
                len(rows) != report.get('source_draws', 0)-config.get('warmup', -1)))):
        raise ValueError('full chronological history required; use --allow-partial for smoke runs')
    arrays = {m: {'brier': [], 'loss': [], 'hits': [], 'ge3': []} for m in METHODS}
    ids = []
    prev_id = None
    for t, item in enumerate(rows):
        draw_id, cutoff = item.get('target_draw_id'), item.get('cutoff_draw_id')
        if (type(draw_id) is not int or type(cutoff) is not int or
                cutoff >= draw_id or (prev_id is not None and
                (draw_id <= prev_id or cutoff != prev_id))):
            raise ValueError(f'nonchronological target/cutoff at index {t}')
        ids.append(draw_id)
        prev_id = draw_id
        label, forecasts = _as_draw(item)
        for name in METHODS:
            rec = forecasts[name]
            p = np.asarray(rec.get('probabilities_1_to_49'), dtype=np.float64)
            picked = rec.get('top6_numbers')
            if (p.shape != (49,) or not np.all(np.isfinite(p)) or
                    np.min(p) < 0 or np.max(p) > 1 or
                    not np.isclose(p.sum(), 6.0, atol=5e-6, rtol=0) or
                    not isinstance(picked, list) or len(picked) != 6 or
                    any(type(v) is not int or not 1 <= v <= 49 for v in picked) or
                    len(set(picked)) != 6):
                raise ValueError(f'invalid forecast probabilities or ticket in {name} target {draw_id}')
            hit = int(label[np.asarray(picked)-1].sum())
            br = float(np.mean((p-label)**2))
            pp = np.clip(p, 1e-7, 1-1e-7)
            loss = float(-np.mean(label*np.log(pp)+(1-label)*np.log1p(-pp)))
            if (rec.get('top6_hits') != hit or
                    not np.isclose(rec.get('brier', float('nan')), br, atol=2e-7, rtol=0) or
                    not np.isclose(rec.get('binary_log_loss', float('nan')), loss,
                                   atol=2e-7, rtol=0)):
                raise ValueError(f'corrupt saved score for {name} target {draw_id}')
            arrays[name]['brier'].append(br)
            arrays[name]['loss'].append(loss)
            arrays[name]['hits'].append(hit)
            arrays[name]['ge3'].append(int(hit>=3))
    original_summary = report.get('summary', {})
    if set(original_summary) != set(METHODS):
        raise ValueError('missing source summary arms')
    for name in METHODS:
        s = original_summary[name]
        a = arrays[name]
        for key, computed in (('mean_brier', float(np.mean(a['brier']))),
                              ('mean_binary_log_loss', float(np.mean(a['loss']))),
                              ('mean_top6_hits', float(np.mean(a['hits']))),
                              ('top6_ge3_rate', float(np.mean(a['ge3'])))):
            if not np.isclose(s.get(key, float('nan')), computed, atol=2e-7, rtol=0):
                raise ValueError(f'summary score mismatch: {name}/{key}')
    bas = arrays[BASE]
    comparisons = {}
    pvals = {}
    for i, name in enumerate(METHODS[1:]):
        a = arrays[name]
        improved = np.asarray(bas['brier'])-np.asarray(a['brier'])
        loss_improved = np.asarray(bas['loss'])-np.asarray(a['loss'])
        hits = np.asarray(a['hits'])-np.asarray(bas['hits'])
        ge3 = np.asarray(a['ge3'], dtype=int)-np.asarray(bas['ge3'], dtype=int)
        only = int((ge3 == 1).sum())
        baseline_only = int((ge3 == -1).sum())
        discordant = only+baseline_only
        pvalue = float(binomtest(only, discordant, 0.5).pvalue) if discordant else 1.0
        pvals[name] = pvalue
        years = defaultdict(list)
        for j, d in enumerate(ids):
            years[str(d//1_000_000+1911)].append(j)
        comparisons[name] = {
            'mean_brier_skill_vs_iid': float(improved.mean()),
            'mean_log_loss_skill_vs_iid': float(loss_improved.mean()),
            'mean_hits_difference': float(hits.mean()),
            'mean_ge3_difference': float(ge3.mean()),
            'model_ge3_count': int(np.sum(a['ge3'])),
            'iid_ge3_count': int(np.sum(bas['ge3'])),
            'model_only_ge3': only, 'iid_only_ge3': baseline_only,
            'both_ge3': int(np.sum(np.asarray(a['ge3']) & np.asarray(bas['ge3']))),
            'mcnemar_two_sided_exact_p': pvalue,
            'brier_skill_draw_ci95': _ci(improved, seed=seed+i*71, unit='draw'),
            'brier_skill_block_ci95': _ci(improved, seed=seed+i*71+1, unit='block'),
            'ge3_delta_draw_ci95': _ci(ge3, seed=seed+i*71+2, unit='draw'),
            'ge3_delta_block_ci95': _ci(ge3, seed=seed+i*71+3, unit='block'),
            'year': {year: {'draws': len(ix),
                            'model_ge3': int(np.sum(np.asarray(a['ge3'])[ix])),
                            'iid_ge3': int(np.sum(np.asarray(bas['ge3'])[ix])),
                            'ge3_difference':float(np.mean(ge3[ix])),
                            'brier_skill':float(np.mean(improved[ix]))}
                     for year, ix in sorted(years.items())},
        }
    adj = _holm(pvals)
    for name in comparisons:
        comparisons[name]['holm_three_model_adjusted_p'] = adj[name]
    raw_json = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)
    return {
        'schema': SCHEMA, 'research_only':True,
        'source_sha256_canonical_json':hashlib.sha256(raw_json.encode()).hexdigest(),
        'target_count':len(rows), 'cutoff_first':rows[0]['cutoff_draw_id'],
        'target_first':ids[0], 'target_last':ids[-1],
        'seed':seed, 'iid_one_bet_ge3_exact':P3,
        'iid_ge3_count':int(np.sum(bas['ge3'])),
        'comparisons': comparisons,
        'limitations':[
            'Retrospective previously inspected results; NOT prospective evidence.',
            'Selection among already-tried metrics/models is not corrected.',
            'McNemar and draw bootstrap assume draw-level exchangeability; overlapping training histories violate simple independence assumptions.',
            '100-draw contiguous-block bootstrap is a sensitivity analysis, not a formal guarantee.',
            'Paired McNemar tests a threshold outcome; probability quality is judged by proper scores.',
            'This JSON-only audit cannot independently authenticate its originating BIN archive.',
        ],
    }
