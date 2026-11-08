"""v0.20: auditable pre-draw *class* forecasts and post-draw reconciliation.

Only a SHA256-sealed snapshot committed publicly *before* the next draw can
provide a documented chronology. These are full categorical probabilities,
not a six-number prediction and not authentication of lottery source records.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from math import exp, log, log1p
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .shape_prospective_v019 import (
    REPORT_SCHEMA, check_freeze, content_hash, file_sha256,
    prospective_evaluate, seal_object, verify_seal,
)
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import LEVELS, classification_keys, category_name

SCHEMA = 'lotto7x7.pre_draw_class_commitment.v0.20'
AUDIT_SCHEMA = 'lotto7x7.pre_draw_class_audit.v0.20'
TAIPEI = ZoneInfo('Asia/Taipei')
DRAW_TIME = time(hour=20, minute=30)


def _events_from_report(report: dict) -> list[dict]:
    events = []
    for item in report.get('events', []):
        events.append({
            'draw_id': int(item['draw_id']),
            'date': str(item['date']),
            'numbers': [int(item[f'n{k}']) for k in range(1, 7)],
        })
    return events


def _check_report(freeze: dict, tables: dict, report: dict) -> list[dict]:
    if (report.get('schema') != REPORT_SCHEMA or
            report.get('freeze_sha256') != freeze.get('self_checksum_sha256')):
        raise ValueError('v0.19 report uses a different freeze or schema')
    if report.get('eligible_new_draws') != len(report.get('events', [])):
        raise ValueError('v0.19 event count is inconsistent')
    if report.get('last_frozen_historical_draw_id') != freeze['source']['last_draw_id']:
        raise ValueError('historical draw boundary mismatch')
    events = _events_from_report(report)
    # Rebuild every reported pre-draw probability and evidence value. A report
    # containing a post-hoc revision to any event is not an admissible input.
    rebuilt, _ = prospective_evaluate(freeze, tables, events, [], previous=report)
    if rebuilt['eligible_new_draws'] != len(events):
        raise AssertionError('re-evaluated event count mismatch')
    return events


def _check_target(freeze: dict, events: list[dict], target_id: int,
                  target_day: date, now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    if target_day <= date.fromisoformat(freeze['freeze_local_date_taipei']):
        raise ValueError('target date must follow original v0.19 freeze')
    if target_day.weekday() not in (1, 4):
        raise ValueError('only standard Tuesday/Friday draw dates accepted; exceptional draws require a separately reviewed protocol')
    if target_day.year - 1911 != target_id // 1_000_000:
        raise ValueError('issue ID ROC year differs from Gregorian target date')
    cutoff_id = int(events[-1]['draw_id']) if events else int(freeze['source']['last_draw_id'])
    yr, issue = divmod(cutoff_id, 1_000_000)
    expected_id = cutoff_id + 1 if target_id // 1_000_000 == yr else (yr + 1) * 1_000_000 + 1
    if target_id != expected_id:
        raise ValueError(f'next target must be contiguous issue {expected_id}; received {target_id}')
    if events and target_day <= date.fromisoformat(events[-1]['date']):
        raise ValueError('target must be strictly later than last scored event date')
    deadline = datetime.combine(target_day, DRAW_TIME, tzinfo=TAIPEI)
    if now.astimezone(TAIPEI) >= deadline:
        raise ValueError('too late to create a genuine pre-draw commitment (20:30 Asia/Taipei cutoff)')
    return deadline


def make_commitment(freeze: dict, tables: dict, report: dict, *,
                    issue_id: int, issue_date: str, now: datetime | None = None,
                    freeze_path: Path | None = None,
                    report_path: Path | None = None) -> dict:
    """Create a sealed class distribution, with all pre-draw info auditable."""
    verify_seal(freeze)
    events = _check_report(freeze, tables, report)
    target = date.fromisoformat(issue_date)
    current = now or datetime.now(timezone.utc)
    deadline = _check_target(freeze, events, int(issue_id), target, current)
    n0 = int(freeze['source']['historical_draw_count'])
    tau = float(freeze['frozen_rule']['tau'])
    distributions: dict[str, list[dict]] = {}
    for lev in LEVELS:
        codes = tables[lev]['codes']
        counts = np.array([int(freeze['frozen_rule']['historical_counts'][lev][str(int(code))])
                           for code in codes], dtype=np.int64)
        if events:
            nums = np.asarray([evt['numbers'] for evt in events], dtype=np.int64)
            new_codes = classification_keys(classify_numbers(nums))[lev]
            ix = np.searchsorted(codes, new_codes)
            if np.any(ix >= len(codes)) or not np.array_equal(codes[ix], new_codes):
                raise ValueError('a previously scored event has an unknown category')
            counts += np.bincount(ix, minlength=len(codes))
        p = np.asarray(tables[lev]['p'], dtype=np.float64)
        q = (counts + tau * p) / (n0 + len(events) + tau)
        if abs(float(q.sum()) - 1.0) > 1e-12:
            raise AssertionError('pre-draw forecast is not normalized')
        distributions[lev] = [
            {'class': category_name(lev, int(code)), 'code': int(code),
             'p_fair': float(p[k]), 'q_before_draw': float(q[k]),
             'relative_ratio': float(q[k]/p[k])}
            for k, code in enumerate(codes)
        ]
    p_rare = float(freeze['tests']['rare_fair_probability'])
    q_rare = float(freeze['tests']['rare_alternative_probability'])
    # The event list is cryptographically bound to this local snapshot;
    # publication timing requires an independently timestamped Git commit.
    result = {
        'schema': SCHEMA,
        'purpose': 'Prospective categorical forecast; NOT predicted lottery numbers',
        'created_at_utc': current.astimezone(timezone.utc).isoformat(timespec='seconds'),
        'target': {'draw_id': int(issue_id), 'date': target.isoformat(),
                   'scheduled_draw_time_taipei': deadline.isoformat()},
        'freeze_sha256': freeze['self_checksum_sha256'],
        'freeze_file_sha256': file_sha256(freeze_path) if freeze_path else None,
        'previous_report_sha256': file_sha256(report_path) if report_path else None,
        'previous_events_count': len(events),
        'previous_events_sha256': content_hash({'events': report['events']}),
        'last_scored_draw_id': (events[-1]['draw_id'] if events else freeze['source']['last_draw_id']),
        'source_csv_manifest': report.get('source_csv_files', []),
        'history_model': {'tau': tau, 'training_draws_through_previous': n0 + len(events),
                          'universe': int(freeze['frozen_rule']['fair_universe']),
                          'fixed_rule_sha256': freeze['frozen_rule']['sha256']},
        'distributions': distributions,
        'rare_class': {'name': freeze['tests']['rare_class'],
                       'p_fair': p_rare, 'q_alternative': q_rare},
        'warning': ('This document has only a local hash seal. A Git commit pushed '
                    'before the drawing is needed to anchor chronology. The input '
                    'CSV report does not independently authenticate official results.')
    }
    return seal_object(result)


def audit_commitment(commitment: dict, freeze: dict, tables: dict,
                     report: dict) -> dict:
    """Verify a published pre-draw class forecast against its scored draw.

    Official source authenticity and Git commit timestamp are **not** checked
    by this offline function and are reported explicitly as unverified.
    """
    verify_seal(commitment)
    verify_seal(freeze)
    if commitment.get('schema') != SCHEMA:
        raise ValueError('unknown v0.20 commitment schema')
    if commitment.get('freeze_sha256') != freeze['self_checksum_sha256']:
        raise ValueError('commitment uses a different v0.19 freeze')
    events = _check_report(freeze, tables, report)
    count = int(commitment['previous_events_count'])
    if len(events) < count:
        raise ValueError('updated prospective report dropped earlier events')
    if content_hash({'events': report['events'][:count]}) != commitment['previous_events_sha256']:
        raise ValueError('earlier prospective events differ from the sealed pre-draw state')
    target_id = int(commitment['target']['draw_id'])
    target_date = str(commitment['target']['date'])
    # Independently recalculate the WHOLE pre-draw class distribution from
    # the prefix, not merely the class that happened to occur.
    prefix, _ = prospective_evaluate(freeze, tables, events[:count], [])
    recreated = make_commitment(
        freeze, tables, prefix, issue_id=target_id, issue_date=target_date,
        now=datetime.fromisoformat(commitment['created_at_utc']))
    for field in ('target', 'history_model', 'distributions', 'rare_class',
                  'previous_events_count', 'previous_events_sha256', 'last_scored_draw_id'):
        if content_hash({'value': recreated[field]}) != content_hash({'value': commitment[field]}):
            raise ValueError(f'pre-draw sealed field {field} is not reproducible')
    if count < len(events):
        nxt = events[count]
        if int(nxt['draw_id']) != target_id or nxt['date'] != target_date:
            raise ValueError('next prospective draw does not match sealed target')
        row = report['events'][count]
        for lev in LEVELS:
            by_class = {c['class']: c for c in commitment['distributions'][lev]}
            actual = row[f'{lev}_class']
            if actual not in by_class:
                raise ValueError(f'{lev}: actual class absent from sealed forecast')
            prior = by_class[actual]
            for key, expected in (
                (f'{lev}_q_before_draw', prior['q_before_draw']),
                (f'{lev}_fair_p', prior['p_fair']),
                (f'{lev}_log_gain', log(prior['q_before_draw']/prior['p_fair'])),
            ):
                if not np.isclose(float(row[key]), float(expected), rtol=1e-12, atol=1e-12):
                    raise ValueError(f'prospective {key} does not match the earlier commitment')
        if int(row['rare_class_hit']) == 1:
            expected_rare = log(commitment['rare_class']['q_alternative'] /
                                commitment['rare_class']['p_fair'])
        else:
            expected_rare = log1p(-commitment['rare_class']['q_alternative']) - log1p(-commitment['rare_class']['p_fair'])
        if not np.isclose(float(row['rare_log_gain']), expected_rare, rtol=1e-12, atol=1e-12):
            raise ValueError('rare-class log-gain mismatch')
        outcome = 'AUDIT_MATCH'
    else:
        if report.get('eligible_new_draws') != count:
            raise ValueError('prospective sample count mismatch')
        outcome = 'AWAITING_DRAW'
    return {
        'schema': AUDIT_SCHEMA,
        'decision': outcome,
        'target': commitment['target'],
        'commitment_sha256': commitment['self_checksum_sha256'],
        'freeze_sha256': freeze['self_checksum_sha256'],
        'verified_class_forecast_arithmetic': outcome == 'AUDIT_MATCH',
        'evidence_timeline_count_after': len(events),
        'source_authenticity_verified': False,
        'external_git_timestamp_verified': False,
        'caution': ('Only v0.19 scoring arithmetic and sealed forecast consistency '
                    'were checked. Confirm official six main numbers and pre-draw '
                    'Git publication timestamp independently before treating this '
                    'as a prospective audit.'),
    }
