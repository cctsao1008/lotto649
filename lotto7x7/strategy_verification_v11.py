"""Independent strategy verification v1.1: exact prize events + frozen M0 audit.

The prize verifier does not call the existing portfolio_prize_v030 dynamic
program. It uses event-intersection counts, binomial inversion, and separate
closed-form per-ticket hypergeometric tier counts. This is a second counting
path, not evidence that any candidate is more likely to be drawn.

M0 verification is a *frozen manifest structural/integrity audit only*.
Original finite-pool rank/optimization scores are NOT independently recomputed
from the full historical 13,983,816-ticket theory; do not claim M0 optimizer PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_FLOOR, localcontext
from hashlib import sha1, sha256
from itertools import combinations
import json
from math import comb, prod
from pathlib import Path

import numpy as np

from .strategy_verification_v1 import verify_selection
from .geometry_convergence_v029 import audit_repository_inputs

SCHEMA = 'lotto649.strategy_verification.v1.1'
M0_SCHEMA = 'lotto7x7.multiobjective.v0.28.m0.candidate_manifest.v1'
M0_GIT_BLOB = '21dff1f975773ca919e6af9e762e93f8ffc3475c'
TIERS = ('jackpot', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'regular')
FIXED_PRIZES = {'fifth': 2000, 'sixth': 1000, 'seventh': 400, 'regular': 400}


def _four_tickets(tickets: object, *, universe: int = 49, draw_size: int = 6) -> tuple[tuple[int, ...], ...]:
    if not isinstance(tickets, (list, tuple)) or len(tickets) != 4:
        raise ValueError('four distinct ticket lines required')
    lines = []
    for line in tickets:
        if not isinstance(line, (list, tuple)) or len(line) != draw_size or any(
            type(x) not in (int, np.int64, np.int32, np.int16) or not 1 <= int(x) <= universe
            for x in line
        ):
            raise ValueError('illegal ticket line')
        item = tuple(int(x) for x in line)
        if tuple(sorted(set(item))) != item:
            raise ValueError('ticket must be sorted and unique')
        lines.append(item)
    if len(set(lines)) != 4:
        raise ValueError('duplicate tickets')
    return tuple(lines)


def _count_main_intersection(group_counts: dict[int, int], subset_size: int,
                             special_mask: int, draw_size: int) -> int:
    """Count main draws for which every ticket in a chosen subset wins.

    Polynomial coefficients are accumulated by group, conditioned on the
    special ball. Unlike the production engine, no 4-ticket win histogram or
    tier grades are accumulated here; each conjunction is counted separately.
    """
    ways: dict[tuple[int, tuple[int, ...]], int] = {(0, (0,) * subset_size): 1}
    for membership, count in sorted(group_counts.items()):
        next_ways: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
        for (used, hits), count_ways in ways.items():
            for chosen in range(min(count, draw_size - used) + 1):
                new_hits = tuple(h + chosen * bool(membership & (1 << i))
                                 for i, h in enumerate(hits))
                next_ways[(used + chosen, new_hits)] += count_ways * comb(count, chosen)
        ways = next_ways
    return sum(weight for (used, hits), weight in ways.items()
               if used == draw_size and all(
                   h >= 3 or (h == 2 and bool(special_mask & (1 << i)))
                   for i, h in enumerate(hits)))


def _grade(hits: int, special: bool) -> str | None:
    if hits == 6:
        return 'jackpot'
    if hits == 5:
        return 'second' if special else 'third'
    if hits == 4:
        return 'fourth' if special else 'fifth'
    if hits == 3:
        return 'sixth' if special else 'regular'
    if hits == 2 and special:
        return 'seventh'
    return None


def independent_exact_prizes(tickets: object, *, universe: int = 49,
                             draw_size: int = 6) -> dict:
    """Exact 0..4 winning lines via intersections and binomial inversion.

    Special ball is selected first: U*C(U-1,k) equally likely pairs.
    This is an alternative to the original all-four-ticket joint-win DP.
    """
    if type(universe) is not int or type(draw_size) is not int or not (3 <= draw_size < universe <= 49):
        raise ValueError('unsupported universe / draw size')
    lines = _four_tickets(tickets, universe=universe, draw_size=draw_size)
    denominator = universe * comb(universe - 1, draw_size)
    membership = Counter(sum(1 << j for j, t in enumerate(lines) if ball in t)
                         for ball in range(1, universe + 1))
    intersections = [0] * 5
    intersections[0] = denominator
    # For k=1..4, accumulate all kth-order win intersections. Each is
    # counted with its own reduced 2^k membership strata and exact choices.
    for k in range(1, 5):
        for subset in combinations(range(4), k):
            reduced = Counter()
            for mask, count in membership.items():
                reduced[sum(1 << j for j, ix in enumerate(subset)
                            if mask & (1 << ix))] += count
            for special_mask, multiplicity in reduced.items():
                remaining = dict(reduced)
                remaining[special_mask] -= 1
                intersections[k] += multiplicity * _count_main_intersection(
                    remaining, k, special_mask, draw_size)
    distribution = [sum((-1) ** (k - j) * comb(k, j) * intersections[k]
                        for k in range(j, 5)) for j in range(5)]
    if min(distribution) < 0 or sum(distribution) != denominator:
        raise AssertionError('nonphysical factorial-moment inversion')

    tiers = Counter()
    if draw_size == 6:
        # Independent analytic hypergeometric counting for one fixed ticket;
        # sum over four tickets. Multiplicities are integer (main,special) pairs.
        for hits in range(7):
            primary = comb(draw_size, hits) * comb(universe - draw_size, draw_size - hits)
            bonus_choices = draw_size - hits
            for is_bonus, multiplicity in ((True, bonus_choices),
                                           (False, universe - draw_size - bonus_choices)):
                name = _grade(hits, is_bonus)
                if name:
                    tiers[name] += 4 * primary * multiplicity
        if tiers['jackpot'] != 4 * (universe - draw_size):
            raise AssertionError('jackpot multiplicity inconsistent')
        if sum(i * count for i, count in enumerate(distribution)) != sum(tiers.values()):
            raise AssertionError('independent tier and winning-line counts differ')
    return {
        'total_equally_likely_outcomes': denominator,
        'exact_counts_winning_lines_0_to_4': distribution,
        'probabilities_winning_lines_0_to_4': [c / denominator for c in distribution],
        'probability_any_prize': sum(distribution[1:]) / denominator,
        'expected_number_winning_lines': sum(i * c for i, c in enumerate(distribution)) / denominator,
        'expected_ticket_counts_by_tier': {t: tiers[t] / denominator for t in TIERS}
        if draw_size == 6 else {},
        'expected_nominal_fixed_lower_tier_payout_ntd': (
            sum(tiers[t] * value for t, value in FIXED_PRIZES.items()) / denominator
        ) if draw_size == 6 else None,
        'fixed_tier_payout_assumption_ntd': FIXED_PRIZES if draw_size == 6 else None,
        'method': 'factorial_moment_inversion_and_single_ticket_hypergeometric_tiers',
        'limitations': 'Fair-draw prize structure only; variable prizes and actual payout excluded',
    }


def _blob_sha(data: bytes) -> str:
    return sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest()


def _m0_partition(line: tuple[int, ...]) -> str:
    """Independent 8-neighbor grid graph; no original taxonomy imported."""
    points = {((v - 1) // 7, (v - 1) % 7) for v in line}
    unseen = points.copy()
    sizes = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        size = 1
        while stack:
            a, b = stack.pop()
            neighbours = {p for p in unseen if max(abs(a - p[0]), abs(b - p[1])) == 1}
            unseen.difference_update(neighbours)
            size += len(neighbours)
            stack.extend(neighbours)
        sizes.append(size)
    return 'P' + '-'.join(map(str, sorted(sizes, reverse=True)))


def _m0_exact_log_bin(line: tuple[int, ...]) -> int:
    with localcontext() as context:
        context.prec = 80
        w = Decimal('0.5')
        minimum = Decimal(prod(range(1, 7))).ln()
        start = int((minimum / w).to_integral_value(rounding=ROUND_FLOOR))
        actual = int((Decimal(prod(line)).ln() / w).to_integral_value(rounding=ROUND_FLOOR))
        return actual - start


def verify_m0_frozen_manifest(payload: bytes) -> dict:
    """Check recorded historic identity and ticket constraints, *not* optimality.

    Frozen SHA1 binds all original fields including historical score claims.
    Those claims remain archived, but cannot be independently re-derived by
    this lightweight manifest checker without original frozen theory/trace.
    """
    checks: dict[str, bool] = {}
    def check(name: str, ok: object) -> None:
        checks[name] = bool(ok)
    normalized = payload.replace(b'\r\n', b'\n')
    check('immutable_manifest_blob', _blob_sha(normalized) == M0_GIT_BLOB)
    try:
        obj = json.loads(normalized)
        check('schema_and_status', obj.get('schema') == M0_SCHEMA and
              obj.get('status') == 'DESCRIPTIVE_CANDIDATE_ONLY_NOT_A_PAID_TICKET')
        cutoff = obj.get('data_cutoff') or {}
        check('original_historical_cutoff', cutoff.get('date') == '2026-11-06' and
              cutoff.get('draw_id') == 115000103 and cutoff.get('draws') == 2497 and
              cutoff.get('historical_prefix_sha256') ==
              'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8')
        method = obj.get('method') or {}
        check('fixed_candidate_search_contract', method.get('seed') == 20261110 and
              method.get('uniform_candidate_sample_size') == 80000 and
              method.get('search_pool_size') == 2000 and method.get('restarts') == 40 and
              method.get('search_certificate') == 'HEURISTIC_UNCERTIFIED')
        entries = obj['primary_candidate_tickets']
        lines = _four_tickets([e['numbers'] for e in entries])
        check('four_disjoint_tickets', len(set(v for t in lines for v in t)) == 24)
        check('frozen_bet_order', [e['bet'] for e in entries] == list('ABCD'))
        for i, (entry, line) in enumerate(zip(entries, lines)):
            check(f'bet_{i + 1}_sum', entry.get('sum') == sum(line))
            check(f'bet_{i + 1}_log_bin', entry.get('log_bin') == _m0_exact_log_bin(line))
            check(f'bet_{i + 1}_shape_partition', entry.get('shape_partition') == _m0_partition(line))
            recorded = entry.get('standardized_IID_SD_gain') or {}
            check(f'bet_{i + 1}_claimed_gains_positive',
                  set(recorded) == {'sum', 'log', 'joint', 'shape'} and
                  all(type(v) in (float, int) and v > 0 for v in recorded.values()))
        check('distinct_shape_partitions', len({_m0_partition(t) for t in lines}) == 4)
        check('all_four_original_log_bins_identical', len({_m0_exact_log_bin(t) for t in lines}) == 1)
        check('exact_fair_universe_size', method.get('fair_6of49_count') == comb(49, 6))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        check('manifest_parsing', False)
    return {
        'decision': 'M0_FROZEN_MANIFEST_STRUCTURAL_PASS_OPTIMIZER_UNVERIFIED'
                    if all(checks.values()) else 'M0_FROZEN_MANIFEST_FAIL',
        'checks': checks,
        'failures': [k for k, v in checks.items() if not v],
        'original_git_blob_sha1': M0_GIT_BLOB,
        'scope': 'frozen M0 manifest integrity and structural constraints only',
        'unverified': ['original per-block score recomputation',
                       'candidate search trace and maximin ranking',
                       'global optimality of finite-pool search',
                       'future-draw predictive advantage'],
    }


def verify_p2b_prize_report(selection: dict, original_verification: dict) -> dict:
    """Check the exact four-ticket prize fields without calling P2B DP."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    def test(key: str, condition: object) -> None:
        checks[key] = bool(condition)
        if not checks[key]:
            failures.append(key)
    test('upstream_strategy_verification_pass',
         original_verification.get('decision') == 'STRATEGY_COMPLIANCE_PASS' and
         all(original_verification.get('checks', {}).values()))
    for arm in ('random', 'coverage', 'repair', 'hybrid'):
        try:
            reported = selection['observed']['arms'][arm]
            exact = independent_exact_prizes(reported['research_only_tickets'])
            old = reported['exact_fair_special_ball_prizes']
            test(f'{arm}.total_outcomes', old['total_equally_likely_outcomes'] ==
                 exact['total_equally_likely_outcomes'])
            test(f'{arm}.winning_line_histogram', old['exact_counts_winning_lines_0_to_4'] ==
                 exact['exact_counts_winning_lines_0_to_4'])
            test(f'{arm}.winning_line_probabilities', len(old['probabilities_winning_lines_0_to_4']) == 5
                 and all(np.isclose(a, b, atol=1e-13, rtol=0)
                         for a, b in zip(old['probabilities_winning_lines_0_to_4'],
                                         exact['probabilities_winning_lines_0_to_4'])))
            test(f'{arm}.tier_expectations', all(
                 np.isclose(old['expected_ticket_counts_by_tier'][k], exact['expected_ticket_counts_by_tier'][k],
                            rtol=0, atol=1e-13) for k in TIERS))
            test(f'{arm}.fixed_prize_assumption',
                 old['fixed_tier_payout_assumption_ntd'] == FIXED_PRIZES)
            test(f'{arm}.any_prize_and_lower_payout',
                 np.isclose(old['probability_any_prize'], exact['probability_any_prize'], rtol=0, atol=1e-13) and
                 np.isclose(old['expected_number_winning_lines'], exact['expected_number_winning_lines'], rtol=0, atol=1e-13) and
                 np.isclose(old['expected_nominal_fixed_lower_tier_payout_ntd'],
                            exact['expected_nominal_fixed_lower_tier_payout_ntd'], rtol=0, atol=1e-10))
        except (KeyError, TypeError, ValueError, IndexError, AssertionError) as exc:
            test(f'{arm}.missing_or_invalid_prize_report', False)
    return {'checks': checks, 'failures': failures,
            'decision': 'EXACT_PRIZE_COMPLIANCE_PASS' if all(checks.values())
                        else 'EXACT_PRIZE_COMPLIANCE_FAIL',
            'scope': 'independent bonus-ball event-intersection histogram and analytical tier multiplicity',
            'no_forecast_claim': True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-report', type=Path, required=True)
    parser.add_argument('--m0-manifest', type=Path,
                        default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    parser.add_argument('--csv-root', type=Path, default=Path('data/csv'))
    parser.add_argument('--bin-root', type=Path, default=Path('data/bin'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    dest = args.out.resolve()
    if not dest.is_relative_to(Path('results').resolve()) or dest.exists():
        parser.error('write-once verification report required under results/')
    source, provenance = audit_repository_inputs(args.csv_root, args.bin_root)
    report_bytes = args.selection_report.read_bytes()
    report = json.loads(report_bytes)
    v1 = verify_selection(report, source, source_gate=provenance)
    bonus = verify_p2b_prize_report(report, v1)
    m0 = verify_m0_frozen_manifest(args.m0_manifest.read_bytes())
    decision = ('STRATEGY_V11_PASS_M0_OPTIMIZER_UNVERIFIED'
                if v1['decision'] == 'STRATEGY_COMPLIANCE_PASS' and
                   bonus['decision'] == 'EXACT_PRIZE_COMPLIANCE_PASS' and
                   m0['decision'] == 'M0_FROZEN_MANIFEST_STRUCTURAL_PASS_OPTIMIZER_UNVERIFIED'
                else 'STRATEGY_V11_FAIL')
    result = {'schema': SCHEMA, 'decision': decision, 'original_selection_sha256':
              sha256(report_bytes).hexdigest(), 'source_gate': provenance,
              'p2b_v1': v1, 'prizes': bonus, 'm0_manifest': m0,
              'limitations': ['M0 historical block scores and maximin optimization NOT independently verified',
                              'P2B selection is finite-pool heuristic; no global optimality certificate',
                              'No prediction validation or outcome performance claim']}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='ascii', newline='\n') as out:
        json.dump(result, out, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        out.write('\n')
    print(f'{decision}: P2B {sum(v1["checks"].values())}/{len(v1["checks"])}; '
          f'prizes {sum(bonus["checks"].values())}/{len(bonus["checks"])}; '
          f'M0 structural {sum(m0["checks"].values())}/{len(m0["checks"])}')
    print('REPORT:', dest)
    return 0 if decision.startswith('STRATEGY_V11_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
