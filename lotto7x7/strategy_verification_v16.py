"""v1.6: nested candidate-pool scaling with immutable rank semantics.

Engineering question: how does finite candidate budget change the best *found*
M0 four-block lexicographic objective, and how tight is a rigorous upper bound?
No forecasts, bet recommendations or mutation of historical/frozen data.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v13 import (
    validate_pool, is_feasible, quality, shape_partition_bound,
    optimization_certificate, constraints,
)
from .strategy_verification_v14 import (
    exact_primary_optimal_reduction, exact_lp_dual_certificate, certified_mip_search,
)

SCHEMA = 'lotto649.strategy_verification.v1.6'
ORIGINAL_POOL_SHA = '366af606a178173d159198fef784a779b8ed430d52babc89d526f1df575f216e'
ORIGINAL_SAMPLE_SHA = '5744bb7b6e5ee9e086cb2c9eb17b7872b3028d512365610fb747672586b92c8c'
ORIGINAL_PREFIX_SHA = 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
QUALIFIED_COUNT = 13704
BASELINE_MAXIMIN = 43549
BASELINE_TOTAL = 192715
BASELINE_JOINT = 52895
# Frozen *experimental* finite-pool optimizer witness; NOT any purchased tickets.
V15_REFERENCE = (
    (11, 26, 29, 32, 42, 46),
    (14, 20, 25, 33, 45, 49),
    (18, 21, 28, 35, 40, 44),
    (13, 19, 31, 37, 43, 47),
)
POOL_SIZES = (2000, 4000, 8000, QUALIFIED_COUNT)


def canonical_sha(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


def nested_pools(original: list[dict], qualified: list[dict],
                 sizes: tuple[int, ...] = POOL_SIZES,
                 *, verify_frozen_sha: bool = True) -> list[dict]:
    """Keep original 2k prefix, add candidates deterministically without re-ranking.

    Never recompute percentile ranks on each pool: that would make the
    cross-budget objective non-comparable. All qualified candidates are scored
    against the same original 13,704-item reference sample.
    """
    validate_pool(original)
    validate_pool(qualified)
    if verify_frozen_sha and canonical_sha(original) != ORIGINAL_POOL_SHA:
        raise ValueError('original 2,000-pool SHA drift')
    original_set = {tuple(p['ticket']) for p in original}
    ref = {tuple(p['ticket']):p for p in qualified}
    if len(original_set) != len(original) or len(ref) != len(qualified):
        raise ValueError('duplicate tickets')
    if any(ref.get(tuple(p['ticket'])) != p for p in original):
        raise ValueError('original candidates or original ranks differ from qualified sample')
    if not sizes or sizes[0] != len(original) or sizes[-1] != len(qualified) or any(
            type(s) is not int or s < len(original) or s > len(qualified)
            for s in sizes) or list(sizes) != sorted(set(sizes)):
        raise ValueError('non-nested pool schedule')
    extras = sorted((p for p in qualified if tuple(p['ticket']) not in original_set),
                    key=lambda p: (-min(p['rank']), -sum(p['rank']), -p['rank'][3], p['ticket']))
    merged = list(original) + extras
    if len(merged) != len(qualified):
        raise AssertionError('nested schedule did not exhaust original qualified set')
    return [{'requested_size': size, 'pool': merged[:size],
             'pool_sha256': canonical_sha(merged[:size])} for size in sizes]


def _masks(tickets: list[tuple[int, ...]]) -> np.ndarray:
    return np.asarray([sum(1 << (int(ball) - 1) for ball in ticket)
                       for ticket in tickets], dtype=np.uint64)


def portfolio_audit(pool: list[dict], selection: list[list[int]]) -> dict:
    ranks, tickets, parts = validate_pool(pool)
    index = {ticket: i for i, ticket in enumerate(tickets)}
    try:
        ids = [index[tuple(ticket)] for ticket in selection]
    except KeyError as exc:
        raise ValueError('incumbent ticket absent from candidate pool') from exc
    if not is_feasible(ids, tickets, parts):
        raise ValueError('incumbent violates four-ticket/partition/disjoint constraints')
    sums = ranks[ids].sum(axis=0).tolist()
    return {'tickets': [list(tickets[i]) for i in ids],
            'rank_sums': [int(s) for s in sums],
            'quality': list(quality(sums, 'balanced')),
            'min_rank': int(min(sums)), 'total_rank': int(sum(sums)),
            'joint_rank': int(sums[2])}


def local_coordinate_ascent(pool: list[dict], selection: list[list[int]],
                            *, max_passes: int = 24) -> dict:
    """Independent deterministic 1-for-1 lexicographic exchange over all candidates."""
    if type(max_passes) is not int or not 1 <= max_passes <= 200:
        raise ValueError('invalid local search pass limit')
    ranks, tickets, parts = validate_pool(pool)
    indices = {ticket: i for i, ticket in enumerate(tickets)}
    try:
        chosen = [indices[tuple(t)] for t in selection]
    except KeyError as exc:
        raise ValueError('unknown starting candidate') from exc
    if not is_feasible(chosen, tickets, parts):
        raise ValueError('starting candidate set is invalid')
    masks = _masks(tickets)
    pcodes = np.asarray(parts, dtype=np.int64)
    rank = np.asarray(ranks, dtype=np.int64)
    swaps = 0
    finished = False
    # This depends on *fixed* rank vectors and exact six-number bitmasks only.
    for iteration in range(max_passes):
        moved = False
        for slot in range(4):
            other = [chosen[k] for k in range(4) if k != slot]
            occupied = int(masks[other[0]]) | int(masks[other[1]]) | int(masks[other[2]])
            allowed = np.flatnonzero(((masks & np.uint64(occupied)) == 0)
                & (pcodes != pcodes[other[0]])
                & (pcodes != pcodes[other[1]])
                & (pcodes != pcodes[other[2]]))
            if not len(allowed):
                raise AssertionError('no feasible replacement, original ticket must be feasible')
            othersum = rank[other].sum(axis=0)
            totals = rank[allowed] + othersum[None, :]
            minscore = totals.min(axis=1)
            totalscore = totals.sum(axis=1)
            # numpy lexsort: last key is primary sort key.
            order = np.lexsort((totals[:, 2], totals[:, 3], totalscore, minscore))
            best_key = (int(minscore[order[-1]]), int(totalscore[order[-1]]),
                        int(totals[order[-1], 3]), int(totals[order[-1], 2]))
            tied = allowed[np.flatnonzero((minscore == best_key[0])
                         & (totalscore == best_key[1])
                         & (totals[:,3] == best_key[2])
                         & (totals[:,2] == best_key[3]))]
            best_idx = min(map(int, tied), key=lambda k: tickets[k])
            current = quality(rank[chosen].sum(axis=0).tolist(), 'balanced')
            if best_key > current:
                chosen[slot] = best_idx
                moved = True
                swaps += 1
        if not moved:
            finished = True
            break
    out = portfolio_audit(pool, [list(tickets[i]) for i in chosen])
    out.update({'swaps': swaps, 'passes': iteration + 1,
                'one_swap_stationary_certified': finished,
                'local_limit_reached': not finished})
    return out


def bounded_mip_primary(pool: list[dict], *, seconds: float) -> dict:
    """Optional numerical primary search; any found witness independently revalidated."""
    if not 1 <= seconds <= 120:
        raise ValueError('invalid MILP time limit')
    ranks, tickets, parts, classes, a, b, e, zmax = constraints(pool)
    n = len(pool)
    costs = np.zeros(n+1, dtype=float)
    costs[-1] = -1
    sol = milp(costs, integrality=np.r_[np.ones(n, dtype=np.int8), 0],
               bounds=Bounds(np.zeros(n+1), np.r_[np.ones(n), zmax]),
               constraints=[LinearConstraint(a, -np.inf, b),
                            LinearConstraint(e, [4.], [4.])],
               options={'time_limit': float(seconds), 'mip_rel_gap':0.,'disp':False})
    info: dict[str,Any] = {'status': int(sol.status), 'message': str(sol.message),
                           'time_limit_seconds':float(seconds),
                           'solver_gap': float(sol.mip_gap) if getattr(sol,'mip_gap',None) is not None else None,
                           'numeric_only':True, 'feasible':False}
    if sol.x is None:
        return info
    selection = [i for i,x in enumerate(sol.x[:n]) if x >= 0.5]
    if len(selection) != 4 or not is_feasible(selection,tickets,parts):
        raise AssertionError('numerical MILP returned an invalid four-ticket portfolio')
    scores = ranks[selection].sum(axis=0)
    z = int(scores.min())
    if float(sol.x[-1]) > z + 1e-5:
        raise AssertionError('numerical MILP objective exceeds integral witness')
    info.update({'feasible':True, 'witness': [list(tickets[i]) for i in selection],
                 'rank_sums': [int(x) for x in scores], 'maximin':z})
    return info


def analyze_budget(pool: list[dict], incumbent: list[list[int]], *,
                   mip_seconds: float = 0., compute_exact_lp:bool = True) -> dict:
    """Return strict lower bound, safe integer upper bound, and uncertified gap."""
    validate_pool(pool)
    start = portfolio_audit(pool, incumbent)
    local = local_coordinate_ascent(pool, incumbent)
    best = local
    mip = None
    if mip_seconds > 0:
        mip = bounded_mip_primary(pool, seconds=mip_seconds)
        if mip.get('feasible'):
            refined = local_coordinate_ascent(pool, mip['witness'])
            if tuple(refined['quality']) > tuple(best['quality']):
                best = refined
    shape = shape_partition_bound(pool)
    u_shape = int(shape['exact_integer_upper_bound'])
    lp = optimization_certificate(pool) if compute_exact_lp else None
    u = min(u_shape, int(lp['integer_upper_bound'])) if lp else u_shape
    if u < best['min_rank']:
        raise AssertionError('valid witness exceeds exact mathematical upper bound')
    return {'size':len(pool), 'baseline_incumbent':start,
            'best_feasible_witness':best, 'local_from_baseline':local,
            'primary_rigorous_upper':u,
            'primary_shape_upper':u_shape,
            'primary_exact_dual_upper':int(lp['integer_upper_bound']) if lp else None,
            'primary_absolute_gap_at_most':u-best['min_rank'],
            'primary_optimality': 'EXACT_FINITE_POOL_PRIMARY_PASS' if u==best['min_rank']
                        else 'BOUNDED_GAP_UNRESOLVED',
            'numerical_mip_exploration':mip,
            'full_lexicographic_optimality': 'NOT_AUDITED_AT_THIS_POOL_SIZE',
            'comparison_basis':'all 4 block rank vectors fixed using the SAME 13704-qualified sample'}



def analyze_secondary(pool: list[dict], *, original_primary: list[list[int]],
                      incumbent: list[list[int]], seconds: float) -> dict:
    """Numerical lexicographic search within an EXACT primary-optimal slice.

    All v1.4 reductions follow from Shape class maxima; they are not
    heuristic pruning. Independent integer dual upper bounds are retained
    distinctly from numerical MIP optimality statuses.
    """
    if not 1 <= seconds <= 120:
        raise ValueError('invalid secondary search limit')
    reduced, proof = exact_primary_optimal_reduction(pool, original_primary)
    before = portfolio_audit(pool, incumbent)
    if before['min_rank'] != BASELINE_MAXIMIN:
        raise AssertionError('incumbent has not reached certified primary optimum')
    upper = exact_lp_dual_certificate(reduced, score_kind='total', primary_bound=BASELINE_MAXIMIN)
    secondary = certified_mip_search(reduced, score_kind='total',
                                     primary_bound=BASELINE_MAXIMIN,seconds=seconds)
    winner = before
    if secondary.get('feasible'):
        trial = portfolio_audit(pool, secondary['tickets'])
        if trial['min_rank'] != BASELINE_MAXIMIN or trial['total_rank'] != secondary['objective_integer']:
            raise AssertionError('secondary MIP witness did not independently verify')
        # No numerical outcome is adopted without a legal, integer-verified witness.
        trial_local = local_coordinate_ascent(pool, trial['tickets'])
        if tuple(trial_local['quality']) > tuple(winner['quality']):
            winner = trial_local
    if winner['total_rank'] > int(upper['integer_upper_bound']):
        raise AssertionError('integer witness exceeds exact secondary bound')
    joint = None
    joint_upper = None
    if secondary.get('solver_certifies_integer_optimum') and secondary.get('feasible'):
        best_total = int(secondary['objective_integer'])
        if winner['total_rank'] != best_total:
            raise AssertionError('secondary numerical-optimality contradicts integer witness')
        joint_upper = exact_lp_dual_certificate(reduced,score_kind='joint',
                        primary_bound=BASELINE_MAXIMIN,total_floor=best_total)
        joint = certified_mip_search(reduced,score_kind='joint',
                    primary_bound=BASELINE_MAXIMIN,total_floor=best_total,seconds=seconds)
        if joint.get('feasible'):
            candidate = portfolio_audit(pool,joint['tickets'])
            if candidate['min_rank']!=BASELINE_MAXIMIN or candidate['total_rank']!=best_total or candidate['joint_rank']!=joint['objective_integer']:
                raise AssertionError('numerical final tie-break witness does not satisfy fixed total')
            if tuple(candidate['quality']) > tuple(winner['quality']):
                winner = candidate
    return {
        'reduced_primary_optimum_candidates':len(reduced),
        'reduction_strict_partition_gap':proof['strict_partition_gap_to_fifth'],
        'total_exact_integer_upper':int(upper['integer_upper_bound']),
        'total_best_feasible_lower':winner['total_rank'],
        'total_rigorous_gap_at_most':int(upper['integer_upper_bound'])-winner['total_rank'],
        'total_solver':secondary,
        'total_certificate_basis':('EXACT_INTEGER_UPPER_EQUAL_WITNESS'
            if winner['total_rank']==int(upper['integer_upper_bound'])
            else 'MIP_ZERO_GAP_NUMERIC_ONLY' if secondary.get('solver_certifies_integer_optimum')
            else 'BOUNDED_UNRESOLVED'),
        'joint_solver':joint,
        'joint_exact_integer_upper':int(joint_upper['integer_upper_bound']) if joint_upper else None,
        'joint_certificate_basis':('EXACT_INTEGER_UPPER_EQUAL_WITNESS'
            if joint_upper is not None and winner['joint_rank']==int(joint_upper['integer_upper_bound'])
            else 'MIP_ZERO_GAP_NUMERIC_ONLY' if joint and joint.get('solver_certifies_integer_optimum')
            else 'BOUNDED_UNRESOLVED'),
        'best_feasible_witness':winner,
    }


def verify_scaling(*, csv_root:Path, bin_root:Path,manifest:Path,golden:Path,
                   sizes: tuple[int,...] = POOL_SIZES,
                   mip_seconds:float=6., dual_for_all:bool=True) -> dict:
    verified = verify_full_m0(csv_root=csv_root,bin_root=bin_root,
        manifest_path=manifest,golden_path=golden,
        include_verified_pool=True,include_qualified_candidates=True)
    if verified['decision'] != 'M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or verified['failed_checks']:
        raise ValueError('frozen M0 152-check gate did not pass')
    original = verified.pop('verified_pool')
    qualified = verified.pop('qualified_candidates')
    if (len(verified['checks'])!=152 or len(original)!=2000 or len(qualified)!=QUALIFIED_COUNT
        or verified['sample']['sha256']!=ORIGINAL_SAMPLE_SHA
        or verified['source']['historical_prefix_sha256']!=ORIGINAL_PREFIX_SHA):
        raise AssertionError('original source, score or qualifying sample drift')
    arms=nested_pools(original,qualified,sizes)
    first = arms[0]['pool']
    baseline = portfolio_audit(first, [list(t) for t in V15_REFERENCE])
    if baseline['quality'] != [BASELINE_MAXIMIN,BASELINE_TOTAL,BASELINE_MAXIMIN,BASELINE_JOINT]:
        raise AssertionError('original v1.5 exact optimum witness unexpectedly changed')
    results=[]
    incumbent=baseline['tickets']
    original_primary=verified['replayed']['primary']['tickets']
    prev_best=tuple(baseline['quality'])
    for i,arm in enumerate(arms):
        budget = analyze_budget(arm['pool'],incumbent,
                mip_seconds=0. if i==0 else mip_seconds,
                compute_exact_lp=(dual_for_all or i==0))
        current=budget['best_feasible_witness']
        # Original 2k full-lex optimum is independently exact-certificated by v1.5.
        if i==0:
            if current['quality']!=baseline['quality']:
                raise AssertionError('original 2k exact full-lex optimum was surpassed')
            budget['full_lexicographic_optimality']='V15_EXACT_FINITE_POOL_PASS'
            budget['secondary_search']={'status':'V15_EXHAUSTIVE_CERTIFICATE',
                                        'best_feasible_witness':baseline}
        else:
            if budget['primary_rigorous_upper']==BASELINE_MAXIMIN and current['min_rank']==BASELINE_MAXIMIN:
                secondary=analyze_secondary(arm['pool'],original_primary=original_primary,
                                             incumbent=current['tickets'],seconds=mip_seconds)
                budget['secondary_search']=secondary
                if tuple(secondary['best_feasible_witness']['quality']) > tuple(current['quality']):
                    current=secondary['best_feasible_witness']
                    budget['best_feasible_witness']=current
                budget['full_lexicographic_optimality']='NUMERIC_ONLY_OR_BOUNDED_GAP'
            else:
                budget['secondary_search']={'status':'SKIPPED_PRIMARY_UNRESOLVED'}
        if tuple(current['quality']) < prev_best:
            raise AssertionError('best feasible objective regressed in nested candidate schedule')
        prev_best=tuple(current['quality'])
        incumbent=current['tickets']
        budget['candidate_pool_sha256']=arm['pool_sha256']
        budget['added_since_previous']=0 if i==0 else budget['size']-results[-1]['size']
        results.append(budget)
    return {'schema':SCHEMA, 'decision':'NESTED_POOL_SCALING_AUDIT_PASS',
      'source':{'draws':verified['source']['historical_prefix_draws'],
                'prefix_sha256':ORIGINAL_PREFIX_SHA,'v12_checks':len(verified['checks']),
                'sample_sha256':ORIGINAL_SAMPLE_SHA},
      'rank_semantics':'frozen empirical ranks from original 13704 eligible candidates; never re-normalize per subset',
      'nested_pool_construction':'original exact 2000 prefix + deterministic remaining candidates sorted by -min(rank),-sum(rank),-shape_rank,ticket',
      'qualified_count':len(qualified),'baseline_original_pool_sha256':ORIGINAL_POOL_SHA,
      'baseline_v15_exhaustive_optimum':{'primary':BASELINE_MAXIMIN,'total':BASELINE_TOTAL,'joint':BASELINE_JOINT},
      'arms':results,
      'limitations':['Nested 2000/4000/8000/13704 schedule uses only the original sampled positive-qualified candidates.',
                     'Each arm has identical frozen score and rank units; no independent prediction tests.',
                     'Optimized feasible witnesses are lower bounds; a rigorous integer upper bound proves primary optimum only when equal.',
                     'MIP outcomes may support a numerical complete lexicographic optimum but do not establish an independent exact integer certificate.',
                     'Candidate pool size impacts optimization opportunity, not the underlying fair-lottery winning probability.',
                     'No source or forecast seals were changed.']}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    parser.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    parser.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    parser.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    parser.add_argument('--mip-seconds',type=float,default=6.)
    parser.add_argument('--no-dual-for-all',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    target=args.out.resolve()
    if not target.is_relative_to(Path('results').resolve()) or target.exists():
        parser.error('new write-once scaling report required under results/')
    result=verify_scaling(csv_root=args.csv_root,bin_root=args.bin_root,manifest=args.manifest,
        golden=args.golden,mip_seconds=args.mip_seconds,dual_for_all=not args.no_dual_for_all)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='ascii',newline='\n') as fp:
        json.dump(result,fp,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        fp.write('\n')
    print(result['decision'])
    for arm in result['arms']:
        q=arm['best_feasible_witness']['quality']
        print('POOL',arm['size'],'FOUND=',q,'PRIMARY_UPPER=',arm['primary_rigorous_upper'],
              'PRIMARY_GAP<=',arm['primary_absolute_gap_at_most'],arm['primary_optimality'])
    print('REPORT',target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
