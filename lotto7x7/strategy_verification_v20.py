"""v2.0: integer full-qualified-source bounds with coupled features and 24-ball disjointness.

This studies frozen M0 four-block percentile-rank OPTIMIZATION QUALITY, not
lottery predictability. The 2,497-draw source cutoff and 13,704 empirical rank
axes are immutable. All exact upper bounds are algebraically valid for the
complete newly sampled four-positive-qualified sources, not just 2,000 screens.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v13 import validate_pool, shape_partition_bound, quality
from .strategy_verification_v16 import portfolio_audit, local_coordinate_ascent, canonical_sha
from .strategy_verification_v17 import sampled_unique, score_uniform_stream, screen_pool, greedy_multistart
from .strategy_verification_v18 import REFERENCE_RANK_AXES_SHA, SOURCE_PREFIX_SHA

SCHEMA = 'lotto649.strategy_verification.v2.0.full_qualified.v1'
SEEDS = (649, 20261111, 20261112)
RAW_SIZES = (160000, 320000)
EXPECTED_REFERENCE_N = 13704
PREREGISTERED_BASELINE = (43549, 195617, 43549, 54184)


def grouped_exact_primary_slice(pool: list[dict], witness: list[list[int]]) -> tuple[list[list[dict]], dict]:
    """Lossless iff 4th Shape-class cap is strictly greater than the 5th.

    Any four disjoint tickets with four distinct Shape classes have
    min(block totals) <= ShapeTotal <= top-four per-class Shape caps.
    If a legal witness hits that upper, every lexically superior portfolio
    must also hit it, hence one cap-attaining candidate from each top class.
    """
    ranks, tickets, parts = validate_pool(pool)
    if len(pool) != len(set(tickets)):
        raise ValueError('duplicate six-number candidates')
    witness_audit = portfolio_audit(pool, witness)
    class_caps = shape_partition_bound(pool)['all_partition_caps']
    if len(class_caps) < 5 or class_caps[3]['max_shape_rank'] <= class_caps[4]['max_shape_rank']:
        raise ValueError('FAIL_CLOSED_NO_UNIQUE_TOP_FOUR_SHAPE_PARTITIONS')
    top = {int(c['partition_code']): int(c['max_shape_rank']) for c in class_caps[:4]}
    bound = sum(top.values())
    if witness_audit['min_rank'] != bound or witness_audit['rank_sums'][3] != bound:
        raise ValueError('FAIL_CLOSED_NO_FEASIBLE_PRIMARY_OPTIMUM_WITNESS')
    groups = [[r for r in pool if r['part'] == p and r['rank'][3] == cap]
              for p,cap in sorted(top.items())]
    if any(not g for g in groups):
        raise AssertionError('empty mandatory Shape partition')
    reduced_keys = {tuple(r['ticket']) for group in groups for r in group}
    if not all(tuple(t) in reduced_keys for t in witness):
        raise AssertionError('witness omitted by mathematically mandatory Shape slice')
    if any(int(row['rank'][3]) != int(top[row['part']]) for group in groups for row in group):
        raise AssertionError('reduction retained non-maximal Shape ticket')
    return groups, {'method':'EXACT_TOP_FOUR_SHAPE_PARTITION_REDUCTION',
                    'primary_upper':bound,
                    'primary_feasible_lower':witness_audit['min_rank'],
                    'strict_4th_5th_gap':class_caps[3]['max_shape_rank']-class_caps[4]['max_shape_rank'],
                    'partitions':[p for p in sorted(top)],
                    'partition_caps':[top[p] for p in sorted(top)],
                    'group_sizes':[len(g) for g in groups],
                    'total_reduced':sum(len(g) for g in groups),
                    'reduced_sha256':canonical_sha([r for g in groups for r in g])}


def coupled_class_bounds(groups: list[list[dict]]) -> dict:
    """Exact per-class maxima of *coupled* ticket score, not axiswise maxima.

    For a legal primary-optimal 4-ticket portfolio, sum of individual rank
    sums is <= sum of each class's max sum of the four ticket rank entries.
    In particular this does not assume that the best Sum, Log and Joint
    independently attainable entries correspond to the same ticket.
    """
    if len(groups) != 4 or any(not g for g in groups):
        raise ValueError('requires four nonempty mandatory Shape groups')
    caps = []
    for g in groups:
        totals=[sum(r['rank']) for r in g]
        high=max(totals)
        idx=min((j for j,v in enumerate(totals) if v == high), key=lambda j: tuple(g[j]['ticket']))
        caps.append({'part':g[0]['part'], 'count':len(g), 'exact_max_coupled_total':int(high),
                     'witness_ticket':g[idx]['ticket'], 'witness_rank':g[idx]['rank'],
                     'max_equal_coupled_total_tickets':int(totals.count(high))})
    upper = sum(v['exact_max_coupled_total'] for v in caps)
    # Primary Shape ranks are per-class constant in this lossless slice.
    return {'method':'SUM_OF_PER_CLASS_MAX_OF_COUPLED_FOUR_BLOCK_RANK',
            'exact_integer_total_upper':upper,
            'per_class_caps':caps,
            'excludes_number_overlap_constraints':True,
            'proof':'Every primary-optimal legal portfolio has exactly one ticket in each mandatory Shape group. Its TotalRank is bounded by the sum of groupwise maxima of the SAME ticket four-rank sum.'}


def _ticket_masks(group: list[dict]) -> np.ndarray:
    return np.asarray([sum(1 << (v-1) for v in r['ticket']) for r in group], dtype=np.uint64)


def max_disjoint_two_class_total(a: list[dict], b: list[dict], *, chunk:int=256) -> dict:
    """Exact max of coupled rank totals of a non-overlapping two-class pair.

    This is a small (matrix batched) integer exhaustive enumeration and its
    result is a mathematically valid but generally non-tight 4-line bound.
    """
    if not a or not b or not (1 <= chunk <= 4096):
        raise ValueError('invalid pair groups or chunk')
    ma = _ticket_masks(a); mb=_ticket_masks(b)
    sa = np.asarray([sum(x['rank']) for x in a], dtype=np.int64)
    sb = np.asarray([sum(x['rank']) for x in b], dtype=np.int64)
    high = -1; idx=(-1,-1); count=0
    for start in range(0,len(a),chunk):
        end=min(len(a),start+chunk)
        valid=(ma[start:end,None]&mb[None,:])==0
        count += int(np.count_nonzero(valid))
        if not valid.any():continue
        combined=sa[start:end,None]+sb[None,:]
        safe=np.where(valid,combined,-1)
        local=int(safe.max())
        if local>high:
            high=local
            ij=np.argwhere(safe==local)[0]
            idx=(int(start+ij[0]),int(ij[1]))
    if high<0:
        raise ValueError('no compatible candidate pair from the two required classes')
    return {'exact_pair_max_total':high,
            'compatible_pair_count':count,
            'pair_cartesian_count':len(a)*len(b),
            'witness_tickets':[a[idx[0]]['ticket'],b[idx[1]]['ticket']]}


def disjoint_pairing_bound(groups:list[list[dict]], *,chunk:int=256) -> dict:
    """Rigorous upper on four-ticket Total using all three two-pair matchings.

    Each valid four-ticket portfolio induces a legal matching in each of the
    3 group pairings; sum of two independently maxed compatible pair scores
    is therefore an upper. Taking min across pairings remains an upper.
    Cross-pair overlaps and >= primary constraints are *not* assumed away.
    """
    if len(groups)!=4:raise ValueError('requires four groups')
    matches=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))
    memo={}; results=[]
    for match in matches:
        pair_reports=[]
        for u,v in match:
            key=(min(u,v),max(u,v))
            if key not in memo:
                memo[key]=max_disjoint_two_class_total(groups[u],groups[v],chunk=chunk)
            pair_reports.append(memo[key])
        upper=sum(x['exact_pair_max_total'] for x in pair_reports)
        results.append({'matching':[list(x) for x in match],
                        'two_compatible_pair_caps':[r['exact_pair_max_total'] for r in pair_reports],
                        'sum_exact_pair_caps':upper,
                        'pair_compatible_counts':[r['compatible_pair_count'] for r in pair_reports]})
    return {'method':'MIN_ACROSS_THREE_PERFECT_MATCHINGS_OF_EXACT_DISJOINT_PAIR_MAXIMA',
            'exact_integer_total_upper':min(x['sum_exact_pair_caps'] for x in results),
            'matchings':results,
            'proof':'Each disjoint four-ticket portfolio induces compatible pairs under all three matchings. Its TotalRank never exceeds either pairwise maximum sum.'}


def certify_full_source(pool:list[dict], witness:list[list[int]], *,
                        pairwise:bool=True, pair_chunk:int=256) -> dict:
    groups, reduction=grouped_exact_primary_slice(pool,witness)
    lower = portfolio_audit(pool,witness)
    coupled = coupled_class_bounds(groups)
    upper=int(coupled['exact_integer_total_upper'])
    pair=None
    if pairwise and upper > lower['total_rank']:
        pair=disjoint_pairing_bound(groups,chunk=pair_chunk)
        upper=min(upper,int(pair['exact_integer_total_upper']))
    if lower['total_rank']>upper:
        raise AssertionError('independently verified feasible witness exceeds rigorous upper')
    # Only when the total is certified can the next tie breaker be certified.
    # Shape is already fixed by the primary proof. Joint <= four classwise
    # maximum Joint ranks, an independently recomputed exact integer cap.
    joint_cap=sum(max(int(r['rank'][2]) for r in g) for g in groups)
    if lower['joint_rank']>joint_cap:
        raise AssertionError('feasible Joint score exceeds integer class caps')
    total_equal=lower['total_rank']==upper
    final_equal=total_equal and lower['joint_rank']==joint_cap
    decision = ('EXACT_FULL_LEXICOGRAPHIC_OPTIMUM_IN_SOURCE' if final_equal else
                'EXACT_TOTAL_BUT_JOINT_TIEBREAKER_UNRESOLVED' if total_equal else
                'VALID_FULL_SOURCE_TOTAL_UPPER_GAP_UNRESOLVED')
    return {'decision':decision,'full_source_candidate_count':len(pool),
            'source_pool_sha256':canonical_sha([{'ticket':r['ticket'],'part':r['part'],'rank':r['rank']} for r in pool]),
            'reduction':reduction,'feasible_witness':lower,
            'coupled_ticket_upper':coupled,'disjoint_pair_upper':pair,
            'exact_total_upper':upper,'exact_feasible_total_lower':lower['total_rank'],
            'integer_total_upper_gap':upper-lower['total_rank'],
            'exact_joint_tiebreak_upper':joint_cap,
            'joint_upper_gap':joint_cap-lower['joint_rank'],
            'primary_exact':True,'full_lexicographic_exact':final_equal,
            'scope':'only source-qualified candidates from frozen historical reference and specified seed/sample; not the entire 6/49 universe'}


def run_source_audit(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                     seeds:tuple[int,...]=SEEDS,
                     raw_sizes:tuple[int,...]=RAW_SIZES,
                     pair_chunk:int=256) -> dict:
    if not seeds or len(set(seeds))!=len(seeds) or not raw_sizes or tuple(sorted(set(raw_sizes)))!=raw_sizes or raw_sizes[-1]>320000:
        raise ValueError('invalid preregistered experiment settings')
    verified=verify_full_m0(csv_root=csv_root,bin_root=bin_root,
            manifest_path=manifest,golden_path=golden,
            include_qualified_candidates=True,include_frozen_score_reference=True)
    if (verified['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED'
         or verified['failed_checks'] or len(verified['checks'])!=152
         or verified['source']['historical_prefix_sha256']!=SOURCE_PREFIX_SHA):
        raise AssertionError('FAIL_CLOSED_ORIGINAL_152_CHECKS_OR_SOURCE_PREFIX')
    ref=verified.pop('frozen_score_reference')
    original=verified.pop('qualified_candidates')
    reference_parity=score_uniform_stream([tuple(r['ticket']) for r in original],ref)[0]
    if {(tuple(r['ticket']),int(r['part']),tuple(r['rank'])) for r in reference_parity} != \
       {(tuple(r['ticket']),int(r['part']),tuple(r['rank'])) for r in original}:
        raise AssertionError('FAIL_CLOSED_ALL_13704_FROZEN_RANK_PARITY')
    ranksha=sha256(json.dumps(ref['sorted_qualified_gains'],separators=(',',':')).encode('ascii')).hexdigest()
    if ranksha!=REFERENCE_RANK_AXES_SHA:raise AssertionError('FAIL_CLOSED_REFERENCE_RANK_AXES')
    records=[]
    previous={}
    for seed in seeds:
        stream,checkpoints=sampled_unique(seed,raw_sizes)
        allscored,_=score_uniform_stream(stream,ref)
        prior=None
        for cp in checkpoints:
            raw=cp['raw_count']
            candidates=[{k:r[k] for k in ('ticket','part','rank')} for r in allscored if r['appearance']<raw]
            ranked=screen_pool([r for r in allscored if r['appearance']<raw],policy='ranked',cap=2000,seed=seed,sample_size=raw)
            base=greedy_multistart(ranked,restarts=6)
            witness=base['tickets']
            if prior is not None and tuple(prior['quality'])>tuple(base['quality']):witness=prior['tickets']
            # The initial baseline is always from the same source; it remains
            # available on nested raw prefixes, not imported from other seeds.
            refined=local_coordinate_ascent(candidates,witness,max_passes=16)
            if tuple(refined['quality'])>tuple(quality_from_pool(candidates,witness)):
                witness=refined['tickets']
            evidence=certify_full_source(candidates,witness,pairwise=True,pair_chunk=pair_chunk)
            previous[(seed,raw)]=evidence
            prior=evidence['feasible_witness']
            records.append({'seed':seed,'raw_unique_sample':raw,
                            'raw_sample_sha256':cp['sample_sha256'],
                            'qualified_count':len(candidates),
                            'screened_2k_baseline_quality':base['quality'],
                            'full_source_one_swap':refined['quality'],
                            'certificate':evidence})
    return {'schema':SCHEMA,'decision':'V20_FULL_SOURCE_BOUND_AUDIT_PASS',
            'source_gate':{'checks':152,'history_draws':verified['source']['historical_prefix_draws'],
                           'history_sha256':SOURCE_PREFIX_SHA,'reference_qualified':len(original),
                           'reference_score_axes_sha256':ranksha,
                           'exact_reference_rank_parity':True},
            'parameters':{'seeds':list(seeds),'raw_sizes':list(raw_sizes),'pair_chunk':pair_chunk},
            'cases':records,
            'limitations':['A proven exact objective is conditional on the positive-qualified source sample, not all 6/49 combinations.',
                           'An unresolved integer upper gap is not evidence an improved feasible portfolio exists.',
                           'All scores are frozen retrospective strategy-ranking units, not draw probabilities.',
                           'Original M0/P3f/P5A seals, historical files and past strategy reports remain unchanged.']}


def quality_from_pool(pool:list[dict], tickets:list[list[int]]) -> tuple[int,...]:
    return tuple(portfolio_audit(pool,tickets)['quality'])


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--raw-sizes',nargs='+',type=int,default=list(RAW_SIZES))
    p.add_argument('--pair-chunk',type=int,default=256)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    output=args.out.resolve()
    if not output.is_relative_to(Path('results').resolve()) or output.exists():
        p.error('new write-once JSON under results required')
    report=run_source_audit(csv_root=args.csv_root,bin_root=args.bin_root,
        manifest=args.manifest,golden=args.golden,seeds=tuple(args.seeds),raw_sizes=tuple(args.raw_sizes),
        pair_chunk=args.pair_chunk)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='ascii',newline='\n') as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(report['decision'])
    for row in report['cases']:
        c=row['certificate']
        print('SEED',row['seed'],'RAW',row['raw_unique_sample'],
              'QUALIFIED',row['qualified_count'],
              'FOUND',c['feasible_witness']['quality'],
              'COUPLED_UPPER',c['coupled_ticket_upper']['exact_integer_total_upper'],
              'PAIR_UPPER',c['disjoint_pair_upper']['exact_integer_total_upper'] if c['disjoint_pair_upper'] else None,
              'EXACT_TOTAL_UPPER',c['exact_total_upper'],
              'GAP',c['integer_total_upper_gap'],
              'DECISION',c['decision'])
    print('REPORT',output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
