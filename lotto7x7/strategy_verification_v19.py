"""v1.9: Diagnose empirical-rank score saturation versus candidate coverage.

Frozen M0 historical-score and 13,704-row rank axes are held fixed; this
module is a DIAGNOSTIC, not a modification of the M0 objective, nor a forecast.
The source and new-sample rank parsers must agree on every original candidate.
A class-separable integer bound optionally certifies an entire raw qualified
candidate sample when a legal witness saturates all necessary per-class caps.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
from math import fsum, prod
from pathlib import Path
import json

import numpy as np

from .strategy_verification_v12 import (
    verify_full_m0, classify_numbers, classification_keys, LEVELS, WEIGHTS,
)
from .strategy_verification_v13 import shape_partition_bound, validate_pool
from .strategy_verification_v16 import (
    BASELINE_MAXIMIN, BASELINE_TOTAL, BASELINE_JOINT, canonical_sha,
    portfolio_audit,
)
from .strategy_verification_v17 import (
    DEFAULT_SEEDS, DEFAULT_RAW_SIZES, POLICIES, sampled_unique,
    score_uniform_stream, screen_pool, greedy_multistart,
)
from .strategy_verification_v18 import REFERENCE_RANK_AXES_SHA, SOURCE_PREFIX_SHA

SCHEMA='lotto649.strategy_verification.v1.9.score_saturation.v1'
AXES=('sum','log','joint','shape')
REFERENCE_COUNT=13704
POOL_CAP=2000
RESTARTS=6


def reference_axes_summary(axes: list[list[float]]) -> list[dict]:
    """Summarize tie/finite-rank resolution independently of any new samples."""
    if len(axes)!=4:
        raise ValueError('need exactly four frozen rank axes')
    out=[]
    for name,ax in zip(AXES,axes):
        if len(ax)!=REFERENCE_COUNT or ax!=sorted(ax):
            raise ValueError('frozen axis length or monotonicity mismatch')
        unique=len(set(ax))
        last=ax[-1]
        first_top=bisect_left(ax,last)
        out.append({'axis':name,'reference_count':len(ax),
                    'distinct_gain_values':unique,
                    'reference_unique_fraction':unique/len(ax),
                    'top_equal_gain_count':len(ax)-first_top,
                    'reference_max_equal_gain_first_rank':first_top,
                    'novel_beyond_reference_max_rank':len(ax),
                    'max_gain':float(last),
                    'top_gain_is_tied':len(ax)-first_top>1})
    return out


def exact_frozen_gains(rows: list[dict], reference: dict) -> list[list[float]]:
    """Compute all four raw block gains and assert strict frozen-rank parity.

    This is deliberately separate from v1.7 scoring: a reference-aware
    diagnostic cannot trust the rank value returned by the candidate scorer.
    """
    if not rows:return []
    validate_pool([{k:r[k] for k in ('ticket','part','rank')} for r in rows])
    cuts=reference['integer_product_thresholds']
    axes=reference['sorted_qualified_gains']
    if len(axes)!=4 or any(len(x)!=REFERENCE_COUNT for x in axes):
        raise ValueError('rank axes reference was modified')
    nums=np.asarray([r['ticket'] for r in rows],dtype=np.uint8)
    cats=classification_keys(classify_numbers(nums))
    levels=reference['shape_gain_tables']
    shape_sd=reference['shape_sd']
    out=[]
    for i,row in enumerate(rows):
        t=row['ticket']
        si=sum(t)-21
        li=bisect_right(cuts,prod(t))-1
        ji=si*(len(cuts)-1)+li
        level_scores={lv:levels[lv][int(cats[lv][i])]/shape_sd[lv] for lv in LEVELS}
        shape=fsum(WEIGHTS[lv]*level_scores[lv] for lv in LEVELS)
        gains=[
            reference['sum_gain_numerators'][si]/reference['sum_denominator']/reference['sum_sd'],
            reference['log_gain_numerators'][li]/reference['log_denominator']/reference['log_sd'],
            reference['joint_gain_numerators'][ji]/reference['joint_denominator']/reference['joint_sd'],
            shape,
        ]
        expected=[bisect_left(axis,g) for axis,g in zip(axes,gains)]
        if expected!=row['rank'] or int(cats['partition'][i])!=row['part']:
            raise AssertionError('FAIL_CLOSED_NEW_CANDIDATE_RANK_OR_SHAPE_PARITY')
        if min(gains)<=0:
            raise AssertionError('candidate did not pass all four positive-block filters')
        out.append([float(x) for x in gains])
    return out


def saturation_counts(rows: list[dict], gains: list[list[float]],
                      axes: list[list[float]]) -> list[dict]:
    """Separate strict above-max clipping from ordinary equal/tied gain values."""
    if len(rows)!=len(gains):raise ValueError('candidate/gain length mismatch')
    result=[]
    for k,name in enumerate(AXES):
        upper=axes[k][-1]
        bucket=defaultdict(set)
        strict=equal=ceiling=0
        for row,gs in zip(rows,gains):
            g=gs[k];rank=row['rank'][k]
            if rank!=bisect_left(axes[k],g):
                raise AssertionError('gain/rank alias parity failure')
            strict+=int(g>upper)
            equal+=int(g==upper)
            ceiling+=int(rank==len(axes[k]))
            bucket[rank].add(g)
        aliased={r for r,values in bucket.items() if len(values)>1}
        result.append({'axis':name,'candidates':len(rows),
                       'strictly_above_reference_max':strict,
                       'equal_to_reference_max':equal,
                       'rank_at_empirical_ceiling':ceiling,
                       'share_strictly_above_reference_max':strict/len(rows) if rows else 0.,
                       'distinct_observed_integer_ranks':len(bucket),
                       'observed_rank_values_with_different_gains':len(aliased),
                       'candidates_in_gain_alias_rank_buckets':sum(
                           1 for r in rows if r['rank'][k] in aliased),
                       'unique_beyond_reference_max_raw_gains':len({g[k] for g in gains if g[k]>upper})})
    return result


def class_separable_envelope(pool: list[dict], witness: list[list[int]]) -> dict:
    """Exact integer per-class envelope on the full qualified source.

    The bound is conditional on reaching primary Shape upper. If a legal
    portfolio reaches that bound, every lexicographically better portfolio
    MUST select one of the top-four unique Shape classes and must attain
    each class's maximum Shape rank. Per-block integer class maxima then
    upper-bound every feasible primary-optimal four-ticket portfolio.
    """
    pool=[{k:r[k] for k in ('ticket','part','rank')} for r in pool]
    validate_pool(pool)
    by_ticket={tuple(row['ticket']):row for row in pool}
    verified=portfolio_audit(pool,witness)
    maxima=shape_partition_bound(pool)['all_partition_caps']
    if len(maxima)<5 or maxima[3]['max_shape_rank']<=maxima[4]['max_shape_rank']:
        return {'status':'NOT_CERTIFIABLE_CLASS_BOUND_TIE','sample_count':len(pool),
                'strict_fourth_vs_fifth_gap':0}
    caps={int(x['partition_code']):int(x['max_shape_rank']) for x in maxima[:4]}
    primary_bound=sum(caps.values())
    if verified['min_rank']!=primary_bound or verified['rank_sums'][3]!=primary_bound:
        return {'status':'PRIMARY_BOUND_NOT_ATTAINED','sample_count':len(pool),
                'primary_upper_bound':primary_bound,
                'found_primary':verified['min_rank']}
    subset={p:[] for p in caps}
    for row in pool:
        if row['part'] in caps and row['rank'][3]==caps[row['part']]:
            subset[row['part']].append(row)
    if any(not entries for entries in subset.values()):
        raise AssertionError('empty top Shape partition in guaranteed reduction')
    for t in witness:
        row=by_ticket[tuple(t)]
        if row not in subset.get(row['part'],[]):
            raise AssertionError('primary-optimal witness omitted by lossless reduction')
    per_part=[]
    block_upper=[0,0,0,0]
    for p in sorted(subset):
        group=subset[p]
        maxs=[max(int(x['rank'][k]) for x in group) for k in range(4)]
        block_upper=[a+b for a,b in zip(block_upper,maxs)]
        per_part.append({'part':p,'primary_eligible_count':len(group),
                         'block_rank_independent_upper':maxs})
    if block_upper[3]!=primary_bound:
        raise AssertionError('wrong Shape partition cap calculation')
    actual=verified['rank_sums']
    if any(a>b for a,b in zip(actual,block_upper)):
        raise AssertionError('feasible portfolio violates class-separable rank upper bound')
    total_upper=sum(block_upper)
    score_quality=[primary_bound,total_upper,primary_bound,block_upper[2]]
    current=verified['quality']
    total_gap=total_upper-current[1]
    # For exact lex certification, reaching the componentwise upper is a
    # sufficient (NOT necessary) condition. Otherwise remain UNSOLVED.
    exact=tuple(current)==tuple(score_quality)
    return {'status':'EXACT_LEXICOGRAPHIC_CERTIFIED_BY_CLASS_CAPS' if exact else 'VALID_UPPER_GAP_UNRESOLVED',
            'sample_count':len(pool),'primary_upper_bound':primary_bound,
            'strict_fourth_vs_fifth_gap':maxima[3]['max_shape_rank']-maxima[4]['max_shape_rank'],
            'primary_eligible_candidates':sum(len(x) for x in subset.values()),
            'per_part':per_part,'componentwise_block_upper':block_upper,
            'full_lexicographic_tuple_component_upper':score_quality,
            'feasible_witness_quality':current,
            'total_upper_minus_feasible':total_gap,
            'joint_upper_minus_feasible':block_upper[2]-current[3],
            'bound_drops_number_exclusivity_and_is_rigorous':True,
            'full_lexicographic_exact_certified':exact}


def audit_saturation(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                     seeds:tuple[int,...]=DEFAULT_SEEDS,
                     raw_sizes:tuple[int,...]=DEFAULT_RAW_SIZES,
                     pool_cap:int=POOL_CAP,restarts:int=RESTARTS) -> dict:
    if (len(set(seeds))!=len(seeds) or not seeds or
        list(raw_sizes)!=sorted(set(raw_sizes)) or not raw_sizes or
        raw_sizes[-1]>320000 or pool_cap<4):
        raise ValueError('invalid preregistered v1.9 matrix')
    v12=verify_full_m0(csv_root=csv_root,bin_root=bin_root,
           manifest_path=manifest,golden_path=golden,
           include_qualified_candidates=True,include_frozen_score_reference=True)
    if (v12['failed_checks'] or len(v12['checks'])!=152 or
        v12['source']['historical_prefix_sha256']!=SOURCE_PREFIX_SHA):
        raise AssertionError('FAIL_CLOSED_ORIGINAL_M0_SOURCE_AND_152_CHECK_GATE')
    ref=v12.pop('frozen_score_reference')
    baseline=v12.pop('qualified_candidates')
    axes=ref['sorted_qualified_gains']
    rank_sha=sha256(json.dumps(axes,separators=(',',':')).encode('ascii')).hexdigest()
    if rank_sha!=REFERENCE_RANK_AXES_SHA or len(baseline)!=REFERENCE_COUNT:
        raise AssertionError('FAIL_CLOSED_FROZEN_RANK_AXIS_DRIFT')
    baseline_gains=exact_frozen_gains(baseline,ref)
    if any(r['rank'][k]!=bisect_left(axes[k],g[k]) for r,g in zip(baseline,baseline_gains) for k in range(4)):
        raise AssertionError('FAIL_CLOSED_BASELINE_EXACT_GAIN_PARITY')
    axis_summary=reference_axes_summary(axes)
    summary=[];policies=[]
    for seed in seeds:
        stream,checkpoints=sampled_unique(seed,raw_sizes)
        new_rows,account=score_uniform_stream(stream,ref)
        gain=exact_frozen_gains(new_rows,ref)
        by_ticket={tuple(row['ticket']):g for row,g in zip(new_rows,gain)}
        for cp in checkpoints:
            sz=cp['raw_count']
            eligible=[row for row in new_rows if row['appearance']<sz]
            eligible_gains=[by_ticket[tuple(x['ticket'])] for x in eligible]
            axes_report=saturation_counts(eligible,eligible_gains,axes)
            if len(eligible)<pool_cap:raise ValueError('not enough qualified candidates')
            conditions=[]
            for policy in POLICIES:
                pool=screen_pool(eligible,policy=policy,cap=pool_cap,seed=seed,sample_size=sz)
                selected_gain=[by_ticket[tuple(x['ticket'])] for x in pool]
                winner=greedy_multistart(pool,restarts=restarts)
                upper=shape_partition_bound(pool)['exact_integer_upper_bound']
                if upper < winner['min_rank']:
                    raise AssertionError('candidate winner exceeds hard upper bound')
                pool_diagnostics=saturation_counts(pool,selected_gain,axes)
                # Source-wide upper/gap check only applies to portfolios
                # whose *witness* is drawn from that source and meets primary.
                conditions.append({'policy':policy,'candidate_pool_sha256':canonical_sha(pool),
                    'feasible_quality':winner['quality'],
                    'primary_upper':int(upper),'primary_gap_at_most':int(upper-winner['min_rank']),
                    'policy_rank_ceiling_counts':{v['axis']:v['rank_at_empirical_ceiling'] for v in pool_diagnostics},
                    'policy_novel_gain_over_max_counts':{v['axis']:v['strictly_above_reference_max'] for v in pool_diagnostics},
                    'screened_2000_full_source_envelope':class_separable_envelope(pool,winner['tickets']),
                    'raw_qualified_source_envelope':class_separable_envelope(eligible,winner['tickets']),
                    'feasible_witness':winner['tickets']})
            summary.append({'seed':seed,'raw_unique_count':sz,'raw_sample_sha256':cp['sample_sha256'],
                            'qualified_count':len(eligible),
                            'qualified_fraction':len(eligible)/sz,
                            'per_axis_source_saturation':axes_report,
                            'policies':conditions})
    return {'schema':SCHEMA,'decision':'V19_FROZEN_SCORE_SATURATION_AUDIT_PASS',
            'source_gate':{'draws':v12['source']['historical_prefix_draws'],
                'historical_prefix_sha256':v12['source']['historical_prefix_sha256'],
                'original_152_checks':len(v12['checks']),
                'original_qualified':len(baseline),
                'reference_axes_sha256':rank_sha,
                'original_qualified_raw_gain_recalculation':'PASS_ALL_13704_RANKS',
                'original_axis_saturation':saturation_counts(baseline,baseline_gains,axes)},
            'frozen_reference_resolution':axis_summary,
            'parameters':{'seeds':list(seeds),'raw_sizes':list(raw_sizes),
                          'policy_screen_cap':pool_cap,'optimizer_restarts':restarts,
                          'policies':list(POLICIES)},
            'studies':summary,
            'limitations':[
                'Rank saturation is an empirical-reference representation property, not evidence of a nonrandom lottery draw process.',
                'A score above the reference maximum is assigned rank 13704 by the original bisect_left scale, losing gain magnitude information.',
                'Per-class upper bounds are rigorous but may be loose; full optimality is claimed only when a legal witness exactly saturates the componentwise upper.',
                'This is a frozen retrospective 2497-draw score reference; no next draw or payout edge is inferred.',
                'Raw-scale gains are reported solely as diagnostic measurements and do not redefine the frozen M0 score objective.',
                'All source, already sealed experiments, and historical data remain unchanged.',
            ]}


def main(argv:list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    p.add_argument('--raw-sizes',type=int,nargs='+',default=list(DEFAULT_RAW_SIZES))
    p.add_argument('--pool-cap',type=int,default=POOL_CAP)
    p.add_argument('--restarts',type=int,default=RESTARTS)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    target=args.out.resolve()
    if not target.is_relative_to(Path('results').resolve()) or target.exists():
        p.error('write-once v1.9 result path under results required')
    report=audit_saturation(csv_root=args.csv_root,bin_root=args.bin_root,
        manifest=args.manifest,golden=args.golden,seeds=tuple(args.seeds),
        raw_sizes=tuple(args.raw_sizes),pool_cap=args.pool_cap,restarts=args.restarts)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='ascii',newline='\n') as f:
        json.dump(report,f,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        f.write('\n')
    print(report['decision'])
    print('FROZEN AXIS DISTINCT COUNTS',[(x['axis'],x['distinct_gain_values'],x['top_equal_gain_count']) for x in report['frozen_reference_resolution']])
    for study in report['studies']:
        print('SEED',study['seed'],'RAW',study['raw_unique_count'],'QUALIFIED',study['qualified_count'],
              'SAT STRICT',{x['axis']:x['strictly_above_reference_max'] for x in study['per_axis_source_saturation']})
        for arm in study['policies']:
            print('POLICY',arm['policy'],'QUALITY',arm['feasible_quality'],
                'FULL SOURCE ENVELOPE',arm['raw_qualified_source_envelope']['status'],
                'GAP',arm['raw_qualified_source_envelope'].get('total_upper_minus_feasible'))
    print('REPORT',target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
