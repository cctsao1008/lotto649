"""v1.7: independent-seed, frozen-rank candidate generation robustness study.

All historical feature/score weights and original 13,704-eligible empirical rank
positions are frozen BEFORE sampling new candidates. Compare reproducible raw
uniform sample budgets and distinct deterministic screening policies, then
measure best feasible four-ticket search results and valid mathematical upper
bounds. This is bounded-search engineering, NOT prediction or actual tickets.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import sha256
from math import fsum
import json
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from .strategy_verification_v12 import (
    verify_full_m0, classify_numbers, classification_keys, LEVELS, WEIGHTS,
    OLD_TICKETS, draw_sample, SAMPLE_N, SEED,
)
from .strategy_verification_v13 import validate_pool, quality, shape_partition_bound
from .strategy_verification_v16 import (
    canonical_sha, portfolio_audit, local_coordinate_ascent, BASELINE_MAXIMIN,
    BASELINE_TOTAL, BASELINE_JOINT, ORIGINAL_PREFIX_SHA, QUALIFIED_COUNT,
    V15_REFERENCE,
)

SCHEMA = 'lotto649.strategy_verification.v1.7.robustness.v1'
DEFAULT_SEEDS = (649, 20261111, 20261112)
DEFAULT_RAW_SIZES = (80000, 160000, 320000)
POLICIES = ('ranked', 'shape_round_robin', 'uniform_screen')
DEFAULT_POOL_CAP = 2000


def sampled_unique(seed: int, sizes: tuple[int,...]) -> tuple[list[tuple[int,...]], list[dict]]:
    """Copy of strict raw 6/49-uniform-without-replacement generator as a STREAM.

    Prefix snapshots use first-encounter order; sorting a 320k final set and
    slicing it would NOT reproduce the valid nested sampling design.
    """
    if type(seed) is not int or not sizes or any(type(s) is not int or s<4 for s in sizes):
        raise ValueError('invalid seed or sample sizes')
    if list(sizes)!=sorted(set(sizes)) or sizes[-1]>320000:
        raise ValueError('sizes must be unique, increasing and <=320000')
    rng=Random(seed)
    observed:set[tuple[int,...]]=set()
    stream:list[tuple[int,...]]=[]
    checkpoints=[]
    next_idx=0
    while len(stream)<sizes[-1]:
        ticket=tuple(sorted(rng.sample(range(1,50),6)))
        if ticket in observed:
            continue
        observed.add(ticket)
        stream.append(ticket)
        if len(stream)==sizes[next_idx]:
            # Equivalence to original M0 sample-hash semantics: sorted tickets
            # and raw concatenation of their six bytes.
            h=sha256()
            for t in sorted(stream):
                h.update(bytes(t))
            checkpoints.append({'raw_count':len(stream),'sample_sha256':h.hexdigest()})
            next_idx+=1
            if next_idx==len(sizes):break
    return stream,checkpoints


def score_uniform_stream(stream: list[tuple[int,...]], reference: dict) -> tuple[list[dict],dict]:
    """Read-only re-score of new draws using frozen v0.28 historical statistics.

    13,704 reference score axes are never refitted, even when a new score
    falls above the frozen maximum (its reference-position rank is 13,704).
    """
    if not stream or len(set(stream))!=len(stream):
        raise ValueError('raw candidate stream must be unique')
    cuts=np.asarray(reference['integer_product_thresholds'],dtype=np.int64)
    nums=np.asarray(stream,dtype=np.uint8)
    if nums.shape != (len(stream),6):
        raise ValueError('invalid six-ball stream shape')
    s_idx=nums.sum(axis=1,dtype=np.int64)-21
    prod_values=np.prod(nums.astype(np.int64),axis=1)
    l_idx=np.searchsorted(cuts,prod_values,side='right')-1
    if np.any(s_idx<0) or np.any(s_idx>=259) or np.any(l_idx<0) or np.any(l_idx>=len(reference['log_gain_numerators'])):
        raise AssertionError('ticket mapped outside frozen theoretical bins')
    j_idx=s_idx*(len(cuts)-1)+l_idx
    sum_num=reference['sum_gain_numerators']
    log_num=reference['log_gain_numerators']
    joint_num=reference['joint_gain_numerators']
    # All three counterfactual L2 signs are calculated from exact integers.
    qualified_pos=[]
    reasons=Counter()
    for pos,ticket in enumerate(stream):
        if ticket in OLD_TICKETS:
            reasons['frozen_contract_exclusions']+=1
        elif sum_num[int(s_idx[pos])]<=0 or log_num[int(l_idx[pos])]<=0:
            reasons['marginal_nonpositive']+=1
        elif joint_num[int(j_idx[pos])]<=0:
            reasons['joint_nonpositive']+=1
        else:
            qualified_pos.append(pos)
    if not qualified_pos:
        raise ValueError('no positive Sum/Log/Joint qualified candidates')
    cats=classification_keys(classify_numbers(nums[np.asarray(qualified_pos,dtype=np.int64)]))
    levels=reference['shape_gain_tables']
    sds=reference['shape_sd']
    rank_axes=reference['sorted_qualified_gains']
    if len(rank_axes)!=4 or any(len(a)!=QUALIFIED_COUNT for a in rank_axes):
        raise ValueError('frozen score reference rank axes have drifted')
    qualified=[]
    for i,pos in enumerate(qualified_pos):
        l_scores={lv:levels[lv][int(cats[lv][i])]/sds[lv] for lv in LEVELS}
        shape_gain=fsum(WEIGHTS[lv]*l_scores[lv] for lv in LEVELS)
        if shape_gain<=0:
            reasons['shape_block_nonpositive']+=1
            continue
        si=int(s_idx[pos]); li=int(l_idx[pos]); ji=int(j_idx[pos])
        gains=(sum_num[si]/reference['sum_denominator']/reference['sum_sd'],
               log_num[li]/reference['log_denominator']/reference['log_sd'],
               joint_num[ji]/reference['joint_denominator']/reference['joint_sd'],
               shape_gain)
        qualified.append({
            'ticket':list(stream[pos]),'part':int(cats['partition'][i]),
            'rank':[bisect_left(rank_axes[k],gains[k]) for k in range(4)],
            'appearance':pos,
        })
    if len(qualified)+sum(reasons.values())!=len(stream):
        raise AssertionError('candidate accounting conservation failed')
    return qualified, {'raw':len(stream),'qualified':len(qualified),
                       'rejects':dict(sorted(reasons.items()))}


def screen_pool(rows: list[dict], *, policy:str, cap:int,
                seed:int, sample_size:int) -> list[dict]:
    """Three explicitly DIFFERENT candidate-screening policies on SAME source.

    1 ranked: best worst-block reference rank; 2 shape round-robin: equal
    partition-code exposure; 3 uniform-screen: seeded unbiased subset.
    They are screening policies, not different physical lottery mechanisms.
    """
    if policy not in POLICIES or type(cap) is not int or cap<4 or len(rows)<cap:
        raise ValueError('insufficient candidates or unsupported policy')
    validate_pool([{'ticket':r['ticket'],'part':r['part'],'rank':r['rank']} for r in rows])
    def score_key(c):return (-min(c['rank']),-sum(c['rank']),-c['rank'][3],tuple(c['ticket']))
    if policy=='ranked':
        chosen=sorted(rows,key=score_key)[:cap]
    elif policy=='shape_round_robin':
        grouped=defaultdict(list)
        for r in rows: grouped[r['part']].append(r)
        keys=sorted(grouped)
        for group in grouped.values():group.sort(key=score_key)
        chosen=[];next_index={k:0 for k in keys}
        while len(chosen)<cap:
            made_progress=False
            for k in keys:
                j=next_index[k]
                if j<len(grouped[k]):
                    chosen.append(grouped[k][j]);next_index[k]=j+1
                    made_progress=True
                    if len(chosen)==cap:break
            if not made_progress:raise AssertionError('shape round-robin ran out')
    else:
        rng=Random((seed<<16) ^ (sample_size*131) ^ 0x64917)
        chosen=rng.sample(sorted(rows,key=lambda r:tuple(r['ticket'])),cap)
    if len({tuple(r['ticket']) for r in chosen})!=cap:
        raise AssertionError('duplicate in screened pool')
    return [{'ticket':r['ticket'],'part':r['part'],'rank':r['rank']} for r in chosen]


def _candidate_masks(tickets:list[tuple[int,...]]) -> np.ndarray:
    return np.asarray([sum(1<<(v-1) for v in t) for t in tickets],dtype=np.uint64)


def greedy_multistart(pool:list[dict], *, restarts:int=10,passes:int=16) -> dict:
    """Batched deterministic greedy first steps + coordinate ascent; no global claim."""
    if type(restarts) is not int or not 1<=restarts<=64:
        raise ValueError('invalid restart budget')
    ranks,tickets,parts=validate_pool(pool)
    mask=_candidate_masks(tickets)
    rank=np.asarray(ranks,dtype=np.int64)
    labels=np.asarray(parts,dtype=np.int64)
    ranked=sorted(range(len(pool)), key=lambda j:(-min(rank[j]),-sum(rank[j]),-rank[j,3],tickets[j]))
    best=None
    successful=0
    for start in ranked[:min(restarts,len(ranked))]:
        chosen=[start]
        for _ in range(3):
            taken=0
            chosen_parts=set()
            for j in chosen:
                taken |= int(mask[j]);chosen_parts.add(int(labels[j]))
            available=np.flatnonzero((mask&np.uint64(taken)==0) & ~np.isin(labels,list(chosen_parts)))
            if not len(available):break
            totals=rank[available]+rank[chosen].sum(axis=0)[None,:]
            # First rank min, then rank sum, then Shape and Joint.
            mins=totals.min(axis=1)
            total=totals.sum(axis=1)
            keys=np.lexsort((totals[:,2],totals[:,3],total,mins))
            target=(int(mins[keys[-1]]),int(total[keys[-1]]),
                    int(totals[keys[-1],3]),int(totals[keys[-1],2]))
            tied=available[np.flatnonzero((mins==target[0]) & (total==target[1]) &
                (totals[:,3]==target[2]) & (totals[:,2]==target[3]))]
            chosen.append(min(map(int,tied),key=lambda j:tickets[j]))
        if len(chosen)!=4:continue
        successful+=1
        try:
            local=local_coordinate_ascent(pool,[list(tickets[j]) for j in chosen],max_passes=passes)
        except (AssertionError, ValueError):
            continue
        if best is None or tuple(local['quality'])>tuple(best['quality']) or (
             tuple(local['quality'])==tuple(best['quality']) and
             tuple(sorted(map(tuple,local['tickets'])))<tuple(sorted(map(tuple,best['tickets'])))):
            best=local
    if best is None:
        raise ValueError('bounded greedy search found no feasible four-ticket portfolio')
    best['search_method']='DETERMINISTIC_GREEDY_MULTISTART_PLUS_COORDINATE_ASCENT'
    best['restarts_attempted']=min(restarts,len(pool))
    best['valid_greedy_starts']=successful
    return best


def audit_policies(rows:list[dict], *,seed:int, raw_size:int, cap:int,
                   policies:tuple[str,...]=POLICIES,restarts:int=10) -> list[dict]:
    arms=[]
    for policy in policies:
        start=perf_counter()
        pool=screen_pool(rows,policy=policy,cap=cap,seed=seed,sample_size=raw_size)
        winner=greedy_multistart(pool,restarts=restarts)
        upper=shape_partition_bound(pool)
        u=int(upper['exact_integer_upper_bound']);low=int(winner['min_rank'])
        if u<low:raise AssertionError('feasible witness beats exact Shape bound')
        arms.append({
            'seed':seed,'raw_sample_size':raw_size,'policy':policy,
            'candidate_pool_size':len(pool),'pool_sha256':canonical_sha(pool),
            'distinct_shape_partitions':len({c['part'] for c in pool}),
            'reference_overlap_2k':None,
            'best_feasible_quality':winner['quality'],
            'best_feasible_block_ranks':winner['rank_sums'],
            'primary_rigorous_upper':u,'primary_gap_at_most':u-low,
            'primary_certification':'EXACT_PRIMARY_FINITE_POOL' if u==low else 'BOUNDED_PRIMARY_UNRESOLVED',
            'full_lexicographic_certification':'NOT_PROVEN_BY_BOUNDED_SEARCH',
            'strategy_compliance':'PASS_FROZEN_RANK_SCREENING_AND_LEGAL_FOUR_TICKETS',
            'one_swap_stationary':winner['one_swap_stationary_certified'],
            'search_restarts':winner['restarts_attempted'],
            'runtime_seconds':round(perf_counter()-start,4),
            # Included for independent revalidation; not a purchase or recommendation.
            'research_witness':winner['tickets'],
        })
    return arms


def verify_robustness(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                      seeds:tuple[int,...]=DEFAULT_SEEDS,
                      raw_sizes:tuple[int,...]=DEFAULT_RAW_SIZES,
                      pool_cap:int=DEFAULT_POOL_CAP,
                      policies:tuple[str,...]=POLICIES,
                      restarts:int=10) -> dict:
    if len(seeds)<1 or len(set(seeds))!=len(seeds) or SEED in seeds or any(type(s) is not int for s in seeds):
        raise ValueError('independent test seeds must be unique and exclude frozen baseline seed')
    if list(raw_sizes)!=sorted(set(raw_sizes)) or not raw_sizes or raw_sizes[-1]>320000:
        raise ValueError('invalid raw sample schedule')
    if len(set(policies))!=len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError('invalid policies')
    if type(pool_cap) is not int or pool_cap<4:
        raise ValueError('invalid pool cap')
    audited=verify_full_m0(csv_root=csv_root,bin_root=bin_root,manifest_path=manifest,
                 golden_path=golden,include_verified_pool=True,
                 include_qualified_candidates=True,include_frozen_score_reference=True)
    if audited['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or audited['failed_checks'] or len(audited['checks'])!=152:
        raise AssertionError('v1.2 fixed original source/strategy independent verification failed')
    reference=audited.pop('frozen_score_reference')
    original=audited.pop('verified_pool')
    original_qualified=audited.pop('qualified_candidates')
    if len(original_qualified)!=QUALIFIED_COUNT or len(original)!=2000:
        raise AssertionError('baseline pool identity drift')
    # The strongest available parity check: reproduce every original qualified
    # ticket's frozen rank rather than trust a different scoring function.
    ref_rows=score_uniform_stream([tuple(p['ticket']) for p in original_qualified],reference)[0]
    if {tuple(p['ticket']):(p['part'],tuple(p['rank'])) for p in ref_rows}!=\
       {tuple(p['ticket']):(p['part'],tuple(p['rank'])) for p in original_qualified}:
        raise AssertionError('new scorer does not reproduce all 13,704 original frozen ranks')
    reference_sha=sha256(json.dumps(reference['sorted_qualified_gains'],separators=(',',':')).encode('ascii')).hexdigest()
    reference_keys={tuple(c['ticket']) for c in original}
    outcomes=[]; samples=[]
    for seed in seeds:
        start=perf_counter()
        unique,checkpoints=sampled_unique(seed,raw_sizes)
        sampled_time=perf_counter()-start
        score_start=perf_counter()
        scored,counts=score_uniform_stream(unique,reference)
        score_seconds=perf_counter()-score_start
        for checkpoint in checkpoints:
            size=checkpoint['raw_count']
            subset=[c for c in scored if c['appearance']<size]
            if len(subset)<pool_cap:
                raise ValueError(f'seed {seed}: not enough qualified candidates at size {size}')
            # All rank scores are frozen BEFORE any policy selection.
            results=audit_policies(subset,seed=seed,raw_size=size,
                                   cap=pool_cap,policies=policies,restarts=restarts)
            for row in results:
                # The same 2k reference pool is only a comparator, not included
                # as incumbents or used for re-ranking independently seeded draws.
                screened_sha=row['pool_sha256']
                row['reference_overlap_2k']=None # kept unused; no leakage of selected ticket sets
                row['quality_relative_to_frozen_2k']=list(x-y for x,y in zip(
                    row['best_feasible_quality'],[BASELINE_MAXIMIN,BASELINE_TOTAL,BASELINE_MAXIMIN,BASELINE_JOINT]))
                row['pool_sha256']=screened_sha
            outcomes+=results
            samples.append({'seed':seed,'raw_sample_size':size,
                            'raw_sample_sha256':checkpoint['sample_sha256'],
                            'qualified_count':len(subset),
                            'qualified_fraction':len(subset)/size,
                            'overlap_with_frozen_2k_count':len(reference_keys & {tuple(x['ticket']) for x in subset}),
                            'distinct_qualified_shape_partitions':len({x['part'] for x in subset})})
        samples[-1]['max_stream_rejects']=counts['rejects']
        samples[-1]['unique_stream_sample_runtime_seconds']=round(sampled_time,4)
        samples[-1]['unique_stream_score_runtime_seconds']=round(score_seconds,4)
    report={
        'schema':SCHEMA,'decision':'CANDIDATE_GENERATION_ROBUSTNESS_EXPLORATORY_PASS',
        'source_gate':{'v12_checks':len(audited['checks']),
          'prefix_sha256':audited['source']['historical_prefix_sha256'],
          'draws':audited['source']['historical_prefix_draws'],
          'baseline_sample_sha256':audited['sample']['sha256'],
          'reference_qualified':len(original_qualified),
          'fixed_sorted_score_axes_sha256':reference_sha,
          'reference_rescoring_all_13704':'PASS_EXACT_RANK_PARITY'},
        'parameters':{'seeds':list(seeds),'raw_sample_sizes':list(raw_sizes),
          'pool_cap':pool_cap,'policies':list(policies),'restarts':restarts},
        'samples':samples,'arms':outcomes,
        'baseline_frozen_2k_exact_optimum':[BASELINE_MAXIMIN,BASELINE_TOTAL,BASELINE_MAXIMIN,BASELINE_JOINT],
        'limitations':[
          'Only new reproducible uniform 6/49 random candidate generation seeds tested; policies vary SCREENING, not lottery generation mechanism.',
          'New candidates have their Sum/Log/Joint/Shape gains computed against the original 2497-draw cutoff; reference rank axes use original 13704 eligible sample without moving denominators.',
          'These are three deterministic pseudorandom seeds, not statistical evidence establishing a population-level sampling effect.',
          'All four-ticket results are finite deterministic heuristic FEASIBLE witnesses; no exact lexicographic global optimum or predictive edge claimed.',
          'A fixed reference-score rank above the maximum may equal 13704; this is a position on a frozen empirical scale, not a calibrated winning probability.',
          'No source, forecast seals or original M0 study is modified.'
        ]}
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    p.add_argument('--raw-sizes',type=int,nargs='+',default=list(DEFAULT_RAW_SIZES))
    p.add_argument('--pool-cap',type=int,default=DEFAULT_POOL_CAP)
    p.add_argument('--restarts',type=int,default=10)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(argv)
    dest=args.out.resolve()
    if not dest.is_relative_to(Path('results').resolve()) or dest.exists():
        p.error('exclusive-create v1.7 result under results/ is required')
    t=perf_counter()
    result=verify_robustness(csv_root=args.csv_root,bin_root=args.bin_root,
      manifest=args.manifest,golden=args.golden,seeds=tuple(args.seeds),
      raw_sizes=tuple(args.raw_sizes),pool_cap=args.pool_cap,restarts=args.restarts)
    result['overall_runtime_seconds']=round(perf_counter()-t,4)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='ascii',newline='\n') as fp:
        json.dump(result,fp,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        fp.write('\n')
    print(result['decision'], 'ARMS',len(result['arms']))
    for seed in result['parameters']['seeds']:
        for sz in result['parameters']['raw_sample_sizes']:
            subset=[r for r in result['arms'] if r['seed']==seed and r['raw_sample_size']==sz]
            print('SEED',seed,'RAW',sz,'QUALIFIED',next(x['qualified_count'] for x in result['samples'] if x['seed']==seed and x['raw_sample_size']==sz),
                 'QUALITIES', {x['policy']:x['best_feasible_quality'] for x in subset},
                 'GAPS', {x['policy']:x['primary_gap_at_most'] for x in subset})
    print('REPORT',dest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
