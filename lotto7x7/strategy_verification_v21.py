"""v2.1: exact full C(49,6) candidate-universe envelope under FROZEN M0 ranks.

Enumerate all 13,983,816 sorted legal six-tuples, once, in bounded-size
batches. Only candidates satisfying frozen four-positive score requirements
(and immutable original strategy exclusions) are eligible for the optimizer.

This is an exact combinatorial optimization audit, not a forecasting claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations, islice
from math import comb, fsum
from pathlib import Path
import json
import numpy as np

from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import classification_keys
from .strategy_verification_v12 import (
    verify_full_m0, OLD_TICKETS, LEVELS, WEIGHTS, JOINT_SHA,
)
from .strategy_verification_v17 import score_uniform_stream
from .strategy_verification_v18 import REFERENCE_RANK_AXES_SHA, SOURCE_PREFIX_SHA

SCHEMA='lotto649.strategy_verification.v2.1.full_universe.v1'
UNIVERSE=comb(49,6)
EXPECTED_REFERENCE=13704
# Research-only feasible comparator from the immutable v2.0 report; NOT a bet.
RESEARCH_BASELINE=(
    (12,26,32,34,37,45),
    (15,24,29,31,38,49),
    (16,22,28,33,39,48),
    (11,20,35,36,40,44),
)
BASELINE_QUALITY=(43549,195617,43549,54184)
BASELINE_RANK_SUMS=(48644,49240,54184,43549)


def _validate_reference(reference:dict)->dict:
    axes=reference['sorted_qualified_gains']
    sha=sha256(json.dumps(axes,separators=(',',':')).encode('ascii')).hexdigest()
    if sha!=REFERENCE_RANK_AXES_SHA or len(axes)!=4 or any(len(v)!=EXPECTED_REFERENCE for v in axes):
        raise AssertionError('FAIL_CLOSED_FROZEN_REFERENCE_SHA_OR_SIZE')
    cuts=np.asarray(reference['integer_product_thresholds'],dtype=np.int64)
    if len(cuts)<3 or np.any(cuts[1:]<=cuts[:-1]):
        raise ValueError('invalid frozen product bins')
    ref={
        'axes':[np.asarray(x,dtype=np.float64) for x in axes],
        'cuts':cuts,
        # Exact gain numerator magnitudes exceed signed int64. Preserve their
        # mathematically exact Python-int signs, and compute reference gains
        # using the identical sequential x / denominator / sd evaluation
        # employed by the original v1.2 score tables. Never overflow to int64.
        'sum_positive':np.asarray([x>0 for x in reference['sum_gain_numerators']],dtype=bool),
        'log_positive':np.asarray([x>0 for x in reference['log_gain_numerators']],dtype=bool),
        'joint_positive':np.asarray([x>0 for x in reference['joint_gain_numerators']],dtype=bool),
        'sum_score':np.asarray([x/reference['sum_denominator']/reference['sum_sd'] for x in reference['sum_gain_numerators']],dtype=np.float64),
        'log_score':np.asarray([x/reference['log_denominator']/reference['log_sd'] for x in reference['log_gain_numerators']],dtype=np.float64),
        'joint_score':np.asarray([x/reference['joint_denominator']/reference['joint_sd'] for x in reference['joint_gain_numerators']],dtype=np.float64),
        'shape_tables':reference['shape_gain_tables'],
        'shape_sd':reference['shape_sd'],
        'edge_cache':{},
        'joint_cols':len(cuts)-1,
        'rank_axes_sha':sha,
    }
    if (len(ref['sum_positive'])!=259 or len(ref['log_positive'])!=ref['joint_cols'] or
        len(ref['joint_positive'])!=259*ref['joint_cols'] or
        not all(np.isfinite(ref[x]).all() for x in ('sum_score','log_score','joint_score'))):
        raise ValueError('inconsistent frozen score dimensions')
    return ref


def _edge_shape_score(edge:int,component:int,part:int,ref:dict)->tuple[float,int]:
    """Compute exact original fsum weighted L1/L2/L3 gains once per shape code."""
    cache=ref['edge_cache']
    if edge not in cache:
        cats={'component':component,'partition':part,'edge':edge}
        score=fsum(WEIGHTS[level]*(ref['shape_tables'][level][cats[level]] / ref['shape_sd'][level])
                   for level in LEVELS)
        rank=int(np.searchsorted(ref['axes'][3],score,side='left'))
        cache[edge]=(float(score),rank)
    return cache[edge]


def score_batch(numbers:np.ndarray, ref:dict, *, disallow:set[tuple[int,...]]=OLD_TICKETS,
                return_detail:bool=False)->tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """Vectorized exact-input screening; run Shape only after cheap 3-block gate."""
    a=np.asarray(numbers,dtype=np.uint8)
    if a.ndim!=2 or a.shape[1]!=6 or not len(a) or np.any(a<1) or np.any(a>49) or np.any(a[:,1:]<=a[:,:-1]):
        raise ValueError('expected sorted strict legal [N,6] combinations')
    s=a.sum(axis=1,dtype=np.int64)-21
    p=np.prod(a.astype(np.int64),axis=1)
    l=np.searchsorted(ref['cuts'],p,side='right')-1
    if np.any(l<0) or np.any(l>=ref['joint_cols']):
        raise AssertionError('sum/log bins outside original support')
    j=s*ref['joint_cols']+l
    pre=ref['sum_positive'][s]&ref['log_positive'][l]&ref['joint_positive'][j]
    chosen=a[pre]; sis=s[pre]; lis=l[pre]; jis=j[pre]
    pre_count=int(np.count_nonzero(pre))
    if not len(chosen):
        empty=np.empty((0,6),dtype=np.uint8)
        return empty,np.empty((0,4),dtype=np.int32),np.empty((0,),dtype=np.int32),{
            'raw_count':len(a),'triple_positive':0,'shape_positive':0,'excluded':0,'qualified':0}
    cats=classification_keys(classify_numbers(chosen))
    ec=cats['edge'].astype(np.int64)
    unique,first,inverse=np.unique(ec,return_index=True,return_inverse=True)
    eranks=np.empty(len(unique),dtype=np.int32)
    egains=np.empty(len(unique),dtype=np.float64)
    for z,(code,i) in enumerate(zip(unique,first)):
        gain,rank=_edge_shape_score(int(code),int(cats['component'][i]),int(cats['partition'][i]),ref)
        egains[z]=gain; eranks[z]=rank
    sg=egains[inverse]; shape_keep=sg>0
    excluded=0
    if disallow:
        # Immutable rare exclusions are applied only AFTER four objective signs.
        # Compare a 6-digit base-50 exact integer key to avoid quadratic sets.
        vals=chosen[shape_keep]
        keys=np.zeros(len(vals),dtype=np.int64)
        for col in range(6):keys=keys*50+vals[:,col]
        blocked=np.asarray([sum(int(v)*50**(5-k) for k,v in enumerate(t)) for t in disallow],dtype=np.int64)
        remaining=~np.isin(keys,blocked)
        excluded=int(np.count_nonzero(~remaining))
        ids=np.flatnonzero(shape_keep)[remaining]
    else:
        ids=np.flatnonzero(shape_keep)
    nums=chosen[ids]
    parts=cats['partition'][ids].astype(np.int32)
    ranks=np.empty((len(nums),4),dtype=np.int32)
    gains=[ref['sum_score'][sis[ids]],ref['log_score'][lis[ids]],
           ref['joint_score'][jis[ids]],sg[ids]]
    for k in range(4):
        ranks[:,k]=np.searchsorted(ref['axes'][k],gains[k],side='left').astype(np.int32)
    details={'raw_count':len(a),'triple_positive':pre_count,
        'shape_positive':int(np.count_nonzero(shape_keep)),'excluded':excluded,'qualified':len(nums)}
    if return_detail:
        details['gains']=np.column_stack(gains)
    return nums,ranks,parts,details


def rank_quality(rows:list[dict])->tuple[tuple[int,int,int,int],list[int]]:
    if len(rows)!=4:raise ValueError('four tickets required')
    masks=[];parts=[]
    for row in rows:
        t=tuple(row['ticket'])
        if len(t)!=6 or len(set(t))!=6 or tuple(sorted(t))!=t or t[0]<1 or t[-1]>49:
            raise ValueError('bad ticket')
        masks.append(sum(1<<(v-1) for v in t));parts.append(row['part'])
    if len(set(parts))!=4 or sum(v.bit_count() for v in masks)!=(masks[0]|masks[1]|masks[2]|masks[3]).bit_count():
        raise ValueError('part/number overlap in portfolio')
    sums=[sum(int(row['rank'][k]) for row in rows) for k in range(4)]
    return (min(sums),sum(sums),sums[3],sums[2]),sums


def _ticket_record(numbers:np.ndarray,ranks:np.ndarray,part:int,i:int)->dict:
    return {'ticket':tuple(map(int,numbers[i])),'part':int(part),
            'rank':tuple(int(v) for v in ranks[i])}


def accumulate_envelopes(groups:dict[int,dict],nums:np.ndarray,ranks:np.ndarray,parts:np.ndarray)->None:
    """Maintain shape cap, per-ticket total cap, and ALL total-cap ties per partition."""
    if len(nums)!=len(ranks) or len(ranks)!=len(parts):raise ValueError('length mismatch')
    for part in np.unique(parts):
        p=int(part); ids=np.flatnonzero(parts==part)
        sub=ranks[ids]
        max_shape=int(sub[:,3].max())
        ix=ids[sub[:,3]==max_shape]
        totals=ranks[ix].sum(axis=1,dtype=np.int64)
        best_total=int(totals.max())
        top=ix[totals==best_total]
        prior=groups.get(p)
        if prior is None or (max_shape,best_total)>(prior['shape_cap'],prior['total_cap']):
            groups[p]={'shape_cap':max_shape,'total_cap':best_total,'ties':[
                _ticket_record(nums,ranks,p,int(i)) for i in top]}
        elif (max_shape,best_total)==(prior['shape_cap'],prior['total_cap']):
            prior['ties'].extend(_ticket_record(nums,ranks,p,int(i)) for i in top)


def find_distinct_witness(groups:list[list[dict]], *,min_primary:int,
                          max_nodes:int=2_000_000, joint_tie_cap:int|None=None)->tuple[list[dict]|None,int,bool]:
    """Seek full coupled-total-bound legal witness; limit is explicit, never silently prove negative."""
    if len(groups)!=4 or any(not group for group in groups):raise ValueError('four nonempty groups')
    order=sorted(range(4),key=lambda i:len(groups[i]))
    sequences=[sorted(groups[i],key=lambda r:(-r['rank'][2],r['ticket'])) for i in order]
    out=[];nodes=0;exhausted=False
    def go(depth:int,occupied:int,block_sum:tuple[int,...])->bool:
        nonlocal nodes,exhausted
        if depth==4:
            if min(block_sum)>=min_primary and (joint_tie_cap is None or block_sum[2]==joint_tie_cap):return True
            return False
        for row in sequences[depth]:
            nodes+=1
            if nodes>max_nodes:
                exhausted=True;return False
            mask=sum(1<<(v-1) for v in row['ticket'])
            if mask & occupied:continue
            next_sum=tuple(block_sum[k]+row['rank'][k] for k in range(4))
            out.append(row)
            if go(depth+1,occupied|mask,next_sum):return True
            out.pop()
            if exhausted:return False
        return False
    ok=go(0,0,(0,0,0,0))
    return (list(out) if ok else None,nodes,not exhausted)


def summarize(groups:dict[int,dict], baseline:list[dict],total_count:int,qualified:int,
              *,max_witness_nodes:int=2_000_000)->dict:
    if total_count!=UNIVERSE:raise AssertionError('FAIL_CLOSED_PARTIAL_UNIVERSE_IS_NOT_A_PROOF')
    quality0,sums0=rank_quality(baseline)
    top=sorted(groups.items(),key=lambda it:(-it[1]['shape_cap'],it[0]))
    if len(top)<5 or top[3][1]['shape_cap']<=top[4][1]['shape_cap']:
        return {'decision':'UNIVERSE_SHAPE_TOP_FOUR_NOT_STRICT','baseline_quality':list(quality0)}
    top4=top[:4]; primary_upper=sum(v['shape_cap'] for _,v in top4)
    total_upper=sum(v['total_cap'] for _,v in top4)
    # Joint tie cap is safe *conditional on every class reaching max coupled total*.
    joint_cap=sum(max(r['rank'][2] for r in v['ties']) for _,v in top4)
    bounded=[[r for r in v['ties'] if r['rank'][2]==max(x['rank'][2] for x in v['ties'])] for _,v in top4]
    witness,nodes,finished=find_distinct_witness(bounded,min_primary=primary_upper,
                                                 joint_tie_cap=joint_cap,max_nodes=max_witness_nodes)
    fallback_nodes=0; fallback_complete=True
    if witness is None:
        witness,fallback_nodes,fallback_complete=find_distinct_witness(
            [v['ties'] for _,v in top4],min_primary=primary_upper,max_nodes=max_witness_nodes)
    score,sums=rank_quality(witness) if witness else (quality0,sums0)
    if quality0>score:score,sums=quality0,sums0;witness=baseline
    if score[0]>primary_upper or (score[0]==primary_upper and score[1]>total_upper):
        raise AssertionError('legal witness exceeds mathematically valid upper bound')
    if (score[0]==primary_upper and score[1]==total_upper and score[3]>joint_cap):
        raise AssertionError('witness exceeds classwise joint tie upper')
    primary_exact=score[0]==primary_upper
    total_exact=primary_exact and score[1]==total_upper
    joint_exact=total_exact and score[3]==joint_cap
    decision=('EXACT_FULL_UNIVERSE_LEXICOGRAPHIC_OPTIMUM' if joint_exact else
              'EXACT_UNIVERSE_PRIMARY_AND_TOTAL_ONLY' if total_exact else
              'EXACT_UNIVERSE_PRIMARY_GAP_UNRESOLVED_TOTAL' if primary_exact else
              'FULL_UNIVERSE_OPTIMIZATION_GAP_UNRESOLVED')
    return {
        'decision':decision,'full_legal_six_tuples':total_count,'qualified_candidates':qualified,
        'distinct_shape_partitions':len(groups),'primary_upper':primary_upper,
        'total_coupled_upper_given_primary':total_upper,
        'joint_upper_given_primary_and_total':joint_cap,
        'baseline_quality':list(quality0),'best_feasible_quality':list(score),
        'best_feasible_rank_sums':sums,
        'primary_certified':primary_exact,'total_certified':total_exact,'joint_tiebreak_certified':joint_exact,
        'total_upper_minus_best_feasible':total_upper-score[1] if primary_exact else None,
        'best_witness':[list(r['ticket']) for r in witness],
        'search_nodes':nodes+fallback_nodes,'search_within_budget':finished and fallback_complete,
        'top_classes':[{'part':p,'shape_rank_cap':v['shape_cap'],
                        'maximum_coupled_ticket_total':v['total_cap'],
                        'ties_at_coupled_total':len(v['ties']),
                        'joint_rank_max_among_total_ties':max(r['rank'][2] for r in v['ties'])}
                       for p,v in top4],
        'fifth_class_shape_cap':top[4][1]['shape_cap'],
        'limitations':['Exact finite all-six-tuple scan under fixed historical score and original strategy exclusions.',
                       'All fair 6/49 combinations were scanned, but only positive-qualified candidates are selectable.',
                       'Candidate rankings use original 13,704-reference empirical axes, not newly refitted ranks.',
                       'No prediction of next draw, odds or payout is implied.'],
    }


def scan_full_universe(ref:dict,baseline:list[dict],*,batch_size:int=60000,
                       allow_partial:bool=False,max_items:int|None=None,
                       max_witness_nodes:int=2_000_000, progress=None)->dict:
    if not 1000<=batch_size<=200000:raise ValueError('unsafe batch size')
    if max_items is not None and (not allow_partial or not 1<=max_items<UNIVERSE):
        raise ValueError('partial enumeration is TEST-ONLY and can NEVER certify')
    limit=UNIVERSE if max_items is None else max_items
    it=combinations(range(1,50),6)
    groups={};qualified=0;prepositive=0;shape_positive=0;excluded=0;seen=0
    rawsha=sha256();qualifiedsha=sha256()
    hist=np.zeros((259,ref['joint_cols']),dtype=np.int64)
    for start in range(0,limit,batch_size):
        batch=np.asarray(list(islice(it,min(batch_size,limit-start))),dtype=np.uint8)
        if not len(batch):raise AssertionError('universe enumeration exhausted unexpectedly')
        # Stable, reference-independent sum/log universe checksum reuses the original
        # exact theoretical distribution; the two implementations must agree.
        ss=batch.sum(axis=1,dtype=np.int64)-21
        pp=np.prod(batch.astype(np.int64),axis=1)
        ll=np.searchsorted(ref['cuts'],pp,side='right')-1
        if np.any(ll<0) or np.any(ll>=ref['joint_cols']):raise AssertionError('universe joint bin')
        np.add.at(hist,(ss,ll),1)
        rawsha.update(batch.tobytes(order='C'))
        chosen,ranks,parts,d=score_batch(batch,ref)
        qualified+=d['qualified'];prepositive+=d['triple_positive']
        shape_positive+=d['shape_positive'];excluded+=d['excluded']
        qualifiedsha.update(chosen.tobytes(order='C'))
        if len(chosen):accumulate_envelopes(groups,chosen,ranks,parts)
        seen+=len(batch)
        if progress and (start//batch_size)%20==0:progress(f'[FULL_UNIVERSE] {seen}/{limit} qualified={qualified}')
    if allow_partial:
        return {'test_only_partial':True,'rows_scanned':seen,'qualified':qualified,
                'certification':'PROHIBITED_FOR_PARTIAL_ENUMERATION'}
    if seen!=UNIVERSE or (next(it,None) is not None):
        raise AssertionError('FAIL_CLOSED_UNIVERSE_COUNT_OR_END')
    theorysha=sha256(json.dumps(hist.tolist(),separators=(',',':')).encode('ascii')).hexdigest()
    if theorysha!=JOINT_SHA or int(hist.sum())!=UNIVERSE:
        raise AssertionError('FAIL_CLOSED_THEORY_JOINT_SHA_MISMATCH')
    result=summarize(groups,baseline,seen,qualified,max_witness_nodes=max_witness_nodes)
    result.update({'raw_enumeration_sha256':rawsha.hexdigest(),
                   'qualified_enumeration_sha256':qualifiedsha.hexdigest(),
                   'independent_fair_joint_theory_sha256':theorysha,
                   'pre_shape_positive_candidates':prepositive,
                   'four_positive_before_frozen_exclusions':shape_positive,
                   'frozen_exclusions_in_eligible':excluded,
                   'max_batch_size':batch_size})
    return result


def check_frozen_exclusion_counterfactual(ref:dict,full:dict)->dict:
    """Bound effect of contract-only exclusions on the ALL positive-score universe.

    No current strategy exclusion is changed. If the seven historically frozen
    exceptions cannot increase the mandatory class shape/total/joint envelopes,
    the same exact optimum also holds when those exceptions are admitted as
    otherwise ordinary four-positive candidates. This audits the strength of
    the full-universe claim without altering source rules or tracking tickets.
    """
    rows=np.asarray(sorted(OLD_TICKETS),dtype=np.uint8)
    vals,ranks,parts,detail=score_batch(rows,ref,disallow=set())
    mandatory={int(g['part']):g for g in full.get('top_classes',[])}
    fourth=min((int(g['shape_rank_cap']) for g in mandatory.values()),default=-1)
    risks=[]
    for nums,score,part in zip(vals,ranks,parts):
        p=int(part); sh=int(score[3]); total=int(score.sum());joint=int(score[2]);
        cap=mandatory.get(p)
        violates=(sh>=fourth if cap is None else
                  sh>cap['shape_rank_cap'] or
                  (sh==cap['shape_rank_cap'] and
                   (total>cap['maximum_coupled_ticket_total'] or
                    (total==cap['maximum_coupled_ticket_total'] and
                     joint>cap['joint_rank_max_among_total_ties']))))
        if violates:risks.append({'part':p,'shape_rank':sh,'coupled_rank':total,'joint_rank':joint})
    return {'original_frozen_exclusions_examined':len(rows),
            'four_positive_excluded_candidates':len(vals),
            'could_expand_full_universe_best_class_envelope':bool(risks),
            'unsafe_envelope_cases':risks,
            'same_optimum_even_without_original_exclusions':not bool(risks) and
                 full.get('decision')=='EXACT_FULL_UNIVERSE_LEXICOGRAPHIC_OPTIMUM',
            'note':'Counterfactual mathematical envelope check only; original frozen eligibility contract is unchanged.'}


def run_universe_audit(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                       batch_size:int=60000,max_witness_nodes:int=2_000_000,progress=None)->dict:
    gate=verify_full_m0(csv_root=csv_root,bin_root=bin_root,manifest_path=manifest,
        golden_path=golden,include_qualified_candidates=True,
        include_frozen_score_reference=True)
    if (gate['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or
        gate['failed_checks'] or len(gate['checks'])!=152 or
        gate['source']['historical_prefix_sha256']!=SOURCE_PREFIX_SHA):
        raise AssertionError('FAIL_CLOSED_ORIGINAL_M0_152_CHECKS')
    reference=gate.pop('frozen_score_reference')
    original=gate.pop('qualified_candidates')
    ref=_validate_reference(reference)
    if len(original)!=EXPECTED_REFERENCE:
        raise AssertionError('FAIL_CLOSED_FROZEN_CANDIDATE_COUNT')
    parity=score_batch(np.asarray([r['ticket'] for r in original],dtype=np.uint8),ref)
    nums,ranks,parts,detail=parity
    expected={(tuple(r['ticket']),int(r['part']),tuple(r['rank'])) for r in original}
    observed={(tuple(map(int,t)),int(p),tuple(int(x) for x in v)) for t,p,v in zip(nums,parts,ranks)}
    if observed!=expected or len(nums)!=EXPECTED_REFERENCE:
        raise AssertionError('FAIL_CLOSED_ALL_13704_FOUR_RANK_AND_SHAPE_PARITY')
    bn,br,bp,bd=score_batch(np.asarray(RESEARCH_BASELINE,dtype=np.uint8),ref)
    if len(bn)!=4:raise AssertionError('research control candidate missing from source')
    baseline=[_ticket_record(bn,br,int(bp[i]),i) for i in range(4)]
    baseline_quality,baseline_sums=rank_quality(baseline)
    if baseline_quality!=BASELINE_QUALITY or tuple(baseline_sums)!=BASELINE_RANK_SUMS:
        raise AssertionError('FAIL_CLOSED_PREVIOUS_SOURCE_WITNESS_DRIFT')
    full=scan_full_universe(ref,baseline,batch_size=batch_size,
                            max_witness_nodes=max_witness_nodes,progress=progress)
    full['exclusion_counterfactual']=check_frozen_exclusion_counterfactual(ref,full)
    decision=full['decision']
    return {'schema':SCHEMA,
            'decision':decision,
            'source_gate':{'historical_draws':gate['source']['historical_prefix_draws'],
               'historical_prefix_sha256':SOURCE_PREFIX_SHA,'v12_checks':152,
               'frozen_reference_candidates':EXPECTED_REFERENCE,'rank_axes_sha256':ref['rank_axes_sha'],
               'reference_rescoring':'PASS_ALL_13704_RANKS_AND_SHAPES',
               'research_source_witness':'PASS_INDEPENDENT_INTEGER_SCORE_PARITY'},
            'universe':full,
            'notes':['Exact full-universe exhaustive independent six-tuple scan under the original fixed rank reference.',
                     'No changes to frozen historical data or previously sealed studies.',
                     'Objective optimization does not imply increased lottery forecast performance.']}


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--batch-size',type=int,default=60000)
    p.add_argument('--max-witness-nodes',type=int,default=2_000_000)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    path=args.out.resolve()
    if not path.is_relative_to(Path('results').resolve()) or path.exists():
        p.error('new exclusive report under results/ required')
    result=run_universe_audit(csv_root=args.csv_root,bin_root=args.bin_root,
         manifest=args.manifest,golden=args.golden,batch_size=args.batch_size,
         max_witness_nodes=args.max_witness_nodes,progress=print)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='ascii',newline='\n') as f:
        json.dump(result,f,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        f.write('\n')
    u=result['universe']
    print('V21 DECISION',result['decision'])
    print('UNIVERSE',u['full_legal_six_tuples'],'QUALIFIED',u['qualified_candidates'],
          'PRIMARY',u['primary_upper'],'TOTAL_UPPER',u['total_coupled_upper_given_primary'],
          'JOINT_UPPER',u['joint_upper_given_primary_and_total'])
    print('FOUND',u['best_feasible_quality'],'EXACT_CERTIFICATES',
          u['primary_certified'],u['total_certified'],u['joint_tiebreak_certified'])
    print('REPORT',path)
    # Audit completion and optimization certification are distinct outcomes.
    # A complete, valid scan with an unresolved gap is still a PASS audit,
    # explicitly labeled as uncertified in the JSON.
    return 0


if __name__=='__main__':
    raise SystemExit(main())
