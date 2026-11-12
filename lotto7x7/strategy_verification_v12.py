"""Independent v0.28 M0 *bounded-strategy* replay, not a forecast.

Reconstructs exact whole-universe references, historical block-score tables,
seeded finite candidate screening, rank pools, and bounded 4-bet selection.
Does not call the original v0.28 multiobjective score or optimizer modules.
Uses the frozen v0.15 taxonomy and v0.27 Decimal bin definition as inputs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import date
from hashlib import sha256
from itertools import combinations, islice
from math import comb, fsum, prod, sqrt
from pathlib import Path
from random import Random
import json

import numpy as np

from .geometry_convergence_v029 import audit_repository_inputs
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import RULE_VERSION, classification_keys
from .log_product_v027 import _grid
from .temporal_dataset_v024 import load_csv_history
from .strategy_verification_v11 import verify_m0_frozen_manifest

SCHEMA = 'lotto649.strategy_verification.v1.2.m0.v1'
SAMPLE_N = 80000
POOL_CAP = 2000
RESTARTS = 40
SEED = 20261110
HISTORY_N = 2497
CUTOFF_ID = 115000103
CUTOFF_DATE = 20261106
PREFIX_SHA = 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
JOINT_SHA = '31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a'
SHAPE_SHA = '341e33d37429c0f1e53c20ed4d54164c6adefa6080f54dc08c5ca9817967f320'
RAW_SAMPLE_SHA = '5744bb7b6e5ee9e086cb2c9eb17b7872b3028d512365610fb747672586b92c8c'
BLOCKS = ('sum','log','joint','shape')
LEVELS = ('component','partition','edge')
WEIGHTS = {'component':.25,'partition':.35,'edge':.40}
OLD_TICKETS = frozenset((
    (4,9,17,24,29,42),(10,13,14,27,28,34),(6,12,19,20,44,48),
    (3,7,18,33,40,47),(1,6,20,39,41,45),(2,5,8,42,47,48),
    (3,4,7,43,46,49),(14,28,31,32,36,37),(1,5,23,35,42,46),
    (3,8,11,20,34,40),(13,22,26,38,39,48),(19,25,30,31,36,37),
))


def prefix_sha256(ids: np.ndarray, dates: np.ndarray, numbers: np.ndarray) -> str:
    digest = sha256()
    for did, when, row in zip(ids, dates, numbers):
        obj = [int(when), int(did)] + [int(v) for v in row]
        digest.update((json.dumps(obj, separators=(',', ':'))+'\n').encode('ascii'))
    return digest.hexdigest()


def derive_fair_counts(thresholds: list[int], *, batch_size: int = 80000) -> tuple[np.ndarray, dict[str, dict[int,int]], dict]:
    """Enumerate each 6-of-49 combination exactly once, without M0 scorer."""
    if len(thresholds) < 3 or any(a >= b for a,b in zip(thresholds,thresholds[1:])):
        raise ValueError('non-increasing theoretical bin thresholds')
    if not 1000 <= batch_size <= 300000:
        raise ValueError('batch size outside bounded range')
    joint = np.zeros((259,len(thresholds)-1),dtype=np.int64)
    shapes: dict[str,Counter] = {k:Counter() for k in LEVELS}
    it = combinations(range(1,50),6)
    total = 0
    while True:
        portion = list(islice(it,batch_size))
        if not portion:
            break
        rows = np.asarray(portion,dtype=np.uint8)
        sums = rows.sum(axis=1, dtype=np.int64)-21
        products = np.prod(rows.astype(np.int64),axis=1)
        bins = np.searchsorted(thresholds,products,side='right')-1
        if np.any(sums < 0) or np.any(sums >= 259) or np.any(bins < 0) or np.any(bins >= joint.shape[1]):
            raise AssertionError('full-universe class outside support')
        np.add.at(joint,(sums,bins),1)
        categories = classification_keys(classify_numbers(rows))
        for level in LEVELS:
            u,c=np.unique(categories[level],return_counts=True)
            shapes[level].update({int(a):int(b) for a,b in zip(u,c)})
        total += len(rows)
    if total != comb(49,6) or int(joint.sum()) != total:
        raise AssertionError('exact universe enumeration incomplete')
    joint_sha=sha256(json.dumps(joint.tolist(),separators=(',',':')).encode('ascii')).hexdigest()
    reference={
        'schema':RULE_VERSION,
        'rule_hash_sha256':sha256(RULE_VERSION.encode('ascii')).hexdigest(),
        'total':total,'complete':True,
        'counts':{k:{str(x):int(v) for x,v in sorted(shapes[k].items())} for k in LEVELS}
    }
    shape_sha=sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()
    if joint_sha!=JOINT_SHA or shape_sha!=SHAPE_SHA:
        raise AssertionError(f'original frozen fair theory mismatch: joint={joint_sha}, shape={shape_sha}')
    return joint, {k:dict(shapes[k]) for k in LEVELS}, {
        'full_universe_count':total,'joint_sha256':joint_sha,
        'shape_sha256':shape_sha,'shape_category_counts':{k:len(shapes[k]) for k in LEVELS},
        'joint_nonzero_cells':int(np.count_nonzero(joint)),
    }


def integer_gain_numerators(observed:list[int], fair:list[int]) -> tuple[list[int],int]:
    """Separate integer-rational rederivation of squared-L2 one-append gain."""
    if len(observed)!=len(fair) or not observed or min(observed)<0 or min(fair)<0:
        raise ValueError('invalid count vectors')
    n=sum(observed)
    m=sum(fair)
    if n<1 or m<1:
        raise ValueError('empty histogram')
    a=[m*h-n*c for h,c in zip(observed,fair)]
    b=[m*h-(n+1)*c for h,c in zip(observed,fair)]
    oldsum=sum(v*v for v in a)
    newsum=sum(v*v for v in b)
    common=m*m*n*n*(n+1)*(n+1)
    numerator=[oldsum*(n+1)**2-n*n*(newsum+2*m*x+m*m) for x in b]
    return numerator,common


def weighted_sd(values: list[float], counts: list[int]) -> float:
    if not values or len(values)!=len(counts) or sum(counts)<=0:
        raise ValueError('bad weighted reference')
    total=sum(counts)
    mean=fsum(float(x)*int(c) for x,c in zip(values,counts))/total
    var=fsum(int(c)*(float(x)-mean)**2 for x,c in zip(values,counts))/total
    if var<=0:
        raise ValueError('constant score cannot be standardized')
    return sqrt(var)


def shape_gain_table(history_counts:dict[int,int], fair_counts:dict[int,int], n:int, *,alpha:float=4.) -> dict[int,float]:
    """Independent before/after regularized L2 objective for each class."""
    keys=sorted(fair_counts)
    probs={c:fair_counts[c]/comb(49,6) for c in keys}
    den={c:probs[c]+alpha/n for c in keys}
    before=fsum((history_counts.get(c,0)/n-probs[c])**2/den[c] for c in keys)
    afterbase=fsum((history_counts.get(c,0)/(n+1)-probs[c])**2/den[c] for c in keys)
    result={}
    for c in keys:
        h=history_counts.get(c,0)
        p=probs[c]
        delta=((h+1)/(n+1)-p)**2-(h/(n+1)-p)**2
        result[c]=before-afterbase-delta/den[c]
    return result


def draw_sample(seed:int=SEED,count:int=SAMPLE_N) -> list[tuple[int,...]]:
    rng=Random(seed)
    seen=set()
    while len(seen)<count:
        seen.add(tuple(sorted(rng.sample(range(1,50),6))))
    return sorted(seen)


def candidate_sample_hash(rows:list[tuple[int,...]]) -> str:
    h=sha256()
    for t in rows: h.update(bytes(t))
    return h.hexdigest()


def _feasible(chosen:list[dict], candidate:dict, omit:int=-1) -> bool:
    occupied=0
    parts=set()
    for j,other in enumerate(chosen):
        if j==omit:
            continue
        occupied |= other['mask']
        parts.add(other['part'])
    return not candidate['mask'] & occupied and candidate['part'] not in parts


def _quality(rows:list[dict],style:str) -> tuple[int,...]:
    rank_sum=tuple(sum(c['rank'][i] for c in rows) for i in range(4))
    low=min(rank_sum)
    if style=='balanced': return (low,sum(rank_sum),rank_sum[3],rank_sum[2])
    if style=='shape':return (rank_sum[3],low,sum(rank_sum),rank_sum[2])
    if style=='joint':return (rank_sum[2],low,sum(rank_sum),rank_sum[3])
    raise ValueError('unknown style')


def replay_bounded_search(pool:list[dict],style:str,restarts:int) -> tuple[list[dict],dict]:
    """Second implementation of fixed seeded greedy/multi-start/exchange contract."""
    sorted_pool=sorted(pool,key=lambda row:(-min(row['rank']),-sum(row['rank']),row['ticket']))
    rng=Random(SEED+{'balanced':0,'shape':1,'joint':2}[style])
    leaders=sorted_pool[:min(200,len(sorted_pool))]
    best=None
    best_key=None
    for restart in range(restarts):
        first=leaders[restart%len(leaders)] if restart<len(leaders) else rng.choice(leaders)
        chosen=[first]
        for _ in range(3):
            options=[c for c in pool if _feasible(chosen,c)]
            if not options:break
            highest=max(_quality(chosen+[c],style) for c in options)
            addition=min((c for c in options if _quality(chosen+[c],style)==highest),key=lambda c:c['ticket'])
            chosen.append(addition)
        if len(chosen)!=4:continue
        for _ in range(2):
            any_changed=False
            for slot in range(4):
                base=_quality(chosen,style)
                better=[]
                for c in pool:
                    if not _feasible(chosen,c,slot) or c['ticket'] in {row['ticket'] for i,row in enumerate(chosen) if i!=slot}:
                        continue
                    proposal=chosen.copy();proposal[slot]=c
                    q=_quality(proposal,style)
                    if q>base:better.append((q,c))
                if better:
                    score=max(q for q,c in better)
                    chosen[slot]=min((c for q,c in better if q==score),key=lambda c:c['ticket'])
                    any_changed=True
            if not any_changed:break
        q=_quality(chosen,style)
        normalized=tuple(sorted(c['ticket'] for c in chosen))
        if best is None or q>best_key or (q==best_key and normalized<tuple(sorted(c['ticket'] for c in best))):
            best=chosen.copy();best_key=q
    if best is None:raise ValueError('no feasible portfolio in fixed pool')
    best=sorted(best,key=lambda c:c['ticket'])
    sums=[sum(c['rank'][i] for c in best) for i in range(4)]
    return best,{'rank_vector_sum':sums,'maximin_rank_total':min(sums),'total_rank_sum':sum(sums),'restarts':restarts}


def one_swap_improvements(pool:list[dict],chosen:list[dict],style:str) -> dict:
    score=_quality(chosen,style)
    best=None
    for slot in range(4):
        for item in pool:
            if item['ticket'] in {a['ticket'] for j,a in enumerate(chosen) if j!=slot}:continue
            if not _feasible(chosen,item,slot):continue
            trial=chosen.copy();trial[slot]=item
            q=_quality(trial,style)
            if q>score and (best is None or q>best['quality']):
                best={'slot':slot,'ticket':list(item['ticket']),'quality':q}
    return {'one_swap_stationary_within_fixed_pool':best is None,'first_improving_swap':best}


def verify_full_m0(*,csv_root:Path, bin_root:Path,manifest_path:Path,
                   golden_path:Path,batch_size:int=80000,include_verified_pool:bool=False) -> dict:
    _,evidence=audit_repository_inputs(csv_root,bin_root)
    data=load_csv_history(csv_root)
    if len(data.ids)!=2498 or int(data.ids[HISTORY_N-1])!=CUTOFF_ID or int(data.dates[HISTORY_N-1])!=CUTOFF_DATE:
        raise ValueError('historical cutoff unavailable or changed')
    dig=prefix_sha256(data.ids[:HISTORY_N],data.dates[:HISTORY_N],data.numbers[:HISTORY_N])
    if dig!=PREFIX_SHA:raise ValueError('historical prefix SHA mismatch')
    frozen=verify_m0_frozen_manifest(manifest_path.read_bytes())
    if frozen['decision']!='M0_FROZEN_MANIFEST_STRUCTURAL_PASS_OPTIMIZER_UNVERIFIED':
        raise ValueError('original immutable M0 manifest failed independent v1.1 gate')
    golden=json.loads(golden_path.read_text('utf-8'))
    if golden['original_report_source']['analysis_input_sha256']!=PREFIX_SHA:
        raise ValueError('unrecognized reference CI source')
    grid=_grid(49,6,'0.5')
    cuts=grid['integer_product_thresholds']
    fair_joint,fair_shapes,reference=derive_fair_counts(cuts,batch_size=batch_size)
    hist=np.asarray(data.numbers[:HISTORY_N],dtype=np.uint8)
    sums=hist.sum(axis=1,dtype=np.int64)-21
    logbins=np.searchsorted(cuts,np.prod(hist.astype(np.int64),axis=1),side='right')-1
    obs_joint=np.zeros_like(fair_joint)
    np.add.at(obs_joint,(sums,logbins),1)
    obs_shape=classification_keys(classify_numbers(hist))
    score_joint,den_joint=integer_gain_numerators(obs_joint.ravel().tolist(),fair_joint.ravel().tolist())
    score_sum,den_sum=integer_gain_numerators(np.bincount(sums,minlength=259).tolist(),fair_joint.sum(axis=1).tolist())
    score_log,den_log=integer_gain_numerators(np.bincount(logbins,minlength=fair_joint.shape[1]).tolist(),fair_joint.sum(axis=0).tolist())
    sd_sum=weighted_sd([s/den_sum for s in score_sum], fair_joint.sum(axis=1).tolist())
    sd_log=weighted_sd([s/den_log for s in score_log], fair_joint.sum(axis=0).tolist())
    sd_joint=weighted_sd([s/den_joint for s in score_joint], fair_joint.ravel().tolist())
    shape_tables={}
    shape_sd={}
    for lv in LEVELS:
        observed=dict(Counter(map(int,obs_shape[lv])))
        table=shape_gain_table(observed,fair_shapes[lv],HISTORY_N)
        shape_tables[lv]=table
        shape_sd[lv]=weighted_sd(list(table.values()),[fair_shapes[lv][k] for k in table])
    sample=draw_sample()
    sample_sha=candidate_sample_hash(sample)
    if sample_sha!=RAW_SAMPLE_SHA:raise AssertionError('original sampling algorithm drift')
    eligible=[t for t in sample if t not in OLD_TICKETS]
    categories=classification_keys(classify_numbers(np.asarray(eligible,dtype=np.uint8)))
    qualifies=[]
    reasons=Counter()
    for i,t in enumerate(eligible):
        si=sum(t)-21
        li=int(np.searchsorted(cuts,prod(t),side='right')-1)
        ji=si*fair_joint.shape[1]+li
        if score_sum[si]<=0 or score_log[li]<=0:
            reasons['marginal_nonpositive']+=1;continue
        if score_joint[ji]<=0:
            reasons['joint_nonpositive']+=1;continue
        lv_scores={lv:shape_tables[lv][int(categories[lv][i])]/shape_sd[lv] for lv in LEVELS}
        sb=fsum(WEIGHTS[lv]*lv_scores[lv] for lv in LEVELS)
        if sb<=0:
            reasons['shape_block_nonpositive']+=1;continue
        gains=(score_sum[si]/den_sum/sd_sum,score_log[li]/den_log/sd_log,
               score_joint[ji]/den_joint/sd_joint,sb)
        qualifies.append({'ticket':t,'mask':sum(1<<(v-1) for v in t),
                          'part':int(categories['partition'][i]),
                          'cats':{lv:int(categories[lv][i]) for lv in LEVELS},
                          'level_scores':lv_scores,'gains':gains})
    for k in ('marginal_nonpositive','joint_nonpositive','shape_block_nonpositive'):
        if reasons[k]!=golden['original_report_search']['candidate_rejection_reasons'][k]:
            raise AssertionError(f'candidate rejection mismatch: {k}: {reasons[k]}')
    if len(qualifies)!=golden['original_report_search']['candidate_passing_four_positive_objective_blocks']:
        raise AssertionError('candidate qualification count diverged')
    sorted_axes=[sorted(c['gains'][i] for c in qualifies) for i in range(4)]
    for c in qualifies:
        c['rank']=[bisect_left(sorted_axes[i],c['gains'][i]) for i in range(4)]
    collection={}
    leaders=sorted(qualifies,key=lambda c:(-min(c['rank']),-sum(c['rank']),c['ticket']))[:max(80,POOL_CAP-800)]
    for c in leaders:collection[c['ticket']]=c
    for axis in range(4):
        peers=sorted(qualifies,key=lambda c:(-c['rank'][axis],-min(c['rank']),c['ticket']))[:max(50,POOL_CAP//10)]
        for c in peers:collection[c['ticket']]=c
    remaining=max(0,POOL_CAP-len(collection))
    if remaining:
        choices=[c for c in qualifies if c['ticket'] not in collection]
        rng=Random(SEED+119)
        for c in rng.sample(choices,min(remaining,len(choices))):collection[c['ticket']]=c
    pool=list(collection.values())
    outcomes={}
    checks={}
    for style,key,count in [('balanced','primary',40),('shape','shape_first',20),('joint','joint_first',20)]:
        selected,metadata=replay_bounded_search(pool,style,count)
        target=golden['primary'] if key=='primary' else golden['sensitivity'][key]
        nums=[list(c['ticket']) for c in selected]
        checks[key+'_selected_tickets']=nums==target['tickets']
        checks[key+'_rank_vector']=metadata['rank_vector_sum']==target['rank_vector_sum']
        checks[key+'_maximin']=metadata['maximin_rank_total']==target['maximin_rank_total']
        checks[key+'_total_rank']=metadata['total_rank_sum']==target['total_rank_sum']
        details=[]
        for line,ref in zip(selected,target['individual']):
            measured={name:float(line['gains'][i]) for i,name in enumerate(BLOCKS)}
            expected=ref['block_gain_in_exact_IID_std_units']
            for b in BLOCKS:
                checks[f'{key}_score_{len(details)+1}_{b}']=abs(measured[b]-expected[b])<=2e-9
                checks[f'{key}_rank_{len(details)+1}_{b}']=line['rank'][BLOCKS.index(b)]==ref['block_percentile_ranks_in_frozen_sample'][b]
            for lv in LEVELS:
                checks[f'{key}_shape_level_{len(details)+1}_{lv}']=abs(line['level_scores'][lv]-ref['shape_internal_L1_L2_L3_scores'][lv])<=2e-9
            details.append({'numbers':list(line['ticket']),'scores':measured,'rank':line['rank']})
        outcomes[key]={'tickets':nums,'maximin':metadata,'selected_details':details,
                       'search_quality':one_swap_improvements(pool,selected,style)}
    checks['source_prefix_sha256']=dig==PREFIX_SHA
    checks['fair_joint_sha256']=reference['joint_sha256']==JOINT_SHA
    checks['fair_shape_sha256']=reference['shape_sha256']==SHAPE_SHA
    checks['sample_sha256']=sample_sha==RAW_SAMPLE_SHA
    checks['qualified_candidates']=len(qualifies)==13704
    checks['pool_size']=len(pool)==2000
    checks['all_strategies_24_disjoint']=all(len({v for t in outcomes[k]['tickets'] for v in t})==24 for k in outcomes)
    checks['original_manifest']=frozen['decision'].startswith('M0_FROZEN_MANIFEST_STRUCTURAL_PASS')
    # Optional read-only audited finite-pool export; default v1.2 output unchanged.
    verified_pool = None
    if include_verified_pool:
        verified_pool = [
            {'ticket': list(c['ticket']), 'part': int(c['part']),
             'rank': [int(v) for v in c['rank']]} for c in pool
        ]
    result = {
        'schema':SCHEMA,
        'decision':'M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' if all(checks.values()) else 'M0_BOUNDED_STRATEGY_REPLAY_FAIL',
        'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
        'source':{'historical_prefix_draws':HISTORY_N,'historical_prefix_sha256':dig,'archive_gate':evidence},
        'full_theory':reference,'sample':{'seed':SEED,'count':len(sample),'sha256':sample_sha,'qualified':len(qualifies),'candidate_pool':len(pool),'reject_counts':dict(reasons)},
        'replayed':outcomes,
        'original_ci_report_sha256':golden['original_ci_json_sha256'],
        'notes':['Independent reproduction of bounded M0 search, not a global optimum over all possible portfolios.',
                 'Shared canonical Shape taxonomy and frozen Log Product bin definition; scoring/ranking/search re-implemented independently.',
                 'No changes to purchased bets, original candidate manifest, P3f/P5A or historical source.',
                 'No evidence of next-draw predictive superiority claimed.'],
    }
    if include_verified_pool:
        result['verified_pool'] = verified_pool
    return result


def main(argv:list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--batch-size',type=int,default=80000)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    dest=args.out.resolve()
    if not dest.is_relative_to(Path('results').resolve()) or dest.exists():
        p.error('exclusive-create report required within results/')
    report=verify_full_m0(csv_root=args.csv_root,bin_root=args.bin_root,manifest_path=args.manifest,
                          golden_path=args.golden,batch_size=args.batch_size)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='ascii',newline='\n') as fp:
        json.dump(report,fp,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False);fp.write('\n')
    print(f"{report['decision']}: {sum(report['checks'].values())}/{len(report['checks'])} checks")
    print(f"M0: {report['sample']['qualified']} qualified; {report['sample']['candidate_pool']} in pool")
    for name in ('primary','shape_first','joint_first'):
        r=report['replayed'][name]
        print(f"{name.upper()}: maximin={r['maximin']['maximin_rank_total']} portfolio={r['tickets']}")
    print('REPORT:',dest)
    return 0 if not report['failed_checks'] else 1


if __name__=='__main__':
    raise SystemExit(main())
