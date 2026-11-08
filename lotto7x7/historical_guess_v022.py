"""v0.22: timed historical 6/49 exact-hit retrieval, not prediction.

Fair unordered six-number samples are generated without replacement inside each
independent trial. The frozen P1 L3 class-frequency ratio orders exactly the
same candidate pool as the uniform baseline. Outcomes are inspected ONLY after
both guess orders have been established. Each trial is capped at 10,000 guesses.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .shape_prospective_v019 import read_sources, file_sha256
from .shape_taxonomy_v015 import classify_numbers, edge_name
from .shape_theory_v016 import UNIVERSE, classification_keys

SCHEMA = 'lotto7x7.historical_guess_benchmark.v0.22'
_BASE50 = np.asarray([50**i for i in range(6)], dtype=np.uint64)


def encode_tickets(tickets: np.ndarray) -> np.ndarray:
    """Injective base-50 integer encoding for six sorted 1..49 labels."""
    a = np.asarray(tickets)
    if a.ndim != 2 or a.shape[1] != 6 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('tickets must be an integer N×6 array')
    if (a < 1).any() or (a > 49).any() or np.any(np.diff(a.astype(np.int16),axis=1) <= 0):
        raise ValueError('tickets must be strictly increasing legal six-number sets')
    return np.sum(a.astype(np.uint64) * _BASE50, axis=1, dtype=np.uint64)


def make_pool(rng: np.random.Generator, count: int, *, chunk: int = 4096) -> np.ndarray:
    """Uniformly sample distinct legal six-sets, same algorithm as v0.21.

    Each unordered six-set has equal chance to be drawn by selecting six of 49
    independent continuous random keys. Rejection of duplicates yields a
    uniform sample without replacement across one trial, not across trials.
    """
    if not 1 <= count <= 10000:
        raise ValueError('per trial count must be 1..10000')
    seen: set[tuple[int,...]] = set()
    out: list[tuple[int,...]] = []
    while len(out) < count:
        keys = rng.random((min(chunk, max(32, count-len(out))), 49))
        choices = np.sort(np.argpartition(keys, 5, axis=1)[:,:6] + 1, axis=1)
        for ticket in choices:
            tup = tuple(int(v) for v in ticket)
            if tup not in seen:
                seen.add(tup)
                out.append(tup)
                if len(out) >= count:
                    break
    return np.asarray(out,dtype=np.uint8)


def build_reference(ids:np.ndarray, numbers:np.ndarray, baseline:dict, *,
                    target:str='all', holdout_count:int=500, tau:float=384.0):
    """Build an exact historical lookup and score from TRAIN ONLY for holdout."""
    ids=np.asarray(ids,dtype=np.int64)
    nums=np.sort(np.asarray(numbers,dtype=np.uint8),axis=1)
    if len(ids)<3 or nums.shape!=(len(ids),6) or tau<=0 or not np.isfinite(tau):
        raise ValueError('invalid source/regularization')
    if target not in ('all','holdout'):
        raise ValueError('target must be all or holdout')
    if target=='holdout' and not 1 <= holdout_count <= len(ids)-2:
        raise ValueError('invalid holdout size')
    split=len(ids) - (holdout_count if target=='holdout' else 0)
    reference_ids,reference_nums=ids[split:],nums[split:]
    if target=='all':
        reference_ids,reference_nums=ids,nums
    hist_codes=encode_tickets(reference_nums)
    lookup:dict[int,list[int]]={}
    for key,did in zip(hist_codes,reference_ids):
        lookup.setdefault(int(key),[]).append(int(did))
    tab=baseline['edge']
    codes=np.asarray(tab['codes'],dtype=np.int64)
    p=np.asarray(tab['p'],dtype=np.float64)
    train_nums=nums[:split]
    g=classification_keys(classify_numbers(train_nums))['edge'].astype(np.int64)
    ix=np.searchsorted(codes,g)
    if np.any(ix>=len(codes)) or not np.array_equal(codes[ix],g):
        raise ValueError('historical class absent from fair baseline')
    counts=np.bincount(ix,minlength=len(p))
    q=(counts+tau*p)/(len(train_nums)+tau)
    ratios=q/p
    if not np.isclose(q.sum(),1.0,atol=1e-12,rtol=0):
        raise AssertionError('shape scorer not normalized')
    return {
        'lookup':lookup,
        'target_count':len(reference_ids),
        'unique_target_sets':len(lookup),
        'training_draws':len(train_nums),
        'training_last_draw_id':int(ids[split-1]),
        'target_first_draw_id':int(reference_ids[0]),
        'target_last_draw_id':int(reference_ids[-1]),
        'class_codes':codes,
        'class_ratio':ratios,
        'all_history_leakage':target=='all',
        'tau':tau,
    }


def trial(candidates:np.ndarray, reference:dict) -> dict:
    """Compare two orders over identical candidates, without peeking at hits."""
    nums=np.asarray(candidates,dtype=np.uint8)
    tags=classification_keys(classify_numbers(nums))['edge'].astype(np.int64)
    codes=reference['class_codes']
    ix=np.searchsorted(codes,tags)
    if np.any(ix>=len(codes)) or not np.array_equal(codes[ix],tags):
        raise AssertionError('candidate shape class absent from fair universe')
    scores=reference['class_ratio'][ix]
    order=np.argsort(-scores,kind='stable')
    rank=np.empty(len(nums),dtype=np.int32)
    rank[order]=np.arange(1,len(nums)+1,dtype=np.int32)
    enc=encode_tickets(nums)
    lookup=reference['lookup']
    matches=[]
    for i,key in enumerate(enc):
        found=lookup.get(int(key))
        if found:
            matches.append({
                'pool_index':i,
                'uniform_rank':i+1,
                'shape_rank':int(rank[i]),
                'numbers':tuple(int(v) for v in nums[i]),
                'historical_draw_ids':found,
                'shape_ratio':float(scores[i]),
                'edge_class':edge_name(int(tags[i])),
            })
    return {
        'hits':matches,
        'uniform_first':min((m['uniform_rank'] for m in matches),default=None),
        'shape_first':min((m['shape_rank'] for m in matches),default=None),
        'candidate_count':len(nums),
        'unique_candidates':len(set(map(int,enc))),
    }


HIT_COLUMNS=('timestamp_utc','elapsed_seconds','trial','pool_index','uniform_rank',
 'shape_rank','numbers','historical_draw_ids','shape_ratio','edge_class')
ROUND_COLUMNS=('trial','elapsed_seconds','candidates','exact_hits','uniform_first',
 'shape_first','earlier_strategy','total_guesses','total_hits')


def _paths(out:Path)->dict[str,Path]:
    p=Path(out)
    if p.suffix.lower()!='.json':raise ValueError('--out must be a .json filename')
    base=p.with_suffix('')
    return {'summary':p,'hits':Path(str(base)+'_hits.csv'),
            'rounds':Path(str(base)+'_rounds.csv')}


def _write_summary(path:Path,report:dict)->None:
    temp=path.with_suffix('.json.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
        f.flush()
    temp.replace(path)


def run_experiment(*, catalog:Path, universe:Path, out:Path,
                   minutes:float=20., per_round:int=10000, seed:int=649,
                   target:str='all', holdout_count:int=500,
                   tau:float=384., max_rounds:int=0,
                   progress_every:int=5,
                   clock:Callable[[],float]=time.perf_counter,
                   emit:Callable[[str],None]=print)->dict:
    if not np.isfinite(minutes) or minutes<=0:
        raise ValueError('minutes must be positive')
    if not 1<=per_round<=10000 or seed<0 or max_rounds<0 or progress_every<1:
        raise ValueError('invalid per-round/seed/round-limit/report interval')
    paths=_paths(out)
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('output already exists; choose a new --out basename')
    for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
    ids,nums,tables,_=read_sources(catalog,universe)
    ref=build_reference(ids,nums,tables,target=target,holdout_count=holdout_count,tau=tau)
    rng=np.random.default_rng(seed)
    expected=per_round*ref['unique_target_sets']/UNIVERSE
    # Hypergeometric without-replacement no-hit probability for the exact trial size.
    p_any=-np.expm1(np.log1p(-ref['unique_target_sets'] /
                              (UNIVERSE - np.arange(per_round,dtype=np.float64))).sum())
    start=clock()
    deadline=start+minutes*60
    summary={
     'schema':SCHEMA,
     'settings':{'requested_minutes':minutes,'per_round':per_round,
       'seed':seed,'target':target,'holdout_count':holdout_count,
       'tau':tau,'max_rounds':max_rounds,'progress_every':progress_every},
     'sources':{'catalog_sha256':file_sha256(catalog),
       'universe_sha256':file_sha256(universe),
       'historical_draw_count':len(ids),
       'training_draws':ref['training_draws'],
       'target_draw_count':ref['target_count'],
       'unique_target_sets':ref['unique_target_sets'],
       'training_last_draw_id':ref['training_last_draw_id'],
       'target_first_draw_id':ref['target_first_draw_id'],
       'target_last_draw_id':ref['target_last_draw_id']},
     'fair_reference':{'universe':UNIVERSE,
       'expected_hits_per_round':expected,
       'probability_at_least_one_per_round_exact':p_any,
       'comment':'Exact hypergeometric without-replacement probability for one trial.'},
     'stats':{'status':'RUNNING','elapsed_seconds':0.,'trials':0,
       'guesses_per_strategy':0,'matching_candidates':0,
       'historical_draw_matches':0,'unique_hit_six_sets':0,
       'unique_historical_draw_ids':0,'hit_trials':0,
       'first_hit_trial':None,'first_uniform_rank':None,
       'first_shape_rank':None,
       'first_uniform_global_guess':None,'first_shape_global_guess':None,
       'uniform_wins':0,'shape_wins':0,'ties':0,
       'sum_first_uniform_ranks_hit_trials':0,
       'sum_first_shape_ranks_hit_trials':0},
     'cautions':[
       'This compares retrieval order of the SAME candidate pool; exact hits per complete trial are identical for both strategies.',
       'Historical target=all uses full target data for shape-frequency training: in-sample retrieval, NOT prediction.',
       'Holdout target excludes last holdout_count draw records from score fitting but was historically inspected during other research.',
       'Candidate rounds are independent: a ticket may reappear in another round.',
       'All valid individual tickets have identical 6/49 fair jackpot probability.',
     ]}
    emit(f'[CONFIG] target={target}, reference={ref["unique_target_sets"]} exact six-sets '
         f'train={ref["training_draws"]} trials <= {per_round:,} guesses each; '
         f'duration={minutes} minutes; seed={seed}')
    emit(f'[IID] per {per_round:,} guesses: expected historical hits={expected:.3f}, '
         f'P(at least one)={p_any:.2%}')
    if target=='all':emit('[WARNING] Full-history shape scorer has trained on the target history. This is retrieval, not future prediction.')
    else:emit('[WARNING] Holdout excludes target rows from scorer, but this previously researched history is not a new blind prospective set.')
    st=summary['stats']
    seen_hit_sets:set[tuple[int,...]]=set()
    seen_hit_ids:set[int]=set()
    with paths['hits'].open('x',encoding='utf-8',newline='') as hf,paths['rounds'].open('x',encoding='utf-8',newline='') as rf:
        hw=csv.DictWriter(hf,fieldnames=HIT_COLUMNS);hw.writeheader();hf.flush()
        rw=csv.DictWriter(rf,fieldnames=ROUND_COLUMNS);rw.writeheader();rf.flush()
        _write_summary(paths['summary'],summary)
        stop_reason='time_limit'
        try:
            while clock()<deadline:
                if max_rounds and st['trials']>=max_rounds:
                    stop_reason='round_limit';break
                trial_n=st['trials']+1
                pool=make_pool(rng,per_round)
                x=trial(pool,ref)
                assert x['unique_candidates']==per_round
                elapsed=round(clock()-start,3)
                hit_count=len(x['hits'])
                if hit_count:
                    st['hit_trials']+=1
                    a=x['uniform_first'];b=x['shape_first']
                    st['sum_first_uniform_ranks_hit_trials']+=a
                    st['sum_first_shape_ranks_hit_trials']+=b
                    if a<b:st['uniform_wins']+=1;winner='uniform'
                    elif b<a:st['shape_wins']+=1;winner='shape_score'
                    else:st['ties']+=1;winner='tie'
                    if st['first_hit_trial'] is None:
                        st['first_hit_trial']=trial_n
                        st['first_uniform_rank']=a
                        st['first_shape_rank']=b
                        st['first_uniform_global_guess']=(trial_n-1)*per_round+a
                        st['first_shape_global_guess']=(trial_n-1)*per_round+b
                else:
                    winner='none'
                for hit in x['hits']:
                    seen_hit_sets.add(hit['numbers'])
                    seen_hit_ids.update(hit['historical_draw_ids'])
                    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
                    formatted=' '.join(f'{n:02d}' for n in hit['numbers'])
                    event={'timestamp_utc':timestamp,'elapsed_seconds':elapsed,
                           'trial':trial_n, 'pool_index':hit['pool_index'],
                           'uniform_rank':hit['uniform_rank'],
                           'shape_rank':hit['shape_rank'],
                           'numbers':formatted,
                           'historical_draw_ids':';'.join(map(str,hit['historical_draw_ids'])),
                           'shape_ratio':hit['shape_ratio'],
                           'edge_class':hit['edge_class']}
                    hw.writerow(event)
                    emit(f'[HIT] elapsed={elapsed:,.1f}s trial={trial_n} '
                         f'uniform=#{hit["uniform_rank"]:,} shape=#{hit["shape_rank"]:,} '
                         f'numbers={formatted} historic={event["historical_draw_ids"]}')
                st['trials']=trial_n
                st['guesses_per_strategy']+=per_round
                st['matching_candidates']+=hit_count
                st['historical_draw_matches']+=sum(len(e['historical_draw_ids']) for e in x['hits'])
                st['unique_hit_six_sets']=len(seen_hit_sets)
                st['unique_historical_draw_ids']=len(seen_hit_ids)
                st['elapsed_seconds']=elapsed
                rw.writerow({'trial':trial_n,'elapsed_seconds':elapsed,
                             'candidates':per_round,'exact_hits':hit_count,
                             'uniform_first':x['uniform_first'] or '',
                             'shape_first':x['shape_first'] or '',
                             'earlier_strategy':winner,
                             'total_guesses':st['guesses_per_strategy'],
                             'total_hits':st['matching_candidates']})
                hf.flush();rf.flush()
                if trial_n%progress_every==0:
                    _write_summary(paths['summary'],summary)
                    emit(f'[PROGRESS] elapsed={elapsed:.1f}s trials={trial_n} '
                         f'guesses/strategy={st["guesses_per_strategy"]:,} '
                         f'hits={st["matching_candidates"]} '
                         f'first-rank wins: random={st["uniform_wins"]} '
                         f'shape={st["shape_wins"]} ties={st["ties"]}')
        except KeyboardInterrupt:
            stop_reason='interrupted'
            emit('[STOP] KeyboardInterrupt; results saved')
        st['elapsed_seconds']=round(clock()-start,3)
        st['status']='COMPLETED' if stop_reason in ('time_limit','round_limit') else 'INTERRUPTED'
        st['stop_reason']=stop_reason
        if st['hit_trials']:
            st['mean_first_uniform_rank_hit_trials']=st['sum_first_uniform_ranks_hit_trials']/st['hit_trials']
            st['mean_first_shape_rank_hit_trials']=st['sum_first_shape_ranks_hit_trials']/st['hit_trials']
        else:
            st['mean_first_uniform_rank_hit_trials']=None
            st['mean_first_shape_rank_hit_trials']=None
        _write_summary(paths['summary'],summary)
    emit(f'[DONE] {st["status"]}; elapsed={st["elapsed_seconds"]:.2f}s '
         f'trials={st["trials"]} guesses/strategy={st["guesses_per_strategy"]:,} '
         f'hit-events={st["matching_candidates"]} unique-history-draws={st["unique_historical_draw_ids"]}')
    for label,path in paths.items():emit(f'[FILE] {label}: {path.resolve()}')
    return summary
