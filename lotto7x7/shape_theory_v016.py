"""P1 v0.16: exact, exhaustive 13,983,816 legal six-hot shape baseline.

The 8-neighbor graph partition and orthogonal/diagonal edge counts are
mathematically fixed classes. This module enumerates each legal combination
exactly once with pure NumPy, so no GPU/JIT/C++ build prerequisite. Simulation
is only used to CALIBRATE the global goodness-of-fit statistic under a known
exact combinatorial probability vector, never to estimate class frequencies.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, islice
from hashlib import sha256
from math import comb
import json

import numpy as np
from scipy.stats import binomtest, chi2

from .shape_taxonomy_v015 import (UNIVERSE,classify_numbers,partition_name,edge_name)

RULE_VERSION='shape8-graph-partition-cardinal-diagonal-v1'
LEVELS=('component','partition','edge')


def classification_keys(data:dict)->dict[str,np.ndarray]:
    return {'component':data['n_components'].astype(np.uint32),
            'partition':data['partition_code'].astype(np.uint32),
            'edge':data['edge_code'].astype(np.uint32)}


def category_name(level:str,code:int)->str:
    if level=='component':return f'C{int(code)}'
    if level=='partition':return partition_name(int(code))
    if level=='edge':return edge_name(int(code))
    raise ValueError('unknown level')


def exhaustive_class_counts(*,batch_size:int=120_000, progress=None,
                             max_items:int|None=None, allow_partial:bool=False)->dict:
    """Stream all C(49,6) sorted six-tuples with bounded memory.

    max_items is solely a TEST-ONLY escape hatch. Production requires exact
    UNIVERSE and rejects any partial run instead of presenting fake theory.
    """
    if batch_size<1:raise ValueError('batch_size must be positive')
    if max_items is not None and (not allow_partial or not 1<=max_items<=UNIVERSE):
        raise ValueError('partial enumeration requires explicit testing consent')
    total_target=UNIVERSE if max_items is None else max_items
    it=combinations(range(1,50),6)
    counters={k:Counter() for k in LEVELS}
    seen=0
    while seen<total_target:
        size=min(batch_size,total_target-seen)
        block=np.asarray(list(islice(it,size)),dtype=np.uint8)
        if block.size==0:raise AssertionError('iterator exhausted early')
        v=classification_keys(classify_numbers(block))
        for level,codes in v.items():
            keys,counts=np.unique(codes,return_counts=True)
            counters[level].update({int(k):int(cnt) for k,cnt in zip(keys,counts)})
        seen+=len(block)
        if progress and (seen==total_target or seen==len(block) or seen//(batch_size*10)!=(seen-len(block))//(batch_size*10)):
            progress(f'[UNIVERSE] {seen}/{total_target}')
    if seen!=total_target or (not allow_partial and seen!=UNIVERSE):
        raise AssertionError('incomplete exact enumeration')
    if not allow_partial:
        try:next(it)
        except StopIteration:pass
        else:raise AssertionError('extra combinations after universe cardinality')
    if any(sum(c.values())!=seen for c in counters.values()):
        raise AssertionError('taxonomy category counts do not sum to exact universe')
    return {'schema':RULE_VERSION,'rule_hash_sha256':sha256(RULE_VERSION.encode()).hexdigest(),
       'total':seen,'complete':bool(seen==UNIVERSE),'counts':{
           level:{str(k):int(v) for k,v in sorted(counter.items())}
           for level,counter in counters.items()}}


def _bonferroni(values:list[float])->list[float]:
    return [min(1.,len(values)*v) for v in values]


def _bh_fdr(values:list[float])->list[float]:
    n=len(values)
    if not n:return []
    q=np.empty(n,dtype=float)
    order=np.argsort(values,kind='stable')
    running=1.0
    for rank in range(n,0,-1):
        idx=int(order[rank-1])
        running=min(running,float(values[idx])*n/rank)
        q[idx]=running
    return q.tolist()


def historical_comparison(numbers:np.ndarray,baseline:dict, *,
                           replicates:int=5000, seed:int=649)->dict:
    """Observed vs exact theoretical multinomial; Monte Carlo null GOF p.

    Fixed category schema is unaffected by history. The simulated draws in
    GOF are multinomial *category counts*, not fabricated near-neighbor image
    matches. Per-category exact binomial tests use global BH correction.
    """
    if not baseline.get('complete') or baseline['total']!=UNIVERSE:
        raise ValueError('must use exact exhaustive 13,983,816 baseline')
    if replicates<99:raise ValueError('at least 99 MC replications required')
    x=np.asarray(numbers)
    n=len(x)
    enc=classification_keys(classify_numbers(x))
    result={}
    rng=np.random.default_rng(seed)
    global_ps=[]
    all_rows=[]
    for level in LEVELS:
        weights={int(k):int(v) for k,v in baseline['counts'][level].items()}
        vals=np.unique(enc[level],return_counts=True)
        obs_map={int(k):int(v) for k,v in zip(*vals)}
        if set(obs_map)-set(weights):raise AssertionError('historical code absent from uniform universe')
        keys=sorted(weights)
        p=np.asarray([weights[k]/UNIVERSE for k in keys],dtype=np.float64)
        if not np.isclose(p.sum(),1.,rtol=0,atol=1e-12):
            raise AssertionError('invalid theoretical probability total')
        observed=np.asarray([obs_map.get(k,0) for k in keys],dtype=np.int64)
        expected=n*p
        chi=float(np.sum((observed-expected)**2/expected))
        # Proper finite-sample null calibration even when expected counts <5.
        simulated=rng.multinomial(n,p,size=replicates)
        null_stats=np.sum((simulated-expected[None,:])**2/expected[None,:],axis=1)
        mc_p=(1+int(np.count_nonzero(null_stats>=chi-1e-10)))/(replicates+1)
        asymptotic=None if float(expected.min())<5 else float(chi2.sf(chi,len(keys)-1))
        rows=[]
        for k,obs,exp,prob in zip(keys,observed,expected,p):
            raw=float(binomtest(int(obs),n,float(prob),alternative='two-sided').pvalue)
            row={'level':level,'code':int(k),'class':category_name(level,int(k)),
                 'observed':int(obs),'expected':float(exp),
                 'theoretical_combinations':int(weights[k]),
                 'theoretical_probability':float(prob),
                 'observed_share':float(obs/n),
                 'observed_minus_expected':float(obs-exp),
                 'binomial_p':raw}
            rows.append(row)
            all_rows.append(row)
        result[level]={'classes':len(keys),'degrees_of_freedom':len(keys)-1,
             'chi_square':chi,'min_expected':float(expected.min()),
             'empirical_multinomial_p':mc_p,'asymptotic_chi_p_if_valid':asymptotic,
             'observed_categories':int(np.count_nonzero(observed)),
             'rows':rows}
        global_ps.append(mc_p)
    adjusted=_bonferroni(global_ps)
    for name,p in zip(LEVELS,adjusted):result[name]['bonferroni_global_p']=p
    fq=_bh_fdr([r['binomial_p'] for r in all_rows])
    for row,q in zip(all_rows,fq):row['fdr_bh_q_across_all_levels']=q
    return {'draws':n,'null':'independent fair uniform 6/49 tickets',
            'exact_universe':UNIVERSE,'simulation_replicates':replicates,
            'schema':RULE_VERSION,'levels':result,
            'notes':['Category probabilities are exact enumerative counts, not MC estimates',
                     'Empirical global p-values use multinomial simulations conditional on fixed classes',
                     'P0/P1 method exploration means nominal p-values are exploratory, not prospective confirmation',
                     'Per-category binomial tests are dependent and BH-FDR is descriptive; global Bonferroni across 3 nested levels',
                     'Observed exact shape families selected from history are not tested individually',
                     'The arbitrary number-to-grid embedding does not change fair future draw probabilities']}


def shape_family_baseline(families:list[dict], historical_count:int)->dict:
    """Analytic exact orbit weights for *selected observed* canonical families.

    Leaves all unseen exact families in a single exact complement; do not
    infer random-draw anomalies from post-selected observed families.
    """
    weights={str(f['shape_id']):int(f['theory_combinations']) for f in families}
    if len(weights)!=len(families):raise ValueError('duplicate IDs')
    used=sum(weights.values())
    if used>UNIVERSE:raise ValueError('non-disjoint orbits')
    return {'observed_shape_families':len(families),'theoretical_total_in_seen_exact_shapes':used,
            'theoretical_total_in_unseen_exact_shapes':UNIVERSE-used,
            'probability_in_seen_exact_shapes':used/UNIVERSE,
            'probability_in_unseen_exact_shapes':(UNIVERSE-used)/UNIVERSE,
            'historical_draws':historical_count,
            'note':'Seen family set is post-selected from history; observed vs seen theoretical mass is NOT an independent significance test'}
