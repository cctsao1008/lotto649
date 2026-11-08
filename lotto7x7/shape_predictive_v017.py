"""P2 v0.17: leakage-safe prequential shape forecasts and fair-ticket tests.

All forecast features are fixed P1 graph codes and exact (49 choose 6) priors.
At index t every policy uses only history[:t]. Historical shape discovery
(P6_O1_D5) is exploratory and NOT retrospectively preregistered evidence.
Random-ticket baselines and future IID simulations test predictive value,
not whether the fixed 7x7 graphics are aesthetically interesting.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import comb, log

import numpy as np

from .shape_taxonomy_v015 import classify_numbers, edge_name
from .shape_theory_v016 import LEVELS, RULE_VERSION, UNIVERSE, classification_keys

SCHEMA='lotto7x7.p2_prequential_shape.v0.17'
RARE_CODE_NAME='P6_O1_D5'
DEFAULT_TAU=384.0


def _baseline_tables(baseline:dict):
    if (baseline.get('schema')!=RULE_VERSION or not baseline.get('complete') or
            int(baseline.get('total',0))!=UNIVERSE or not isinstance(baseline.get('counts'),dict)):
        raise ValueError('must use verified exact P1 universe data')
    tables={}
    for level in LEVELS:
        raw=baseline['counts'][level]
        key=np.asarray(sorted(map(int,raw)),dtype=np.int64)
        masses=np.asarray([int(raw[str(int(c))]) for c in key],dtype=np.int64)
        if len(key)<2 or (masses<=0).any() or masses.sum()!=UNIVERSE:
            raise ValueError(f'incomplete exact {level} theory')
        tables[level]={'codes':key,'mass':masses,'p':masses.astype(np.float64)/UNIVERSE}
    return tables


def _index_tags(numbers:np.ndarray,tables:dict)->dict:
    codes=classification_keys(classify_numbers(numbers))
    out={}
    for level in LEVELS:
        k=tables[level]['codes']; ix=np.searchsorted(k,codes[level])
        if np.any(ix>=len(k)) or not np.array_equal(k[ix],codes[level].astype(np.int64)):
            raise ValueError(f'{level}: found a code absent from the exact universe')
        out[level]=ix.astype(np.int32)
    return out


def validate_history(ids:np.ndarray,numbers:np.ndarray):
    x=np.asarray(numbers)
    d=np.asarray(ids)
    if d.ndim!=1 or x.shape!=(len(d),6) or len(d)<12 or (np.diff(d.astype(np.int64))<=0).any():
        raise ValueError('draws must be in strictly increasing historical order')
    classify_numbers(x)
    if len(set(map(int,d)))!=len(d):raise ValueError('duplicate draw IDs')
    return np.asarray(d,dtype=np.int64),np.sort(x.astype(np.uint8),axis=1)


def _prequential_one(indices:np.ndarray,prior:np.ndarray,min_train:int,tau:float)->np.ndarray:
    """Per-event log(model probability / mathematically exact fair probability).

    The history at t is updated strictly AFTER scoring event t.
    """
    x=np.asarray(indices,dtype=np.int32)
    if not 1<=min_train<len(x) or tau<=0 or not np.isfinite(tau):raise ValueError('invalid time horizon / tau')
    p=np.asarray(prior,dtype=np.float64)
    counts=np.zeros(len(p),dtype=np.int64)
    out=np.full(len(x),np.nan,dtype=np.float64)
    for t,k in enumerate(x):
        if t>=min_train:
            out[t]=np.log((counts[k]+tau*p[k])/(t+tau)/p[k])
        counts[k]+=1
    return out


def _iid_prequential_simulation(tables:dict,min_train:int,n:int,tau:float,reps:int,seed:int,
                                on_progress=None)->dict:
    """IID fair whole-record simulations; the three nested levels are coupled.

    Simulate at the *fine* edge layer, then derive both coarser graph levels.
    Each replication refits the online count model on its own past.
    """
    if reps<1:raise ValueError('IID replicate count must be positive')
    rng=np.random.default_rng(seed)
    edge=tables['edge']['codes']
    ec_ix=rng.choice(len(edge),size=(reps,n),p=tables['edge']['p'])
    partition_of_edge=edge>>10
    component_of_partition=np.count_nonzero(
        ((partition_of_edge[:,None]>>np.array([15,12,9,6,3,0]))&7)>0,axis=1)
    lookup={
        'edge':np.arange(len(edge),dtype=np.int32),
        'partition':np.searchsorted(tables['partition']['codes'],partition_of_edge).astype(np.int32),
        'component':np.searchsorted(tables['component']['codes'],component_of_partition).astype(np.int32)}
    if np.any(tables['component']['codes'][lookup['component']]!=component_of_partition):
        raise AssertionError('nested taxonomy component mapping mismatch')
    if np.any(tables['partition']['codes'][lookup['partition']]!=partition_of_edge):
        raise AssertionError('nested taxonomy partition mapping mismatch')
    result={}
    ridx=np.arange(reps)
    for level in LEVELS:
        ix=lookup[level][ec_ix]
        prior=tables[level]['p']
        ct=np.zeros((reps,len(prior)),dtype=np.int32)
        gains=np.zeros(reps,dtype=np.float64)
        for t in range(n):
            k=ix[:,t]
            if t>=min_train:
                gains+=np.log((ct[ridx,k]+tau*prior[k])/(t+tau)/prior[k])
            ct[ridx,k]+=1
        result[level]=gains
        if on_progress:on_progress(f'[IID LOGSCORE] {level}: {reps} complete independent sequences')
    return result


def _fresh_uniform_tickets(rng:np.random.Generator,pool:int)->np.ndarray:
    """Uniform 6/49 without replacement within each ticket, distinct tickets."""
    if pool<1:raise ValueError('invalid candidate pool size')
    out=[];seen=set()
    while len(out)<pool:
        block=np.argpartition(rng.random((max(16,pool-len(out)),49)),6,axis=1)[:,:6]+1
        for row in block:
            key=tuple(sorted(map(int,row)))
            if key not in seen:
                seen.add(key);out.append(key)
                if len(out)==pool:break
    return np.asarray(out,dtype=np.uint8)


def _ticket_matches(tickets:np.ndarray,winners:np.ndarray)->np.ndarray:
    if tickets.ndim!=3 or tickets.shape[2]!=6 or winners.shape!=(len(tickets),6):raise ValueError('invalid ticket array')
    return (tickets[:,:,:,None]==winners[:,None,None,:]).sum(axis=(2,3)).astype(np.uint8)


def _ticket_summary(matches:np.ndarray)->dict:
    x=np.asarray(matches)
    if x.ndim!=2 or x.size==0:raise ValueError('match array required')
    return {'total_bets':int(x.size),'total_match_at_least_3':int(np.sum(x>=3)),
            'total_match_at_least_4':int(np.sum(x>=4)),
            'total_match_at_least_5':int(np.sum(x>=5)),
            'total_match_6':int(np.sum(x==6)),
            'draws_with_any_3plus':int(np.sum((x>=3).any(axis=1))),
            'histogram':{str(i):int(np.sum(x==i)) for i in range(7)},
            'mean_matches_per_ticket':float(x.mean()),
            'mean_best_matches_per_draw':float(x.max(axis=1).mean())}


def _ticket_conditional_null(tickets:dict[str,np.ndarray],*,reps:int,seed:int,
                             on_progress=None)->dict:
    """Independent future fair 6/49 tickets evaluated against *frozen* picks.

    Exact conditional-on-bets reference distribution; because ticket sequences
    were selected adaptively on the actual history, this is not a complete
    refit-the-whole-strategy simulation and is marked as such in the report.
    """
    if reps<1:raise ValueError('invalid ticket null replication count')
    n=len(next(iter(tickets.values())))
    out={k:np.zeros((reps,3),dtype=np.float64) for k in tickets}
    rng=np.random.default_rng(seed)
    for lo in range(0,reps,16):
        size=min(16,reps-lo)
        draws=(np.argpartition(rng.random((size,n,49)),6,axis=2)[:,:,:6]+1).astype(np.uint8)
        for strategy,candidates in tickets.items():
            # [R, N, bets, selected_values, winning_values] boolean view.
            counts=(candidates[None,:,:,:,None]==draws[:,:,None,None,:]).sum(axis=(-1,-2))
            out[strategy][lo:lo+size,0]=(counts>=3).sum(axis=(1,2))
            out[strategy][lo:lo+size,1]=(counts>=4).sum(axis=(1,2))
            out[strategy][lo:lo+size,2]=counts.max(axis=2).mean(axis=1)
        if on_progress and (lo==0 or lo+size==reps or (lo+size)%128==0):
            on_progress(f'[IID TICKET NULL] {lo+size}/{reps}')
    return out


def candidate_backtest(numbers:np.ndarray,edge_tags:np.ndarray,tables:dict,*,
                       min_train:int,tau:float,seed:int,pool_size:int,bets:int,
                       ticket_null_reps:int,on_progress=None)->tuple[dict,dict]:
    """Same freshly randomized candidate pool per event; select without future.

    Baseline: first bets from the uniform independent pool.
    Experimental: largest *previously estimated* L3 q/p among that pool.
    Both use exactly the same number of 6/49 tickets per draw.
    """
    x=np.asarray(numbers,dtype=np.uint8)
    n=len(x)
    if not 1<=bets<=pool_size<=2000 or min_train>=n:raise ValueError('invalid ticket settings')
    rng=np.random.default_rng(seed)
    prior=tables['edge']['p'];codes=tables['edge']['codes']
    counts=np.bincount(edge_tags[:min_train],minlength=len(prior)).astype(np.int64)
    selected={k:np.empty((n-min_train,bets,6),dtype=np.uint8) for k in ('uniform','shape_ranked')}
    ranking=[]
    for t in range(min_train,n):
        fresh=_fresh_uniform_tickets(rng,pool_size)
        pool_codes=classification_keys(classify_numbers(fresh))['edge']
        ix=np.searchsorted(codes,pool_codes)
        if np.any(ix>=len(codes)) or not np.array_equal(codes[ix],pool_codes.astype(np.int64)):
            raise AssertionError('universe taxonomy mismatch')
        # Proper conditional categorical factor q(class)/p(class).
        ratio=(counts[ix]+tau*prior[ix])/((t+tau)*prior[ix])
        order=np.argsort(-ratio,kind='stable')
        selected['uniform'][t-min_train]=fresh[:bets]
        selected['shape_ranked'][t-min_train]=fresh[order[:bets]]
        ranking.append({'draw_id_index':t,'best_ratio':float(ratio[order[0]]),
                        'median_pool_ratio':float(np.median(ratio))})
        counts[int(edge_tags[t])]+=1
    actual={name:_ticket_matches(tk,x[min_train:]) for name,tk in selected.items()}
    summaries={k:_ticket_summary(val) for k,val in actual.items()}
    p_match={str(j):comb(6,j)*comb(43,6-j)/UNIVERSE for j in range(7)}
    null=_ticket_conditional_null(selected,reps=ticket_null_reps,seed=seed+199,
                                  on_progress=on_progress)
    for name in summaries:
        obs=summaries[name];a=null[name]
        obs['iid_expected_3plus_tickets']=float(a[:,0].mean())
        obs['iid_expected_4plus_tickets']=float(a[:,1].mean())
        obs['iid_conditional_p_ge_3plus']=(1+int(np.count_nonzero(a[:,0]>=obs['total_match_at_least_3'])))/(len(a)+1)
        obs['iid_conditional_p_ge_4plus']=(1+int(np.count_nonzero(a[:,1]>=obs['total_match_at_least_4'])))/(len(a)+1)
        obs['iid_3plus_reference_p95']=float(np.quantile(a[:,0],0.95))
    report={'strategy_definitions':{
           'uniform':'First k legal independently uniform candidate tickets from the same pool before each draw',
           'shape_ranked':'Top k tickets by online Dirichlet-smoothed L3 q/p, using only earlier historical draws',
           'pool_generation':'Uniform independent legal 6/49, unique within pool; exact fair conditional chances unchanged'},
        'pool_size':pool_size,'tickets_per_draw':bets,'draws_evaluated':n-min_train,
        'iid_exact_match_probability_per_ticket':1/UNIVERSE,
        'iid_exact_per_ticket_match_distribution':p_match,
        'strategies':summaries,'iid_ticket_null_replicates':ticket_null_reps,
        'null_caution':'Conditional fixed-ticket IID reference; NOT a full rerun of the adaptive strategy on an IID training sequence',
        'expected_six_hits_per_strategy':float((n-min_train)*bets/UNIVERSE)}
    detail={'tickets':selected,'matches':actual,'ranking':ranking}
    return report,detail


def analyze(ids:np.ndarray,numbers:np.ndarray,baseline:dict,*,min_train:int=500,
           tau:float=DEFAULT_TAU,iid_replicates:int=1000,ticket_null_replicates:int=128,
           seed:int=649,pool_size:int=64,bets:int=5,on_progress=None)->tuple[dict,dict]:
    """One historical chronological OOS run; all models and pools predeclared."""
    draw_ids,nums=validate_history(ids,numbers)
    n=len(draw_ids)
    if not 10<=min_train<n or tau<=0 or not np.isfinite(tau):raise ValueError('bad online configuration')
    tables=_baseline_tables(baseline)
    tags=_index_tags(nums,tables)
    if on_progress:on_progress(f'[HISTORY] strictly ordered={n} warmup={min_train} future={n-min_train}')
    real={level:_prequential_one(tags[level],tables[level]['p'],min_train,tau)
          for level in LEVELS}
    null=_iid_prequential_simulation(tables,min_train,n,tau,iid_replicates,seed+100,on_progress)
    scores={}
    for level in LEVELS:
        g=real[level][min_train:]
        total=float(g.sum())
        sim=null[level]
        p_ge=(1+int(np.count_nonzero(sim>=total)))/(iid_replicates+1)
        scores[level]={'cumulative_log_likelihood_ratio':total,
          'mean_log_gain_per_draw_nats':float(g.mean()),
          'mean_log_gain_per_draw_bits':float(g.mean()/log(2)),
          'model_vs_fair_e_value':float(np.exp(min(700,total))),
          'iid_null_mean_cumulative_gain':float(sim.mean()),
          'iid_null_p05_p50_p95':[float(v) for v in np.quantile(sim,[.05,.5,.95])],
          'iid_refit_upper_tail_p':p_ge,
          'iid_refit_upper_tail_bonferroni_across_three_levels':min(1.,3*p_ge),
          'positive_gain_draws':int(np.sum(g>0))}
    if on_progress:on_progress('[PREQUENTIAL] online L1/L2/L3 scored against exact IID probabilities')
    ticket,ticket_arrays=candidate_backtest(nums,tags['edge'],tables,
       min_train=min_train,tau=tau,seed=seed+33,pool_size=pool_size,bets=bets,
       ticket_null_reps=ticket_null_replicates,on_progress=on_progress)
    rare_codes=np.array([edge_name(c)==RARE_CODE_NAME for c in tables['edge']['codes']],dtype=bool)
    if int(rare_codes.sum())!=1:raise AssertionError('rare category not in exact P1 theory')
    rare_ix=int(np.flatnonzero(rare_codes)[0])
    occurrences=np.flatnonzero(tags['edge']==rare_ix)
    rare={'class':RARE_CODE_NAME,'theoretical_combinations':int(tables['edge']['mass'][rare_ix]),
          'exact_fair_probability':float(tables['edge']['p'][rare_ix]),
          'observed_total':len(occurrences),'observed_pre_warmup':int(np.sum(occurrences<min_train)),
          'observed_in_retrospective_evaluation':int(np.sum(occurrences>=min_train)),
          'draw_ids':[int(draw_ids[i]) for i in occurrences],
          'restriction':'Discovered using all history before P2: cannot treat ANY historical segment as independent prospective confirmation. Future-only preregistration required.'}
    final_edge_counts=np.bincount(tags['edge'],minlength=len(tables['edge']['p']))
    posterior=(final_edge_counts+tau*tables['edge']['p'])/(n+tau)
    ratio=posterior/tables['edge']['p']
    top_order=np.argsort(-ratio,kind='stable')[:15]
    final_classes=[{'class':edge_name(int(tables['edge']['codes'][i])),
             'historical':int(final_edge_counts[i]),'fair_p':float(tables['edge']['p'][i]),
             'online_predicted_p_next':float(posterior[i]),
             'ratio_to_fair':float(ratio[i])} for i in top_order]
    report={'schema':SCHEMA,'research_only':True,
      'data':{'history_draws':n,'history_first_draw_id':int(draw_ids[0]),
              'history_last_draw_id':int(draw_ids[-1]),'evaluated_draws':n-min_train,
              'fair_universe':UNIVERSE,'fixed_graph_rule':RULE_VERSION},
      'settings':{'min_train':min_train,'tau':tau,'iid_replicates':iid_replicates,
           'ticket_null_replicates':ticket_null_replicates,'pool_size':pool_size,
           'bets_per_draw':bets,'seed':seed},
      'prequential_log_scores':scores,
      'ticket_backtest':ticket,
      'rare_shape_watchlist':rare,
      'next_draw_descriptive_class_ratios':final_classes,
      'decision':'NO CLAIM OF PREDICTIVE ADVANTAGE; inspect IID calibration and multiple comparisons',
      'cautions':['Every fair 6/49 ticket has exact same future full-hit probability',
        'Historical P0/P1 method selection precludes calling retrospective results independent discovery',
        'Dirichlet models update after scoring each observed draw; no future information enters forecasts',
        'The 7x7 number-to-pixel mapping is a research convention, not physical draw equipment',
        'Full ticket-null simulations freeze actual candidate tickets; not a full adaptive null refit',
        'P6_O1_D5 is post-selected historically and receives no special model advantage']}
    arrays={'draw_ids':draw_ids,'numbers':nums,
      **{f'tag_{lev}':tags[lev] for lev in LEVELS},
      **{f'gain_{lev}':real[lev] for lev in LEVELS},
      **{f'tickets_{name}':tk for name,tk in ticket_arrays['tickets'].items()},
      **{f'matches_{name}':m for name,m in ticket_arrays['matches'].items()}}
    if on_progress:on_progress('[P2] chronological backtest and ticket controls completed')
    return report,arrays


def per_draw_rows(arrays:dict, min_train:int):
    ids=arrays['draw_ids']
    for t in range(min_train,len(ids)):
        i=t-min_train
        yield {'draw_id':int(ids[t]),'train_draws_before':t,
               **{f'n{j+1}':int(arrays['numbers'][t,j]) for j in range(6)},
               **{f'gain_{k}_nats':float(arrays[f'gain_{k}'][t]) for k in LEVELS},
               **{f'{s}_best_match':int(arrays[f'matches_{s}'][i].max()) for s in ('uniform','shape_ranked')},
               **{f'{s}_3plus_ticket_count':int((arrays[f'matches_{s}'][i]>=3).sum()) for s in ('uniform','shape_ranked')}}


def last_candidate_rows(arrays:dict,report:dict):
    """Show last backtested *pre-draw* candidate tickets with observed match.

    Does not label them as next draw predictions. Explorer supports fresh query.
    """
    i=-1
    last_id=int(arrays['draw_ids'][-1])
    for s in ('uniform','shape_ranked'):
        for j,ticket in enumerate(arrays[f'tickets_{s}'][i]):
            yield {'strategy':s,'evaluated_draw_id':last_id,'ticket_position':j+1,
                    **{f'n{k+1}':int(ticket[k]) for k in range(6)},
                    'matched_main_numbers':int(arrays[f'matches_{s}'][i,j])}
