"""v2.2: chronological exploratory falsification of M0-style historical repair.

NO future outcome enters prefix training or candidate selection. This is NOT a
prospective independent experiment: the strategy family and data were explored
historically. Four-ticket portfolios have exactly equal IID official prize
coverage by construction, regardless of their descriptive optimization scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
from math import exp, fsum, prod
from pathlib import Path
import json

import numpy as np

from .geometry_convergence_v029 import audit_repository_inputs
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import classification_keys
from .strategy_verification_v12 import (
    PREFIX_SHA, HISTORY_N, CUTOFF_ID, CUTOFF_DATE, JOINT_SHA, SHAPE_SHA,
    LEVELS, WEIGHTS, _grid, derive_fair_counts, integer_gain_numerators,
    weighted_sd, shape_gain_table, prefix_sha256,
)
from .temporal_dataset_v024 import load_csv_history
from .strategy_ab_v026 import (
    checked_bonus_by_draw, disjoint_portfolio, draw_awards, paired_exact_p,
)
from .deficit_compensation_v025 import load_csv_history as load_records
from .official_prize_v026 import official_prize_coverage
from .strategy_verification_v17 import sampled_unique, screen_pool, greedy_multistart

SCHEMA = 'lotto649.strategy_validation.v2.2.chronological_exploratory.v1'
PRIMARY_ENDPOINT = 'history-repair versus history-blind fair main-ball pairwise ordinal score'
TRAIN_MIN = 1850
HORIZON = 25
STRIDE = 50
MAX_FOLDS = 13
RAW_CANDIDATES = 40000
POOL_CAP = 1000
RESTARTS = 6
CONTROL_DRAWS_PER_TARGET = 16


def prefix_score_model(history: np.ndarray, fair_joint: np.ndarray,
                       fair_shapes: dict[str,dict[int,int]], cuts: list[int]) -> dict:
    """Fit only strictly older draw counts; fixed exact fair theory is time-free."""
    h=np.asarray(history,dtype=np.uint8)
    if h.ndim!=2 or h.shape[1]!=6 or len(h)<256 or np.any(h[:,1:]<=h[:,:-1]):
        raise ValueError('invalid sorted historical six-number prefix')
    sums=h.sum(axis=1,dtype=np.int64)-21
    products=np.prod(h.astype(np.int64),axis=1)
    logs=np.searchsorted(cuts,products,side='right')-1
    if np.any(logs<0) or np.any(logs>=fair_joint.shape[1]):
        raise ValueError('historical product outside exact fair grid')
    obs_joint=np.zeros_like(fair_joint)
    np.add.at(obs_joint,(sums,logs),1)
    numerators={}; denoms={}; sds={}; gains={}
    sources={
        'sum':(np.bincount(sums,minlength=259).tolist(),fair_joint.sum(axis=1).tolist()),
        'log':(np.bincount(logs,minlength=fair_joint.shape[1]).tolist(),fair_joint.sum(axis=0).tolist()),
        'joint':(obs_joint.ravel().tolist(),fair_joint.ravel().tolist()),
    }
    for key,(observed,fair) in sources.items():
        num,den=integer_gain_numerators(observed,fair)
        sd=weighted_sd([x/den for x in num],fair)
        numerators[key]=num;denoms[key]=den;sds[key]=sd
        gains[key]=np.asarray([x/den/sd for x in num],dtype=np.float64)
    cats=classification_keys(classify_numbers(h))
    shape_tables={};shape_sd={}
    for lv in LEVELS:
        observed=dict(Counter(map(int,cats[lv])))
        table=shape_gain_table(observed,fair_shapes[lv],len(h))
        shape_tables[lv]=table
        shape_sd[lv]=weighted_sd([table[k] for k in sorted(table)],
                                  [fair_shapes[lv][k] for k in sorted(table)])
    return {'n':len(h),'cuts':np.asarray(cuts,dtype=np.int64),'joint_cols':fair_joint.shape[1],
            'positive':{k:np.asarray([x>0 for x in numerators[k]],dtype=bool) for k in sources},
            'gains':gains,'shape_tables':shape_tables,'shape_sd':shape_sd,
            'shape_cache':{}}


def feature_scores(numbers: np.ndarray, model: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """All candidate/draw four-block raw standardized gains. Negative allowed.

    Use precomputed category-level scores; no future outcome changes this model.
    """
    a=np.asarray(numbers,dtype=np.uint8)
    if a.ndim!=2 or a.shape[1]!=6 or len(a)==0 or np.any(a<1) or np.any(a>49) or np.any(a[:,1:]<=a[:,:-1]):
        raise ValueError('sorted distinct legal six-number matrix required')
    sums=a.sum(axis=1,dtype=np.int64)-21
    logs=np.searchsorted(model['cuts'],np.prod(a.astype(np.int64),axis=1),side='right')-1
    if np.any(logs<0) or np.any(logs>=model['joint_cols']):raise ValueError('log bin outside theory')
    joints=sums*model['joint_cols']+logs
    cats=classification_keys(classify_numbers(a))
    edge=cats['edge']
    unique,inverse=np.unique(edge,return_inverse=True)
    score_table=[]
    for code in unique:
        code=int(code)
        if code not in model['shape_cache']:
            idx=int(np.flatnonzero(edge==code)[0]); lv_scores=[]
            for lv in LEVELS:
                cid=int(cats[lv][idx]);lv_scores.append(
                    model['shape_tables'][lv][cid]/model['shape_sd'][lv])
            model['shape_cache'][code]=fsum(w*s for w,s in zip(
                (WEIGHTS[lv] for lv in LEVELS),lv_scores))
        score_table.append(model['shape_cache'][code])
    raw=np.column_stack((model['gains']['sum'][sums],
                         model['gains']['log'][logs],
                         model['gains']['joint'][joints],
                         np.asarray(score_table,dtype=np.float64)[inverse]))
    qualified=(model['positive']['sum'][sums]&model['positive']['log'][logs]&
               model['positive']['joint'][joints]&(raw[:,3]>0))
    return raw,cats['partition'].astype(np.int64),qualified


def select_repair_portfolio(candidate_stream: list[tuple[int,...]], model:dict,
                            *, seed:int,pool_cap:int=POOL_CAP,
                            restarts:int=RESTARTS)->tuple[list[list[int]],dict]:
    """Fold-local empirical ranks; M0-inspired BOUNDED selector, not v2.1 global M0."""
    raw,parts,positive=feature_scores(np.asarray(candidate_stream,dtype=np.uint8),model)
    selected=np.flatnonzero(positive)
    if len(selected)<pool_cap:raise ValueError('insufficient four-positive prefix candidates')
    allpos=raw[selected]
    sorted_axes=[np.sort(allpos[:,k]) for k in range(4)]
    ranks=np.column_stack([np.searchsorted(sorted_axes[k],allpos[:,k],side='left')
                           for k in range(4)])
    rows=[{'ticket':list(candidate_stream[int(i)]),'part':int(parts[int(i)]),
           'rank':[int(v) for v in ranks[j]],'appearance':int(i)}
          for j,i in enumerate(selected)]
    pool=screen_pool(rows,policy='shape_round_robin',cap=pool_cap,seed=seed,
                     sample_size=len(candidate_stream))
    best=greedy_multistart(pool,restarts=restarts)
    tickets=best['tickets']
    if len({v for row in tickets for v in row})!=24 or len(tickets)!=4:
        raise AssertionError('M0 four-ticket constraint violation')
    return tickets, {'training_draws':model['n'],'sample_raw':len(candidate_stream),
                     'qualified':len(rows),'pool_size':len(pool),
                     'selection_quality_in_prefix_sample':best['quality'],
                     'optimizer':'fold-local four-block sample-rank/shape round-robin/greedy',
                     'not_global_optimum':True}


def paired_score(actual:np.ndarray,controls:np.ndarray,model:dict)->float:
    """Against independent fair tickets: E[paired_score|past] = 1/2 under IID.

    Avoid sample-reference percentile artifacts; real and control tickets are
    scored by the *same* prefix-only model and rank ties receive half credit.
    """
    a=np.asarray(actual,dtype=np.uint8).reshape(1,6)
    c=np.asarray(controls,dtype=np.uint8)
    if c.ndim!=2 or c.shape[1]!=6 or len(c)<1:raise ValueError('controls must be fair six-number tickets')
    raw,_,_=feature_scores(np.concatenate((a,c)),model)
    obs=float(raw[0].min())
    ref=np.min(raw[1:],axis=1)
    return float((np.count_nonzero(obs>ref)+0.5*np.count_nonzero(obs==ref))/len(ref))


def validate_schedule(draws:int,train_min:int,horizon:int,stride:int,max_folds:int)->list[tuple[int,int]]:
    if not (256<=train_min<draws and 1<=horizon<=stride and 1<=max_folds<=64):
        raise ValueError('nonoverlap expanding schedule required')
    schedule=[]
    for start in range(train_min,draws,stride):
        if len(schedule)==max_folds:break
        end=start+horizon
        if end>draws:break   # no truncated, unequally sized holdouts
        schedule.append((start,end))
    if not schedule:raise ValueError('no complete holdout fold')
    if any(schedule[i][1]>schedule[i+1][0] for i in range(len(schedule)-1)):
        raise AssertionError('holdout windows overlap')
    return schedule


def analyze_pairwise_results(event_pairs:list[tuple[bool,bool]],scores:list[float])->dict:
    if not event_pairs or not scores:raise ValueError('need outcomes and scores')
    a_only=sum(a and not b for a,b in event_pairs)
    b_only=sum(b and not a for a,b in event_pairs)
    both=sum(a and b for a,b in event_pairs)
    neither=sum(not a and not b for a,b in event_pairs)
    n=len(event_pairs);avg=fsum(scores)/len(scores)
    if n!=len(scores) or any(not 0<=x<=1 for x in scores):raise ValueError('mismatched bounded outcomes')
    # Hoeffding's bounded martingale inequality: under IID fair, actual and
    # independently sampled control draws are conditionally exchangeable.
    azuma_p=min(1.,2*exp(-2*len(scores)*(avg-0.5)**2))
    return {'draws':n,'history_blind_any_award':a_only+both,
            'history_based_any_award':b_only+both,
            'history_blind_only':a_only,'history_based_only':b_only,
            'both_awarded':both,'neither_awarded':neither,
            'history_based_minus_blind_award_rate':(b_only-a_only)/n,
            'paired_exact_two_sided_p_exploratory':paired_exact_p(a_only,b_only),
            'future_draw_higher_score_fraction':avg,
            'future_draw_score_minus_fair_pairwise_null':avg-0.5,
            'score_two_sided_azuma_hoeffding_p_bound':azuma_p,
            'score_inference':'bounded conditional exchangeable fair control; conservative martingale tail bound'}


def run_walkforward(*, csv_root:Path,bin_root:Path,train_min:int=TRAIN_MIN,
                    horizon:int=HORIZON,stride:int=STRIDE,max_folds:int=MAX_FOLDS,
                    raw_candidates:int=RAW_CANDIDATES,pool_cap:int=POOL_CAP,
                    restarts:int=RESTARTS,controls_per_target:int=CONTROL_DRAWS_PER_TARGET,
                    source_gate:bool=True)->dict:
    if not (1000<=raw_candidates<=100000 and 4<=pool_cap<=raw_candidates and
            1<=restarts<=20 and 1<=controls_per_target<=64):
        raise ValueError('preregistered computational budget outside limits')
    if source_gate: audit_repository_inputs(csv_root,bin_root)
    data=load_csv_history(csv_root)
    if len(data.ids)<HISTORY_N or int(data.ids[HISTORY_N-1])!=CUTOFF_ID or int(data.dates[HISTORY_N-1])!=CUTOFF_DATE:
        raise AssertionError('required historical prefix end missing')
    digest=prefix_sha256(data.ids[:HISTORY_N],data.dates[:HISTORY_N],data.numbers[:HISTORY_N])
    if digest!=PREFIX_SHA:raise AssertionError('original historical prefix changed')
    # Strictly restrict all exploratory holdouts to the archived historical prefix.
    history=np.asarray(data.numbers[:HISTORY_N],dtype=np.uint8)
    ids=np.asarray(data.ids[:HISTORY_N],dtype=np.int64)
    dates=np.asarray(data.dates[:HISTORY_N],dtype=np.int64)
    records, _=load_records(csv_root,'2026-11-06')
    if len(records)!=HISTORY_N or any(r[1]!=int(ids[i]) or tuple(r[2])!=tuple(map(int,history[i])) for i,r in enumerate(records)):
        raise AssertionError('independent main-number CSV loaders disagree')
    bonus=checked_bonus_by_draw(csv_root,records)
    schedule=validate_schedule(HISTORY_N,train_min,horizon,stride,max_folds)
    grid=_grid(49,6,'0.5');cuts=grid['integer_product_thresholds']
    fair_joint,fair_shapes,proof=derive_fair_counts(cuts,batch_size=80000)
    if proof['joint_sha256']!=JOINT_SHA or proof['shape_sha256']!=SHAPE_SHA:
        raise AssertionError('fair theory hash drift')
    baseline_profile=None;folds=[];paired=[];ordinal=[];tier_counts={'a':Counter(),'b':Counter()}
    for fi,(start,end) in enumerate(schedule):
        if not (ids[start-1]<ids[start] and dates[start-1]<dates[start]):raise AssertionError('chronology failure')
        train=history[:start]
        model=prefix_score_model(train,fair_joint,fair_shapes,cuts)
        seed=704011+fi*101
        stream,checkpoints=sampled_unique(seed,(raw_candidates,))
        strategy_b,metadata=select_repair_portfolio(stream,model,seed=seed,
                                                     pool_cap=pool_cap,restarts=restarts)
        strategy_a=disjoint_portfolio(903801+fi)
        a_prof=official_prize_coverage(strategy_a)
        b_prof=official_prize_coverage(strategy_b)
        parity_keys=('any_prize_outcomes','awarded_ticket_count_outcomes','any_ticket_by_tier','expected_tickets_by_tier')
        if any(a_prof[k]!=b_prof[k] for k in parity_keys):
            raise AssertionError('official exact IID prize-coverage parity violation')
        if baseline_profile is None:baseline_profile=a_prof
        elif any(a_prof[k]!=baseline_profile[k] for k in parity_keys):
            raise AssertionError('fold exposure not identical to fixed fair four-ticket coverage')
        fold_pairs=[];fold_scores=[]
        from random import Random
        rng=Random(202200000+fi)
        for idx in range(start,end):
            t=tuple(map(int,history[idx])); day=datetime.strptime(str(int(dates[idx])),'%Y%m%d').date().isoformat()
            key=(day,int(ids[idx])); special=bonus[key]
            aa=draw_awards(strategy_a,t,special)
            bb=draw_awards(strategy_b,t,special)
            paired.append((aa['any_award'],bb['any_award']));fold_pairs.append((aa['any_award'],bb['any_award']))
            tier_counts['a'].update(aa['tiers']);tier_counts['b'].update(bb['tiers'])
            fair_controls=np.asarray([sorted(rng.sample(range(1,50),6)) for _ in range(controls_per_target)],dtype=np.uint8)
            signal=paired_score(np.asarray(t,dtype=np.uint8),fair_controls,model)
            ordinal.append(signal);fold_scores.append(signal)
        result=analyze_pairwise_results(fold_pairs,fold_scores)
        folds.append({'fold':fi,'train_draws':start,
                      'train_end_id':int(ids[start-1]),'train_end_date':str(dates[start-1]),
                      'test_start_id':int(ids[start]),'test_end_id':int(ids[end-1]),
                      'test_draws':end-start,
                      'candidate_sample_sha256':checkpoints[0]['sample_sha256'],
                      'history_based_tickets':strategy_b,
                      'history_blind_tickets':strategy_a,
                      'selection_meta':metadata,
                      'outcome':result})
    aggregate=analyze_pairwise_results(paired,ordinal)
    fair_probability=float(baseline_profile['any_prize_probability'])
    aggregate.update({'fair_iid_any_award_probability_both':fair_probability,
                      'fair_expected_any_award_events_per_arm':fair_probability*aggregate['draws'],
                      'history_based_tiers':dict(sorted(tier_counts['b'].items())),
                      'history_blind_tiers':dict(sorted(tier_counts['a'].items())),
                      'comparable_ticket_exposure_each_arm':4*aggregate['draws'],
                      'both_arms_same_theoretical_award_profile':True,
                      'no_claim_about_net_payout_or_unknown_parimutuel_prizes':True})
    # Significance p-values are descriptive due to historical strategy exploration.
    positive=(aggregate['history_based_minus_blind_award_rate']>0 and
              aggregate['paired_exact_two_sided_p_exploratory']<0.05)
    signal=(aggregate['future_draw_score_minus_fair_pairwise_null']>0 and
            aggregate['score_two_sided_azuma_hoeffding_p_bound']<0.05)
    return {'schema':SCHEMA,'decision':'EXPLORATORY_SIGNAL_ONLY_NOT_PROSPECTIVE' if positive or signal else 'EXPLORATORY_PREDICTIVE_NO_GO',
            'source_gate':{'prefix_draws':HISTORY_N,'prefix_sha256':digest,
                           'last_training_eligible_draw_id':int(ids[-1]),
                           'theory_full_universe_count':proof['full_universe_count'],
                           'theory_joint_sha256':proof['joint_sha256'],
                           'theory_shape_sha256':proof['shape_sha256'],
                           'bonus_verified_against_independent_csv_loader':True},
            'parameters':{'train_min':train_min,'horizon':horizon,'stride':stride,
                          'max_folds':max_folds,'candidate_raw':raw_candidates,
                          'screened_pool':pool_cap,'restarts':restarts,
                          'independent_fair_control_tickets_per_target':controls_per_target},
            'folds':folds,'aggregate':aggregate,
            'limitations':['Historical walk-forward is exploratory: these dates/model families were examined when rules were developed.',
                           'The M0-inspired per-prefix 40k candidate rank refit is a BOUNDED selection policy, not the exact v2.1 frozen global optimizer.',
                           'The primary score test compares realized drawn ticket repair preference against independent fair ticket scores, not an identified full combinatorial predictive probability model.',
                           'M0 historical training is strictly expanding; exact 6/49 fair theory and score formulas are independent of future outcomes.',
                           'Both four-ticket strategies have mathematically identical IID prize probabilities for every fold.',
                           'Any historical score difference requires a separate, pre-outcome future-only study and familywise multiple-testing control.',
                           'No prize return is inferred from top-tier pari-mutuel pools; no lottery ticket purchase or forecast recommendation is made.']}


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--train-min',type=int,default=TRAIN_MIN)
    p.add_argument('--horizon',type=int,default=HORIZON)
    p.add_argument('--stride',type=int,default=STRIDE)
    p.add_argument('--folds',type=int,default=MAX_FOLDS)
    p.add_argument('--candidate-raw',type=int,default=RAW_CANDIDATES)
    p.add_argument('--pool-cap',type=int,default=POOL_CAP)
    p.add_argument('--restarts',type=int,default=RESTARTS)
    p.add_argument('--fair-controls',type=int,default=CONTROL_DRAWS_PER_TARGET)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    path=args.out.resolve()
    if not path.is_relative_to(Path('results').resolve()) or path.exists():p.error('write-once under results required')
    report=run_walkforward(csv_root=args.csv_root,bin_root=args.bin_root,
        train_min=args.train_min,horizon=args.horizon,stride=args.stride,
        max_folds=args.folds,raw_candidates=args.candidate_raw,
        pool_cap=args.pool_cap,restarts=args.restarts,controls_per_target=args.fair_controls)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='ascii',newline='\n') as file:
        json.dump(report,file,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        file.write('\n')
    agg=report['aggregate']
    print('V22 DECISION',report['decision'])
    print('DRAWS',agg['draws'],'FOLDS',len(report['folds']))
    print('PRIZE A/B',agg['history_blind_any_award'],agg['history_based_any_award'],
          'A_ONLY',agg['history_blind_only'],'B_ONLY',agg['history_based_only'],
          'PAIRED_EXPLORATORY_P',agg['paired_exact_two_sided_p_exploratory'])
    print('FAIR SCORE PAIR',agg['future_draw_higher_score_fraction'],
          'AZUMA_BOUND',agg['score_two_sided_azuma_hoeffding_p_bound'])
    print('REPORT',path)
    return 0


if __name__=='__main__':raise SystemExit(main())
