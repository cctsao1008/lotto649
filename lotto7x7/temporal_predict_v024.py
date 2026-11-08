"""v0.24 P4/P5: causal, regularized multioutput next-draw ridge and walk-forward.

The predictor at draw index t sees only rows <t.  Hyperparameter alpha is
chosen exclusively from earlier chronological validation predictions.  All
scores are proper: number marginal Brier/Bernoulli log and exact ticket log.
"""
from __future__ import annotations

from math import comb,log
import json
from pathlib import Path
import numpy as np

from .temporal_dataset_v024 import TemporalDataset, chronological_split
from .temporal_features_v024 import temporal_features
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import classification_keys

SCHEMA='lotto7x7.temporal_walkforward.v0.24'
FAIR_P=6/49
FAIR_TICKET_LOG=-log(comb(49,6))


def _project_capped_simplex(raw:np.ndarray, total:float=6)->np.ndarray:
    """Euclidean projection on {0<=p<=1, sum(p)=6}, deterministic."""
    v=np.asarray(raw,dtype=float)
    if v.shape!=(49,) or not np.all(np.isfinite(v)):
        raise ValueError('expected finite 49 probabilities')
    lo=float(v.min()-1.);hi=float(v.max()+1.)
    for _ in range(65):
        middle=(lo+hi)/2
        if np.clip(v-middle,0.,1.).sum()>total:lo=middle
        else:hi=middle
    out=np.clip(v-(lo+hi)/2,0.,1.)
    if not np.isclose(out.sum(),total,atol=1e-10):
        raise AssertionError('probability constraint not achieved')
    return out


def exact_ticket_logprob(probabilities:np.ndarray,numbers:np.ndarray)->float:
    """A normalized conditional-six product-odds model over all C(49,6) tickets.

    P(S)=prod(w_i for i in S) / e_6(w), w_i=p_i/(1-p_i), |S|=6.
    Note p_i are ridge-projected scores; these are weight parameters, not the
    exact marginals of this conditional-six distribution.
    """
    p=np.asarray(probabilities,dtype=float)
    ticket=np.asarray(numbers,dtype=np.int64)
    if p.shape!=(49,) or ticket.shape!=(6,) or np.any(ticket<1) or np.any(ticket>49) or len(set(ticket.tolist()))!=6:
        raise ValueError('invalid probabilities/ticket')
    if np.any(~np.isfinite(p)) or np.any(p<0) or np.any(p>1):
        raise ValueError('probability weights must be in [0,1]')
    q=np.clip(p,1e-5,1-1e-5)
    w=q/(1-q)
    w=w/np.mean(w)  # Common factor cancels; protect DP range.
    es=np.zeros(7,dtype=float);es[0]=1.
    for i,weight in enumerate(w,1):
        for k in range(min(6,i),0,-1):
            es[k]+=weight*es[k-1]
    return float(np.log(w[ticket-1]).sum()-np.log(es[6]))


def _brier(p:np.ndarray,y:np.ndarray)->float:
    return float(np.mean((p-y)**2))


def _bern_log(p:np.ndarray,y:np.ndarray)->float:
    q=np.clip(p,1e-6,1-1e-6)
    return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q)))


def _standardize_freeze(mat:np.ndarray,train_end:int,warmup:int)->tuple[np.ndarray,dict]:
    first=mat[warmup:train_end-1]
    if np.any(~np.isfinite(first)):
        raise ValueError('training features contain noncausal/missing lag')
    mu=first.mean(axis=0)
    scale=np.maximum(first.std(axis=0),0.1)
    z=(mat-mu)/scale
    z[:warmup]=0.0  # Never fitted; boundary values not forecastable.
    if np.any(~np.isfinite(z[warmup:])):
        raise ValueError('nonfinite features after warmup')
    return np.column_stack([np.ones(len(z)),z]),{'train_only_mean':mu.tolist(),'train_only_scale':scale.tolist()}


def _suff_stats(x:np.ndarray,y:np.ndarray,start_target:int,end_target:int)->tuple[np.ndarray,np.ndarray]:
    xx=x[start_target-1:end_target-1]
    yy=y[start_target:end_target]-FAIR_P
    return xx.T@xx,xx.T@yy


def _ridge_predict(gram:np.ndarray,cross:np.ndarray,v:np.ndarray,alpha:float)->np.ndarray:
    diag=np.full(len(gram),alpha);diag[0]=0.0
    beta=np.linalg.solve(gram+np.diag(diag),cross)
    return _project_capped_simplex(FAIR_P + v@beta)


def _eval_window(x:np.ndarray,y:np.ndarray,ids:np.ndarray,nums:np.ndarray,
                 *,start:int,end:int,train_start:int,alpha:float,
                 rolling_prior:float=100.,save_predictions:bool=False)->tuple[list[dict],np.ndarray]:
    gram,cross=_suff_stats(x,y,train_start,start)
    hits=y[train_start:start].sum(axis=0)
    ntrain=start-train_start
    out=[]
    for t in range(start,end):
        raw=_ridge_predict(gram,cross,x[t-1],alpha)
        baseline=(hits+rolling_prior*FAIR_P)/(ntrain+rolling_prior)
        if not np.isclose(baseline.sum(),6,atol=1e-10):
            raise AssertionError('static marginal smoothing should sum to six')
        current=y[t]
        result={'target_index':int(t),'draw_id':int(ids[t]),
            'fair_brier':_brier(np.full(49,FAIR_P),current),
            'static_brier':_brier(baseline,current),
            'temporal_brier':_brier(raw,current),
            'fair_bernoulli_logloss':_bern_log(np.full(49,FAIR_P),current),
            'static_bernoulli_logloss':_bern_log(baseline,current),
            'temporal_bernoulli_logloss':_bern_log(raw,current),
            'fair_ticket_logprob':FAIR_TICKET_LOG,
            'static_ticket_logprob':exact_ticket_logprob(baseline,nums[t]),
            'temporal_ticket_logprob':exact_ticket_logprob(raw,nums[t])}
        if save_predictions:
            result['temporal_49_parameters']=[float(p) for p in raw]
            result['static_49_parameters']=[float(p) for p in baseline]
        out.append(result)
        v=x[t-1]
        gram+=np.outer(v,v)
        cross+=np.outer(v,current-FAIR_P)
        hits+=current
        ntrain+=1
    return out, gram


def _block_bootstrap_ci(values:np.ndarray, *,seed:int=649,
                        reps:int=1000,block:int=10)->list[float]:
    d=np.asarray(values,dtype=float)
    n=len(d)
    if n<block*2 or reps<50:raise ValueError('not enough time points or resamples')
    rng=np.random.default_rng(seed)
    means=np.empty(reps)
    for j in range(reps):
        starts=rng.integers(0,n,size=(n+block-1)//block)
        draws=(starts[:,None]+np.arange(block))%n
        means[j]=d[draws.ravel()[:n]].mean()
    return np.quantile(means,[0.025,0.975]).tolist()


def _shape_logscore(data:TemporalDataset,*,universe:Path|None,
                    test_start:int,test_end:int,tau:float=384.) -> dict:
    if universe is None:
        return {'available':False,'reason':'Pass --universe to enable train-only L3 shape-only exact-ticket log-score.'}
    source=json.loads(Path(universe).read_text(encoding='utf-8'))
    if source.get('total')!=comb(49,6) or not source.get('complete'):
        raise ValueError('universe is not the full exact fair six-of-49 catalog')
    counts=source['counts']['edge']
    codes=np.array(sorted(int(k) for k in counts),dtype=np.int64)
    fair=np.array([counts[str(i)] for i in codes],dtype=float)/comb(49,6)
    # Use each draw's L3 class only AFTER generating the pre-draw score.
    observed=classification_keys(classify_numbers(data.numbers))['edge'].astype(np.int64)
    indices=np.searchsorted(codes,observed)
    if np.any(indices>=len(codes)) or not np.array_equal(codes[indices],observed):
        raise ValueError('observed shape class is absent from exact IID universe')
    hist=np.bincount(indices[:test_start],minlength=len(codes)).astype(float)
    n=float(test_start)
    logp=[]
    for t in range(test_start,test_end):
        k=indices[t]
        q=(hist[k]+tau*fair[k])/(n+tau)
        logp.append(float(FAIR_TICKET_LOG+np.log(q/fair[k])))
        hist[k]+=1.;n+=1
    return {'available':True,'model':'per-draw train-only L3 Dirichlet q/p shape ticket score',
            'mean_ticket_logprob':float(np.mean(logp)),
            'mean_delta_vs_fair':float(np.mean(logp)-FAIR_TICKET_LOG),
            'per_draw_logprob':logp}


def walk_forward(data:TemporalDataset, *,universe:Path|None=None,
                 alphas:tuple[float,...]=(1.,10.,100.,1000.),
                 bootstrap_seed:int=649,bootstrap_reps:int=1000)->tuple[dict,list[dict]]:
    n=len(data.ids)
    cut=chronological_split(n)
    warm=cut['warmup'];train_end=cut['train_end'];val_end=cut['validation_end']
    features,_=temporal_features(data)
    x,scaler=_standardize_freeze(features,train_end,warm)
    y=data.images.reshape(n,49).astype(float)
    train_start=warm+1
    validations={}
    for alpha in alphas:
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('ridge alpha must be positive')
        val,_=_eval_window(x,y,data.ids,data.numbers,start=train_end,
            end=val_end,train_start=train_start,alpha=alpha)
        validations[str(alpha)]=float(np.mean([v['temporal_brier'] for v in val]))
    chosen=float(min(alphas,key=lambda a:(validations[str(a)],a)))
    # Crucial: no test observation was consulted to choose alpha/scaling.
    rows,_=_eval_window(x,y,data.ids,data.numbers,start=val_end,end=n,
        train_start=train_start,alpha=chosen,save_predictions=True)
    metric_names=('fair_brier','static_brier','temporal_brier',
                 'fair_bernoulli_logloss','static_bernoulli_logloss','temporal_bernoulli_logloss',
                 'fair_ticket_logprob','static_ticket_logprob','temporal_ticket_logprob')
    average={field:float(np.mean([row[field] for row in rows])) for field in metric_names}
    fair=np.array([r['fair_brier'] for r in rows]);temp=np.array([r['temporal_brier'] for r in rows])
    static=np.array([r['static_brier'] for r in rows])
    c_fair=_block_bootstrap_ci(fair-temp,seed=bootstrap_seed,reps=bootstrap_reps)
    c_static=_block_bootstrap_ci(static-temp,seed=bootstrap_seed+1,reps=bootstrap_reps)
    shape=_shape_logscore(data,universe=universe,test_start=val_end,test_end=n)
    if shape['available']:
        for item,score in zip(rows,shape.pop('per_draw_logprob')):
            item['shape_ticket_logprob']=score
    delta=float(average['fair_brier']-average['temporal_brier'])
    decision='NO_GO'
    # A retrospective positive result is NEVER sufficient for a real release.
    preregistered_future_confirmatory=False
    if (c_fair[0]>0 and c_static[0]>0 and preregistered_future_confirmatory):
        decision='REVIEW_FOR_INTEGRATION'
    report={'schema':SCHEMA,'status':'COMPLETED','draws':n,'split':cut,
        'training_target_count':int(train_end-train_start),
        'validation_target_count':int(val_end-train_end),
        'test_target_count':len(rows),'alpha_candidates':list(alphas),
        'validation_mean_brier_by_alpha':validations,'selected_alpha':chosen,
        'standardization':scaler,
        'score_means':average,
        'delta_fair_minus_temporal_brier':delta,
        'paired_block_bootstrap_95ci_fair_minus_temporal':c_fair,
        'paired_block_bootstrap_95ci_static_minus_temporal':c_static,
        'block_bootstrap_reps':bootstrap_reps,
        'shape_only_ticket_log_score':shape,
        'decision':decision,'needs_preregistered_new_draws':True,
        'cautions':[
          'Evaluation is chronological one-step ahead. At t all fit targets and all features stop strictly before t.',
          'Ridge features and scaling are frozen from initial training; alpha selected on validation only.',
          'This is an exploratory retrospective split, not independent preregistered future evidence.',
          'Static frequency is a separate marginal baseline; shape-only is a separate normalized exact-ticket log-score.',
          'Ridge projected scores sum to six; conditional-ticket normalized weights do not imply exact marginals.',
          'Brier is a proper score for individual number marginals, conditional exact-ticket log-score for complete six-sets.',
          'No model is automatically integrated into paid lottery selections.']}
    return report,rows
