"""v0.24 P1/P2: causal per-draw 7x7 temporal features and honest IID controls.

Row t is computed solely from draw indices <= t, suitable for forecasting t+1.
All lag-1 / rolling analyses declare warmup and reject false lookahead.
"""
from __future__ import annotations

from math import comb
import numpy as np
from scipy.fft import dctn
from scipy.signal import correlate2d

from .temporal_dataset_v024 import TemporalDataset
from .shape_taxonomy_v015 import classify_numbers

SCHEMA='lotto7x7.temporal_features.v0.24'
LAGS=(1,2,3,5,10,20)
FEATURE_NAMES=(
    'centroid_row','centroid_col','spread','cardinal_edges','diagonal_edges',
    'components_8','dct_low_energy',
    *(f'overlap_lag_{i}' for i in LAGS),
    'centroid_motion','edge_change','overlap_mean_5','overlap_mean_10',
)


def iid_overlap_probs() -> dict[str,float]:
    denominator=comb(49,6)
    return {str(k):comb(6,k)*comb(43,6-k)/denominator for k in range(7)}


def d4_translation_max_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Max intersection/union over 8 D4 transforms and any grid translation.

    This compares shape independent of absolute number/pixel positions; it is
    explicitly NOT a property of the physical lottery.
    """
    before=np.asarray(a,dtype=np.int8)
    after=np.asarray(b,dtype=np.int8)
    if before.shape!=(7,7) or after.shape!=(7,7) or before.sum()!=6 or after.sum()!=6:
        raise ValueError('expected two six-on 7x7 masks')
    best=0
    for turns in range(4):
        rotated=np.rot90(after,turns)
        for arr in (rotated,np.fliplr(rotated)):
            best=max(best,int(correlate2d(before,arr,mode='full').max()))
            if best==6:return 1.0
    return best/(12-best)


def base_descriptors(data: TemporalDataset) -> dict[str,np.ndarray]:
    # Geometry follows the chosen raster mapping, never numerical label order.
    nums=(np.nonzero(data.images.reshape(-1,49))[1].reshape(len(data.ids),6)+1).astype(np.uint8)
    r=((nums.astype(np.int16)-1)//7).astype(float)
    c=((nums.astype(np.int16)-1)%7).astype(float)
    rmean=r.mean(axis=1);cmean=c.mean(axis=1)
    spread=np.sqrt(((r-rmean[:,None])**2+(c-cmean[:,None])**2).mean(axis=1))
    tax=classify_numbers(nums)
    dct=dctn(data.images.astype(float),axes=(1,2),norm='ortho')
    # DC is exactly 6/7 on a six-on seven-by-seven orthonormal DCT.
    if not np.allclose(dct[:,0,0],6/7,atol=1e-12):
        raise AssertionError('nonconstant 6/7 DCT DC across six-on images')
    low=(dct[:,:3,:3]**2).sum(axis=(1,2))-dct[:,0,0]**2
    return {'centroid_row':rmean,'centroid_col':cmean,'spread':spread,
            'cardinal_edges':tax['cardinal_edges'].astype(float),
            'diagonal_edges':tax['diagonal_edges'].astype(float),
            'components_8':tax['n_components'].astype(float),
            'dct_low_energy':low}


def temporal_features(data:TemporalDataset) -> tuple[np.ndarray,dict[str,np.ndarray]]:
    """Compute all causal features; unavailable initial lags remain NaN.

    The model is allowed to use row t only for predicting draw t+1.
    """
    n=len(data.ids)
    src=base_descriptors(data)
    flat=data.images.reshape(n,49)
    for lag in LAGS:
        v=np.full(n,np.nan)
        if lag<n:
            v[lag:]=(flat[lag:]&flat[:-lag]).sum(axis=1)
        src[f'overlap_lag_{lag}']=v
    motion=np.full(n,np.nan)
    motion[1:]=np.hypot(np.diff(src['centroid_row']),np.diff(src['centroid_col']))
    src['centroid_motion']=motion
    edges=src['cardinal_edges']+src['diagonal_edges']
    delta=np.full(n,np.nan)
    delta[1:]=np.diff(edges)
    src['edge_change']=delta
    overlaps=src['overlap_lag_1']
    for window in (5,10):
        v=np.full(n,np.nan)
        for t in range(window,n):
            v[t]=float(np.mean(overlaps[t-window+1:t+1]))
        src[f'overlap_mean_{window}']=v
    matrix=np.column_stack([src[name] for name in FEATURE_NAMES])
    return matrix,src


def transition_rows(data:TemporalDataset, *, full_shape:bool=True) -> list[dict]:
    """Causal P1 rows for each t>0, includes exact and D4-translation IoU."""
    _,f=temporal_features(data)
    out=[]
    for t in range(1,len(data.ids)):
        common=int(f['overlap_lag_1'][t]);removed=6-common
        record={'previous_draw_id':int(data.ids[t-1]),'current_draw_id':int(data.ids[t]),
                'previous_date':int(data.dates[t-1]) if data.dates is not None else None,
                'current_date':int(data.dates[t]) if data.dates is not None else None,
                'retained':common,'removed':removed,'added':removed,
                'xor_pixels':2*removed,'exact_jaccard':common/(12-common),
                'centroid_motion':float(f['centroid_motion'][t]),
                'edge_change':float(f['edge_change'][t])}
        if full_shape:
            record['d4_translation_iou']=d4_translation_max_iou(data.images[t-1],data.images[t])
        out.append(record)
    return out


def p1_summary(rows:list[dict]) -> dict:
    observed=np.bincount([r['retained'] for r in rows],minlength=7)
    probs=iid_overlap_probs()
    n=len(rows)
    mean=float(np.dot(np.arange(7),observed)/n)
    return {'schema':SCHEMA,'transition_count':n,
            'overlap_histogram':{str(i):int(observed[i]) for i in range(7)},
            'iid_overlap_probability':probs,
            'iid_expected_counts':{str(i):n*probs[str(i)] for i in range(7)},
            'mean_retained':mean,'iid_mean_retained':36/49,
            'mean_retained_delta':mean-36/49,
            'note':'An exact overlap histogram is descriptive. Overlapping transitions are not independently replicated experiments.'}


def autocorrelation(x:np.ndarray, max_lag:int=30)->np.ndarray:
    """Ordinary centered sample ACF; NaNs omitted only at leading warmup."""
    a=np.asarray(x,dtype=float)
    valid=a[np.isfinite(a)]
    if max_lag<0 or len(valid)<max_lag+3:
        raise ValueError('too few finite feature values for requested ACF')
    z=valid-valid.mean()
    power=float(z@z)
    if power<=1e-20:
        return np.r_[1.0,np.zeros(max_lag)]
    return np.array([1.0]+[float(z[:-lag]@z[lag:]/power) for lag in range(1,max_lag+1)])


def feature_acf_report(matrix:np.ndarray, *, max_lag:int=20,
                       columns:tuple[str,...]=FEATURE_NAMES)->dict:
    return {name:{'acf':autocorrelation(matrix[:,i],max_lag).tolist(),
                  'finite_count':int(np.isfinite(matrix[:,i]).sum())}
            for i,name in enumerate(columns)}


def generate_iid_numbers(rng:np.random.Generator,n:int) ->np.ndarray:
    """Every generated unordered six-set has exact 1/C(49,6) probability."""
    key=rng.random((n,49))
    return np.sort(np.argpartition(key,5,axis=1)[:,:6]+1,axis=1).astype(np.uint8)


def p2_iid_acf_control(data:TemporalDataset, *, max_lag:int=20,
                       replicates:int=80,seed:int=649)->dict:
    """Full IID draw-series surrogate (not naive shuffling rolling metrics)."""
    if replicates<2:raise ValueError('replicates >=2 required')
    mat,_=temporal_features(data)
    names=('overlap_lag_1','overlap_mean_5','centroid_motion','cardinal_edges','dct_low_energy')
    indices=[FEATURE_NAMES.index(x) for x in names]
    rng=np.random.default_rng(seed)
    observed=np.array([autocorrelation(mat[:,j],max_lag)[1:] for j in indices])
    sims=np.empty((replicates,len(indices),max_lag),dtype=float)
    for rep in range(replicates):
        n=generate_iid_numbers(rng,len(data.ids))
        # Fill using exact six-on bitmap; dates/IDs do not affect classification.
        from .temporal_dataset_v024 import rasterize
        sim=TemporalDataset(data.ids,n,rasterize(n),None,{})
        mx,_=temporal_features(sim)
        for k,j in enumerate(indices):
            sims[rep,k]=autocorrelation(mx[:,j],max_lag)[1:]
    null_max=np.max(np.abs(sims),axis=(1,2))
    obs_max=float(np.max(np.abs(observed)))
    return {'schema':'lotto7x7.temporal_acf_iid.v0.24',
            'features':list(names),'max_lag':max_lag,'replicates':replicates,'seed':seed,
            'observed_acf_by_lag':{name:observed[k].tolist() for k,name in enumerate(names)},
            'iid_mean_acf_by_lag':{name:sims[:,k].mean(axis=0).tolist() for k,name in enumerate(names)},
            'familywise_statistic_max_abs_acf':obs_max,
            'iid_max_abs_acf_pvalue':float((1+np.count_nonzero(null_max>=obs_max))/(replicates+1)),
            'caution':'Surrogates preserve rolling-window artifacts because every replicate re-computes the same overlapping features.'}
