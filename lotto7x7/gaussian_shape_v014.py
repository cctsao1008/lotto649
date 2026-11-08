"""v0.14: Gaussian-kernel shape density and GMM taxonomy from cached v0.13 distances.

A Gaussian radial *similarity score* over a non-Euclidean shape quotient is
not automatically a normalized probability density or a positive-definite
kernel. Classical MDS supplies explicitly approximate Euclidean coordinates
for diagonal-covariance Gaussian mixtures. Neither establishes draw odds.
"""
from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from sklearn.mixture import GaussianMixture

from .image_classification_v011 import iid_legal_images, numbers
from .pairwise_similarity_v013 import _device, pairwise_matrices, point_coordinates

SCHEMA = 'lotto7x7.gaussian_shape_density.v0.14'
DEFAULT_BANDWIDTHS = (0.35, 0.55, 0.8, 1.1)
DEFAULT_COMPONENTS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


def load_existing(npz_path: Path, csv_path: Path):
    """Read *unchanged* cached distance/metadata and verify row alignment."""
    with np.load(npz_path, allow_pickle=False) as saved:
        if 'shape' not in saved or 'draw_ids' not in saved:
            raise ValueError('v0.13 NPZ must include shape and draw_ids')
        dist = np.array(saved['shape'], dtype=np.float32, copy=True)
        draw_ids = np.array(saved['draw_ids'], dtype=np.int64, copy=True)
    n = len(draw_ids)
    if (n < 8 or dist.shape != (n, n) or len(np.unique(draw_ids)) != n
            or not np.all(np.isfinite(dist)) or np.min(dist) < -1e-5
            or np.max(np.abs(dist - dist.T)) > 2e-4
            or np.max(np.abs(np.diag(dist))) > 2e-4):
        raise ValueError('inconsistent v0.13 symmetric distance matrix')
    dist = np.maximum(0.0, (dist + dist.T) / 2)
    np.fill_diagonal(dist, 0)
    rows = {}
    with Path(csv_path).open('r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            draw_id = int(row['draw_id'])
            ns = [int(row[f'n{i}']) for i in range(1, 7)]
            if len(set(ns)) != 6 or min(ns) < 1 or max(ns) > 49 or draw_id in rows:
                raise ValueError('invalid CSV draw / duplicate')
            rows[draw_id] = ns
    if set(rows) != set(map(int, draw_ids)):
        raise ValueError('CSV and NPZ draw_id sets differ')
    ns = np.array([rows[int(x)] for x in draw_ids], dtype=np.int16)
    return dist, draw_ids, ns


def kde_density(dist: np.ndarray, bandwidths, *, device='auto') -> np.ndarray:
    """Leave-one-out Gaussian radial similarity score, one score per draw/bandwidth.

    Excludes self-match; values are dimensionless and in [0,1], NOT PDFs.
    """
    d = np.asarray(dist)
    n = len(d)
    bws = np.array(tuple(bandwidths), dtype=np.float64)
    if d.shape != (n, n) or n < 3 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('invalid square distance matrix')
    if not len(bws) or not np.isfinite(bws).all() or (bws <= 0).any():
        raise ValueError('bandwidths must be finite positive values')
    dev = _device(device)
    dd = torch.as_tensor(d, device=dev, dtype=torch.float32)
    sq = dd.square()
    out=[]
    with torch.no_grad():
        for b in bws:
            sim=torch.exp(-sq / (2.0 * float(b)**2))
            # An exact diagonal of zero yields 1.0 self-similarity.
            value = (sim.sum(dim=1) - torch.diag(sim)) / (n - 1)
            out.append(value.cpu().numpy())
    return np.stack(out, axis=1)


def density_metrics(values: np.ndarray) -> dict:
    v=np.asarray(values,dtype=float).reshape(-1)
    if len(v)<3 or (v<0).any() or not np.isfinite(v).all():
        raise ValueError('need finite nonnegative scores')
    mean=float(v.mean())
    return {'mean':mean,'std':float(v.std()),
            'cv':float(v.std()/max(mean,1e-12)),
            'p50':float(np.median(v)),'p90':float(np.quantile(v,0.9)),
            'max':float(v.max()),
            'mean_log_score':float(np.log(np.maximum(v,1e-12)).mean())}


def density_modes(dist:np.ndarray, densities:np.ndarray, *, peak_radius:float,
                  assignment_radius:float, max_peaks:int=24, min_peak_quantile:float=0.75) -> dict:
    """Select separated density peaks; nearest-peak labels are NOT complete-link clusters."""
    n=len(densities)
    if (dist.shape!=(n,n) or peak_radius<=0 or assignment_radius<=0
            or max_peaks<1 or not (0<=min_peak_quantile<1)):
        raise ValueError('invalid mode finder parameters')
    order=np.argsort(-densities, kind='stable')
    cutoff=float(np.quantile(densities,min_peak_quantile))
    peaks=[]
    for i in order:
        if densities[i] < cutoff or len(peaks)>=max_peaks:break
        if all(float(dist[i,p])>peak_radius for p in peaks):peaks.append(int(i))
    if not peaks:
        peaks=[int(order[0])]
    nearest=np.argmin(dist[:,peaks],axis=1)
    mindist=dist[np.arange(n), np.array(peaks)[nearest]]
    labels=np.where(mindist<=assignment_radius, nearest, -1).astype(np.int32)
    # Peak indices must always be assigned to themselves.
    for k,p in enumerate(peaks):labels[p]=k
    return {'peaks':peaks,'labels':labels,
            'nearest_distance':mindist.astype(np.float32),
            'unassigned':int(np.count_nonzero(labels<0))}


def mds_embedding(dist:np.ndarray, dimensions:int=8, *, seed:int=649):
    """Positive-eigenvalue classical MDS; report actual shape-distance distortion.

    D4+translation minimum need not be isometrically Euclidean. Eigenvalues of
    centered -D**2/2 may be negative. Do not claim this is an exact embedding.
    """
    d=np.asarray(dist,dtype=np.float64)
    n=len(d)
    if d.shape!=(n,n) or n<8 or dimensions<1 or dimensions>=n-1:
        raise ValueError('bad embedding dimensions')
    sq=d*d
    row_mean=sq.mean(axis=1)
    gram=-0.5*(sq-row_mean[:,None]-row_mean[None,:]+sq.mean())
    gram=(gram+gram.T)*0.5
    k=min(dimensions+2,n-2)
    if n<=80:
        val,vec=np.linalg.eigh(gram)
        val,vec=val[-k:],vec[:,-k:]
    else:
        val,vec=eigsh(gram,k=k,which='LA',tol=1e-6,maxiter=12000,
                      v0=np.random.default_rng(seed).normal(size=n))
    order=np.argsort(val)[::-1]
    val=val[order];vec=vec[:,order]
    keep=(val>max(float(val[0]),1.0)*1e-9)
    val=val[keep][:dimensions];vec=vec[:,keep][:,:dimensions]
    if not len(val):raise ValueError('no positive MDS eigenvalues')
    coord=vec*np.sqrt(val)[None,:]
    # Deterministic sign convention for reproducible saved coordinate axes.
    for j in range(coord.shape[1]):
        i=np.argmax(np.abs(coord[:,j]))
        if coord[i,j]<0:coord[:,j]*=-1
    rng=np.random.default_rng(seed)
    a=rng.integers(0,n,size=min(5000,n*(n-1)//2))
    b=rng.integers(0,n,size=len(a))
    diff=np.linalg.norm(coord[a]-coord[b],axis=1)
    original=d[a,b]
    stress=float(np.sqrt(np.sum((diff-original)**2)/max(np.sum(original**2),1e-12)))
    return coord.astype(np.float32),{'dimensions_used':len(val),
            'largest_positive_eigenvalues':[float(x) for x in val],
            'sample_relative_stress':stress,
            'note':'Positive-only classical MDS approximates a potentially non-Euclidean D4+translation metric'}


def gaussian_mixture(embedding:np.ndarray, dist:np.ndarray, components=DEFAULT_COMPONENTS,
                     *,seed=649, n_init=2):
    """Select diagonal-covariance GMM with BIC *within one fixed MDS embedding*."""
    x=np.asarray(embedding,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<8 or dist.shape!=(len(x),len(x)):
        raise ValueError('bad GMM input shape')
    choices=sorted({int(k) for k in components if 1<=int(k)<=min(50,len(x)//4)})
    if not choices:raise ValueError('no valid GMM component choices')
    result=[];best=None
    for k in choices:
        model=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-4,
                     n_init=n_init,max_iter=200,random_state=seed,init_params='kmeans')
        model.fit(x)
        bic=float(model.bic(x))
        result.append({'k':k,'bic':bic,'converged':bool(model.converged_)})
        if best is None or bic<best[0]:best=(bic,model)
    model=best[1]
    probs=model.predict_proba(x)
    labels=probs.argmax(axis=1).astype(np.int32)
    confidence=probs.max(axis=1).astype(np.float32)
    classes=[]
    for k in range(model.n_components):
        members=np.flatnonzero(labels==k)
        if not len(members):continue
        sub=dist[np.ix_(members,members)]
        medoid=int(members[np.argmin(np.sum(sub,axis=1))])
        classes.append({'class_index':int(k),'count':int(len(members)),
              'percentage':float(100*len(members)/len(x)),
              'weight':float(model.weights_[k]),
              'mean_confidence':float(confidence[members].mean()),
              'medoid_index':medoid,'mean_shape_distance_to_medoid':float(dist[members,medoid].mean()),
              'p90_shape_distance_to_medoid':float(np.quantile(dist[members,medoid],0.9))})
    classes.sort(key=lambda c:(-c['count'],c['class_index']))
    base_bic=next((q['bic'] for q in result if q['k']==1),None)
    return {'best_k':int(model.n_components),'bic_table':result,
            'bic_improvement_vs_k1':(float(base_bic-best[0]) if base_bic is not None else None),
            'mean_confidence':float(confidence.mean()),
            'selected_at_search_ceiling':int(model.n_components)==int(max(choices)),
            'caution_if_ceiling':'BIC minimum occurs at largest tested K, so no natural class count has been identified',
            'uncertain_lt_0p6':int(np.count_nonzero(confidence<0.6)),
            'classes':classes,'labels':labels,'confidence':confidence,
            'posterior':probs.astype(np.float32)}


def matched_null_controls(dist, bandwidth, *, seed=649, draws=128, repeats=9,
                         device='auto', batch_size=512, compare_gmm=True,
                         control_gmm_components=(1,2,4,8), control_mds_dimensions=6,
                         on_progress=None):
    """Same-size actual subsamples vs independently generated *legal* 6/49.

    These are descriptive paired contrasts. Repeated historical subsets overlap,
    hence pairs/replicates cannot be treated as independent hypothesis tests.
    """
    n=len(dist)
    if repeats<0 or draws<8 or draws>n or bandwidth<=0:raise ValueError('bad control settings')
    rng=np.random.default_rng(seed)
    reps=[]
    for r in range(repeats):
        idx=rng.choice(n,draws,replace=False)
        historical=dist[np.ix_(idx,idx)]
        iid=iid_legal_images(draws,rng)
        pts=point_coordinates(iid)
        iid_dist=pairwise_matrices(pts,device=device,batch_size=batch_size,modes=('shape',))['shape']
        obs=density_metrics(kde_density(historical,[bandwidth],device=device)[:,0])
        null=density_metrics(kde_density(iid_dist,[bandwidth],device=device)[:,0])
        record={'replicate':r+1,'historical':obs,'iid_legal_6of49':null,
                     'delta_cv':obs['cv']-null['cv'],
                     'delta_mean_log_score':obs['mean_log_score']-null['mean_log_score']}
        if compare_gmm:
            descriptors=[]
            for tag,dsub in [('historical',historical),('iid_legal_6of49',iid_dist)]:
                emb,_=mds_embedding(dsub,min(control_mds_dimensions,draws-2),seed=seed+r)
                gmm=gaussian_mixture(emb,dsub,control_gmm_components,seed=seed+r,n_init=1)
                descriptors.append({'name':tag,'best_k':gmm['best_k'],
                    'bic_improvement_vs_k1':gmm['bic_improvement_vs_k1'],
                    'selected_at_search_ceiling':gmm['selected_at_search_ceiling']})
            record['gmm_matched_control']=descriptors
            record['delta_gmm_bic_gain']=(descriptors[0]['bic_improvement_vs_k1']
                                         -descriptors[1]['bic_improvement_vs_k1'])
        reps.append(record)
        if on_progress:on_progress(f'[IID MATCHED] {r+1}/{repeats}')
    delta_cv=np.array([r['delta_cv'] for r in reps],dtype=float)
    delta_log=np.array([r['delta_mean_log_score'] for r in reps],dtype=float)
    delta_bic=np.array([r['delta_gmm_bic_gain'] for r in reps],dtype=float) if compare_gmm and repeats else np.array([])
    return {'draws_per_group':int(draws),'replicates':reps,
            'mean_delta_cv':(float(delta_cv.mean()) if repeats else None),
            'mean_delta_mean_log_score':(float(delta_log.mean()) if repeats else None),
            'gmm_control_enabled':bool(compare_gmm),
            'mean_delta_gmm_bic_gain':(float(delta_bic.mean()) if len(delta_bic) else None),
            'delta_cv_range':([float(delta_cv.min()),float(delta_cv.max())] if repeats else []),
            'caution':'Exploratory matched-size contrasts; no independent p-value or prospective winning-number evidence'}


def analyze(dist,draw_ids,numbers_six, *,device='auto',
            bandwidths=DEFAULT_BANDWIDTHS,mode_bandwidth=0.55,
            peak_radius=0.6,assignment_radius=0.85,max_peaks=24,
            mds_dimensions=8,components=DEFAULT_COMPONENTS,
            control_draws=128,control_replicates=9,batch_size=512,seed=649,
            compare_gmm_controls=True,
            on_progress=None):
    n=len(dist);ids=np.asarray(draw_ids);ns=np.asarray(numbers_six)
    if ids.shape!=(n,) or ns.shape!=(n,6) or not np.all(np.isfinite(dist)):
        raise ValueError('bad history input')
    bandwidths=tuple(float(b) for b in bandwidths)
    if mode_bandwidth not in bandwidths:bandwidths=tuple(sorted(set(bandwidths+(mode_bandwidth,))))
    scores=kde_density(dist,bandwidths,device=device)
    per_bw={str(b):density_metrics(scores[:,j]) for j,b in enumerate(bandwidths)}
    preferred=scores[:,list(bandwidths).index(mode_bandwidth)]
    modes=density_modes(dist,preferred,peak_radius=peak_radius,assignment_radius=assignment_radius,max_peaks=max_peaks)
    if on_progress:on_progress(f'[KDE] bandwidths={bandwidths} peaks={len(modes["peaks"])} unassigned={modes["unassigned"]}')
    coord,embed_stats=mds_embedding(dist,mds_dimensions,seed=seed)
    if on_progress:on_progress(f'[MDS] dimensions={coord.shape[1]} stress={embed_stats["sample_relative_stress"]:.4f}')
    mix=gaussian_mixture(coord,dist,components,seed=seed)
    if on_progress:on_progress(f'[GMM] best_k={mix["best_k"]} mean_confidence={mix["mean_confidence"]:.4f}')
    controls=matched_null_controls(dist,mode_bandwidth,seed=seed,draws=control_draws,
                           repeats=control_replicates,device=device,batch_size=batch_size,
                           compare_gmm=compare_gmm_controls,
                           on_progress=on_progress)
    mode_classes=[]
    for k,p in enumerate(modes['peaks']):
        members=np.flatnonzero(modes['labels']==k)
        if not len(members):continue
        mode_classes.append({'class_index':k,'count':len(members),
            'percentage':float(100*len(members)/n),'medoid_index':p,
            'center_draw_id':int(ids[p]),'center_numbers':ns[p].tolist(),
            'density_score':float(preferred[p]),
            'mean_distance':float(modes['nearest_distance'][members].mean()),
            'p90_distance':float(np.quantile(modes['nearest_distance'][members],0.9)),
            'max_distance':float(modes['nearest_distance'][members].max())})
    mode_classes.sort(key=lambda x:(-x['count'],x['class_index']))
    for cl in mix['classes']:
        p=cl['medoid_index'];cl['medoid_draw_id']=int(ids[p]);cl['medoid_numbers']=ns[p].tolist()
    report={'schema':SCHEMA,'research_only':True,'draw_count':n,
         'device':str(_device(device)),'distance_source':'v0.13 cached NPZ shape, NOT recomputed historical distances',
         'bandwidths':list(bandwidths),'kde':{'per_bandwidth':per_bw,'preferred_bandwidth':mode_bandwidth,
                 'note':'Leave-one-out radial similarity scores, not normalized Gaussian probability densities'},
         'mode_taxonomy':{'selection':'density-ranked separated peaks; nearest peak within radius, otherwise unassigned',
               'peak_radius':peak_radius,'assignment_radius':assignment_radius,
               'peak_count':len(modes['peaks']),'peak_selection_hit_cap':len(modes['peaks'])>=max_peaks,
               'unassigned_count':modes['unassigned'],
               'classes':mode_classes,'warning':'Nearest-peak members are not guaranteed mutually close'},
         'mds':embed_stats,
         'gmm':{k:v for k,v in mix.items() if k not in ('labels','confidence','posterior')},
         'iid_matched_controls':controls,
         'cautions':['A D4 + translation quotient distance need not be a Euclidean distance; MDS stress measures projection distortion',
                     'KDE peaks and BIC-selected Gaussian components are descriptive and do not establish natural lottery probabilities',
                     'GMM labels may differ with coordinate dimensions or sampling',
                     'The 7x7 positions are a researcher-chosen arrangement of numbers, not physical draw locations',
                     'All legal 6/49 tickets have equal future odds if the draw mechanism is IID fair']}
    arrays={'draw_ids':ids.astype(np.int64),'numbers':ns.astype(np.int16),
        'bandwidths':np.asarray(bandwidths,dtype=np.float32),'density':scores.astype(np.float32),
        'embedding':coord,'mode_labels':modes['labels'],
        'mode_nearest_distances':modes['nearest_distance'],
        'gmm_labels':mix['labels'],'gmm_confidence':mix['confidence'],'gmm_posterior':mix['posterior']}
    return report,arrays


def _bitmap(ns, color='#17867f', cell=13):
    chosen=set(map(int,ns));pixels=[]
    for i in range(49):
        r,c=divmod(i,7)
        fill=color if (i+1) in chosen else '#e5eaf0'
        pixels.append(f'<rect x="{c*cell+1}" y="{r*cell+1}" width="{cell-2}" height="{cell-2}" rx="1.5" fill="{fill}"/>')
    return f'<svg width="105" viewBox="0 0 {7*cell} {7*cell}" role="img" aria-label="7 by 7 historical lottery bitmap">'+''.join(pixels)+'</svg>'


def html_gallery(report,arrays, *, gallery_count=8):
    import html
    if gallery_count<1:raise ValueError('gallery_count must be positive')
    ns=arrays['numbers'];ids=arrays['draw_ids'];modes=arrays['mode_labels'];gmm=arrays['gmm_labels']
    score=arrays['density'][:,np.argmin(abs(arrays['bandwidths']-report['kde']['preferred_bandwidth']))]
    chunks=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>Lotto7x7 v0.14 Gaussian Shape Distribution</title>',
       '<style>body{max-width:1420px;margin:1.8rem auto;padding:0 1rem;background:#f6f8fa;color:#253742;font-family:system-ui}section{margin:1.4rem 0;border-top:2px solid #d4dfe4;padding-top:1rem}article{background:white;border:1px solid #dce4e8;border-radius:12px;padding:.9rem;margin:.7rem 0}.samples{display:flex;flex-wrap:wrap;gap:.5rem}.sample{display:flex;flex-direction:column;max-width:115px}.small{color:#637786;font-size:.8rem}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #dce4e8}</style>',
       '<h1>7×7 Gaussian Shape Density and Class Quantification</h1>',
       '<p>Historical image neighborhoods and approximate Gaussian mixtures; descriptive only, not prediction of winning numbers.</p>']
    counts,edges=np.histogram(score,bins=24)
    hmax=max(int(counts.max()),1)
    bars=[]
    for i,count in enumerate(counts):
        height=100*int(count)/hmax
        bars.append(f'<rect x="{8+i*18}" y="{116-height:.1f}" width="13" height="{height:.1f}" fill="#17867f"/>')
    chunks.extend(['<section><h2>Gaussian KDE similarity-score distribution</h2>',
           '<p>Leave-one-out density scores at bandwidth '
           +str(report['kde']['preferred_bandwidth'])
           +'; unnormalized radial similarities, not a Gaussian PDF.</p>',
           '<svg viewBox="0 0 450 138" width="100%" style="max-width:720px">'
           +''.join(bars)+'<path d="M5 116 H445" stroke="#344"/></svg>',
           f'<p>Range: {float(edges[0]):.5f}–{float(edges[-1]):.5f}; '
           f'peak selection cap hit: {report["mode_taxonomy"]["peak_selection_hit_cap"]}; '
           f'not assigned to selected peaks: {report["mode_taxonomy"]["unassigned_count"]}</p>',
           '</section><section><h2>GMM Bayesian information criterion</h2>',
           '<p>Lower is better among the tested K values, at one fixed 8D MDS representation. '
           'The numerical optimum is not proof of a natural class count.</p>',
           '<table><thead><tr><th>Components (K)</th><th>BIC</th></tr></thead><tbody>'])
    for b in report['gmm']['bic_table']:
        chunks.append(f'<tr><td>{b["k"]}</td><td>{b["bic"]:.2f}</td></tr>')
    chunks.append('</tbody></table></section>')
    control=report['iid_matched_controls']
    chunks.append('<section><h2>Legal 6/49 IID controls (matched-size refits)</h2>'
                  f'<p>Matched control draws: {control["draws_per_group"]}; '
                  f'replicates: {len(control["replicates"])}. '
                  f'Mean observed-minus-IID density CV: {control["mean_delta_cv"]}; '
                  f'mean observed-minus-IID GMM BIC gain: {control["mean_delta_gmm_bic_gain"]}. '
                  'Exploratory paired contrasts; not an independent significance test.</p></section>')
    sections=[('Shape KDE density peaks',report['mode_taxonomy']['classes'],modes),
              ('GMM categories in approximate MDS space',report['gmm']['classes'],gmm)]
    for title,groups,labels in sections:
        chunks.append(f'<section><h2>{html.escape(title)}</h2>')
        for cl in groups:
            k=cl['class_index'];member=np.flatnonzero(labels==k)
            center=int(cl['medoid_index'])
            if len(member):
                # For KDE, show densest assigned members; for GMM show those close to medoid (by density ranking).
                rank=member[np.argsort(-score[member],kind='stable')][:gallery_count]
            else:rank=np.array([],dtype=int)
            chunks.append(f'<article><h3>Class {k+1:03d} · {len(member)} historical images ({100*len(member)/len(ns):.2f}%)</h3>')
            chunks.append('<div class="samples"><div class="sample">'+_bitmap(ns[center])+
                  f'<span class="small">Representative draw {int(ids[center])}<br>{ns[center].tolist()}</span></div>')
            for i in rank:
                chunks.append('<div class="sample">'+_bitmap(ns[i],color='#668cb5')+
                   f'<span class="small">{int(ids[i])}<br>density {float(score[i]):.4f}</span></div>')
            chunks.append('</div></article>')
        chunks.append('</section>')
    chunks.append('<section><h2>Experimental caution</h2><p>Density scores are not calibrated PDFs. High density is not increased lottery odds. The GMM is fitted to an approximate MDS embedding with measurable distortion.</p></section></html>')
    return '\n'.join(chunks)
