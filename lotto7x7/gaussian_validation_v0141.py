"""P0 Batch 1: honest, matched-IID Gaussian shape validation from cached distances.

The v0.13 distance defines the 7x7 shape geometry, not a draw probability.
Full-size and matched-subset IID controls refit their own MDS/GMM; identical
sample size, coordinates dimension, component grid and optimization settings.
The MDS geometry may not be Euclidean; GMM is only a descriptive approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .gaussian_shape_v014 import (load_existing, kde_density, density_metrics,
                                    density_modes, mds_embedding)
from .image_classification_v011 import iid_legal_images
from .pairwise_similarity_v013 import point_coordinates, pairwise_matrices

SCHEMA = 'lotto7x7.gaussian_shape_validation.v0.14.1'
DEFAULT_DIMS = (4, 8, 12, 16)
DEFAULT_K = (1, 2, 3, 4, 6, 8, 12, 16, 24)
DEFAULT_BW = (0.35, 0.55, 0.8, 1.1)


def validate_parameters(dist, dimensions, components, bandwidths, matched_size,
                        matched_replicates, full_iid_replicates, stability_seeds,
                        stability_fraction, base_dim, max_peaks):
    a = np.asarray(dist)
    n = len(a)
    dims = tuple(sorted(set(int(x) for x in dimensions)))
    ks = tuple(sorted(set(int(x) for x in components)))
    bws = tuple(sorted(set(float(x) for x in bandwidths)))
    seeds = tuple(int(s) for s in stability_seeds)
    if (a.shape != (n, n) or n < 24 or not np.isfinite(a).all()
            or not np.allclose(a, a.T, atol=2e-4)
            or np.max(np.abs(np.diag(a))) > 2e-4 or np.min(a) < -1e-5):
        raise ValueError('expected nonnegative symmetric shape-distance matrix, n>=24')
    if (not dims or min(dims)<2 or max(dims)>=matched_size-2
            or not ks or min(ks)!=1 or max(ks)>matched_size//4
            or not bws or min(bws)<=0 or not np.isfinite(bws).all()
            or int(base_dim) not in dims or matched_size<24 or matched_size>n
            or matched_replicates<1 or full_iid_replicates<0 or len(seeds)<2
            or len(set(seeds))!=len(seeds) or not (0.5<=stability_fraction<1)
            or max_peaks<1):
        raise ValueError('invalid dimension/grid/bandwidth/matched/stability settings')
    return dims,ks,bws,seeds


def fit_gmm_grid(embedding: np.ndarray, components, *, seed: int, n_init: int=2):
    """Identical diagonal GMM recipe for historical and independently drawn IID.

    Model labels carry no absolute meaning; ARI/NMI below are permutation-invariant.
    The BIC comparison is within each embedding, not across data-set scales.
    """
    x = np.asarray(embedding, dtype=np.float64)
    if x.ndim != 2 or len(x) < 16 or n_init<1:
        raise ValueError('invalid embedding or n_init')
    choices = sorted(set(map(int,components)))
    if not choices or choices[0] < 1 or choices[-1] > len(x)//4:
        raise ValueError('GMM component grid incompatible with group size')
    rows=[];winner=None
    with threadpool_limits(limits=1):
        for k in choices:
            m = GaussianMixture(n_components=k, covariance_type='diag',
                 reg_covar=1e-4, n_init=n_init, max_iter=200,
                 random_state=seed, init_params='kmeans')
            m.fit(x)
            bic=float(m.bic(x))
            rows.append({'k':k,'bic':bic,'converged':bool(m.converged_)})
            if winner is None or bic < winner[0]:
                winner=(bic,m)
    model=winner[1]
    labels=model.predict(x).astype(np.int32)
    confidence=model.predict_proba(x).max(axis=1)
    silhouette=(float(silhouette_score(x, labels,
        sample_size=min(len(x),1000),random_state=seed))
        if 1<len(np.unique(labels))<len(x) else None)
    base = next((z['bic'] for z in rows if z['k']==1), None)
    return {
        'best_k':int(model.n_components),
        'bic_gain_vs_k1':float(base-winner[0]) if base is not None else None,
        'bic_table':rows,'selected_at_ceiling':int(model.n_components)==choices[-1],
        'mean_confidence':float(confidence.mean()),
        'silhouette':silhouette,'labels':labels,
    }


def _compact_fit(fit):
    return {k:v for k,v in fit.items() if k!='labels'}


def _embedding_stress(original, coordinates, indices):
    aa,bb=indices
    o=original[aa,bb]
    p=np.linalg.norm(coordinates[aa]-coordinates[bb],axis=1)
    return float(np.sqrt(np.sum((o-p)**2)/max(np.sum(o**2),1e-12)))


def embed_once(dist, max_dimensions, *, seed):
    # The leading classical-MDS eigenvectors are the same for all shorter
    # dimension grids. Compute only once per dataset for a fair, fast comparison.
    return mds_embedding(dist, max_dimensions, seed=seed)[0]


def dimension_comparison(dist, dims, ks, *, seed, n_init, on_progress=None):
    coords=embed_once(dist,max(dims),seed=seed)
    rng=np.random.default_rng(seed+172)
    n=len(dist)
    aa=rng.integers(0,n,size=4000)
    bb=rng.integers(0,n,size=4000)
    rows=[];fits={}
    for d in dims:
        if d>coords.shape[1]:raise ValueError('MDS returned too few positive dimensions')
        fit=fit_gmm_grid(coords[:,:d],ks,seed=seed,n_init=n_init)
        rows.append({'dimensions':int(d),'relative_stress':_embedding_stress(dist,coords[:,:d],(aa,bb)),
              **_compact_fit(fit)})
        fits[int(d)]=fit
        if on_progress:on_progress(f'[DIM] d={d} best_k={fit["best_k"]} stress={rows[-1]["relative_stress"]:.4f}')
    return coords,rows,fits


def _spearman(x,y):
    a=rankdata(x);b=rankdata(y)
    if a.std()<1e-12 or b.std()<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])


def bandwidth_comparison(dist, bws, *, device, peak_radius, assignment_radius, max_peaks):
    scores=kde_density(dist,bws,device=device)
    bws=tuple(bws)
    rows=[]
    n=len(dist)
    count=max(1,math.ceil(.1*n))
    top_sets=[set(np.argsort(-scores[:,i],kind='stable')[:count]) for i in range(len(bws))]
    for j,b in enumerate(bws):
        modes=density_modes(dist,scores[:,j],peak_radius=peak_radius,
              assignment_radius=assignment_radius,max_peaks=max_peaks)
        rank_ref=_spearman(scores[:,j],scores[:,0])
        jaccard=(len(top_sets[j]&top_sets[0])/len(top_sets[j]|top_sets[0]))
        rows.append({'bandwidth':float(b),'density':density_metrics(scores[:,j]),
          'spearman_vs_smallest_bandwidth':rank_ref,
          'top10pct_jaccard_vs_smallest':float(jaccard),
          'selected_representatives':int(len(modes['peaks'])),
          'hit_peak_cap':bool(len(modes['peaks'])==max_peaks),
          'unassigned_count':int(modes['unassigned'])})
    return scores,rows


def iid_distance_sample(n, *, rng, device, batch_size):
    images=iid_legal_images(n,rng)
    return pairwise_matrices(point_coordinates(images),device=device,
                     batch_size=batch_size,modes=('shape',))['shape']


def matched_controls(history_dist, dims, ks, bws, *, matched_size, repeats,
                     device, batch_size, seed, n_init, on_progress=None):
    """Equal-N/equal-grid paired experiments; do NOT treat resamples as independent."""
    rng=np.random.default_rng(seed+900)
    n=len(history_dist)
    results=[]
    for r in range(repeats):
        idx=rng.choice(n,matched_size,replace=False)
        obs=history_dist[np.ix_(idx,idx)]
        iid=iid_distance_sample(matched_size,rng=rng,device=device,batch_size=batch_size)
        cohorts={}
        for key,dd in [('historical',obs),('iid',iid)]:
            emb=embed_once(dd,max(dims),seed=seed+r)
            f=fit_gmm_grid(emb[:,:max(dims)],ks,seed=seed+r,n_init=n_init)
            dens=kde_density(dd,bws,device=device)
            cohorts[key]={'n':matched_size,'mds_dimensions':int(max(dims)),
                 'best_k':f['best_k'],'bic_gain_vs_k1':f['bic_gain_vs_k1'],
                 'bic_table':f['bic_table'],'silhouette':f['silhouette'],
                 'mean_confidence':f['mean_confidence'],
                 'density':{str(b):density_metrics(dens[:,i]) for i,b in enumerate(bws)}}
        results.append({'replicate':r+1,'cohorts':cohorts})
        if on_progress:on_progress(f'[MATCHED IID] {r+1}/{repeats} n={matched_size} historical_K={cohorts["historical"]["best_k"]} iid_K={cohorts["iid"]["best_k"]}')
    return results


def full_size_iid_controls(history_fit, n, dims, ks, bws, *, repeats,
                           seed, device, batch_size, n_init, on_progress=None):
    """Full-N 2497-vs-2497 null comparison; one replicate is NOT significance."""
    rng=np.random.default_rng(seed+19000)
    rows=[]
    for r in range(repeats):
        iid=iid_distance_sample(n,rng=rng,device=device,batch_size=batch_size)
        # Keep solver/MDS seed and optimization settings identical to the
        # historical full-N baseline; only the independently drawn data vary.
        emb=embed_once(iid,max(dims),seed=seed)
        f=fit_gmm_grid(emb[:,:max(dims)],ks,seed=seed,n_init=n_init)
        dens=kde_density(iid,bws,device=device)
        rows.append({'replicate':r+1,'n':n,'mds_dimensions':int(max(dims)),
            'iid_best_k':f['best_k'],'iid_bic_gain_vs_k1':f['bic_gain_vs_k1'],
            'iid_bic_table':f['bic_table'],'iid_silhouette':f['silhouette'],
            'iid_density':{str(b):density_metrics(dens[:,j]) for j,b in enumerate(bws)},
            'historical_best_k':history_fit['best_k'],
            'historical_bic_gain_vs_k1':history_fit['bic_gain_vs_k1']})
        if on_progress:on_progress(f'[FULL IID] {r+1}/{repeats} n={n} iid_K={f["best_k"]}')
        # One full-size distance matrix at a time; discard before next IID refit.
        del iid,emb,dens
    return rows


def classification_stability(full_embedding, selected_k, *, fraction, seeds,
                             dimension, n_init=1, on_progress=None):
    """Refit GMM on overlapping independently sampled subsets, fixing MDS axes.

    Held-out samples contribute to the MDS axes (transductive stability). No
    prospective generalization or fully refitted MDS stability is claimed.
    """
    x=np.asarray(full_embedding[:,:dimension],dtype=np.float64)
    n=len(x)
    count=int(math.ceil(n*fraction))
    if count<=selected_k*3:raise ValueError('sample too small for selected K')
    records=[]
    with threadpool_limits(limits=1):
        for s in seeds:
            rng=np.random.default_rng(s)
            a=np.sort(rng.choice(n,count,replace=False))
            b=np.sort(rng.choice(n,count,replace=False))
            common,ia,ib=np.intersect1d(a,b,return_indices=True)
            def fit(idx):
                m=GaussianMixture(n_components=selected_k,covariance_type='diag',
                   reg_covar=1e-4,n_init=n_init,max_iter=200,
                   random_state=s,init_params='kmeans')
                m.fit(x[idx])
                return m.predict(x[common])
            la=fit(a);lb=fit(b)
            records.append({'seed':s,'samples_per_refit':count,
                'intersection_size':int(len(common)),
                'ari':float(adjusted_rand_score(la,lb)),
                'nmi':float(normalized_mutual_info_score(la,lb))})
            if on_progress:on_progress(f'[STABILITY] seed={s} overlap={len(common)} ARI={records[-1]["ari"]:.4f} NMI={records[-1]["nmi"]:.4f}')
    return {'kind':'overlapping_subsample_fixed_full_MDS_axes',
        'warning':'MDS axes fixed from full historical data; not independent MDS refits or out-of-sample prediction',
        'class_count_fixed':int(selected_k),'refit_fraction':float(fraction),
        'runs':records,'mean_ari':float(np.mean([z['ari'] for z in records])),
        'mean_nmi':float(np.mean([z['nmi'] for z in records]))}


def analyze(dist, draw_ids, numbers_six, *, device='auto',dimensions=DEFAULT_DIMS,
            components=DEFAULT_K, bandwidths=DEFAULT_BW, comparison_dimension=8,
            matched_size=384, matched_replicates=9, full_iid_replicates=1,
            stability_seeds=(649,650,651,652), stability_fraction=.8,
            peak_radius=.6, assignment_radius=.85,max_peaks=24,
            n_init=2, batch_size=512, seed=649, on_progress=None):
    dims,ks,bws,seeds=validate_parameters(dist,dimensions,components,bandwidths,
        matched_size,matched_replicates,full_iid_replicates,stability_seeds,
        stability_fraction,comparison_dimension,max_peaks)
    if n_init<1 or batch_size<1:raise ValueError('invalid batch_size or n_init')
    d=np.asarray(dist,dtype=np.float32)
    ids=np.asarray(draw_ids,dtype=np.int64)
    ns=np.asarray(numbers_six,dtype=np.int16)
    if ids.shape!=(len(d),) or ns.shape!=(len(d),6):
        raise ValueError('draw IDs and six numbers must align with historical distances')
    coords,dimrows,fits=dimension_comparison(d,dims,ks,seed=seed,n_init=n_init,on_progress=on_progress)
    scores,bwrows=bandwidth_comparison(d,bws,device=device,peak_radius=peak_radius,
                            assignment_radius=assignment_radius,max_peaks=max_peaks)
    if on_progress:on_progress('[KDE] completed multi-bandwidth density-rank and mode sensitivity')
    matches=matched_controls(d,(comparison_dimension,),ks,bws,matched_size=matched_size,
         repeats=matched_replicates,device=device,batch_size=batch_size,
         seed=seed,n_init=n_init,on_progress=on_progress)
    fulliid=full_size_iid_controls(fits[comparison_dimension],len(d),
         (comparison_dimension,),ks,bws,repeats=full_iid_replicates,
         seed=seed,device=device,batch_size=batch_size,n_init=n_init,
         on_progress=on_progress)
    stability=classification_stability(coords,fits[comparison_dimension]['best_k'],
         fraction=stability_fraction,seeds=seeds,dimension=comparison_dimension,
         n_init=n_init,on_progress=on_progress)
    report={'schema':SCHEMA,'research_only':True,'data':{
            'draw_count':len(d),'pair_count':len(d)*(len(d)-1)//2,
            'distance_source':'cached v0.13 shape distance; no historical point matching recomputation',
            'draw_ids_first_last':[int(ids[0]),int(ids[-1])]},
         'settings':{'device':str(device),'dimensions':list(dims),'comparison_dimension':comparison_dimension,
             'gmm_components':list(ks),'gmm_n_init':n_init,
             'bandwidths':list(bws),'matched_size':matched_size,
             'matched_replicates':matched_replicates,'full_size_iid_replicates':full_iid_replicates,
             'stability_seeds':list(seeds),'stability_fraction':stability_fraction,
             'peak_radius':peak_radius,'assignment_radius':assignment_radius,
             'max_peaks':max_peaks,'seed':seed},
         'dimension_sensitivity':dimrows,'bandwidth_sensitivity':bwrows,
         'matched_iid_controls':matches,'full_size_iid_controls':fulliid,
         'stability':stability,
         'comparison_historical_baseline':next(z for z in dimrows if z['dimensions']==comparison_dimension),
         'cautions':['BIC best K depends on MDS dimension and diagonal-covariance GMM assumptions',
             'Independent legal 6/49 IID controls get their own MDS fits but identical dimension, grid and n_init',
             'Matched-size subsets have fewer points than the full history; do not compare their BIC values to full-N BIC',
             'One full-size IID replicate does not imply a valid significance test',
             'Repeated overlapping historical subsamples are NOT independent tests',
             'KDE scores are unnormalized radial similarities, not Gaussian probability PDFs',
             'D4+centroid translation shape geometry is a researcher-chosen representation, not a physical lottery variable',
             'No verified next-draw prediction or wagering advantage']}
    arrays={'draw_ids':ids,'numbers':ns,'embedding_maxdim':coords,'bandwidths':np.asarray(bws),
            'density_scores':scores.astype(np.float32),
            'baseline_labels':fits[comparison_dimension]['labels']}
    return report,arrays


def csv_records(report):
    """Long-form CSV rows for easy inspection and reproducible comparisons."""
    base={'cohort':'historical','replicate':0,'n':report['data']['draw_count']}
    for z in report['dimension_sensitivity']:
        yield {'kind':'dimension',**base,'dimensions':z['dimensions'],'best_k':z['best_k'],
             'bic_gain':z['bic_gain_vs_k1'],'stress':z['relative_stress'],
             'silhouette':z['silhouette']}
    for z in report['bandwidth_sensitivity']:
        yield {'kind':'bandwidth',**base,'bandwidth':z['bandwidth'],
           'density_cv':z['density']['cv'],'rank_corr':z['spearman_vs_smallest_bandwidth'],
           'top10_jaccard':z['top10pct_jaccard_vs_smallest'],
           'peak_count':z['selected_representatives'],'unassigned':z['unassigned_count']}
    for rep in report['matched_iid_controls']:
        for cohort,z in rep['cohorts'].items():
            for bw,metrics in z['density'].items():
                yield {'kind':'matched', 'cohort':cohort,'replicate':rep['replicate'],
                     'n':z['n'],'dimensions':z['mds_dimensions'],'bandwidth':bw,
                     'best_k':z['best_k'],'bic_gain':z['bic_gain_vs_k1'],
                     'density_cv':metrics['cv'],'silhouette':z['silhouette']}
    for rep in report['full_size_iid_controls']:
        yield {'kind':'full_iid','cohort':'iid','replicate':rep['replicate'],
             'n':rep['n'],'dimensions':rep['mds_dimensions'],
             'best_k':rep['iid_best_k'],'bic_gain':rep['iid_bic_gain_vs_k1'],
             'silhouette':rep['iid_silhouette']}
    for rep in report['stability']['runs']:
        yield {'kind':'stability','cohort':'historical','replicate':rep['seed'],
              'n':rep['samples_per_refit'],'overlap':rep['intersection_size'],
              'ari':rep['ari'],'nmi':rep['nmi']}


def html_report(report):
    from html import escape
    def fmt(x,precision=4):return '—' if x is None else f'{x:.{precision}f}' if isinstance(x,(float,np.floating)) else str(x)
    def table(headers,rows):
        h='<thead><tr>'+''.join(f'<th>{escape(str(s))}</th>' for s in headers)+'</tr></thead>'
        body='<tbody>'+''.join('<tr>'+''.join(f'<td>{escape(fmt(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody>'
        return '<table>'+h+body+'</table>'
    sections=['<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>Lotto7x7 P0 Gaussian Validation</title>',
       '<style>body{font:16px system-ui;max-width:1150px;margin:auto;padding:2rem;background:#f7f9fb;color:#273740}section{background:white;padding:1.1rem;border-radius:12px;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{border-bottom:1px solid #dde3e8;text-align:right;padding:.45rem}th:first-child,td:first-child{text-align:left}h1{color:#16776e}p{line-height:1.55}</style>',
       '<h1>Batch 1 / P0 — Gaussian Shape Model Validation</h1>',
       '<p>Descriptive comparisons, not evidence of improved winning odds. All GMMs use the same diagonal covariance, regularization, optimizer and candidate K grid within each matched trial.</p>']
    sections.append('<section><h2>Full historical MDS / GMM sensitivity</h2>')
    sections.append(table(['Dimensions','Stress','Selected K','BIC gain over K=1','Silhouette'],
      [[z['dimensions'],z['relative_stress'],z['best_k'],z['bic_gain_vs_k1'],z['silhouette']] for z in report['dimension_sensitivity']]))
    sections.append('</section><section><h2>Bandwidth sensitivity</h2>')
    sections.append(table(['Bandwidth','Density CV','Spearman to smallest','Top 10% overlap (Jaccard)','Peak prototypes','Unassigned'],
      [[z['bandwidth'],z['density']['cv'],z['spearman_vs_smallest_bandwidth'],z['top10pct_jaccard_vs_smallest'],
         z['selected_representatives'],z['unassigned_count']] for z in report['bandwidth_sensitivity']]))
    sections.append('</section><section><h2>Matched-size independent IID refits</h2>')
    matches=[]
    for z in report['matched_iid_controls']:
        a,b=z['cohorts']['historical'],z['cohorts']['iid']
        matches.append([z['replicate'],a['n'],a['mds_dimensions'],a['best_k'],b['best_k'],
             a['bic_gain_vs_k1'],b['bic_gain_vs_k1']])
    sections.append(table(['Trial','N each','MDS dimension','Historical best K','IID best K','History gain','IID gain'],matches))
    sections.append('</section><section><h2>Full-N independently drawn IID</h2>')
    sections.append(table(['Trial','N each','History K','IID K','History BIC gain','IID BIC gain'],
       [[z['replicate'],z['n'],z['historical_best_k'],z['iid_best_k'],
         z['historical_bic_gain_vs_k1'],z['iid_bic_gain_vs_k1']] for z in report['full_size_iid_controls']]))
    sections.append('</section><section><h2>GMM stability (overlapping 80% subsets)</h2>')
    sections.append(table(['Seed','Overlap','ARI','NMI'],
        [[z['seed'],z['intersection_size'],z['ari'],z['nmi']] for z in report['stability']['runs']]))
    sections.append(f'<p>Mean ARI: {report["stability"]["mean_ari"]:.4f}; Mean NMI: {report["stability"]["mean_nmi"]:.4f}. Fixed full-history MDS axes; transductive, not prospective.</p></section>')
    sections.append('<section><h2>Limits</h2><ul>'+''.join(f'<li>{escape(c)}</li>' for c in report['cautions'])+'</ul></section></html>')
    return '\n'.join(sections)
