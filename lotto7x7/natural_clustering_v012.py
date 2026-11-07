"""v0.12: natural 7x7 six-hot image classes, stability, and random-null controls.

Clustering is descriptive. Density noise is not forced into arbitrary classes.
The 7x7 mapping is an arbitrary representation of six unordered lottery labels.
"""
from __future__ import annotations

from collections import Counter
from html import escape
from math import sqrt
from pathlib import Path
import hashlib

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, cut_tree

from .image_classification_v011 import (
    _embeddings, _normalize_features, iid_legal_images, morphology, numbers, validate,
)

SCHEMA = 'lotto7x7.natural_image_clustering.v0.12'
DEFAULT_K = (4, 6, 8, 12, 16, 24, 32)


def device_name(device: str) -> str:
    if device not in ('cuda', 'cpu', 'auto'):
        raise ValueError('invalid device')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return 'cuda' if (device == 'cuda' or device == 'auto' and torch.cuda.is_available()) else 'cpu'


def feature_space(images: np.ndarray, *, feature_mode: str, checkpoint: Path | None,
                  device: str, batch_size: int = 1024, reference: dict | None = None
                  ) -> tuple[np.ndarray, dict]:
    """Build D4-invariant morphology or frozen CNN/fusion features; no fitting CNN."""
    if feature_mode not in ('geometry', 'cnn', 'fused') or (feature_mode != 'geometry' and checkpoint is None):
        raise ValueError('invalid feature mode or missing CNN checkpoint')
    raw, _ = morphology(images)
    geometry_ref = None if reference is None else reference['geometry']
    g, gr = _normalize_features(raw, geometry_ref)
    r = {'geometry': gr, 'cnn': None}
    if feature_mode == 'geometry':
        return g, r
    deep = _embeddings(images, Path(checkpoint), device=torch.device(device), batch_size=batch_size)
    if reference is None:
        mn, std = deep.mean(axis=0), np.maximum(deep.std(axis=0), 1e-5)
    else:
        mn, std = reference['cnn']
    z = ((deep-mn)/std/sqrt(deep.shape[1])).astype(np.float32)
    r['cnn'] = (mn, std)
    return (z if feature_mode == 'cnn' else np.column_stack((g,z))).astype(np.float32), r


def pairwise_gpu(features: np.ndarray, *, device: str, batch_size: int = 512) -> np.ndarray:
    """One pairwise Euclidean distance matrix, CUDA-blocked when available."""
    d = device_name(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=d)
    if x.ndim != 2 or len(x) < 3 or not bool(torch.isfinite(x).all()):
        raise ValueError('invalid feature matrix')
    n = len(x)
    out = np.empty((n,n), dtype=np.float32)
    with torch.no_grad():
        for i in range(0,n,batch_size):
            out[i:i+batch_size] = torch.cdist(x[i:i+batch_size], x).cpu().numpy()
    # Finite precision cdist can have tiny nonzero diagonal or asymmetric distances.
    out = (out+out.T)*np.float32(.5)
    np.fill_diagonal(out,0.)
    return out


def adjusted_silhouette(distance: np.ndarray, labels: np.ndarray, *, sample: int,
                        seed: int) -> float | None:
    labels = np.asarray(labels)
    allowed = np.flatnonzero(labels >= 0)
    distinct = np.unique(labels[allowed])
    if len(distinct)<2 or len(allowed) <= len(distinct):
        return None
    if len(allowed)>sample:
        rng = np.random.default_rng(seed)
        # Sample within every class to avoid missing small minority groups.
        per_cluster = []
        for v in distinct:
            group = allowed[labels[allowed] == v]
            per_cluster.append(int(group[rng.integers(len(group))]))
        if len(per_cluster)>=sample:
            return None
        remaining=np.setdiff1d(allowed,np.array(per_cluster),assume_unique=False)
        idx=np.concatenate((per_cluster,rng.choice(remaining,size=sample-len(per_cluster),replace=False)))
    else:
        idx=allowed
    subset=distance[np.ix_(idx,idx)].astype(np.float64,copy=True)
    np.fill_diagonal(subset,0.)
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(subset,labels[idx],metric='precomputed'))


def ward_candidates(features: np.ndarray, distance: np.ndarray, k_values: tuple[int,...],
                    *, sample: int, seed: int) -> tuple[dict,dict[int,np.ndarray]]:
    ks=tuple(sorted(set(int(k) for k in k_values if 2<=k<len(features))))
    if not ks:
        raise ValueError('need at least one candidate class count below draw count')
    tree=linkage(features.astype(np.float64,copy=False),method='ward',metric='euclidean',optimal_ordering=False)
    matrix=cut_tree(tree,n_clusters=list(ks))
    rows=[]; saved={}
    for j,k in enumerate(ks):
        labels=matrix[:,j].astype(np.int32)
        saved[k]=labels
        s=adjusted_silhouette(distance,labels,sample=sample,seed=seed)
        rows.append({'k':int(k),'silhouette':s, 'min_class_size':int(np.bincount(labels).min())})
    # Winner = best score only within this predefined family of K values.
    best=max(rows,key=lambda row:(-np.inf if row['silhouette'] is None else row['silhouette'],-row['k']))
    return {'candidate_scores':rows,'selected_k':best['k'],'selected_silhouette':best['silhouette'],
            'selection_rule':'best_sampled_silhouette_within_explicit_k_grid_not_an_intrinsic_class_count'},saved


def density_hdbscan(distance: np.ndarray, *, min_cluster_size: int, min_samples: int):
    from sklearn.cluster import HDBSCAN
    model=HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,
                  metric='precomputed',cluster_selection_method='eom',allow_single_cluster=False,copy=True)
    labels=model.fit_predict(np.asarray(distance,dtype=np.float64))
    probs=np.asarray(model.probabilities_,dtype=np.float32)
    return np.asarray(labels,dtype=np.int32),probs


def kmedoids(distance: np.ndarray, k:int, *, seed:int, max_iter:int=25) -> tuple[np.ndarray,np.ndarray]:
    """PAM-style alternate assign/update using actual training images as centers."""
    n=len(distance)
    if not 2<=k<n:
        raise ValueError('invalid medoid count')
    rng=np.random.default_rng(seed)
    medoids=[int(rng.integers(n))]
    nearest=np.square(distance[:,medoids[0]])
    for _ in range(1,k):
        weights=nearest.copy();weights[medoids]=0.
        if weights.sum() <= 1e-12:
            medoid=int(rng.choice(np.setdiff1d(np.arange(n),medoids)))
        else:
            medoid=int(rng.choice(n,p=weights/weights.sum()))
        medoids.append(medoid)
        nearest=np.minimum(nearest,np.square(distance[:,medoid]))
    medoids=np.array(medoids,dtype=int)
    for _ in range(max_iter):
        labels=np.argmin(distance[:,medoids],axis=1)
        new=medoids.copy()
        for j in range(k):
            idx=np.flatnonzero(labels==j)
            if len(idx):
                row_sum=distance[np.ix_(idx,idx)].sum(axis=1)
                new[j]=idx[np.argmin(row_sum)]
            else:
                far=np.min(distance[:,medoids],axis=1)
                far[medoids]=-1
                new[j]=int(np.argmax(far))
        if np.array_equal(new,medoids):
            break
        medoids=new
    labels=np.argmin(distance[:,medoids],axis=1).astype(np.int32)
    return labels,medoids


def class_summary(images:np.ndarray,draw_ids:np.ndarray,distance:np.ndarray,
                  labels:np.ndarray, *, method:str, gallery_count:int=10,
                  fitted_medoids: np.ndarray | None=None,
                  confidences:np.ndarray | None=None) -> tuple[list[dict],list[dict]]:
    n=len(labels); families=[]
    vals=[int(v) for v in sorted(np.unique(labels)) if v>=0]
    items=[]
    for v in vals:
        idx=np.flatnonzero(labels==v)
        if fitted_medoids is not None:
            center=int(fitted_medoids[v])
        else:
            # Exact medoid of cluster in the searched metric; a real observed drawing.
            center=int(idx[np.argmin(distance[np.ix_(idx,idx)].sum(axis=1))])
        order=idx[np.argsort(distance[center,idx],kind='stable')]
        near=order[:gallery_count]
        far=order[-gallery_count:][::-1]
        d=distance[center,idx]
        vals_near=np.sort(d)
        family={'class_id':f'C{len(items)+1:03}', 'raw_label':v,'count':len(idx),'share':len(idx)/n,
                'representative_index':center,'representative_draw_id':int(draw_ids[center]),
                'representative_numbers':numbers(images[center]),
                'mean_distance_to_medoid':float(d.mean()),'p90_distance_to_medoid':float(np.quantile(d,.9)),
                'max_distance_to_medoid':float(d.max()),
                'within_class_nearest_distances':{'p50':float(np.median(vals_near)),'p90':float(np.quantile(vals_near,.9))},
                'gallery_nearest':[{'draw_id':int(draw_ids[i]),'numbers':numbers(images[i]),'distance':float(distance[center,i])} for i in near],
                'gallery_boundary':[{'draw_id':int(draw_ids[i]),'numbers':numbers(images[i]),'distance':float(distance[center,i])} for i in far]}
        if confidences is not None:
            family['mean_density_membership_strength']=float(confidences[idx].mean())
        items.append(family)
    items.sort(key=lambda g:(-g['count'],g['raw_label']))
    remap={g['raw_label']:f'C{i+1:03}' for i,g in enumerate(items)}
    for i,g in enumerate(items):g['class_id']=f'C{i+1:03}'
    for i in range(n):
        label=int(labels[i])
        families.append({'draw_id':int(draw_ids[i]),'class_id':remap.get(label,'NOISE'),
                         'numbers':numbers(images[i]),
                         'membership_strength':float(confidences[i]) if confidences is not None else None})
    return items,families


def stability_subsamples(features:np.ndarray, labels:np.ndarray, *, method:str,
                         k:int|None,min_cluster_size:int,min_samples:int,
                         repeats:int, fraction:float,seed:int, device:str) -> dict:
    if repeats<1:return {'ari':None,'valid_comparisons':0,'note':'disabled'}
    from sklearn.metrics import adjusted_rand_score
    n=len(features); rng=np.random.default_rng(seed); scores=[];covered=[]
    for rep in range(repeats):
        take=np.sort(rng.choice(n,size=max(6,int(round(n*fraction))),replace=False))
        if method=='ward':
            tree=linkage(features[take].astype(float),method='ward')
            sub=cut_tree(tree,n_clusters=[k])[:,0]
        else:
            dist=pairwise_gpu(features[take],device=device)
            if method=='hdbscan':
                sub,_=density_hdbscan(dist,min_cluster_size=max(2,int(round(min_cluster_size*fraction))),
                                      min_samples=min_samples)
            else:
                sub,_=kmedoids(dist,k,seed=seed+rep)
        total_classes=len(set(sub[sub>=0]))
        if method=='hdbscan' and (total_classes<2 or len(set(labels[take][labels[take]>=0]))<2):
            continue
        if method=='hdbscan':
            # Noise is a legitimate categorical state for ARI, but also report coverage.
            covered.append(float(np.mean(sub>=0)))
        scores.append(float(adjusted_rand_score(labels[take],sub)))
    return {'ari':float(np.mean(scores)) if scores else None,
            'ari_each':scores,'valid_comparisons':len(scores),
            'subsample_fraction':fraction,'mean_subsample_coverage':float(np.mean(covered)) if covered else None,
            'note':'ARI comparing full-fit labels to independent subset refits on shared indices; high ARI does not prove nonrandom categories'}


def _run_methods(features, distance, *, ks, min_cluster_size,min_samples,silhouette_sample,
                 seed, stability_repeats,stability_fraction,device, include_kmedoids):
    ward_report, ward_labels=ward_candidates(features,distance,ks,sample=silhouette_sample,seed=seed)
    ward_k=int(ward_report['selected_k']); wc=ward_labels[ward_k]
    hlabels,hprob=density_hdbscan(distance,min_cluster_size=min_cluster_size,min_samples=min_samples)
    methods={}
    for name,labels,k,medoids,conf in [('hdbscan',hlabels,None,None,hprob),('ward',wc,ward_k,None,None)]:
        c=len(set(labels[labels>=0])); coverage=float(np.mean(labels>=0));
        methods[name]={'labels':labels,'medoids':medoids,'confidences':conf,
                       'class_count':int(c),'noise_count':int(np.sum(labels<0)),
                       'coverage':coverage,'silhouette':adjusted_silhouette(distance,labels,sample=silhouette_sample,seed=seed),
                       'stability':stability_subsamples(features,labels,method=name,k=k,min_cluster_size=min_cluster_size,
                           min_samples=min_samples,repeats=stability_repeats,fraction=stability_fraction,
                           seed=seed+991,device=device)}
    if include_kmedoids:
        ml,mmed=kmedoids(distance,ward_k,seed=seed)
        methods['kmedoids']={'labels':ml,'medoids':mmed,'confidences':None,
                            'class_count':int(len(np.unique(ml))),'noise_count':0,'coverage':1.,
                            'silhouette':adjusted_silhouette(distance,ml,sample=silhouette_sample,seed=seed),
                            'stability':stability_subsamples(features,ml,method='kmedoids',k=ward_k,
                                min_cluster_size=min_cluster_size,min_samples=min_samples,repeats=stability_repeats,
                                fraction=stability_fraction,seed=seed+991,device=device)}
    return ward_report,methods


def report(images:np.ndarray,draw_ids:np.ndarray,*,feature_mode='geometry',checkpoint:Path|None=None,
           device='auto', k_values=DEFAULT_K,min_cluster_size=30,min_samples=10,
           silhouette_sample=1200,stability_repeats=2,stability_fraction=.8,
           iid_replicates=9,iid_samples=10000,batch_size=1024,gallery_count=10,
           seed=649,on_progress=None)->dict:
    x,ids=validate(images,draw_ids); n=len(x);dev=device_name(device)
    if not (2<=min_cluster_size<n and 1<=min_samples<n and silhouette_sample>=32
            and stability_repeats>=0 and 0<stability_fraction<1 and
            iid_replicates>=0 and iid_samples>=100 and batch_size>=1 and
            gallery_count>=1 and seed>=0):
        raise ValueError('invalid parameters')
    ks=tuple(sorted({int(k) for k in k_values if 2<=int(k)<n}))
    if not ks:raise ValueError('empty valid candidate class counts')
    z,reference=feature_space(x,feature_mode=feature_mode,checkpoint=checkpoint,device=dev,batch_size=batch_size)
    d=pairwise_gpu(z,device=dev,batch_size=batch_size)
    ward,methods=_run_methods(z,d,ks=ks,min_cluster_size=min_cluster_size,min_samples=min_samples,
                      silhouette_sample=silhouette_sample,seed=seed,stability_repeats=stability_repeats,
                      stability_fraction=stability_fraction,device=dev,include_kmedoids=True)
    summary={}
    for key,v in methods.items():
        classes,assign=class_summary(x,ids,d,v['labels'],method=key,gallery_count=gallery_count,
                                     fitted_medoids=v['medoids'],confidences=v['confidences'])
        noise_indices=np.flatnonzero(v['labels']<0)[:gallery_count]
        noise_examples=[{'draw_id':int(ids[i]),'numbers':numbers(x[i])} for i in noise_indices]
        size_hist=Counter(c['count'] for c in classes)
        summary[key]={'class_count':v['class_count'],'noise_count':v['noise_count'],'coverage':v['coverage'],
                      'silhouette':v['silhouette'],'stability':v['stability'],
                      'class_size_histogram':{str(k):int(q) for k,q in sorted(size_hist.items())},
                      'noise_examples':noise_examples,'classes':classes,'assignments':assign}
    observed_metrics={key:{'class_count':v['class_count'],'noise_count':v['noise_count'],
                           'coverage':v['coverage'],'silhouette':v['silhouette']} for key,v in methods.items()}
    if on_progress:on_progress(f'[OBSERVED] method class_count noise silhouette '+str({k:(v['class_count'],v['noise_count'],v['silhouette']) for k,v in observed_metrics.items()}))
    # Full-refit null for cluster existence/quality; this does NOT treat
    # k selected from real data as if it had been predeclared.
    null_rows=[];rng=np.random.default_rng(seed+3009)
    for rep in range(iid_replicates):
        synthetic=iid_legal_images(n,rng)
        syn_features,_=feature_space(synthetic,feature_mode=feature_mode,checkpoint=checkpoint,device=dev,batch_size=batch_size)
        sd=pairwise_gpu(syn_features,device=dev,batch_size=batch_size)
        nw,nm=_run_methods(syn_features,sd,ks=ks,min_cluster_size=min_cluster_size,
                           min_samples=min_samples,silhouette_sample=silhouette_sample,
                           seed=seed+rep+100,stability_repeats=0,stability_fraction=stability_fraction,
                           device=dev,include_kmedoids=False)
        null_rows.append({'ward_best_silhouette':nw['selected_silhouette'],
                          'ward_selected_k':nw['selected_k'],
                          'hdbscan_class_count':nm['hdbscan']['class_count'],
                          'hdbscan_coverage':nm['hdbscan']['coverage'],
                          'hdbscan_silhouette':nm['hdbscan']['silhouette']})
        if on_progress and (rep==0 or (rep+1)%max(1,iid_replicates//4)==0):
            on_progress(f'[IID REFIT] {rep+1}/{iid_replicates}')
    def empirical_p(values, observed):
        if observed is None:return None
        vals=[float(v) for v in values if v is not None]
        if not vals:return None
        return (1+sum(v>=observed-1e-10 for v in vals))/(len(vals)+1)
    controls={
      'replicates':iid_replicates,
      'ward_selected_best_silhouette_empirical_p':empirical_p([r['ward_best_silhouette'] for r in null_rows],ward['selected_silhouette']),
      'hdbscan_class_count_empirical_p':empirical_p([r['hdbscan_class_count'] for r in null_rows],observed_metrics['hdbscan']['class_count']),
      'hdbscan_coverage_empirical_p':empirical_p([r['hdbscan_coverage'] for r in null_rows],observed_metrics['hdbscan']['coverage']),
      'hdbscan_silhouette_empirical_p':empirical_p([r['hdbscan_silhouette'] for r in null_rows],observed_metrics['hdbscan']['silhouette']),
      'null_summary':null_rows,
      'limitations':'Monte Carlo exploratory controls; small replicate count has coarse p-value resolution. Frozen CNN encoder is not refitted per IID data; null is conditional. Do not infer lottery advantage.'}
    # A separate fixed-medoid IID descriptive *class distribution* reference.
    ref_counts={name:np.zeros(len(v['classes']),dtype=np.int64) for name,v in summary.items()}
    # Ward/k-medoids: nearest observed medoid. The resulting Voronoi cells
    # are NOT the same as the observed hierarchical/PAM assignments.
    # Report actual cells under the same distance rule for a valid comparison.
    # native prediction; report distance profiles only, never force noise to class.
    realmedoids={name:np.asarray([r['representative_index'] for r in v['classes']],dtype=int) for name,v in summary.items()}
    for i in range(0,iid_samples,batch_size):
        synthetic=iid_legal_images(min(batch_size,iid_samples-i),rng)
        syn_z,_=feature_space(synthetic,feature_mode=feature_mode,checkpoint=checkpoint,
                               device=dev,batch_size=batch_size,reference=reference)
        # Distances measured in same fixed historical feature space.
        for name,medoid_ids in realmedoids.items():
            if name=='hdbscan':continue  # HDBSCAN has no standard centroid assignment.
            if len(medoid_ids)==0:continue
            dst=torch.cdist(torch.as_tensor(syn_z,device=dev),torch.as_tensor(z[medoid_ids],device=dev)).argmin(dim=1).cpu().numpy()
            ref_counts[name]+=np.bincount(dst,minlength=len(medoid_ids))
    for name in ('ward','kmedoids'):
        medoid_ids=realmedoids[name]
        historical_fixed=torch.cdist(torch.as_tensor(z,device=dev),torch.as_tensor(z[medoid_ids],device=dev)).argmin(dim=1).cpu().numpy()
        historical_voronoi=np.bincount(historical_fixed,minlength=len(medoid_ids))
        for k,g in enumerate(summary[name]['classes']):
            g['fixed_medoid_historical_share']=float(historical_voronoi[k]/n)
            g['fixed_medoid_iid_share']=float(ref_counts[name][k]/iid_samples)
            g['fixed_medoid_iid_expected_count']=float(ref_counts[name][k]*n/iid_samples)
    result={'schema':SCHEMA,'research_only':True,'source_sha256':hashlib.sha256(x.tobytes()+ids.tobytes()).hexdigest(),
            'draw_count':n,'first_draw_id':int(ids[0]),'last_draw_id':int(ids[-1]),
            'features':feature_mode,'device':dev,'checkpoint_sha256':(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint is not None else None),
            'settings':{'k_values':list(ks),'hdbscan_min_cluster_size':min_cluster_size,'hdbscan_min_samples':min_samples,
                        'silhouette_sample':silhouette_sample,'stability_repeats':stability_repeats,'stability_fraction':stability_fraction,
                        'iid_replicates':iid_replicates,'iid_samples':iid_samples,'gallery_count':gallery_count,'seed':seed},
            'ward_model_selection':ward,'methods':summary,'iid_full_refit_controls':controls,
            'cautions':['Exploratory classification, not next-draw prediction.',
                        'Ward K selected from a finite tested grid; no claim of a unique natural number of classes.',
                        'HDBSCAN noise means unclassified; a zero-class result is valid.',
                        'D4 geometry ignores orientation and position; frozen CNN may not be rotation/translation invariant.',
                        'Fair 6/49 draws can exhibit highly structured image clusters under any chosen metric.',
                        'IID nearest-medoid fractions must be compared to historical nearest-medoid fractions, not to Ward/PAM raw class member counts.',
                        'HDBSCAN native out-of-sample assignment is deliberately omitted; no forced assignment of IID draws.',
                        'Existing v0.10 checkpoint is research-only and not known predictive.']}
    return result


def html_gallery(item:dict)->str:
    """Standalone gallery: 7x7 medoid, nearest/boundary actual historical samples."""
    def mini(nums):
        selected=set(nums)
        cells=[]
        for p in range(49):
            active=p+1 in selected
            cells.append(f'<rect x="{p%7*12+1}" y="{p//7*12+1}" width="10" height="10" rx="1.3" fill="{("#0e807a" if active else "#e2e7ea")}"/>')
        return '<svg viewBox="0 0 84 84" role="img" aria-label="seven by seven six point bitmap">'+''.join(cells)+'</svg>'
    rows=[]
    for name,group in item['methods'].items():
        families=[]
        for cl in group['classes']:
            near=''.join('<div class="sample">'+mini(v['numbers'])+f'<small>{v["draw_id"]} · d={v["distance"]:.2f}</small></div>' for v in cl['gallery_nearest'])
            far=''.join('<div class="sample">'+mini(v['numbers'])+f'<small>{v["draw_id"]} · d={v["distance"]:.2f}</small></div>' for v in cl['gallery_boundary'])
            fair='' if 'fixed_medoid_iid_share' not in cl else f' · nearest-medoid regions (history/IID): {cl["fixed_medoid_historical_share"]:.1%} / {cl["fixed_medoid_iid_share"]:.1%}'
            families.append('<details class="class" '+('open' if len(families)<3 else '')+'><summary>'+escape(cl['class_id'])+f' · {cl["count"]} images ({cl["share"]:.1%})'+fair+'</summary>'+
                            '<div class="represent">'+mini(cl['representative_numbers'])+
                            f'<div><strong>Actual medoid</strong><p>Draw {cl["representative_draw_id"]}: {escape(str(cl["representative_numbers"]))}</p>'+
                            f'<p>Mean radius {cl["mean_distance_to_medoid"]:.3f} · p90 {cl["p90_distance_to_medoid"]:.3f} · max {cl["max_distance_to_medoid"]:.3f}</p></div></div>'+
                            '<h4>Nearest historical images</h4><div class="samples">'+near+'</div><h4>Class boundary images (most distant from medoid)</h4><div class="samples">'+far+'</div></details>')
        noise=''.join('<div class="sample">'+mini(v['numbers'])+f'<small>{v["draw_id"]} · NOISE</small></div>' for v in group['noise_examples'])
        noise_block='<details class="class"><summary>UNCLASSIFIED / NOISE · '+str(group['noise_count'])+' drawings</summary><p>HDBSCAN does not force these drawings into any group. Examples:</p><div class="samples">'+noise+'</div></details>' if group['noise_count'] else ''
        score=group['silhouette']; quality=('n/a' if score is None else f'{score:.3f}')
        stability=group['stability']['ari']; stable=('n/a' if stability is None else f'{stability:.3f}')
        rows.append(f'<section><h2>{escape(name.upper())}</h2><p>{group["class_count"]} classes · {group["noise_count"]} noise · coverage {group["coverage"]:.1%} · silhouette {quality} · subsample ARI {stable}</p>'+''.join(families)+noise_block+'</section>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lotto 7×7 Natural Shape Clustering</title>
<style>body{max-width:1450px;margin:2rem auto;padding:0 1rem;font-family:system-ui, sans-serif;color:#26333e;background:#f5f8fa}h1{font-size:1.8rem}h2{margin-top:2.2rem}p{color:#596b76}section{border-top:2px solid #ccd9de}details.class{background:white;border:1px solid #dbe2e6;border-radius:12px;margin:.7rem 0;padding:1rem}summary{font-weight:600;cursor:pointer}.represent{display:flex;align-items:center;gap:1rem;padding:1rem 0}.represent svg{width:115px;flex-shrink:0}.samples{display:flex;flex-wrap:wrap;gap:.6rem}.sample{width:91px;display:flex;flex-direction:column;gap:4px;border:1px solid #edf1f2;border-radius:6px;padding:4px}.sample svg{width:84px}.sample small{font-size:.63rem;color:#637b89}h4{margin:1rem 0 .3rem;font-size:.88rem}@media(max-width:650px){.represent{display:block}}</style>
<h1>7×7 Natural Image Similarity &amp; Quantification</h1>'''+f'<p>{item["draw_count"]} historical draws · {escape(item["features"])} features · {escape(item["device"])} · not winning-number probabilities</p>'+''.join(rows)+'</html>'
