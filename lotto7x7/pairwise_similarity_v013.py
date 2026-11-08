"""v0.13: GPU 6-point optimal-matching distances and threshold-bounded classes.

Distinguishes position-specific, D4 (right-angle rotation + reflection) and
D4 + centroid-translation-invariant *geometric* similarity. For 6 identical
unit-mass pixels the exact Hungarian/optimal-transport solution is obtained
by exhaustively minimizing over all 6! permutations on batched GPU tensors.
Continuous centroid alignment minimizes squared transport error but does not
imply a physically possible integer translation on the finite 7x7 grid.

All 6/49 lotteries remain equiprobable under an independent fair mechanism.
"""
from __future__ import annotations

from collections import Counter
from html import escape
from itertools import permutations
from math import comb, sqrt
from pathlib import Path
import hashlib

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .image_classification_v011 import iid_legal_images, numbers, validate

SCHEMA = 'lotto7x7.pairwise_optimal_matching.v0.13'
MODES = ('position', 'd4', 'shape', 'shape_hausdorff')
DEFAULT_QUANTILES = (0.001, 0.005, 0.01)
PERMUTATIONS = np.asarray(list(permutations(range(6))), dtype=np.int64)


def _device(value: str) -> torch.device:
    if value not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if value == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA but no CUDA device is available')
    return torch.device('cuda' if value == 'cuda' or value == 'auto' and torch.cuda.is_available() else 'cpu')


def point_coordinates(images: np.ndarray) -> np.ndarray:
    """Sorted 6x2 integer point arrays from a batch of legal 7x7 images."""
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (7, 7) or not np.all((arr == 0) | (arr == 1)) or not np.all(arr.sum(axis=(1, 2)) == 6):
        raise ValueError('expected legal [N,7,7] six-hot images')
    flat = arr.reshape(-1, 49).astype(bool)
    row, col = np.nonzero(flat)
    return np.stack((col // 7, col % 7), axis=1).reshape(len(arr), 6, 2).astype(np.float32)


def _d4_coordinates(points: torch.Tensor) -> torch.Tensor:
    """Eight isometries of the 7x7 board around (3,3), identity first."""
    x = points - 3.
    variants = []
    for flip in (False, True):
        p = x.clone()
        if flip:
            p[..., 1] = -p[..., 1]
        for turns in range(4):
            q = p
            for _ in range(turns):
                q = torch.stack((-q[..., 1], q[..., 0]), dim=-1)
            variants.append(q + 3.)
    return torch.stack(variants, dim=1)


def _point_distance_blocks(a: torch.Tensor, b: torch.Tensor, perms: torch.Tensor) -> dict[str, torch.Tensor]:
    """For B pairs, return four exact/defined distances in cell units.

    shape uses optimal translation of centroids (fractional cell displacement
    allowed). Shape Hausdorff is symmetric nearest-point Hausdorff after D4
    and centroid alignment (NOT integer nonwrapping lattice translation).
    """
    if a.ndim != 3 or b.shape != a.shape or a.shape[1:] != (6, 2):
        raise ValueError('two [B,6,2] batches required')
    if perms.shape != (720, 6):
        raise ValueError('expected 720 six-point permutations')
    transforms = _d4_coordinates(b)  # [B,8,6,2]
    # Pair-to-pair dot matrix, then exact 6! assignment (no Sinkhorn approx).
    dots = torch.einsum('bid,btjd->btij', a, transforms)  # [B,8,6,6]
    order = torch.arange(6, device=a.device)
    # [B,8,720,6]: select each of 720 one-to-one correspondence scores.
    selected = dots[:, :, order[None, :], perms]
    best_dot = selected.sum(dim=-1).amax(dim=-1)  # [B,8]
    norm_a = a.double().square().sum(dim=(1,2), keepdim=False)[:, None]
    norm_b = transforms.double().square().sum(dim=(2,3))
    squared = (norm_a + norm_b - 2. * best_dot).clamp_min(0.)
    means_a = a.double().mean(dim=1)[:,None,:]
    means_b = transforms.double().mean(dim=2)
    # Optimal free translation reduces the 6-point squared cost by
    # 6 * ||mu_A - mu_B||^2, independent of the chosen permutation.
    centroid_shift_cost = 6. * (means_a-means_b).square().sum(dim=-1)
    shape_squared = (squared - centroid_shift_cost).clamp_min(0.)
    position = torch.sqrt(squared[:,0] / 6.)
    d4 = torch.sqrt(squared.min(dim=1).values / 6.)
    shape = torch.sqrt(shape_squared.min(dim=1).values / 6.)
    # Hausdorff after same centroid alignment, using actual distances.
    centered_a = a.double() - a.double().mean(dim=1, keepdim=True)
    centered_b = transforms.double() - means_b[:,:,None,:]
    delta = centered_a[:,None,:,None,:] - centered_b[:,:,None,:,:]
    squared_all = delta.square().sum(dim=-1).clamp_min(0.)
    directed_a = squared_all.min(dim=3).values.amax(dim=2)
    directed_b = squared_all.min(dim=2).values.amax(dim=2)
    haus = torch.sqrt(torch.maximum(directed_a,directed_b).min(dim=1).values)
    return {'position':position, 'd4':d4, 'shape':shape, 'shape_hausdorff':haus}


def pair_distances(points_a: np.ndarray, points_b: np.ndarray, *, device='auto', batch_size=128, modes=MODES) -> dict[str,np.ndarray]:
    """Compute exact 6-point assignment distances for corresponding pairs.

    Runtime: O(pairs * 8 * 6! * 6), with memory bounded by batch_size.
    All returned distances are symmetric, nonnegative, and in cell units.
    """
    a,b = np.asarray(points_a),np.asarray(points_b)
    if a.ndim!=3 or a.shape[1:]!=(6,2) or b.shape!=a.shape or batch_size<1 or not set(modes).issubset(MODES):
        raise ValueError('need paired [B,6,2] points and supported distance modes')
    dev=_device(device)
    perms=torch.as_tensor(PERMUTATIONS,device=dev,dtype=torch.long)
    out={m:np.empty(len(a),dtype=np.float32) for m in modes}
    for lo in range(0,len(a),batch_size):
        hi=min(len(a),lo+batch_size)
        aa=torch.as_tensor(a[lo:hi],device=dev,dtype=torch.float32)
        bb=torch.as_tensor(b[lo:hi],device=dev,dtype=torch.float32)
        with torch.no_grad():
            chunk=_point_distance_blocks(aa,bb,perms)
            for m in modes:out[m][lo:hi]=chunk[m].cpu().numpy()
    return out


def pairwise_matrices(points: np.ndarray, *, device='auto',batch_size=128,modes=MODES,on_progress=None) -> dict[str,np.ndarray]:
    """All unordered pairs, including 0-valued diagonals; no pair subsampling."""
    pts=np.asarray(points)
    if pts.ndim!=3 or pts.shape[1:]!=(6,2) or len(pts)<2:
        raise ValueError('need at least two six-point images')
    n=len(pts)
    # 2497 draws => 3,116,256 exact pair computations.
    ix,jx=np.triu_indices(n,k=1)
    out={m:np.zeros((n,n),dtype=np.float32) for m in modes}
    # Cache both CUDA device and the 720 assignment permutations outside
    # the pair loop to avoid thousands of repeated host-to-device copies.
    dev=_device(device)
    perms=torch.as_tensor(PERMUTATIONS,device=dev,dtype=torch.long)
    for lo in range(0,len(ix),batch_size):
        hi=min(len(ix),lo+batch_size)
        aa=torch.as_tensor(pts[ix[lo:hi]],device=dev,dtype=torch.float32)
        bb=torch.as_tensor(pts[jx[lo:hi]],device=dev,dtype=torch.float32)
        with torch.no_grad():
            batch=_point_distance_blocks(aa,bb,perms)
        for mode in modes:
            ds=batch[mode].cpu().numpy().astype(np.float32,copy=False)
            out[mode][ix[lo:hi],jx[lo:hi]]=ds
            out[mode][jx[lo:hi],ix[lo:hi]]=ds
        if on_progress and (lo==0 or hi==len(ix) or hi//(batch_size*100)>(lo//(batch_size*100))):
            on_progress(f'[PAIRWISE] {hi}/{len(ix)}')
    return out


def distance_stats(matrix:np.ndarray, *, iid_distances:np.ndarray|None=None, iqs=DEFAULT_QUANTILES) ->dict:
    n=len(matrix)
    real=matrix[np.triu_indices(n,k=1)]
    info={'pair_count':int(len(real)),'mean':float(real.mean()),
          'quantiles':{str(q):float(np.quantile(real,q)) for q in (0.,.001,.005,.01,.05,.1,.25,.5,.75,.9,.99,1.)}}
    if iid_distances is not None:
        iid=np.asarray(iid_distances)
        info['iid_pair_count']=int(len(iid))
        info['iid_mean']=float(iid.mean())
        info['iid_quantiles']={str(q):float(np.quantile(iid,q)) for q in (0.,.001,.005,.01,.05,.1,.25,.5,.75,.9,.99,1.)}
        info['predeclared_iid_thresholds']=[{'target_quantile':float(q),'distance':float(t),
            'historical_pair_share_leq':float(np.mean(real<=t+1e-5)),
            'iid_pair_share_leq':float(np.mean(iid<=t+1e-5))}
            for q in iqs for t in [np.quantile(iid,q)]]
    return info



def _aligned_shape_pair(a:np.ndarray,b:np.ndarray)->dict:
    """Describe the best D4 and centroid overlay for a selected pair.

    The expensive SciPy assignment is used only for a handful of top pairs,
    NEVER for the N(N-1)/2 exact pairwise GPU computation.
    """
    from scipy.optimize import linear_sum_assignment
    ref=np.asarray(a,dtype=float)
    moving=torch.as_tensor(np.asarray(b,dtype=np.float64),dtype=torch.float64).unsqueeze(0)
    variants=_d4_coordinates(moving)[0].numpy()
    best=None
    for t,pts in enumerate(variants):
        aligned=pts+ref.mean(axis=0)-pts.mean(axis=0)
        costs=np.square(ref[:,None,:]-aligned[None,:,:]).sum(axis=-1)
        rows,cols=linear_sum_assignment(costs)
        rms=float(np.sqrt(costs[rows,cols].sum()/6))
        if best is None or rms<best['distance']-1e-9:
            best={'d4_transform_index':t,'distance':rms,
                  'translation':(ref.mean(axis=0)-pts.mean(axis=0)).tolist(),
                  'reference_points':ref.tolist(),'aligned_points':aligned.tolist()}
    return best


def _closest_pairs(matrix:np.ndarray, points:np.ndarray,draw_ids:np.ndarray, top:int=20, align=False) ->list[dict]:
    if top<1:return []
    i,j=np.triu_indices(len(points),k=1)
    d=matrix[i,j]
    chosen=np.argpartition(d,min(top,len(d))-1)[:min(top,len(d))]
    chosen=chosen[np.lexsort((j[chosen],i[chosen],d[chosen]))]
    result=[]
    for k in chosen:
        a,b=int(i[k]),int(j[k])
        vals_a=sorted((points[a,:,0].astype(int)*7+points[a,:,1].astype(int)+1).tolist())
        vals_b=sorted((points[b,:,0].astype(int)*7+points[b,:,1].astype(int)+1).tolist())
        row={'draw_id_a':int(draw_ids[a]),'draw_id_b':int(draw_ids[b]),
             'numbers_a':vals_a,'numbers_b':vals_b,
             'shared_number_count':int(len(set(vals_a)&set(vals_b))),
             'distance':float(d[k])}
        if align:row['aligned_shape_overlay']=_aligned_shape_pair(points[a],points[b])
        result.append(row)
    return result


def _threshold_classes(matrix:np.ndarray,draw_ids:np.ndarray,points:np.ndarray, thresholds:list[tuple[float,float]], *,gallery_count=6,max_groups=20):
    """Complete-link classes guarantee every within-class pair <= threshold."""
    n=len(matrix)
    # SciPy complete linkage is deterministic for a fixed input order.
    condensed=squareform(matrix.astype(np.float64,copy=False),checks=False)
    tree=linkage(condensed,method='complete',optimal_ordering=False)
    results=[];assignments={}
    for q,th in thresholds:
        raw=fcluster(tree,t=float(th)+1e-7,criterion='distance')
        unique,inv,counts=np.unique(raw,return_inverse=True,return_counts=True)
        members=[np.flatnonzero(inv==k) for k in range(len(unique))]
        members=sorted(members,key=lambda m:(-len(m),int(m[0])))
        multi=[m for m in members if len(m)>=2]
        dist_q=matrix[np.triu_indices(n,1)]
        pairs=int(np.count_nonzero(dist_q<=th+1e-5))
        group_list=[]
        for ordinal,m in enumerate(multi[:max_groups],1):
            segment=matrix[np.ix_(m,m)]
            medoid=int(m[np.argmin(segment.sum(axis=1))])
            nearest=m[np.argsort(matrix[medoid,m],kind='stable')[:gallery_count]]
            maximum=float(segment.max())
            group_list.append({'class_id':f'C{ordinal:03d}','count':int(len(m)),
                'representative_draw_id':int(draw_ids[medoid]),
                'representative_numbers':sorted((points[medoid,:,0].astype(int)*7+points[medoid,:,1].astype(int)+1).tolist()),
                'diameter':maximum,
                'examples':[{'draw_id':int(draw_ids[t]),'numbers':sorted((points[t,:,0].astype(int)*7+points[t,:,1].astype(int)+1).tolist()),
                             'distance_to_medoid':float(matrix[medoid,t])} for t in nearest]})
        labels=np.array(['SINGLETON']*n,dtype=object)
        for ordinal,m in enumerate(multi,1):labels[m]=f'C{ordinal:03d}'
        results.append({'target_iid_quantile':float(q),'threshold':float(th),
               'pairs_within_threshold':pairs,'pair_share':pairs/len(dist_q),
               'multi_member_class_count':len(multi),
               'multi_member_images':int(sum(map(len,multi))),
               'singleton_images':int(sum(len(m)==1 for m in members)),
               'largest_class':int(len(members[0])),
               'classes_shown':group_list,
               'class_method':'complete_linkage_all_pairs_within_threshold'})
        assignments[str(q)]=labels.tolist()
    return results,assignments


def iid_reference_pairs(count:int,*,seed:int,device:str,batch_size:int,modes=MODES):
    if count<100:raise ValueError('iid_pairs count must be at least 100')
    rng=np.random.default_rng(seed)
    # independent uniformly chosen legal 6/49 pairs
    draws=iid_legal_images(count*2,rng)
    pts=point_coordinates(draws)
    return pair_distances(pts[:count],pts[count:],device=device,batch_size=batch_size,modes=modes)


def analyze(images:np.ndarray,draw_ids:np.ndarray,*,device='auto',batch_size=128,
    iid_pairs=30000, iid_quantiles=DEFAULT_QUANTILES, cluster_mode='shape',
    control_draws=384, control_replicates=9,top_pairs=25,gallery_count=8,
    seed=649,on_progress=None):
    arr,ids=validate(images,draw_ids)
    if (cluster_mode not in MODES or iid_pairs<100 or batch_size<1 or control_draws<12 or
        control_draws>len(arr) or control_replicates<0 or top_pairs<1 or gallery_count<1 or seed<0 or
        any(not 0.<q<1. for q in iid_quantiles) or len(set(iid_quantiles))!=len(iid_quantiles)):
        raise ValueError('invalid similarity settings')
    dev=str(_device(device))
    pts=point_coordinates(arr)
    pair_matrices=pairwise_matrices(pts,device=dev,batch_size=batch_size,on_progress=on_progress)
    if on_progress:on_progress('[NULL] independent legal 6/49 pair distances')
    null_pairs=iid_reference_pairs(iid_pairs,seed=seed+391,device=dev,batch_size=batch_size)
    distributions={m:distance_stats(pair_matrices[m],iid_distances=null_pairs[m],iqs=iid_quantiles) for m in MODES}
    thresholds=[(float(q),float(np.quantile(null_pairs[cluster_mode],q))) for q in sorted(iid_quantiles)]
    classes,assignments=_threshold_classes(pair_matrices[cluster_mode],ids,pts,thresholds,gallery_count=gallery_count)
    if on_progress:on_progress('[CLUSTERS] complete linkage on actual image distances')
    # Null refits are full independent legal-draw *subsets*, same size as the
    # sampled actual subsets. The original 2497x2497 distance matrix is reused.
    # Thresholds were fixed from independent IID pairs above, not chosen on real.
    rng=np.random.default_rng(seed+29)
    diagnostic=[]
    for rep in range(control_replicates):
        take=np.sort(rng.choice(len(arr),size=control_draws,replace=False))
        real=pair_matrices[cluster_mode][np.ix_(take,take)]
        synth=iid_legal_images(control_draws,rng)
        synth_pts=point_coordinates(synth)
        sim=pairwise_matrices(synth_pts,device=dev,batch_size=batch_size,modes=(cluster_mode,))[cluster_mode]
        dummy_ids=np.arange(1,control_draws+1)
        real_classes,_=_threshold_classes(real,dummy_ids,pts[take],thresholds,gallery_count=1,max_groups=0)
        sim_classes,_=_threshold_classes(sim,dummy_ids,synth_pts,thresholds,gallery_count=1,max_groups=0)
        diagnostic.append({'replicate':rep+1,'observed_sample':[{'quantile':a['target_iid_quantile'],
            'multi_member_images':a['multi_member_images'],'class_count':a['multi_member_class_count'],
            'pairs':a['pairs_within_threshold']} for a in real_classes],
            'iid_sample':[{'quantile':a['target_iid_quantile'],
            'multi_member_images':a['multi_member_images'],'class_count':a['multi_member_class_count'],
            'pairs':a['pairs_within_threshold']} for a in sim_classes]})
        if on_progress:on_progress(f'[IID SUBSET REFIT] {rep+1}/{control_replicates}')
    closest={m:_closest_pairs(pair_matrices[m],pts,ids,top=top_pairs,align=(m=='shape')) for m in MODES}
    consistency={'pos_ge_d4':bool(np.all(pair_matrices['position']+5e-4>=pair_matrices['d4'])),
                 'd4_ge_shape':bool(np.all(pair_matrices['d4']+5e-4>=pair_matrices['shape'])),
                 'symmetry':bool(all(np.allclose(d,d.T,rtol=0,atol=5e-5) for d in pair_matrices.values())),
                 'diagonal_zero':bool(all(np.all(np.diag(d)==0) for d in pair_matrices.values()))}
    if not all(consistency.values()):raise AssertionError('pairwise distance invariants failed: '+str(consistency))
    report={'schema':SCHEMA,'research_only':True,
       'source_sha256':hashlib.sha256(arr.tobytes()+ids.tobytes()).hexdigest(),
       'first_draw_id':int(ids[0]),'last_draw_id':int(ids[-1]),'draw_count':len(arr),
       'unordered_pair_count':int(len(arr)*(len(arr)-1)//2),
       'device':dev,'settings':{'batch_size':batch_size,'iid_pairs':iid_pairs,
           'iid_quantiles':list(iid_quantiles),'cluster_mode':cluster_mode,
           'control_draws':control_draws,'control_replicates':control_replicates,
           'top_pairs':top_pairs,'gallery_count':gallery_count,'seed':seed},
       'distance_definitions':{
           'position':'optimal one-to-one six-point RMS distance on original 7x7 coordinates',
           'd4':'minimum position RMS after 8 rotations/reflections around board center; no translation',
           'shape':'minimum RMS after D4 and free continuous centroid translation, not integer-grid-constrained',
           'shape_hausdorff':'symmetric Hausdorff after D4 and centroid translation, not optimal six-point assignment'},
       'distance_distributions':distributions,
       'closest_pairs':closest,
       'threshold_classes':classes,'assignments_at_threshold':assignments,
       'iid_matched_subset_refits':diagnostic,
       'sanity':consistency,
       'warnings':['Descriptive similarity, not a validated lottery selection edge.',
         'The 7x7 number layout is arbitrary; rotations, reflections and centering do not correspond to the lottery mechanism.',
         'Continuous centroid alignment may move pixels off the integer grid; it is a shape distance, not an exact translated lattice-image equivalence.',
         'Complete-link clusters (unlike connected components) have diameter at most the threshold, including after transitive merges.',
         'Quantiles chosen from independent IID pairs; null-refit diagnostics use small subsamples, not a whole-search significance test.',
         'Closest-pair and cutoff searches inspect the same historical data; future lottery numbers are not made more likely by visual similarity.']}
    return report,pair_matrices


def _bitmap(nums:list[int],*, width=110):
    active=set(nums)
    rects=[]
    for i in range(49):
        x=(i%7)*15+1;y=(i//7)*15+1
        rects.append(f'<rect x="{x}" y="{y}" width="13" height="13" rx="1.4" fill="{("#19877e" if i+1 in active else "#e1e6ea")}"/>')
    return f'<svg viewBox="0 0 106 106" width="{width}" aria-label="seven by seven image">'+''.join(rects)+'</svg>'



def _overlay_svg(overlay:dict)->str:
    """Plot both point sets in aligned *continuous* 2D coordinates."""
    def circles(items,color):
        return ''.join(f'<circle cx="{10+15*p[1]:.2f}" cy="{10+15*p[0]:.2f}" r="5.8" fill="{color}" fill-opacity=".62"/>' for p in items)
    return ('<svg viewBox="-18 -18 142 142" width="110" aria-label="shape-aligned overlay">'
            +circles(overlay['reference_points'],'#197f79')
            +circles(overlay['aligned_points'],'#dc8840')+'</svg>')


def html_gallery(report:dict)->str:
    panels=[]
    for mode in MODES:
        pairs=report['closest_pairs'][mode][:18]
        cells=[]
        for i,p in enumerate(pairs,1):
            overlay=p.get('aligned_shape_overlay')
            overlay_html=('<div>'+_overlay_svg(overlay)+'<small>Aligned overlay</small></div>') if overlay else ''
            cells.append('<article class="pair"><strong>#'+str(i)+' · d='+f'{p["distance"]:.3f}'+'</strong><div class="twins">'+
                         '<div>'+_bitmap(p['numbers_a'])+'<small>'+str(p['draw_id_a'])+'</small></div>'+
                         '<div>'+_bitmap(p['numbers_b'])+'<small>'+str(p['draw_id_b'])+'</small></div>'+
                         overlay_html+'</div>'+f'<p>Common actual numbers: {p["shared_number_count"]}</p></article>')
        panels.append('<section><h2>'+escape(mode)+'</h2><div class="grid">'+''.join(cells)+'</div></section>')
    for section in report['threshold_classes']:
        cards=[]
        for g in section['classes_shown']:
            examples=''.join('<div>'+_bitmap(ex['numbers'],width=75)+'<small>'+str(ex['draw_id'])+'</small></div>' for ex in g['examples'])
            cards.append('<article><h3>'+g['class_id']+' · '+str(g['count'])+' images</h3><p>Diameter '+f'{g["diameter"]:.3f}'+' cell RMS</p>'+_bitmap(g['representative_numbers'])+'<div class="examples">'+examples+'</div></article>')
        panels.append('<section><h2>Complete-link classes · fair IID quantile '+str(section['target_iid_quantile'])+'</h2>'+f'<p>Threshold {section["threshold"]:.4f} · {section["multi_member_class_count"]} non-singleton groups · {section["multi_member_images"]} member images · {section["singleton_images"]} singletons</p>'+'<div class="grid">'+''.join(cards)+'</div></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Lotto7x7 v0.13 Exact Geometric Similarity</title><style>body{background:#f7f9fb;color:#26333c;font-family:system-ui;margin:2rem auto;padding:0 1rem;max-width:1500px}h1{font-size:1.65rem}section{border-top:2px solid #ccd7e0;margin:1.6rem 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:.8rem}article{background:#fff;border:1px solid #dce2e7;border-radius:12px;padding:.85rem}.twins,.examples{display:flex;gap:.7rem;flex-wrap:wrap}.twins>div,.examples>div{display:flex;flex-direction:column}small{color:#637786}h3{font-size:.95rem}p{color:#60717e;font-size:.85rem}</style>'
    '<h1>GPU Optimal-Assignment 7×7 Image Similarity</h1><p>All image pairs · exact six-point matching · fair IID pair baseline · no selection claim</p>' + ''.join(panels)+'</html>')
