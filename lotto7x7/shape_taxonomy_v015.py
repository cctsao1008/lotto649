"""P1 v0.15: discrete multiscale image-shape taxonomy with exact stable codes.

Exploratory density modes are *separate* from the frozen deterministic graph
codes used for exact 6/49 theoretical frequencies. Gaussian radial similarity
on shape distances is a score, not a normalized Gaussian density PDF.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from math import comb
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


UNIVERSE=comb(49,6)
BANDWIDTHS=(0.35,0.55,0.8,1.1)
RADII=(0.70,0.75,0.80)
POPCOUNT=np.asarray([int(i).bit_count() for i in range(64)],dtype=np.uint8)
BITS=np.asarray([1<<i for i in range(6)],dtype=np.uint8)
I,J=np.triu_indices(6,1)


def classify_numbers(numbers: np.ndarray) -> dict[str,np.ndarray]:
    """D4-invariant, origin-independent exact combinatorial graph classes.

    Returns coded topology of all six points on the original 7x7 board:
    L1 = 8-neighbor connected component count;
    L2 = descending multiset of component sizes (integer packed 6x3 bits);
    L3 = L2 plus cardinal and diagonal 8-neighbor edge counts.

    Vectorized Nx6 closure, constant working memory per input chunk. Pure NumPy;
    no optional JIT/compiler/CUDA dependency for exact universe enumeration.
    """
    a=np.asarray(numbers)
    if a.ndim!=2 or a.shape[1]!=6 or len(a)<1:
        raise ValueError('expected [N,6] legal numbers 1..49')
    if not np.issubdtype(a.dtype,np.integer) or np.any(a<1) or np.any(a>49):
        raise ValueError('numbers must be 1..49 integers')
    a=np.sort(a.astype(np.uint8,copy=False),axis=1)
    if np.any(a[:,1:]==a[:,:-1]):raise ValueError('numbers must be distinct')
    pos=a-1
    rr=(pos//7).astype(np.int8)
    cc=(pos%7).astype(np.int8)
    dr=np.abs(rr[:,:,None]-rr[:,None,:]); dc=np.abs(cc[:,:,None]-cc[:,None,:])
    neighbor=(dr<=1)&(dc<=1)
    reach=np.sum(neighbor*BITS[None,None,:],axis=2,dtype=np.uint8)
    # Warshall transitive closure, using one pivot vertex per pass.
    for pivot in range(6):
        connected=(reach & BITS[pivot]) != 0
        reach |= np.where(connected,reach[:,pivot,None],np.uint8(0))
    # Only the lowest-index member represents each component.
    first=(reach & (BITS-1)[None,:])==0
    component_sizes=np.sort(POPCOUNT[reach]*first,axis=1)[:,::-1].astype(np.uint32)
    shifts=np.asarray([15,12,9,6,3,0],dtype=np.uint32)
    partition=np.bitwise_or.reduce(component_sizes << shifts[None,:],axis=1).astype(np.uint32)
    cardinal=np.count_nonzero((((dr==1)&(dc==0))|((dr==0)&(dc==1)))[:,I,J],axis=1).astype(np.uint8)
    diagonal=np.count_nonzero(((dr==1)&(dc==1))[:,I,J],axis=1).astype(np.uint8)
    code=((partition << 10) | (cardinal.astype(np.uint32)<<5) | diagonal.astype(np.uint32)).astype(np.uint32)
    ncomp=np.count_nonzero(component_sizes,axis=1).astype(np.uint8)
    return {'n_components':ncomp,'partition_code':partition,'edge_code':code,
            'cardinal_edges':cardinal,'diagonal_edges':diagonal}


def decode_partition(code:int) -> tuple[int,...]:
    vals=[int((int(code)>>(3*(5-i)))&7) for i in range(6)]
    result=tuple(x for x in vals if x)
    if sum(result)!=6 or tuple(sorted(result,reverse=True))!=result:
        raise ValueError('invalid graph partition code')
    return result


def partition_name(code:int)->str:
    return 'P'+('-'.join(map(str,decode_partition(code))))


def edge_name(code:int)->str:
    pc=int(code)>>10
    return f'{partition_name(pc)}_O{(int(code)>>5)&31}_D{int(code)&31}'


def shape_key(numbers: Iterable[int])->str:
    a=np.asarray(list(numbers),dtype=np.int64)
    if a.shape!=(6,) or len(set(a.tolist()))!=6 or a.min()<1 or a.max()>49:
        raise ValueError('need six distinct 1..49 numbers')
    points=np.stack(((a-1)//7,(a-1)%7),axis=1)
    from .image_classification_v011 import canonical
    return canonical(points)


def exact_shape_families(numbers:np.ndarray,draw_ids:np.ndarray)->tuple[list[dict],np.ndarray]:
    """Observed exact D4+translation families; exact orbit mass is analytic.

    All unobserved families are exactly accounted for in one explicit remainder
    class; the finite fair universe size is not estimated from Monte Carlo.
    """
    from .image_classification_v011 import orbit_size
    a=np.asarray(numbers)
    ids=np.asarray(draw_ids)
    if ids.shape!=(len(a),) or len(set(map(int,ids)))!=len(ids):raise ValueError('draw IDs')
    codes=classify_numbers(a)
    groups=defaultdict(list)
    for i,row in enumerate(a):groups[shape_key(row)].append(i)
    families=[]
    family_labels=np.empty(len(a),dtype='<U20')
    for key,indices in groups.items():
        digest=sha256(key.encode('utf-8')).hexdigest()[:16]
        name='S'+digest
        i=indices[0]
        orbit=int(orbit_size(key))
        if not (1<=orbit<=UNIVERSE):raise AssertionError('illegal orbit size')
        fam={'shape_id':name,'canonical_key':key,'count':len(indices),
             'theory_combinations':orbit,'theory_probability':orbit/UNIVERSE,
             'expected_draws':len(a)*orbit/UNIVERSE,
             'representative_draw_id':int(ids[i]),
             'representative_numbers':list(map(int,a[i])),
             'component_count':int(codes['n_components'][i]),
             'partition':partition_name(int(codes['partition_code'][i])),
             'edge_class':edge_name(int(codes['edge_code'][i])),
             'member_draw_ids':[int(ids[j]) for j in indices]}
        families.append(fam)
        family_labels[indices]=name
    families.sort(key=lambda r:(-r['count'],r['shape_id']))
    orbit_total=sum(r['theory_combinations'] for r in families)
    if orbit_total>UNIVERSE:raise AssertionError('overlapping orbits')
    return families,family_labels


def sample_local_modes(distance:np.ndarray,scores:np.ndarray,radius:float)->dict:
    """All empirical discrete KDE local maxima in closed metric balls.

    This is a peak among the *observed points* at fixed radius, NOT an
    analytic continuous-kernel stationary point; no arbitrary cap on modes.
    """
    d=np.asarray(distance,dtype=np.float32)
    y=np.asarray(scores,dtype=np.float64)
    n=len(y)
    if n<3 or d.shape!=(n,n) or not np.isfinite(d).all() or not np.isfinite(y).all() or radius<=0:
        raise ValueError('invalid distance or density')
    # Deterministic ties by lowest index: this breaks equal-density duplicate images.
    maxima=np.ones(n,dtype=bool)
    for i in range(n):
        within=(d[i]<=radius+1e-7)
        better=(y>y[i]+1e-10)|((abs(y-y[i])<=1e-10)&(np.arange(n)<i))
        if np.any(within&better):maxima[i]=False
    peaks=np.flatnonzero(maxima)
    rank=peaks[np.argsort(-y[peaks],kind='stable')]
    return {'peaks':rank,'peak_count':int(len(rank))}


def nearest_mode_assignments(distance:np.ndarray,peaks:np.ndarray,assignment_radius:float)->tuple[np.ndarray,np.ndarray]:
    if len(peaks)==0 or assignment_radius<=0:raise ValueError('positive radius and nonempty peaks')
    nearest=np.argmin(distance[:,peaks],axis=1)
    nearest_dist=distance[np.arange(len(distance)),peaks[nearest]]
    labels=np.where(nearest_dist<=assignment_radius+1e-7,nearest,-1).astype(np.int32)
    for j,p in enumerate(peaks):labels[p]=j
    return labels,nearest_dist


def mode_stability_multiscale(distance:np.ndarray,density:np.ndarray,bandwidths=BANDWIDTHS,
                              radii=RADII,assignment_radius=0.85)->tuple[dict,dict]:
    """Enumerate every discrete local mode at all bandwidths x radii.

    All settings remain descriptive; no peak count is asserted to be natural.
    Main display uses median bandwidth index and median radius index.
    """
    d=np.asarray(distance)
    score=np.asarray(density)
    if score.shape!=(len(d),len(bandwidths)):raise ValueError('density matrix alignment')
    radius_values=sorted({float(x) for x in radii})
    if not radius_values or any(x<=0 for x in radius_values):raise ValueError('radii')
    records=[];all_peaks={};arrangements={}
    for col,bw in enumerate(bandwidths):
        for radius in radius_values:
            modes=sample_local_modes(d,score[:,col],radius)
            peaks=modes['peaks']
            labels,mindist=nearest_mode_assignments(d,peaks,assignment_radius)
            records.append({'bandwidth':float(bw),'radius':radius,
                 'peak_count':len(peaks),'assigned':int(np.count_nonzero(labels>=0)),
                 'unassigned':int(np.count_nonzero(labels<0)),
                 'max_class_size':int(np.bincount(labels[labels>=0]).max()) if np.any(labels>=0) else 0})
            all_peaks[(float(bw),radius)]=set(map(int,peaks))
            arrangements[(float(bw),radius)]=(peaks,labels,mindist)
    main_bw=float(bandwidths[len(bandwidths)//2-1 if len(bandwidths)%2==0 else len(bandwidths)//2])
    main_radius=radius_values[len(radius_values)//2]
    main_peaks,main_labels,main_dist=arrangements[(main_bw,main_radius)]
    stability=[]
    for p in main_peaks:
        persistence=0
        for alt_peaks in all_peaks.values():
            if not alt_peaks:continue
            if float(d[int(p),list(alt_peaks)].min())<=main_radius+1e-7:
                persistence+=1
        stability.append({'index':int(p),'presence_rate':persistence/len(all_peaks),
                          'density':float(score[int(p),list(bandwidths).index(main_bw)])})
    summary={'records':records,'main_bandwidth':main_bw,'main_radius':main_radius,
          'main_peak_count':len(main_peaks),'main_unassigned':int(np.count_nonzero(main_labels<0)),
          'main_assigned':int(np.count_nonzero(main_labels>=0)),
          'note':'Empirical discrete maxima only; radius and bandwidth change mode count; nearest-mode classes are not diameter-bounded'}
    arrays={'peaks':main_peaks.astype(np.int32),'labels':main_labels,
            'nearest_distance':main_dist,'persistence':stability}
    return summary,arrays


def pairwise_diameter_metrics(distance:np.ndarray,labels:np.ndarray,max_classes:int=1000)->list[dict]:
    """Exact diameter and average within/between distance of finite groups.

    Only grouped labels 0..max_classes (ignore -1 noise), sampling no pairs.
    """
    d=np.asarray(distance);lab=np.asarray(labels)
    unique=[int(x) for x in np.unique(lab) if x>=0]
    if len(unique)>max_classes:raise ValueError('too many clusters')
    result=[]
    for k in unique:
        idx=np.flatnonzero(lab==k)
        sub=d[np.ix_(idx,idx)]
        tri=sub[np.triu_indices(len(idx),k=1)]
        ext=d[np.ix_(idx,np.flatnonzero((lab>=0)&(lab!=k)))]
        result.append({'class_index':k,'size':int(len(idx)),'diameter':float(tri.max()) if len(tri) else 0.,
                       'mean_within':float(tri.mean()) if len(tri) else 0.,
                       'nearest_other_min':float(ext.min()) if ext.size else None})
    return result
