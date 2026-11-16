"""v0.9 GPU-first multi-domain discovery of *candidate* periodic image features.

Works on a strictly historical prefix and calibrates the whole scan using two
Monte Carlo null ensembles: fair independent legal 6-of-49 draws and shuffled
observed draws. No future draw prediction or claimed lottery advantage.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "lotto7x7.feature_discovery.v0.9"


def resolve_device(mode: str) -> torch.device:
    if mode not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto/cuda/cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda" if mode == "cuda" or (mode == "auto" and torch.cuda.is_available()) else "cpu")


def validate_images(images: np.ndarray) -> np.ndarray:
    a = np.asarray(images)
    if (a.ndim != 3 or a.shape[1:] != (7,7) or len(a) < 70
            or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
            or not np.logical_or(a == 0, a == 1).all()
            or not (a.sum(axis=(1,2)) == 6).all()):
        raise ValueError("requires >=70 ordered legal 7x7 six-hot images")
    return a.astype(np.float32,copy=True)


def _dct_basis(*, device: torch.device) -> torch.Tensor:
    # Orthonormal cosine basis for the 7x7 spatial DCT-II, fixed by geometry.
    i = torch.arange(7, dtype=torch.float32, device=device)
    k = torch.arange(7, dtype=torch.float32, device=device)
    c = torch.cos(math.pi / 7.0 * (i[None,:] + .5) * k[:,None])
    c[0,:] *= math.sqrt(1.0/7.0)
    c[1:,:] *= math.sqrt(2.0/7.0)
    return torch.kron(c,c)


def _feature_bank(tensor: torch.Tensor) -> tuple[torch.Tensor,list[str],list[str]]:
    """Features ending at each known draw; triplet difference uses only <=t.

    Accept [B,T,7,7], return [B,T-2,F] on the original device. Every feature
    transformation is deterministic and uses no fitted parameters.
    """
    if tensor.ndim != 4 or tensor.shape[2:] != (7,7) or tensor.shape[1] < 6:
        raise ValueError("feature tensor must be [batch,time>=6,7,7]")
    b,t,_,_ = tensor.shape
    x = tensor.to(dtype=torch.float32)
    flat = x.reshape(b,t,49)
    parts: list[torch.Tensor] = []
    names: list[str] = []
    families: list[str] = []

    def add(values: torch.Tensor, family: str, columns: list[str]):
        if values.ndim != 3 or values.shape[:2] != (b,t-2) or len(columns) != values.shape[-1]:
            raise AssertionError("feature implementation dimensional mismatch")
        parts.append(values)
        names.extend(f"{family}/{column}" for column in columns)
        families.extend([family]*len(columns))

    add(flat[:,2:], "raw", [f"number_{i:02}" for i in range(1,50)])
    add(flat[:,2:]-flat[:,1:-1], "signed_diff",
        [f"number_{i:02}" for i in range(1,50)])
    add(flat[:,2:]-2*flat[:,1:-1]+flat[:,:-2], "central_second_diff",
        [f"number_{i:02}" for i in range(1,50)])

    dct = _dct_basis(device=x.device)
    cos = flat[:,2:] @ dct.T
    add(cos[:,:,1:],"spatial_dct",[f"coefficient_{i:02}" for i in range(1,49)])

    fft = torch.fft.rfft2(x[:,2:]).abs().square().reshape(b,t-2,-1)
    # Excluding DC; individual FFT bins may be related by conjugate symmetry.
    add(torch.log1p(fft[:,:,1:]),"spatial_fft_power",
        [f"coefficient_{i:02}" for i in range(1,28)])

    current = x[:,2:]
    coordinates = torch.arange(7,device=x.device,dtype=x.dtype)
    r = coordinates.view(1,1,7,1)
    c = coordinates.view(1,1,1,7)
    row_sum=current.sum(-1)
    col_sum=current.sum(-2)
    mr=(current*r).sum(dim=(-1,-2))/6
    mc=(current*c).sum(dim=(-1,-2))/6
    varr=(current*(r-mr[:,:,None,None]).square()).sum(dim=(-1,-2))/6
    varc=(current*(c-mc[:,:,None,None]).square()).sum(dim=(-1,-2))/6
    covariance=(current*(r-mr[:,:,None,None])*(c-mc[:,:,None,None])).sum(dim=(-1,-2))/6
    row_prob=row_sum/6
    col_prob=col_sum/6
    er=-(row_prob*row_prob.clamp(min=1.e-10).log()).sum(dim=-1)
    ec=-(col_prob*col_prob.clamp(min=1.e-10).log()).sum(dim=-1)
    neighbors_h=(current[:,:,:,:-1]*current[:,:,:,1:]).sum(dim=(-1,-2))
    neighbors_v=(current[:,:,:-1,:]*current[:,:,1:,:]).sum(dim=(-1,-2))
    neighbors_diag=(current[:,:,:-1,:-1]*current[:,:,1:,1:]).sum(dim=(-1,-2))
    neighbors_antidiag=(current[:,:,:-1,1:]*current[:,:,1:,:-1]).sum(dim=(-1,-2))
    geo=torch.stack([mr,mc,varr,varc,covariance,er,ec,
                     row_sum.square().sum(-1),col_sum.square().sum(-1),
                     neighbors_h,neighbors_v,neighbors_diag,neighbors_antidiag],dim=-1)
    add(geo,"geometry_graph",["centroid_row","centroid_col","variance_row","variance_col",
                              "covariance","row_entropy","col_entropy","row_occupancy_concentration",
                              "col_occupancy_concentration","horizontal_edges","vertical_edges",
                              "diag_edges","antidiag_edges"])

    # Candidate 7x7 graph pairwise displacement motifs (15 nonredundant scales).
    pair_features = []
    pair_names=[]
    for dr in range(4):
        for dc in range(4):
            if dr==0 and dc==0:
                continue
            v=(current[:,:,:7-dr,:7-dc]*current[:,:,dr:,dc:]).sum(dim=(-1,-2))
            pair_features.append(v)
            pair_names.append(f"pair_offset_{dr}_{dc}")
    add(torch.stack(pair_features,dim=-1),"pair_motif",pair_names)

    # Spatial edge / wavelet-like filter bank; these encode differential structure.
    kernels=torch.tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]],
                          [[[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]],
                          [[[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]]],
                          [[[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]]]],
                         device=x.device,dtype=x.dtype)
    conv=F.conv2d(current.reshape(b*(t-2),1,7,7),kernels,padding=1)
    energy=conv.square().mean(dim=(-1,-2)).reshape(b,t-2,4)
    add(energy,"spatial_filter_energy",["sobel_x","sobel_y","laplacian_4","laplacian_8"])

    output=torch.cat(parts,dim=-1)
    if not torch.isfinite(output).all():
        raise AssertionError("nonfinite generated features")
    return output,names,families


def iid_six_hot(batch:int, draws:int, *, generator:torch.Generator,
                device:torch.device) -> torch.Tensor:
    """Legal IID 6-out-of-49 without replacement, sampled entirely on device."""
    if batch<1 or draws<1:
        raise ValueError("positive IID batch and draw counts required")
    order=torch.rand((batch,draws,49),generator=generator,device=device).topk(6,dim=-1).indices
    result=torch.zeros((batch,draws,49),device=device,dtype=torch.float32)
    return result.scatter_(2,order,1.0).reshape(batch,draws,7,7)


def shuffled_history(observed:torch.Tensor,batch:int,*,generator:torch.Generator)->torch.Tensor:
    """Same historical draw multiset, independently permuted chronological order."""
    if observed.ndim!=3 or observed.shape[1:]!=(7,7) or batch<1:
        raise ValueError("expected historical images [time,7,7]")
    n=len(observed)
    ranks=torch.rand((batch,n),generator=generator,device=observed.device)
    idx=ranks.argsort(dim=1)
    return observed[idx]


def _scan_periods(bank:torch.Tensor, names:list[str], *, recent_windows:tuple[int,...],
                  half_lives:tuple[int,...], min_period:float,max_period:float,
                  top_count:int) -> tuple[torch.Tensor,list[dict]]:
    """Maximum relative periodogram peak across all searched feature/period/radius/window.

    Z-plane mapping: for time-reversed age k, r=e^(ln2/h), and exponential
    weighting r^(-k) times Fourier factor e^(-j omega k). Windowed spectrum
    is a finite-sample Z-transform evaluated on that circle.
    """
    if bank.ndim!=3 or bank.shape[-1]!=len(names):
        raise ValueError("bank and name shape mismatch")
    b,n,f=bank.shape
    winners=[]
    global_max=torch.zeros(b,dtype=torch.float32,device=bank.device)
    windows=tuple(dict.fromkeys([n]+[min(n,x) for x in recent_windows if x>=16]))
    for w in windows:
        if w<min_period*2:
            continue
        segment=bank[:,-w:,:]
        centered=segment-segment.mean(dim=1,keepdim=True)
        std=centered.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1.e-5)
        norm=centered/std
        freqs=torch.fft.rfftfreq(w,d=1.,device=bank.device)
        ok=(freqs>0.)&(freqs>=1.0/max_period)&(freqs<=1.0/min_period)
        if not ok.any():
            continue
        selected_freqs=freqs[ok]
        for h in half_lives:
            ages=torch.arange(w-1,-1,-1,device=bank.device,dtype=torch.float32)
            if h>0:
                weights=torch.exp((-math.log(2.)/h)*ages)
                weights=weights/(weights.square().mean().sqrt().clamp_min(1.e-8))
                z=norm*weights.view(1,w,1)
            else:
                z=norm
            power=torch.fft.rfft(z,dim=1).abs().square()[:,ok,:]
            strength=power/(power.sum(dim=1,keepdim=True).clamp_min(1.e-12))
            best,strength_freq_ix=strength.max(dim=1)
            max_strength,feature_ix=best.max(dim=1)
            global_max=torch.maximum(global_max,max_strength)
            if top_count>0 and b==1:
                values,top_ix=best[0].topk(min(top_count,f))
                for v,fi in zip(values.tolist(),top_ix.tolist()):
                    k=int(strength_freq_ix[0,fi])
                    winners.append({"feature":names[fi],"feature_index":fi,
                                    "strength":float(v),"window":w,"z_half_life":int(h),
                                    "frequency_cycles_per_draw":float(selected_freqs[k]),
                                    "period_draws":float(1.0/selected_freqs[k])})
    winners.sort(key=lambda x:x['strength'],reverse=True)
    # Present distinct feature channels; the underlying multiplicity correction
    # still covers EVERY feature/window/radius/period tested, including those
    # not listed in the top candidate summary.
    distinct = {}
    for row in winners:
        distinct.setdefault(row['feature'], row)
    return global_max,list(distinct.values())[:top_count]


def discover(images:np.ndarray,draw_ids:np.ndarray,*,device="auto",holdout_draws=500,
             surrogates=199,surrogate_batch=8,recent_windows=(256,512,1024),
             half_lives=(0,64,256),min_period=2.,max_period=512.,
             top_count=20,seed=649,on_progress:Callable[[str],None]|None=None)->dict:
    """GPU-wide multiplicity-controlled feature periodicity discovery, no model fitting.

    Global max-null calibration accounts for all searched features, periods,
    recent windows, and Z radii. Two independently calibrated null ensembles:
    IID legal 6/49 images, and permutations of the observed draw multiset.
    Holdout suffix is never touched by feature search.
    """
    x=validate_images(images)
    ids=np.asarray(draw_ids)
    if (ids.shape!=(len(x),) or not np.issubdtype(ids.dtype,np.integer) or
            np.any(np.diff(ids)<=0) or np.any(ids<=0)):
        raise ValueError("invalid chronological draw IDs")
    if not (type(holdout_draws)==int and 0<=holdout_draws<len(x)-70 and
            type(surrogates)==int and surrogates>=9 and
            type(surrogate_batch)==int and surrogate_batch>=1 and
            type(seed)==int and seed>=0 and type(top_count)==int and top_count>=1 and
            2<=min_period<max_period and
            len(recent_windows)>0 and all(type(k)==int and k>=16 for k in recent_windows) and
            len(half_lives)>0 and all(type(h)==int and h>=0 for h in half_lives)):
        raise ValueError("invalid feature-discovery configuration")
    dev=resolve_device(device)
    n=len(x)-holdout_draws
    observed=torch.as_tensor(x[:n],device=dev)
    bank,names,families=_feature_bank(observed[None])
    observed_stat,candidates=_scan_periods(bank,names,recent_windows=recent_windows,
                                          half_lives=half_lives,min_period=min_period,
                                          max_period=max_period,top_count=top_count)
    maxima={"iid_legal_6of49":[],"shuffled_historical_draws":[]}
    gens={key:torch.Generator(device=dev).manual_seed(seed+offset) for key,offset in
          [("iid_legal_6of49",101),("shuffled_historical_draws",202)]}
    for label in maxima:
        for start in range(0,surrogates,surrogate_batch):
            amount=min(surrogate_batch,surrogates-start)
            if label=="iid_legal_6of49":
                artificial=iid_six_hot(amount,n,generator=gens[label],device=dev)
            else:
                artificial=shuffled_history(observed,amount,generator=gens[label])
            artificial_bank,_,_=_feature_bank(artificial)
            values,_=_scan_periods(artificial_bank,names,recent_windows=recent_windows,
                                  half_lives=half_lives,min_period=min_period,
                                  max_period=max_period,top_count=0)
            maxima[label].extend(float(v) for v in values.detach().cpu())
            if on_progress and (start+amount==surrogates or (start//surrogate_batch)%8==0):
                on_progress(f"[NULL] {label} {start+amount}/{surrogates}")
    for candidate in candidates:
        for label,values in maxima.items():
            count=sum(v>=candidate['strength']-1e-8 for v in values)
            candidate[f"global_max_p_{label}"]=(count+1)/(surrogates+1)
        candidate['passes_both_global_005']=(all(candidate[f'global_max_p_{label}']<.05
                                                  for label in maxima))
    families_summary=dict(sorted(Counter(families).items()))
    return {"schema":SCHEMA,"status":"EXPLORATORY_FEATURE_DISCOVERY_ONLY",
            "device":str(dev),"gpu_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else None,
            "dataset":{"source_draws":len(x),"discovery_prefix_draws":n,
                       "holdout_suffix_draws":holdout_draws,
                       "discovery_first_draw_id":int(ids[0]),
                       "discovery_last_draw_id":int(ids[n-1]),
                       "source_last_draw_id":int(ids[-1]),
                       "full_history_sha256":hashlib.sha256(x.tobytes()+ids.astype('<i8').tobytes()).hexdigest(),
                       "discovery_prefix_sha256":hashlib.sha256(x[:n].tobytes()+ids[:n].astype('<i8').tobytes()).hexdigest()},
            "feature_families":families_summary,
            "feature_count":len(names),"feature_names":names,
            "periodicity_scan":{"recent_windows":list(recent_windows),
                                 "z_plane_exponential_half_lives":list(half_lives),
                                 "min_period":min_period,"max_period":max_period,
                                 "global_observed_max":float(observed_stat[0]),
                                 "method":"peak relative finite-window Z-transform power"},
            "null_controls":{label:{"surrogates":surrogates,
                                    "global_max_mean":float(np.mean(values)),
                                    "global_max_quantile95":float(np.quantile(values,.95)),
                                    "global_max_p_for_best_observed":float((1+sum(v>=float(observed_stat[0])-1e-8 for v in values))/(surrogates+1))}
                             for label,values in maxima.items()},
            "top_candidates":candidates,
            "warnings":["Feature periodicity is not a validated next-draw predictive edge.",
                        "The entire family/window/frequency/Z-radius search is corrected via global maximum surrogates.",
                        "IID legal draws and shuffled actual histories answer different null hypotheses.",
                        "Differencing mechanically creates spectral structure; comparison with the same transformed null is essential.",
                        "A frequency measured in draw index is not a real-time/calendar frequency.",
                        "The holdout is reserved in this experiment but prior work may already have inspected it.",
                        "Any discovered candidate requires separate prefix-only model selection and forward-only validation.",
                        "All spatial grid relations depend on a human-defined number-to-pixel mapping."]}
