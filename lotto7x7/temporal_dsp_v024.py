"""v0.24 P3: immutable six-pixel spatial spectra, temporal periodograms, IID nulls.

Every simulated drawing contains six distinct numbers (not 49 independent
Bernoulli pixels). Alternative raster mappings are explicit and deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, rfft, rfftfreq

from .temporal_dataset_v024 import TemporalDataset, rasterize
from .temporal_features_v024 import temporal_features,generate_iid_numbers

SCHEMA='lotto7x7.temporal_dsp.v0.24'
KINDS=('sequential','snake','permuted')


def layout_indices(kind:str, *,seed:int=649)->np.ndarray:
    """New raster index -> original sequential number index, a bijection."""
    if kind=='sequential':return np.arange(49)
    if kind=='snake':
        a=np.arange(49).reshape(7,7)
        a[1::2]=a[1::2,::-1]
        return a.ravel()
    if kind=='permuted' and seed>=0:
        return np.random.default_rng(seed).permutation(49)
    raise ValueError('unsupported mapping')


def mapped_data(data:TemporalDataset,kind:str='sequential', *, seed:int=649)->TemporalDataset:
    remap=layout_indices(kind,seed=seed)
    x=data.images.reshape(-1,49)[:,remap].reshape(-1,7,7).copy()
    return TemporalDataset(data.ids,data.numbers,x,data.dates,{**data.sources,'layout':kind,'layout_seed':seed})


def spatial_energy(images:np.ndarray)->dict:
    x=np.asarray(images,dtype=float)
    if x.ndim!=3 or x.shape[1:]!=(7,7) or np.any((x!=0)&(x!=1)) or np.any(x.sum(axis=(1,2))!=6):
        raise ValueError('expected six-on 7x7 binary images')
    dct=dctn(x,axes=(1,2),norm='ortho')
    fft=np.fft.fft2(x,axes=(1,2),norm='ortho')
    direct_energy=(x*x).sum(axis=(1,2))
    dct_energy=(dct*dct).sum(axis=(1,2))
    fft_energy=(abs(fft)**2).sum(axis=(1,2))
    if not np.allclose(dct_energy,direct_energy,atol=1e-10) or not np.allclose(fft_energy,direct_energy,atol=1e-10):
        raise AssertionError('Parseval equality failed')
    if not np.allclose(dct[:,0,0],6/7,atol=1e-11) or not np.allclose(fft[:,0,0],6/7,atol=1e-11):
        raise AssertionError('DC not constant for six-on maps')
    return {'dc_power':36/49,'mean_total_energy':float(direct_energy.mean()),
            'mean_dct_power':(dct*dct).mean(axis=0).tolist(),
            'mean_fft_power':(abs(fft)**2).mean(axis=0).tolist(),
            'mean_non_dc_energy':float((dct_energy-dct[:,0,0]**2).mean()),
            'note':'All six-on images have exactly six units of total energy and DC power 36/49; the DC peak is not a discovery.'}


def periodogram_peak(x:np.ndarray)->dict:
    values=np.asarray(x,dtype=float)
    finite=values[np.isfinite(values)]
    if len(finite)<64:raise ValueError('too few data to estimate temporal spectrum')
    centered=finite-finite.mean()
    power=np.abs(rfft(centered))**2
    power=power[1:]
    if power.size==0:raise ValueError('empty spectrum')
    freq=rfftfreq(len(finite),d=1)[1:]
    if power.sum()<1e-18:
        return {'peak_frequency_cycles_per_draw':None,'peak_period_draws':None,
                'peak_power_share':0.0,'constant':True}
    k=int(np.argmax(power))
    return {'peak_frequency_cycles_per_draw':float(freq[k]),
            'peak_period_draws':float(1/freq[k]),
            'peak_power_share':float(power[k]/power.sum()),'constant':False}


def dsp_iid_report(data:TemporalDataset, *, replicates:int=80,
                   seed:int=649,perm_seed:int=2026)->dict:
    if replicates<2:raise ValueError('replicates must be >=2')
    rng=np.random.default_rng(seed)
    series=('centroid_row','spread','overlap_lag_1','dct_low_energy')
    from .temporal_features_v024 import FEATURE_NAMES
    cols=[FEATURE_NAMES.index(s) for s in series]
    mappings={}
    max_actual=0.0
    for kind in KINDS:
        converted=mapped_data(data,kind,seed=perm_seed)
        matrix,_=temporal_features(converted)
        obs={name:periodogram_peak(matrix[:,col]) for name,col in zip(series,cols)}
        max_actual=max(max_actual,max(m['peak_power_share'] for m in obs.values()))
        mappings[kind]={'layout_seed':perm_seed if kind=='permuted' else None,
                        'spectrum':spatial_energy(converted.images),
                        'temporal_peak_by_feature':obs}
    # Independent full draw-series controls for all mappings and features.
    maximum=np.empty(replicates)
    for rep in range(replicates):
        values=generate_iid_numbers(rng,len(data.ids))
        independent=TemporalDataset(data.ids,values,rasterize(values),None,{})
        local_max=0.0
        for kind in KINDS:
            converted=mapped_data(independent,kind,seed=perm_seed)
            mat,_=temporal_features(converted)
            local_max=max(local_max,max(periodogram_peak(mat[:,col])['peak_power_share'] for col in cols))
        maximum[rep]=local_max
    return {'schema':SCHEMA,'seed':seed,'permutation_seed':perm_seed,
            'replicates':replicates,'sample_draws':len(data.ids),
            'peak_max_12_searches':max_actual,
            'familywise_iid_pvalue':float((1+np.count_nonzero(maximum>=max_actual))/(replicates+1)),
            'mapping_diagnostics':mappings,
            'cautions':['The 7x7 coordinate assignment is a researcher-defined bijection, not physical draw geometry.',
                        'All tested mappings and selected temporal features enter the max-peak IID calibration.',
                        'Potential periodicity is exploratory and must pass out-of-sample forecast scores.',
                        'Red-line interpolation is not used: no artificial analog waveform is fitted.']}
