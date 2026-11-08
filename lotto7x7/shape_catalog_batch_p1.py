"""P1 Batch 2 v0.15+v0.16: fixed shape catalog and complete fair universe.

Two intentionally separate lenses:
  - empirical multiscale KDE neighborhood maxima on observed distances,
    exploratory and parameter-sensitive (NOT a complete lottery density);
  - fixed 8-neighbor graph codes / exact D4+translation shape orbits,
    with exact C(49,6) theoretical mass and exploratory IID tests.

No future-draw inference is supported by these descriptive shape operations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

from .gaussian_shape_v014 import kde_density
from .shape_taxonomy_v015 import (
    BANDWIDTHS,RADII,UNIVERSE,classify_numbers,decode_partition,partition_name,
    edge_name,exact_shape_families,mode_stability_multiscale,pairwise_diameter_metrics
)
from .shape_theory_v016 import (LEVELS,classification_keys,category_name,
    exhaustive_class_counts,historical_comparison,shape_family_baseline)

SCHEMA='lotto7x7.shape_catalog_exhaustive.v0.15+v0.16'


def _filesha(file:Path)->str:
    h=sha256()
    with Path(file).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def _representative(distance:np.ndarray, members:np.ndarray)->tuple[int,float,float,float]:
    """Medoid, mean-to-medoid, within-pairwise maximum and mean (original shape D)."""
    member=np.asarray(members,dtype=np.int64)
    sub=distance[np.ix_(member,member)]
    center=int(member[np.argmin(sub.sum(axis=1))])
    dist=distance[member,center]
    tri=sub[np.triu_indices(len(member),k=1)]
    return center,float(dist.mean()),float(tri.max()) if len(tri) else 0.,float(tri.mean()) if len(tri) else 0.


def frozen_shape_catalog(distance:np.ndarray,ids:np.ndarray,numbers:np.ndarray,
                         baseline:dict, comparisons:dict)->dict:
    """Catalog levels 1-3 exactly match the *same fixed* universe classifier."""
    tags=classification_keys(classify_numbers(numbers))
    levels={}
    for level in LEVELS:
        candidates=[]
        for row in comparisons['levels'][level]['rows']:
            code=int(row['code'])
            members=np.flatnonzero(tags[level]==code)
            if len(members):
                medoid,mean_center,diameter,mean_pair=_representative(distance,members)
                repre={'medoid_index':medoid,'medoid_draw_id':int(ids[medoid]),
                       'medoid_numbers':list(map(int,numbers[medoid])),
                       'mean_distance_to_medoid':mean_center,'class_diameter':diameter,
                       'mean_pairwise_distance':mean_pair}
            else:
                repre={'medoid_index':None,'medoid_draw_id':None,'medoid_numbers':[],
                       'mean_distance_to_medoid':None,'class_diameter':None,'mean_pairwise_distance':None}
            if len(members):
                others=np.flatnonzero(tags[level]!=code)
                cross=distance[np.ix_(members,others)]
                repre.update({'min_other_class_distance':float(cross.min()) if cross.size else None,
                              'mean_other_class_distance':float(cross.mean()) if cross.size else None})
            else:
                repre.update({'min_other_class_distance':None,'mean_other_class_distance':None})
            candidates.append({**row,**repre})
        levels[level]=candidates
    return levels


def analyze(distance:np.ndarray,ids:np.ndarray,numbers:np.ndarray,*,
            baseline:dict,device='auto',bandwidths=BANDWIDTHS,radii=RADII,
            assignment_radius=0.85,mc_replicates=5000,seed=649,on_progress=None)->tuple[dict,dict]:
    n=len(distance)
    arr=np.asarray(numbers,dtype=np.int16)
    draw_ids=np.asarray(ids,dtype=np.int64)
    d=np.asarray(distance,dtype=np.float32)
    if (arr.shape!=(n,6) or draw_ids.shape!=(n,) or n<8 or d.shape!=(n,n)
            or not np.isfinite(d).all() or abs(d-d.T).max()>3e-4
            or np.max(np.abs(np.diag(d)))>1e-4):raise ValueError('cached distance / metadata mismatch')
    if not baseline.get('complete') or baseline['total']!=UNIVERSE:raise ValueError('requires exact full universe')
    bandwidths=tuple(float(v) for v in bandwidths)
    radii=tuple(float(v) for v in radii)
    if any(v<=0 for v in bandwidths+radii) or len(set(bandwidths))!=len(bandwidths):
        raise ValueError('positive unique bandwidths and radii required')
    if on_progress:on_progress(f'[CATALOG] historical draws={n}; frozen family hierarchy')
    tags=classification_keys(classify_numbers(arr))
    exact_families,exact_labels=exact_shape_families(arr,draw_ids)
    fam_baseline=shape_family_baseline(exact_families,n)
    if on_progress:on_progress(f'[CATALOG] observed exact shape families={len(exact_families)}')
    comp=historical_comparison(arr,baseline,replicates=mc_replicates,seed=seed)
    frozen=frozen_shape_catalog(d,draw_ids,arr,baseline,comp)
    separation={}
    for level in ('component','partition','edge'):
        k=tags[level]
        unique=np.unique(k)
        separation[level]={'observed_classes':int(len(unique)),
             'silhouette_original_shape_distance':float(silhouette_score(d,k,metric='precomputed'))
                    if 1<len(unique)<n else None,
             'note':'Descriptive separation in v0.13 original shape distance, not a predictive accuracy'}
    if on_progress:on_progress('[THEORY] historical versus complete fixed shape categories')
    density=kde_density(d,bandwidths,device=device)
    ms,mode_arr=mode_stability_multiscale(d,density,bandwidths,radii,assignment_radius)
    main_peaks=mode_arr['peaks'];mode_labels=mode_arr['labels']
    peak_info=[]
    for k,p in enumerate(main_peaks):
        members=np.flatnonzero(mode_labels==k)
        if not len(members):raise AssertionError('peak without members')
        medoid,mc,diam,meanpair=_representative(d,members)
        peak_info.append({'mode_index':int(k),'peak_draw_id':int(draw_ids[p]),
            'peak_numbers':list(map(int,arr[p])),'peak_index':int(p),
            'count':int(len(members)),'mean_distance_to_medoid':mc,
            'class_diameter':diam,'mean_pairwise_distance':meanpair,
            'peak_density_score':float(density[p,bandwidths.index(ms['main_bandwidth'])]),
            'peak_persistence':next(x['presence_rate'] for x in mode_arr['persistence'] if x['index']==p)})
    peak_info.sort(key=lambda x:(-x['count'],x['mode_index']))
    assigned=np.flatnonzero(mode_labels>=0)
    num_assigned_labels=len(np.unique(mode_labels[assigned]))
    separation['empirical_modes']={'observed_classes':num_assigned_labels,
        'assigned':len(assigned),
        'silhouette_original_shape_distance':float(silhouette_score(
             d[np.ix_(assigned,assigned)],mode_labels[assigned],metric='precomputed'))
             if 1<num_assigned_labels<len(assigned) else None,
        'note':'Silhouette excludes unassigned; cannot be compared directly to full-coverage graph scores'}
    if on_progress:on_progress(f'[MODES] empirical main modes={len(main_peaks)} unassigned={ms["main_unassigned"]}')
    report={'schema':SCHEMA,'research_only':True,
          'data':{'draw_count':int(n),'unordered_pair_count':int(n*(n-1)//2),
                  'exact_universe_count':UNIVERSE,
                  'distance_source':'v0.13 cached D4+centroid-translation shape distance',
                  'fixed_graph_rule':baseline['schema'],
                  'fixed_graph_rule_hash':baseline['rule_hash_sha256']},
          'settings':{'bandwidths':list(bandwidths),'radii':list(radii),
                 'mode_assignment_radius':float(assignment_radius),
                 'mc_replicates':int(mc_replicates),'seed':int(seed),'device':device},
          'classification_levels':{'L1':'8-neighbor component count',
                 'L2':'descending component size multiset (integer partition of 6)',
                 'L3':'L2 plus cardinal and diagonal 8-neighbor edges',
                 'L4':'exact D4+translation canonical six-point shape',
                 'density_modes':'separate exploratory multiscale empirical KDE local-maxima overlay'},
          'theory':{'enumeration_complete':True,'processed':baseline['total'],
                    'class_counts':baseline['counts'],
                    'num_classes':{k:len(v) for k,v in baseline['counts'].items()}},
          'comparison':comp,
          'frozen_catalog':frozen,
          'separation_diagnostics':separation,
          'exact_shape_family_theory':fam_baseline,
          'exact_shape_families':exact_families,
          'kde_multiscale':{**ms,'peak_classes':peak_info,
                 'warning':'Empirical sample maxima, not continuous KDE modes or stable intrinsic natural class count'},
          'cautions':['The 7x7 number-to-cell mapping is arbitrary and does not alter fair draw odds',
              'Graph categories are deterministic descriptive codes; density modes are exploratory',
              'The theoretical class counts are exhaustive; the multinomial p-value is Monte Carlo calibrated',
              'Significance testing after choosing a representation is exploratory, not prospective confirmation',
              'Observed exact shape family list was selected after inspecting history; no individual family p-value',
              'Nearest-mode assigned members need not have bounded pairwise diameter',
              'Density and class frequency are not next-draw probability improvements']}
    arrays={'draw_ids':draw_ids.astype(np.int64),'numbers':arr.astype(np.int16),
            'component':tags['component'],'partition':tags['partition'],'edge':tags['edge'],
            'family_id':exact_labels,
            'bandwidths':np.asarray(bandwidths,dtype=np.float32),
            'density':density.astype(np.float32),'mode_peaks':main_peaks,
            'mode_labels':mode_labels,'mode_distance_to_peak':mode_arr['nearest_distance']}
    return report,arrays


def rows_draws(arrays:dict):
    a=arrays;d=a['draw_ids'];n=a['numbers']
    for i in range(len(d)):
        row={'draw_id':int(d[i]),**{f'n{j+1}':int(n[i,j]) for j in range(6)},
             'component_class':f'C{int(a["component"][i])}',
             'partition_class':partition_name(int(a['partition'][i])),
             'edge_class':edge_name(int(a['edge'][i])),
             'exact_shape_id':str(a['family_id'][i]),
             'mode_class':f'M{int(a["mode_labels"][i])+1:03d}' if a['mode_labels'][i]>=0 else 'UNASSIGNED',
             'mode_distance':float(a['mode_distance_to_peak'][i])}
        row.update({f'kde_sigma_{float(b):g}':float(v) for b,v in zip(a['bandwidths'],a['density'][i])})
        yield row


def rows_theory(report:dict):
    for level in LEVELS:
        for row in report['frozen_catalog'][level]:
            yield dict(row)


def rows_exact_families(report:dict):
    for f in report['exact_shape_families']:
        yield {**f,'representative_numbers':';'.join(map(str,f['representative_numbers'])),
               'member_draw_ids':';'.join(map(str,f['member_draw_ids']))}


def _bitmap(numbers,width:int=95,fill='#187d74')->str:
    nums=set(map(int,numbers));cells=[]
    for i in range(49):
        c,r=i%7,i//7
        color=fill if i+1 in nums else '#e8edf0'
        cells.append(f'<rect x="{1+c*13}" y="{1+r*13}" width="11" height="11" rx="1.5" fill="{color}"/>')
    return f'<svg viewBox="0 0 91 91" width="{int(width)}" role="img" aria-label="six selected cells on 7 by 7">'+''.join(cells)+'</svg>'


def html_report(report:dict, *,shown_modes:int=24,shown_classes:int=20,shown_shapes:int=16)->str:
    if min(shown_modes,shown_classes,shown_shapes)<1:raise ValueError('positive display limits')
    p=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>Lotto649 P1 Shape Catalog & Exact Fair Universe</title>',
       '<style>body{background:#f6f9fa;color:#243543;font:15px system-ui;max-width:1350px;margin:auto;padding:1.4rem}section{margin:1rem 0;background:white;border:1px solid #e1e6ea;padding:1rem;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:.4rem;border-bottom:1px solid #e4e7eb;text-align:right}th:first-child,td:first-child{text-align:left}.cards{display:flex;flex-wrap:wrap;gap:1rem}.card{border:1px solid #d9e3e7;padding:.7rem;border-radius:9px;min-width:115px;max-width:230px}.small{font-size:.76rem;color:#60717f}h1{color:#187d74}</style>',
       '<h1>Batch 2 / P1: Natural Shape Taxonomy + Exact 6/49 Baseline</h1>',
       '<p>Every legal 6/49 combination counted exactly once under frozen graph rules. KDE sample modes are a separate exploratory overlay. No forecast or wagering advantage.</p>']
    tc=report['theory'];exact=report['exact_shape_family_theory']
    p.append('<section><h2>Full Universe</h2>'+
          f'<p>Exhaustive: <b>{tc["processed"]:,}</b> combinations; '
          f'L1/L2/L3 classes: {tc["num_classes"]["component"]} / '
          f'{tc["num_classes"]["partition"]} / {tc["num_classes"]["edge"]}.</p>'+
          f'<p>Observed exact families: {exact["observed_shape_families"]:,}; '
          f'combinations in observed families: {exact["theoretical_total_in_seen_exact_shapes"]:,}; '
          f'other legal combinations: {exact["theoretical_total_in_unseen_exact_shapes"]:,}.</p></section>')
    p.append('<section><h2>Original-shape-distance separation</h2><table><tr><th>Partition</th><th>Observed classes</th><th>Silhouette</th></tr>')
    for name,z in report['separation_diagnostics'].items():
        silhouette=z['silhouette_original_shape_distance']
        p.append(f'<tr><td>{escape(name)}</td><td>{z["observed_classes"]}</td><td>{silhouette:.4f}</td></tr>' if silhouette is not None
            else f'<tr><td>{escape(name)}</td><td>{z["observed_classes"]}</td><td>N/A</td></tr>')
    p.append('</table><p class="small">Empirical mode silhouette uses assigned images only; graph taxonomy uses full history. They are not directly comparable.</p></section>')
    for level,label in [('component','L1 Connected components'),('partition','L2 Component partitions'),('edge','L3 Edge profiles')]:
        q=report['comparison']['levels'][level]
        p.append(f'<section><h2>{label}</h2><p>Global Pearson chi-square={q["chi_square"]:.3f}; '
                 f'multinomial MC p={q["empirical_multinomial_p"]:.4f}; '
                 f'Bonferroni-across-3-levels p={q["bonferroni_global_p"]:.4f}. '
                 f'Categories: {q["classes"]}.</p>')
        p.append('<table><tr><th>Class</th><th>Observed</th><th>Exact count</th><th>Expected</th><th>Ratio</th><th>BH q</th></tr>')
        rows=sorted(report['frozen_catalog'][level],key=lambda x:(-x['observed'],x['class']))
        if level=='edge':rows=rows[:shown_classes]
        for r in rows:
            ratio=r['observed']/r['expected'] if r['expected']>0 else 0
            p.append(f'<tr><td>{escape(r["class"])}</td><td>{r["observed"]}</td><td>{r["theoretical_combinations"]:,}</td>'
                     f'<td>{r["expected"]:.2f}</td><td>{ratio:.2f}</td>'
                     f'<td>{r["fdr_bh_q_across_all_levels"]:.3f}</td></tr>')
        p.append('</table>')
        if level=='edge':p.append(f'<p class="small">Showing top {min(shown_classes,len(q["rows"]))} historical classes; full 94-class table in theory CSV.</p>')
        p.append('</section>')
    p.append('<section><h2>Multiscale empirical KDE local maxima</h2>'+
          '<p>Each peak is a maximal density point among observed samples within its neighborhood radius. '
          'Not an analytic continuous KDE mode or a fixed graph family. Peak counts change with bandwidth and radius.</p>')
    p.append('<table><tr><th>σ</th><th>radius</th><th>sample modes</th><th>assigned</th><th>unassigned</th><th>largest basin</th></tr>')
    for r in report['kde_multiscale']['records']:
        p.append(f'<tr><td>{r["bandwidth"]:.2f}</td><td>{r["radius"]:.2f}</td>'
                 f'<td>{r["peak_count"]}</td><td>{r["assigned"]}</td><td>{r["unassigned"]}</td><td>{r["max_class_size"]}</td></tr>')
    p.append('</table></section>')
    p.append('<section><h2>Largest empirical shape-mode basins</h2><div class="cards">')
    for r in report['kde_multiscale']['peak_classes'][:shown_modes]:
        p.append('<article class="card">'+_bitmap(r['peak_numbers'])+
          f'<div><b>Mode {r["mode_index"]+1:03d}</b> · {r["count"]} draws</div>'+
          f'<div class="small">draw {r["peak_draw_id"]}</div>'+
          f'<div class="small">diameter {r["class_diameter"]:.3f}; persistence {r["peak_persistence"]:.2f}</div></article>')
    p.append('</div></section>')
    p.append('<section><h2>Frequently observed exact shape families</h2><div class="cards">')
    for r in report['exact_shape_families'][:shown_shapes]:
        p.append('<article class="card">'+_bitmap(r['representative_numbers'],fill='#5778a9')+
            f'<div><b>{escape(r["shape_id"])}</b></div>'+
            f'<div>{r["count"]} historical · {r["theory_combinations"]} exact legal placements</div>'+
            f'<div class="small">{escape(r["edge_class"])}</div></article>')
    p.append('</div></section>')
    p.append('<section><h2>Scientific limitations</h2><p>Graph-theory distributions are exact; empirical significance uses Monte Carlo multinomial calibration for the fixed classes. '
             'Multiple methods were explored on this same history, so all statistical p-values remain exploratory. '
             'Observed exact-shape families are selected from the sample and must not be individually treated as preregistered hypotheses. '
             'A 7x7 image of 6/49 draw numbers is researcher-chosen and does not imply greater next-draw winning odds.</p></section></html>')
    return '\n'.join(p)
