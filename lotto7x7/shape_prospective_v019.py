"""v0.19 future-only, preregistered 6/49 shape monitoring.

A fixed historical snapshot and P1 exact fair masses seed three adaptive,
properly normalized categorical likelihood-ratio e-processes. An independent
predeclared one-sided Bernoulli alternative (10x rare-class fair rate) adds a
fourth. Their equally weighted mixture is an anytime-valid e-process under
fair IID draws. No retrospective data enter the prospective evidence counter.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from glob import glob
from hashlib import sha256
import json
from math import exp, log, log1p
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import logsumexp

from .shape_predictive_v017 import _baseline_tables, _index_tags, validate_history, RARE_CODE_NAME
from .shape_taxonomy_v015 import classify_numbers, edge_name
from .shape_theory_v016 import RULE_VERSION, LEVELS, UNIVERSE, classification_keys, category_name

FREEZE_SCHEMA='lotto7x7.prospective_freeze.v0.19'
REPORT_SCHEMA='lotto7x7.prospective_monitor.v0.19'
RARE_MULTIPLIER=10.0
TAU=384.0
ALPHA=0.05


def file_sha256(path:Path)->str:
    h=sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()


def content_hash(obj:dict)->str:
    return sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,
                             separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal_object(obj:dict)->dict:
    result=dict(obj)
    result['self_checksum_sha256']=content_hash(obj)
    return result


def verify_seal(obj:dict)->None:
    checksum=obj.get('self_checksum_sha256')
    raw={k:v for k,v in obj.items() if k!='self_checksum_sha256'}
    if not isinstance(checksum,str) or content_hash(raw)!=checksum:
        raise ValueError('snapshot integrity checksum mismatch')


def read_sources(catalog:Path,universe:Path)->tuple[np.ndarray,np.ndarray,dict,dict]:
    with Path(universe).open(encoding='utf-8') as f:baseline=json.load(f)
    if baseline.get('rule_hash_sha256')!=sha256(RULE_VERSION.encode()).hexdigest():
        raise ValueError('frozen taxonomy rule hash differs from P1')
    tables=_baseline_tables(baseline)
    with np.load(catalog,allow_pickle=False) as z:
        if not {'draw_ids','numbers','component','partition','edge'}<=set(z.files):
            raise ValueError('P1 catalog missing required arrays')
        ids,nums=validate_history(z['draw_ids'],z['numbers'])
        actual=classification_keys(classify_numbers(nums))
        for level in LEVELS:
            if not np.array_equal(actual[level],z[level]):
                raise ValueError(f'P1 cached {level} class does not match fixed rule')
    tags=_index_tags(nums,tables)
    return ids,nums,tables,tags


def make_freeze(catalog:Path,universe:Path,watchlist:Path,*,
                frozen_at:datetime|None=None,tau:float=TAU)->dict:
    if tau<=0 or not np.isfinite(tau):raise ValueError('tau must be positive and finite')
    ids,nums,tables,tags=read_sources(catalog,universe)
    with Path(watchlist).open(encoding='utf-8') as f:w=json.load(f)
    if (w.get('schema')!='lotto7x7.p2_future_only_watchlist.v0.17' or
            w.get('source_draw_id_last')!=int(ids[-1]) or
            w.get('frozen_rule_hash_sha256')!=sha256(RULE_VERSION.encode()).hexdigest() or
            w.get('fixed_category',{}).get('class')!=RARE_CODE_NAME or
            w.get('fixed_category',{}).get('theoretical_combinations')!=744):
        raise ValueError('P2 preregistration does not match historical freeze boundary')
    now=frozen_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:raise ValueError('freeze time must have timezone')
    utc=now.astimezone(timezone.utc)
    local_date=now.astimezone(ZoneInfo('Asia/Taipei')).date().isoformat()
    exact_count={lev:{str(int(k)):int(v) for k,v in zip(tables[lev]['codes'],
              np.bincount(tags[lev],minlength=len(tables[lev]['codes'])))} for lev in LEVELS}
    rare=np.flatnonzero([edge_name(int(c))==RARE_CODE_NAME for c in tables['edge']['codes']])
    if len(rare)!=1:raise AssertionError('rare class missing or repeated')
    rare_probability=float(tables['edge']['p'][rare[0]])
    if abs(rare_probability-744/UNIVERSE)>1e-16:raise AssertionError('rare fair mass mismatch')
    result={
      'schema':FREEZE_SCHEMA,'frozen_at_utc':utc.isoformat(timespec='seconds'),
      'freeze_local_date_taipei':local_date,
      'source':{'catalog_sha256':file_sha256(catalog),
                'universe_sha256':file_sha256(universe),
                'p2_watchlist_sha256':file_sha256(watchlist),
                'last_draw_id':int(ids[-1]),'first_draw_id':int(ids[0]),
                'historical_draw_count':int(len(ids)),
                'historical_record_sha256':sha256(ids.astype('<i8').tobytes()+nums.astype('u1').tobytes()).hexdigest()},
      'frozen_rule':{'schema':RULE_VERSION,'sha256':sha256(RULE_VERSION.encode()).hexdigest(),
                     'fair_universe':UNIVERSE,'tau':float(tau),
                     'category_names':['component','partition','edge'],
                     'historical_counts':exact_count},
      'tests':{'e_processes':['component','partition','edge','rare_class'],
               'rare_class':RARE_CODE_NAME,'rare_fair_probability':rare_probability,
               'rare_alternative_multiplier':RARE_MULTIPLIER,
               'rare_alternative_probability':RARE_MULTIPLIER*rare_probability,
               'mixture_weights':[0.25]*4,
               'anytime_familywise_alpha':ALPHA,'mixture_e_threshold':1/ALPHA},
      'eligibility':{'minimum_draw_id_exclusive':int(ids[-1]),
                     'minimum_taipei_draw_date_exclusive':local_date,
                     'requires_gregorian_draw_date':True,
                     'note':'A later draw ID alone is insufficient: actual draw date must be after freeze date; verify records independently from their source.'},
      'interpretation':'Preregistered research monitoring only; no evidence of greater probability for any individual future fair ticket.'}
    return seal_object(result)


def check_freeze(freeze:dict,catalog:Path,universe:Path):
    verify_seal(freeze)
    if freeze.get('schema')!=FREEZE_SCHEMA:raise ValueError('wrong freeze schema')
    if freeze.get('source',{}).get('catalog_sha256')!=file_sha256(catalog):
        raise ValueError('P1 catalog bytes changed after freeze')
    if freeze.get('source',{}).get('universe_sha256')!=file_sha256(universe):
        raise ValueError('exact P1 fair universe bytes changed after freeze')
    ids,nums,tables,tags=read_sources(catalog,universe)
    source=freeze['source']
    if (source['historical_draw_count']!=len(ids) or source['first_draw_id']!=int(ids[0]) or
            source['last_draw_id']!=int(ids[-1]) or
            source['historical_record_sha256']!=sha256(ids.astype('<i8').tobytes()+nums.astype('u1').tobytes()).hexdigest()):
        raise ValueError('frozen historical sample differs')
    expected=freeze['frozen_rule']['historical_counts']
    for lev in LEVELS:
        cs=np.bincount(tags[lev],minlength=len(tables[lev]['codes']))
        if expected[lev]!={str(int(k)):int(v) for k,v in zip(tables[lev]['codes'],cs)}:
            raise ValueError(f'frozen {lev} counts differ')
    tests=freeze['tests']
    if (tests['e_processes']!=['component','partition','edge','rare_class'] or
        tests['mixture_weights']!=[0.25]*4 or tests['rare_alternative_multiplier']!=RARE_MULTIPLIER or
        tests['rare_class']!=RARE_CODE_NAME or freeze['frozen_rule']['schema']!=RULE_VERSION or
        freeze['frozen_rule']['sha256']!=sha256(RULE_VERSION.encode()).hexdigest()):
        raise ValueError('unsupported or altered prospective hypothesis family')
    return ids,nums,tables,tags


def _parse_date(value:str)->date:
    raw=value.strip().replace('/','-')
    try:result=date.fromisoformat(raw)
    except ValueError as e:raise ValueError(f'invalid Gregorian draw date {value!r}; use YYYY-MM-DD') from e
    if not 2004<=result.year<=2100:raise ValueError('draw date outside Gregorian 2004–2100')
    return result


def _expand_csv(paths:list[str])->list[Path]:
    items=[]
    for arg in paths:
        p=Path(arg)
        if p.is_dir():items.extend(sorted(p.rglob('*.csv')))
        elif any(c in arg for c in '*?['):items.extend(Path(s) for s in sorted(glob(arg,recursive=True)))
        else:items.append(p)
    result=[];seen=set()
    for p in items:
        key=str(p.resolve())
        if key not in seen:result.append(p);seen.add(key)
    return result


def read_updates(files:list[str],*,freeze:dict,reference_ids:np.ndarray,
                 reference_numbers:np.ndarray)->tuple[list[dict],list[dict]]:
    historical={int(i):tuple(map(int,x)) for i,x in zip(reference_ids,reference_numbers)}
    sources=[];future=[];seen=set()
    cutoff=freeze['source']['last_draw_id']
    min_date=date.fromisoformat(freeze['freeze_local_date_taipei'])
    allfiles=_expand_csv(files)
    for path in allfiles:
        if not path.is_file():raise FileNotFoundError(f'CSV file not found: {path}')
        sources.append({'file':str(path),'sha256':file_sha256(path)})
        with path.open(newline='',encoding='utf-8-sig') as fh:
            reader=csv.DictReader(fh)
            if not reader.fieldnames or not {'draw_id','n1','n2','n3','n4','n5','n6'}<=set(reader.fieldnames):
                raise ValueError(f'{path}: required English headers draw_id,n1..n6; date mandatory for new draws')
            for lineno,row in enumerate(reader,2):
                try:
                    draw_id=int(row['draw_id'].strip())
                    nums=tuple(sorted(int(row[f'n{i}'].strip()) for i in range(1,7)))
                except (ValueError,AttributeError,TypeError) as e:
                    raise ValueError(f'{path}:{lineno} invalid draw ID or six numbers') from e
                if draw_id in seen:raise ValueError(f'duplicate draw ID in supplied CSV sources: {draw_id}')
                seen.add(draw_id)
                # Validate ALL rows, including old historical records, not only future ones.
                classify_numbers(np.array([nums],dtype=np.int64))
                if draw_id<=cutoff:
                    if draw_id in historical and nums!=historical[draw_id]:
                        raise ValueError(f'{path}:{lineno}: existing historical draw {draw_id} differs from frozen data')
                    continue
                date_text=(row.get('date') or row.get('draw_date') or '').strip()
                if not date_text:raise ValueError(f'{path}:{lineno}: future draw requires date or draw_date (YYYY-MM-DD)')
                day=_parse_date(date_text)
                if day<=min_date:raise ValueError(f'{path}:{lineno}: draw date {day} predates or equals freeze date {min_date}')
                if day.year-1911!=draw_id//1_000_000:
                    raise ValueError(f'{path}:{lineno}: draw ID ROC year mismatches draw date')
                future.append({'draw_id':draw_id,'date':day.isoformat(),'numbers':list(nums)})
    future.sort(key=lambda x:x['draw_id'])
    previous=cutoff; previous_date=min_date
    for row in future:
        current=row['draw_id']; year,issue=divmod(current,1_000_000)
        prev_year,prev_issue=divmod(previous,1_000_000)
        if year==prev_year and issue!=prev_issue+1:
            raise ValueError(f'non-contiguous prospective issue IDs {previous} -> {current}; cannot cherry-pick draws')
        if year!=prev_year and (year!=prev_year+1 or issue!=1):
            raise ValueError(f'year-transition draw ID discontinuity {previous} -> {current}')
        this_date=date.fromisoformat(row['date'])
        if this_date<previous_date:raise ValueError('draw dates must be nondecreasing by issue')
        previous=current;previous_date=this_date
    return future,sources


def prospective_evaluate(freeze:dict,tables:dict,forward:list[dict],sources:list[dict],
                         *,previous:dict|None=None)->tuple[dict,list[dict]]:
    frozen=freeze['frozen_rule'];tau=float(frozen['tau']); n0=freeze['source']['historical_draw_count']
    counts={}
    for lev in LEVELS:
        codes=tables[lev]['codes'];c=frozen['historical_counts'][lev]
        counts[lev]=np.array([int(c[str(int(x))]) for x in codes],dtype=np.int64)
    log_e={lev:0. for lev in LEVELS};log_e['rare_class']=0.
    p_rare=float(freeze['tests']['rare_fair_probability'])
    q_rare=float(freeze['tests']['rare_alternative_probability'])
    alpha=float(freeze['tests']['anytime_familywise_alpha'])
    rare_code=[int(x) for x in tables['edge']['codes'] if edge_name(int(x))==RARE_CODE_NAME]
    if len(rare_code)!=1:raise AssertionError('rare lookup ambiguous')
    rows=[]; rare_hits=0
    for j,evt in enumerate(forward):
        nums=np.array([evt['numbers']],dtype=np.uint8)
        enc=classification_keys(classify_numbers(nums))
        out={'draw_id':int(evt['draw_id']),'date':evt['date'],
             **{f'n{k+1}':int(evt['numbers'][k]) for k in range(6)}}
        for lev in LEVELS:
            code=int(enc[lev][0]); codes=tables[lev]['codes']
            ix=int(np.searchsorted(codes,code))
            if ix>=len(codes) or codes[ix]!=code:raise AssertionError('code not in exact fair baseline')
            p=float(tables[lev]['p'][ix]);q=float((counts[lev][ix]+tau*p)/(n0+j+tau))
            assert 0<q<=1 and 0<p<=1
            gain=log(q/p)
            log_e[lev]+=gain
            out.update({f'{lev}_class':category_name(lev,code),
                       f'{lev}_q_before_draw':q,f'{lev}_fair_p':p,
                       f'{lev}_log_gain':gain,f'{lev}_cumulative_log_e':log_e[lev]})
        hit=int(int(enc['edge'][0])==rare_code[0]);rare_hits+=hit
        rare_gain=log(q_rare/p_rare) if hit else log1p(-q_rare)-log1p(-p_rare)
        log_e['rare_class']+=rare_gain
        out['rare_class_hit']=hit
        out['rare_cumulative_hits']=rare_hits
        out['rare_log_gain']=rare_gain
        out['rare_cumulative_log_e']=log_e['rare_class']
        log_mixture=float(logsumexp(list(log_e.values()))-log(4))
        out['mixture_log_e']=log_mixture
        out['mixture_e']=float(exp(min(log_mixture,700)))
        out['anytime_p_upper_bound']=float(1. if log_mixture<=0 else exp(-min(log_mixture,700)))
        rows.append(out)
        # The observed draw modifies the NEXT prediction, never its own.
        for lev in LEVELS:
            code=int(enc[lev][0]);idx=int(np.searchsorted(tables[lev]['codes'],code))
            counts[lev][idx]+=1
    if previous is not None:
        if previous.get('schema')!=REPORT_SCHEMA or previous.get('freeze_sha256')!=freeze['self_checksum_sha256']:
            raise ValueError('previous report from different preregistered experiment')
        prior=previous.get('events',[])
        if len(prior)>len(rows):raise ValueError('new input shorter than previous observation history')
        for idx,row in enumerate(prior):
            updated=rows[idx]
            if row.keys()!=updated.keys():
                raise ValueError('previously scored event fields changed')
            for key,value in row.items():
                actual=updated[key]
                if isinstance(value,float) and isinstance(actual,(float,int)):
                    match=bool(np.isclose(value,actual,rtol=1e-12,atol=1e-12))
                else:
                    match=(value==actual)
                if not match:
                    raise ValueError(f'previously scored prospective event {idx} field {key} was removed or changed')
        if previous.get('eligible_new_draws')!=len(prior):
            raise ValueError('previous report count differs from its event log')
        baseline_mixture=1. if not prior else prior[-1]['mixture_e']
        if not np.isclose(previous.get('mixture_e',-1),baseline_mixture,rtol=1e-12,atol=1e-12):
            raise ValueError('previous report aggregate differs from event log')
    total=len(rows)
    logmix=float(logsumexp(list(log_e.values()))-log(4))
    mix_e=float(exp(min(logmix,700)))
    rare_mass=total*p_rare
    report={
      'schema':REPORT_SCHEMA,'freeze_sha256':freeze['self_checksum_sha256'],
      'status':'NO_PROSPECTIVE_DRAWS' if total==0 else 'MONITORING',
      'eligible_new_draws':total,
      'first_new_draw_id':rows[0]['draw_id'] if rows else None,
      'last_new_draw_id':rows[-1]['draw_id'] if rows else None,
      'source_csv_files':sources,
      'frozen_at_utc':freeze['frozen_at_utc'],
      'last_frozen_historical_draw_id':freeze['source']['last_draw_id'],
      'model':{'tau':tau,'fair_universe':UNIVERSE,'rare_alternative_probability':q_rare,
               'rare_fair_probability':p_rare,'e_processes':list(log_e)},
      'rare_future_count':rare_hits,'rare_future_fair_expectation':rare_mass,
      'cumulative_log_e':log_e,
      'mixture_e':mix_e,'anytime_p_upper_bound':float(1. if logmix<=0 else exp(-min(logmix,700))),
      'anytime_alpha':alpha,'signal_threshold':1/alpha,
      'decision':'REVIEW_DEVIATION_CANDIDATE' if mix_e>=1/alpha else ('NO_FUTURE_DATA' if total==0 else 'NO_CONFIRMED_DEVIATION'),
      'events':rows,
      'previous_report_file_sha256':previous.get('_file_sha256') if previous is not None else None,
      'cautions':['Frozen protocol applies only to verified external lottery draw outcomes after the freeze date',
                 'User-provided CSV dates are not independently authenticated by this tool',
                 'Mixture e-value controls optional stopping for this predeclared four-process family under fair IID',
                 'Research review threshold does not prove bias, causality or prediction of particular future numbers',
                 'No alternate ticket becomes more likely to win under a fair uniform 6/49 draw']}
    return report,rows


def report_html(report:dict)->str:
    from html import escape
    seen=report['eligible_new_draws'];result=report['decision']
    cells=[]
    for row in report['events']:
        cells.append('<tr>'+''.join(f'<td>{escape(str(row[col]))}</td>' for col in
                          ('draw_id','date','component_class','partition_class','edge_class','rare_class_hit','mixture_e'))+'</tr>')
    return '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Lotto649 v0.19 Prospective Validation</title>
<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f5f8fa;color:#243943}section{background:#fff;border:1px solid #dce5e9;border-radius:12px;padding:1rem;margin:1rem 0}th,td{padding:.45rem;border-bottom:1px solid #e2e7ea;text-align:left}table{width:100%;border-collapse:collapse}h1{color:#157e74}</style>
<h1>Lotto649 — v0.19 真正未來期別驗證</h1>
<p>每一期使用前期已知結果產生當期機率，再更新模型。不能將研究期間已看過的開獎紀錄列為未來驗證樣本。</p>
<section><h2>研究狀態</h2><p>新期別：<b>'''+str(seen)+'''</b>，判定：<b>'''+escape(result)+'''</b></p>
<p>Mixture e-value：<b>'''+f'{report["mixture_e"]:.6g}'+'''</b>（預先凍結門檻：20）</p>
<p>Anytime-valid p-value upper bound：'''+f'{report["anytime_p_upper_bound"]:.6g}'+'''</p>
<p>罕見類別未來出現：'''+str(report['rare_future_count'])+'''；公平基準期望：'''+f'{report["rare_future_fair_expectation"]:.6g}'+'''</p></section>
<section><h2>逐期審計</h2><table><tr><th>期別</th><th>日期</th><th>L1</th><th>L2</th><th>L3</th><th>罕見類別</th><th>累積 e-value</th></tr>'''+''.join(cells)+'''</table></section>
<section><h2>限制</h2><p>資料日期仍須獨立核實。這是對固定四項檢定的統計監測，不是提高六碼中獎率的預測系統。若無新資料，本次報告沒有任何前瞻驗證證據。</p></section></html>'''
