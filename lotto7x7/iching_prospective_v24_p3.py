"""v2.4 P3: future-only USNO sky/parity-hexagram martingale shadow audit.

The 100-draw model and 5:4 tilt are fixed before eligible outcomes. Astronomical
variables never use winning numbers. Two archived independent source pages are
required before any real observation is admitted. No tickets or wagers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
from fractions import Fraction
from hashlib import sha256
import io
import json
from math import comb
from pathlib import Path

from . import iching_gua_qi_v24_p1 as p1
from . import iching_solar_lunar_v24_p2 as p2
from . import iching_hexagram_association_v24_p2a as p2a

SCHEMA = 'lotto649.v2.4.p3.usno_parity_eprocess.v1'
SEAL_SCHEMA = 'lotto649.v2.4.p3.future_sky_and_model_seal.v1'
SKY_RULE = p2.RULE
BASE_EVENTS_SHA256 = p1.EXPECTED_USNO_EVENTS_SHA256
BASE_P2_SHA256 = p2a.SOURCE_SHA256
FIRST_ID = 115000106
FIRST_DATE = '2026-11-17'
HORIZON = 100
SKY_START = date(2026, 11, 17)
SKY_END = date(2027, 11, 30)
REVIEW_THRESHOLD = 20
ALT_WEIGHT_NUM = 5
ALT_WEIGHT_DEN = 4
SKY_FILENAME = 'v24_p3_sky_calendar_20261114.csv'
SEAL_FILENAME = 'v24_p3_usno_hexagram_static100_20261114_seal.json'
REPORT_SCHEMA = 'lotto649.v2.4.p3.prospective_evaluation.v1'
SKY_COLS = ('date','reference_at_taipei','solar_sector_8','lunar_sector_8',
            'astronomical_hexagram_code','mapping_version')


def digest(b: bytes) -> str:
    return sha256(b).hexdigest()


def encode_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=SKY_COLS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _verified_2027(root: Path) -> tuple[list[dict], dict]:
    """2027 USNO add-on is a separate namespace, never mutates base 2004-26."""
    from scripts.fetch_usno_astronomy import validate_json

    man_path = root / 'manifest_2027_2027.json'
    csv_path = root / 'usno_events_2027_2027.csv'
    meta = json.loads(man_path.read_text(encoding='utf-8'))
    if (meta.get('years') != [2027,2027] or meta.get('successful_sources') != 2 or
            meta.get('failed_sources') != 0 or len(meta.get('source_records', [])) != 2):
        raise ValueError('2027 USNO source manifest is incomplete')
    original = []
    raw_hashes = {}
    for rec in meta['source_records']:
        kind=rec.get('kind')
        if (rec.get('year') != 2027 or kind not in ('moon_phases','seasons') or
                rec.get('status') != 'ok' or kind in raw_hashes or
                rec.get('path') != f'raw/{kind}/2027.json'):
            raise ValueError('2027 USNO source record drift')
        raw=(root / 'raw' / kind / '2027.json').read_bytes()
        if digest(raw)!=rec['sha256'] or len(raw)!=rec['size_bytes']:
            raise ValueError('2027 USNO JSON bytes changed')
        rows=validate_json(raw,year=2027,kind=kind,source_url=rec['url'])
        if len(rows)!=rec['event_count']:
            raise ValueError('2027 USNO events count changed')
        original.extend(rows)
        raw_hashes[kind]=digest(raw)
    if len(raw_hashes)!=2 or len(original)!=meta.get('events'):
        raise ValueError('2027 USNO raw source coverage mismatch')
    original.sort(key=lambda r:(r['event_at_utc'],r['event_type'],r['event_name']))
    # Match the exact normalized CSV produced by the already-frozen fetcher.
    source=csv_path.read_bytes()
    if digest(source)!=meta['events_csv_sha256']:
        raise ValueError('2027 USNO normalized CSV hash mismatch')
    parsed=list(csv.DictReader(source.decode('utf-8-sig').splitlines()))
    if len(parsed)!=len(original):
        raise ValueError('2027 USNO normalized event count mismatch')
    for a,b in zip(parsed,original):
        if any(a[k]!=str(b[k]) for k in ('event_at_utc','event_at_taipei','event_type',
                                       'event_name','source_year','source_url','raw_sha256')):
            raise ValueError('2027 normalized events disagree with raw JSON')
    return original, {'manifest_sha256':digest(man_path.read_bytes()),
                      'csv_sha256':digest(source),'raw_json_sha256':raw_hashes,
                      'event_count':len(original)}


def build_sky_table(*,old_root:Path, future_root:Path) -> tuple[list[dict],dict]:
    old,_,old_audit=p1.verify_source_archive(
        old_root/'usno_events_2004_2026.csv',old_root/'manifest_2004_2026.json',old_root/'raw')
    anchors,moons=p1.parse_anchors(old)
    future,extra=_verified_2027(future_root)
    season_extra=[]; moon_extra=[]
    for row in future:
        when=p1.parse_local(row['event_at_taipei'])
        if row['event_type']=='seasons' and row['event_name'] in ('equinox','solstice'):
            name=p1.SEASON_MONTHS.get((when.month,row['event_name']))
            if not name:
                raise ValueError('unknown 2027 equinox/solstice')
            season_extra.append((when,name))
        elif row['event_type']=='moon_phases' and row['event_name']=='new_moon':
            moon_extra.append(when)
    if len(season_extra)!=4 or not 12<=len(moon_extra)<=13:
        raise ValueError('2027 astronomical anchor coverage incomplete')
    anchors += season_extra; moons += moon_extra
    if any(anchors[i][0]>=anchors[i+1][0] for i in range(len(anchors)-1)):
        raise ValueError('solar chronology changed')
    if any(moons[i]>=moons[i+1] for i in range(len(moons)-1)):
        raise ValueError('moon chronology changed')
    day=SKY_START
    rows=[]
    while day<=SKY_END:
        ref=datetime(day.year,day.month,day.day,12,tzinfo=p1.TZ)
        solar=p2.solar_segment(ref,anchors)
        lunar=p2.lunar_segment(ref,moons)
        if not solar or not lunar:
            raise ValueError('P3 date lacks both precomputed USNO boundary anchors: '+day.isoformat())
        sky=p2.hexagram_from_sectors(int(solar['solar_sector_8']),int(lunar['lunar_sector_8']))
        rows.append({'date':day.isoformat(),'reference_at_taipei':ref.isoformat(timespec='minutes'),
                     'solar_sector_8':solar['solar_sector_8'],
                     'lunar_sector_8':lunar['lunar_sector_8'],
                     'astronomical_hexagram_code':sky['astronomical_hexagram_code'],
                     'mapping_version':SKY_RULE})
        day+=timedelta(days=1)
    expected_days=(SKY_END-SKY_START).days+1
    if len(rows)!=expected_days:
        raise AssertionError('wrong sky table size')
    source={'base_events_sha256':old_audit['usno_event_csv_sha256'],
            'base_manifest_sha256':old_audit['usno_manifest_sha256'],
            'base_json_count':old_audit['usno_original_json_files'],
            'future_2027':extra,
            'mapping_version':SKY_RULE,
            'date_span':[SKY_START.isoformat(),SKY_END.isoformat()]}
    return rows,source


def parity_counts()->list[int]:
    values=p2a.exact_parity_masses()
    if len(values)!=64 or sum(values)!=comb(49,6) or len(set(values))==1:
        raise AssertionError('incorrect fair hexagram class masses')
    return values


def tilted_normalizer(sky:int,masses:list[int],anti:bool)->int:
    if type(sky) is not int or sky not in range(64) or len(masses)!=64:
        raise ValueError('illegal astronomical code')
    return sum(masses[y] * (ALT_WEIGHT_NUM**(d if anti else 6-d)) *
               (ALT_WEIGHT_DEN**(6-d if anti else d))
               for y in range(64) for d in [(sky^y).bit_count()])


def lr_step(sky:int, out:int, masses:list[int],anti:bool)->Fraction:
    if type(out) is not int or out not in range(64):
        raise ValueError('illegal observed parity code')
    d=(sky^out).bit_count()
    k=d if anti else 6-d
    weight=ALT_WEIGHT_NUM**k * ALT_WEIGHT_DEN**(6-k)
    return Fraction(weight*sum(masses), tilted_normalizer(sky,masses,anti))


def primary_process(states:list[tuple[int,int]],masses:list[int])->dict:
    plus=minus=Fraction(1,1)
    max_e=Fraction(1,1)
    steps=[]
    for sky,out in states:
        plus*=lr_step(sky,out,masses,False)
        minus*=lr_step(sky,out,masses,True)
        mix=(plus+minus)/2
        max_e=max(max_e,mix)
        steps.append({'sky_code':sky,'outcome_code_OUTCOME_ONLY':out,
                      'hamming_distance_OUTCOME_ONLY':(sky^out).bit_count(),
                      'e_numerator':str(mix.numerator),'e_denominator':str(mix.denominator),
                      'review_threshold_crossed':mix>=REVIEW_THRESHOLD})
    mixture=(plus+minus)/2
    return {'observed_count':len(states),'e_numerator':str(mixture.numerator),
            'e_denominator':str(mixture.denominator),'e_approx':float(mixture),
            'ever_crossed_review_threshold':max_e>=REVIEW_THRESHOLD,
            'terminal_crossed_review_threshold':mixture>=REVIEW_THRESHOLD,
            'steps':steps}


def make_seal(source:dict,sky_csv_sha256:str)->dict:
    masses=parity_counts()
    return {
        'schema':SEAL_SCHEMA,'study_id':'v24-p3-usno-sky-parity100-20261114',
        'study_type':'FUTURE_ONLY_NON_WAGER_PROSPECTIVE_SHADOW',
        'first_eligible_draw':{'draw_id':FIRST_ID,'date':FIRST_DATE},
        'horizon_consecutive_verified_draws':HORIZON,
        'sky_table_path':f'prospective/{SKY_FILENAME}',
        'sky_table_sha256':sky_csv_sha256,'sky_table_date_span':source['date_span'],
        'source_hashes':source,
        'exact_fair_parity_counts_by_code_0_to_63':masses,
        'exact_fair_universe':sum(masses),
        'model':{
            'astronomical_sky_code':'USNO solar8 upper Gray + newmoon-time8 lower Gray',
            'outcome_code':'ascending six main numbers odd yang=1 even yin=0, OUTCOME_ONLY',
            'odds_ratio_per_matching_or_mismatching_bit':'5/4',
            'likelihoods':'w_plus=5^(6-d)*4^d; w_minus=5^d*4^(6-d); exact normalizers Z_s=sum_y p0(y)*w(s,y)',
            'primary_e_process':'0.5*product_t(w_plus/Z_plus)+0.5*product_t(w_minus/Z_minus)',
            'one_primary_endpoint_only':True,
            'review_threshold_e_value':REVIEW_THRESHOLD,
            'never_automatically_conclude_predictive_go':True,
            'reference_time':'12:00 Asia/Taipei, not verified draw time',
            'astronomy_timestamp_vs_lottery_outcomes':'precomputed astronomy is independent of winning numbers',
        },
        'verification':'REQUIRE_TWO_PUBLISHER_EVIDENCE_PAGES_PER_OFFICIAL_DRAW',
        'previous_exploratory_results':'P1A/P2A NO_EVIDENCE; no retrospective tuning',
        'separate_existing_seals':('v23_m0_global_static200_20261113_seal.json',
                                  'v027_p3f_static10_20261110_seal.json',
                                  'v029_p5a_static9_20261113_seal.json'),
        'current_result':'PENDING_0_OF_100_NO_PREDICTIVE_GO',
    }


def freeze(*, old_root:Path, future_root:Path, out_dir:Path)->tuple[dict,bytes]:
    """Write-once sky table and pre-outcome protocol seal; reject existing paths."""
    table_path=out_dir/SKY_FILENAME;seal_path=out_dir/SEAL_FILENAME
    if table_path.exists() or seal_path.exists():
        raise FileExistsError('P3 sky table / original seal already exists; never replace')
    rows,source=build_sky_table(old_root=old_root,future_root=future_root)
    raw=encode_csv(rows)
    seal=make_seal(source,digest(raw))
    out_dir.mkdir(parents=True,exist_ok=True)
    with table_path.open('xb') as f:f.write(raw)
    with seal_path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(seal,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.write('\n')
    return seal,raw


def load_seal_and_sky(*,seal_path:Path, sky_path:Path,
                      old_root:Path|None=None,future_root:Path|None=None)->tuple[dict,dict[str,int]]:
    seal=json.loads(seal_path.read_text(encoding='utf-8'))
    if (seal.get('schema')!=SEAL_SCHEMA or seal.get('study_id')!='v24-p3-usno-sky-parity100-20261114' or
        seal.get('study_type')!='FUTURE_ONLY_NON_WAGER_PROSPECTIVE_SHADOW' or
        seal.get('first_eligible_draw')!={'draw_id':FIRST_ID,'date':FIRST_DATE} or
        seal.get('horizon_consecutive_verified_draws')!=HORIZON or
        seal.get('sky_table_path')!=f'prospective/{SKY_FILENAME}' or
        seal.get('sky_table_date_span')!=[SKY_START.isoformat(),SKY_END.isoformat()] or
        seal.get('source_hashes',{}).get('base_events_sha256')!=BASE_EVENTS_SHA256 or
        seal.get('source_hashes',{}).get('base_json_count')!=46 or
        seal.get('source_hashes',{}).get('mapping_version')!=SKY_RULE or
        seal.get('exact_fair_parity_counts_by_code_0_to_63')!=parity_counts() or
        seal.get('exact_fair_universe')!=comb(49,6) or
        seal.get('model',{}).get('odds_ratio_per_matching_or_mismatching_bit')!='5/4' or
        seal.get('model',{}).get('one_primary_endpoint_only') is not True or
        seal.get('model',{}).get('review_threshold_e_value')!=REVIEW_THRESHOLD or
        seal.get('model',{}).get('never_automatically_conclude_predictive_go') is not True):
        raise ValueError('P3 pre-outcome model protocol drift')
    table=sky_path.read_bytes()
    if digest(table)!=seal.get('sky_table_sha256'):
        raise ValueError('P3 future sky table changed after precommit')
    rows=list(csv.DictReader(table.decode('utf-8-sig').splitlines()))
    if not rows or tuple(rows[0])!=SKY_COLS:
        raise ValueError('P3 sky CSV schema drift')
    all_days=(SKY_END-SKY_START).days+1
    if len(rows)!=all_days or rows[0]['date']!=SKY_START.isoformat() or rows[-1]['date']!=SKY_END.isoformat():
        raise ValueError('P3 future sky date coverage incomplete')
    sky={}
    for j,r in enumerate(rows):
        day=SKY_START+timedelta(days=j)
        if r['date']!=day.isoformat() or r['reference_at_taipei']!=datetime(day.year,day.month,day.day,12,tzinfo=p1.TZ).isoformat(timespec='minutes') or r['mapping_version']!=SKY_RULE:
            raise ValueError('P3 future sky row has wrong date/ref/version')
        solar,lunar=int(r['solar_sector_8']),int(r['lunar_sector_8'])
        if not (0<=solar<=7 and 0<=lunar<=7):raise ValueError('invalid sky sector')
        code=(p2.gray3(solar)<<3)|p2.gray3(lunar)
        if r['astronomical_hexagram_code']!=str(code):
            raise ValueError('sky code not derived from frozen solar/lunar sectors')
        sky[r['date']]=code
    if old_root is not None and future_root is not None:
        recalculated,source=build_sky_table(old_root=old_root,future_root=future_root)
        if digest(encode_csv(recalculated))!=seal['sky_table_sha256'] or source!=seal['source_hashes']:
            raise ValueError('original USNO raw archives do not reproduce sealed future sky table')
    return seal,sky


def verified_future_100(csv_root:Path,bin_root:Path,evidence_root:Path)->tuple[list[dict],dict,list[dict]]:
    """Reuse v2.3 trusted independent evidence gate, limited to our first 100 events."""
    from .draw_ingestion_m2 import audit_source, _from_csv_bytes, normalize_draw
    from .official_draw_evidence_m2 import verify_local_evidence
    from scripts.export_lotto649_csv import generate_yearly_csv

    archive=audit_source(csv_root,bin_root)
    if archive['total_draws']<2498 or archive['last_draw_id']<115000104:
        raise ValueError('missing frozen 2498-draw research baseline')
    yearly,_=generate_yearly_csv(bin_root)
    all_rows=[row for rel,b in sorted(yearly.items()) for row in _from_csv_bytes(rel,b)]
    anchors=[x for x in all_rows if x['draw_id']==115000104]
    if len(anchors)!=1 or anchors[0]!={'draw_id':115000104,'date':'2026-11-10',
                                      'main':[11,23,30,32,40,41],'bonus':28}:
        raise ValueError('original official 2498-draw anchor changed')
    future=[x for x in all_rows if x['draw_id']>=FIRST_ID][:HORIZON]
    # Equivalent strict chronological identity audit to the frozen v2.3 trial,
    # but directly on M2 canonical records without importing v2.3's optional
    # NumPy-dependent prize/coverage engine.
    prev_id=None
    prev_day=None
    for i,row in enumerate(future):
        normalized=normalize_draw(row)
        if normalized!=row:
            raise ValueError('M2 normalization changed a future draw')
        day=date.fromisoformat(row['date'])
        if i==0 and (row['draw_id'],row['date'])!=(FIRST_ID,FIRST_DATE):
            raise ValueError('cannot start at a different future official draw')
        if i>0:
            old_year=prev_id//1000000
            new_year=row['draw_id']//1000000
            contiguous=(new_year==old_year and row['draw_id']==prev_id+1) or (
                new_year==old_year+1 and row['draw_id']==new_year*1000000+1)
            if not contiguous or day<=prev_day:
                raise ValueError('gap or reordered future draw in research series')
        prev_id=row['draw_id']
        prev_day=day
    proofs=[]
    for r in future:
        e=verify_local_evidence(r['draw_id'],evidence_root)
        if e['draw']!=r:raise ValueError('publisher evidence disagrees with archived official draw')
        proofs.append({'source_index_sha256':e['source_index_sha256'],'page_sha256':e['page_sha256']})
    return future,archive,proofs


def evaluate(*, seal_path:Path,sky_path:Path,csv_root:Path,bin_root:Path,
             evidence_root:Path,previous_report:Path|None=None,old_root:Path|None=None,
             future_root:Path|None=None)->dict:
    seal,sky=load_seal_and_sky(seal_path=seal_path,sky_path=sky_path,
                               old_root=old_root,future_root=future_root)
    rows,archive,proofs=verified_future_100(csv_root,bin_root,evidence_root)
    masses=parity_counts();pairs=[];observed=[]
    for row,e in zip(rows,proofs):
        day=row['date'];s=sky.get(day)
        if s is None:raise ValueError('verified official draw date outside precommitted sky table')
        outcome=p2a.outcome_code(tuple(row['main']))
        pairs.append((s,outcome))
        observed.append({'date':day,'draw_id':row['draw_id'],
                         'main':row['main'],'bonus':row['bonus'],
                         'sky_code_precommitted':s,'outcome_code_OUTCOME_ONLY':outcome,
                         'source_evidence':e})
    progress=primary_process(pairs,masses)
    for item,p in zip(observed,progress['steps']):
        item.update({'hamming_distance_OUTCOME_ONLY':p['hamming_distance_OUTCOME_ONLY'],
                     'e_numerator':p['e_numerator'],'e_denominator':p['e_denominator'],
                     'review_threshold_crossed':p['review_threshold_crossed']})
    n=len(rows)
    status=('COMPLETED_100_OF_100_REVIEW_ONLY_NO_AUTOMATIC_GO' if n==HORIZON else
            'PENDING_REVIEW_ONLY_NO_AUTOMATIC_GO' if progress['ever_crossed_review_threshold'] else
            'PENDING_NO_PREDICTIVE_GO')
    report={'schema':REPORT_SCHEMA,'study_id':seal['study_id'],
            'seal_sha256':digest(seal_path.read_bytes()),
            'sky_table_sha256':seal['sky_table_sha256'],
            'status':status,'observed_draw_count':n,'pending_draw_count':HORIZON-n,
            'first_future_draw':{'draw_id':FIRST_ID,'date':FIRST_DATE},
            'observed_draws':observed,
            'anytime_e_process':{k:v for k,v in progress.items() if k!='steps'},
            'review_threshold':REVIEW_THRESHOLD,
            'source_archive_gate':{'status':archive['status'],'archive_total_draws':archive['total_draws'],
                                   'last_draw_id':archive['last_draw_id'],
                                   'manifest_sha256':archive['source_fingerprints']['manifest_sha256'],
                                   'verified_new_draws':len(proofs)},
            'no_outcome_leakage':True,'predictive_go':False,'actual_wagers_placed':False,
            'no_independent_familywise_control_across_other_trials':True,
            'notes':['Future-only independent source verification via two archived publisher pages each',
                     'One primary fixed 50:50 mixture of alignment / anti-alignment exponentials',
                     'E>=20 is manual methodology review ONLY, not automatic predictive confirmation',
                     'Cannot establish historic lottery astronomical causality; 100 draws may have low power',
                     'Zero-count status is not a successful forecasting result']}
    if previous_report is not None:
        previous=json.loads(previous_report.read_text('utf-8'))
        if (previous.get('schema')!=REPORT_SCHEMA or previous.get('seal_sha256')!=report['seal_sha256'] or
            previous.get('sky_table_sha256')!=report['sky_table_sha256'] or
            previous.get('observed_draws')!=observed[:len(previous.get('observed_draws',[]))] or
            previous.get('observed_draw_count',-1)>n):
            raise ValueError('P3 previous report not an unchanged externally authenticated prefix')
    return report


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    build=sub.add_parser('freeze',help='once only, pre-outcome future calendar + seal')
    build.add_argument('--old-root',type=Path,default=Path('data/astronomy/usno_official'))
    build.add_argument('--future-root',type=Path,default=Path('data/astronomy/usno_future_2027'))
    build.add_argument('--out-dir',type=Path,default=Path('prospective'))
    run=sub.add_parser('evaluate',help='source-verified read-only report; do not download or buy')
    run.add_argument('--seal',type=Path,default=Path('prospective')/SEAL_FILENAME)
    run.add_argument('--sky',type=Path,default=Path('prospective')/SKY_FILENAME)
    run.add_argument('--old-root',type=Path,default=Path('data/astronomy/usno_official'))
    run.add_argument('--future-root',type=Path,default=Path('data/astronomy/usno_future_2027'))
    run.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    run.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    run.add_argument('--evidence-root',type=Path,default=Path('data/private/lotto649-source'))
    run.add_argument('--previous',type=Path)
    run.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    if args.cmd=='freeze':
        seal,table=freeze(old_root=args.old_root,future_root=args.future_root,out_dir=args.out_dir)
        print('P3 PRECOMMITTED',seal['study_id'],'days',table.count(b'\n')-1,'table sha',seal['sky_table_sha256'])
        return 0
    p=args.out.resolve()
    if p.exists() or not p.is_relative_to(Path('results').resolve()):
        ap.error('report must be a new file inside results/ (no overwrite)')
    report=evaluate(seal_path=args.seal,sky_path=args.sky,csv_root=args.csv_root,
                    bin_root=args.bin_root,evidence_root=args.evidence_root,
                    previous_report=args.previous,old_root=args.old_root,future_root=args.future_root)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
    print('P3',report['status'],'verified',report['observed_draw_count'],'remaining',
          report['pending_draw_count'],'E',report['anytime_e_process']['e_approx'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())