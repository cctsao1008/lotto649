"""v2.4 P0: source-audited historical-draw, astronomy and six-line taxonomy data.

Read-only inputs. CWA calendar values are imported only from separately supplied
CSV, never simulated and never silently inferred from an ephemeris. Astronomy
columns are deterministic ephemeris calculations at 12:00 Asia/Taipei on the
Gregorian draw date, NOT purported historical observations at draw time.

The parity-to-hexagram mapping is a research encoding, not traditional divination.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone, timedelta
from hashlib import sha256
from io import StringIO
import json
from math import comb, cos, pi, sin, degrees
from pathlib import Path

SCHEMA = "lotto649.astronomy_hexagram.v2.4.p0"
UNIVERSE = comb(49, 6)
TZ_TAIPEI = timezone(timedelta(hours=8))
HEX_ENCODING = "ascending_6_numbers_bottom_to_top_odd_yang_even_yin_v1"
CWA_DATASET = "https://data.gov.tw/dataset/157677"
CWA_ID = "A-A0087-001"
CWA_FORMAT_ALIASES = {
    'date': ('GregorianCalendar','格里曆日期','日期','date'),
    'lunar_year_label': ('AgriculturalCalendar','農曆年','農曆歲次'),
    'lunar_month': ('AgriculturalCalendarMonth','農曆月'),
    'lunar_day': ('AgriculturalCalendarDay','農曆日'),
    'roc_year': ('RepublicOfChinaCalendar','民國年','民國曆年'),
    'week': ('Week','星期'),
}
SOLAR_TERMS = (
    'spring_equinox','clear_and_bright','grain_rain','start_summer','grain_full',
    'grain_in_ear','summer_solstice','minor_heat','major_heat','start_autumn',
    'end_of_heat','white_dew','autumn_equinox','cold_dew','frost_descent',
    'start_winter','minor_snow','major_snow','winter_solstice','minor_cold',
    'major_cold','start_spring','rain_water','awakening_insects',
)
COLUMNS = (
    'date','draw_id','n1','n2','n3','n4','n5','n6','bonus',
    'weekday_iso','day_of_year','draw_gap_days','previous_hexagram_code',
    'hexagram_code','hexagram_binary_bottom_to_top','lower_trigram_code',
    'upper_trigram_code','yang_count','changed_lines_from_previous_draw',
    'solar_ecliptic_longitude_deg','solar_term_noon_bin','solar_cycle_sin','solar_cycle_cos',
    'moon_ecliptic_longitude_deg','moon_sun_elongation_deg','lunar_cycle_sin','lunar_cycle_cos',
    'moon_illumination_percent','days_since_astronomical_new_moon',
    'days_until_astronomical_new_moon','astronomy_reference_time_local',
    'astronomy_provenance','cwa_lunar_year_label','cwa_lunar_month','cwa_lunar_leap_month',
    'cwa_lunar_day','cwa_calendar_provenance',
)


def hex_code(ticket: tuple[int,...] | list[int]) -> int:
    if (len(ticket)!=6 or any(type(v) is not int or not 1<=v<=49 for v in ticket)
            or list(ticket)!=sorted(set(ticket))):
        raise ValueError('six strictly ascending unique 1..49 integers required')
    return sum((v&1)<<i for i,v in enumerate(ticket))


def exact_hexagram_universe(*,ball_count:int=49,ticket_size:int=6) -> dict[int,int]:
    """Exact DP over sorted ticket ordering. No assumption of uniform 64 codes."""
    if not 1<=ticket_size<=ball_count<=49:raise ValueError('invalid game')
    dp=[defaultdict(int) for _ in range(ticket_size+1)];dp[0][0]=1
    for num in range(1,ball_count+1):
        for k in range(min(num,ticket_size),0,-1):
            for code,count in list(dp[k-1].items()):
                dp[k][code|((num&1)<<(k-1))]+=count
    counts=dict(sorted(dp[ticket_size].items()))
    if sum(counts.values())!=comb(ball_count,ticket_size):raise AssertionError('universe incomplete')
    return counts


def parse_cwa_calendar(content:bytes) -> dict[str,dict]:
    """Accept documented CWA A-A0087-001 6-column CSV with strong date checks.

    Raw bytes are retained outside the program by the operator. Parsing a
    CWA-shaped file does NOT cryptographically prove it came from the CWA.
    """
    text=None
    for codec in ('utf-8-sig','cp950','big5'):
        try:
            text=content.decode(codec,'strict');break
        except UnicodeError:continue
    if text is None:raise ValueError('unknown CWA CSV encoding')
    reader=csv.DictReader(StringIO(text))
    if reader.fieldnames is None:raise ValueError('no CWA header')
    fields={f.strip().lstrip('\ufeff'):f for f in reader.fieldnames if f is not None}
    found={}
    for field,aliases in CWA_FORMAT_ALIASES.items():
        matched=[fields[x] for x in aliases if x in fields]
        if len(matched)!=1:raise ValueError(f'CWA required field absent/ambiguous: {field}')
        found[field]=matched[0]
    out={}
    for row in reader:
        ds=row[found['date']].strip().replace('/','-')
        if len(ds)==8 and ds.isdigit():ds=f'{ds[:4]}-{ds[4:6]}-{ds[6:]}'
        d=date.fromisoformat(ds)
        if d.isoformat()!=ds:raise ValueError('CWA date format must be YYYY-MM-DD')
        roc=int(row[found['roc_year']].strip())
        if roc!=d.year-1911:raise ValueError('CWA year mismatch')
        lm=row[found['lunar_month']].strip().replace('閏','闰')
        leap=lm.startswith('闰')
        month=int(lm[1:] if leap else lm)
        day=int(row[found['lunar_day']].strip())
        weekday=int(row[found['week']].strip())
        if not(1<=month<=12 and 1<=day<=30 and weekday==d.isoweekday()):
            raise ValueError('CWA weekday or lunar date outside range')
        if ds in out:raise ValueError('duplicate official calendar date')
        out[ds]={
            'cwa_lunar_year_label':row[found['lunar_year_label']].strip(),
            'cwa_lunar_month':str(month),'cwa_lunar_leap_month':str(int(leap)),
            'cwa_lunar_day':str(day),
        }
    if not out:raise ValueError('no CWA calendar observations')
    return out


def _read_source_draws(csv_root:Path,bin_root:Path):
    """All original rows must equal independently regenerated BIN-backed CSV."""
    from .draw_ingestion_m2 import audit_source, _from_csv_bytes
    from scripts.export_lotto649_csv import generate_yearly_csv
    gate=audit_source(csv_root,bin_root)
    regenerated,_=generate_yearly_csv(bin_root)
    rows=[]
    for relative,raw in sorted(regenerated.items()):
        rows.extend(_from_csv_bytes(relative,raw))
    if len(rows)!=gate['total_draws'] or rows[-1]['draw_id']!=gate['last_draw_id']:
        raise AssertionError('archive and generated draws differ')
    # Fixed P0 historical research cutoff. Later archived outcomes must never
    # silently alter the original 2004--2026 P0 dataset or its hashes.
    rows=[x for x in rows if x['draw_id']<=115000104]
    if len(rows)!=2498 or rows[-1]['date']!='2026-11-10' or rows[-1]['draw_id']!=115000104:
        raise ValueError('v2.4 P0 exact 2498-draw baseline unavailable')
    return rows,gate


def astro_features(day:str) -> dict:
    """PyEphem astronomical *calculation*, noon Taipei; not CWA measured data."""
    try:import ephem
    except ModuleNotFoundError as exc:raise RuntimeError('install ephem>=4.2,<5 for calculation') from exc
    local=datetime.combine(date.fromisoformat(day),datetime.min.time(),tzinfo=TZ_TAIPEI)+timedelta(hours=12)
    utc=local.astimezone(timezone.utc)
    e=ephem.Date(utc.replace(tzinfo=None))
    sun=ephem.Sun(e);moon=ephem.Moon(e)
    sunlon=(degrees(float(ephem.Ecliptic(sun).lon))%360.)
    moonlon=(degrees(float(ephem.Ecliptic(moon).lon))%360.)
    elong=(moonlon-sunlon)%360.
    prev=ephem.previous_new_moon(e)
    nxt=ephem.next_new_moon(e)
    age=float(e-prev);remaining=float(nxt-e)
    if not 0<=age<33 or not 0<remaining<33:raise AssertionError('invalid lunar phase ephemeris interval')
    r=pi/180.
    return {
        'solar_ecliptic_longitude_deg':f'{sunlon:.6f}',
        'solar_term_noon_bin':SOLAR_TERMS[int(sunlon//15)%24],
        'solar_cycle_sin':f'{sin(sunlon*r):.8f}',
        'solar_cycle_cos':f'{cos(sunlon*r):.8f}',
        'moon_ecliptic_longitude_deg':f'{moonlon:.6f}',
        'moon_sun_elongation_deg':f'{elong:.6f}',
        'lunar_cycle_sin':f'{sin(elong*r):.8f}',
        'lunar_cycle_cos':f'{cos(elong*r):.8f}',
        'moon_illumination_percent':f'{float(moon.phase):.4f}',
        'days_since_astronomical_new_moon':f'{age:.6f}',
        'days_until_astronomical_new_moon':f'{remaining:.6f}',
        'astronomy_reference_time_local':local.isoformat(timespec='minutes'),
        'astronomy_provenance':'PYEPHEM_CALCULATED_LOCAL_NOON_NOT_CWA_OBSERVED',
    }


def join_data(rows:list[dict],calendar:dict[str,dict]|None=None,*,astro_func=astro_features):
    if not isinstance(rows,list) or not rows:raise ValueError('no draw rows')
    cwa=calendar or {}
    cache={};out=[];previous_code=None;previous_date=None
    for x in rows:
        d=date.fromisoformat(x['date']);ds=d.isoformat()
        numbers=x['main'];code=hex_code(numbers)
        if previous_date is not None and d<=previous_date:raise ValueError('draws not sorted by date')
        if ds not in cache:cache[ds]=astro_func(ds)
        daydata=cache[ds]
        line={
            'date':ds,'draw_id':str(x['draw_id']),
            **{f'n{i}':f'{numbers[i-1]:02d}' for i in range(1,7)},
            'bonus':f'{x["bonus"]:02d}',
            'weekday_iso':str(d.isoweekday()),'day_of_year':str(d.timetuple().tm_yday),
            'draw_gap_days':str((d-previous_date).days) if previous_date else '',
            'previous_hexagram_code':str(previous_code) if previous_code is not None else '',
            'hexagram_code':str(code),
            'hexagram_binary_bottom_to_top':''.join('1' if n%2 else '0' for n in numbers),
            'lower_trigram_code':str(code&7),'upper_trigram_code':str(code>>3),
            'yang_count':str(code.bit_count()),
            'changed_lines_from_previous_draw':str((code^previous_code).bit_count()) if previous_code is not None else '',
            **daydata,**(cwa.get(ds) or {}),
            'cwa_calendar_provenance':'CWA_FORMAT_SOURCE_PROVIDED_NOT_CRYPT_AUTHENTICATED' if ds in cwa else 'NOT_AVAILABLE',
        }
        if set(line)-set(COLUMNS):raise ValueError('unexpected data column')
        out.append({c:str(line.get(c,'')) for c in COLUMNS})
        previous_date=d;previous_code=code
    return out


def build_dataset(csv_root:Path,bin_root:Path,*,cwa_csv:Path|None=None,astro_func=astro_features):
    rows,gate=_read_source_draws(csv_root,bin_root)
    raw=cwa_csv.read_bytes() if cwa_csv else None
    cwa=parse_cwa_calendar(raw) if raw is not None else {}
    joined=join_data(rows,cwa,astro_func=astro_func)
    theoretical=exact_hexagram_universe()
    counts=Counter(int(x['hexagram_code']) for x in joined)
    code_rows=[{'hexagram_code':i,'binary_bottom_to_top':''.join(str((i>>p)&1) for p in range(6)),
                'theory_combinations':theoretical[i],
                'theory_probability':theoretical[i]/UNIVERSE,
                'historical_draw_count':counts[i]} for i in range(64)]
    cwa_used=sum(x['cwa_calendar_provenance']!='NOT_AVAILABLE' for x in joined)
    years=Counter(x['date'][:4] for x in joined)
    cwa_yrs=Counter(x['date'][:4] for x in joined if x['cwa_calendar_provenance']!='NOT_AVAILABLE')
    report={
        'schema':SCHEMA,'decision':'DATASET_BUILD_PASS_WITH_OFFICIAL_CWA_COVERAGE' if cwa_used==len(joined) else 'DATASET_BUILD_PASS_CWA_PARTIAL_OR_PENDING',
        'source_archive_gate':gate,
        'rows':len(joined),'first_date':joined[0]['date'],'last_date':joined[-1]['date'],
        'earliest_draw_id':int(joined[0]['draw_id']),'latest_draw_id':int(joined[-1]['draw_id']),
        'per_year_draw_count':dict(sorted(years.items())),
        'cwa_calendar':{'dataset_id':CWA_ID,'reference':CWA_DATASET,
            'raw_bytes_sha256':sha256(raw).hexdigest() if raw is not None else None,
            'raw_rows':len(cwa),'matched_draws':cwa_used,
            'unmatched_draws':len(joined)-cwa_used,'per_year_matched':dict(sorted(cwa_yrs.items())),
            'authenticity':'local CSV format verified; publisher origin must be established by download custody' if cwa else 'NO_CWA_FILE_SUPPLIED_NO_CWA_VALUES_SYNTHESIZED'},
        'ephemeris':{'engine':'PyEphem','version_note':'calculated via library at 12:00 Asia/Taipei, not CWA measurement or draw time','time_assumption':'local_noon_date_only'},
        'hexagram':{'encoding':HEX_ENCODING,'source':'observed historical main six numbers after drawing; target/leakage-sensitive',
                    'theoretical_universe_size':UNIVERSE,'theoretical_count_total':sum(theoretical.values()),
                    'theoretical_min':min(theoretical.values()),'theoretical_max':max(theoretical.values()),
                    'categories':code_rows},
        'warnings':['No relationship to extra winning probability is established.',
                    'Do not use same-draw hexagram_code/changed_lines as a pre-draw predictor.',
                    'No unspecified weather-station climate data is imputed.',
                    'Solar terms are approximate noon-longitude bins, not exact CWA published transition times.',
                    'CWA data are only included if a separate locally archived CSV is supplied and verified.'],
    }
    return joined,report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--cwa-calendar',type=Path,default=None)
    p.add_argument('--out-csv',type=Path,required=True)
    p.add_argument('--out-json',type=Path,required=True)
    args=p.parse_args(argv)
    outcsv=args.out_csv.resolve();outjson=args.out_json.resolve();root=Path('results').resolve()
    if (outcsv==outjson or not outcsv.is_relative_to(root) or not outjson.is_relative_to(root) or
            outcsv.exists() or outjson.exists()):p.error('both outputs must be new distinct files inside results/')
    rows,report=build_dataset(args.csv_root,args.bin_root,cwa_csv=args.cwa_calendar)
    outcsv.parent.mkdir(parents=True,exist_ok=True)
    with outcsv.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report['output_csv_sha256']=sha256(outcsv.read_bytes()).hexdigest()
    with outjson.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,sort_keys=True,indent=2,ensure_ascii=False);f.write('\n')
    print('V24 DATASET',report['decision'],'ROWS',report['rows'],'CWA_MATCH',report['cwa_calendar']['matched_draws'])
    print('HEXAGRAM THEORY',report['hexagram']['theoretical_count_total'],'SHA256',report['output_csv_sha256'])
    print('CSV',outcsv,'JSON',outjson)
    return 0


if __name__=='__main__':raise SystemExit(main())
