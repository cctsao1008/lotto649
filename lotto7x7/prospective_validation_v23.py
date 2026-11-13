"""v2.3: strictly pre-registered static M0 global-score shadow against a fair control.

This module does not place bets, fetch web data, generate post-outcome selections,
or change previously sealed studies. Only two matching archived publisher-page
snapshots can admit each future evaluation row. A full objective-score optimum
is not itself evidence of future predictive advantage.
"""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha1, sha256
from itertools import chain
from math import comb
from pathlib import Path
from random import Random
import json

from .draw_ingestion_m2 import audit_source, _from_csv_bytes, normalize_draw
from .official_draw_evidence_m2 import verify_local_evidence
from .official_prize_v026 import official_prize_coverage, prize_tier, PRIZE_ORDER

SCHEMA = 'lotto649.v2.3.prospective.fixed_m0_full_universe.v1'
REPORT_SCHEMA = 'lotto649.v2.3.prospective.static200.evaluation.v1'
SEAL_PATH = Path('prospective/v23_m0_global_static200_20261113_seal.json')
SEAL_BLOB_SHA1 = 'f02a94b37bc2287f89cfbc6dd7f6ab2daa4d16be'
FIRST_ID = 115000106
FIRST_DAY = '2026-11-17'
HORIZON = 200
STARTING_ANCHOR_ID = 115000104
EXPECTED_OPTIMAL_QUALITY = [43549, 195617, 43549, 54184]
EXPECTED_B = [[9,30,33,35,38,41], [10,26,29,36,42,43],
              [11,21,31,37,40,46], [12,24,27,32,44,47]]
EXPECTED_BASELINE_MANIFEST_SHA256 = '003ed03eda889952654b679859f2d465ffb7349c7f005a6df3aa4436be3d54a3'


def _git_blob_sha1(raw: bytes) -> str:
    return sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def _canonical_seal_bytes(raw: bytes) -> bytes:
    if b'\r' in raw:
        if b'\r' in raw.replace(b'\r\n', b''):
            raise ValueError('mixed CRLF in seal')
        raw = raw.replace(b'\r\n', b'\n')
    if _git_blob_sha1(raw) != SEAL_BLOB_SHA1:
        raise ValueError('frozen v2.3 Git blob SHA1 mismatch; never modify preregistration')
    return raw


def _validate_tickets(tickets: object) -> list[list[int]]:
    if not isinstance(tickets, list) or len(tickets) != 4:
        raise ValueError('four tickets per arm required')
    rows = []
    for ticket in tickets:
        if (not isinstance(ticket,list) or len(ticket)!=6 or
            any(type(v) is not int or not 1<=v<=49 for v in ticket) or
            ticket!=sorted(set(ticket))):
            raise ValueError('six ordered unique legal numbers required per ticket')
        rows.append(ticket.copy())
    if len(set(chain.from_iterable(rows))) != 24:
        raise ValueError('24 disjoint numbers required in each arm')
    return rows


def _control_seed_tickets(seed:int) -> list[list[int]]:
    labels=list(range(1,50))
    Random(seed).shuffle(labels)
    return [sorted(labels[6*i:6*(i+1)]) for i in range(4)]


def validate_manifest(obj: dict) -> dict:
    """Fail closed on protocol drift, including non-wager scope and first outcome."""
    if not isinstance(obj,dict) or obj.get('schema')!=SCHEMA or obj.get('experiment_id')!='v23-m0-full-universe-shadow-20261113':
        raise ValueError('unknown v2.3 preregistration schema or study identifier')
    if obj.get('study_type')!='PROSPECTIVE_STATIC_FROZEN_SHADOW_NO_REAL_WAGERS':
        raise ValueError('study is shadow-only')
    s=obj['selection'];p=obj['prospective_protocol'];base=obj['source_baseline']
    b=_validate_tickets(s['history_optimized_strategy']['tickets'])
    a=_validate_tickets(s['history_blind_control']['tickets'])
    if (b!=EXPECTED_B or s['history_optimized_strategy']['v21_certified_quality']!=EXPECTED_OPTIMAL_QUALITY or
        s['history_optimized_strategy']['cutoff_draw_id']!=115000103 or
        s['history_optimized_strategy']['v21_result_json_sha256']!='9daf72152925a7a873418155b3f194bfbdce6e4cd37174ddc52e5d68ab811b86'):
        raise ValueError('globally optimized M0 source witness drift')
    if (s['history_blind_control']['seed']!=20261113 or a!=_control_seed_tickets(20261113) or
        not s['fixed_until_end'] or not s['same_exact_iid_prize_profile_required'] or
        s['tickets_per_arm_per_draw']!=4 or s['distinct_numbers_per_arm']!=24 or
        s['purchases_authorized'] is not False):
        raise ValueError('control RNG / eligible four-ticket rule drift')
    if (base['last_archived_draw_id_at_seal']!=STARTING_ANCHOR_ID or
        base['source_manifest_sha256_at_seal']!=EXPECTED_BASELINE_MANIFEST_SHA256 or
        base['last_archived_draw_count_at_seal']!=2498):
        raise ValueError('initial source archive changed')
    monitor=p['anytime_valid_monitor']
    if (p['first_draw_id']!=FIRST_ID or p['first_draw_date']!=FIRST_DAY or
        p['horizon_consecutive_official_draws']!=HORIZON or
        monitor['B_only_factor']!='13/10' or monitor['A_only_factor']!='7/10' or
        monitor['both_or_neither_factor']!='1' or monitor['review_threshold_e_value']!='20' or
        p['one_primary_test_only'] is not True):
        raise ValueError('prospective horizon or predeclared process drift')
    original_profile=official_prize_coverage(a)
    tested_profile=official_prize_coverage(b)
    keys=('any_prize_outcomes','awarded_ticket_count_outcomes','any_ticket_by_tier','expected_tickets_by_tier')
    if any(original_profile[k]!=tested_profile[k] for k in keys):
        raise AssertionError('fair IID official prize parity between A/B is required')
    return {
        'arms':{'A':a,'B':b},
        'probability_any_official_prize_per_arm':original_profile['any_prize_probability'],
        'theoretical_full_official_prize_profile_equal':True,
        'first_draw':{'draw_id':FIRST_ID,'date':FIRST_DAY},
        'horizon':HORIZON,
    }


def load_frozen_seal(path:Path=SEAL_PATH) -> tuple[dict,dict]:
    raw=_canonical_seal_bytes(Path(path).read_bytes())
    obj=json.loads(raw.decode('utf-8'))
    validated=validate_manifest(obj)
    return obj,validated


def _check_contiguous_future(rows:list[dict]) -> list[dict]:
    if not isinstance(rows,list) or len(rows)>HORIZON:
        raise ValueError('observations must be a future-prefix list no longer than sealed horizon')
    out=[]
    last_id=None;last_date=None
    for i,row in enumerate(rows):
        item=normalize_draw(row)
        current_date=date.fromisoformat(item['date'])
        if i==0:
            if (item['draw_id'],item['date'])!=(FIRST_ID,FIRST_DAY):
                raise ValueError('future trial cannot start after first locked draw')
        else:
            previous_year=last_id//1000000
            current_year=item['draw_id']//1000000
            next_ok=(current_year==previous_year and item['draw_id']==last_id+1) or (
                current_year==previous_year+1 and item['draw_id']==(current_year*1000000+1))
            if not next_ok:
                raise ValueError('missing, duplicate, reversed or skipped future draw in official sequence')
            if current_date<=last_date:
                raise ValueError('prospective dates must increase strictly')
        last_date=current_date;last_id=item['draw_id']
        out.append(item)
    return out


def _awards(ticket_lines:list[list[int]], mains:list[int], bonus:int) -> dict:
    drawn=set(mains)
    tiers=[prize_tier(len(drawn.intersection(t)),bonus in t) for t in ticket_lines]
    awarded=[tier for tier in tiers if tier is not None]
    return {'any_award':bool(awarded),'award_count':len(awarded),
            'tier_counts':{k:awarded.count(k) for k in PRIZE_ORDER if k in awarded}}


def exact_one_sided_discordance_p(b_only:int,a_only:int)->float:
    if any(type(x) is not int or x<0 for x in (b_only,a_only)):
        raise ValueError('nonnegative integer paired-discordance counts required')
    d=b_only+a_only
    if d>HORIZON:raise ValueError('discordances exceed sealed horizon')
    return sum(comb(d,k) for k in range(b_only,d+1))/(2**d)


def _evaluate_sealed_rows(rows:list[dict], seal:dict,meta:dict,
                          *, external_source_verified:bool=False,
                          external_evidence:list[dict]|None=None)->dict:
    """Pure pairing; actual CLI is the only code path that authenticates sources."""
    rows=_check_contiguous_future(rows)
    if external_source_verified and (external_evidence is None or len(external_evidence)!=len(rows)):
        raise ValueError('source evidence must cover every measured draw')
    arms=meta['arms']; observed=[]
    a_only=b_only=both=neither=0
    tiers={'A':{x:0 for x in PRIZE_ORDER},'B':{x:0 for x in PRIZE_ORDER}}
    for i,row in enumerate(rows):
        events={k:_awards(v,row['main'],row['bonus']) for k,v in arms.items()}
        a=events['A']['any_award'];b=events['B']['any_award']
        a_only+=int(a and not b);b_only+=int(b and not a)
        both+=int(a and b);neither+=int(not a and not b)
        for key in arms:
            for grade,count in events[key]['tier_counts'].items():tiers[key][grade]+=count
        num=pow(13,b_only)*pow(7,a_only)
        den=pow(10,b_only+a_only)
        observed.append({**row,'arms':events,
            'e_process_num':str(num),'e_process_den':str(den),
            'threshold_crossed':num>=20*den,
            'source_evidence':external_evidence[i] if external_source_verified else None})
    n=len(rows)
    num=pow(13,b_only)*pow(7,a_only)
    den=pow(10,b_only+a_only)
    crossed=any(row['threshold_crossed'] for row in observed)
    complete=n==HORIZON
    if not external_source_verified and n:
        status='UNVERIFIED_INPUT_REJECT_FOR_SCIENTIFIC_REPORT'
    elif complete:
        status='COMPLETE_MANUAL_REVIEW_NO_AUTOMATIC_PREDICTIVE_GO'
    elif crossed:
        status='PENDING_REVIEW_THRESHOLD_CROSSED_NO_GO'
    else:
        status='PENDING_NO_PREDICTIVE_GO'
    return {
        'schema':REPORT_SCHEMA,'experiment_id':seal['experiment_id'],
        'seal_git_blob_sha1':SEAL_BLOB_SHA1,
        'source_verification':('TWO_ARCHIVED_INDEPENDENT_SOURCE_PAGES_PER_DRAW' if external_source_verified
                               else 'PURE_FUNCTION_ONLY_NO_EXTERNAL_SOURCE_CLAIM'),
        'status':status,'planned_draws':HORIZON,'observed_draw_count':n,
        'pending_draw_count':HORIZON-n,
        'first_future_draw':{'draw_id':FIRST_ID,'date':FIRST_DAY},
        'next_draw_expected_by_id':FIRST_ID if not rows else ('AFTER_LATEST_CONTIGUOUS_OFFICIAL_DRAW' if n<HORIZON else None),
        'iid_theoretical_any_prize_per_draw_equal':meta['probability_any_official_prize_per_arm'],
        'fair_full_award_profile_parity':meta['theoretical_full_official_prize_profile_equal'],
        'paired_counts':{'A_only':a_only,'B_only':b_only,'both':both,'neither':neither},
        'observed_any_award_draws':{'A':a_only+both,'B':b_only+both},
        'observed_award_rate_difference_B_minus_A':(b_only-a_only)/n if n else None,
        'all_tiers':tiers,
        'anytime_e_process':{'numerator':str(num),'denominator':str(den),
                            'approximate_e_value_for_display_only':num/den,
                            'review_threshold':20,
                            'ever_crossed_review_threshold':crossed,
                            'anytime_p_upper_bound_single_study_for_display_only':min(1.0,den/num)},
        'terminal_one_sided_discordance_p':(exact_one_sided_discordance_p(b_only,a_only) if complete else None),
        'predicted_winning_probability_advantage_established':False,
        'predictive_go':False,
        'actual_wagers_placed':False,
        'observed_draws':observed,
        'caveats':[
            'A precommitted global optimizer score does not imply increased fair-IID draw or prize odds.',
            'Shadow portfolio performance is not observed real-money profit or return.',
            'This study controls only its own one-sided e-process, not multiplicity across other prior or concurrent model studies.',
            'Crossing E >= 20 initiates source and methodology review only, not automatic prediction or betting approval.',
            'Input-only pure evaluation does not authenticate draw source; real CLI requires two archived publisher pages for each observation.'
        ],
    }


def assert_append_only(previous:dict, current:dict)->None:
    if not isinstance(previous,dict) or previous.get('schema')!=REPORT_SCHEMA:
        raise ValueError('prior report schema invalid')
    if previous.get('seal_git_blob_sha1')!=current.get('seal_git_blob_sha1'):
        raise ValueError('seal identity changed')
    old=previous.get('observed_draws')
    new=current.get('observed_draws')
    if not isinstance(old,list) or not isinstance(new,list) or len(old)>len(new):
        raise ValueError('observed prospective sequence regressed')
    if old!=new[:len(old)]:
        raise ValueError('prior observed or externally verified evidence was changed')


def read_source_verified_future(csv_root:Path,bin_root:Path,evidence_root:Path,
                                *,max_draws:int=HORIZON)->tuple[list[dict],dict,list[dict]]:
    from scripts.export_lotto649_csv import generate_yearly_csv
    if not 0<=max_draws<=HORIZON:raise ValueError('max draw limit exceeds sealed horizon')
    audit=audit_source(csv_root,bin_root)
    if audit['total_draws']<2498 or audit['last_draw_id']<STARTING_ANCHOR_ID:
        raise ValueError('baseline archive incomplete')
    yearly,_=generate_yearly_csv(Path(bin_root))
    all_rows=[row for rel,data in sorted(yearly.items()) for row in _from_csv_bytes(rel,data)]
    anchor=[x for x in all_rows if x['draw_id']==STARTING_ANCHOR_ID]
    if len(anchor)!=1 or anchor[0] != {'draw_id':115000104,'date':'2026-11-10','main':[11,23,30,32,40,41],'bonus':28}:
        raise AssertionError('pre-prospective source anchor changed')
    rows=[x for x in all_rows if x['draw_id']>=FIRST_ID][:HORIZON]
    _check_contiguous_future(rows)
    if max_draws<len(rows):
        raise ValueError('cannot deliberately truncate an available verified prospective sequence')
    evidence=[]
    for row in rows:
        proof=verify_local_evidence(row['draw_id'],Path(evidence_root))
        if proof['draw']!=row:
            raise ValueError('archive disagrees with independent publisher-page evidence')
        evidence.append({'source_index_sha256':proof['source_index_sha256'],
                         'page_sha256':proof['page_sha256']})
    return rows,audit,evidence


def run_pending_audit(*,seal_path:Path=SEAL_PATH,csv_root:Path=Path('data/csv'),
                      bin_root:Path=Path('data/bin'),evidence_root:Path=Path('data/private/lotto649-source'),
                      previous:Path|None=None)->dict:
    seal,meta=load_frozen_seal(seal_path)
    rows,source_audit,evidence=read_source_verified_future(csv_root,bin_root,evidence_root)
    report=_evaluate_sealed_rows(rows,seal,meta,external_source_verified=True,external_evidence=evidence)
    report['source_archive_gate']={'status':source_audit['status'],
                                   'archive_total_draws':source_audit['total_draws'],
                                   'last_draw_id':source_audit['last_draw_id'],
                                   'manifest_sha256':source_audit['source_fingerprints']['manifest_sha256'],
                                   'verified_new_draws':len(evidence)}
    if previous is not None:
        assert_append_only(json.loads(Path(previous).read_text('utf-8')),report)
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',type=Path,default=SEAL_PATH)
    parser.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    parser.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    parser.add_argument('--evidence-root',type=Path,default=Path('data/private/lotto649-source'))
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    p=args.out.resolve();root=Path('results').resolve()
    if not p.is_relative_to(root) or p.exists():
        parser.error('--out must be a NEW non-overwritten report path inside results/')
    result=run_pending_audit(seal_path=args.seal,csv_root=args.csv_root,
         bin_root=args.bin_root,evidence_root=args.evidence_root,previous=args.previous)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='ascii',newline='\n') as w:
        json.dump(result,w,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        w.write('\n')
    print('V23 STATUS',result['status'],'OBSERVED',result['observed_draw_count'],
          'PENDING',result['pending_draw_count'],'E',result['anytime_e_process']['approximate_e_value_for_display_only'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
