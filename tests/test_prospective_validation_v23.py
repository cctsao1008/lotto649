"""Deterministic, fail-closed tests for the v2.3 immutable future-only protocol.

All generated rows in tests are SYNTHETIC and must never be added to the
historical draw archive or misrepresented as actual prospective outcomes.
"""
from __future__ import annotations
from datetime import date,timedelta
from copy import deepcopy
from pathlib import Path
import unittest

from lotto7x7.prospective_validation_v23 import (
    SEAL_PATH,SEAL_BLOB_SHA1,EXPECTED_B,HORIZON,FIRST_ID,
    _git_blob_sha1,_canonical_seal_bytes,load_frozen_seal,validate_manifest,
    _control_seed_tickets,_check_contiguous_future,_evaluate_sealed_rows,
    exact_one_sided_discordance_p,assert_append_only,
)


def synthetic_row(issue:int,day:str,balls:list[int]|None=None,bonus:int=1)->dict:
    balls=sorted(balls if balls is not None else [2,10,20,30,40,48])
    if bonus in balls:bonus=49
    return {'draw_id':issue,'date':day,'main':balls,'bonus':bonus}


class V23FixedProspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seal,cls.meta=load_frozen_seal(SEAL_PATH)

    def test_01_sha_is_a_real_git_blob_sha_and_immutable(self):
        raw=Path(SEAL_PATH).read_bytes()
        canonical=_canonical_seal_bytes(raw)
        # Windows Git may check out an LF Git blob as CRLF in the working tree.
        # Verify the canonical committed bytes, never a platform-specific raw
        # working-tree representation. Do not rewrite the sealed manifest.
        self.assertIn(raw,(canonical,canonical.replace(b'\n',b'\r\n')))
        self.assertEqual(_git_blob_sha1(canonical),SEAL_BLOB_SHA1)
        with self.assertRaises(ValueError): _canonical_seal_bytes(canonical.replace(b'200',b'201',1))
        self.assertEqual(_canonical_seal_bytes(canonical.replace(b'\n',b'\r\n')),canonical)
        with self.assertRaises(ValueError): _canonical_seal_bytes(canonical.replace(b'\n',b'\r',1))

    def test_02_exact_v21_witness_and_rng_control_match_seal(self):
        self.assertEqual(self.meta['arms']['B'],EXPECTED_B)
        self.assertEqual(self.meta['arms']['A'],_control_seed_tickets(20261113))
        for key in ('A','B'):
            self.assertEqual(len({n for t in self.meta['arms'][key] for n in t}),24)
        self.assertEqual(self.meta['horizon'],200)
        self.assertEqual(self.meta['first_draw'],{'draw_id':115000106,'date':'2026-11-17'})
        self.assertTrue(self.meta['theoretical_full_official_prize_profile_equal'])
        self.assertAlmostEqual(self.meta['probability_any_official_prize_per_arm'],.1227314589619088)

    def test_03_protocol_tamper_fail_closed(self):
        x=deepcopy(self.seal);x['prospective_protocol']['horizon_consecutive_official_draws']=201
        with self.assertRaises(ValueError): validate_manifest(x)
        x=deepcopy(self.seal);x['selection']['history_blind_control']['seed']=12
        with self.assertRaises(ValueError):validate_manifest(x)
        x=deepcopy(self.seal);x['selection']['history_optimized_strategy']['tickets'][0][0]=8
        with self.assertRaises(ValueError):validate_manifest(x)

    def test_04_no_future_draw_is_pending_not_zero_performance(self):
        r=_evaluate_sealed_rows([],self.seal,self.meta,external_source_verified=True,external_evidence=[])
        self.assertEqual(r['observed_draw_count'],0)
        self.assertEqual(r['pending_draw_count'],HORIZON)
        self.assertIsNone(r['observed_award_rate_difference_B_minus_A'])
        self.assertIsNone(r['terminal_one_sided_discordance_p'])
        self.assertEqual(r['status'],'PENDING_NO_PREDICTIVE_GO')
        self.assertFalse(r['predictive_go'])
        self.assertEqual(r['anytime_e_process']['numerator'],'1')

    def test_05_future_date_schedule_and_id_are_strict(self):
        a=synthetic_row(FIRST_ID,'2026-11-17')
        b=synthetic_row(FIRST_ID+1,'2026-11-20')
        self.assertEqual(len(_check_contiguous_future([a,b])),2)
        for bad in ([b],[a,a],[a,synthetic_row(FIRST_ID+2,'2026-11-20')],
                    [a,synthetic_row(FIRST_ID+1,'2026-11-17')],
                    [a,synthetic_row(FIRST_ID+1,'2027-01-01')]):
            with self.assertRaises(ValueError): _check_contiguous_future(bad)

    def test_06_year_rollover_is_allowed_with_first_new_roc_id(self):
        a=synthetic_row(FIRST_ID,'2026-11-17')
        b=synthetic_row(116000001,'2027-01-01')
        # A year rollover after the first eligible draw is allowed; source
        # archive integrity is separately checked before the real CLI runs.
        self.assertEqual(len(_check_contiguous_future([a,b])),2)
        with self.assertRaises(ValueError):_check_contiguous_future([a,synthetic_row(116000002,'2027-01-01')])

    def test_07_exact_discordance_one_sided_p(self):
        self.assertEqual(exact_one_sided_discordance_p(0,0),1.)
        self.assertEqual(exact_one_sided_discordance_p(3,0),.125)
        self.assertEqual(exact_one_sided_discordance_p(0,3),1.)
        self.assertAlmostEqual(exact_one_sided_discordance_p(2,1),.5)
        with self.assertRaises(ValueError):exact_one_sided_discordance_p(-1,2)
        with self.assertRaises(ValueError):exact_one_sided_discordance_p(201,0)

    def test_08_b_only_and_a_only_betting_factors_are_exact(self):
        b=synthetic_row(FIRST_ID,'2026-11-17',EXPECTED_B[0],1)
        a=synthetic_row(FIRST_ID+1,'2026-11-20',self.meta['arms']['A'][0],1)
        r=_evaluate_sealed_rows([b,a],self.seal,self.meta)
        self.assertEqual(r['paired_counts']['B_only'],1)
        self.assertEqual(r['paired_counts']['A_only'],1)
        self.assertEqual(r['paired_counts']['both'],0)
        self.assertEqual(r['anytime_e_process']['numerator'],'91')
        self.assertEqual(r['anytime_e_process']['denominator'],'100')
        self.assertEqual(r['status'],'UNVERIFIED_INPUT_REJECT_FOR_SCIENTIFIC_REPORT')
        self.assertFalse(r['predictive_go'])

    def test_09_exact_threshold_and_no_early_go(self):
        day=date(2026,11,17);rows=[]
        for i in range(12):
            dt=day+timedelta(days=i)
            rows.append(synthetic_row(FIRST_ID+i,dt.isoformat(),EXPECTED_B[0],1))
        r=_evaluate_sealed_rows(rows,self.seal,self.meta,external_source_verified=True,
                                external_evidence=[{'proof':'SYNTHETIC-UNIT-TEST-ONLY'}]*len(rows))
        self.assertEqual(r['paired_counts']['B_only'],12)
        self.assertTrue(r['anytime_e_process']['ever_crossed_review_threshold'])
        self.assertEqual(r['status'],'PENDING_REVIEW_THRESHOLD_CROSSED_NO_GO')
        self.assertFalse(r['predictive_go'])
        self.assertIsNone(r['terminal_one_sided_discordance_p'])

    def test_10_append_only_previous_reports(self):
        a=synthetic_row(FIRST_ID,'2026-11-17')
        b=synthetic_row(FIRST_ID+1,'2026-11-20')
        p=_evaluate_sealed_rows([a],self.seal,self.meta)
        n=_evaluate_sealed_rows([a,b],self.seal,self.meta)
        assert_append_only(p,n)
        with self.assertRaises(ValueError):assert_append_only(n,p)
        modified=deepcopy(n);modified['observed_draws'][0]['main']=[1,2,3,4,5,6]
        with self.assertRaises(ValueError):assert_append_only(p,modified)

    def test_11_external_evidence_mandatory_for_verified_source(self):
        a=synthetic_row(FIRST_ID,'2026-11-17')
        with self.assertRaises(ValueError):
            _evaluate_sealed_rows([a],self.seal,self.meta,external_source_verified=True,external_evidence=[])
        with self.assertRaises(ValueError):
            _check_contiguous_future([{**a,'bonus':a['main'][0]}])

    def test_12_terminal_only_after_exactly_200_events(self):
        base=date(2026,11,17); rows=[]
        for i in range(HORIZON):
            day=base+timedelta(days=i)
            year=day.year-1911
            # For synthetic fixture, enumerate 2026+ with a reset at rollover
            if i==0:seq=106
            elif day.year!=(base+timedelta(days=i-1)).year:seq=1
            else:seq+=1
            rows.append(synthetic_row(year*1000000+seq,day.isoformat()))
        self.assertEqual(len(_check_contiguous_future(rows)),200)
        r=_evaluate_sealed_rows(rows,self.seal,self.meta,external_source_verified=True,
                                external_evidence=[{'proof':'SYNTHETIC-TEST-ONLY'}]*len(rows))
        self.assertEqual(r['observed_draw_count'],200)
        self.assertEqual(r['pending_draw_count'],0)
        self.assertIsNotNone(r['terminal_one_sided_discordance_p'])
        self.assertFalse(r['predictive_go'])
        with self.assertRaises(ValueError):_check_contiguous_future(rows+[rows[-1]])


if __name__=='__main__': unittest.main()
