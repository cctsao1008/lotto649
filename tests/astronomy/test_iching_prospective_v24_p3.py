"""P3 preregistration / exact likelihood ratio / fail-closed synthetic tests."""
from __future__ import annotations

from datetime import date,datetime,timedelta
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest
from unittest.mock import patch

from lotto7x7 import iching_prospective_v24_p3 as p3


class P3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mass=p3.parity_counts()

    def test_parity_exact_full_universe_not_uniform(self):
        self.assertEqual(sum(self.mass),13983816)
        self.assertEqual(len(self.mass),64)
        self.assertEqual(min(self.mass),134596)
        self.assertEqual(max(self.mass),296010)
        self.assertNotEqual(self.mass[0],self.mass[63])

    def test_likelihood_ratio_normalized_all_states_both_directions(self):
        universe=sum(self.mass)
        for s in range(64):
            for anti in (False, True):
                z=p3.tilted_normalizer(s,self.mass,anti)
                self.assertGreater(z,0)
                # Exact conditional fair-IID E[L | given sky]=1.
                v=sum((Fraction(self.mass[y],universe)*p3.lr_step(s,y,self.mass,anti)
                       for y in range(64)),Fraction(0))
                self.assertEqual(v,Fraction(1))

    def test_one_primary_process_zero_starts_at_one(self):
        r=p3.primary_process([],self.mass)
        self.assertEqual((r['e_numerator'],r['e_denominator']),('1','1'))
        self.assertFalse(r['ever_crossed_review_threshold'])
        self.assertEqual(r['steps'],[])

    def test_e_process_exact_one_observation(self):
        s,y=31,26
        a=p3.lr_step(s,y,self.mass,False)
        b=p3.lr_step(s,y,self.mass,True)
        expected=(a+b)/2
        result=p3.primary_process([(s,y)],self.mass)
        self.assertEqual(Fraction(int(result['e_numerator']),int(result['e_denominator'])),expected)
        self.assertEqual(result['steps'][0]['hamming_distance_OUTCOME_ONLY'],(s^y).bit_count())

    def test_eprocess_two_steps_is_mixture_of_products_not_product_of_mixtures(self):
        cases=[(0,0),(31,7)]
        a=p3.lr_step(*cases[0],self.mass,False)*p3.lr_step(*cases[1],self.mass,False)
        b=p3.lr_step(*cases[0],self.mass,True)*p3.lr_step(*cases[1],self.mass,True)
        r=p3.primary_process(cases,self.mass)
        self.assertEqual(Fraction(int(r['e_numerator']),int(r['e_denominator'])),(a+b)/2)

    def test_invalid_outcome_codes(self):
        for bad in (-1,64,100):
            with self.assertRaises(ValueError):
                p3.lr_step(0,bad,self.mass,False)
        for bad in (-1,64,100,0.0):
            with self.assertRaises(ValueError):
                p3.tilted_normalizer(bad,self.mass,False)

    def test_positive_threshold_in_exact_rationals(self):
        r=p3.primary_process([(0,0)]*35,self.mass)
        self.assertTrue(r['ever_crossed_review_threshold'])
        self.assertGreaterEqual(int(r['e_numerator']),20*int(r['e_denominator']))

    def test_past_observation_cannot_redefine_future_horizon(self):
        self.assertEqual((p3.FIRST_ID,p3.FIRST_DATE,p3.HORIZON),
                         (115000106,'2026-11-17',100))
        self.assertEqual(p3.BASE_P2_SHA256,'a46f08269fc8021369f1a045116051e4f9591ba83801d5cdbe8c1a08873edf39')

    def make_files(self,dir_:Path):
        rows=[];day=p3.SKY_START
        while day<=p3.SKY_END:
            s,l=3,6
            rows.append({'date':day.isoformat(),
                         'reference_at_taipei':datetime(day.year,day.month,day.day,12,tzinfo=p3.p1.TZ).isoformat(timespec='minutes'),
                         'solar_sector_8':str(s),'lunar_sector_8':str(l),
                         'astronomical_hexagram_code':str(p3.p2.hexagram_from_sectors(s,l)['astronomical_hexagram_code']),
                         'mapping_version':p3.SKY_RULE})
            day+=timedelta(days=1)
        data=p3.encode_csv(rows)
        source={'base_events_sha256':p3.BASE_EVENTS_SHA256,'base_json_count':46,
                'mapping_version':p3.SKY_RULE,
                'date_span':[p3.SKY_START.isoformat(),p3.SKY_END.isoformat()]}
        seal=p3.make_seal(source,p3.digest(data))
        seal_path=dir_/'seal.json'; sky_path=dir_/'sky.csv'
        seal_path.write_text(json.dumps(seal),encoding='utf-8')
        sky_path.write_bytes(data)
        return seal_path,sky_path

    def test_synthetic_future_sky_seal_is_valid(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            obj,rows=p3.load_seal_and_sky(seal_path=seal,sky_path=sky)
            self.assertEqual(len(rows),(p3.SKY_END-p3.SKY_START).days+1)
            self.assertEqual(obj['exact_fair_universe'],13983816)
            self.assertEqual(rows['2026-11-17'],(p3.p2.gray3(3)<<3)|p3.p2.gray3(6))

    def test_freeze_refuses_replacing_a_precommitted_table_or_seal(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            day=p3.SKY_START
            rows=[]
            while day<=p3.SKY_END:
                rows.append({'date':day.isoformat(),
                  'reference_at_taipei':datetime(day.year,day.month,day.day,12,tzinfo=p3.p1.TZ).isoformat(timespec='minutes'),
                  'solar_sector_8':'0','lunar_sector_8':'0',
                  'astronomical_hexagram_code':'0',
                  'mapping_version':p3.SKY_RULE})
                day+=timedelta(days=1)
            source={'date_span':[p3.SKY_START.isoformat(),p3.SKY_END.isoformat()],
                    'mapping_version':p3.SKY_RULE,
                    'base_events_sha256':p3.BASE_EVENTS_SHA256,'base_json_count':46}
            with patch.object(p3,'build_sky_table',return_value=(rows,source)):
                p3.freeze(old_root=root,future_root=root,out_dir=root)
                original=(root/p3.SKY_FILENAME).read_bytes()
                with self.assertRaises(FileExistsError):
                    p3.freeze(old_root=root,future_root=root,out_dir=root)
                self.assertEqual((root/p3.SKY_FILENAME).read_bytes(),original)

    def test_sky_csv_tampering_rejected(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            sky.write_bytes(sky.read_bytes()+b'# wrong\n')
            with self.assertRaises(ValueError):
                p3.load_seal_and_sky(seal_path=seal,sky_path=sky)

    def test_precommitted_model_parameter_drift_rejected(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            obj=json.loads(seal.read_text('utf-8'))
            obj['model']['odds_ratio_per_matching_or_mismatching_bit']='2/1'
            seal.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                p3.load_seal_and_sky(seal_path=seal,sky_path=sky)

    def test_seal_missing_future_sky_model_rejected(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            obj=json.loads(seal.read_text('utf-8'))
            obj['horizon_consecutive_verified_draws']=200
            seal.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                p3.load_seal_and_sky(seal_path=seal,sky_path=sky)

    def test_initial_pending_without_any_authentic_outcomes(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            archive={'status':'PASS','total_draws':2498,'last_draw_id':115000104,
                     'source_fingerprints':{'manifest_sha256':'frozen-manifest'}}
            with patch.object(p3,'verified_future_100',return_value=([],archive,[])):
                r=p3.evaluate(seal_path=seal,sky_path=sky,csv_root=Path(''),
                              bin_root=Path(''),evidence_root=Path(''))
            self.assertEqual(r['status'],'PENDING_NO_PREDICTIVE_GO')
            self.assertEqual((r['observed_draw_count'],r['pending_draw_count']),(0,100))
            self.assertEqual((r['anytime_e_process']['e_numerator'],
                              r['anytime_e_process']['e_denominator']),('1','1'))
            self.assertFalse(r['predictive_go'])

    def test_positive_verified_future_not_prediction_approval(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            archive={'status':'PASS','total_draws':2500,'last_draw_id':115000106,
                     'source_fingerprints':{'manifest_sha256':'frozen-manifest'}}
            item={'date':'2026-11-17','draw_id':115000106,
                  'main':[1,2,3,4,5,6],'bonus':7}
            proof=[{'source_index_sha256':'verified','page_sha256':['a','b']}]
            with patch.object(p3,'verified_future_100',return_value=([item],archive,proof)):
                r=p3.evaluate(seal_path=seal,sky_path=sky,csv_root=Path(''),
                              bin_root=Path(''),evidence_root=Path(''))
            self.assertEqual(r['observed_draw_count'],1)
            self.assertFalse(r['predictive_go'])
            self.assertEqual(r['observed_draws'][0]['outcome_code_OUTCOME_ONLY'],p3.p2a.outcome_code((1,2,3,4,5,6)))

    def test_missing_target_date_rejected_instead_of_skipped(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            archive={'status':'PASS','total_draws':2600,'last_draw_id':116000100,
                     'source_fingerprints':{'manifest_sha256':'x'}}
            item={'date':'2028-01-15','draw_id':115000106,'main':[1,2,3,4,5,6],'bonus':7}
            with patch.object(p3,'verified_future_100',return_value=([item],archive,[{}])):
                with self.assertRaises(ValueError):
                    p3.evaluate(seal_path=seal,sky_path=sky,csv_root=Path(''),
                                bin_root=Path(''),evidence_root=Path(''))

    def test_change_of_prior_report_fails_closed(self):
        with TemporaryDirectory() as tmp:
            seal,sky=self.make_files(Path(tmp))
            previous=Path(tmp)/'prev.json'
            previous.write_text(json.dumps({'schema':p3.REPORT_SCHEMA,
                'seal_sha256':p3.digest(seal.read_bytes()),
                'sky_table_sha256':json.loads(seal.read_text())['sky_table_sha256'],
                'observed_draw_count':1,
                'observed_draws':[{'fabricated':'do not permit'}]}))
            archive={'status':'PASS','total_draws':2498,'last_draw_id':115000104,
                     'source_fingerprints':{'manifest_sha256':'frozen-manifest'}}
            with patch.object(p3,'verified_future_100',return_value=([],archive,[])):
                with self.assertRaises(ValueError):
                    p3.evaluate(seal_path=seal,sky_path=sky,csv_root=Path(''),
                                bin_root=Path(''),evidence_root=Path(''),previous_report=previous)


if __name__=='__main__':
    unittest.main()
