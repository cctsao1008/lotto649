"""Independent BIN/score audit and multi-seed no-pseudo-replication contracts."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from math import comb
from pathlib import Path
import struct
import tempfile
from unittest import TestCase, mock

import numpy as np

from lotto7x7.full_history_v072 import NAMES, run_full_history, uniform_pool
from lotto7x7.full_history_audit_v073 import (
    AUDIT_SCHEMA, MULTISEED_SCHEMA, _disjoint_exact, _pair_p, audit_report,
    multiseed_report, read_verified_history,
)
from lotto7x7.full_history_audit_cli_v073 import configure_parsers, run


class FullHistoryIndependentAuditTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(2804)
        images = np.zeros((54,7,7), dtype=np.uint8)
        for image in images:
            image.flat[rng.choice(49,6,replace=False)] = 1
        ids = np.arange(115000001,115000055,dtype=np.int64)
        cls.images, cls.ids=images,ids
        image_bytes=bytearray(struct.pack('<8sII',b'L649IMG2',54,53))
        history_bytes=bytearray(struct.pack('<8sII',b'L649HIS1',54,15))
        for i,(image,draw_id) in enumerate(zip(images,ids)):
            main=tuple(int(n+1) for n in np.flatnonzero(image))
            bonus=next(n for n in range(1,50) if n not in main)
            datecode=int((date(2026,1,1)+timedelta(days=i)).strftime('%Y%m%d'))
            image_bytes.extend(struct.pack('<I49B',int(draw_id),*image.reshape(-1).tolist()))
            history_bytes.extend(struct.pack('<II7B',int(draw_id),datecode,*main,bonus))
        (cls.root/'lotto649_images_v2.bin').write_bytes(image_bytes)
        (cls.root/'lotto649_history_v1.bin').write_bytes(history_bytes)
        m={'dataset':'lotto649.single-source.biga.v1','source_site':'https://2019.biga.com.tw/',
           'total_draws':54,'first_draw_id':int(ids[0]),'last_draw_id':int(ids[-1]),
           'files':{name:{'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
                    for name,blob in [('lotto649_images_v2.bin',image_bytes),
                                      ('lotto649_history_v1.bin',history_bytes)]}}
        (cls.root/'lotto649_source_manifest.json').write_text(json.dumps(m),encoding='utf8')

        class LightweightCatalog:
            def __init__(self):self.last=None
            def add(self,image,draw_id):self.last=draw_id
            def make(self,*,size,families,seed):return uniform_pool(size,seed)

        with mock.patch('lotto7x7.full_history_v072.PrefixShapePool',LightweightCatalog):
            cls.reports=[run_full_history(images,ids,warmup=50,count=2,size=20,
                                          families=5,seed=seed,scenarios=512,device='cpu')
                         for seed in (649,650)]
        cls.hist = read_verified_history(cls.root/'lotto649_images_v2.bin',
                                         cls.root/'lotto649_source_manifest.json')
        cls.audits=[audit_report(rep,cls.hist,audit_seed=123) for rep in cls.reports]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_01_independent_wire_format_manifest_and_archive_hash(self):
        self.assertEqual(len(self.hist.ids),54)
        self.assertEqual(self.hist.ids[0],115000001)
        self.assertEqual(self.hist.numbers[0],tuple(int(x+1) for x in np.flatnonzero(self.images[0])))
        self.assertEqual(self.hist.archive_sha256,self.reports[0]['parameters']['archive_sha256'])
        self.assertEqual(len(self.hist.images_sha256),64)
        self.assertEqual(len(self.hist.history_sha256),64)

    def test_02_all_cpu_scores_match_report(self):
        for x in self.audits:
            self.assertEqual(x['schema'],AUDIT_SCHEMA)
            self.assertTrue(x['validated_full_history'])
            self.assertEqual(x['target_count'],4)
            self.assertEqual(x['independent_cpu_scored_ticket_sets'],24)
            for name in NAMES:
                self.assertEqual(x['summary'][name]['ge3_count'],
                                 self.reports[self.audits.index(x)]['summary'][name]['ge3_counts'])

    def test_03_independent_disjoint_math_and_conditional_mcnemar(self):
        ref=_disjoint_exact(5)
        self.assertEqual(ref['6'],5/comb(49,6))
        self.assertAlmostEqual(ref['3'],.09290168,places=6)
        self.assertAlmostEqual(_pair_p(9,5),.4239501953125,places=9)
        self.assertAlmostEqual(_pair_p(10,5),.3017578125,places=9)
        self.assertAlmostEqual(_pair_p(8,6),.79052734375,places=9)
        self.assertEqual(_pair_p(0,0),1)

    def test_04_broken_bin_pixels_or_manifest_rejected(self):
        path=self.root/'lotto649_images_v2.bin'
        src=path.read_bytes()
        bad=self.root/'bad.bin'
        wrong=bytearray(src)
        wrong[16+4+8]=4
        bad.write_bytes(wrong)
        with self.assertRaisesRegex(ValueError,'manifest SHA256'):
            read_verified_history(bad,self.root/'lotto649_source_manifest.json')
        old=json.loads((self.root/'lotto649_source_manifest.json').read_text())
        old['files']['lotto649_images_v2.bin']['sha256']='0'*64
        fake=self.root/'fake_manifest.json'
        fake.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError,'manifest SHA256'):
            read_verified_history(path,fake)

    def test_05_score_tamper_detected(self):
        changed=deepcopy(self.reports[0]);changed['targets'][0]['actual']=[1,2,3,4,5,6]
        if changed['targets'][0]['actual']==list(self.hist.numbers[50]):
            changed['targets'][0]['actual']=[7,8,9,10,11,12]
        with self.assertRaisesRegex(ValueError,'actual numbers'):
            audit_report(changed,self.hist)
        changed=deepcopy(self.reports[0]);changed['summary'][NAMES[0]]['ge3_counts']+=1
        with self.assertRaisesRegex(ValueError,'summary mismatch'):
            audit_report(changed,self.hist)

    def test_06_ticket_fair_and_cutoff_tamper_detected(self):
        changed=deepcopy(self.reports[0]);changed['targets'][0]['fair_coverage'][NAMES[0]]['3']+=.01
        with self.assertRaisesRegex(ValueError,'fair>=3'):
            audit_report(changed,self.hist)
        changed=deepcopy(self.reports[0]);changed['targets'][0]['cutoff_draw_id']+=1
        with self.assertRaisesRegex(ValueError,'cutoff mismatch'):
            audit_report(changed,self.hist)
        changed=deepcopy(self.reports[0]); changed['targets'][1]['bets'][NAMES[0]][0]=changed['targets'][1]['bets'][NAMES[0]][1]
        with self.assertRaisesRegex(ValueError,'duplicate ticket'):
            audit_report(changed,self.hist)
        changed=deepcopy(self.reports[0]);changed['parameters']['archive_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archive SHA256'):
            audit_report(changed,self.hist)

    def test_07_strict_full_history_versus_tail(self):
        short=deepcopy(self.reports[0])
        short['targets']=short['targets'][-2:]
        short['skipped_eligible_targets']=2
        short['evaluated_draws']=2
        # Aggregate summary is from four targets; recompute two with a real tail run.
        class LightweightCatalog:
            def __init__(self):pass
            def add(self,image,draw_id):pass
            def make(self,*,size,families,seed):return uniform_pool(size,seed)
        with mock.patch('lotto7x7.full_history_v072.PrefixShapePool',LightweightCatalog):
            tail=run_full_history(self.images,self.ids,warmup=50,tail_targets=2,
                                  count=2,size=20,families=5,seed=649,scenarios=512,device='cpu')
        with self.assertRaisesRegex(ValueError,'full historical'):
            audit_report(tail,self.hist)
        x=audit_report(tail,self.hist,allow_partial=True)
        self.assertEqual(x['target_count'],2)
        self.assertFalse(x['validated_full_history'])

    def test_08_multiseed_draw_level_only(self):
        m=multiseed_report(self.reports,self.audits,seed=649)
        self.assertEqual(m['schema'],MULTISEED_SCHEMA)
        self.assertEqual(m['draw_targets'],4)
        self.assertEqual(m['seeds'],[649,650])
        self.assertEqual(m['draw_clustered_s1_minus_b1']['random']['independent_draw_targets'],4)
        self.assertEqual(m['draw_clustered_s1_minus_b1']['random']['seed_replicates_per_target'],2)
        self.assertEqual(len(m['seed_paired_holm_pvalues']),4)
        self.assertIn('2026',m['draw_clustered_s1_minus_b1']['random']['year'])
        self.assertEqual(len(m['draw_clustered_fair_calibration']),6)

    def test_09_multiseed_mismatch_or_duplicate_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique integer seeds'):
            multiseed_report([self.reports[0],self.reports[0]],
                             [self.audits[0],self.audits[0]])
        x=deepcopy(self.reports[1]);x['targets'][0]['actual']=[1,2,3,4,5,6]
        changed_audit=deepcopy(self.audits[1])
        changed_audit['source_report_sha256']=hashlib.sha256(json.dumps(
            x,sort_keys=True,separators=(',',':'),ensure_ascii=True,
            allow_nan=False).encode('utf8')).hexdigest()
        with self.assertRaisesRegex(ValueError,'target numbers'):
            multiseed_report([self.reports[0],x],[self.audits[0],changed_audit])
        x=deepcopy(self.audits[1]);x['source_report_sha256']='00'*32
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            multiseed_report(self.reports,[self.audits[0],x])

    def test_10_cli_parses_expected_options(self):
        import argparse
        parser=argparse.ArgumentParser()
        sub=parser.add_subparsers(dest='command',required=True)
        configure_parsers(sub)
        x=parser.parse_args(['full-history-audit','--report','file.json','--out','audit.json'])
        self.assertEqual(x.report,Path('file.json'))
        self.assertFalse(x.allow_partial)
        x=parser.parse_args(['full-history-multiseed','--reports','a.json','b.json',
                             '--audits','aa.json','bb.json','--out','out.json'])
        self.assertEqual(len(x.reports),2)

    def test_11_cli_smoke(self):
        import argparse
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            source=folder/'gpu.json';source.write_text(json.dumps(self.reports[0]))
            audit_path=folder/'audit.json'
            args=argparse.Namespace(command='full-history-audit',report=source,
                data=self.root/'lotto649_images_v2.bin',history_bin=None,
                manifest=self.root/'lotto649_source_manifest.json',
                audit_seed=649,allow_partial=False,out=audit_path)
            run(args)
            self.assertEqual(json.loads(audit_path.read_text())['target_count'],4)
            with self.assertRaises(FileExistsError):run(args)

if __name__=='__main__':
    import unittest
    unittest.main()
