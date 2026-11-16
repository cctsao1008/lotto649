"""Reboot-only data, time-order permutation, held-out replay tests."""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
import numpy as np
from lotto7x7.core import encode
from lotto7x7.research import (load_history, normalized_signatures, research,
                               split, temporal_audit, year_shuffle)
from lotto7x7.__main__ import gallery


def fixture(path, values):
    dtype=np.dtype([('date','<u4'),('draw_id','<u4'),('numbers','u1',(6,)),('special','u1')])
    x=np.zeros(len(values),dtype=dtype)
    for i, row in enumerate(values):
        x[i]['date']=int((date(2018,1,1)+timedelta(days=3*i)).strftime('%Y%m%d'))
        x[i]['draw_id']=107000001+i
        x[i]['numbers']=row
        x[i]['special']=next(n for n in range(1,50) if n not in row)
    np.save(path,x,allow_pickle=False)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'draws.npy'
        rng=np.random.default_rng(440)
        values=np.stack([rng.choice(np.arange(1,50),6,replace=False) for _ in range(80)])
        fixture(self.path,values)

    def test_source_loaded_without_modification(self):
        before=self.path.read_bytes()
        images, dates, ids=load_history(self.path)
        self.assertEqual(images.shape,(80,7,7))
        self.assertEqual(len(ids),80)
        self.assertEqual(before,self.path.read_bytes())

    def test_split_chronological(self):
        a,b,c=split(80,window=3)
        self.assertEqual((a[0],a[-1],b[0],b[-1],c[0],c[-1]),(3,47,48,63,64,79))

    def test_constant_history_does_not_create_false_time_signal(self):
        image=encode(np.arange(1,7))
        images=np.repeat(image[None],80,axis=0)
        years=np.full(80,20180101)
        z=normalized_signatures(images,train_end=48)
        np.testing.assert_array_equal(z,np.zeros_like(z))
        audit=temporal_audit(images,years,train_end=48,start=64,end=80,shuffles=9)
        self.assertAlmostEqual(audit['lag1_delta'],0.0)
        self.assertAlmostEqual(audit['two_sided_tail'],1.0)

    def test_year_shuffle_preserves_multiset(self):
        images, dates, _=load_history(self.path)
        years=dates//10000
        out=year_shuffle(images,years,start=10,end=65,rng=np.random.default_rng(4))
        np.testing.assert_array_equal(out[:10],images[:10])
        np.testing.assert_array_equal(out[65:],images[65:])
        for year in np.unique(years[10:65]):
            idx=np.flatnonzero((years==year)&(np.arange(80)>=10)&(np.arange(80)<65))
            self.assertEqual(sorted(x.tobytes() for x in out[idx]),sorted(x.tobytes() for x in images[idx]))

    def test_replay_methods_and_visual_report(self):
        report=research(self.path, window=3,baseline=2,shuffles=9)
        self.assertEqual(report['evaluated_draws'],16)
        self.assertEqual(len(report['rows']),16)
        self.assertIn(report['model_selected_on_validation_only'],report['evaluation'])
        for method,result in report['evaluation'].items():
            self.assertLessEqual(result['unique_patterns']['unique'],16)
            self.assertTrue(np.isfinite(result['adjusted_mean']))
        page=gallery(report,count=2)
        self.assertIn('<svg ',page)
        self.assertEqual(page.count('<article>'),2)

    def test_wrong_year_fails(self):
        with self.assertRaises(ValueError): research(self.path,year=2026,baseline=2,shuffles=9,window=3)


if __name__=='__main__': unittest.main()
