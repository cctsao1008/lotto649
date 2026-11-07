"""v0.8 GPU probability forecast contract, chronology, score and CPU fallback."""
from __future__ import annotations
import copy
import math
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import torch

from lotto7x7.prediction_v08 import (
    ProbabilityCNNGRU, METHODS, _baseline_curves, _top6, _validate,
    _windows, device_for, project_six, replay)


def fake_history(n=64, seed=921):
    rng = np.random.default_rng(seed)
    x=np.zeros((n,7,7), dtype=np.uint8)
    for row in x: row.flat[rng.choice(49,6,replace=False)] = 1
    return x, np.arange(93000001,93000001+n,dtype=np.int64)

class PredictionV08Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_projection_has_exact_six_marginal_mass(self):
        logits=torch.randn(10,49)*4
        p=project_six(logits)
        self.assertTrue(torch.allclose(p.sum(1),torch.full((10,),6.),atol=2e-4))
        self.assertTrue(bool(torch.all((p>=0)&(p<=1))))
        self.assertTrue(torch.equal(torch.argsort(p,dim=1),torch.argsort(logits,dim=1)))

    def test_projection_rejects_nan_and_shape(self):
        with self.assertRaises(ValueError):project_six(torch.zeros(2,7))
        a=torch.zeros(1,49);a[0,3]=float('nan')
        with self.assertRaises(ValueError):project_six(a)

    def test_prefix_baselines_use_no_future_outcome(self):
        x,_=fake_history()
        flat=x.reshape(len(x),49)
        a,b=_baseline_curves(flat)
        tampered=flat.copy()
        tampered[60]=np.roll(tampered[60],7)
        c,d=_baseline_curves(tampered)
        np.testing.assert_array_equal(a[:61],c[:61])
        np.testing.assert_array_equal(b[:61],d[:61])
        np.testing.assert_allclose(a.sum(1),6,atol=1e-10)
        np.testing.assert_allclose(b.sum(1),6,atol=1e-10)

    def test_window_target_excluded(self):
        x,_=fake_history()
        flat=x.reshape(len(x),49)
        v=_windows(flat,np.array([6,7,8]),5)
        np.testing.assert_array_equal(v[0].ravel(),flat[1:6].ravel())
        changed=flat.copy(); changed[6]=np.roll(changed[6],5)
        np.testing.assert_array_equal(_windows(changed,np.array([6]),5)[0],v[0])

    def test_cnn_head_is_49_logits(self):
        model=ProbabilityCNNGRU()
        self.assertEqual(tuple(model(torch.rand(3,5,7,7)).shape),(3,49))
        with self.assertRaises(ValueError):model(torch.rand(3,7,7))

    def test_device_modes(self):
        self.assertEqual(str(device_for('cpu')),'cpu')
        with mock.patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):device_for('cuda')
        with self.assertRaises(ValueError):device_for('metal')

    def test_invalid_image_or_ids(self):
        x,ids=fake_history()
        broken=x.copy();broken[8]=0
        with self.assertRaises(ValueError):_validate(broken,ids)
        swapped=ids.copy();swapped[3]=swapped[2]
        with self.assertRaises(ValueError):_validate(x,swapped)

    def test_full_replay_score_calibrated_and_next_forecast(self):
        x,ids=fake_history()
        r=replay(x,ids,warmup=60,block_size=2,window=4,validation=12,
                 epochs=2,patience=1,batch_size=16,seed=9,device='cpu')
        self.assertEqual(r['evaluated_draws'],4)
        self.assertEqual(len(r['training_blocks']),2)
        self.assertEqual(r['source_draws'],64)
        self.assertEqual(len(r['targets']),4)
        for record in r['targets']:
            self.assertLess(record['cutoff_draw_id'],record['target_draw_id'])
            for method in METHODS:
                row=record['predictions'][method]
                self.assertEqual(len(row['top6_numbers']),6)
                self.assertEqual(row['top6_hits'],len(set(row['top6_numbers'])&set(record['actual_main_numbers'])))
                self.assertAlmostEqual(sum(row['probabilities_1_to_49']),6,places=3)
        self.assertEqual(r['next_forecast']['after_last_known_draw_id'],int(ids[-1]))
        self.assertTrue(r['next_forecast']['actual_next_draw_not_available'])
        self.assertEqual(set(r['next_forecast']['models']),set(METHODS))
        self.assertAlmostEqual(r['iid_reference']['one_ticket_mean_hits'],36/49)

    def test_future_target_mutation_does_not_change_its_own_predictions(self):
        x,ids=fake_history(n=61)
        kwargs=dict(warmup=60,window=4,validation=12,epochs=1,patience=1,
                    seed=19,device='cpu')
        a=replay(x,ids,**kwargs)
        changed=x.copy();changed[-1]=0;changed[-1].flat[10:16]=1
        b=replay(changed,ids,**kwargs)
        self.assertNotEqual(a['targets'][0]['actual_main_numbers'],b['targets'][0]['actual_main_numbers'])
        for method in METHODS:
            self.assertEqual(a['targets'][0]['predictions'][method]['top6_numbers'],
                             b['targets'][0]['predictions'][method]['top6_numbers'])
            self.assertEqual(a['targets'][0]['predictions'][method]['probabilities_1_to_49'],
                             b['targets'][0]['predictions'][method]['probabilities_1_to_49'])

    def test_invalid_params(self):
        x,ids=fake_history()
        with self.assertRaises(ValueError):replay(x,ids,warmup=50,device='cpu')
        with self.assertRaises(ValueError):replay(x,ids,warmup=60,epochs=0,device='cpu')
        with self.assertRaises(ValueError):replay(x,ids,warmup=60,tail_targets=0,device='cpu')

    def test_iid_top6_reproducible(self):
        p=np.full(49,6/49)
        a=_top6(p,seed=649,t=200,method='iid-uniform')
        b=_top6(p,seed=649,t=200,method='iid-uniform')
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),6)
