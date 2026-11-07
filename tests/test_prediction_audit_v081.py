"""Focused v0.8.1 score-only audit tests; no old test suites or retraining."""
import copy
from math import log
import unittest

import numpy as np

from lotto7x7.prediction_audit_v081 import audit, METHODS, SCHEMA


class ProbabilityAuditV081Tests(unittest.TestCase):
    def setUp(self):
        n=6
        self.rows=[]
        for t in range(n):
            actual=list(range(t+1,t+7))
            labels=np.zeros(49,dtype=float)
            labels[np.asarray(actual)-1]=1
            models={}
            for j,name in enumerate(METHODS):
                p=np.full(49,6/49)
                # Test different but valid candidate top6 and numerically consistent scores.
                if name=='image-cnn-gru':
                    p=np.full(49,6/49)
                    p[0]+=0.001; p[1]-=0.001
                top=list(range(j+1,j+7))
                score=float(np.mean((p-labels)**2))
                loss=float(-np.mean(labels*np.log(p)+(1-labels)*np.log1p(-p)))
                models[name]={'probabilities_1_to_49':p.tolist(),
                               'top6_numbers':top,
                               'top6_hits':len(set(actual)&set(top)),
                               'brier':score,'binary_log_loss':loss}
            self.rows.append({'target_draw_id':115000001+t,
                              'cutoff_draw_id':115000000+t,
                              'actual_main_numbers':actual,'predictions':models})
        summary={}
        for name in METHODS:
            p=[r['predictions'][name] for r in self.rows]
            summary[name]={'mean_brier':float(np.mean([x['brier'] for x in p])),
                           'mean_binary_log_loss':float(np.mean([x['binary_log_loss'] for x in p])),
                           'mean_top6_hits':float(np.mean([x['top6_hits'] for x in p])),
                           'top6_ge3_rate':float(np.mean([x['top6_hits']>=3 for x in p]))}
        self.report={'schema':'lotto7x7.probability_replay.v0.8',
                     'source_draws':56,'evaluated_draws':6,
                     'config':{'warmup':50,'tail_targets':None},
                     'targets':self.rows,'summary':summary}

    def test_valid_frozen_json(self):
        r=audit(self.report)
        self.assertEqual(r['schema'],SCHEMA)
        self.assertEqual(r['target_count'],6)
        self.assertEqual(set(r['comparisons']),set(METHODS[1:]))
        self.assertIn('year',r['comparisons']['image-cnn-gru'])

    def test_reject_bad_stored_hit(self):
        s=copy.deepcopy(self.report)
        s['targets'][1]['predictions']['image-cnn-gru']['top6_hits']=0
        with self.assertRaisesRegex(ValueError,'corrupt saved score'):
            audit(s)

    def test_reject_corrupt_probs_or_brier(self):
        s=copy.deepcopy(self.report)
        s['targets'][0]['predictions']['recent-ewma']['probabilities_1_to_49'][0]=0.0
        with self.assertRaisesRegex(ValueError,'invalid forecast'):
            audit(s)
        t=copy.deepcopy(self.report)
        t['targets'][0]['predictions']['recent-ewma']['brier']+=0.01
        with self.assertRaisesRegex(ValueError,'corrupt saved score'):
            audit(t)

    def test_reject_cutoff_gap(self):
        s=copy.deepcopy(self.report)
        s['targets'][2]['cutoff_draw_id']=115000000
        with self.assertRaisesRegex(ValueError,'nonchronological'):
            audit(s)

    def test_no_need_full_history_for_partial_smoke(self):
        s=copy.deepcopy(self.report)
        s['config']['tail_targets']=6
        with self.assertRaisesRegex(ValueError,'full chronological history required'):
            audit(s)
        self.assertEqual(audit(s,allow_partial=True)['target_count'],6)


if __name__=='__main__':
    unittest.main()
