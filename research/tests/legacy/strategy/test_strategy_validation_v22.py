"""v2.2 independent negative tests of no-leak and comparator decisions."""
from __future__ import annotations

from math import isclose
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from lotto7x7.strategy_validation_v22 import (
    analyze_pairwise_results,feature_scores,paired_score,prefix_score_model,
    validate_schedule,select_repair_portfolio,
)
from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_theory_v016 import classification_keys
from lotto7x7.strategy_verification_v12 import LEVELS


class V22Test(TestCase):
    def test_01_nonoverlap_schedule(self):
        schedule=validate_schedule(2497,1850,25,50,13)
        self.assertEqual(len(schedule),13)
        self.assertEqual(schedule[0],(1850,1875))
        self.assertEqual(schedule[-1],(2450,2475))
        self.assertTrue(all(a[1] <= b[0] for a,b in zip(schedule,schedule[1:])))

    def test_02_reject_overlapping_folds(self):
        with self.assertRaises(ValueError):validate_schedule(2000,1600,26,25,8)

    def test_03_reject_history_too_short(self):
        with self.assertRaises(ValueError):validate_schedule(2000,2000,25,25,1)

    def test_04_reject_partial_holdout(self):
        self.assertEqual(validate_schedule(1000,900,30,30,20),[(900,930),(930,960),(960,990)])

    def test_05_invalid_history_model_fails_closed(self):
        with self.assertRaises(ValueError):prefix_score_model(np.ones((257,6),dtype=np.uint8),np.ones((259,2),dtype=np.int64),{},[0,10,100])

    def test_06_features_fail_on_unsorted_or_duplicate(self):
        with self.assertRaises(ValueError):feature_scores(np.asarray([[1,2,2,4,5,6]],dtype=np.uint8),{})
        with self.assertRaises(ValueError):feature_scores(np.asarray([[6,5,4,3,2,1]],dtype=np.uint8),{})

    def test_07_feature_score_shapes_preserved(self):
        tickets=np.asarray([[1,2,3,4,5,6],[8,11,12,17,21,33]],dtype=np.uint8)
        cats=classification_keys(classify_numbers(tickets))
        tables={lv:{int(v):1. for v in set(map(int,cats[lv]))} for lv in LEVELS}
        model={'cuts':np.asarray([0,1000,100000000000],dtype=np.int64),
               'joint_cols':2,'gains':{'sum':np.ones(259),'log':np.ones(2),'joint':np.ones(518)},
               'positive':{'sum':np.ones(259,dtype=bool),'log':np.ones(2,dtype=bool),'joint':np.ones(518,dtype=bool)},
               'shape_tables':tables,'shape_sd':{lv:1. for lv in LEVELS},'shape_cache':{}}
        raw,parts,valid=feature_scores(tickets,model)
        self.assertEqual(raw.shape,(2,4))
        np.testing.assert_allclose(raw,np.ones((2,4)))
        self.assertEqual(parts.tolist(),cats['partition'].tolist())
        self.assertTrue(valid.all())

    def test_08_paired_fair_controls_tie_is_half(self):
        dummy=np.ones((4,4),dtype=np.float64)
        with patch('lotto7x7.strategy_validation_v22.feature_scores',return_value=(dummy,np.zeros(4),np.ones(4))):
            x=paired_score(np.arange(1,7),np.asarray([[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]),{})
        self.assertEqual(x,.5)

    def test_09_paired_exact_no_advantage(self):
        rows=[(True,False),(False,True),(True,True),(False,False)]
        r=analyze_pairwise_results(rows,[.5,.5,.5,.5])
        self.assertEqual(r['history_blind_only'],1)
        self.assertEqual(r['history_based_only'],1)
        self.assertEqual(r['history_based_minus_blind_award_rate'],0.)
        self.assertEqual(r['paired_exact_two_sided_p_exploratory'],1.)
        self.assertEqual(r['score_two_sided_azuma_hoeffding_p_bound'],1.)

    def test_10_paired_score_bounds_and_length(self):
        with self.assertRaises(ValueError):analyze_pairwise_results([(False,False)],[1.1])
        with self.assertRaises(ValueError):analyze_pairwise_results([(False,False)],[.4,.6])

    def test_11_forced_score_extreme_valid_bound(self):
        r=analyze_pairwise_results([(False,False)]*100,[1.]*100)
        self.assertLess(r['score_two_sided_azuma_hoeffding_p_bound'],1e-15)

    def test_12_fair_4_ticket_prize_coverage_from_engine(self):
        from lotto7x7.strategy_ab_v026 import disjoint_portfolio
        from lotto7x7.official_prize_v026 import official_prize_coverage
        a,b=disjoint_portfolio(1),disjoint_portfolio(649)
        p,q=official_prize_coverage(a),official_prize_coverage(b)
        for field in ('any_prize_outcomes','awarded_ticket_count_outcomes',
                      'any_ticket_by_tier','expected_tickets_by_tier'):
            self.assertEqual(p[field],q[field])
        self.assertTrue(isclose(p['any_prize_probability'],.1227314589619088,rel_tol=0,abs_tol=1e-15))


if __name__=='__main__':
    import unittest
    unittest.main()
