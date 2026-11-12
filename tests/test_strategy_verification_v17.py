"""v1.7 independent-seed and candidate-screening anti-regression tests."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from hashlib import sha256
from math import fsum
from pathlib import Path
from random import Random
import unittest

import numpy as np

from lotto7x7.strategy_verification_v12 import draw_sample, LEVELS
from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_theory_v016 import classification_keys
from lotto7x7.strategy_verification_v13 import quality, validate_pool
from lotto7x7.strategy_verification_v17 import (
    sampled_unique, score_uniform_stream, screen_pool, greedy_multistart,
    audit_policies, verify_robustness, POLICIES, QUALIFIED_COUNT,
)


def toy():
    return [
        {'ticket':[1,2,3,4,5,6], 'rank':[5,8,7,7], 'part':100},
        {'ticket':[7,8,9,10,11,12], 'rank':[8,6,7,9], 'part':200},
        {'ticket':[13,14,15,16,17,18], 'rank':[6,9,8,8], 'part':300},
        {'ticket':[19,20,21,22,23,24], 'rank':[9,8,6,10], 'part':400},
        {'ticket':[25,26,27,28,29,30], 'rank':[3,4,5,7], 'part':100},
        {'ticket':[31,32,33,34,35,36], 'rank':[7,9,8,7], 'part':200},
        {'ticket':[37,38,39,40,41,42], 'rank':[8,8,6,8], 'part':300},
        {'ticket':[43,44,45,46,47,48], 'rank':[5,7,9,8], 'part':400},
    ]


def fake_ref(stream):
    cats=classification_keys(classify_numbers(np.asarray(stream,dtype=np.uint8)))
    tables={k:{int(x):2. for x in np.unique(cats[k])} for k in LEVELS}
    # Maps all legal 6-ball combinations onto a positive reference dummy
    # and provides 13,704-bin frozen ranks. No historic source is needed.
    return {'integer_product_thresholds':[0,10**12,10**15],
            'sum_gain_numerators':[2]*259,'log_gain_numerators':[2,2],
            'joint_gain_numerators':[2]*(259*2),
            'sum_denominator':1,'log_denominator':1,'joint_denominator':1,
            'sum_sd':1.,'log_sd':1.,'joint_sd':1.,
            'shape_gain_tables':tables,'shape_sd':{k:1. for k in LEVELS},
            'sorted_qualified_gains':[[0.]*QUALIFIED_COUNT for _ in range(4)]}


class TestV17(unittest.TestCase):
    def test_01_unique_sample_is_prefix_nested_and_repeatable(self):
        rows,stamps=sampled_unique(222,(25,50,100))
        old,again=sampled_unique(222,(25,50,100))
        self.assertEqual(rows,old)
        self.assertEqual(stamps,again)
        self.assertEqual(len(set(rows)),100)
        self.assertEqual(sorted(rows[:50]),draw_sample(seed=222,count=50))
        self.assertNotEqual(sampled_unique(223,(25,50,100))[1][-1],stamps[-1])

    def test_02_first_checkpoint_hash_matches_original_M0_definition(self):
        rows,stamps=sampled_unique(15,(25,))
        h=sha256()
        for t in sorted(rows): h.update(bytes(t))
        self.assertEqual(stamps[0]['sample_sha256'],h.hexdigest())

    def test_03_fixed_scores_do_not_refit_with_candidate_count(self):
        stream,_=sampled_unique(900,(12,20))
        ref=fake_ref(stream)
        a,_=score_uniform_stream(stream[:12],ref)
        b,_=score_uniform_stream(stream,ref)
        expected={tuple(x['ticket']):(x['part'],x['rank']) for x in b[:12]}
        actual={tuple(x['ticket']):(x['part'],x['rank']) for x in a}
        self.assertEqual(actual,expected)
        self.assertEqual(len(b),20)
        self.assertEqual(b[0]['rank'],[QUALIFIED_COUNT]*4)

    def test_04_score_reference_must_have_complete_frozen_axes(self):
        stream,_=sampled_unique(901,(10,))
        ref=fake_ref(stream)
        ref['sorted_qualified_gains'][0].pop()
        with self.assertRaises(ValueError): score_uniform_stream(stream,ref)

    def test_05_scoring_rejects_duplicate_raw_tickets(self):
        stream,_=sampled_unique(901,(10,))
        ref=fake_ref(stream)
        with self.assertRaises(ValueError):score_uniform_stream(stream+[stream[0]],ref)

    def test_06_screening_is_deterministic_but_policy_distinct(self):
        rows=[dict(x) for x in toy()]
        for p in POLICIES:
            a=screen_pool(rows,policy=p,cap=4,seed=19,sample_size=20)
            b=screen_pool(rows,policy=p,cap=4,seed=19,sample_size=20)
            self.assertEqual(a,b)
            validate_pool(a)
            self.assertEqual(len(a),4)
        with self.assertRaises(ValueError):screen_pool(rows,policy='nonsense',cap=4,seed=19,sample_size=20)

    def test_07_bounded_multistart_produces_legal_deterministic_witness(self):
        rows=toy()
        a=greedy_multistart(rows,restarts=5)
        b=greedy_multistart(rows,restarts=5)
        self.assertEqual(a['quality'],b['quality'])
        self.assertEqual(a['tickets'],b['tickets'])
        self.assertEqual(len({n for t in a['tickets'] for n in t}),24)
        self.assertEqual(a['quality'],list(quality(a['rank_sums'],'balanced')))
        self.assertTrue(a['one_swap_stationary_certified'])

    def test_08_candidate_budgets_validate_bad_parameters(self):
        with self.assertRaises(ValueError):sampled_unique(1,(20,10))
        with self.assertRaises(ValueError):sampled_unique(1,(20,20))
        with self.assertRaises(ValueError):sampled_unique(1,(320001,))
        with self.assertRaises(ValueError):greedy_multistart(toy(),restarts=0)
        with self.assertRaises(ValueError):screen_pool(toy(),policy='ranked',cap=9,seed=1,sample_size=20)

    def test_09_bad_test_seed_is_rejected_before_live_data_access(self):
        kwargs=dict(csv_root=Path('absent'),bin_root=Path('absent'),
                    manifest=Path('absent'),golden=Path('absent'))
        with self.assertRaises(ValueError): verify_robustness(**kwargs,seeds=(20261110,))
        with self.assertRaises(ValueError): verify_robustness(**kwargs,seeds=(1,1))
        with self.assertRaises(ValueError): verify_robustness(**kwargs,raw_sizes=(160000,80000))

    def test_10_rank_gain_is_not_lottery_probability(self):
        stream,_=sampled_unique(944,(10,))
        r=fake_ref(stream)
        outcome,info=score_uniform_stream(stream,r)
        self.assertEqual(info['qualified'],10)
        self.assertEqual(info['raw'],10)
        self.assertTrue(all(all(j==QUALIFIED_COUNT for j in x['rank']) for x in outcome))


if __name__=='__main__':unittest.main()
