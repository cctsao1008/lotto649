"""M0 v1.2 independent audit boundary and deterministic calculation checks."""
from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import unittest

import numpy as np

from lotto7x7.strategy_verification_v12 import (
    BLOCKS, JOINT_SHA, OLD_TICKETS, PREFIX_SHA, RAW_SAMPLE_SHA, SHAPE_SHA,
    _feasible, _quality, candidate_sample_hash, draw_sample,
    integer_gain_numerators, one_swap_improvements, prefix_sha256,
    replay_bounded_search, shape_gain_table, weighted_sd,
)


class TestM0VerifierV12(unittest.TestCase):
    def test_01_exact_gain_numerators_agree_with_direct_fraction(self):
        from fractions import Fraction
        observed=[3,0,2,1]
        fair=[8,5,6,3]
        gain,den=integer_gain_numerators(observed,fair)
        n=sum(observed);m=sum(fair)
        base=sum((Fraction(h,n)-Fraction(c,m))**2 for h,c in zip(observed,fair))
        for k in range(len(fair)):
            after=sum((Fraction(h+int(i==k),n+1)-Fraction(c,m))**2 for i,(h,c) in enumerate(zip(observed,fair)))
            self.assertEqual(Fraction(gain[k],den),base-after)

    def test_02_shape_regularized_gain_matches_direct_before_after(self):
        from math import comb
        # The verifier formula has fixed 6-of-49 theory normalization.
        fair={1:comb(49,6)//2,2:comb(49,6)-comb(49,6)//2}
        counts={1:8,2:12}; n=sum(counts.values()); alpha=4.
        scored=shape_gain_table(counts,fair,n,alpha=alpha)
        for key in fair:
            p={k:v/comb(49,6) for k,v in fair.items()}
            weight={k:p[k]+alpha/n for k in fair}
            initial=sum((counts.get(k,0)/n-p[k])**2/weight[k] for k in fair)
            evolved=sum(((counts.get(k,0)+int(k==key))/(n+1)-p[k])**2/weight[k] for k in fair)
            self.assertAlmostEqual(scored[key],initial-evolved,places=13)

    def test_03_weighted_sigma(self):
        self.assertAlmostEqual(weighted_sd([0.,2.],[1,1]),1.)
        self.assertAlmostEqual(weighted_sd([-1.,3.],[1,3]),3**0.5)
        with self.assertRaises(ValueError):weighted_sd([2.,2.],[1,3])

    def test_04_sample_determinism(self):
        a=draw_sample(seed=649,count=30)
        b=draw_sample(seed=649,count=30)
        self.assertEqual(a,b)
        self.assertEqual(len(a),30)
        self.assertEqual(len(set(a)),30)
        self.assertTrue(all(len(set(t))==6 and t==tuple(sorted(t)) for t in a))
        self.assertNotEqual(candidate_sample_hash(a),candidate_sample_hash(draw_sample(seed=650,count=30)))

    def test_05_prefix_hash_strict_order(self):
        ids=np.asarray([93000001,93000002],dtype=np.int64)
        days=np.asarray([20040102,20040103],dtype=np.int64)
        rows=np.asarray([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        x=prefix_sha256(ids,days,rows)
        self.assertEqual(x,sha256(b'[20040102,93000001,1,2,3,4,5,6]\n[20040103,93000002,2,3,4,5,6,7]\n').hexdigest())
        self.assertNotEqual(x,prefix_sha256(ids[::-1],days[::-1],rows[::-1]))

    def test_06_four_ticket_disjointness_and_partition(self):
        chosen=[{'ticket':(1,2,3,4,5,6),'mask':sum(1<<(i-1) for i in range(1,7)),'part':1,'rank':[1,1,1,1]}]
        conflict={'ticket':(6,7,8,9,10,11),'mask':sum(1<<(i-1) for i in range(6,12)),'part':2,'rank':[1,1,1,1]}
        duplicate_part={'ticket':(7,8,9,10,11,12),'mask':sum(1<<(i-1) for i in range(7,13)),'part':1,'rank':[1,1,1,1]}
        good={'ticket':(7,8,9,10,11,12),'mask':sum(1<<(i-1) for i in range(7,13)),'part':2,'rank':[1,1,1,1]}
        self.assertFalse(_feasible(chosen,conflict))
        self.assertFalse(_feasible(chosen,duplicate_part))
        self.assertTrue(_feasible(chosen,good))

    def test_07_bounded_heuristic_tiny_pool(self):
        pool=[]
        for k in range(6):
            lo=k*6+1
            row=tuple(range(lo,lo+6))
            pool.append({'ticket':row,'mask':sum(1<<(n-1) for n in row),'part':k,
                         'rank':[10*k,9*k,8*k,7*k]})
        selected,stats=replay_bounded_search(pool,'balanced',restarts=10)
        self.assertEqual(len(selected),4)
        self.assertEqual(len({n for p in selected for n in p['ticket']}),24)
        self.assertTrue(one_swap_improvements(pool,selected,'balanced')['one_swap_stationary_within_fixed_pool'])
        self.assertEqual(stats['maximin_rank_total'],min(stats['rank_vector_sum']))

    def test_08_archive_reference_predeclared(self):
        p=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json')
        obj=json.loads(p.read_text('utf-8'))
        self.assertEqual(obj['original_ci_json_sha256'],
                         '7528a541d0d983ee1e36a300ff14c5fcbfd0c06a46af4eb894ff00f83db23520')
        self.assertEqual(obj['original_git_manifest_blob_sha1'],
                         '21dff1f975773ca919e6af9e762e93f8ffc3475c')
        self.assertEqual(obj['original_report_theory']['joint_sha256'],JOINT_SHA)
        self.assertEqual(obj['original_report_theory']['shape_counts_sha256'],SHAPE_SHA)
        self.assertEqual(obj['original_report_search']['sample_sha256'],RAW_SAMPLE_SHA)
        self.assertEqual(obj['original_report_search']['candidate_passing_four_positive_objective_blocks'],13704)
        self.assertEqual(obj['original_report_source']['analysis_input_sha256'],PREFIX_SHA)
        self.assertEqual(len(obj['primary']['tickets']),4)
        self.assertEqual(obj['primary']['maximin_rank_total'],43549)

    def test_09_reference_mutation_invalidates_declared_hash(self):
        p=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json')
        obj=json.loads(p.read_text('utf-8'))
        original=obj['original_report_search']['sample_sha256']
        self.assertEqual(original,RAW_SAMPLE_SHA)
        mutated=original[:-1]+('0' if original[-1]!='0' else '1')
        self.assertNotEqual(mutated,RAW_SAMPLE_SHA)

    def test_10_no_prediction_claim(self):
        obj=json.loads(Path('tests/fixtures/v028_m0_original_ci_golden_v12.json').read_text())
        self.assertEqual(obj['primary']['status'],'HEURISTIC_UNCERTIFIED')
        self.assertEqual(len(set(x for row in obj['primary']['tickets'] for x in row)),24)


if __name__=='__main__':unittest.main()
