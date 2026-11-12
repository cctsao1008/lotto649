"""Exact small-pool tests and adversarial examples for v1.4."""
from __future__ import annotations

from itertools import combinations
import unittest
import numpy as np
from lotto7x7.strategy_verification_v13 import validate_pool, is_feasible
from lotto7x7.strategy_verification_v14 import (
    exact_primary_optimal_reduction, model_matrix, exact_lp_dual_certificate,
    certified_mip_search, audit_lexicographic,
)


def toy_pool():
    return [
        {'ticket':[1,2,3,4,5,6], 'part':100,'rank':[10,10,10,10]},
        {'ticket':[1,2,3,4,5,7], 'part':100,'rank':[15,15,15,10]},
        {'ticket':[7,8,9,10,11,12], 'part':200,'rank':[10,10,10,9]},
        {'ticket':[8,9,10,11,12,26], 'part':200,'rank':[15,15,15,9]},
        {'ticket':[13,14,15,16,17,18], 'part':300,'rank':[10,10,10,8]},
        {'ticket':[19,20,21,22,23,24], 'part':400,'rank':[10,10,10,7]},
        {'ticket':[25,27,28,29,30,31], 'part':500,'rank':[100,100,100,6]},
    ]


class TestV14(unittest.TestCase):
    def setUp(self):
        self.pool=toy_pool()
        self.original=[self.pool[i]['ticket'] for i in (0,2,4,5)]
        self.reduced, self.red=exact_primary_optimal_reduction(
            self.pool,self.original,expected_bound=34,expected_total=154,expected_joint=40)

    def _enumerate(self):
        ranks,tickets,parts=validate_pool(self.reduced)
        out=[]
        for inds in combinations(range(len(self.reduced)),4):
            if not is_feasible(list(inds),tickets,parts):continue
            total=ranks[list(inds)].sum(axis=0)
            if total.min()>=34:
                out.append((int(total.sum()),int(total[2]),tuple(inds)))
        return out

    def test_01_shape_reduction_is_safe_and_exact(self):
        self.assertEqual(self.red['frozen_primary_maximin'],34)
        self.assertEqual(self.red['strict_partition_gap_to_fifth'],1)
        self.assertEqual(len(self.reduced),6)
        self.assertEqual(sum(self.red['top_partition_candidate_counts'].values()),6)

    def test_02_secondary_heuristic_counterexample_and_joint(self):
        results=self._enumerate()
        self.assertTrue(results)
        self.assertEqual(max(x[0] for x in results),184)
        self.assertGreater(max(x[0] for x in results),154)
        c=exact_lp_dual_certificate(self.reduced,score_kind='total',primary_bound=34)
        self.assertGreaterEqual(c['integer_upper_bound'],184)
        m=certified_mip_search(self.reduced,score_kind='total',primary_bound=34,seconds=3)
        self.assertTrue(m['feasible'])
        self.assertEqual(m['objective_integer'],184)
        self.assertTrue(m['solver_certifies_integer_optimum'])
        joint=certified_mip_search(self.reduced,score_kind='joint',primary_bound=34,total_floor=184,seconds=3)
        self.assertEqual(joint['objective_integer'],50)
        self.assertTrue(joint['solver_certifies_integer_optimum'])
        jbound=exact_lp_dual_certificate(self.reduced,score_kind='joint',primary_bound=34,total_floor=184)
        self.assertGreaterEqual(jbound['integer_upper_bound'],50)

    def test_03_full_lex_detects_secondary_suboptimality(self):
        r=audit_lexicographic(self.pool,self.original,primary_bound=34,mip_seconds=3,
                              expected_total=154,expected_joint=40)
        self.assertEqual(r['decision'],'ORIGINAL_PRIMARY_MAXIMIN_OPTIMAL_SECONDARY_SUBOPTIMAL')
        self.assertEqual(r['complete_lexicographic_optimum_quality'][0],34)
        self.assertEqual(r['complete_lexicographic_optimum_quality'][1],184)

    def test_04_dual_exact_upper_beyond_all_feasible_combinations(self):
        feasible=self._enumerate()
        for kind in ('total','joint'):
            threshold=184 if kind=='joint' else None
            c=exact_lp_dual_certificate(self.reduced,score_kind=kind,primary_bound=34,total_floor=threshold)
            eligible=[row[0] if kind=='total' else row[1] for row in feasible
                      if threshold is None or row[0]>=threshold]
            self.assertGreaterEqual(c['integer_upper_bound'],max(eligible))

    def test_05_fail_closed_on_tampered_originals(self):
        with self.assertRaises(ValueError):
            exact_primary_optimal_reduction(self.pool,self.original,expected_bound=35,
                                            expected_total=154,expected_joint=40)
        wrong=[self.pool[i]['ticket'] for i in (1,2,4,5)]
        with self.assertRaises(ValueError):
            exact_primary_optimal_reduction(self.pool,wrong,expected_bound=34,
                                            expected_total=154,expected_joint=40)

    def test_06_reject_invalid_model_or_time_limit(self):
        with self.assertRaises(ValueError):
            model_matrix(self.reduced,primary_bound=35)
        with self.assertRaises(ValueError):
            certified_mip_search(self.reduced,score_kind='unknown',primary_bound=34,seconds=5)
        with self.assertRaises(ValueError):
            certified_mip_search(self.reduced,score_kind='total',primary_bound=34,seconds=0)


if __name__=='__main__':unittest.main()
