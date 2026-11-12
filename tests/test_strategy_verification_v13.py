"""Exact small-universe checks of the v1.3 finite-pool dual certificate."""
from __future__ import annotations

import unittest
from itertools import combinations
import sys
from pathlib import Path

import numpy as np

# Tests run against the actual package when invoked from repository root.
from lotto7x7.strategy_verification_v13 import (validate_pool, constraints,
     exact_dual_upper_bound, optimization_certificate, explain_selection, is_feasible,
     quality, shape_partition_bound, CERT_SCALE)


def fixture():
    return [
        {'ticket':[1,2,3,4,5,6], 'part':100,'rank':[10,20,20,20]},
        {'ticket':[7,8,9,10,11,12], 'part':200,'rank':[20,10,20,20]},
        {'ticket':[13,14,15,16,17,18], 'part':300,'rank':[20,20,10,20]},
        {'ticket':[19,20,21,22,23,24], 'part':400,'rank':[20,20,20,10]},
        {'ticket':[1,7,13,19,25,26], 'part':500,'rank':[90,0,5,5]},
        {'ticket':[27,28,29,30,31,32], 'part':600,'rank':[7,7,7,7]},
    ]


class TestV13(unittest.TestCase):
    def test_lp_certificate_upper_bounds_every_feasible_subset(self):
        pool=fixture()
        ranks,tickets,parts=validate_pool(pool)
        feasible=[]
        for ids in combinations(range(len(pool)),4):
            if is_feasible(list(ids),tickets,parts):
                feasible.append(int(np.min(ranks[list(ids)].sum(axis=0))))
        self.assertTrue(feasible)
        r=optimization_certificate(pool)
        self.assertGreaterEqual(r['integer_upper_bound'],max(feasible))
        self.assertGreaterEqual(r['upper_bound_float'],max(feasible))
        self.assertEqual(r['certificate'],'EXACT_INTEGER_LAGRANGIAN_UPPER_BOUND_FOR_FINITE_POOL')
        self.assertGreaterEqual(r['nonzero_dual_multipliers'],1)

    def test_exact_certificate_arbitrary_multipliers_are_valid(self):
        pool=fixture();ranks,tickets,parts,shapes,_,_,_,zmax=constraints(pool)
        m=49+len(shapes)+4
        cert=exact_dual_upper_bound(ranks=ranks,tickets=tickets,parts=parts,
          shape_keys=shapes,multiplier_int=[0]*m,lambda_int=0,zmax=zmax)
        self.assertEqual(cert['integer_upper_bound'],zmax)
        with self.assertRaises(ValueError):
            exact_dual_upper_bound(ranks=ranks,tickets=tickets,parts=parts,
             shape_keys=shapes,multiplier_int=[-1]+[0]*(m-1),lambda_int=0,zmax=zmax)

    def test_untrusted_pool_rejected(self):
        pool=fixture();pool[-1]['ticket']=[1,2,3,4,5,6]
        with self.assertRaises(ValueError):validate_pool(pool)
        pool=fixture();pool[-1]['ticket']=[27,28,29,30,30,32]
        with self.assertRaises(ValueError):validate_pool(pool)
        pool=fixture();pool[-1]['rank']=[1,2,3,0.5]
        with self.assertRaises(ValueError):validate_pool(pool)

    def test_explanation_describes_same_rank_arithmetic(self):
        pool=fixture()
        portfolio=[p['ticket'] for p in pool[:4]]
        result=explain_selection(pool,portfolio)
        self.assertEqual(result['rank_vector_sum'],[70,70,70,70])
        self.assertEqual(result['bottleneck_blocks'],['sum','log','joint','shape'])
        self.assertEqual(len(result['each_ticket_and_best_alternative']),4)
        self.assertTrue(all(v['feasible_one_ticket_replacements']>0 for v in result['each_ticket_and_best_alternative']))
        self.assertTrue(result['one_swap_stationary'])
        suboptimal=[pool[i]['ticket'] for i in (0,1,2,5)]
        self.assertFalse(explain_selection(pool,suboptimal)['one_swap_stationary'])

    def test_invalid_selection_rejected(self):
        pool=fixture()
        with self.assertRaises(ValueError):
            explain_selection(pool,[pool[i]['ticket'] for i in (0,1,2,4)])
        with self.assertRaises(ValueError):
            explain_selection(pool,[pool[i]['ticket'] for i in (0,1,2,2)])

    def test_shape_partition_upper_bound_exact_small_pool(self):
        pool=fixture()
        ranks,tickets,parts=validate_pool(pool)
        bound=shape_partition_bound(pool)
        for ids in combinations(range(len(pool)),4):
            if is_feasible(list(ids),tickets,parts):
                self.assertLessEqual(int(min(ranks[list(ids)].sum(axis=0))),
                                     bound['exact_integer_upper_bound'])
        self.assertEqual(len(bound['top_four_partition_codes']),4)

    def test_milp_optional_candidate_feasible(self):
        pool=fixture()
        r=optimization_certificate(pool,test_milp_seconds=3.)
        info=r['mip_exploration']
        self.assertTrue(info.get('feasible'))
        self.assertLessEqual(info['main_maximin'],r['integer_upper_bound'])

if __name__=='__main__': unittest.main()
