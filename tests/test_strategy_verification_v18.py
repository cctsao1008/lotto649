"""v1.8 independent exact-optimum and fail-closed reduction regression tests."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lotto7x7.strategy_verification_v15 import (
    compile_enumerator, enumerate_exact, python_reference,
)
from lotto7x7.strategy_verification_v18 import (
    reduce_primary_optimal_candidates, certify_pool, EXPECTED,
)


def toy_pool():
    return [
        {'ticket':[1,2,3,4,5,6],'part':100,'rank':[10,10,10,10]},
        {'ticket':[1,2,3,4,5,7],'part':100,'rank':[15,15,15,10]},
        {'ticket':[7,8,9,10,11,12],'part':200,'rank':[10,10,10,9]},
        {'ticket':[8,9,10,11,12,26],'part':200,'rank':[15,15,15,9]},
        {'ticket':[13,14,15,16,17,18],'part':300,'rank':[10,10,10,8]},
        {'ticket':[19,20,21,22,23,24],'part':400,'rank':[10,10,10,7]},
        {'ticket':[25,27,28,29,30,31],'part':500,'rank':[100,100,100,6]},
    ]


class TestV18(unittest.TestCase):
    def setUp(self):
        self.pool=toy_pool()
        self.witness=[self.pool[i]['ticket'] for i in (0,2,4,5)]

    def test_01_exact_lossless_reduction(self):
        reduced,proof=reduce_primary_optimal_candidates(self.pool,self.witness)
        self.assertEqual(len(reduced),6)
        self.assertEqual(proof['primary_upper'],34)
        self.assertEqual(proof['strict_fourth_vs_fifth_cap_gap'],1)
        self.assertEqual(sum(proof['per_partition_count'].values()),6)

    def test_02_exact_matches_independent_python_oracle(self):
        reduced,proof=reduce_primary_optimal_candidates(self.pool,self.witness)
        with TemporaryDirectory() as td:
            binary=Path(td)/'exact'
            compile_enumerator('g++',binary)
            exact=enumerate_exact(reduced,primary_bound=proof['primary_upper'],executable=binary)
        independent=python_reference(reduced,primary_bound=34)
        self.assertEqual(exact['best_total_rank'],independent['best_total_rank'])
        self.assertEqual(exact['best_joint_rank'],independent['best_joint_rank'])
        self.assertEqual(exact['primary_feasible_portfolios'],independent['primary_feasible_portfolios'])
        self.assertEqual(exact['tied_best_lexicographic_portfolios'],independent['tied_best_lexicographic_portfolios'])
        self.assertEqual(exact['best_total_rank'],184)
        self.assertEqual(exact['best_joint_rank'],50)

    def test_03_fails_on_tied_fourth_and_fifth_classes(self):
        pool=toy_pool()
        pool[-1]['rank'][-1]=7
        with self.assertRaisesRegex(ValueError,'CAP_TIE'):
            reduce_primary_optimal_candidates(pool,self.witness)

    def test_04_fails_on_nonprimary_witness(self):
        other=[self.pool[i]['ticket'] for i in (0,2,4,6)]
        with self.assertRaisesRegex(ValueError,'NO_PRIMARY_OPTIMAL_WITNESS'):
            reduce_primary_optimal_candidates(self.pool,other)

    def test_05_fails_on_corrupted_scores_or_ticket(self):
        pool=toy_pool()
        pool[1]['rank'][3]=11
        with self.assertRaises(ValueError):
            reduce_primary_optimal_candidates(pool,self.witness)
        pool=toy_pool()
        pool[0]['ticket']=[1,2,3,4,5,5]
        with self.assertRaises(ValueError):
            reduce_primary_optimal_candidates(pool,self.witness)

    def test_06_fails_on_modified_hash_before_search(self):
        with TemporaryDirectory() as td:
            bogus=Path(td)/'executable'
            with self.assertRaisesRegex(ValueError,'POOL_SHA_DRIFT'):
                certify_pool(self.pool,expected_sha='0'*64,executable=bogus)

    def test_07_case_fingerprints_unique_and_expected(self):
        self.assertEqual(len(EXPECTED),4)
        self.assertEqual(len(set((seed,n,policy) for seed,n,policy,_ in EXPECTED)),4)
        self.assertTrue(all(len(sha)==64 for _,_,_,sha in EXPECTED))


if __name__=='__main__':unittest.main()
