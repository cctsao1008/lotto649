"""v1.5 standalone C++ exhaustive search vs independent Python oracle."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from lotto7x7.strategy_verification_v15 import (
    compile_enumerator,enumerate_exact,input_stream,python_reference,SOURCE_CPP,
)


def pool_small():
    """Four Shape partitions; intentionally conflicting legal six-ball sets."""
    return [
        {'ticket':[1,2,3,4,5,6],'part':11,'rank':[13,14,12,11]},
        {'ticket':[1,2,3,4,5,9],'part':11,'rank':[15,13,16,11]},
        {'ticket':[7,8,9,10,11,12],'part':22,'rank':[12,16,10,9]},
        {'ticket':[7,8,9,10,11,13],'part':22,'rank':[14,15,12,9]},
        {'ticket':[13,14,15,16,17,18],'part':33,'rank':[12,14,17,8]},
        {'ticket':[19,20,21,22,23,24],'part':33,'rank':[16,14,16,8]},
        {'ticket':[25,26,27,28,29,30],'part':44,'rank':[14,16,13,7]},
        {'ticket':[31,32,33,34,35,36],'part':44,'rank':[15,17,15,7]},
    ]


class TestV15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.bin=Path(cls.tmp.name)/'v15_audit'
        compile_enumerator('g++',cls.bin)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _match(self,pool,bound):
        truth=python_reference(pool,primary_bound=bound)
        got=enumerate_exact(pool,primary_bound=bound,executable=self.bin)
        for k in ('best_total_rank','best_joint_rank','primary_feasible_portfolios',
                  'tied_best_lexicographic_portfolios','selected_candidate_ids'):
            self.assertEqual(got[k],truth[k],k)
        self.assertEqual(got['best_rank_sums'][3],bound)
        return got

    def test_01_small_pool_matches_exhaustive_python(self):
        self._match(pool_small(),35)

    def test_02_permutation_of_pool_order(self):
        data=pool_small();data=list(reversed(data))
        self._match(data,35)

    def test_03_lexicographic_tiebreak_joints(self):
        data=pool_small()
        for row in data:
            row['rank'][0]=50;row['rank'][1]=50
        self._match(data,35)

    def test_04_reject_malformed_shape_partition_bound(self):
        data=pool_small();data[2]['rank'][3]=8
        with self.assertRaises(ValueError):input_stream(data,primary_bound=35)
        with self.assertRaises(ValueError):input_stream(pool_small(),primary_bound=36)

    def test_05_reject_duplicate_tickets(self):
        data=pool_small();data[1]['ticket']=data[0]['ticket'][:]
        with self.assertRaises(ValueError):input_stream(data,primary_bound=35)

    def test_06_reject_invalid_balls(self):
        data=pool_small();data[0]['ticket']=[1,2,3,4,5,50]
        with self.assertRaises(ValueError):input_stream(data,primary_bound=35)

    def test_07_candidate_mask_overlap_is_a_real_constraint(self):
        got=self._match(pool_small(),35)
        self.assertLess(got['disjoint_four_portfolios'],got['cartesian_count'])
        self.assertGreater(got['primary_feasible_portfolios'],0)

    def test_08_correct_primary_threshold_preserved(self):
        got=self._match(pool_small(),35)
        self.assertEqual(sum(got['best_rank_sums']),got['best_total_rank'])


if __name__=='__main__':unittest.main()
