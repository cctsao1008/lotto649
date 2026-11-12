"""Small synthetic oracles and fail-closed gates for nested M0 candidate budgets."""
from __future__ import annotations

from itertools import combinations
import unittest

from lotto7x7.strategy_verification_v13 import validate_pool, is_feasible, quality
from lotto7x7.strategy_verification_v16 import (
    nested_pools, canonical_sha, portfolio_audit, local_coordinate_ascent,
    analyze_budget, bounded_mip_primary,
)


def small_pool():
    return [
        {'ticket': [1, 2, 3, 4, 5, 6], 'part': 10, 'rank': [10, 10, 10, 10]},
        {'ticket': [7, 8, 9, 10, 11, 12], 'part': 20, 'rank': [10, 10, 10, 10]},
        {'ticket': [13, 14, 15, 16, 17, 18], 'part': 30, 'rank': [10, 10, 10, 10]},
        {'ticket': [19, 20, 21, 22, 23, 24], 'part': 40, 'rank': [10, 10, 10, 10]},
        {'ticket': [25, 26, 27, 28, 29, 30], 'part': 10, 'rank': [20, 20, 20, 20]},
        {'ticket': [31, 32, 33, 34, 35, 36], 'part': 20, 'rank': [18, 18, 18, 18]},
    ]


class TestNestedScaling(unittest.TestCase):
    def setUp(self):
        self.all = small_pool()
        self.base = self.all[:4]
        self.original = [row['ticket'] for row in self.base]

    def test_01_nested_prefix_all_qualified_and_hashes(self):
        stages = nested_pools(self.base, self.all, (4, 5, 6), verify_frozen_sha=False)
        self.assertEqual([len(s['pool']) for s in stages], [4, 5, 6])
        self.assertEqual(stages[0]['pool'], self.base)
        self.assertEqual(set(tuple(p['ticket']) for p in stages[-1]['pool']),
                         set(tuple(p['ticket']) for p in self.all))
        for a, b in zip(stages, stages[1:]):
            self.assertEqual(a['pool'], b['pool'][:len(a['pool'])])
            self.assertEqual(a['pool_sha256'], canonical_sha(a['pool']))
        self.assertEqual(stages[-1]['pool_sha256'], canonical_sha(stages[-1]['pool']))

    def test_02_no_normalization_or_mutation_of_ranks(self):
        stages = nested_pools(self.base, self.all, (4, 6), verify_frozen_sha=False)
        original = {tuple(c['ticket']):c['rank'] for c in self.all}
        self.assertEqual({tuple(c['ticket']):c['rank'] for c in stages[-1]['pool']}, original)
        self.assertEqual(self.base[0]['rank'], [10,10,10,10])

    def test_03_source_rank_drift_fails_closed(self):
        damaged = [dict(x) for x in self.all]
        damaged[0] = {**damaged[0], 'rank':[11,10,10,10]}
        with self.assertRaises(ValueError):
            nested_pools(self.base, damaged, (4,6), verify_frozen_sha=False)
        with self.assertRaises(ValueError):
            nested_pools(self.base, self.all, (5,6), verify_frozen_sha=False)
        with self.assertRaises(ValueError):
            nested_pools(self.base, self.all, (4,6,5), verify_frozen_sha=False)

    def test_04_portfolio_legality_and_fixed_quality(self):
        x=portfolio_audit(self.base,self.original)
        self.assertEqual(x['quality'], [40,160,40,40])
        with self.assertRaises(ValueError):
            portfolio_audit(self.base,[self.original[0]]*4)
        with self.assertRaises(ValueError):
            portfolio_audit(self.base,[*self.original[:3], [25,26,27,28,29,30]])

    def test_05_local_exchange_improves_computable_objective(self):
        local=local_coordinate_ascent(self.all,self.original)
        self.assertEqual(local['quality'][0], 58)
        self.assertEqual(local['rank_sums'], [58]*4)
        self.assertTrue(local['one_swap_stationary_certified'])
        self.assertGreaterEqual(local['swaps'],2)

    def test_06_exact_small_pool_oracle_and_primary_bound(self):
        arm=analyze_budget(self.all,self.original,mip_seconds=0.,compute_exact_lp=False)
        self.assertEqual(arm['primary_shape_upper'],58)
        self.assertEqual(arm['best_feasible_witness']['min_rank'],58)
        self.assertEqual(arm['primary_absolute_gap_at_most'],0)
        self.assertEqual(arm['primary_optimality'],'EXACT_FINITE_POOL_PRIMARY_PASS')
        ranks,tickets,parts=validate_pool(self.all)
        best=None
        for ids in combinations(range(len(self.all)),4):
            if not is_feasible(list(ids),tickets,parts):continue
            q=quality(ranks[list(ids)].sum(axis=0).tolist(),'balanced')
            if best is None or q>best:best=q
        self.assertEqual(tuple(arm['best_feasible_witness']['quality']),best)

    def test_07_numerical_mip_witness_is_independently_legal(self):
        x=bounded_mip_primary(self.all,seconds=3.)
        self.assertTrue(x['feasible'])
        measured=portfolio_audit(self.all,x['witness'])
        self.assertEqual(x['maximin'],measured['min_rank'])
        self.assertEqual(x['maximin'],58)

    def test_08_bad_limits_and_damaged_pool_fail(self):
        with self.assertRaises(ValueError):
            local_coordinate_ascent(self.all,self.original,max_passes=0)
        with self.assertRaises(ValueError):
            bounded_mip_primary(self.all,seconds=0)
        bad=[*self.base, {**self.all[0]}]
        with self.assertRaises(ValueError):
            nested_pools(self.base,bad,(4,5),verify_frozen_sha=False)

    def test_09_reproducible_hash_and_local_selection(self):
        a=local_coordinate_ascent(self.all,self.original)
        b=local_coordinate_ascent(self.all,self.original)
        self.assertEqual(a,b)
        self.assertEqual(canonical_sha(a),canonical_sha(b))


if __name__=='__main__':
    unittest.main()
