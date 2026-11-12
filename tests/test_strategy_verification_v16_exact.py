"""Independent mathematical sandwich-scope checks for v1.6 full qualified pool."""
from __future__ import annotations
import unittest
from lotto7x7.strategy_verification_v16_exact import nested_sandwich_certificate

class TestExactQualifiedSandwich(unittest.TestCase):
    def test_01_equality_proves_every_intermediate_optimum(self):
        r=nested_sandwich_certificate((3,100,3,20),(3,100,3,20))
        self.assertTrue(r['all_nested_intermediate_optima_proven_equal'])

    def test_02_higher_full_optimum_does_not_prove_middle(self):
        r=nested_sandwich_certificate((3,100,3,20),(3,101,3,20))
        self.assertFalse(r['all_nested_intermediate_optima_proven_equal'])

    def test_03_lexicographic_order_not_simple_total(self):
        r=nested_sandwich_certificate((3,100,3,50),(4,99,3,10))
        self.assertFalse(r['all_nested_intermediate_optima_proven_equal'])

    def test_04_cannot_have_full_superset_worse(self):
        with self.assertRaises(AssertionError):
            nested_sandwich_certificate((4,99,3,10),(3,200,3,20))

    def test_05_bad_objective_components_fail(self):
        with self.assertRaises(ValueError):
            nested_sandwich_certificate((1,2,3),(1,2,3,4))
        with self.assertRaises(ValueError):
            nested_sandwich_certificate((1,2,3,True),(1,2,3,4))

if __name__ == '__main__':
    unittest.main()
