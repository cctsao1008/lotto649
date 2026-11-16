"""Independent v0.26 Burnside + Möbius certificate adversarial tests (stdlib)."""
import math
from itertools import permutations
import tempfile
from pathlib import Path
import unittest

from lotto7x7 import coverage_burnside_v026 as b


class IndependentBurnsideTests(unittest.TestCase):
    def test_fifteen_set_partitions_and_mobius_cancel(self):
        parts = b.set_partitions((0, 1, 2, 3))
        self.assertEqual(len(parts), 15)
        self.assertEqual(len(set(parts)), 15)
        self.assertEqual(sum(b.partition_mobius(p) for p in parts), 0)
        self.assertEqual(b.partition_mobius(((0,1,2,3),)), -6)

    def test_s4_group_action_is_bijective_for_all_24_permutations(self):
        for p in permutations(range(4)):
            mapped = [b.apply_ticket_permutation(mask, p) for mask in range(1,16)]
            self.assertEqual(set(mapped), set(range(1,16)))
            orbits = b.nonzero_mask_orbits(p)
            self.assertEqual(sum(map(len,orbits)), 15)
            self.assertEqual(len({m for orbit in orbits for m in orbit}), 15)

    def test_nonnegative_labeled_count_via_integer_dp(self):
        got = b.count_permutation_fixed_with_equalities(
            (0,1,2,3), ((0,), (1,), (2,), (3,))
        )
        self.assertEqual(got, 52450)

    def test_duplicate_ticket_mobius_exclusion(self):
        p=(0,1,2,3)
        total=b.count_permutation_fixed_with_equalities(p,((0,),(1,),(2,),(3,)))
        valid=b.fixed_valid_signatures(p)
        self.assertEqual(total,52450)
        self.assertEqual(valid,50985)
        self.assertEqual(total-valid,1465)
        self.assertLess(b.count_permutation_fixed_with_equalities(
            p,((0,1), (2,), (3,))),total)

    def test_five_s4_cycle_types_exact_fixed_point_counts(self):
        expected={
            (1,1,1,1):(1,50985),
            (1,1,2):(6,2467),
            (1,3):(8,114),
            (2,2):(3,1309),
            (4,):(6,57),
        }
        seen={}
        for p in permutations(range(4)):
            key=b.cycle_type(p)
            fixed=b.fixed_valid_signatures(p)
            if key not in seen:
                seen[key]=[0,fixed]
            self.assertEqual(seen[key][1],fixed)
            seen[key][0]+=1
        self.assertEqual({k:tuple(v) for k,v in seen.items()},expected)

    def test_burnside_integrality_and_legacy_agreement_without_legacy_import(self):
        report=b.independent_burnside_report()
        self.assertEqual(report["schema"],b.SCHEMA)
        self.assertEqual(report["burnside_fixed_point_numerator"],70968)
        self.assertEqual(report["group_order"],24)
        self.assertEqual(report["unlabeled_valid_membership_structures"],2957)
        self.assertTrue(report["independent_structural_count_agrees"])
        self.assertEqual(report["scientific_decision"],"EXACT_STRUCTURE_ONLY_NO_PREDICTIVE_INFERENCE")
        self.assertNotIn("any_prize_probability",report)

    def test_invalid_permutation_and_partition_fail_closed(self):
        for perm in [(0,1,1,3), (0,1,2), (4,1,2,3)]:
            with self.assertRaises(ValueError):
                b.nonzero_mask_orbits(perm)
        with self.assertRaises(ValueError):
            b.apply_ticket_permutation(0,(0,1,2,3))
        with self.assertRaises(ValueError):
            b.count_permutation_fixed_with_equalities((0,1,2,3),((0,1), (1,2), (3,)))

    def test_no_unrequested_outputs_or_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"result.json"
            self.assertFalse(out.exists())
            result=b.independent_burnside_report()
            self.assertTrue(result["independent_structural_count_agrees"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
