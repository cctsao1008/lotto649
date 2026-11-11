"""v0.29 P1 exact 6/49 region and pair laws, with deterministic MC controls."""
from math import comb
import unittest

from lotto7x7 import geometry_theory_v029 as theory
from lotto7x7 import geometry_v029 as geo


class TestGeometryTheoryV029P1(unittest.TestCase):
    def test_all_single_regions_exact_and_nonnegative(self):
        self.assertEqual(13983816, theory.UNIVERSE)
        for name in geo.REGIONS:
            counts = theory.occupancy_numerators(name)
            self.assertEqual(7, len(counts))
            self.assertEqual(theory.UNIVERSE, sum(counts), name)
            self.assertTrue(all(type(n) is int and n >= 0 for n in counts))
            expected_mean = 6 * len(geo.REGIONS[name]) / 49
            observed_mean = sum(k * counts[k] for k in range(7)) / theory.UNIVERSE
            self.assertAlmostEqual(expected_mean, observed_mean, places=12)

    def test_radius_three_hypergeometric_fixture(self):
        counts = theory.occupancy_numerators("disk_r3")
        self.assertEqual(29, len(geo.REGIONS["disk_r3"]))
        for k in range(7):
            self.assertEqual(comb(29, k) * comb(20, 6 - k), counts[k])
        self.assertAlmostEqual(3.5510204081632653, 6 * 29 / 49, places=12)
        self.assertAlmostEqual(0.5265236613525235,
                               sum(counts[4:]) / theory.UNIVERSE, places=12)

    def test_all_pair_joint_tables_have_exact_marginals(self):
        names = list(geo.REGIONS)
        for first in names:
            for second in names:
                m = theory.pair_joint_numerators(first, second)
                self.assertEqual(theory.UNIVERSE, sum(map(sum, m)))
                self.assertTrue(all(v >= 0 for row in m for v in row))
                self.assertEqual([sum(row) for row in m],
                                 theory.occupancy_numerators(first))
                self.assertEqual([sum(m[i][j] for i in range(7)) for j in range(7)],
                                 theory.occupancy_numerators(second))

    def test_covariance_matches_joint_distribution(self):
        for first, second in (
            ("disk_r3", "border"), ("disk_r2", "disk_r3"),
            ("disk_r1", "disk_r1"), ("ellipse_horizontal", "ellipse_vertical"),
        ):
            joint = theory.pair_joint_numerators(first, second)
            e_ab = sum(i * j * joint[i][j] for i in range(7) for j in range(7)) / theory.UNIVERSE
            e_a = 6 * len(geo.REGIONS[first]) / 49
            e_b = 6 * len(geo.REGIONS[second]) / 49
            self.assertAlmostEqual(e_ab - e_a * e_b,
                                   theory.pair_covariance(first, second), places=11)
            self.assertAlmostEqual(theory.pair_covariance(first, second),
                                   theory.pair_covariance(second, first), places=12)

    def test_mutually_exclusive_and_identical_regions(self):
        self.assertEqual(0, len(geo.REGION_SETS["disk_r1"] & geo.REGION_SETS["border"]))
        m = theory.pair_joint_numerators("disk_r1", "disk_r1")
        self.assertTrue(all(m[i][j] == 0 for i in range(7)
                            for j in range(7) if i != j))
        disjoint = theory.pair_joint_numerators("disk_r1", "border")
        self.assertTrue(all(disjoint[i][j] == 0 for i in range(7)
                            for j in range(7) if i + j > 6))

    def test_theory_schema_and_fixed_geometry_identity(self):
        report = theory.exact_reference()
        self.assertEqual(theory.SCHEMA, report["schema"])
        self.assertEqual(10, report["region_count"])
        self.assertEqual(45, report["pair_count"])
        self.assertEqual(13983816, report["universe_combinations"])
        self.assertEqual(geo.catalog()["definition_sha256"],
                         report["geometry_definition_sha256"])

    def test_seeded_monte_carlo_reproducibility(self):
        a = theory.monte_carlo_check(trials=1200, seed=649)
        b = theory.monte_carlo_check(trials=1200, seed=649)
        c = theory.monte_carlo_check(trials=1200, seed=650)
        self.assertEqual(a, b)
        self.assertNotEqual(a["region_counts_k0_to_k6"],
                            c["region_counts_k0_to_k6"])
        self.assertEqual(1200, sum(a["region_counts_k0_to_k6"]["disk_r3"]))
        self.assertEqual(1200, sum(map(sum, a["predeclared_pair"]["empirical_joint_counts"])))
        self.assertEqual(geo.catalog()["definition_sha256"],
                         a["geometry_definition_sha256"])

    def test_monte_carlo_cross_checks_exact_expected_values(self):
        sample = theory.monte_carlo_check(trials=20000, seed=649)
        estimated = sample["region_empirical_means"]["disk_r3"]
        self.assertLess(abs(estimated - 6 * 29 / 49), 0.05)
        p_ge4 = sum(sample["region_counts_k0_to_k6"]["disk_r3"][4:]) / 20000
        self.assertLess(abs(p_ge4 - 0.5265236613525235), 0.03)

    def test_invalid_options_fail_closed(self):
        with self.assertRaises(ValueError):
            theory.occupancy_numerators("not_a_region")
        with self.assertRaises(ValueError):
            theory.pair_joint_numerators("disk_r3", "not_a_region")
        with self.assertRaises(ValueError):
            theory.pair_covariance("not_a_region", "disk_r3")
        for trials in (0, -1, True, 1.1, 1000001):
            with self.assertRaises(ValueError):
                theory.monte_carlo_check(trials=trials)
        with self.assertRaises(ValueError):
            theory.monte_carlo_check(trials=1, seed=-1)
        with self.assertRaises(ValueError):
            theory.monte_carlo_check(trials=1, seed=True)


if __name__ == "__main__":
    unittest.main()
