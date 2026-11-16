"""Independent safety and null-mechanics tests for the frozen P1A association screen."""
import math
from pathlib import Path
import tempfile
import unittest

from lotto7x7 import iching_gua_qi_stats_v24_p1a as m


def fixtures(*, n_per_state=20, confounding=False, dependence=False):
    rows=[]
    for state in range(12):
        for k in range(n_per_state):
            year=2020 if (not confounding or state<6) else 2021
            weekday=(2,5)[k%2]
            if confounding:
                nums=(1,2,3,4,5,6) if year==2020 else (44,45,46,47,48,49)
            elif dependence:
                nums=(1,2,3,4,5,6) if state<6 else (44,45,46,47,48,49)
            else:
                # Many repeating legal combinations, but all tests are on
                # permutation validity rather than the assumption of fair data.
                base=(state*3+k*7)%44+1
                nums=tuple(range(base,base+6))
            rows.append({'state':state, 'year':year, 'weekday':weekday,
                        'sum':sum(nums),'odd':sum(n&1 for n in nums),
                        'shape':m.component_count(nums)})
    return rows


class TestGuaQiStatsP1A(unittest.TestCase):
    def test_exact_fair_sum_moment(self):
        exact=m.fair_moments()
        self.assertEqual(exact['sum']['mean'],150)
        self.assertAlmostEqual(exact['sum']['variance'],1075)

    def test_exact_hypergeometric_odd_moment(self):
        exact=m.fair_moments()
        self.assertAlmostEqual(exact['odd']['mean'],6*25/49)
        self.assertAlmostEqual(exact['odd']['variance'],6*(25/49)*(24/49)*(43/48))

    def test_shape_universe_enumeration(self):
        fair=m.fair_moments()['shape_L1_full_distribution']
        self.assertEqual(sum(fair['counts_by_number_of_components_1_to_6']),13983816)
        self.assertAlmostEqual(sum(fair['probabilities']),1.0)

    def test_shape_components_connected_row(self):
        self.assertEqual(m.component_count((1,2,3,4,5,6)),1)

    def test_shape_components_six_isolated(self):
        self.assertEqual(m.component_count((1,3,5,7,29,31)),6)

    def test_shape_components_three_pairs(self):
        self.assertEqual(m.component_count((1,2,15,16,29,30)),3)

    def test_shape_components_vertical(self):
        self.assertEqual(m.component_count((1,8,15,22,29,36)),1)

    def test_illegal_numbers_rejected(self):
        for v in [(1,1,2,3,4,5), (0,2,3,4,5,6), (1,2,3,4,5,50), (2,1,3,4,5,6), (1,2,3)]:
            with self.subTest(v=v),self.assertRaises(ValueError):
                m.component_count(tuple(v))

    def test_strata_membership_and_count(self):
        rows=fixtures()
        cnt, cats, vals, _, _=m.sufficient_statistics(rows)
        self.assertEqual(cnt,[20]*12)
        self.assertEqual(sum(map(sum,cats)),240)
        self.assertEqual(sum(v[0] for v in vals),sum(r['sum'] for r in rows))

    def test_reproducible_seed(self):
        rows=fixtures()
        one=m.audit(rows,{'synthetic':True},reps=99,seed=649)
        two=m.audit(rows,{'synthetic':True},reps=99,seed=649)
        self.assertEqual(one,two)

    def test_year_confounding_does_not_create_season_claim(self):
        rows=fixtures(confounding=True)
        report,_=m.audit(rows,{'synthetic':True},reps=99)
        for v in report['tests']['endpoints'].values():
            self.assertEqual(v['permutation_p'],1.0)

    def test_strong_seasonal_difference_detectable(self):
        rows=fixtures(dependence=True)
        report,_=m.audit(rows,{'synthetic':True},reps=399,seed=649)
        x=report['tests']['endpoints']['sum']
        self.assertLessEqual(x['permutation_p'],0.01)
        self.assertLess(x['bonferroni_four_families_p'],0.05)

    def test_no_seasonal_difference_is_no_evidence(self):
        rows=fixtures(confounding=True)
        report,_=m.audit(rows,{'synthetic':True},reps=99)
        self.assertEqual(report['decision'],'EXPLORATORY_ASSOCIATION_SCREEN_NO_EVIDENCE')

    def test_multiple_comparison_correction(self):
        rows=fixtures()
        report,_=m.audit(rows,{'synthetic':True},reps=99)
        for v in report['tests']['endpoints'].values():
            self.assertAlmostEqual(v['bonferroni_four_families_p'],min(1.,4*v['permutation_p']))

    def test_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'test.json'
            m.write_new(f,'one\n')
            with self.assertRaises(FileExistsError):
                m.write_new(f,'two\n')
            self.assertEqual(f.read_text(),'one\n')

    def test_sha_mismatch_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'fake.csv'
            f.write_text('date,draw_id\n2004-01-05,93000001\n')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                m.read_frozen_source(f)


if __name__=='__main__':
    unittest.main()