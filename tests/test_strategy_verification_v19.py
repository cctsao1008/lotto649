"""Adversarial, small synthetic audits of frozen-score saturation and bounds."""
import unittest
from lotto7x7.strategy_verification_v19 import (
    AXES, REFERENCE_COUNT, reference_axes_summary, saturation_counts,
    class_separable_envelope,
)


def five_class_pool(extra=False):
    rows=[]
    for i,shape_rank in enumerate((11,10,9,8,7)):
        row={'ticket':list(range(6*i+1,6*i+7)),
             'part':i,'rank':[10,10,10,shape_rank]}
        rows.append(row)
    if extra:
        rows.append({'ticket':[31,32,33,34,35,36],
                     'part':0,'rank':[20,20,20,11]})
    return rows


class V19Test(unittest.TestCase):
    def test_reference_resolution_distinguishes_reference_ties(self):
        base=list(range(REFERENCE_COUNT-2))+[REFERENCE_COUNT-3]*2
        report=reference_axes_summary([base]*4)
        self.assertEqual(len(report),4)
        self.assertEqual(report[0]['top_equal_gain_count'],3)
        self.assertEqual(report[0]['reference_max_equal_gain_first_rank'],REFERENCE_COUNT-3)
        self.assertTrue(report[0]['top_gain_is_tied'])

    def test_strict_above_max_saturates_empirical_rank(self):
        axes=[list(range(REFERENCE_COUNT)) for _ in AXES]
        rows=[{'rank':[REFERENCE_COUNT,0,0,0]}, {'rank':[REFERENCE_COUNT,0,0,0]}]
        gains=[[REFERENCE_COUNT+1.,0,0,0],[REFERENCE_COUNT+100.,0,0,0]]
        out=saturation_counts(rows,gains,axes)
        self.assertEqual(out[0]['strictly_above_reference_max'],2)
        self.assertEqual(out[0]['rank_at_empirical_ceiling'],2)
        self.assertEqual(out[0]['unique_beyond_reference_max_raw_gains'],2)
        self.assertEqual(out[0]['observed_rank_values_with_different_gains'],1)
        self.assertEqual(out[0]['candidates_in_gain_alias_rank_buckets'],2)

    def test_equal_to_max_not_the_same_as_strict_clipping(self):
        axes=[list(range(REFERENCE_COUNT)) for _ in AXES]
        rows=[{'rank':[REFERENCE_COUNT-1,0,0,0]}]
        gains=[[float(REFERENCE_COUNT-1),0.,0.,0.]]
        out=saturation_counts(rows,gains,axes)
        self.assertEqual(out[0]['strictly_above_reference_max'],0)
        self.assertEqual(out[0]['equal_to_reference_max'],1)
        self.assertEqual(out[0]['rank_at_empirical_ceiling'],0)

    def test_bad_rank_is_fail_closed(self):
        axes=[list(range(REFERENCE_COUNT)) for _ in AXES]
        with self.assertRaises(AssertionError):
            saturation_counts([{'rank':[REFERENCE_COUNT-1,0,0,0]}],[[float(REFERENCE_COUNT+1),0,0,0]],axes)

    def test_exact_finite_class_separable_certificate(self):
        pool=five_class_pool()
        witness=[p['ticket'] for p in pool[:4]]
        result=class_separable_envelope(pool,witness)
        self.assertEqual(result['status'],'EXACT_LEXICOGRAPHIC_CERTIFIED_BY_CLASS_CAPS')
        self.assertEqual(result['total_upper_minus_feasible'],0)
        self.assertEqual(result['primary_upper_bound'],38)
        self.assertTrue(result['full_lexicographic_exact_certified'])

    def test_upper_gap_is_not_falsely_certified(self):
        pool=five_class_pool(extra=True)
        witness=[p['ticket'] for p in pool[:4]]
        out=class_separable_envelope(pool,witness)
        self.assertEqual(out['status'],'VALID_UPPER_GAP_UNRESOLVED')
        self.assertGreater(out['total_upper_minus_feasible'],0)
        self.assertFalse(out['full_lexicographic_exact_certified'])

    def test_unattained_primary_does_not_claim_exact(self):
        pool=five_class_pool()
        witness=[pool[0]['ticket'],pool[1]['ticket'],pool[2]['ticket'],pool[4]['ticket']]
        out=class_separable_envelope(pool,witness)
        self.assertEqual(out['status'],'PRIMARY_BOUND_NOT_ATTAINED')

    def test_no_strict_top_four_bound_fail_closed(self):
        pool=five_class_pool()
        pool[-1]['rank'][3]=pool[-2]['rank'][3]
        witness=[p['ticket'] for p in pool[:4]]
        out=class_separable_envelope(pool,witness)
        self.assertEqual(out['status'],'NOT_CERTIFIABLE_CLASS_BOUND_TIE')


if __name__=='__main__':
    unittest.main()
