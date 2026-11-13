"""v2.0 exact integer upper bound positive and adversarial regressions."""
import itertools
import unittest

from lotto7x7.strategy_verification_v20 import (
    grouped_exact_primary_slice, coupled_class_bounds,
    max_disjoint_two_class_total, disjoint_pairing_bound, certify_full_source,
)


def toy(extra_kind='none'):
    # Four required Shape classes plus a strictly lower fifth class.
    rows=[]
    for i,s in enumerate([11,10,9,8,7]):
        rows.append({'ticket':list(range(6*i+1,6*i+7)),
                     'part':i, 'rank':[10,10,10,s]})
    # A deliberately different Sum gain but worse SAME-ticket total:
    # separate axis maxima cannot be simultaneously attained.
    if extra_kind=='compensated':
        rows.append({'ticket':[1,2,3,4,5,37], 'part':0, 'rank':[12,4,10,11]})
    if extra_kind=='improved':
        rows.append({'ticket':[1,2,3,4,5,37], 'part':0, 'rank':[20,10,10,11]})
    if extra_kind=='joint_tie':
        rows[0]['rank']=[11,10,9,11]
        rows.append({'ticket':[1,2,3,4,5,37], 'part':0, 'rank':[9,10,11,11]})
    return rows,[rows[k]['ticket'] for k in range(4)]


class TestV20(unittest.TestCase):
    def test_primary_shape_reduction_lossless(self):
        rows,chosen=toy('compensated')
        groups,proof=grouped_exact_primary_slice(rows,chosen)
        self.assertEqual(proof['primary_upper'],38)
        self.assertEqual(proof['group_sizes'],[2,1,1,1])
        self.assertEqual(proof['strict_4th_5th_gap'],1)
        self.assertEqual(sum(len(g) for g in groups),5)

    def test_coupled_ticket_bound_closes_axiswise_gap(self):
        rows,chosen=toy('compensated')
        groups,_=grouped_exact_primary_slice(rows,chosen)
        u=coupled_class_bounds(groups)
        self.assertEqual(u['exact_integer_total_upper'],158)
        self.assertEqual(u['per_class_caps'][0]['exact_max_coupled_total'],41)
        result=certify_full_source(rows,chosen,pairwise=True)
        self.assertEqual(result['decision'],'EXACT_FULL_LEXICOGRAPHIC_OPTIMUM_IN_SOURCE')
        self.assertEqual(result['integer_total_upper_gap'],0)
        self.assertTrue(result['full_lexicographic_exact'])
        self.assertIsNone(result['disjoint_pair_upper'])

    def test_unresolved_bound_remains_unresolved(self):
        rows,chosen=toy('improved')
        a=certify_full_source(rows,chosen,pairwise=False)
        self.assertGreater(a['integer_total_upper_gap'],0)
        self.assertEqual(a['decision'],'VALID_FULL_SOURCE_TOTAL_UPPER_GAP_UNRESOLVED')
        self.assertFalse(a['full_lexicographic_exact'])

    def test_pairwise_compatibility_is_stronger_than_class_caps(self):
        rows,chosen=toy('none')
        # High-total alternatives for the two classes overlap one another.
        rows.append({'ticket':[31,32,33,34,35,36], 'part':0, 'rank':[20,10,10,11]})
        rows.append({'ticket':[31,32,33,34,35,37], 'part':1, 'rank':[20,10,10,10]})
        a=certify_full_source(rows,chosen,pairwise=True)
        self.assertEqual(a['coupled_ticket_upper']['exact_integer_total_upper'],178)
        self.assertEqual(a['disjoint_pair_upper']['exact_integer_total_upper'],168)
        self.assertEqual(a['exact_total_upper'],168)
        self.assertGreaterEqual(a['exact_total_upper'],a['feasible_witness']['total_rank'])

    def test_pair_max_matches_independent_python_brute_force(self):
        rows,chosen=toy('none')
        g0=[rows[0],{'ticket':[31,32,33,34,35,36], 'part':0, 'rank':[20,10,10,11]}]
        g1=[rows[1],{'ticket':[31,32,33,34,35,37], 'part':1, 'rank':[20,10,10,10]}]
        brute=[]
        for a,b in itertools.product(g0,g1):
            if not set(a['ticket']) & set(b['ticket']):brute.append(sum(a['rank'])+sum(b['rank']))
        result=max_disjoint_two_class_total(g0,g1,chunk=1)
        self.assertEqual(result['exact_pair_max_total'],max(brute))
        self.assertEqual(result['compatible_pair_count'],len(brute))
        self.assertEqual(result['exact_pair_max_total'],91)

    def test_joint_tiebreak_not_mistaken_for_full_certificate(self):
        rows,chosen=toy('joint_tie')
        result=certify_full_source(rows,chosen,pairwise=True)
        self.assertEqual(result['integer_total_upper_gap'],0)
        self.assertEqual(result['decision'],'EXACT_TOTAL_BUT_JOINT_TIEBREAKER_UNRESOLVED')
        self.assertFalse(result['full_lexicographic_exact'])

    def test_invalid_primary_witness_fails_closed(self):
        rows,chosen=toy()
        chosen=chosen[:3]+[rows[-1]['ticket']]
        with self.assertRaisesRegex(ValueError,'PRIMARY_OPTIMUM'):
            certify_full_source(rows,chosen)

    def test_tied_fourth_and_fifth_fails_closed(self):
        rows,chosen=toy()
        rows[-1]['rank'][3]=8
        with self.assertRaisesRegex(ValueError,'UNIQUE_TOP_FOUR'):
            certify_full_source(rows,chosen)

    def test_invalid_duplicate_or_illegal_ticket_fails_closed(self):
        rows,chosen=toy()
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            certify_full_source(rows,chosen)
        rows,chosen=toy()
        rows[0]['ticket']=[1,2,3,4,5,5]
        with self.assertRaises(ValueError):
            certify_full_source(rows,chosen)

    def test_pairing_three_matchings_exact_small_oracle(self):
        rows,chosen=toy()
        groups,proof=grouped_exact_primary_slice(rows,chosen)
        u=disjoint_pairing_bound(groups,chunk=2)
        self.assertEqual(u['exact_integer_total_upper'],158)
        self.assertEqual(len(u['matchings']),3)


if __name__=='__main__':
    unittest.main()
