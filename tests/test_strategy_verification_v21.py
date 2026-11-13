"""Independent tiny-universe and adversarial gates for v2.1 full source audit."""
from __future__ import annotations
from itertools import combinations
from math import comb
import unittest

import numpy as np

from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_theory_v016 import classification_keys
from lotto7x7.strategy_verification_v21 import (
    _validate_reference,score_batch,accumulate_envelopes,rank_quality,
    find_distinct_witness,summarize,scan_full_universe,UNIVERSE,
)


def fake_ref(rows):
    allrows=np.asarray(rows,dtype=np.uint8)
    categories=classification_keys(classify_numbers(allrows))
    gain={level:{int(v):1.0 for v in categories[level]} for level in ('component','partition','edge')}
    axes=[np.linspace(-1,2,13704).tolist() for _ in range(4)]
    return {
        'axes':[np.asarray(a,dtype=np.float64) for a in axes],
        'cuts':np.asarray([0,10**11,10**12],dtype=np.int64),
        'sum_num':np.ones(259,dtype=np.int64),
        'log_num':np.ones(2,dtype=np.int64),
        'joint_num':np.ones(518,dtype=np.int64),
        'sum_den':1.0,'log_den':1.0,'joint_den':1.0,
        'shape_tables':gain,'shape_sd':{x:1. for x in gain},
        'edge_cache':{},'joint_cols':2,
    }


class ExactFullUniverseV21Tests(unittest.TestCase):
    def test_01_separate_scoring_axes_valid_batch(self):
        lines=[(1,2,3,4,5,6),(7,8,9,10,11,12),(13,14,15,16,17,18)]
        ref=fake_ref(lines)
        out,ranks,parts,d=score_batch(np.asarray(lines,dtype=np.uint8),ref,disallow=set())
        self.assertEqual(d['qualified'],3)
        self.assertEqual(d['raw_count'],3)
        self.assertEqual(ranks.shape,(3,4))
        self.assertEqual(len(set(parts)),len(set(classification_keys(classify_numbers(out))['partition'])))
        self.assertTrue((ranks>=0).all())

    def test_02_baseline_single_immutable_exclusion_is_deterministic(self):
        lines=[(1,2,3,4,5,6),(7,8,9,10,11,12)]
        ref=fake_ref(lines)
        out,ranks,parts,d=score_batch(np.asarray(lines,dtype=np.uint8),ref,disallow={lines[0]})
        self.assertEqual(len(out),1)
        self.assertEqual(tuple(out[0]),lines[1])
        self.assertEqual(d['excluded'],1)

    def test_03_malformed_numbers_fail_closed(self):
        ref=fake_ref([(1,2,3,4,5,6)])
        with self.assertRaisesRegex(ValueError,'sorted strict'):
            score_batch(np.asarray([[1,2,3,4,4,6]],dtype=np.uint8),ref)
        with self.assertRaisesRegex(ValueError,'sorted strict'):
            score_batch(np.asarray([[1,2,3,4,5,0]],dtype=np.uint8),ref)

    def test_04_update_envelope_across_chunks_and_keep_ties(self):
        groups={}
        a=np.asarray([[1,2,3,4,5,6],[7,8,9,10,11,12]],dtype=np.uint8)
        ranks=np.asarray([[1,1,1,10],[2,2,2,10]],dtype=np.int32)
        parts=np.asarray([7,7],dtype=np.int32)
        accumulate_envelopes(groups,a,ranks,parts)
        self.assertEqual(groups[7]['total_cap'],16)
        self.assertEqual(len(groups[7]['ties']),1)
        accumulate_envelopes(groups,np.asarray([[13,14,15,16,17,18]],dtype=np.uint8),
                             np.asarray([[2,2,2,10]],dtype=np.int32),np.asarray([7],dtype=np.int32))
        self.assertEqual(len(groups[7]['ties']),2)
        accumulate_envelopes(groups,np.asarray([[19,20,21,22,23,24]],dtype=np.uint8),
                             np.asarray([[1,1,1,11]],dtype=np.int32),np.asarray([7],dtype=np.int32))
        self.assertEqual(groups[7]['shape_cap'],11)
        self.assertEqual(len(groups[7]['ties']),1)

    def test_05_four_ticket_constraint_rejects_overlap_and_same_part(self):
        rows=[]
        for k in range(4):
            rows.append({'ticket':tuple(range(6*k+1,6*k+7)), 'part':k, 'rank':(10,10,10,10-k)})
        q,s=rank_quality(rows)
        self.assertEqual(q,(34,154,34,40))
        rows[1]['ticket']=rows[0]['ticket']
        with self.assertRaisesRegex(ValueError,'overlap'):
            rank_quality(rows)
        rows[1]['ticket']=tuple(range(7,13));rows[1]['part']=0
        with self.assertRaisesRegex(ValueError,'overlap'):
            rank_quality(rows)

    def test_06_disjoint_search_enforces_exact_primary_and_joint(self):
        def row(lo,part,joint):
            return {'ticket':tuple(range(lo,lo+6)), 'part':part, 'rank':(10,10,joint,10)}
        groups=[[row(1,1,10),row(2,1,20)], [row(7,2,10)], [row(13,3,10)], [row(19,4,10)]]
        candidate,nodes,finished=find_distinct_witness(groups,min_primary=40)
        self.assertIsNotNone(candidate)
        self.assertTrue(finished)
        self.assertGreater(nodes,0)
        with self.assertRaises(ValueError):
            find_distinct_witness(groups[:3],min_primary=40)
        exhausted,nodes,finished=find_distinct_witness(groups,min_primary=40,max_nodes=1)
        self.assertIsNone(exhausted)
        self.assertFalse(finished)

    def test_07_tiny_oracle_proves_finite_class_bound(self):
        g={}
        rows=[]
        for p,(lo,shape) in enumerate([(1,11),(7,10),(13,9),(19,8),(25,7)]):
            rows.append({'ticket':tuple(range(lo,lo+6)),'part':p,'rank':(20,20,20,shape)})
        for r in rows:
            a=np.asarray([r['ticket']],dtype=np.uint8)
            b=np.asarray([r['rank']],dtype=np.int32)
            c=np.asarray([r['part']],dtype=np.int32)
            accumulate_envelopes(g,a,b,c)
        report=summarize(g,rows[:4],UNIVERSE,100)
        self.assertEqual(report['decision'],'EXACT_FULL_UNIVERSE_LEXICOGRAPHIC_OPTIMUM')
        self.assertEqual(report['primary_upper'],38)
        self.assertEqual(report['total_coupled_upper_given_primary'],278)
        self.assertEqual(report['best_feasible_quality'],[38,278,38,80])
        self.assertEqual(report['fifth_class_shape_cap'],7)
        with self.assertRaisesRegex(AssertionError,'PARTIAL_UNIVERSE'):
            summarize(g,rows[:4],100,100)

    def test_08_tied_top_shape_is_not_certified(self):
        g={}
        rows=[]
        for p,(lo,shape) in enumerate([(1,11),(7,10),(13,9),(19,8),(25,8)]):
            r={'ticket':tuple(range(lo,lo+6)),'part':p,'rank':(20,20,20,shape)}
            rows.append(r)
            accumulate_envelopes(g,np.asarray([r['ticket']],dtype=np.uint8),
                                 np.asarray([r['rank']],dtype=np.int32),np.asarray([p],dtype=np.int32))
        out=summarize(g,rows[:4],UNIVERSE,50)
        self.assertEqual(out['decision'],'UNIVERSE_SHAPE_TOP_FOUR_NOT_STRICT')

    def test_09_test_only_partial_universe_cannot_be_certified(self):
        ref=fake_ref([(1,2,3,4,5,6),(7,8,9,10,11,12)])
        dummy=[{'ticket':(1,2,3,4,5,6),'part':1,'rank':(1,1,1,1)}]*4
        with self.assertRaisesRegex(ValueError,'TEST-ONLY'):
            scan_full_universe(ref,dummy,max_items=50)
        partial=scan_full_universe(ref,dummy,batch_size=1000,max_items=1,allow_partial=True)
        self.assertTrue(partial['test_only_partial'])
        self.assertEqual(partial['rows_scanned'],1)
        self.assertEqual(partial['certification'],'PROHIBITED_FOR_PARTIAL_ENUMERATION')

    def test_10_mathematical_universe_cardinality(self):
        self.assertEqual(UNIVERSE,13983816)
        self.assertEqual(UNIVERSE,comb(49,6))
        first=next(combinations(range(1,50),6))
        self.assertEqual(first,(1,2,3,4,5,6))


if __name__=='__main__':
    unittest.main()
