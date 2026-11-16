#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v17.py' -v

python -m research.lotto7x7.strategy_verification_v17 \
  --csv-root data/csv --bin-root data/bin \
  --seeds 649 20261111 20261112 \
  --raw-sizes 80000 160000 320000 \
  --pool-cap 2000 --restarts 6 \
  --out results/strategy_v17_candidate_robustness.json
python -c "import json; r=json.load(open('results/strategy_v17_candidate_robustness.json')); assert r['source_gate']['v12_checks']==152; assert r['source_gate']['reference_rescoring_all_13704']=='PASS_EXACT_RANK_PARITY'; assert r['parameters']['seeds']==[649,20261111,20261112]; assert r['parameters']['raw_sample_sizes']==[80000,160000,320000]; assert len(r['samples'])==9 and len(r['arms'])==27; assert all(a['strategy_compliance'].startswith('PASS_') for a in r['arms']); assert all(a['primary_gap_at_most']>=0 for a in r['arms']); print('V17 SOURCE_GATED ROBUSTNESS PASS',len(r['arms']),'arms')"
python -c "import json; r=json.load(open('results/strategy_v17_candidate_robustness.json')); a=r['arms']; ranked=[x for x in a if x['policy']=='ranked']; balanced=[x for x in a if x['policy']=='shape_round_robin']; uniform=[x for x in a if x['policy']=='uniform_screen']; assert r['source_gate']['fixed_sorted_score_axes_sha256']=='18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b'; assert len(ranked)==len(balanced)==len(uniform)==9; assert all(x['best_feasible_quality'][0]==43549 and x['primary_gap_at_most']==0 for x in ranked+balanced); assert sum(x['best_feasible_quality'][1]>192715 for x in ranked)==8; assert sum(x['best_feasible_quality'][0]==43549 for x in uniform)==1; assert max(x['best_feasible_quality'][1] for x in ranked)==195617; assert max(x['best_feasible_quality'][3] for x in ranked)==54184; print('V17 REGRESSION-LOCKED POLICY FINDINGS PASS')"
git diff --exit-code
if [[ -f 'results/strategy_v17_candidate_robustness.json' ]]; then cp -- 'results/strategy_v17_candidate_robustness.json' "$LEGACY_ARTIFACT_DIR/"; fi
