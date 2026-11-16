#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v18.py' -v

python -m research.lotto7x7.strategy_verification_v18 \
  --csv-root data/csv --bin-root data/bin --timeout 240 \
  --out results/strategy_v18_new_seed_exact.json
python - <<'PY'
import json
p='results/strategy_v18_new_seed_exact.json'
r=json.load(open(p))
assert r['decision']=='NEW_SEED_2000_POOL_EXACT_CERTIFICATES_COMPLETE'
assert r['source_gate']['v12_checks']==152
assert r['source_gate']['reference_rescoring']=='PASS_ALL_13704_FOUR_RANK_VECTORS'
assert len(r['cases'])==4
assert all(x['certification']=='EXACT_FINITE_2000_POOL_LEXICOGRAPHIC_CERTIFIED' for x in r['cases'])
assert all(x['lossless_reduction']['primary_upper']==43549 for x in r['cases'])
assert all(x['exhaustive_quality']>=x['greedy_quality'] for x in r['cases'])
assert all(x['greedy_was_optimal'] for x in r['cases'])
assert all(x['exhaustive_quality']==[43549,195617,43549,54184] for x in r['cases'])
assert [x['lossless_reduction']['primary_optimal_reduced_count'] for x in r['cases']]==[854,681,997,982]
assert [x['exhaustive_counts']['primary_feasible_portfolios'] for x in r['cases']]==[1712,1597,31577,30284]
assert [x['exhaustive_counts']['tied_best_lexicographic_portfolios'] for x in r['cases']]==[1,1,27,11]
assert [x['exhaustive_counts']['cartesian_count'] for x in r['cases']]==[338814900,218400000,869103900,698025984]
print('PASS four independent exact candidate-pool certificates')
PY
git diff --exit-code
if [[ -f 'results/strategy_v18_new_seed_exact.json' ]]; then cp -- 'results/strategy_v18_new_seed_exact.json' "$LEGACY_ARTIFACT_DIR/"; fi
