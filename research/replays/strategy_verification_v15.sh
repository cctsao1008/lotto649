#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v15.py' -v

python -m research.lotto7x7.strategy_verification_v15 --out results/strategy_v15_m0_exhaustive.json
python -c "import json; x=json.load(open('results/strategy_v15_m0_exhaustive.json')); assert x['decision']=='EXACT_INTEGER_FINITE_POOL_LEXICOGRAPHIC_OPTIMUM_CONFIRMED'; assert x['exact_exhaustive_evidence']['best_total_rank']==192715; assert x['exact_exhaustive_evidence']['best_joint_rank']==52895; assert x['original_primary_secondary_suboptimal']; e=x['exact_exhaustive_evidence']; assert e['cartesian_count']==1014135000; assert e['disjoint_four_portfolios']==3853084; assert e['primary_feasible_portfolios']==2059; assert e['tied_best_lexicographic_portfolios']==1; print('PASS complete exact finite-pool certificate')"
git diff --exit-code
if [[ -f 'results/strategy_v15_m0_exhaustive.json' ]]; then cp -- 'results/strategy_v15_m0_exhaustive.json' "$LEGACY_ARTIFACT_DIR/"; fi
