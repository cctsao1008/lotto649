#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v16_exact.py' -v

python -m lotto7x7.strategy_verification_v16_exact --timeout 360 --out results/strategy_v16_all_qualified_exact.json
python -c "import json; r=json.load(open('results/strategy_v16_all_qualified_exact.json')); assert r['source_gate']['original_v12_checks']==152; assert r['source_gate']['all_qualified']==13704; assert r['exact_full_qualified_enumeration']['candidates']>=893; assert r['decision']=='EXACT_FULL_QUALIFIED_OPTIMUM_MATCHES_ORIGINAL_2K'; assert r['nested_sandwich_proof']['all_nested_intermediate_optima_proven_equal']; e=r['exact_full_qualified_enumeration']; assert e['candidates']==3328; assert e['cartesian_count']==92677006400; assert e['disjoint_four_portfolios']==351603285; assert e['primary_feasible_portfolios']==2164; assert e['tied_best_lexicographic_portfolios']==1; assert e['best_total_rank']==192715; assert e['best_joint_rank']==52895; print('EXACT AUDIT:',r['decision'])"
git diff --exit-code
if [[ -f 'results/strategy_v16_all_qualified_exact.json' ]]; then cp -- 'results/strategy_v16_all_qualified_exact.json' "$LEGACY_ARTIFACT_DIR/"; fi
