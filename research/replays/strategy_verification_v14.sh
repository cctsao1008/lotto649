#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v14.py' -v

python -m research.lotto7x7.strategy_verification_v14 --csv-root data/csv --bin-root data/bin --mip-seconds 90 --out results/strategy_v14_m0_lexicographic.json
python -c "import json; r=json.load(open('results/strategy_v14_m0_lexicographic.json')); assert r['v12_checks']==152; assert r['historical_draws']==2497; assert r['audit']['reduction']['frozen_primary_maximin']==43549; assert r['decision']=='ORIGINAL_PRIMARY_MAXIMIN_OPTIMAL_SECONDARY_SUBOPTIMAL'; assert r['audit']['original_total_rank_sum']==189817; assert r['audit']['secondary_total_rank']['solver']['objective_integer']==192715; assert r['audit']['tertiary_joint_rank']['solver']['objective_integer']==52895; assert r['audit']['secondary_total_rank']['strict_improvement_found']; assert r['audit']['secondary_total_rank']['certificate_basis']=='MIP_ZERO_GAP_NUMERICAL_ONLY'; assert r['audit']['tertiary_joint_rank']['certificate_basis']=='MIP_ZERO_GAP_NUMERICAL_ONLY'; print('AUDITED:',r['decision'])"
git diff --exit-code
if [[ -f 'results/strategy_v14_m0_lexicographic.json' ]]; then cp -- 'results/strategy_v14_m0_lexicographic.json' "$LEGACY_ARTIFACT_DIR/"; fi
