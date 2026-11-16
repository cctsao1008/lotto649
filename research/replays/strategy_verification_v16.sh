#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v16.py' -v

python -m research.lotto7x7.strategy_verification_v16 --csv-root data/csv --bin-root data/bin --mip-seconds 6 --out results/strategy_v16_nested_pool_scaling.json
python -c "import json; r=json.load(open('results/strategy_v16_nested_pool_scaling.json')); assert r['decision']=='NESTED_POOL_SCALING_AUDIT_PASS'; assert r['source']['v12_checks']==152; assert [x['size'] for x in r['arms']]==[2000,4000,8000,13704]; assert r['arms'][0]['full_lexicographic_optimality']=='V15_EXACT_FINITE_POOL_PASS'; assert all(x['primary_rigorous_upper']>=x['best_feasible_witness']['min_rank'] for x in r['arms']); print('PASS nested candidate quality audit')"
git diff --exit-code
if [[ -f 'results/strategy_v16_nested_pool_scaling.json' ]]; then cp -- 'results/strategy_v16_nested_pool_scaling.json' "$LEGACY_ARTIFACT_DIR/"; fi
