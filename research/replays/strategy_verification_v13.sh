#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v13.py' -v

python -m research.lotto7x7.strategy_verification_v13 --csv-root data/csv --bin-root data/bin --mip-seconds 15 --out results/strategy_v13_full.json
python -c "import json; r=json.load(open('results/strategy_v13_full.json')); assert r['decision']=='M0_STRATEGY_EXPLANATION_AND_GAP_AUDIT_PASS'; assert r['bounds']['upper']>=43549; assert all(v['one_swap_stationary'] for v in r['explanations'].values()); print('V13 PASS:',r['bounds']['original_primary_main_maximin_lower'],r['bounds']['upper'],r['primary_maximin_status'])"
git diff --exit-code
if [[ -f 'results/strategy_v13_full.json' ]]; then cp -- 'results/strategy_v13_full.json' "$LEGACY_ARTIFACT_DIR/"; fi
