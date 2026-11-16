#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v12.py' -v

python -m research.lotto7x7.strategy_verification_v12 --csv-root data/csv --bin-root data/bin --manifest research/experiments/v028_m0_four_block_candidates_20261110.json --golden research/tests/fixtures/v028_m0_original_ci_golden_v12.json --out results/strategy_v12_m0_full.json
python -c "import json; r=json.load(open('results/strategy_v12_m0_full.json')); assert r['decision']=='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED',r['failed_checks']; print('CONFIRMED:',len(r['checks']),'independent checks PASS')"
git diff --exit-code
if [[ -f 'results/strategy_v12_m0_full.json' ]]; then cp -- 'results/strategy_v12_m0_full.json' "$LEGACY_ARTIFACT_DIR/"; fi
