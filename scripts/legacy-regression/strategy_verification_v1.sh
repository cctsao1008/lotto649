#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s tests -p 'test_strategy_verification_v1.py' -v

python -m lotto7x7.portfolio_benchmark_v030 --pool-size 128 --random-portfolios 32 --min-hybrid-unique 22 --iid-reps 3 --seed 649 --out results/strategy_v1_source.json
python -m lotto7x7.strategy_verification_v1 --selection-report results/strategy_v1_source.json --out results/strategy_v1_verified.json
python -c "import json; x=json.load(open('results/strategy_v1_verified.json')); assert x['decision']=='STRATEGY_COMPLIANCE_PASS'; print('PASS: strategy audit',len(x['checks']), 'checks')"
git diff --exit-code

python -m lotto7x7.portfolio_benchmark_v030 --pool-size 512 --random-portfolios 256 --min-hybrid-unique 22 --iid-reps 19 --seed 649 --out results/strategy_v1_full_source.json
python -m lotto7x7.strategy_verification_v1 --selection-report results/strategy_v1_full_source.json --out results/strategy_v1_full_verified.json
python -c "import json; r=json.load(open('results/strategy_v1_full_verified.json')); assert r['decision']=='STRATEGY_COMPLIANCE_PASS'; assert len(r['checks'])==25; print('PASS: full P2B 512-candidate independent audit')"
git diff --exit-code
if [[ -f 'results/strategy_v1_source.json' ]]; then cp -- 'results/strategy_v1_source.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v1_verified.json' ]]; then cp -- 'results/strategy_v1_verified.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v1_full_source.json' ]]; then cp -- 'results/strategy_v1_full_source.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v1_full_verified.json' ]]; then cp -- 'results/strategy_v1_full_verified.json' "$LEGACY_ARTIFACT_DIR/"; fi
