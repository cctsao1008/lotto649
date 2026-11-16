#!/usr/bin/env bash
set -euo pipefail

python -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2'

python -m unittest discover -s research/tests -p 'test_strategy_verification_v1.py' -v
python -m unittest discover -s research/tests -p 'test_strategy_verification_v11.py' -v

python -m lotto7x7.portfolio_benchmark_v030 --pool-size 128 --random-portfolios 32 --min-hybrid-unique 22 --iid-reps 3 --seed 649 --out results/strategy_v11_smoke_source.json
python -m lotto7x7.strategy_verification_v11 --selection-report results/strategy_v11_smoke_source.json --out results/strategy_v11_smoke_verified.json
python -c "import json; a=json.load(open('results/strategy_v11_smoke_verified.json')); assert a['decision']=='STRATEGY_V11_PASS_M0_OPTIMIZER_UNVERIFIED'; print('PASS smoke:',sum(a['prizes']['checks'].values()),'independent-prize checks')"

python -m lotto7x7.portfolio_benchmark_v030 --pool-size 512 --random-portfolios 256 --min-hybrid-unique 22 --iid-reps 19 --seed 649 --out results/strategy_v11_full_source.json
python -m lotto7x7.strategy_verification_v11 --selection-report results/strategy_v11_full_source.json --out results/strategy_v11_full_verified.json
python -c "import json; a=json.load(open('results/strategy_v11_full_verified.json')); assert a['decision']=='STRATEGY_V11_PASS_M0_OPTIMIZER_UNVERIFIED'; assert len(a['prizes']['checks'])==25; print('PASS full:',len(a['p2b_v1']['checks']),'P2B checks',len(a['prizes']['checks']),'prize checks',len(a['m0_manifest']['checks']),'M0 structural checks')"
git diff --exit-code
if [[ -f 'results/strategy_v11_smoke_source.json' ]]; then cp -- 'results/strategy_v11_smoke_source.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v11_smoke_verified.json' ]]; then cp -- 'results/strategy_v11_smoke_verified.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v11_full_source.json' ]]; then cp -- 'results/strategy_v11_full_source.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f 'results/strategy_v11_full_verified.json' ]]; then cp -- 'results/strategy_v11_full_verified.json' "$LEGACY_ARTIFACT_DIR/"; fi
