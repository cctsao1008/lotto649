#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile lotto7x7/portfolio/log_product_p3c_exact_iid_v027.py
python -m pytest -q \
  tests/portfolio/test_distribution_repair_v027.py \
  tests/portfolio/test_distribution_repair_scoring_v027.py \
  tests/portfolio/test_distribution_repair_portfolio_v027.py \
  tests/portfolio/test_log_product_v027.py \
  tests/portfolio/test_log_product_dual_scoring_v027.py \
  tests/portfolio/test_log_product_dual_portfolio_v027.py \
  tests/portfolio/test_log_product_joint_p3_v027.py \
  tests/portfolio/test_log_product_p3b_pilot_v027.py \
  tests/portfolio/test_log_product_p3c_exact_iid_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_sum_p0.json --md-out /tmp/v027_sum_p0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_sum_p0.json --out /tmp/v027_sum_p1.json --md-out /tmp/v027_sum_p1.md
python -m lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_log_p0.json --md-out /tmp/v027_log_p0.md
python -m lotto7x7.log_product_dual_scoring_v027 --sum-p1 /tmp/v027_sum_p1.json --log-p0 /tmp/v027_log_p0.json --out /tmp/v027_dual_p1.json --md-out /tmp/v027_dual_p1.md

python -m lotto7x7.log_product_p3b_pilot_v027 \
  --sum-p1 /tmp/v027_sum_p1.json --log-p0 /tmp/v027_log_p0.json \
  --dual-p1 /tmp/v027_dual_p1.json --csv-dir data/csv --cutoff 2026-11-06 \
  --warmup 300 --stride 64 --replicates 24 \
  --seed 649 --per-cell 32 --restarts 12 --cell-limit 24 \
  --out /tmp/v027_p3b_pilot.json --md-out /tmp/v027_p3b_pilot.md

python -m lotto7x7.log_product_p3c_exact_iid_v027 \
  --p3b /tmp/v027_p3b_pilot.json \
  --out /tmp/v027_p3c_exact_iid.json --md-out /tmp/v027_p3c_exact_iid.md
python - <<'PY'
import json,math
from pathlib import Path
r=json.loads(Path('/tmp/v027_p3c_exact_iid.json').read_text())
assert r['schema']=='lotto7x7.log_product.v0.27.p3c_exact_iid'
assert r['source']['historical_draws']==2497
assert r['source']['last_draw_id']==115000103
assert r['source']['analysis_input_sha256']=='d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert r['historical']['fixed_checkpoints']==35
assert r['fair_reference']['total_outcomes']==math.comb(49,6)*43
assert abs(r['fair_reference']['any_official_award_probability']-0.1227314589619088)<1e-13
t=r['historical']['exact_binomial_tests']
assert (t['primary']['successes'],t['distinct_sums']['successes'],t['random']['successes'])==(6,2,5)
assert abs(t['primary']['upper_tail_probability']-0.25442857495591076)<1e-12
assert abs(r['two_policy_familywise_upper_tail']['adjusted_p_upper_bound']-0.5088571499118215)<1e-12
assert r['original_cached_pilot_reference']['pilot_MC_p']==0.52
assert 'PREDICTIVE_NO_GO_OR_INCONCLUSIVE' in r['decision']
assert len(r['input_p3b_report_sha256'])==64
print('[PASS] independently exact FAIR-IID Binomial(n,p), two-policy family-wise Bonferroni bound and complete archived checkpoint audit')
PY
if [[ -f '/tmp/v027_p3b_pilot.json' ]]; then cp -- '/tmp/v027_p3b_pilot.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p3b_pilot.md' ]]; then cp -- '/tmp/v027_p3b_pilot.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p3c_exact_iid.json' ]]; then cp -- '/tmp/v027_p3c_exact_iid.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p3c_exact_iid.md' ]]; then cp -- '/tmp/v027_p3c_exact_iid.md' "$LEGACY_ARTIFACT_DIR/"; fi
