#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile lotto7x7/portfolio/log_product_joint_p3_v027.py
python -m pytest -q \
  tests/portfolio/test_distribution_repair_v027.py \
  tests/portfolio/test_distribution_repair_scoring_v027.py \
  tests/portfolio/test_distribution_repair_portfolio_v027.py \
  tests/portfolio/test_log_product_v027.py \
  tests/portfolio/test_log_product_dual_scoring_v027.py \
  tests/portfolio/test_log_product_dual_portfolio_v027.py \
  tests/portfolio/test_log_product_joint_p3_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_sum_p0.json --md-out /tmp/v027_sum_p0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_sum_p0.json --out /tmp/v027_sum_p1.json --md-out /tmp/v027_sum_p1.md
python -m lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_log_p0.json --md-out /tmp/v027_log_p0.md
python -m lotto7x7.log_product_dual_scoring_v027 --sum-p1 /tmp/v027_sum_p1.json --log-p0 /tmp/v027_log_p0.json --out /tmp/v027_dual_p1.json --md-out /tmp/v027_dual_p1.md
python -m lotto7x7.log_product_joint_p3_v027 \
  --sum-p1 /tmp/v027_sum_p1.json \
  --log-p0 /tmp/v027_log_p0.json \
  --dual-p1 /tmp/v027_dual_p1.json \
  --csv-dir data/csv --cutoff 2026-11-06 --warmup 300 \
  --out /tmp/v027_joint_p3a.json --md-out /tmp/v027_joint_p3a.md

python - <<'PY'
import json,math
from pathlib import Path
r=json.loads(Path('/tmp/v027_joint_p3a.json').read_text())
assert r['schema'] == 'lotto7x7.log_product.v0.27.p3a'
assert r['source']['historical_draws'] == 2497
assert r['source']['last_draw_id'] == 115000103
assert r['source']['analysis_input_sha256'] == 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
j=r['joint']
assert sum(map(sum,j['exact_theory_counts'])) == math.comb(49,6)
assert sum(map(sum,j['historical_counts'])) == 2497
assert len(j['exact_theory_counts'])==259
assert all(len(row)==34 for row in j['exact_theory_counts'])
assert r['repairs']['marginal_dual_positive_combinations']==4_014_395
assert (r['repairs']['dual_and_joint_positive_combinations']+
        r['repairs']['dual_positive_but_joint_negative_combinations']+
        r['repairs']['dual_positive_with_joint_zero_gain_combinations'])==4_014_395
assert .85 < r['correlation']['theory_unbinned']['sum_log_pearson_correlation'] < .98
p=r['prequential']
assert p['warmup']==300
assert p['evaluated_draws']==2197
assert 0 < p['IID_expected_average_positive_rate'] < 1
assert 0 <= p['observed_positive_rate'] <= 1
assert len(p['records'])==2197
assert p['records'][0]['prior_history_draws']==300
assert 'PREDICTIVE_NO_GO_OR_INCONCLUSIVE' in r['decision']
print('[PASS] exact 6/49 joint theory, correlated repairs, leak-free conditional IID single-draw event audit')
PY
if [[ -f '/tmp/v027_sum_p0.json' ]]; then cp -- '/tmp/v027_sum_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p0.md' ]]; then cp -- '/tmp/v027_sum_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.json' ]]; then cp -- '/tmp/v027_sum_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.md' ]]; then cp -- '/tmp/v027_sum_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.json' ]]; then cp -- '/tmp/v027_log_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.md' ]]; then cp -- '/tmp/v027_log_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.json' ]]; then cp -- '/tmp/v027_dual_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.md' ]]; then cp -- '/tmp/v027_dual_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_joint_p3a.json' ]]; then cp -- '/tmp/v027_joint_p3a.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_joint_p3a.md' ]]; then cp -- '/tmp/v027_joint_p3a.md' "$LEGACY_ARTIFACT_DIR/"; fi
