#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile research/lotto7x7/portfolio/log_product_dual_portfolio_v027.py
python -m pytest -q \
  research/tests/portfolio/test_distribution_repair_v027.py \
  research/tests/portfolio/test_distribution_repair_scoring_v027.py \
  research/tests/portfolio/test_distribution_repair_portfolio_v027.py \
  research/tests/portfolio/test_log_product_v027.py \
  research/tests/portfolio/test_log_product_dual_scoring_v027.py \
  research/tests/portfolio/test_log_product_dual_portfolio_v027.py

python -m research.lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_sum_p0.json --md-out /tmp/v027_sum_p0.md
python -m research.lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_sum_p0.json --out /tmp/v027_sum_p1.json --md-out /tmp/v027_sum_p1.md
python -m research.lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_log_p0.json --md-out /tmp/v027_log_p0.md
python -m research.lotto7x7.log_product_dual_scoring_v027 --sum-p1 /tmp/v027_sum_p1.json --log-p0 /tmp/v027_log_p0.json --out /tmp/v027_dual_p1.json --md-out /tmp/v027_dual_p1.md
python -m research.lotto7x7.log_product_dual_portfolio_v027 \
  --sum-p1 /tmp/v027_sum_p1.json \
  --log-p0 /tmp/v027_log_p0.json \
  --dual-p1 /tmp/v027_dual_p1.json \
  --out /tmp/v027_dual_p2.json \
  --md-out /tmp/v027_dual_p2.md \
  --seed 649

python - <<'PY'
from pathlib import Path
import json
r=json.loads(Path('/tmp/v027_dual_p2.json').read_text())
assert r['schema'] == 'lotto7x7.log_product.v0.27.p2'
assert r['source']['historical_draws']==2497
assert r['source']['last_draw_id']==115000103
assert r['source']['analysis_input_sha256']=='d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert r['theory_checksums']['log']=='6bfaf80d57451c3e5eb4577691a27113dd4ee7b9a50fe8363460a0645e1ba826'
j=r['theoretical_joint_support']
assert j['full_universe']==13_983_816
assert j['dual_positive_ticket_count']==4_014_395
assert j['dual_positive_cells'] > 0
assert len(j['joint_counts_sha256']) == 64
p=r['portfolios']['primary']
assert p['tickets'], "no primary feasible disjoint dual-positive portfolio"
assert p['status'] in ('GLOBAL_OPTIMUM_CERTIFIED','HEURISTIC_UNCERTIFIED')
assert p['certified_global_optimum']==(p['objective_exact_integer']==p['upper_bound_exact_integer'])
assert p['unique_numbers']==24
assert len(p['tickets']) == 4
assert all(x['dual_positive'] for x in p['ticket_details'])
assert p['coverage']['exact_official_eight_tier_parity']
assert p['coverage']['exact_main_hit_coverage_parity']
assert abs(p['coverage']['any_prize_probability'] - .1227314589619088) < 1e-12
other=r['portfolios']['distinct_sum_sensitivity']
if other['tickets']:
    assert other['unique_numbers']==24
    assert other['unique_sum_bins']==4
    assert all(x['dual_positive'] for x in other['ticket_details'])
    assert other['coverage']['exact_official_eight_tier_parity']
    assert other['certified_global_optimum']==(other['objective_exact_integer']==other['upper_bound_exact_integer'])
    assert int(p['objective_exact_integer']) >= int(other['objective_exact_integer']), 'unrestricted optimum search must not underperform known diversity witness'
assert r['sum_only_p2_baseline_unchanged']['sum_bins']==[152]*4
assert r['objective']['no_joint_append_to_draw_history']
assert 'PREDICTIVE_ADVANTAGE_NOT_EVALUATED' in r['decision']
print('[PASS] exact dual-positive support; 4-disjoint valid tickets, 8-tier parity, certificate integrity')
PY
if [[ -f '/tmp/v027_sum_p0.json' ]]; then cp -- '/tmp/v027_sum_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p0.md' ]]; then cp -- '/tmp/v027_sum_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.json' ]]; then cp -- '/tmp/v027_sum_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.md' ]]; then cp -- '/tmp/v027_sum_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.json' ]]; then cp -- '/tmp/v027_log_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.md' ]]; then cp -- '/tmp/v027_log_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.json' ]]; then cp -- '/tmp/v027_dual_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.md' ]]; then cp -- '/tmp/v027_dual_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p2.json' ]]; then cp -- '/tmp/v027_dual_p2.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p2.md' ]]; then cp -- '/tmp/v027_dual_p2.md' "$LEGACY_ARTIFACT_DIR/"; fi
