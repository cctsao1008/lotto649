#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile research/lotto7x7/portfolio/distribution_repair_v027.py research/lotto7x7/portfolio/distribution_repair_scoring_v027.py research/lotto7x7/portfolio/distribution_repair_portfolio_v027.py
python -m pytest -q research/tests/portfolio/test_distribution_repair_v027.py research/tests/portfolio/test_distribution_repair_scoring_v027.py research/tests/portfolio/test_distribution_repair_portfolio_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_p0.json --md-out /tmp/v027_p0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_p0.json --out /tmp/v027_p1.json --md-out /tmp/v027_p1.md
python -m lotto7x7.distribution_repair_portfolio_v027 --p1 /tmp/v027_p1.json --out /tmp/v027_p2.json --md-out /tmp/v027_p2.md --seed 649
python - <<'PY'
from math import comb, isclose
import json
from pathlib import Path
p2 = json.loads(Path('/tmp/v027_p2.json').read_text())
assert p2['source']['historical_draws'] == 2497
assert p2['source']['last_draw_id'] == 115000103
assert p2['source']['analysis_input_sha256'] == 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
for name in ('primary','distinct_sums_sensitivity'):
    x = p2['optimization'][name]
    assert x['certificate']['globally_optimal_proven']
    assert x['certificate']['status'] == 'GLOBAL_OPTIMUM_CERTIFIED'
    assert x['objective']['gap_in_exact_integer_units'] == 0
    assert x['unique_numbers'] == 24
    assert x['coverage']['exact_official_eight_tier_parity']
    assert x['coverage']['exact_main_ball_coverage_parity']
    assert isclose(x['coverage']['official_probability'], .1227314589619088, abs_tol=1e-16)
    assert x['coverage']['official_total_outcomes'] == comb(49, 6) * 43
assert p2['optimization']['primary']['sum_bins'] == [152] * 4
assert set(p2['optimization']['distinct_sums_sensitivity']['sum_bins']) == {152, 116, 153, 164}
assert p2['optimization']['diversity_tradeoff']['coverage_probability_change'] == 0
print('[PASS] P2 exact global certificates for both constraints; official eight-tier parity')
PY
if [[ -f '/tmp/v027_p0.json' ]]; then cp -- '/tmp/v027_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p0.md' ]]; then cp -- '/tmp/v027_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p1.json' ]]; then cp -- '/tmp/v027_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p1.md' ]]; then cp -- '/tmp/v027_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p2.json' ]]; then cp -- '/tmp/v027_p2.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p2.md' ]]; then cp -- '/tmp/v027_p2.md' "$LEGACY_ARTIFACT_DIR/"; fi
