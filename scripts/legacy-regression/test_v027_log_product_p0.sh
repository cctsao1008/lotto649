#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile lotto7x7/portfolio/log_product_v027.py
python -m pytest -q tests/portfolio/test_distribution_repair_v027.py tests/portfolio/test_distribution_repair_scoring_v027.py tests/portfolio/test_distribution_repair_portfolio_v027.py tests/portfolio/test_log_product_v027.py

python -m lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 \
  --expected-prefix-sha256 d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8 \
  --out /tmp/v027_log_p0.json --md-out /tmp/v027_log_p0.md
python - <<'PY'
from math import comb, isclose
from pathlib import Path
import json
report = json.loads(Path('/tmp/v027_log_p0.json').read_text())
theory = report['theory']
source = report['source']
assert report['schema'] == 'lotto7x7.log_product.v0.27.p0'
assert source['historical_draws'] == 2497
assert source['last_draw_id'] == 115000103
assert source['cutoff'] == '2026-11-06'
assert source['analysis_input_sha256'] == 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert theory['universe'] == comb(49, 6)
assert sum(theory['counts']) == comb(49, 6)
assert sum(report['observed']['counts']) == 2497
assert len(theory['counts']) == 34
assert theory['grid']['edge_start'] == '6.5'
assert theory['grid']['edge_end'] == '23.5'
assert theory['moments']['skewness'] < 0
assert theory['strict_interior_local_peaks'] == 1
assert len(theory['checksum_sha256']) == 64
assert isclose(theory['moments']['mean'],17.701927830164678,abs_tol=1e-10)
assert isclose(theory['moments']['skewness'],-0.46564263743495726,abs_tol=1e-10)
assert report['metric']['name'] == 'squared_l2_probability_v1'
assert 'PREDICTIVE_ADVANTAGE_NOT_EVALUATED' in report['decision']
print('[PASS] exact 13,983,816 Log Product theoretical counts / left skew / frozen historical deficit baseline')
PY
if [[ -f '/tmp/v027_log_p0.json' ]]; then cp -- '/tmp/v027_log_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.md' ]]; then cp -- '/tmp/v027_log_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
