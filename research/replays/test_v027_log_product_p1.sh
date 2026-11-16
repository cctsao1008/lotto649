#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile research/lotto7x7/portfolio/log_product_v027.py research/lotto7x7/portfolio/log_product_dual_scoring_v027.py
python -m pytest -q research/tests/portfolio/test_distribution_repair_v027.py research/tests/portfolio/test_distribution_repair_scoring_v027.py research/tests/portfolio/test_distribution_repair_portfolio_v027.py research/tests/portfolio/test_log_product_v027.py research/tests/portfolio/test_log_product_dual_scoring_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_sum_p0.json --md-out /tmp/v027_sum_p0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_sum_p0.json --out /tmp/v027_sum_p1.json --md-out /tmp/v027_sum_p1.md
python -m lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_log_p0.json --md-out /tmp/v027_log_p0.md
python -m lotto7x7.log_product_dual_scoring_v027 --sum-p1 /tmp/v027_sum_p1.json --log-p0 /tmp/v027_log_p0.json --out /tmp/v027_dual_p1.json --md-out /tmp/v027_dual_p1.md --ticket 04,09,17,24,29,42 --ticket 06,15,23,29,39,40
python - <<'PY'
import json, math
from pathlib import Path
d=json.loads(Path('/tmp/v027_dual_p1.json').read_text())
assert d['schema'] == 'lotto7x7.log_product.v0.27.p1'
assert d['source']['historical_draws'] == 2497
assert d['source']['last_draw_id'] == 115000103
assert d['source']['analysis_input_sha256'] == 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert d['log_theory']['checksum_sha256'] == '6bfaf80d57451c3e5eb4577691a27113dd4ee7b9a50fe8363460a0645e1ba826'
assert len(d['sum_bins']) == 259
assert len(d['log_bins']) == 34
assert d['qualifying']['dual_positive_ticket_count'] > 0
assert d['qualifying']['dual_positive_ticket_count'] < math.comb(49,6)
assert sum(d['qualifying']['per_sum_counts']) == d['qualifying']['dual_positive_ticket_count']
assert sum(d['qualifying']['per_log_bin_counts']) == d['qualifying']['dual_positive_ticket_count']
assert d['conditional_iid']['dual_positive_fair_probability'] == d['qualifying']['dual_positive_ticket_count']/math.comb(49,6)
assert d['qualifying']['feasibility'] == 'NONEMPTY'
assert len(d['ticket_examples']) == 2
assert 'PREDICTIVE_ADVANTAGE_NOT_EVALUATED' in d['decision']
print('[PASS] exact rational dual gain signs and exhaustive conditional IID qualification mass')
PY
if [[ -f '/tmp/v027_sum_p0.json' ]]; then cp -- '/tmp/v027_sum_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p0.md' ]]; then cp -- '/tmp/v027_sum_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.json' ]]; then cp -- '/tmp/v027_sum_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_sum_p1.md' ]]; then cp -- '/tmp/v027_sum_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.json' ]]; then cp -- '/tmp/v027_log_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_log_p0.md' ]]; then cp -- '/tmp/v027_log_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.json' ]]; then cp -- '/tmp/v027_dual_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_dual_p1.md' ]]; then cp -- '/tmp/v027_dual_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
