#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile research/lotto7x7/portfolio/distribution_repair_v027.py research/lotto7x7/portfolio/distribution_repair_scoring_v027.py
python -m pytest -q tests/portfolio/test_distribution_repair_v027.py tests/portfolio/test_distribution_repair_scoring_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_p0.json --md-out /tmp/v027_p0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v027_p0.json --out /tmp/v027_p1.json --md-out /tmp/v027_p1.md --ticket 04,09,17,24,29,42
python - <<'PY'
import json, math
from pathlib import Path
r = json.loads(Path('/tmp/v027_p1.json').read_text())
assert r['source']['historical_draws'] == 2497
assert r['source']['last_draw_id'] == 115000103
assert r['source']['analysis_input_sha256'] == 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert math.isclose(r['baseline_loss'], 0.000382723193838, abs_tol=1e-15)
assert len(r['sums']) == 259
assert sum(x['theory_combinations'] for x in r['sums']) == math.comb(49,6)
assert math.isclose(sum(x['probability'] for x in r['iid_null']['law_by_exact_rank_integer']),1)
assert r['decision'] == 'REPAIR_DESCRIPTIVE_PASS; PREDICTIVE_ADVANTAGE_NOT_EVALUATED'
assert len(r['ticket_examples']) == 1
print('[PASS] real exact conditional IID repair law and frozen cutoff')
PY
if [[ -f '/tmp/v027_p0.json' ]]; then cp -- '/tmp/v027_p0.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p0.md' ]]; then cp -- '/tmp/v027_p0.md' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p1.json' ]]; then cp -- '/tmp/v027_p1.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_p1.md' ]]; then cp -- '/tmp/v027_p1.md' "$LEGACY_ARTIFACT_DIR/"; fi
