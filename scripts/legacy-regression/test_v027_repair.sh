#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy

python -m py_compile lotto7x7/portfolio/distribution_repair_v027.py
python -m pytest -q tests/portfolio/test_distribution_repair_v027.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v027_baseline.json --md-out /tmp/v027_baseline.md
python - <<'PY'
import json, math
from pathlib import Path
r = json.loads(Path('/tmp/v027_baseline.json').read_text())
assert r['source']['historical_draws'] == 2497
assert r['source']['last_draw_id'] == 115000103
assert r['source']['cutoff'] == '2026-11-06'
assert r['theory']['universe'] == math.comb(49, 6)
assert r['theory']['mean'] == 150 and r['theory']['variance'] == 1075
assert len(r['bins']) == 259
assert sum(b['observed_count'] for b in r['bins']) == 2497
assert r['decision'].startswith('DESCRIPTIVE_BASELINE_PASS')
assert Path('/tmp/v027_baseline.md').is_file()
print('[PASS] verified 2,497-draw historical P0 baseline')
PY
if [[ -f '/tmp/v027_baseline.json' ]]; then cp -- '/tmp/v027_baseline.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v027_baseline.md' ]]; then cp -- '/tmp/v027_baseline.md' "$LEGACY_ARTIFACT_DIR/"; fi
