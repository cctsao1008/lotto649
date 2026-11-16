#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s research/tests -p 'test_coverage_burnside_v026.py' -v

mkdir -p results
python -m research.lotto7x7.coverage_burnside_v026 --out results/v026_burnside_independent.json
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('results/v026_burnside_independent.json').read_text(encoding='utf-8'))
assert r['total_nonnegative_labeled_mask_structures']==52450
assert r['distinct_ticket_labeled_mask_structures']==50985
assert r['duplicate_ticket_labeled_mask_structures']==1465
assert r['burnside_fixed_point_numerator']==70968
assert r['unlabeled_valid_membership_structures']==2957
assert r['independent_structural_count_agrees']
assert r['scientific_decision']=='EXACT_STRUCTURE_ONLY_NO_PREDICTIVE_INFERENCE'
print('PASS: 2957 independent orbits; no lottery forecast inferred')
PY
git diff --exit-code
if [[ -f 'results/v026_burnside_independent.json' ]]; then cp -- 'results/v026_burnside_independent.json' "$LEGACY_ARTIFACT_DIR/"; fi
