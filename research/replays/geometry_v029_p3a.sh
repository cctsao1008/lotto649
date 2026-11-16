#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check "numpy>=1.26,<3" "scipy>=1.12,<2"

python -m unittest discover -s research/tests -p 'test_geometry_redundancy_v029_p3a.py' -v

python -m lotto7x7.geometry_redundancy_v029 \
  --csv-root data/csv --bin-root data/bin \
  --iid-sample 50000 --seed 649 \
  --out results/v029_p3a_ci_conditional_ablation.json
python - <<'PY'
import json
from pathlib import Path
j=json.loads(Path("results/v029_p3a_ci_conditional_ablation.json").read_text())
assert j["source_gate"]["status"] == "PASS_REPOSITORY_MANIFEST_CSV_AND_BOTH_BINS"
assert j["source_gate"]["draws"] >= 2498
assert j["decision"] == "STRUCTURAL_NONIDENTITY_PROVEN_PREDICTIVE_NO_GO"
assert j["certificate"]["structural_nonidentity_proven"]
assert [r["pair_geometry"]["pair_squared_distance_sum"] for r in j["certificate"]["tickets"]] == [172,258]
assert j["certificate"]["same_v028_baseline"]["log_bin"] == 20
assert j["fair_iid_descriptive_collision_audit"]["sample_draws"] == 50000
assert j["historical_descriptive_collision_audit"]["sample_draws"] >= 2498
print("PASS: P3A strict conditional non-identity and validated historical gate")
print("Historical:",j["historical_descriptive_collision_audit"])
print("Synthetic:",j["fair_iid_descriptive_collision_audit"])
PY
if [[ -f 'results/v029_p3a_ci_conditional_ablation.json' ]]; then cp -- 'results/v029_p3a_ci_conditional_ablation.json' "$LEGACY_ARTIFACT_DIR/"; fi
