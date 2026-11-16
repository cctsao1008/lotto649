#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check "numpy>=1.26,<3" "scipy>=1.12,<2"

python -m unittest discover -s tests -p 'test_geometry_v029_p0.py' -v
python -m unittest discover -s tests -p 'test_geometry_theory_v029_p1.py' -v
python -m unittest discover -s tests -p 'test_geometry_convergence_v029_p2a.py' -v
python -m unittest discover -s tests -p 'test_geometry_structure_v029_p2b.py' -v

python -m lotto7x7.geometry_structure_v029 \
  --csv-root data/csv --bin-root data/bin \
  --iid-reps 199 --seed 649 \
  --out results/v029_p2b_ci_structure.json
python - <<'PY'
import json
from pathlib import Path
j = json.loads(Path("results/v029_p2b_ci_structure.json").read_text())
assert j["source_gate"]["status"] == "PASS_REPOSITORY_MANIFEST_CSV_AND_BOTH_BINS"
assert j["draws"] >= 2498
assert len(j["mapping_ids"]) == 4
assert len(j["features"]["tested_structural_pair_features"]) == 3
assert j["source_geometry_sha256"] == "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"
assert j["decision"] == "EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE"
null = j["same_pipeline_iid"]
assert null["replicates"] == 199
assert 0 < null["pair_max_z_empirical_p"] <= 1
assert 0 < null["graph_control_max_z_empirical_p"] <= 1
print("PASS: P2B exact pair moments, historical integrity, and IID null calibration")
print("observed", j["observed"]["maximum_absolute_pair_z"],
      j["observed"]["maximum_absolute_graph_control_z"])
print("familywise p", null["pair_max_z_empirical_p"],
      null["graph_control_max_z_empirical_p"])
print("adjusted p", null["pair_two_endpoint_bonferroni_p"],
      null["graph_two_endpoint_bonferroni_p"])
PY
if [[ -f 'results/v029_p2b_ci_structure.json' ]]; then cp -- 'results/v029_p2b_ci_structure.json' "$LEGACY_ARTIFACT_DIR/"; fi
