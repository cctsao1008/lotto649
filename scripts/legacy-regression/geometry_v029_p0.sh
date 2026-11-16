#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -p 'test_geometry_v029_p0.py' -v
python -m lotto7x7.geometry_v029 --catalog > /tmp/geometry_catalog.json
python -m lotto7x7.geometry_v029 --numbers 1 10 17 24 31 49 > /tmp/geometry_example.json
python - <<'PY'
import json
from pathlib import Path
cat = json.loads(Path("/tmp/geometry_catalog.json").read_text())
example = json.loads(Path("/tmp/geometry_example.json").read_text())
assert cat["regions"]["disk_r3"]["lattice_cell_count"] == 29
assert cat["definition_sha256"] == "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"
assert example["region_occupancies"]["disk_r3"] == 4
assert sum(example["graph"]["pair_distance_squared_histogram"]) == 15
print("PASS: fixed region catalog and known geometry example")
PY
