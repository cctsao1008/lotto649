#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check "numpy>=1.26,<3" "scipy>=1.12,<2"

python -m unittest discover -s tests -p 'test_geometry_repair_v029_p3b.py' -v

python -m lotto7x7.geometry_repair_v029 \
  --csv-root data/csv --bin-root data/bin \
  --pool-size 1024 --random-portfolios 256 \
  --iid-reps 99 --seed 649 \
  --out results/v029_p3b_ci_repair.json
python - <<'PY'
import json
from pathlib import Path
j=json.loads(Path('results/v029_p3b_ci_repair.json').read_text())
assert j['source_gate']['status'] == 'PASS_REPOSITORY_MANIFEST_CSV_AND_BOTH_BINS'
assert j['history_draws'] >= 2498
assert j['decision'] == 'DESCRIPTIVE_COUNTERFACTUAL_ONLY_NO_PREDICTIVE_GO'
assert j['control_design']['iid_replicates'] == 99
assert len(j['observed']['research_only_four_ticket_candidate']) == 4
assert len({n for t in j['observed']['research_only_four_ticket_candidate'] for n in t}) == 24
assert j['observed']['optimized_four_append_gain'] >= j['observed']['random_portfolio_gain_quantiles_0_25_50_75_100'][-1] - 1e-10
p=j['matched_whole_pipeline_iid']
assert 0 < p['excess_over_random_median_empirical_upper_tail_p'] <= 1
assert 0 < p['optimized_gain_empirical_upper_tail_p'] <= 1
print('PASS: P3B exact covariance, four-disjoint repair, and whole-pipeline IID')
print('Gains:',j['observed']['optimized_four_append_gain'],j['observed']['random_portfolio_gain_median'])
print('Fair-IID upper-tail p:',p['excess_over_random_median_empirical_upper_tail_p'],p['optimized_gain_empirical_upper_tail_p'])
PY
if [[ -f 'results/v029_p3b_ci_repair.json' ]]; then cp -- 'results/v029_p3b_ci_repair.json' "$LEGACY_ARTIFACT_DIR/"; fi
