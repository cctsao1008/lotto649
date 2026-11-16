#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s research/tests -p 'test_prospective_readiness_v032.py' -v

mkdir -p results
python -m lotto7x7.prospective_readiness_v032 --expected-last-id 115000105 --expected-draws 2499 --out results/v032_readiness_20261114.json
python - <<'PY'
import json
from pathlib import Path
j=json.loads(Path('results/v032_readiness_20261114.json').read_text(encoding='utf-8'))
assert j['scope']=='READINESS_ONLY_NO_OUTCOME_SCORING'
assert j['source_archive']['total_draws']==2499
assert j['source_archive']['latest_draw_id']==115000105
assert j['source_archive']['archive_integrity_is_not_two_publisher_verification']
assert not j['predictive_go'] and not j['wagers_placed']
counts={x['study_id']:x['source_archived_targets_not_yet_admitted_for_scoring'] for x in j['studies']}
assert counts=={'v027_p3f_static10':2,'v029_p5a_static9':1,'v23_m0_static200':0,'v24_p3_iching_static100':0}, counts
assert all(x['prospective_observations_admitted_by_this_tool']==0 for x in j['studies'])
assert all(x['publisher_evidence_per_target']=='NOT_EXAMINED_BY_THIS_PREFLIGHT' for x in j['studies'])
assert j['joint_study_interpretation']['union_bound_any_one_of_two_valid_e_processes_under_joint_fair_null_at_most']==0.10
print('PASS: 4 original seals; 2499 source audited rows; zero scored trials; joint-study multiplicity flagged')
PY
git diff --exit-code
if [[ -f 'results/v032_readiness_20261114.json' ]]; then cp -- 'results/v032_readiness_20261114.json' "$LEGACY_ARTIFACT_DIR/"; fi
