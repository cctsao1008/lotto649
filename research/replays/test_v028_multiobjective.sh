#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pytest numpy scipy

python -m py_compile research/lotto7x7/portfolio/multiobjective_v028.py
python -m pytest -q research/tests/portfolio/test_multiobjective_v028.py

python -m lotto7x7.distribution_repair_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v028_s0.json --md-out /tmp/v028_s0.md
python -m lotto7x7.distribution_repair_scoring_v027 --p0 /tmp/v028_s0.json --out /tmp/v028_s1.json --md-out /tmp/v028_s1.md
python -m lotto7x7.log_product_v027 --csv-dir data/csv --cutoff 2026-11-06 --out /tmp/v028_l0.json --md-out /tmp/v028_l0.md
python -m lotto7x7.log_product_dual_scoring_v027 --sum-p1 /tmp/v028_s1.json --log-p0 /tmp/v028_l0.json --out /tmp/v028_d1.json --md-out /tmp/v028_d1.md

python -m lotto7x7.multiobjective_v028 \
  --sum-p1 /tmp/v028_s1.json --log-p0 /tmp/v028_l0.json \
  --dual-p1 /tmp/v028_d1.json --csv-dir data/csv \
  --seed 20261110 --sample-count 80000 --pool-cap 2000 \
  --restarts 40 --batch-size 120000 \
  --out /tmp/v028_multiobjective.json \
  --md-out /tmp/v028_multiobjective.md

python - <<'PY'
import json,math,pathlib
r=json.loads(pathlib.Path('/tmp/v028_multiobjective.json').read_text())
assert r['schema']=='lotto7x7.multiobjective.v0.28.m0'
assert r['source']['last_draw_id']==115000103
assert r['source']['historical_draws']==2497
assert r['source']['analysis_input_sha256']=='d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
assert r['theory']['enumerated']==math.comb(49,6)
assert r['theory']['joint_sha256']=='31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a'
assert all(r['theory']['shape_category_counts'][k]>0 for k in ('component','partition','edge'))
assert r['search']['candidate_passing_four_positive_objective_blocks']>=100
assert r['search']['candidate_eligible_after_old_ticket_exclusion']>=1000
assert r['search']['search_optimality']=='HEURISTIC_UNCERTIFIED'
assert r['theory']['shape_counts_sha256']=='341e33d37429c0f1e53c20ed4d54164c6adefa6080f54dc08c5ca9817967f320'
assert r['search']['sample_sha256']=='5744bb7b6e5ee9e086cb2c9eb17b7872b3028d512365610fb747672586b92c8c'
assert r['search']['candidate_passing_four_positive_objective_blocks']==13704
assert r['search']['bounded_search_pool_size']==2000
assert r['primary']['tickets']==[
    [10,27,29,35,38,47],
    [15,25,30,32,39,49],
    [16,18,24,40,42,46],
    [20,23,28,31,33,43],
]
manifest=json.loads(pathlib.Path('research/experiments/v028_m0_four_block_candidates_20261110.json').read_text())
assert [item['numbers'] for item in manifest['primary_candidate_tickets']]==r['primary']['tickets']
assert manifest['method']['exact_shape_baseline_sha256']==r['theory']['shape_counts_sha256']
assert manifest['method']['exact_joint_theory_sha256']==r['theory']['joint_sha256']
assert manifest['status']=='DESCRIPTIVE_CANDIDATE_ONLY_NOT_A_PAID_TICKET'
assert len(r['search']['sample_sha256'])==64
for v in (r['primary'],*r['sensitivity'].values()):
    assert len(v['tickets'])==4
    assert len({k for ticket in v['tickets'] for k in ticket})==24
    assert v['shape_partition_count']==4
    assert v['old_ticket_overlap_same_line_count']==0
    assert v['status']=='HEURISTIC_UNCERTIFIED'
    assert abs(v['exact_official_any_prize_probability']-.1227314589619088)<1e-12
    assert all(all(z>0 for z in item['block_gain_in_exact_IID_std_units'].values()) for item in v['individual'])
assert 'NO_GO' in r['decision']
print('[PASS] 4 genuinely distinct objectives, exact fair joint/shape, Pareto maximin search, no old ticket reuse')
PY
if [[ -f '/tmp/v028_multiobjective.json' ]]; then cp -- '/tmp/v028_multiobjective.json' "$LEGACY_ARTIFACT_DIR/"; fi
if [[ -f '/tmp/v028_multiobjective.md' ]]; then cp -- '/tmp/v028_multiobjective.md' "$LEGACY_ARTIFACT_DIR/"; fi
