"""Issue #15 P3a: joint exact-fair support and prequential no-lookahead checks."""
from __future__ import annotations

import csv
from itertools import combinations
from math import comb, prod, isclose
import json
from pathlib import Path

import pytest

from lotto7x7.distribution_repair_v027 import baseline
from lotto7x7.distribution_repair_scoring_v027 import score_p0
from lotto7x7.log_product_v027 import historical_baseline, _grid
from lotto7x7.log_product_dual_scoring_v027 import dual_score, exact_gain_numerators
from lotto7x7.log_product_joint_p3_v027 import (
    SCHEMA, exact_event_mass, prequential_test, analyze, markdown_report,
    _history_prefix, _observed_features, _theory_pearson_sum_log, main,
)
from lotto7x7.log_product_dual_portfolio_v027 import enumerate_joint_counts


def test_exact_joint_event_probability_not_independence_product():
    # 10 equally likely full outcomes by exact combinatorial mass.
    joint = [[1, 1],[3, 5]]
    m = exact_event_mass([True,False],[False,True],joint)
    assert m["dual_positive_mass"] == 1
    assert m["sum_positive_mass"] == 2
    assert m["log_positive_mass"] == 6
    assert m["dual_positive_probability"] == pytest.approx(0.1)
    assert m["independence_product_not_valid"] == pytest.approx(0.12)
    assert m["dependence_difference_from_independence"] == pytest.approx(-0.02)
    for bad in (
        ([True],[True,False],joint),
        ([True,1],[True,False],joint),
        ([True,False],[True],joint),
        ([True,False],[True,False],[[1,-1],[1,2]]),
    ):
        with pytest.raises(ValueError):
            exact_event_mass(*bad)


def test_joint_repair_exact_rational_vs_independent_brute_force():
    joint = [[1,2],[3,4]]
    observed=[[1,0],[1,0]]
    flat=[x for row in observed for x in row]
    theory=[x for row in joint for x in row]
    from lotto7x7.distribution_repair_v027 import squared_l2_probability
    original=squared_l2_probability(flat,theory)
    gains=exact_gain_numerators(flat,theory)
    for i,g in enumerate(gains["numerators"]):
        new=flat[:]
        new[i]+=1
        direct=original-squared_l2_probability(new,theory)
        assert g/gains["denominator"] == pytest.approx(direct,abs=1e-14)


def test_synthetic_prequential_temporal_prefix_only():
    joint=[[1,2],[3,4]]
    features=[
        (0,0,0.),(1,1,1.),(0,1,0.5),
        (0,0,0.),(1,0,0.9),(1,1,1.1),(0,1,0.4),
    ]
    r=prequential_test(features,joint,warmup=3)
    assert r["evaluated_draws"] == 4
    assert len(r["records"]) == 4
    assert r["records"][0]["prior_history_draws"]==3
    # Independently compute the FIRST forecast using only first 3 draws.
    sc=[2,1];lc=[1,2]
    sg=exact_gain_numerators(sc,[3,7])
    lg=exact_gain_numerators(lc,[4,6])
    expected=exact_event_mass([x>0 for x in sg["numerators"]],
                              [x>0 for x in lg["numerators"]],joint)
    assert r["records"][0]["conditional_fair_probability"] == expected["dual_positive_probability"]
    assert r["realized_positive_count"] == sum(v["realized_dual_positive"] for v in r["records"])
    assert r["expected_positive_count"] == pytest.approx(sum(v["conditional_fair_probability"] for v in r["records"]))
    assert 0 <= r["hoeffding_azuma_upper_tail_bound"] <= 1
    assert 0 <= r["hoeffding_azuma_lower_tail_bound"] <= 1
    assert prequential_test(features,joint,warmup=3) == r
    for warmup in (-1,0,3.0,7,True):
        with pytest.raises(ValueError):
            prequential_test(features,joint,warmup=warmup)


def test_theory_correlation_positive_less_than_one_and_unbinned():
    result=_theory_pearson_sum_log()
    corr=result["sum_log_pearson_correlation"]
    assert .85<corr<.98
    assert result["sum_log_covariance"]>0
    assert "unbinned" in result["note"]


ROWS=[
    ("2026/10/27",115000100,1,12,18,23,35,47,49),
    ("2026/10/30",115000101,4,22,25,30,34,37,48),
    ("2026/11/03",115000102,2,23,24,25,29,39,31),
    ("2026/11/06",115000103,5,9,26,32,41,48,23),
    ("2026/11/10",115000104,1,2,3,4,5,6,7),
]


def write_annual(folder, rows):
    with (folder/"lotto649_2026.csv").open("w",encoding="utf-8",newline="") as f:
        out=csv.writer(f)
        out.writerow(("date","draw_id","n1","n2","n3","n4","n5","n6","bonus"))
        out.writerows(rows)


@pytest.fixture(scope="module")
def fixture_joint(tmp_path_factory):
    folder=tmp_path_factory.mktemp("joint_synthetic")
    write_annual(folder,ROWS[:4])
    s0=baseline(folder,cutoff="2026-11-06")
    s1=score_p0(s0)
    l0=historical_baseline(folder,cutoff="2026-11-06")
    d1=dual_score(s1,l0)
    r=analyze(s1,l0,d1,folder,warmup=3)
    return folder,s1,l0,d1,r


def test_synthetic_pipeline_trajectory_and_no_lookahead(fixture_joint):
    folder,s1,l0,d1,r=fixture_joint
    assert r["schema"]==SCHEMA
    assert r["source"]["last_draw_id"]==115000103
    assert r["source"]["historical_draws"]==4
    assert r["source"]["analysis_input_sha256"]==s1["source"]["analysis_input_sha256"]
    j=r["joint"]
    assert sum(map(sum,j["exact_theory_counts"]))==comb(49,6)
    assert sum(map(sum,j["historical_counts"]))==4
    assert j["nonzero_joint_cells"]>0
    assert j["squared_l2_probability_v1"]>0
    assert len(r["theory_checksums"]["joint_counts"])==64
    p=r["prequential"]
    assert p["warmup"]==3 and p["evaluated_draws"]==1
    assert p["records"][0]["prior_history_draws"]==3
    assert "full four-ticket" in r["warnings"][2] or "portfolio" in r["warnings"][2]
    assert "Chronological" in markdown_report(r)

    # Appending a later draw cannot alter any values computed up to frozen cutoff.
    write_annual(folder,ROWS)
    data,n=_history_prefix(folder,d1["source"])
    assert n==4
    assert len(data.numbers)==5
    changed=analyze(s1,l0,d1,folder,warmup=3)
    assert changed==r


def test_cli_frozen_gate_and_no_overwrite(tmp_path,fixture_joint,capsys):
    folder,s1,l0,d1,r=fixture_joint
    paths=[]
    for label,data in (("s1",s1),("l0",l0),("d1",d1)):
        path=tmp_path/f"{label}.json"
        path.write_text(json.dumps(data),encoding="utf-8")
        paths.append(str(path))
    out=tmp_path/"joint.json"
    md=tmp_path/"joint.md"
    args=["--sum-p1",paths[0],"--log-p0",paths[1],"--dual-p1",paths[2],
          "--csv-dir",str(folder),"--cutoff","2026-11-06",
          "--warmup","3","--out",str(out),"--md-out",str(md)]
    main(args)
    saved=json.loads(out.read_text(encoding="utf-8"))
    assert saved["schema"]==SCHEMA
    assert saved["joint"]==r["joint"]
    assert "[JOINT-P3A PASS]" in capsys.readouterr().out
    assert "prequential" in md.read_text(encoding="utf-8").lower()
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(ValueError,match="cutoff"):
        main([*args[:7],"2026-11-05",*args[8:]])
