"""Issue #15 P3c: exact null for ANY pre-draw 4-disjoint strategy."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import comb
import json

import pytest

from lotto7x7.distribution_repair_portfolio_v027 import validate_four_disjoint
from lotto7x7.log_product_p3c_exact_iid_v027 import (
    SCHEMA, exact_binomial_tail, exact_fair_probability, audit_pilot,
    markdown_report, main,
)
from lotto7x7.log_product_p3b_pilot_v027 import evaluate_draw
from lotto7x7.official_prize_v026 import official_prize_coverage
from lotto7x7.deficit_compensation_v025 import REFERENCE


TICKETS_YES=((1,2,3,4,5,6),(7,8,9,10,11,12),
             (13,14,15,16,17,18),(19,20,21,22,23,24))
TICKETS_NO=((7,8,9,10,11,12),(13,14,15,16,17,18),
            (19,20,21,22,23,24),(25,26,27,28,29,30))


def test_exact_official_fair_odds_identical_after_relabeling():
    p=exact_fair_probability()
    assert isinstance(p,Fraction)
    assert float(p)==pytest.approx(0.1227314589619088,abs=1e-15)
    assert p.denominator <= comb(49,6)*43
    for portfolio in (REFERENCE,TICKETS_YES,TICKETS_NO,
                      ((1,3,5,7,9,11),(2,4,6,8,10,12),
                       (13,15,17,19,21,23),(14,16,18,20,22,24))):
        validate_four_disjoint(portfolio)
        data=official_prize_coverage(portfolio)
        assert Fraction(data["any_prize_outcomes"],data["total_outcomes"]) == p
        assert data["ticket_count"]==4


def test_exact_binomial_tails_and_half_integer_cases():
    r=exact_binomial_tail(1,1,Fraction(1,4))
    assert r["upper_tail_probability"]==0.25
    assert r["lower_tail_probability"]==1.
    assert r["upper_tail_exact_numerator"]=="1"
    assert r["upper_tail_exact_denominator"]=="4"
    r=exact_binomial_tail(2,1,Fraction(1,4))
    assert r["upper_tail_probability"]==0.4375
    assert r["lower_tail_probability"]==0.9375
    r=exact_binomial_tail(35,6,exact_fair_probability())
    assert r["upper_tail_probability"] == pytest.approx(0.25442857495591076,rel=1e-12)
    for n,k,p in ((0,0,Fraction(1,4)),(2,3,Fraction(1,4)),
                  (4,-1,Fraction(1,4)),(5,3,0.1),(5,3,Fraction(1,1)),
                  (True,0,Fraction(1,4))):
        with pytest.raises(ValueError):
            exact_binomial_tail(n,k,p)


def synthetic_pilot():
    n=35
    expected={"primary":6,"distinct_sums":2,"random":5}
    stats={}
    records=[]
    for name,k in expected.items():
        stats[name]={"selected":n,"wins":k,"total_awards":k,
                     "failures":0,"sum_max_main_hits":0}
    for i in range(n):
        decisions={}
        for name,k in expected.items():
            ticket=TICKETS_YES if i<k else TICKETS_NO
            outcome=evaluate_draw(ticket,(1,2,3,4,5,6),40)
            assert outcome["any_award"]==int(i<k)
            decisions[name]={"tickets":[list(t) for t in ticket],**outcome}
        t=300+64*i
        records.append({
            "training_count":t,"target_index_zero_based":t,
            "target_main":[1,2,3,4,5,6],"target_bonus":40,
            "decisions":decisions,
        })
    return {
        "schema":"lotto7x7.log_product.v0.27.p3b_pilot",
        "decision":"EXPLORATORY_PILOT; PREDICTIVE_NO_GO_OR_INCONCLUSIVE",
        "source":{"cutoff":"2026-11-06","last_draw_id":115000103,
                  "historical_draws":2497,
                  "analysis_input_sha256":"d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"},
        "policy":{"policy":"cached_joint_cell_greedy_v1",
                  "warmup":300,"stride":64,
                  "catalogue_checksum_sha256":"0"*64},
        "theory_checksums":{
            "joint":"31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a"},
        "fair_reference":{
            "official_eight_tier_any_prize_probability":float(exact_fair_probability())},
        "historical":{"evaluated_checkpoints":n,"records":records,
                      "stats":stats},
        "bonus_prefix_sha256":"1"*64,
        "iid_null":{"max_statistic_multiple_policy_adjusted_upper_tail_p":0.52,
                    "replay_count":24},
    }


def test_full_35_checkpoint_exact_binomial_and_two_policy_adjustment():
    fake=synthetic_pilot()
    result=audit_pilot(fake)
    assert result["schema"]==SCHEMA
    assert "PREDICTIVE_NO_GO" in result["decision"]
    assert result["historical"]["fixed_checkpoints"]==35
    expected={"primary":6,"distinct_sums":2,"random":5}
    for key,number in expected.items():
        row=result["historical"]["exact_binomial_tests"][key]
        assert row["successes"]==number
        assert row["trials"]==35
        assert row["exact_fair_expected_successes"]==pytest.approx(4.295601063666808)
    assert result["two_policy_familywise_upper_tail"]["adjusted_p_upper_bound"]==pytest.approx(
        0.5088571499118215,rel=1e-11)
    assert result["original_cached_pilot_reference"]["pilot_MC_p"]==0.52
    assert "Binomial" in markdown_report(result)
    assert "original P2" in markdown_report(result)


def test_failure_does_not_silently_drop_adaptive_checkpoints():
    fake=synthetic_pilot()
    first=fake["historical"]["records"][0]["decisions"]["primary"]
    assert first["any_award"]==1
    fake["historical"]["records"][0]["decisions"]["primary"]={"no_feasible_witness":True}
    fake["historical"]["stats"]["primary"]["selected"]-=1
    fake["historical"]["stats"]["primary"]["wins"]-=1
    fake["historical"]["stats"]["primary"]["total_awards"]-=1
    fake["historical"]["stats"]["primary"]["failures"]+=1
    result=audit_pilot(fake)
    assert result["decision"]=="INCONCLUSIVE_ADAPTIVE_SELECTION_FAILURES"
    assert "two_policy_familywise_upper_tail" not in result


@pytest.mark.parametrize("mutation", [
    lambda x: x["historical"]["records"][0].update(training_count=1234),
    lambda x: x["historical"]["records"][0].update(target_bonus=1),
    lambda x: x["historical"]["records"][1]["decisions"]["primary"].update(any_award=0),
    lambda x: x["policy"].update(stride=3),
    lambda x: x["policy"].update(catalogue_checksum_sha256="wrong"),
    lambda x: x["source"].update(analysis_input_sha256="different"),
    lambda x: x["historical"]["stats"]["primary"].update(wins=123),
])
def test_reject_tampered_outcomes_or_schedule(mutation):
    fake=synthetic_pilot()
    mutation(fake)
    with pytest.raises(ValueError):
        audit_pilot(fake)


def test_cli_p3c_from_archived_report_and_no_overwrite(tmp_path,capsys):
    fake=synthetic_pilot()
    inp=tmp_path/"p3b.json"
    inp.write_text(json.dumps(fake),encoding="utf-8")
    out=tmp_path/"p3c.json";md=tmp_path/"p3c.md"
    args=["--p3b",str(inp),"--out",str(out),"--md-out",str(md)]
    main(args)
    result=json.loads(out.read_text(encoding="utf-8"))
    assert result["schema"]==SCHEMA
    assert result["historical"]["fixed_checkpoints"]==35
    assert "[EXACT FAMILYWISE]" in capsys.readouterr().out
    assert "0.50885715" in md.read_text(encoding="utf-8")
    assert len(result["input_p3b_report_sha256"])==64
    with pytest.raises(FileExistsError):
        main(args)
