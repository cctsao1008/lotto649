"""Issue #15 P3e: 35 fixed original-P2 targets, deterministic sharding and audit."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lotto7x7.log_product_p3d_legacy_replay_v027 import CONFIG
from lotto7x7.log_product_p3e_merge_v027 import (
    CHECKPOINTS, SCHEMA, expected_shard_indices, merge_reports,
    markdown_report, main,
)
from lotto7x7.log_product_p3b_pilot_v027 import evaluate_draw
from lotto7x7.log_product_p3c_exact_iid_v027 import exact_fair_probability

YES = ((1,2,3,4,5,6),(7,8,9,10,11,12),
       (13,14,15,16,17,18),(19,20,21,22,23,24))
NO = ((7,8,9,10,11,12),(13,14,15,16,17,18),
      (19,20,21,22,23,24),(25,26,27,28,29,30))


def sample_report(i):
    indices = expected_shard_indices(i)
    records = []
    main, bonus = [1,2,3,4,5,6], 49
    for t in indices:
        pos = CHECKPOINTS.index(t)
        choices = {}
        for mode, winning in (("primary",pos < 6),
                              ("distinct_sums",pos < 2)):
            ticket = YES if winning else NO
            choices[mode] = {
                "status":"HEURISTIC_UNCERTIFIED",
                "selection_failure":False,
                "tickets":[list(x) for x in ticket],
                "outcome":evaluate_draw(ticket,main,bonus),
            }
        random_ticket = YES if pos < 5 else NO
        records.append({
            "target_index_zero_based":t,
            "training_count":t,
            "previous_draw_id":1000000+t,
            "target_draw_id":1000001+t,
            "previous_draw_date":"2026-09-01",
            "target_draw_date":"2026-09-02",
            "source_prefix_sha256":"a"*64,
            "original_P2_schema":"lotto7x7.log_product.v0.27.p2",
            "original_P2_config":CONFIG,
            "target_main":main,
            "target_bonus":bonus,
            "original_P2_portfolios":choices,
            "random_control":{"tickets":[list(x) for x in random_ticket],
                              "outcome":evaluate_draw(random_ticket,main,bonus)},
        })
    stats = {}
    for mode in ("primary","distinct_sums","random"):
        total = sum(
            (r["random_control"]["outcome"]["any_award"] if mode=="random" else
             r["original_P2_portfolios"][mode]["outcome"]["any_award"])
            for r in records)
        stats[mode]={"selected":len(records),"selection_failures":0,
                     "prize_events":total}
    return {
        "schema":"lotto7x7.log_product.v0.27.p3d_legacy_replay",
        "source":{
            "analysis_input_sha256":"d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8",
            "last_draw_id":115000103,"historical_draws":2497,"cutoff":"2026-11-06",
        },
        "actual_original_optimizer":"lotto7x7.log_product_dual_portfolio_v027.optimize",
        "optimizer_parameters":CONFIG,
        "target_indices":list(indices),
        "checkpoint_count":len(indices),
        "individual_replays":records,
        "bonus_prefix_sha256":"b"*64,
        "empirical_results":stats,
    }


def all_reports():
    return [sample_report(i) for i in range(5)]


def test_shard_cover_exactly_same_35_p3b_fixed_checkpoints():
    assert len(CHECKPOINTS)==35
    assert CHECKPOINTS[0]==300 and CHECKPOINTS[-1]==2476
    assert sorted(t for i in range(5) for t in expected_shard_indices(i))==list(CHECKPOINTS)
    assert all(len(expected_shard_indices(i))==7 for i in range(5))
    for bad in (-1,5,True,1.5):
        with pytest.raises(ValueError):
            expected_shard_indices(bad)


def test_merge_reaudits_all_original_portfolios_and_exact_null():
    result=merge_reports(all_reports())
    assert result["schema"]==SCHEMA
    assert result["checkpoint_count"]==35
    assert result["checkpoint_indices"]==list(CHECKPOINTS)
    assert len(result["records"])==35
    assert result["empirical"]["primary"]["awards"]==6
    assert result["empirical"]["distinct_sums"]["awards"]==2
    assert result["empirical"]["random"]["awards"]==5
    for name in ("primary","distinct_sums","random"):
        stat=result["empirical"][name]
        assert stat["selected"]==35
        assert stat["failures"]==0
        assert stat["exact_binomial"]["trials"]==35
        assert stat["fair_expected_count"]==pytest.approx(35*float(exact_fair_probability()))
    assert result["two_policy_bonferroni_upper_tail_p"]==pytest.approx(0.5088571499118215,abs=1e-12)
    assert result["matched_p3b_cached_comparison"] is None
    assert "retrospective" in markdown_report(result).lower()
    assert "2006" not in result["stage"]


def test_reject_tampered_timeline_duplicate_source_or_award():
    for mutator in (
        lambda r: r[0]["target_indices"].reverse(),
        lambda r: r[1]["source"].update(analysis_input_sha256="0"*64),
        lambda r: r[0]["individual_replays"][0]["original_P2_portfolios"]["primary"]["outcome"].update(any_award=0),
        lambda r: r[3]["individual_replays"][0].update(training_count=999),
        lambda r: r[1]["bonus_prefix_sha256"]="wrong",
        lambda r: r[2]["empirical_results"]["primary"].update(prize_events=999),
        lambda r: r[4]["individual_replays"][0]["original_P2_portfolios"]["primary"].update(tickets=[[1,2,3,4,5,6]]*4),
    ):
        copies=all_reports()
        mutator(copies)
        with pytest.raises((ValueError,AssertionError)):
            merge_reports(copies)
    with pytest.raises(ValueError):
        merge_reports(all_reports()[:4])


def test_cli_merge_writes_full_audit_and_refuses_overwrite(tmp_path,capsys):
    paths=[]
    for i,r in enumerate(all_reports()):
        p=tmp_path / f"p3d_shard_{i}.json"
        p.write_text(json.dumps(r),encoding="utf-8")
        paths.append(p)
    out=tmp_path/"p3e.json"
    md=tmp_path/"p3e.md"
    arguments=[arg for p in paths for arg in ("--shard",str(p))]
    args=[*arguments,"--out",str(out),"--md-out",str(md)]
    main(args)
    result=json.loads(out.read_text(encoding="utf-8"))
    assert result["schema"]==SCHEMA
    assert len(result["shard_report_SHA256"])==5
    assert "[P3E MERGE PASS]" in capsys.readouterr().out
    assert "35-Checkpoint" in md.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(args)
