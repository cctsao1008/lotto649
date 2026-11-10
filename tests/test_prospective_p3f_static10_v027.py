"""Issue #15 P3f: preregistered pre-draw frozen ten-draw shadow validation.

Unlike past backtests, neither strategy nor selected tickets may be changed
after future outcomes. This test intentionally copies the sealed 2497-row
reference dataset into scratch space when simulating forthcoming draws.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from lotto7x7.log_product_p3b_pilot_v027 import evaluate_draw
from lotto7x7.log_product_p3c_exact_iid_v027 import exact_fair_probability
from lotto7x7.prospective_p3f_static10_v027 import (
    SEAL, SCHEMA, SEAL_COMMIT, PINNED_GIT_BLOB_SHA1, REPORT_SCHEMA,
    _blob_sha1, _canonical_seal_bytes, load_frozen_seal, validate_manifest,
    read_verified_history, evaluate_frozen_future, markdown_report, main,
)


@pytest.fixture(scope="module")
def pinned_seal():
    manifest,provenance=load_frozen_seal(SEAL)
    return manifest,provenance


def test_immutable_source_timestamp_and_exact_git_blob_pin(pinned_seal):
    seal,provenance=pinned_seal
    assert seal["schema"]==SCHEMA
    raw=SEAL.read_bytes()
    canonical,mode=_canonical_seal_bytes(raw)
    assert _blob_sha1(canonical)==PINNED_GIT_BLOB_SHA1
    assert provenance["first_pre_draw_seal_commit"]==SEAL_COMMIT
    assert provenance["seal_bytes_sha256"]==sha256(raw).hexdigest()
    assert provenance["canonical_seal_sha256"]==sha256(canonical).hexdigest()
    assert provenance["working_tree_line_endings"]==mode
    assert seal["historical_cutoff"]["last_draw_id"]==115000103
    assert seal["historical_cutoff"]["draws"]==2497
    assert seal["historical_cutoff"]["last_draw_date"]=="2026-11-06"
    assert seal["source"]["original_p2_module_git_blob_sha1"]=="e190bfbc847658301d69cc9d91a4c26dba27617e"
    assert len(seal["experiment"]["planned_draws"])==10
    assert [v["draw_id"] for v in seal["experiment"]["planned_draws"]]==list(range(115000104,115000114))
    assert seal["experiment"]["planned_draws"][0]["date"]=="2026-11-10"
    assert seal["experiment"]["planned_draws"][-1]["date"]=="2026-12-11"
    assert seal["experiment"]["prespecified_primary_hypotheses"]==[
        "original_p2_primary","original_p2_distinct_sums"
    ]
    assert seal["experiment"]["user_existing_paid_ticket_policy"].startswith("Existing purchased")


def test_public_shadow_bets_are_legal_and_ten_draws_are_fixed(pinned_seal):
    m,_=pinned_seal
    for name in ("original_p2_primary","original_p2_distinct_sums",
                 "history_blind_random_control"):
        rows=m["strategies"][name]["tickets"]
        assert len(rows)==4
        assert all(len(t)==6 and sorted(t)==t for t in rows)
        assert len(set(n for t in rows for n in t))==24
    assert len({sum(t) for t in
                m["strategies"]["original_p2_distinct_sums"]["tickets"]})==4
    assert all("tickets" in m["strategies"][k] for k in m["strategies"])
    assert m["status"]=="SHADOW_PRE_REGISTERED_NO_PURCHASE"
    validate_manifest(m)


def test_crlf_checkout_is_equivalent_to_pre_draw_git_blob(tmp_path,pinned_seal):
    # Reproduce Windows core.autocrlf=true on Linux CI AND Linux LF on Windows.
    working_tree = SEAL.read_bytes()
    canonical,mode = _canonical_seal_bytes(working_tree)
    assert mode in ("LF","CRLF_CHECKOUT")
    assert b"\r" not in canonical
    assert _blob_sha1(canonical)==PINNED_GIT_BLOB_SHA1
    windows = canonical.replace(b"\n",b"\r\n")
    assert windows != canonical
    assert _blob_sha1(windows)!=PINNED_GIT_BLOB_SHA1
    assert _canonical_seal_bytes(windows)==(canonical,"CRLF_CHECKOUT")
    copy = tmp_path/"windows_checkout.json"
    copy.write_bytes(windows)
    manifest,provenance = load_frozen_seal(copy)
    assert manifest == pinned_seal[0]
    assert provenance["pinned_git_blob_sha1"]==PINNED_GIT_BLOB_SHA1
    assert provenance["working_tree_line_endings"]=="CRLF_CHECKOUT"
    assert provenance["seal_bytes_sha256"]==sha256(windows).hexdigest()
    assert provenance["canonical_seal_sha256"]==sha256(canonical).hexdigest()
    # CRLF is the only permitted transformation; mixed/stray CR is rejected.
    mixed = canonical.replace(b"\n",b"\r\n",1)
    with pytest.raises(ValueError,match="mixed or invalid line endings"):
        _canonical_seal_bytes(mixed)
    with pytest.raises(ValueError,match="mixed or invalid line endings"):
        _canonical_seal_bytes(canonical+b"\r")
    # Even a valid JSON edit in a Windows-CRLF working tree must fail.
    edited=windows.replace(b'"fixed_draw_count": 10',b'"fixed_draw_count": 11')
    copy.write_bytes(edited)
    with pytest.raises(ValueError,match="PROSPECTIVE SEAL MODIFIED"):
        load_frozen_seal(copy)


def test_any_modified_seal_rejected_even_if_valid_json(tmp_path,pinned_seal):
    original=SEAL.read_bytes()
    altered=tmp_path/"altered.json"
    altered.write_bytes(original.replace(b'"fixed_draw_count": 10',b'"fixed_draw_count": 11'))
    with pytest.raises(ValueError,match="PROSPECTIVE SEAL MODIFIED"):
        load_frozen_seal(altered)
    altered.write_bytes(original+b"\n")
    with pytest.raises(ValueError,match="PROSPECTIVE SEAL MODIFIED"):
        load_frozen_seal(altered)
    copies=[
        lambda x:x["experiment"].update(fixed_draw_count=11),
        lambda x:x["strategies"]["original_p2_primary"]["tickets"][0].__setitem__(0,2),
        lambda x:x["experiment"]["planned_draws"][0].update(date="2026-11-11"),
        lambda x:x["source"]["selection_parameters"].update(seed=100),
        lambda x:x["strategies"]["original_p2_distinct_sums"]["tickets"][0].__setitem__(0,48),
    ]
    for mutate in copies:
        m=deepcopy(pinned_seal[0])
        mutate(m)
        with pytest.raises((ValueError,TypeError)):
            validate_manifest(m)


def _future(m, i=0, main=None, bonus=7):
    schedule=m["experiment"]["planned_draws"][i]
    return {
        "draw_id":schedule["draw_id"],
        "date":schedule["date"],
        "main":main or [1,2,3,4,5,6],
        "bonus":bonus,
    }


def test_pending_not_recorded_as_loss_nor_tested_before_tenth_draw(pinned_seal):
    m,prov=pinned_seal
    result=evaluate_frozen_future(m,[],prov)
    assert result["schema"]==REPORT_SCHEMA
    assert result["observed_draws"]==0
    assert result["pending_draws"]==10
    assert result["earliest_missing_draw"]==115000104
    assert all(x["status"]=="PENDING" for x in result["status_per_draw"])
    assert all(x["awarded_draws"]==0 and x["observed"]==0 for x in result["strategy_statistics"].values())
    assert all(x["iid_expected_any_prize_draws"]==0 for x in result["strategy_statistics"].values())
    assert result["final_exact_one_sided_binomial_tests"] is None
    assert result["two_P2_policy_bonferroni_upper_tail_p"] is None
    assert "NOT evaluated" in markdown_report(result)
    out=evaluate_frozen_future(m,[_future(m)],prov)
    assert out["observed_draws"]==1 and out["pending_draws"]==9
    assert out["final_exact_one_sided_binomial_tests"] is None
    assert out["status_per_draw"][1]["status"]=="PENDING"
    for k in m["strategies"]:
        predicted=evaluate_draw(m["strategies"][k]["tickets"],(1,2,3,4,5,6),7)
        assert out["status_per_draw"][0]["evaluations"][k]==predicted


def test_final_ten_draw_statistical_gate_and_control_not_in_two_policy_adjustment(pinned_seal):
    m,provenance=pinned_seal
    # These are *SYNTHETIC* future draws for a UNIT TEST, never an assertion
    # about actual future results. Every future draw date/ID is still checked.
    rows=[_future(m,i,main=[1,2,3,4,5,6],bonus=7) for i in range(10)]
    report=evaluate_frozen_future(m,rows,provenance)
    assert report["observed_draws"]==10
    assert report["pending_draws"]==0
    assert report["earliest_missing_draw"] is None
    assert "COMPLETE" in report["decision"]
    assert len(report["final_exact_one_sided_binomial_tests"])==3
    expected=10*float(exact_fair_probability())
    for k,stat in report["strategy_statistics"].items():
        assert stat["iid_expected_any_prize_draws"]==pytest.approx(expected,abs=1e-12)
        assert report["final_exact_one_sided_binomial_tests"][k]["trials"]==10
    pvals=[report["final_exact_one_sided_binomial_tests"][k]["upper_tail_probability"]
           for k in ("original_p2_primary","original_p2_distinct_sums")]
    assert report["two_P2_policy_bonferroni_upper_tail_p"]==pytest.approx(min(1.,2*min(pvals)))
    assert "Bonferroni" in markdown_report(report)


@pytest.mark.parametrize("edit", [
    lambda x,m:x.append(_future(m)),
    lambda x,m:x[0].update(draw_id=115000103),
    lambda x,m:x[0].update(date="2026-11-11"),
    lambda x,m:x[0].update(bonus=1),
    lambda x,m:x[0].update(main=[1,1,3,4,5,6]),
    lambda x,m:x[0].update(main=[1,2,3,4,5,50]),
    lambda x,m:x[0].update(main=[6,5,4,3,2,1]),
    lambda x,m:x[0].update(bonus="7"),
])
def test_reject_future_outcome_tampering(edit,pinned_seal):
    m,prov=pinned_seal
    rows=[_future(m)]
    edit(rows,m)
    with pytest.raises((ValueError,TypeError)):
        evaluate_frozen_future(m,rows,prov)


def test_noncontiguous_or_out_of_order_future_results_cannot_be_silently_selected(pinned_seal):
    m,prov=pinned_seal
    with pytest.raises(ValueError,match="non-contiguous"):
        evaluate_frozen_future(m,[_future(m,1)],prov)
    # Even if provided in a different list order, properly contiguous
    # observed draws are resequenced by the fixed preregistration schedule.
    rows=[_future(m,1),_future(m,0)]
    result=evaluate_frozen_future(m,rows,prov)
    assert [x["draw_id"] for x in result["status_per_draw"][:2]]==[115000104,115000105]
    assert result["observed_draws"]==2


def test_frozen_csv_baseline_and_synthetic_unseen_future(tmp_path,pinned_seal):
    m,prov=pinned_seal
    src=Path("data/csv")
    # If the repo subsequently adds outcomes, the test copies only the
    # exact 2497 frozen draws and appends a test-only outcome in scratch.
    root=tmp_path/"csv"
    root.mkdir()
    for path in src.glob("lotto649_????.csv"):
        text=path.read_text(encoding="utf-8-sig")
        if path.name=="lotto649_2026.csv":
            text="".join(line+"\n" for line in text.splitlines()
                         if line.startswith("date,") or
                         (line.split(",")[1].isdigit()
                          and int(line.split(",")[1])<=115000103))
        (root/path.name).write_text(text,encoding="utf-8")
    assert read_verified_history(root,m)==[]
    with (root/"lotto649_2026.csv").open("a",encoding="utf-8") as f:
        f.write("2026/11/10,115000104,01,02,03,04,05,06,07\n")
    observed=read_verified_history(root,m)
    assert observed==[_future(m)]
    report=evaluate_frozen_future(m,observed,prov)
    assert report["observed_draws"]==1
    assert report["final_exact_one_sided_binomial_tests"] is None
    # A changed pre-2026-11-06 record must break the historical prefix SHA.
    p=root/"lotto649_2026.csv"
    corrupt=p.read_text(encoding="utf-8")
    original="2026/11/06,115000103,05,09,26,32,41,48,23"
    altered="2026/11/06,115000103,05,09,26,32,41,47,23"
    assert original in corrupt
    p.write_text(corrupt.replace(original,altered),encoding="utf-8")
    with pytest.raises(ValueError,match="SHA256"):
        read_verified_history(root,m)


def test_cli_baseline_status_and_no_overwrite(tmp_path,pinned_seal,capsys):
    out=tmp_path/"shadow.json";md=tmp_path/"shadow.md"
    args=["--seal",str(SEAL),"--csv-dir","data/csv",
          "--out",str(out),"--md-out",str(md)]
    main(args)
    report=json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"]==REPORT_SCHEMA
    assert report["fixed_draws_total"]==10
    assert report["frozen_seal_provenance"]["pinned_git_blob_sha1"]==PINNED_GIT_BLOB_SHA1
    assert "[P3F SEAL PASS]" in capsys.readouterr().out
    assert "Shadow Test" in md.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(args)
