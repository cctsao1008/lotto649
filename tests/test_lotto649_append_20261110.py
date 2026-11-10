"""End-to-end source integrity for appended draw #115000104 (2026-11-10).

Exactly one new record in annual CSV, 15-byte history BIN and 53-byte
image BIN; first 2497 draws must be reproducible byte-for-byte after
restoring the historical 2497-record header count. Immutable research
cutoff / P3f seal still evaluates the new draw *after* the model cutoff.
"""
from __future__ import annotations

from datetime import date
from hashlib import sha256
import csv
import json
import struct
from pathlib import Path

from lotto7x7.distribution_repair_v027 import baseline
from lotto7x7.prospective_p3f_static10_v027 import (
    load_frozen_seal, read_verified_history, evaluate_frozen_future
)

ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/"data"/"bin"
CSV=ROOT/"data"/"csv"
SOURCE=BIN/"lotto649_source_manifest.json"
HISTORY=BIN/"lotto649_history_v1.bin"
IMAGE=BIN/"lotto649_images_v2.bin"
OLD_HISTORY_SHA256="2418e5a3f3a7254035cacad04aab43f2e6dd4c70e83da64498a1f57602274e2e"
OLD_IMAGE_SHA256="736f598a31abfa43cd91b7d58f878f52c48af850b2581b909ac6fbf649278b4b"
ORIGINAL_PREFIX_SHA256="d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"
DRAW=(115000104,20261110,(11,23,30,32,40,41),28)
HEADER=struct.Struct("<8sII")
HROW=struct.Struct("<II7B")
IROW=struct.Struct("<I49B")


def _read_binary(path,magic,record):
    raw=path.read_bytes()
    actual_magic,n,width=HEADER.unpack_from(raw)
    assert (actual_magic,n,width)==(magic,2498,record.size)
    assert len(raw)==HEADER.size+n*record.size
    return raw


def test_20261110_exact_csv_record_and_date_id_nonoverlap():
    rows=list(csv.DictReader((CSV/"lotto649_2026.csv").open(encoding="utf-8-sig",newline="")))
    assert len(rows)==104
    assert [int(row["draw_id"]) for row in rows]==list(range(115000001,115000105))
    row=rows[-1]
    assert row["date"]=="2026/11/10"
    assert int(row["draw_id"])==DRAW[0]
    assert tuple(int(row[f"n{i}"]) for i in range(1,7))==DRAW[2]
    assert int(row["bonus"])==DRAW[3]
    assert list(row)==["date","draw_id","n1","n2","n3","n4","n5","n6","bonus"]
    assert all(1<=x<=49 for x in (*DRAW[2],DRAW[3]))
    assert len(set((*DRAW[2],DRAW[3])))==7


def test_history_v1_bin_only_one_appended_record_original_2497_unchanged():
    raw=_read_binary(HISTORY,b"L649HIS1",HROW)
    record=HROW.unpack_from(raw,16+2497*HROW.size)
    assert record==(DRAW[0],DRAW[1],*DRAW[2],DRAW[3])
    restored=bytearray(raw[:-HROW.size])
    struct.pack_into("<I",restored,8,2497)
    assert sha256(restored).hexdigest()==OLD_HISTORY_SHA256


def test_image_v2_bin_only_one_appended_record_original_2497_unchanged():
    raw=_read_binary(IMAGE,b"L649IMG2",IROW)
    draw_id,*pixels=IROW.unpack_from(raw,16+2497*IROW.size)
    assert draw_id==DRAW[0]
    assert all(p in (0,1) for p in pixels)
    assert sum(pixels)==6
    assert tuple(i+1 for i,p in enumerate(pixels) if p)==DRAW[2]
    restored=bytearray(raw[:-IROW.size])
    struct.pack_into("<I",restored,8,2497)
    assert sha256(restored).hexdigest()==OLD_IMAGE_SHA256


def test_manifest_matches_binary_checksums_and_independent_20261110_provenance():
    manifest=json.loads(SOURCE.read_text(encoding="utf-8"))
    assert manifest["dataset"]=="lotto649.appended-verified.v2"
    assert manifest["cutoff"]=="2026-11-10"
    assert manifest["total_draws"]==2498
    assert manifest["first_draw_id"]==93000001
    assert manifest["last_draw_id"]==DRAW[0]
    assert manifest["by_year"]["2026"]==104
    assert sum(manifest["by_year"].values())==2498
    old=manifest["original_source_snapshot"]
    assert (old["total_draws"],old["last_draw_id"],old["cutoff"]) == (
        2497,115000103,"2026-11-06")
    assert old["prefix_sha256"]==ORIGINAL_PREFIX_SHA256
    assert len(manifest["appended_draws"])==1
    append=manifest["appended_draws"][0]
    assert append["draw_id"]==DRAW[0] and append["date"]=="2026-11-10"
    assert append["main"]==list(DRAW[2]) and append["bonus"]==DRAW[3]
    assert append["source_status"]=="CORROBORATED_BY_INDEPENDENT_NON_OFFICIAL_SOURCES"
    assert {x["name"] for x in append["sources"]}=={
        "Central News Agency","H8 Lottery Results"}
    assert len(manifest["source_pages"])==25
    assert sum(row["rows"] for row in manifest["source_pages"].values())==2497
    for name in ("lotto649_history_v1.bin","lotto649_images_v2.bin"):
        raw=(BIN/name).read_bytes()
        assert manifest["files"][name]=={"bytes":len(raw),"sha256":sha256(raw).hexdigest()}


def test_all_23_annual_csvs_byte_match_history_and_images_without_network():
    # Preserve the original established BIN -> yearly CSV verification.
    import importlib.util
    import sys
    path=ROOT/"scripts"/"export_lotto649_csv.py"
    spec=importlib.util.spec_from_file_location("_lotto649_export_append_test",path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    counts=module.export(BIN,CSV,verify=True)
    assert sum(counts.values())==2498
    assert counts[2026]==104
    assert len(counts)==23


def test_original_research_prefix_sha_unchanged_and_prospective_one_draw_observed():
    report=baseline(CSV,cutoff="2026-11-06")
    source=report["source"]
    assert source["historical_draws"]==2497
    assert source["last_draw_id"]==115000103
    assert source["analysis_input_sha256"]==ORIGINAL_PREFIX_SHA256
    assert source["post_cutoff_rows_excluded"]==1
    seal,provenance=load_frozen_seal()
    observed=read_verified_history(CSV,seal)
    assert observed==[{"draw_id":DRAW[0],"date":"2026-11-10",
                       "main":list(DRAW[2]),"bonus":DRAW[3]}]
    graded=evaluate_frozen_future(seal,observed,provenance)
    assert graded["observed_draws"]==1
    assert graded["pending_draws"]==9
    assert graded["earliest_missing_draw"]==115000105
    assert graded["final_exact_one_sided_binomial_tests"] is None
    assert graded["two_P2_policy_bonferroni_upper_tail_p"] is None
    assert graded["status_per_draw"][0]["actual_main"]==list(DRAW[2])
    assert graded["status_per_draw"][0]["actual_bonus"]==DRAW[3]
    assert all(record["status"]=="PENDING" for record in graded["status_per_draw"][1:])
