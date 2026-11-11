"""M2 verified new-draw append PREVIEW; never mutate the active archive.

One future result is accepted only when two archived publisher pages agree,
the current original archive passes every source gate, and the date/ID matches
the frozen P5A next chronological draw. Output is a newly created, isolated
and self-audited candidate archive under ignored results/ for human review.

There is no apply/commit/push operation: the original files remain untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory

from .draw_ingestion_m2 import audit_source, normalize_draw
from .geometry_shadow_v029 import SEAL_PATH
from .official_draw_evidence_m2 import TRUSTED, verify_local_evidence
from .prospective_evaluation_m4 import verify_p5a_seal
from scripts.export_lotto649_csv import HEAD, HISTORY, IMAGE, generate_yearly_csv

SCHEMA = "lotto649.m2.verified_append_preview.v1"
HISTORY_NAME = "lotto649_history_v1.bin"
IMAGE_NAME = "lotto649_images_v2.bin"
MANIFEST_NAME = "lotto649_source_manifest.json"


def _history_row(draw: dict) -> bytes:
    datecode = int(draw["date"].replace("-", ""))
    return HISTORY.pack(draw["draw_id"], datecode, *draw["main"], draw["bonus"])


def _image_row(draw: dict) -> bytes:
    selected = set(draw["main"])
    return IMAGE.pack(draw["draw_id"], *(int(i in selected) for i in range(1, 50)))


def _append_binary(original: bytes, *, record: struct.Struct,
                   magic: bytes, new_row: bytes, expected_n: int) -> bytes:
    if len(original) != HEAD.size + record.size * expected_n:
        raise ValueError("original binary is not the expected length")
    stored_magic, n, width = HEAD.unpack_from(original)
    if (stored_magic, n, width) != (magic, expected_n, record.size):
        raise ValueError("source binary header and verified draw count disagree")
    if len(new_row) != record.size:
        raise AssertionError("new binary row width differs from original format")
    return HEAD.pack(magic, n + 1, record.size) + original[HEAD.size:] + new_row


def build_preview(
    draw_id: int,
    output: Path,
    *,
    csv_root: Path = Path("data/csv"),
    bin_root: Path = Path("data/bin"),
    evidence_root: Path = Path("data/private/lotto649-source"),
    seal_path: Path = SEAL_PATH,
) -> dict:
    """Write a fresh audited standalone candidate archive, never active source."""
    if type(draw_id) is not int or draw_id < 1:
        raise ValueError("positive integer draw ID is required")
    output = Path(output)
    resolved = output.resolve()
    allowed = Path("results").resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise ValueError("preview output must be a NEW directory below results/")
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing preview: {output}")
    seal = verify_p5a_seal(seal_path)
    current = audit_source(csv_root, bin_root)
    records = seal["experiment"]["planned_draws"]
    already = current["last_draw_id"] - seal["historical_cutoff"]["last_draw_id"]
    if not 0 <= already < len(records):
        raise ValueError("current archive cutoff is outside remaining P5A future window")
    expected = records[already]
    if draw_id != expected["draw_id"]:
        raise ValueError("preview requires the immediate next P5A draw, without gaps")
    evidence_root = Path(evidence_root)
    verified = verify_local_evidence(draw_id, evidence_root)
    draw = normalize_draw(verified["draw"])
    if draw["date"] != expected["date"]:
        raise ValueError("publisher pages contradict frozen P5A schedule")
    manifest_file = Path(bin_root) / MANIFEST_NAME
    original_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (original_manifest["last_draw_id"] != current["last_draw_id"] or
            original_manifest["total_draws"] != current["total_draws"]):
        raise ValueError("manifest cutoff changed during validation")
    pages_index = json.loads((evidence_root / f"{draw_id}.json").read_text("utf-8"))
    evidence_sources = [
        {"name": "CTBC published Lotto649", "url": TRUSTED["ctbc"],
         "page_sha256": verified["page_sha256"]["ctbc"]},
        {"name": "Independent lottery.com.tw publication",
         "url": TRUSTED["independent"],
         "page_sha256": verified["page_sha256"]["independent"]},
    ]
    # Legacy exporter requires this exact status token and two source objects.
    # It is a provenance record, NOT proof of official operator authentication.
    appended = {
        **draw, "source_status": "CORROBORATED_BY_INDEPENDENT_NON_OFFICIAL_SOURCES",
        "sources": evidence_sources,
        "source_index_sha256": verified["source_index_sha256"],
    }
    originals = {}
    for name in (HISTORY_NAME, IMAGE_NAME):
        originals[name] = (Path(bin_root) / name).read_bytes()
        if sha256(originals[name]).hexdigest() != current["source_fingerprints"][
            "history_bin_sha256" if name == HISTORY_NAME else "image_bin_sha256"
        ]:
            raise ValueError("source binary changed during preview verification")
    updated = {
        HISTORY_NAME: _append_binary(
            originals[HISTORY_NAME], record=HISTORY, magic=b"L649HIS1",
            new_row=_history_row(draw), expected_n=current["total_draws"]),
        IMAGE_NAME: _append_binary(
            originals[IMAGE_NAME], record=IMAGE, magic=b"L649IMG2",
            new_row=_image_row(draw), expected_n=current["total_draws"]),
    }
    draft_manifest = deepcopy(original_manifest)
    draft_manifest["total_draws"] += 1
    draft_manifest["last_draw_id"] = draw_id
    draft_manifest["cutoff"] = draw["date"]
    draft_manifest["by_year"][draw["date"][:4]] += 1
    draft_manifest["appended_draws"].append(appended)
    for name, raw in updated.items():
        draft_manifest["files"][name]["bytes"] = len(raw)
        draft_manifest["files"][name]["sha256"] = sha256(raw).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    # Work in an isolated sibling, verify before publishing its final name.
    with TemporaryDirectory(prefix=".m2-append-preview-", dir=output.parent) as temporary:
        root = Path(temporary)
        temp_bin = root / "data" / "bin"
        temp_csv = root / "data" / "csv"
        proof_dir = root / "evidence"
        temp_bin.mkdir(parents=True)
        temp_csv.mkdir(parents=True)
        proof_dir.mkdir(parents=True)
        for name, raw in updated.items():
            (temp_bin / name).write_bytes(raw)
        (temp_bin / MANIFEST_NAME).write_text(
            json.dumps(draft_manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii", newline="\n")
        generated, _ = generate_yearly_csv(temp_bin)
        for relative, raw in generated.items():
            (temp_csv / relative).write_bytes(raw)
        for key in (HISTORY_NAME, IMAGE_NAME):
            if originals[key][HEAD.size:] != updated[key][HEAD.size:
                  HEAD.size + current["total_draws"] * (
                      HISTORY.size if key == HISTORY_NAME else IMAGE.size
                  )]:
                raise AssertionError("original binary prefix modified in candidate archive")
        for page in pages_index["pages"]:
            digest = page["sha256"]
            src = evidence_root / (digest + ".html")
            dest = proof_dir / (digest + ".html")
            shutil.copyfile(src, dest)
            if sha256(dest.read_bytes()).hexdigest() != digest:
                raise ValueError("copied publisher evidence corrupted")
        shutil.copyfile(evidence_root / f"{draw_id}.json",
                        proof_dir / f"{draw_id}.json")
        verification = audit_source(temp_csv, temp_bin)
        if (verification["total_draws"] != current["total_draws"] + 1 or
                verification["last_draw_id"] != draw_id):
            raise AssertionError("candidate archive failed final source gate")
        output_summary = {
            "schema": SCHEMA,
            "status": "PASS_STANDALONE_VERIFIED_APPEND_PREVIEW_NO_SOURCE_MUTATION",
            "draw": draw,
            "source_evidence_index_sha256": verified["source_index_sha256"],
            "source_page_sha256": verified["page_sha256"],
            "previous_manifest_sha256": current["source_fingerprints"]["manifest_sha256"],
            "new_manifest_sha256": verification["source_fingerprints"]["manifest_sha256"],
            "new_draw_count": verification["total_draws"],
            "new_history_sha256": verification["source_fingerprints"]["history_bin_sha256"],
            "new_image_sha256": verification["source_fingerprints"]["image_bin_sha256"],
            "active_repository_files_changed": False,
            "can_claim_authenticated_operator_source": False,
            "ready_to_commit_to_main": False,
            "next_action": (
                "Human review of raw pages, draw number, hashes, full preview diff and "
                "separately authorized append transaction. Never modify P3f/P5A seals."
            ),
        }
        (root / "preview_summary.json").write_text(
            json.dumps(output_summary, sort_keys=True, indent=2) + "\n",
            encoding="ascii", newline="\n")
        # Atomic sibling directory rename on same filesystem. No overwrite.
        root.rename(output)
    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--evidence-root", type=Path,
                        default=Path("data/private/lotto649-source"))
    args = parser.parse_args()
    report = build_preview(
        args.draw_id, args.out_dir,
        csv_root=args.csv_root, bin_root=args.bin_root,
        evidence_root=args.evidence_root,
    )
    print(f"PASS M2 PREVIEW: draw #{report['draw']['draw_id']} "
          f"candidate archive has {report['new_draw_count']} records")
    print("NO ORIGINAL SOURCE FILES WERE MODIFIED / NOT AUTHORIZED TO COMMIT")
    print(f"PREVIEW: {args.out_dir}")


if __name__ == "__main__":
    main()
