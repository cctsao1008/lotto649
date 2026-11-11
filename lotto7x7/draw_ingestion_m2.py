"""M2: read-only Lotto649 source audit and fail-closed staged-draw validation.

This validates original CSV/BIN/manifest integrity and one independently
published result anchor; source metadata supplied for *new* draws is not a
substitute for reviewing the source itself. No network calls or file mutation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

SCHEMA = "lotto649.m2.draw_source_audit.v1"
STAGE_SCHEMA = "lotto649.m2.official_source_staging.v1"
ANCHOR = {
    "draw_id": 115000104,
    "date": "2026-11-10",
    "main": [11, 23, 30, 32, 40, 41],
    "bonus": 28,
}
ANCHOR_SOURCE = {
    "publisher": "CTBC Bank Taiwan Lottery results publication",
    "url": "https://lotto.ctbcbank.com/result_all.htm",
    "review_date": "2026-11-11",
    "attestation": "INDEPENDENT_PUBLISHED_RESULT_MANUALLY_REVIEWED_NOT_LIVE_RETRIEVED",
}
PINNED_SOURCE_MANIFEST_SHA256 = "003ed03eda889952654b679859f2d465ffb7349c7f005a6df3aa4436be3d54a3"
PINNED_FIRST_2498_HISTORY_SHA256 = "57de2ead78147f998b7cac22dfbd94461a6af38584ee106d6a1a26c1dd97f3b8"
PINNED_FIRST_2498_IMAGE_SHA256 = "9069034085cfd901405dbfba0f0993c6e1fd8b1494ecea87c20681b52fe808b9"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_draw(record: object) -> dict:
    """Validate strict draw ID, civil date, six main balls and distinct bonus."""
    _require(isinstance(record, dict) and set(record) ==
             {"draw_id", "date", "main", "bonus"}, "unexpected draw fields")
    draw_id, day, main, bonus = (record[k] for k in ("draw_id", "date", "main", "bonus"))
    _require(type(draw_id) is int and draw_id > 0, "draw_id must be positive integer")
    _require(isinstance(day, str), "date must be canonical ISO")
    try:
        parsed = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("invalid Gregorian draw date") from exc
    _require(day == parsed.isoformat() and draw_id // 1000000 == parsed.year - 1911,
             "draw_id/date ROC year mismatch")
    _require(isinstance(main, list) and len(main) == 6 and
             all(type(x) is int and 1 <= x <= 49 for x in main) and
             main == sorted(set(main)), "six main balls must be ascending unique 1..49")
    _require(type(bonus) is int and 1 <= bonus <= 49 and bonus not in main,
             "special ball must be 1..49 and distinct from six main balls")
    return {"draw_id": draw_id, "date": day, "main": main.copy(), "bonus": bonus}


def _from_csv_bytes(relative: Path, raw: bytes) -> list[dict]:
    """Read the *already generated and SHA-checked* canonical annual CSV."""
    year = int(relative.stem.split("_")[-1])
    rows = list(csv.DictReader(StringIO(raw.decode("ascii"), newline="")))
    out = []
    expected_keys = ("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus")
    for row in rows:
        _require(tuple(row) == expected_keys and None not in row.values(),
                 "annual CSV columns differ from nine-field contract")
        normalized = normalize_draw({
            "draw_id": int(row["draw_id"]),
            "date": row["date"].replace("/", "-"),
            "main": [int(row[f"n{i}"]) for i in range(1, 7)],
            "bonus": int(row["bonus"]),
        })
        _require(normalized["draw_id"] // 1000000 == year - 1911,
                 "CSV filename/ROC draw year mismatch")
        out.append(normalized)
    return out


def audit_source(csv_root: Path = Path("data/csv"),
                 bin_root: Path = Path("data/bin")) -> dict:
    """Validate all existing year files, both binary formats and source manifest.

    Archived-source consistency and a publication cross-check are intentionally
    distinct claims: neither automatically certifies all 2004..2026 draws.
    """
    from scripts.export_lotto649_csv import generate_yearly_csv, verify_files

    csv_root, bin_root = Path(csv_root), Path(bin_root)
    generated, year_counts = generate_yearly_csv(bin_root)
    verify_files(csv_root, generated)
    all_draws = [item for relative, raw in sorted(generated.items())
                 for item in _from_csv_bytes(relative, raw)]
    _require(bool(all_draws), "empty draw archive")
    by_id = {}
    previous_id = 0
    previous_date = ""
    for item in all_draws:
        draw_id = item["draw_id"]
        _require(draw_id > previous_id and item["date"] > previous_date,
                 "chronology is not strictly ascending")
        _require(draw_id not in by_id, "duplicate draw ID")
        if previous_id and draw_id // 1000000 == previous_id // 1000000:
            _require(draw_id == previous_id + 1, "missing draw ID within year")
        by_id[draw_id] = item
        previous_id, previous_date = draw_id, item["date"]
    _require(ANCHOR["draw_id"] in by_id, "independent 2026-11-10 anchor absent")
    _require(by_id[ANCHOR["draw_id"]] == ANCHOR,
             "published result anchor contradicts archived CSV/BIN")
    manifest_raw = (bin_root / "lotto649_source_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    _require(len(all_draws) == manifest["total_draws"]
             and all_draws[0]["draw_id"] == manifest["first_draw_id"]
             and all_draws[-1]["draw_id"] == manifest["last_draw_id"],
             "archive and source manifest endpoints do not agree")
    canonical_manifest = manifest_raw.replace(b"\r\n", b"\n")
    _require(b"\r" not in canonical_manifest and
             (b"\r" not in manifest_raw or b"\n" not in manifest_raw.replace(b"\r\n", b"")),
             "source manifest has mixed or invalid line endings")
    fingerprint = sha256(canonical_manifest).hexdigest()
    _require(manifest.get("dataset") == "lotto649.appended-verified.v2",
             "unexpected archive dataset version")
    history_bytes = (bin_root / "lotto649_history_v1.bin").read_bytes()
    image_bytes = (bin_root / "lotto649_images_v2.bin").read_bytes()
    if len(all_draws) == 2498:
        _require(fingerprint == PINNED_SOURCE_MANIFEST_SHA256,
                 "frozen initial 2498-draw manifest differs")
        _require(sha256(history_bytes).hexdigest() == PINNED_FIRST_2498_HISTORY_SHA256
                 and sha256(image_bytes).hexdigest() == PINNED_FIRST_2498_IMAGE_SHA256,
                 "frozen first 2498-draw binary snapshot differs")
    else:
        _require(len(all_draws) > 2498, "archive regressed below fixed shadow cutoff")
        # Pinned source draws must be preserved when later draws are appended.
        from scripts.export_lotto649_csv import HEAD, HISTORY, IMAGE
        import struct
        for raw, row, magic, expected in (
            (history_bytes, HISTORY, b"L649HIS1", PINNED_FIRST_2498_HISTORY_SHA256),
            (image_bytes, IMAGE, b"L649IMG2", PINNED_FIRST_2498_IMAGE_SHA256),
        ):
            prefix = bytearray(raw[:HEAD.size + 2498 * row.size])
            _require(HEAD.unpack_from(prefix)[0] == magic,
                     "original binary version changed")
            struct.pack_into("<I", prefix, 8, 2498)
            _require(sha256(prefix).hexdigest() == expected,
                     "previously sealed 2498-draw prefix was modified")
    return {
        "schema": SCHEMA,
        "status": "PASS_ARCHIVE_INTEGRITY_AND_SINGLE_PUBLISHED_DRAW_CROSSCHECK",
        "source_authenticity_scope": (
            "One independently published CTBC 2026-11-10 draw manually crosschecked; "
            "other historic rows remain source-archive consistent but are NOT individually "
            "certified against official Taiwan Lottery downloads."
        ),
        "total_draws": len(all_draws), "annual_csv_files": len(generated),
        "first_draw_id": all_draws[0]["draw_id"], "last_draw_id": all_draws[-1]["draw_id"],
        "last_date": all_draws[-1]["date"], "crosscheck": {
            "anchor": ANCHOR, "source": ANCHOR_SOURCE, "exact_match": True,
        },
        "source_fingerprints": {
            "manifest_sha256": fingerprint,
            "history_bin_sha256": sha256(history_bytes).hexdigest(),
            "image_bin_sha256": sha256(image_bytes).hexdigest(),
        },
        "annual_counts": {str(year): count for year, count in year_counts.items()},
    }


def validate_staged_batch(stage: object, audit: dict) -> dict:
    """Schema-check future observations; never claim raw web evidence was fetched.

    Raw source SHA values are declarations awaiting independent review.
    This function intentionally does NOT append to CSV/BIN/manifest.
    """
    _require(isinstance(stage, dict)
             and set(stage) == {"schema", "base_manifest_sha256", "records"},
             "unexpected staged bundle fields")
    _require(stage["schema"] == STAGE_SCHEMA, "unsupported staging schema")
    _require(stage["base_manifest_sha256"] ==
             audit["source_fingerprints"]["manifest_sha256"],
             "staged base manifest SHA changed; revalidate before importing")
    rows = stage["records"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= 30,
             "staging requires 1..30 strictly ordered draws")
    current_id = audit["last_draw_id"]
    current_day = date.fromisoformat(audit["last_date"])
    accepted = []
    for entry in rows:
        _require(isinstance(entry, dict) and set(entry) == {"draw", "sources"},
                 "unexpected stage row fields")
        draw = normalize_draw(entry["draw"])
        day = date.fromisoformat(draw["date"])
        _require(day > current_day, "draw date must follow archive cutoff")
        if day.year == current_day.year:
            _require(draw["draw_id"] == current_id + 1,
                     "staged draw gap or duplicate within year")
        else:
            _require(day.year == current_day.year + 1
                     and draw["draw_id"] % 1000000 == 1,
                     "invalid new-year draw ID transition")
        sources = entry["sources"]
        _require(isinstance(sources, list) and len(sources) >= 2,
                 "two separately identified published sources required")
        kinds = set()
        hosts = set()
        for src in sources:
            _require(isinstance(src, dict) and set(src) ==
                     {"publisher", "kind", "url", "page_sha256", "observed_draw"},
                     "source declaration requires strict provenance fields")
            _require(src["kind"] in ("lottery_operator", "licensed_bank", "other_publisher"),
                     "unsupported source kind")
            _require(isinstance(src["publisher"], str) and src["publisher"].strip(),
                     "source publisher missing")
            _require(isinstance(src["url"], str), "source URL must be a string")
            url = urlsplit(src["url"])
            _require(url.scheme == "https" and bool(url.hostname)
                     and not url.username and not url.password,
                     "source evidence must use a valid public HTTPS URL")
            _require(url.hostname not in hosts,
                     "two source records must have distinct published hosts")
            hosts.add(url.hostname)
            _require(type(src["page_sha256"]) is str
                     and _SHA.fullmatch(src["page_sha256"]) is not None,
                     "source page SHA256 must be 64 lowercase hex chars")
            _require(normalize_draw(src["observed_draw"]) == draw,
                     "source declaration disagrees with staged draw")
            kinds.add(src["kind"])
        _require(bool(kinds & {"lottery_operator", "licensed_bank"}),
                 "at least one declared source must be operator or licensed bank")
        accepted.append(draw)
        current_id, current_day = draw["draw_id"], day
    return {
        "status": "PASS_STAGING_SCHEMA_ONLY_SOURCE_PAGES_NOT_FETCHED_OR_AUTHENTICATED",
        "staged_draws": len(accepted), "first_staged_draw": accepted[0],
        "last_staged_draw": accepted[-1],
        "ready_to_mutate_archive": False,
        "next_action": (
            "Independently retrieve and verify original published source pages and "
            "their hashes, then perform a separately authorized append transaction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--stage", type=Path, help="Optional untrusted future draw JSON to validate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    _require(destination.is_relative_to(Path("results").resolve()),
             "M2 audit report must be beneath results/")
    report = audit_source(args.csv_root, args.bin_root)
    if args.stage:
        report["staging"] = validate_staged_batch(
            json.loads(args.stage.read_text(encoding="utf-8")), report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, ensure_ascii=True)
        fp.write("\n")
    print(f"PASS M2: {report['total_draws']} draws / {report['annual_csv_files']} annual CSVs, "
          "both BINs and manifest verified")
    print("SOURCE: #115000104 matches independent published CTBC results")
    if args.stage:
        print("STAGING: SCHEMA_ONLY / NOT LIVE SOURCE VERIFIED / NO DATA MUTATION")
    print(f"REPORT: {destination}")


if __name__ == "__main__":
    main()
