"""Read-only public experiment anchors and write-once private paid-ticket ledger.

The private ledger is a local audit tool, not proof of a pre-draw timestamp.
No lottery outcomes are stored in registration records.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import re

PUBLIC_SCHEMA = "lotto649.public_experiment_registry.v1"
PAID_SCHEMA = "lotto649.private_paid_registration.v1"
ENTRY_SCHEMA = "lotto649.private_ledger_entry.v1"
DEFAULT_REGISTRY = Path("experiments/ledger_registry_v1.json")
DEFAULT_PRIVATE_DIR = Path("data/private/experiment-ledger")
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,79}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_keys(obj: object, keys: set[str], label: str) -> None:
    _require(isinstance(obj, dict) and set(obj) == keys,
             f"{label}: unexpected or missing fields")


def _canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _parse_day(value: object) -> date:
    _require(isinstance(value, str), "date must be ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be ISO YYYY-MM-DD") from exc
    _require(parsed.isoformat() == value, "date must be canonical ISO YYYY-MM-DD")
    return parsed


def validate_paid_registration(record: object) -> None:
    _object_keys(record, {"schema", "experiment_id", "strategy", "registration_status",
                          "purchase_date_reported", "source_evidence", "historical_cutoff",
                          "draw_schedule", "tickets", "disjoint_tickets",
                          "cost_per_line_twd", "total_cost_twd"}, "paid registration")
    _require(record["schema"] == PAID_SCHEMA, "invalid paid registration schema")
    _require(isinstance(record["experiment_id"], str) and
             _ID.fullmatch(record["experiment_id"]) is not None,
             "unsafe experiment_id")
    _require(isinstance(record["strategy"], str) and bool(record["strategy"].strip()),
             "strategy must be nonempty")
    _require(record["registration_status"] == "RETROSPECTIVE_UNSEALED",
             "late registrations cannot claim an independent pre-draw seal")
    _require(record["source_evidence"] in ("USER_REPORTED", "USER_CONFIRMED_PHOTO"),
             "unsupported evidence classification")
    purchase_date = _parse_day(record["purchase_date_reported"])
    cutoff = record["historical_cutoff"]
    _object_keys(cutoff, {"date", "last_draw_id", "draws", "prefix_sha256"},
                 "historical cutoff")
    cutoff_day = _parse_day(cutoff["date"])
    _require(type(cutoff["last_draw_id"]) is int and cutoff["last_draw_id"] > 0 and
             type(cutoff["draws"]) is int and cutoff["draws"] > 0 and
             isinstance(cutoff["prefix_sha256"], str) and
             _SHA256.fullmatch(cutoff["prefix_sha256"]) is not None,
             "invalid historical cutoff anchor")
    _require(cutoff["last_draw_id"] // 1000000 == cutoff_day.year - 1911,
             "historical cutoff draw ID/date mismatch")
    schedule = record["draw_schedule"]
    _require(isinstance(schedule, list) and 1 <= len(schedule) <= 200,
             "invalid draw_schedule length")
    previous_id = cutoff["last_draw_id"]
    previous_day = cutoff_day
    for item in schedule:
        _object_keys(item, {"draw_id", "date"}, "schedule row")
        draw_id = item["draw_id"]
        day = _parse_day(item["date"])
        _require(type(draw_id) is int and draw_id > previous_id and
                 draw_id // 1000000 == day.year - 1911 and day > previous_day,
                 "schedule must be strictly ordered with matching draw IDs and dates")
        if day.year == previous_day.year:
            _require(draw_id == previous_id + 1,
                     "missing draw ID within scheduled year")
        else:
            _require(day.year == previous_day.year + 1 and draw_id % 1000000 == 1,
                     "invalid scheduled year transition")
        previous_id, previous_day = draw_id, day
    _require(purchase_date <= _parse_day(schedule[0]["date"]),
             "reported purchase must not follow the first scheduled draw")
    tickets = record["tickets"]
    _require(isinstance(tickets, list) and 1 <= len(tickets) <= 10,
             "invalid ticket count")
    seen = set()
    for ticket in tickets:
        _require(isinstance(ticket, list) and len(ticket) == 6 and
                 all(type(n) is int and 1 <= n <= 49 for n in ticket) and
                 ticket == sorted(set(ticket)), "invalid six-number ticket")
        _require(tuple(ticket) not in seen, "duplicate ticket")
        seen.add(tuple(ticket))
    _require(type(record["disjoint_tickets"]) is bool, "disjoint_tickets must be boolean")
    if record["disjoint_tickets"]:
        _require(len(set(n for ticket in tickets for n in ticket)) == 6 * len(tickets),
                 "tickets must be pairwise disjoint")
    price = record["cost_per_line_twd"]
    total = record["total_cost_twd"]
    _require(type(price) is int and price > 0 and type(total) is int and
             total == price * len(tickets) * len(schedule), "paid cost mismatch")


def _git_blob_sha1(payload: bytes) -> str:
    return sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def verify_public_registry(registry_path: Path = DEFAULT_REGISTRY) -> int:
    registry_path = Path(registry_path)
    root = registry_path.resolve().parent.parent
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _object_keys(registry, {"schema", "entries"}, "public registry")
    _require(registry["schema"] == PUBLIC_SCHEMA, "invalid public registry schema")
    entries = registry["entries"]
    _require(isinstance(entries, list) and bool(entries), "empty public registry")
    seen_ids = set()
    for item in entries:
        _object_keys(item, {"experiment_id", "role", "artifact", "git_blob_sha1"},
                     "public reference")
        _require(isinstance(item["experiment_id"], str) and item["experiment_id"] not in seen_ids,
                 "duplicate or invalid experiment ID")
        seen_ids.add(item["experiment_id"])
        _require(item["role"] in ("FROZEN_SHADOW", "CANDIDATE_ONLY"),
                 "public registry must not store paid tickets")
        _require(isinstance(item["git_blob_sha1"], str) and
                 _SHA1.fullmatch(item["git_blob_sha1"]) is not None,
                 "invalid Git blob SHA1")
        artifact_name = item["artifact"]
        _require(isinstance(artifact_name, str) and not Path(artifact_name).is_absolute(),
                 "artifact path must be relative")
        artifact = (root / artifact_name).resolve()
        _require(artifact.is_relative_to(root) and artifact.suffix == ".json",
                 "unsafe public artifact path")
        content = artifact.read_bytes()
        normalized = content.replace(b"\r\n", b"\n")
        _require(b"\r" not in normalized, "mixed or invalid public artifact line endings")
        _require(_git_blob_sha1(normalized) == item["git_blob_sha1"],
                 f"public artifact differs from frozen Git blob: {artifact_name}")
        source = json.loads(normalized)
        _require(source.get("experiment_id", source.get("experiment")) == item["experiment_id"],
                 f"experiment ID differs from referenced artifact: {artifact_name}")
        if item["role"] == "FROZEN_SHADOW":
            _require("strategies" in source and "experiment" in source,
                     "shadow registry entry is not a preregistration")
        else:
            _require(source.get("status") == "DESCRIPTIVE_CANDIDATE_ONLY_NOT_A_PAID_TICKET",
                     "candidate reference misclassified")
    return len(entries)


def register_paid(record: dict, directory: Path = DEFAULT_PRIVATE_DIR,
                  *, registered_at_utc: str | None = None) -> tuple[Path, str]:
    validate_paid_registration(record)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Recorded time is an ingestion timestamp, never independent pre-draw evidence.
    recorded = registered_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _require(isinstance(recorded, str) and recorded.endswith("+00:00"),
             "registered_at_utc must be timezone-aware UTC")
    digest = sha256(_canonical(record)).hexdigest()
    entry = {"schema": ENTRY_SCHEMA, "registered_at_utc": recorded,
             "registration_sha256": digest, "registration": record}
    path = directory / (record["experiment_id"] + ".json")
    content = json.dumps(entry, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, digest


def verify_private_ledger(directory: Path = DEFAULT_PRIVATE_DIR) -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    _require(directory.is_dir(), "private ledger path is not a directory")
    files = sorted(directory.iterdir())
    for path in files:
        _require(path.is_file() and path.suffix == ".json" and not path.is_symlink(),
                 "private ledger contains unexpected files")
        entry = json.loads(path.read_text(encoding="utf-8"))
        _object_keys(entry, {"schema", "registered_at_utc", "registration_sha256",
                             "registration"}, "ledger entry")
        _require(entry["schema"] == ENTRY_SCHEMA, "invalid ledger entry schema")
        validate_paid_registration(entry["registration"])
        _require(path.stem == entry["registration"]["experiment_id"],
                 "private ledger filename and experiment ID disagree")
        _require(sha256(_canonical(entry["registration"])).hexdigest() ==
                 entry["registration_sha256"], "private ledger checksum mismatch")
        _require(isinstance(entry["registered_at_utc"], str) and
                 entry["registered_at_utc"].endswith("+00:00"),
                 "invalid ledger ingestion timestamp")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register-paid", help="Register a local private paid ticket, write once")
    reg.add_argument("--input", type=Path, required=True)
    verify = sub.add_parser("verify", help="Verify public anchors and private registrations")
    verify.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    if args.command == "register-paid":
        record = json.loads(args.input.read_text(encoding="utf-8"))
        path, digest = register_paid(record)
        print(f"PASS: private registration written once: {path}; SHA256={digest}")
    else:
        public = verify_public_registry(args.registry)
        private = verify_private_ledger()
        print(f"PASS: {public} public artifact anchors; {private} private paid registrations")
        if private == 0:
            print("NOTE: no private paid tickets registered locally yet")


if __name__ == "__main__":
    main()
