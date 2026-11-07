"""Versioned, lossless Lotto649 CSV-to-BIN draw archive.

Source columns: draw_id, six main numbers, and special number.
The record excludes the date by design; the source CSV remains authoritative
for dates and sales metadata. Binary records are ordered by draw_id.

Wire format v1:
    16-byte header: <8sII, magic=b'L649BIN1', count, record_bytes=11
    11-byte record: <I7B, draw_id:uint32, main[6]:uint8, special:uint8

All integers are unsigned, little-endian where multi-byte, and unaligned.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import os
import re
import struct
import tempfile

import numpy as np

MAGIC = b"L649BIN1"
HEADER = struct.Struct("<8sII")
RECORD = struct.Struct("<I7B")
DTYPE = np.dtype([
    ("draw_id", "<u4"),
    ("numbers", "u1", (6,)),
    ("special", "u1"),
], align=False)
SOURCE_PATTERN = re.compile(r"^大樂透_\d{4}(?:\d{2}_\d{6})?\.csv$")
REQUIRED = ("遊戲名稱", "期別", "開獎日期", "獎號1", "獎號2", "獎號3",
            "獎號4", "獎號5", "獎號6", "特別號")
DEFAULT_CSV = Path("data/csv")
DEFAULT_BIN = Path("data/bin/lotto649_draws_v1.bin")

if DTYPE.itemsize != RECORD.size:
    raise RuntimeError("binary structured record size differs from schema")


@dataclass(frozen=True)
class Draw:
    draw_id: int
    main: tuple[int, int, int, int, int, int]
    special: int
    date: int
    source: str


def discover_sources(csv_root: Path) -> list[Path]:
    """Discover both normal names and 2018/2020 yyyyMM_yyyyMM variants."""
    root = Path(csv_root)
    sources = sorted(p for p in root.glob("*/*.csv")
                     if p.is_file() and p.parent.name.isdigit()
                     and len(p.parent.name) == 4
                     and SOURCE_PATTERN.fullmatch(p.name))
    if not sources:
        raise ValueError(f"no Lotto649 draw CSV files found under {root}")
    years = sorted({int(p.parent.name) for p in sources})
    expected = list(range(years[0], years[-1] + 1))
    if years != expected:
        raise ValueError(f"missing calendar year folders: {sorted(set(expected)-set(years))}")
    return sources


def _positive_int(text: str, *, label: str, origin: str) -> int:
    try:
        raw = str(text).strip()
        if not raw.isdecimal():
            raise ValueError("non-decimal integer")
        value = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{origin}: invalid {label}: {text!r}") from exc
    return value


def parse_csv(path: Path) -> list[Draw]:
    """Parse columns by header; tolerate BOM and unrelated trailing columns."""
    path = Path(path)
    year = int(path.parent.name)
    results: list[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [field.strip() for field in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"{path}: empty CSV") from exc
        missing = [field for field in REQUIRED if field not in header]
        if missing:
            raise ValueError(f"{path}: missing columns: {missing}")
        positions = {field: header.index(field) for field in REQUIRED}
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            origin = f"{path}:{line_number}"
            if len(row) <= max(positions.values()):
                raise ValueError(f"{origin}: truncated row")
            if row[positions["遊戲名稱"]].strip() != "大樂透":
                raise ValueError(f"{origin}: unexpected game name")
            draw_id = _positive_int(row[positions["期別"]], label="draw_id", origin=origin)
            if not 0 < draw_id <= 0xFFFFFFFF or draw_id // 1000000 != year - 1911:
                raise ValueError(f"{origin}: draw_id/year mismatch or overflow")
            raw_date = row[positions["開獎日期"]].strip()
            try:
                date = datetime.strptime(raw_date, "%Y/%m/%d")
            except ValueError as exc:
                raise ValueError(f"{origin}: invalid draw date {raw_date!r}") from exc
            if date.year != year:
                raise ValueError(f"{origin}: draw date/year mismatch")
            labels = ("獎號1", "獎號2", "獎號3", "獎號4", "獎號5", "獎號6", "特別號")
            nums = tuple(_positive_int(row[positions[key]], label=key, origin=origin)
                         for key in labels)
            if any(not 1 <= x <= 49 for x in nums) or len(set(nums)) != 7:
                raise ValueError(f"{origin}: numbers must be seven distinct values in 1..49")
            results.append(Draw(draw_id, nums[:6], nums[6],
                                int(date.strftime("%Y%m%d")), str(path)))
    if not results:
        raise ValueError(f"{path}: no draw rows")
    return results


def load_csv_directory(root: Path = DEFAULT_CSV) -> tuple[list[Draw], list[Path]]:
    """Read and validate every source; sort by globally increasing draw ID."""
    sources = discover_sources(root)
    draws = [draw for path in sources for draw in parse_csv(path)]
    draws.sort(key=lambda x: x.draw_id)
    for previous, current in zip(draws, draws[1:]):
        if previous.draw_id == current.draw_id:
            raise ValueError(f"duplicate draw ID {current.draw_id}")
        if previous.date >= current.date:
            raise ValueError(f"chronology mismatch near {current.draw_id}")
    return draws, sources


def encode_bin(draws: list[Draw]) -> bytes:
    """Serialize the exact eight requested fields with a versioned header."""
    if not draws:
        raise ValueError("cannot serialize zero draws")
    payload = bytearray(HEADER.pack(MAGIC, len(draws), RECORD.size))
    last = 0
    for entry in draws:
        if entry.draw_id <= last or not 0 < entry.draw_id <= 0xFFFFFFFF:
            raise ValueError("draw IDs must be unique ascending uint32 values")
        numbers = (*entry.main, entry.special)
        if len(entry.main) != 6 or any(not 1 <= n <= 49 for n in numbers) or len(set(numbers)) != 7:
            raise ValueError(f"invalid six-main/special payload at {entry.draw_id}")
        payload += RECORD.pack(entry.draw_id, *numbers)
        last = entry.draw_id
    return bytes(payload)


def decode_bin(payload: bytes) -> np.ndarray:
    """Read and validate all records; do not silently accept damaged BIN data."""
    if len(payload) < HEADER.size:
        raise ValueError("truncated BIN header")
    magic, count, width = HEADER.unpack_from(payload)
    if magic != MAGIC or width != RECORD.size:
        raise ValueError("unsupported BIN magic/version or record size")
    if not count or len(payload) != HEADER.size + count * width:
        raise ValueError("BIN count does not match actual byte length")
    records = np.frombuffer(payload, dtype=DTYPE, count=count, offset=HEADER.size)
    ids = records["draw_id"]
    if np.any(ids == 0) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("BIN draw IDs are not strictly increasing")
    numbers = np.column_stack([records["numbers"], records["special"]])
    if np.any((numbers < 1) | (numbers > 49)) or any(len(set(row.tolist())) != 7 for row in numbers):
        raise ValueError("BIN contains invalid or repeated numbers")
    return records


def build(csv_root: Path = DEFAULT_CSV, out: Path = DEFAULT_BIN) -> dict:
    """Atomically write one canonical combined BIN after read-back verification."""
    draws, sources = load_csv_directory(csv_root)
    payload = encode_bin(draws)
    restored = decode_bin(payload)
    for original, record in zip(draws, restored):
        if (original.draw_id != int(record["draw_id"])
                or original.main != tuple(map(int, record["numbers"]))
                or original.special != int(record["special"])):
            raise AssertionError(f"serialization round-trip failed at {original.draw_id}")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp",
                                         delete=False) as f:
            temp_path = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    reread = decode_bin(path.read_bytes())
    if not np.array_equal(restored, reread):
        raise AssertionError("written BIN failed read-back validation")
    counts = dict(sorted(Counter(str(d.date)[:4] for d in draws).items()))
    return {"file": str(path), "bytes": len(payload), "records": len(draws),
            "source_files": len(sources), "counts_by_year": counts,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "first_draw_id": draws[0].draw_id, "last_draw_id": draws[-1].draw_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lossless Lotto649 CSV to versioned BIN")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--verify", type=Path,
                        help="Validate an existing BIN file without rewriting")
    args = parser.parse_args()
    if args.verify is not None:
        records = decode_bin(args.verify.read_bytes())
        print(f"[PASS] Binary: {args.verify} | records={len(records)} | record_bytes={RECORD.size}")
        print(f"[PASS] First draw ID: {int(records['draw_id'][0])}")
        print(f"[PASS] Last draw ID: {int(records['draw_id'][-1])}")
        return
    result = build(args.csv_root, args.out)
    print(f"[PASS] Source CSV files: {result['source_files']}")
    print(f"[PASS] Parsed draws: {result['records']}")
    for year, count in result["counts_by_year"].items():
        print(f"[YEAR] {year}: {count}")
    print(f"[PASS] Fixed BIN record: {RECORD.size} bytes")
    print(f"[PASS] BIN size: {result['bytes']} bytes")
    print(f"[PASS] SHA256: {result['sha256']}")
    print(f"[DONE] Binary dataset: {Path(result['file']).resolve()}")


if __name__ == "__main__":
    main()
