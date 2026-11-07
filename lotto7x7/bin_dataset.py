"""Canonical Lotto649 CSV to 7x7 six-pixel BIN v2 converter.

CSV remains the source of truth. Each record stores only draw_id and a
sequentially mapped 7x7 six-pixel binary mask. No bonus number or date is
stored in this file; the ROC year can be recovered from the draw ID.

Header (16 bytes): <8sII  magic=L649IMG2, count, record_bytes=53
Record (53 bytes): <I49B  draw_id:uint32 and 49 raster uint8 cells
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np

from .core import decode, encode

MAGIC = b"L649IMG2"
HEADER = struct.Struct("<8sII")
RECORD = struct.Struct("<I49B")
DTYPE = np.dtype([("draw_id", "<u4"), ("image", "u1", (7, 7))], align=False)
SOURCE_PATTERN = re.compile(r"^大樂透_\d{4}(?:\d{2}_\d{6})?\.csv$")
REQUIRED = ("期別", "開獎日期", "獎號1", "獎號2", "獎號3",
            "獎號4", "獎號5", "獎號6")
DEFAULT_CSV = Path("data/csv")
DEFAULT_BIN = Path("data/bin/lotto649_images_v2.bin")

if HEADER.size != 16 or RECORD.size != 53 or DTYPE.itemsize != RECORD.size:
    raise RuntimeError("BIN v2 schema has an unexpected byte size")


@dataclass(frozen=True)
class Draw:
    draw_id: int
    main: tuple[int, int, int, int, int, int]
    date: int
    source: str


def discover_sources(csv_root: Path = DEFAULT_CSV) -> list[Path]:
    """Discover annual Lotto649 CSVs, including 2018 and 2020 range names."""
    root = Path(csv_root)
    sources = sorted(p for p in root.glob("*/*.csv")
                     if p.is_file() and p.parent.name.isdecimal()
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
    raw = str(text).strip()
    if not raw.isdecimal():
        raise ValueError(f"{origin}: invalid {label}: {text!r}")
    return int(raw)


def parse_csv(path: Path) -> list[Draw]:
    """Validate source fields by header, tolerating BOM and trailing columns.

    The special-number column is deliberately ignored. It is never encoded
    into this six-pixel dataset and is not a training label.
    """
    path = Path(path)
    if not path.parent.name.isdecimal():
        raise ValueError(f"{path}: numeric year directory required")
    year = int(path.parent.name)
    result: list[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"{path}: empty CSV") from exc
        missing = [key for key in REQUIRED if key not in header]
        if missing:
            raise ValueError(f"{path}: missing columns: {missing}")
        columns = {key: header.index(key) for key in REQUIRED}
        for lineno, row in enumerate(reader, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            origin = f"{path}:{lineno}"
            if len(row) <= max(columns.values()):
                raise ValueError(f"{origin}: truncated row")
            if "遊戲名稱" in header and row[header.index("遊戲名稱")].strip() != "大樂透":
                raise ValueError(f"{origin}: unexpected game name")
            draw_id = _positive_int(row[columns["期別"]],
                                    label="draw_id", origin=origin)
            if not 0 < draw_id <= 0xFFFFFFFF or draw_id // 1000000 != year - 1911:
                raise ValueError(f"{origin}: draw_id/year mismatch or overflow")
            date_text = row[columns["開獎日期"]].strip()
            try:
                date = datetime.strptime(date_text, "%Y/%m/%d")
            except ValueError as exc:
                raise ValueError(f"{origin}: invalid draw date {date_text!r}") from exc
            if date.year != year:
                raise ValueError(f"{origin}: draw date/year mismatch")
            main = tuple(_positive_int(row[columns[f"獎號{i}"]],
                                       label=f"main_{i}", origin=origin)
                         for i in range(1, 7))
            if any(n < 1 or n > 49 for n in main) or len(set(main)) != 6:
                raise ValueError(f"{origin}: six main numbers must be distinct in 1..49")
            result.append(Draw(draw_id, main,
                               int(date.strftime("%Y%m%d")), str(path)))
    if not result:
        raise ValueError(f"{path}: no draw records")
    return result


def load_csv_directory(csv_root: Path = DEFAULT_CSV) -> tuple[list[Draw], list[Path]]:
    """Read every annual file and verify globally unique IDs and chronology."""
    sources = discover_sources(csv_root)
    draws = sorted((r for path in sources for r in parse_csv(path)),
                   key=lambda r: r.draw_id)
    for left, right in zip(draws, draws[1:]):
        if left.draw_id == right.draw_id:
            raise ValueError(f"duplicate draw ID {right.draw_id}")
        if left.date >= right.date:
            raise ValueError(f"chronology mismatch near draw {right.draw_id}")
    return draws, sources


def encode_bin(draws: list[Draw]) -> bytes:
    """Serialize sequential 7x7 raster images as 53-byte fixed-width records."""
    if not draws:
        raise ValueError("cannot serialize zero draws")
    out = bytearray(HEADER.pack(MAGIC, len(draws), RECORD.size))
    last_id = 0
    for row in draws:
        if (not 0 < row.draw_id <= 0xFFFFFFFF or row.draw_id <= last_id
                or row.draw_id // 1000000 < 1
                or len(row.main) != 6):
            raise ValueError("draw IDs must be valid ascending uint32 values")
        mask = encode(np.asarray(row.main, dtype=np.int64), mapping="sequential")
        out.extend(RECORD.pack(row.draw_id, *mask.ravel().tolist()))
        last_id = row.draw_id
    return bytes(out)


def decode_bin(payload: bytes) -> np.ndarray:
    """Reject incompatible header versions, malformed masks or damaged sizes."""
    data = memoryview(payload)
    if len(data) < HEADER.size:
        raise ValueError("truncated BIN header")
    magic, count, width = HEADER.unpack_from(data)
    if magic != MAGIC or width != RECORD.size:
        raise ValueError("unsupported BIN v2 magic or record width")
    if not count or len(data) != HEADER.size + count * width:
        raise ValueError("BIN record count does not match byte length")
    records = np.frombuffer(data, dtype=DTYPE, count=count, offset=HEADER.size)
    draw_ids = records["draw_id"].astype(np.int64)
    if np.any(draw_ids < 1_000_000) or np.any(np.diff(draw_ids) <= 0):
        raise ValueError("invalid, duplicated or unordered BIN draw IDs")
    pixels = records["image"]
    if np.any(pixels > 1) or np.any(pixels.sum(axis=(1, 2)) != 6):
        raise ValueError("each BIN image must have exactly six binary active pixels")
    return records


def verify(path: Path = DEFAULT_BIN, csv_root: Path = DEFAULT_CSV) -> dict:
    """Check both the wire format and byte equality to fresh CSV conversion."""
    content = Path(path).read_bytes()
    records = decode_bin(content)
    source_draws, sources = load_csv_directory(csv_root)
    regenerated = encode_bin(source_draws)
    if content != regenerated:
        raise ValueError("BIN differs from original CSV-derived six-point masks")
    return {
        "records": len(records), "bytes": len(content),
        "source_files": len(sources),
        "first_draw_id": int(records["draw_id"][0]),
        "last_draw_id": int(records["draw_id"][-1]),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build(csv_root: Path = DEFAULT_CSV, out: Path = DEFAULT_BIN) -> dict:
    """Atomically generate, read back and CSV-verify the canonical BIN file."""
    draws, sources = load_csv_directory(csv_root)
    content = encode_bin(draws)
    restored = decode_bin(content)
    if len(restored) != len(draws):
        raise AssertionError("binary roundtrip changed number of draws")
    for row, decoded in zip(draws, restored):
        if row.draw_id != int(decoded["draw_id"]) or set(row.main) != set(
                decode(decoded["image"], mapping="sequential")):
            raise AssertionError(f"binary roundtrip mismatch at {row.draw_id}")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=path.name+".",
                suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    validated = verify(path, csv_root)
    validated["counts_by_year"] = dict(sorted(Counter(
        str(x.date)[:4] for x in draws).items()))
    validated["file"] = str(path)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify 7x7 Lotto649 BIN v2 from canonical CSV files")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--verify", type=Path,
                        help="Compare an existing BIN v2 byte-for-byte against original CSVs")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify, args.csv_root)
        print(f"[PASS] BIN v2: records={result['records']} bytes={result['bytes']}")
        print(f"[PASS] CSV source files: {result['source_files']}")
        print(f"[PASS] CSV parity and SHA256: {result['sha256']}")
        return
    result = build(args.csv_root, args.out)
    print("Lotto649 7x7 Image BIN v2")
    print(f"[PASS] CSV sources: {result['source_files']}")
    print(f"[PASS] Draw images: {result['records']}")
    print(f"[PASS] Record size: {RECORD.size} bytes")
    print(f"[PASS] Total size: {result['bytes']} bytes")
    print(f"[PASS] SHA256: {result['sha256']}")
    print(f"[DONE] BIN: {Path(result['file']).resolve()}")


if __name__ == "__main__":
    main()
