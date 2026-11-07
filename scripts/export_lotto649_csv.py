#!/usr/bin/env python3
"""Export one minimal Lotto649 CSV per year from the validated Biga history BIN.

Single upstream source: Biga -> history_v1.bin -> annual CSVs.
The binary carries normalized draw IDs: ROC year * 1_000_000 + sequence.
For 2004, 93000001 is written as 093000001 (nine-character display ID).
Only nine columns are emitted: date, ID, six ascending main numbers, bonus.
No new network requests or secondary sources.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, date
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import struct
import tempfile

HEAD = struct.Struct("<8sII")
HISTORY = struct.Struct("<II7B")
IMAGE = struct.Struct("<I49B")
HEADER = ("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus")
DEFAULT_BIN = Path("data/bin")
DEFAULT_OUT = Path("data/csv")


def _checked_binary(path: Path, *, magic: bytes, record: struct.Struct,
                    sha256: str) -> memoryview:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 differs from source manifest")
    if len(payload) < HEAD.size:
        raise ValueError(f"{path}: truncated header")
    stored_magic, count, record_bytes = HEAD.unpack_from(payload)
    if stored_magic != magic or record_bytes != record.size:
        raise ValueError(f"{path}: unsupported binary version or record width")
    if count < 1 or len(payload) != HEAD.size + count * record_bytes:
        raise ValueError(f"{path}: invalid record count or byte length")
    return memoryview(payload)


def generate_yearly_csv(bin_dir: Path = DEFAULT_BIN) -> tuple[dict[Path, bytes], dict[int, int]]:
    """Validate both BINs and produce all CSV bytes in memory before any writes."""
    bin_dir = Path(bin_dir)
    manifest = json.loads((bin_dir / "lotto649_source_manifest.json").read_text("utf-8"))
    if manifest.get("dataset") != "lotto649.single-source.biga.v1":
        raise ValueError("unexpected source dataset; expected Biga single-source archive")
    files = manifest["files"]
    history = _checked_binary(
        bin_dir / "lotto649_history_v1.bin",
        magic=b"L649HIS1", record=HISTORY,
        sha256=files["lotto649_history_v1.bin"]["sha256"]
    )
    images = _checked_binary(
        bin_dir / "lotto649_images_v2.bin",
        magic=b"L649IMG2", record=IMAGE,
        sha256=files["lotto649_images_v2.bin"]["sha256"]
    )
    count = HEAD.unpack_from(history)[1]
    if count != HEAD.unpack_from(images)[1] or count != manifest["total_draws"]:
        raise ValueError("history / image / manifest draw counts disagree")
    by_year: dict[int, list[tuple[date, int, tuple[int, ...], int]]] = defaultdict(list)
    previous_id, previous_date = 0, None

    for index in range(count):
        h = HISTORY.unpack_from(history, HEAD.size + index * HISTORY.size)
        image = IMAGE.unpack_from(images, HEAD.size + index * IMAGE.size)
        draw_id, yyyymmdd, *balls = h
        main, special = tuple(balls[:6]), balls[6]
        try:
            draw_date = datetime.strptime(str(yyyymmdd), "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"invalid date {yyyymmdd} at record {index}") from exc
        year = draw_date.year
        roc = year - 1911
        if not 2004 <= year <= 2026 or draw_id // 1000000 != roc:
            raise ValueError(f"draw ID / date year mismatch: {draw_id} / {yyyymmdd}")
        if draw_id <= previous_id or (previous_date is not None and draw_date <= previous_date):
            raise ValueError(f"duplicate or unordered draw: {draw_id}")
        if len(main) != 6 or tuple(sorted(set(main))) != main or any(
            not 1 <= n <= 49 for n in (*main, special)
        ) or special in main:
            raise ValueError(f"invalid main/special numbers at {draw_id}")
        if image[0] != draw_id or tuple(image[1:]) != tuple(
            1 if n in main else 0 for n in range(1, 50)
        ):
            raise ValueError(f"history/image BIN mismatch at draw {draw_id}")
        by_year[year].append((draw_date, draw_id, main, special))
        previous_id, previous_date = draw_id, draw_date

    actual_counts = {year: len(rows) for year, rows in sorted(by_year.items())}
    expected_counts = {int(year): int(n) for year, n in manifest["by_year"].items()}
    if actual_counts != expected_counts:
        raise ValueError(f"annual counts disagree with manifest: {actual_counts}")
    if (previous_id != manifest["last_draw_id"] or
            next(iter(by_year.values()))[0][1] != manifest["first_draw_id"]):
        raise ValueError("first/last draw ID differs from manifest")

    output: dict[Path, bytes] = {}
    for year, draws in sorted(by_year.items()):
        sequence = [draw_id % 1000000 for _, draw_id, _, _ in draws]
        if sequence != list(range(1, len(draws) + 1)):
            raise ValueError(f"missing or duplicate annual sequence: {year}")
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(HEADER)
        for day, draw_id, main, special in draws:
            writer.writerow([day.strftime("%Y/%m/%d"), f"{draw_id:09d}",
                             *(f"{n:02d}" for n in main), f"{special:02d}"])
        rel = Path(f"lotto649_{year}.csv")
        output[rel] = buffer.getvalue().encode("ascii")
    return output, actual_counts


def verify_files(output_dir: Path, expected: dict[Path, bytes]) -> None:
    existing = {p.relative_to(output_dir) for p in output_dir.rglob("*.csv")}
    if existing != set(expected):
        raise ValueError(f"CSV filename set mismatch: missing={set(expected)-existing}, "
                         f"unexpected={existing-set(expected)}")
    if any(path.is_dir() for path in output_dir.iterdir()):
        raise ValueError("CSV directory must be flat: year subdirectories are not allowed")
    for relative, data in expected.items():
        if (output_dir / relative).read_bytes() != data:
            raise ValueError(f"CSV content differs from history BIN: {relative}")


def export(bin_dir: Path = DEFAULT_BIN, output_dir: Path = DEFAULT_OUT,
           *, verify: bool = False) -> dict[int, int]:
    expected, counts = generate_yearly_csv(bin_dir)
    output_dir = Path(output_dir)
    if verify:
        verify_files(output_dir, expected)
    else:
        # Migrate only the known previous-year directory layout; never remove unknown files.
        nested = list(output_dir.glob("*/*.csv"))
        legacy = [p for p in nested if p.parent.name.isdecimal()
                  and len(p.parent.name) == 4
                  and p.name == f"大樂透_{p.parent.name}.csv"]
        if len(legacy) != len(nested):
            raise ValueError("unexpected nested CSV: refusing destructive migration")
        for relative, contents in expected.items():
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                                 prefix=destination.name + ".",
                                                 suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(contents)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        for previous in legacy:
            previous.unlink()
        for folder in sorted({p.parent for p in legacy}):
            folder.rmdir()
        verify_files(output_dir, expected)
    print(f"PASS: {sum(counts.values())} Lotto649 draws, "
          f"{len(counts)} annual CSVs, matching history and image BINs", flush=True)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing CSVs without writing")
    args = parser.parse_args()
    export(args.bin_dir, args.out_dir, verify=args.verify)
