#!/usr/bin/env python3
"""Export one minimal Lotto649 CSV per year from validated history/image BINs.

Initial 2497-draw snapshot came from Biga. Later individually corroborated
draws may be appended only with explicit immutable per-draw provenance in the
v2 source manifest; previous original source-page hashes remain unchanged.
The binary carries normalized draw IDs: ROC year * 1_000_000 + sequence.
For 2004, 93000001 is written as 093000001 (nine-character display ID).
Only nine columns: date, ID, six ascending main numbers, and bonus.
No network requests or unverified invented outcomes.
"""
from __future__ import annotations

import argparse
import csv
from io import StringIO
import os
from pathlib import Path
import tempfile
import sys

# Support direct invocation from the scripts directory on a clean interpreter.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lotto649.draw_archive import (
    DEFAULT_BIN, HEAD, HISTORY, IMAGE, read_verified_draws,
    valid_appended_source_provenance,
)


HEADER = ("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus")
DEFAULT_OUT = Path("data/csv")


def generate_yearly_csv(bin_dir: Path = DEFAULT_BIN) -> tuple[dict[Path, bytes], dict[int, int]]:
    """Produce all CSV bytes from the verified archive before any writes."""
    by_year = read_verified_draws(bin_dir)
    output: dict[Path, bytes] = {}
    for year, draws in by_year.items():
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(HEADER)
        for day, draw_id, main, special in draws:
            writer.writerow([day.strftime("%Y/%m/%d"), f"{draw_id:09d}",
                             *(f"{n:02d}" for n in main), f"{special:02d}"])
        rel = Path(f"lotto649_{year}.csv")
        output[rel] = buffer.getvalue().encode("ascii")
    return output, {year: len(draws) for year, draws in by_year.items()}


def verify_files(output_dir: Path, expected: dict[Path, bytes]) -> None:
    existing = {p.relative_to(output_dir) for p in output_dir.rglob("*.csv")}
    if existing != set(expected):
        raise ValueError(f"CSV filename set mismatch: missing={set(expected)-existing}, "
                         f"unexpected={existing-set(expected)}")
    if any(path.is_dir() for path in output_dir.iterdir()):
        raise ValueError("CSV directory must be flat: year subdirectories are not allowed")
    for relative, data in expected.items():
        actual = (output_dir / relative).read_bytes()
        if actual == data:
            continue
        # Git on Windows may check out canonical LF text files using CRLF.
        # Accept only a complete LF-to-CRLF conversion, never modified fields,
        # extra whitespace, mixed line endings, or missing/extra records.
        if actual == data.replace(b"\n", b"\r\n"):
            continue
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
                  and p.name == f"\u5927\u6a02\u900f_{p.parent.name}.csv"]
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
