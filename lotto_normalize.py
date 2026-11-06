"""Lotto649 normalized/compact dataset builder v0.0.1.

Input: data/binary/lotto649_YYYY.parquet and lotto649_bonus_YYYY.parquet
Output: data/normalized/*.parquet and *.npy
No network access, and never modifies original ZIP / source Parquet files.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "0.0.1"
SOURCE_DIR = Path("data/binary")
OUTPUT_DIR = Path("data/normalized")

BASE_DTYPE = np.dtype([
    ("draw_id", "<u4"), ("date", "<u4"),
    ("sales", "<u8"), ("tickets", "<u4"), ("prize", "<u8"),
    ("numbers", "u1", (6,)), ("special", "u1"),
], align=False)
BONUS_DTYPE = np.dtype([
    ("draw_id", "<u4"), ("date", "<u4"),
    ("activity_id", "<u2"), ("numbers", "u1", (6,)),
], align=False)


def parse_date(value: str, year: int) -> int:
    match = re.fullmatch(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", value.strip())
    if not match:
        raise ValueError(f"invalid date: {value!r}")
    d = date(*map(int, match.groups()))
    if d.year != year:
        raise ValueError(f"archive year {year} != date {d.isoformat()}")
    return int(d.strftime("%Y%m%d"))


def integer(value, label, low=0, high=2**64 - 1):
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"{label}: not an unsigned integer: {value!r}")
    num = int(text)
    if not low <= num <= high:
        raise ValueError(f"{label}: out of range: {num}")
    return num


def validate_numbers(row, kind, source, line):
    nums = tuple(integer(row[f"field_{i:02d}"], f"{source} row {line} n{i-6}", 1, 49)
                 for i in range(7, 13))
    if len(set(nums)) != 6:
        raise ValueError(f"{source} row {line}: duplicate main numbers: {nums}")
    return nums


def read_year(path: Path, kind: str, activities: set):
    year = int(re.search(r"(20\d{2})\.parquet$", path.name).group(1))
    table = pq.read_table(path)
    d = table.to_pydict()
    expected_fields = 13 if kind == "base" else 12
    for i in range(1, expected_fields + 1):
        if f"field_{i:02d}" not in d:
            raise ValueError(f"{path.name}: missing field_{i:02d}")
    # Prohibit silently dropping meaningful trailing columns.
    for name, values in d.items():
        m = re.fullmatch(r"field_(\d+)", name)
        if m and int(m.group(1)) > expected_fields:
            if any(v is not None and str(v).strip() for v in values):
                raise ValueError(f"{path.name}: nonempty unexpected {name}")
    rows = []
    for pos in range(table.num_rows):
        row = {key: values[pos] for key, values in d.items()}
        loc = f"{path.name} row {pos + 2}"
        draw_id = integer(row["field_02"], loc + " draw_id", 1, 2**32 - 1)
        if draw_id // 1_000_000 != year - 1911:
            raise ValueError(f"{loc}: unexpected ROC year in draw_id={draw_id}")
        dt = parse_date(row["field_03"], year)
        numbers = validate_numbers(row, kind, path.name, pos + 2)
        sales = integer(row["field_04"], loc + " sales")
        tickets = integer(row["field_05"], loc + " tickets", 0, 2**32-1)
        prize = integer(row["field_06"], loc + " prize")
        if kind == "base":
            if str(row["field_01"]).strip() != "大樂透":
                raise ValueError(f"{loc}: unexpected game name {row['field_01']!r}")
            special = integer(row["field_13"], loc + " special", 1, 49)
            if special in numbers:
                raise ValueError(f"{loc}: special overlaps main six")
            rows.append(dict(draw_id=draw_id, date=dt, sales=sales, tickets=tickets,
                             prize=prize, numbers=numbers, special=special))
        else:
            activity = str(row["field_01"]).strip()
            if not activity:
                raise ValueError(f"{loc}: empty activity")
            activities.add(activity)
            rows.append(dict(draw_id=draw_id, date=dt, sales=sales, tickets=tickets,
                             prize=prize, numbers=numbers, activity=activity))
    return rows


def atomic_npy(path, array):
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with tmp.open("wb") as f:
            np.save(f, array, allow_pickle=False)
        with tmp.open("rb") as f:
            check = np.load(f, allow_pickle=False)
        if not np.array_equal(array, check):
            raise ValueError(f"npy round-trip mismatch: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_parquet(path, table):
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        pq.write_table(table, tmp, compression="zstd")
        if not table.equals(pq.read_table(tmp), check_metadata=True):
            raise ValueError(f"Parquet round-trip mismatch: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    print(f"Lotto649 Normalizer v{VERSION}")
    base_files = sorted(SOURCE_DIR.glob("lotto649_????.parquet"))
    bonus_files = sorted(SOURCE_DIR.glob("lotto649_bonus_????.parquet"))
    if len(base_files) != 20 or len(bonus_files) != 16:
        raise ValueError(f"expected 20 base + 16 bonus, got {len(base_files)} + {len(bonus_files)}")

    activity_names = set()
    base = [rec for f in base_files for rec in read_year(f, "base", activity_names)]
    bonus = [rec for f in bonus_files for rec in read_year(f, "bonus", activity_names)]
    if not base or not bonus:
        raise ValueError("empty base or bonus dataset")

    base.sort(key=lambda r: (r["date"], r["draw_id"]))
    bonus.sort(key=lambda r: (r["date"], r["draw_id"], r["activity"], r["numbers"]))
    ids = [r["draw_id"] for r in base]
    dates = [r["date"] for r in base]
    if len(set(ids)) != len(ids) or len(set(dates)) != len(dates):
        raise ValueError("duplicate base draw ID or draw date")

    base_by_id = {r["draw_id"]: r for r in base}
    unmatched = {(r["draw_id"], r["date"]) for r in bonus
                 if r["draw_id"] not in base_by_id
                 or base_by_id[r["draw_id"]]["date"] != r["date"]}
    if unmatched:
        print(f"[WARN] {len(unmatched)} bonus period/date keys not matching a base draw; kept as separate events")

    names = sorted(activity_names)
    activity_ids = {name: i + 1 for i, name in enumerate(names)}
    a = np.empty(len(base), dtype=BASE_DTYPE)
    for i, r in enumerate(base):
        a[i] = (r["draw_id"], r["date"], r["sales"], r["tickets"],
                r["prize"], r["numbers"], r["special"])
    b = np.empty(len(bonus), dtype=BONUS_DTYPE)
    for i, r in enumerate(bonus):
        b[i] = (r["draw_id"], r["date"], activity_ids[r["activity"]], r["numbers"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_npy(OUTPUT_DIR / "lotto649_base.npy", a)
    atomic_npy(OUTPUT_DIR / "lotto649_bonus.npy", b)

    def as_table(records, include_special):
        cols = {
            "draw_id": pa.array([r["draw_id"] for r in records], type=pa.uint32()),
            "date": pa.array([r["date"] for r in records], type=pa.uint32()),
            "sales": pa.array([r["sales"] for r in records], type=pa.uint64()),
            "tickets": pa.array([r["tickets"] for r in records], type=pa.uint32()),
            "prize": pa.array([r["prize"] for r in records], type=pa.uint64()),
        }
        if not include_special:
            cols["activity"] = pa.array([r["activity"] for r in records], type=pa.string())
        for j in range(6):
            cols[f"n{j+1}"] = pa.array([r["numbers"][j] for r in records], type=pa.uint8())
        if include_special:
            cols["special"] = pa.array([r["special"] for r in records], type=pa.uint8())
        return pa.table(cols)

    atomic_parquet(OUTPUT_DIR / "lotto649_base.parquet", as_table(base, True))
    atomic_parquet(OUTPUT_DIR / "lotto649_bonus.parquet", as_table(bonus, False))
    (OUTPUT_DIR / "activities.json").write_text(
        json.dumps({str(activity_ids[n]): n for n in names}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    repeated = sum(count - 1 for count in Counter((r["draw_id"], r["activity"], r["numbers"]) for r in bonus).values() if count > 1)
    print(f"[PASS] base  : {len(a):,} records, {a.dtype.itemsize} bytes/record")
    print(f"[PASS] bonus : {len(b):,} records, {b.dtype.itemsize} bytes/record")
    print(f"[INFO] activity types: {len(names)}")
    print(f"[INFO] repeated bonus combinations (same draw/activity): {repeated}")
    print(f"[DONE] {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
