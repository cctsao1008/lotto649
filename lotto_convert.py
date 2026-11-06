
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "0.0.4"

RAW_DIR = Path("data/raw")
BIN_DIR = Path("data/binary")

# Supports:
# 大樂透_2017.csv
# 大樂透_201801_201812.csv
# 大樂透加開獎項_2017.csv
# 大樂透加開獎項_201801_201812.csv
CSV_PATTERN = re.compile(
    r"^(大樂透|大樂透加開獎項)_"
    r"(20\d{2}|20\d{4}_20\d{4})\.csv$",
    re.IGNORECASE,
)

TYPES = {
    "大樂透": "base",
    "大樂透加開獎項": "bonus",
}


def decode_name(info):
    name = info.filename

    if info.flag_bits & 0x800:
        return name

    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name

    for enc in ("utf-8", "cp950", "big5"):
        try:
            result = raw.decode(enc)
            if "大樂透" in result:
                return result
        except UnicodeDecodeError:
            continue

    return name


def parse_csv(data):
    for enc in ("utf-8-sig", "cp950", "big5"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unsupported CSV encoding")

    records = list(
        csv.reader(io.StringIO(text, newline=""))
    )

    # Ignore physically empty lines.
    records = [r for r in records if r]

    if len(records) < 2:
        raise ValueError("Missing header or data rows")

    header = records[0]
    rows = records[1:]

    widths = Counter(len(r) for r in rows)
    max_width = max(
        len(header),
        max(len(r) for r in rows),
    )

    names = [
        f"field_{i + 1:02d}"
        for i in range(max_width)
    ]

    original_widths = [len(r) for r in rows]

    padded = [
        r + [""] * (max_width - len(r))
        for r in rows
    ]

    return (
        names, padded, original_widths,
        header, widths, enc,
    )


def convert(year, kind, source, data):
    (
        names, rows, original_widths,
        header, widths, encoding,
    ) = parse_csv(data)

    if kind == "base":
        filename = f"lotto649_{year}.parquet"
    else:
        filename = f"lotto649_bonus_{year}.parquet"

    output = BIN_DIR / filename
    temporary = output.with_suffix(".parquet.part")

    columns = {
        name: pa.array(
            [row[i] for row in rows],
            type=pa.string(),
        )
        for i, name in enumerate(names)
    }

    columns["__source_field_count"] = pa.array(
        original_widths,
        type=pa.int32(),
    )

    table = pa.table(columns)

    metadata = {
        b"source_filename": source.encode("utf-8"),
        b"source_year": str(year).encode(),
        b"lotto_type": kind.encode(),
        b"csv_encoding": encoding.encode(),
        b"original_header": json.dumps(
            header, ensure_ascii=False
        ).encode("utf-8"),
        b"row_widths": json.dumps(
            dict(widths)
        ).encode("utf-8"),
    }

    table = table.replace_schema_metadata(metadata)

    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
        )

        check = pq.read_table(temporary)

        if not table.equals(
            check, check_metadata=True
        ):
            raise ValueError("Arrow round-trip mismatch")

        restored = check.to_pylist()

        # Verify every original field.
        for idx, original in enumerate(rows):
            result = restored[idx]

            if result["__source_field_count"] != original_widths[idx]:
                raise ValueError(
                    f"Row width mismatch: {idx + 2}"
                )

            for col, expected in zip(names, original):
                if result[col] != expected:
                    raise ValueError(
                        f"Value mismatch: row {idx + 2}, {col}"
                    )

        temporary.replace(output)

    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"[PASS] {year} {kind}: "
        f"{len(rows)} rows, "
        f"{output.stat().st_size:,} bytes"
    )

    if len(header) != max(widths) or len(widths) != 1:
        print(
            f"       [WARN] Schema: "
            f"header={len(header)}, "
            f"rows={dict(widths)}"
        )

    return len(rows)


def process_zip(path):
    match = re.search(
        r"(20\d{2})\.zip$",
        path.name,
    )

    if not match:
        raise ValueError("Unknown ZIP year")

    year = int(match.group(1))

    found = {"base": [], "bonus": []}

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            decoded = decode_name(info)
            basename = decoded.replace(
                "\\", "/"
            ).split("/")[-1]

            match = CSV_PATTERN.fullmatch(basename)
            if not match:
                continue

            kind = TYPES[match.group(1)]
            found[kind].append((info, decoded))

        results = {}
        errors = {}

        for kind in ("base", "bonus"):
            matches = found[kind]

            if not matches:
                if kind == "base":
                    errors[kind] = "Base CSV missing"
                else:
                    print(
                        f"[INFO] {year}: "
                        "No bonus CSV in archive"
                    )
                continue

            if len(matches) != 1:
                errors[kind] = (
                    f"Expected one {kind} CSV, "
                    f"found {len(matches)}"
                )
                continue

            info, source = matches[0]

            print(f"[INFO] {path.name}")
            print(f"       Type: {kind}")
            print(f"       Source: {source}")

            try:
                results[kind] = convert(
                    year,
                    kind,
                    source,
                    zf.read(info),
                )
            except Exception as exc:
                errors[kind] = str(exc)

        return results, errors


def main():
    print(f"Lotto649 Binary Converter v{VERSION}")

    BIN_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.zip"))

    if not files:
        print("[FAIL] No ZIP files found")
        return

    stats = {
        "base": {"files": 0, "rows": 0},
        "bonus": {"files": 0, "rows": 0},
    }

    failed = 0

    for path in files:
        try:
            results, errors = process_zip(path)

            for kind, count in results.items():
                stats[kind]["files"] += 1
                stats[kind]["rows"] += count

            for kind, reason in errors.items():
                failed += 1
                print(
                    f"[FAIL] {path.name} "
                    f"{kind}: {reason}"
                )

        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {exc}")

    print("\n========== SUMMARY ==========")
    print(f"ZIP scanned : {len(files)}")

    for kind in ("base", "bonus"):
        print(
            f"{kind:5s} : "
            f"{stats[kind]['files']} files, "
            f"{stats[kind]['rows']} rows"
        )

    print(f"Failures    : {failed}")
    print(f"Output      : {BIN_DIR.resolve()}")


if __name__ == "__main__":
    main()
