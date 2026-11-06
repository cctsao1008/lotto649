
import json
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq

VERSION = "0.0.1"

BIN_DIR = Path("data/binary")
REPORT = Path("data/schema_report.txt")

SAMPLE_YEARS = {2007, 2011, 2018, 2023, 2024, 2026}


def decode_metadata(schema):
    metadata = schema.metadata or {}

    result = {}

    for key, value in metadata.items():
        k = key.decode("utf-8")
        v = value.decode("utf-8")

        if k in ("original_header", "row_widths"):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass

        result[k] = v

    return result


def analyze(path):
    table = pq.read_table(path)
    metadata = decode_metadata(table.schema)

    rows = table.num_rows
    columns = table.column_names

    lines = []

    lines.append("=" * 72)
    lines.append(f"FILE: {path.name}")
    lines.append(f"ROWS: {rows}")
    lines.append(
        f"TYPE: {metadata.get('lotto_type', 'unknown')}"
    )
    lines.append(
        f"SOURCE: {metadata.get('source_filename', 'unknown')}"
    )
    lines.append(
        f"HEADER: {metadata.get('original_header', [])}"
    )
    lines.append(
        f"ROW WIDTHS: {metadata.get('row_widths', {})}"
    )
    lines.append("")

    data = table.to_pydict()

    for index, name in enumerate(columns):
        if name == "__source_field_count":
            continue

        values = data[name]

        nonempty = [
            v for v in values
            if v is not None and str(v).strip()
        ]

        distinct = len(set(nonempty))
        examples = list(dict.fromkeys(nonempty))[:5]

        lines.append(
            f"{name:10s} "
            f"nonempty={len(nonempty):6d}/{rows:<6d} "
            f"distinct={distinct:<6d} "
            f"examples={examples}"
        )

    year = int(metadata.get("source_year", 0))

    if year in SAMPLE_YEARS:
        lines.append("")
        lines.append("SAMPLE ROWS:")

        for i in range(min(3, rows)):
            lines.append(f"  Row {i + 1}:")
            for col in columns:
                lines.append(
                    f"    {col}: {data[col][i]!r}"
                )

    return "\n".join(lines)


def main():
    print(f"Lotto649 Schema Discovery v{VERSION}")

    paths = sorted(BIN_DIR.glob("*.parquet"))

    if not paths:
        print("[FAIL] No Parquet files found")
        return

    sections = []
    failures = 0

    for path in paths:
        try:
            sections.append(analyze(path))
            print(f"[PASS] {path.name}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {path.name}: {exc}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )

    print()
    print("========== SUMMARY ==========")
    print(f"Files scanned : {len(paths)}")
    print(f"Failures      : {failures}")
    print(f"Report        : {REPORT.resolve()}")


if __name__ == "__main__":
    main()
