# Lotto649 Draw Archive BIN v1

This format is a **lossless binary serialization** of the eight requested CSV fields: draw ID, the six main numbers, and the special number. It is **not** a compressed or learned 7x7 bitmap. The six main numbers can be turned into the existing 7x7 bitmap with `lotto7x7.core.encode`. Dates, sales and prize fields remain authoritative only in the original source CSV.

## Wire format

- Encoding: little-endian, no implicit padding, fixed schema.
- 16-byte header: Python `struct.Struct('<8sII')` — ASCII magic `L649BIN1` (8 bytes, includes schema version), unsigned 32-bit number of records, unsigned 32-bit record size (always 11).
- 11-byte record: Python `struct.Struct('<I7B')` — unsigned 32-bit draw ID, six one-byte main numbers in the source CSV order, and one one-byte special number.
- All records ordered by ascending draw ID. The special number must differ from every main number. Number fields must each be 1–49, and the six main numbers must be distinct.
- No date, flags, filler, text, or hidden metadata is stored in a record. No byte alignment is assumed. The header and records are fully deterministic.

### Example

Draw ID `115000001` and main numbers `03,07,16,19,40,42`, special `12` are stored as exactly 11 bytes, matching `struct.pack('<I7B', 115000001, 3, 7, 16, 19, 40, 42, 12)`.

## Source discovery

The canonical source is `data/csv/<year>/大樂透_*.csv`. Discover all year folders (including the 2018/2020 `大樂透_YYYYMM_YYYYMM.csv` variants). Only the exact `大樂透_...csv` pattern is accepted; `大樂透加開獎項` and every other game are excluded. Decode using `utf-8-sig` and match headers by name because some early files have extra trailing columns. Files are never modified or deleted by the converter.

## Build and verify (Windows PowerShell)

```powershell
cd D:\my-github\lotto649
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset --verify data/bin/lotto649_draws_v1.bin
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Default output: `data/bin/lotto649_draws_v1.bin` (one combined file for all included years). To select another output: `--out <path>`. To select another CSV root: `--csv-root <path>`.

Regenerating with unchanged CSV input yields **byte-identical BIN output**. The converter rejects missing year folders, duplicated IDs, non-increasing chronology, invalid dates, truncations, repeated or out-of-range numbers, and malformed headers. It writes atomically and verifies the file after writing.

### Python reading example

```python
from pathlib import Path
from lotto7x7.bin_dataset import decode_bin
from lotto7x7.core import encode

records = decode_bin(Path('data/bin/lotto649_draws_v1.bin').read_bytes())
first = records[0]
image = encode(first['numbers'])
assert image.shape == (7, 7) and image.sum() == 6
```

Use `np.dtype([('draw_id','<u4'),('numbers','u1',(6,)),('special','u1')], align=False)` only after the header and version have been validated; record size is **11** bytes.

## Usage with image learning

Keep BIN conversion separate from training. The BIN archive is a canonical numerical representation, while 7x7 images are a reversible view of the six main numbers. This change does not introduce predictive information or improve a neural model's accuracy by itself. Once the format is verified, a future input adapter can read BIN instead of the old normalized NumPy data, while CSV remains the provenance source.
