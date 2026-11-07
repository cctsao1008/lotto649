# Lotto7x7 canonical image BIN v2

The data source of truth is the repository's original `data/csv/<year>/大樂透_*.csv` files. The canonical machine-learning input is **`data/bin/lotto649_images_v2.bin`**, a fully deterministic binary archive containing **only the official period identifier and the six-main-number 7×7 image**. The special number is deliberately excluded, as is the exact draw date.

## On-disk specification

All records use the fixed **sequential mapping** from number 1 through 49 to row-major raster positions `0..48`. Pixel values are `0` (not drawn) or `1` (drawn), with exactly six `1`s per frame. Neither a seventh special-number pixel nor a duplicate six-number list is present.

| Offset | Width | Type | Meaning |
|---:|---:|---|---|
| 0 | 8 bytes | ASCII | File magic: `L649IMG2` |
| 8 | 4 bytes | little-endian `uint32` | Number of draws |
| 12 | 4 bytes | little-endian `uint32` | Record size, always **53** |
| 16 + 53×i | 4 bytes | little-endian `uint32` | Official draw ID (`期別`) |
| 20 + 53×i | 49 bytes | `uint8[7][7]` | Six-on 7×7 image, row-major |

Header size: **16 bytes**. Record size: **53 bytes**. For the 2,183-draw 2007–2026 source snapshot (ending 2026-10-30), file size is **115,715 bytes** (16 + 2,183×53).

The converter includes `struct.Struct('<8sII')` for the header, `struct.Struct('<I49B')` for a record, and NumPy's **unaligned** dtype:

```python
import numpy as np

DTYPE = np.dtype([
    ('draw_id', '<u4'),
    ('image', 'u1', (7, 7)),
], align=False)
assert DTYPE.itemsize == 53
```

There is no padding, appended checksum, filename, game name, special number or date in a record. The file format is **not** a compressed tensor: 49 raw pixels are stored directly so the CNN can receive the array with a `uint8 → float32` conversion.

### Date and year handling

The original date is available **only from CSV**. To determine the calendar year for train/test reports and year filters without inventing a date, use the official Taiwanese ROC-year draw ID:

`Gregorian_year = draw_id // 1_000_000 + 1911`.

A BIN-based report uses `date: null` and a separate `year` field. A HTML gallery shows `Year 2023` rather than an invented `20230000` or a fake day.

Because a bitmap is a set of activated positions, it preserves the six drawn numbers **as a set**. Original ordering of the six CSV columns is not preserved, intentionally. For any `image`, use the same sequential mapping to recover six ascending numbers with `lotto7x7.core.decode(image)`.

## Dataset generation and verification

The converter discovers all annual CSV names for `大樂透`, including 2018's and 2020's `大樂透_YYYYMM_YYYYMM.csv` naming. It ignores unrelated games and additional-prize files. It parses by column header with `utf-8-sig`, accepts unused trailing fields and unpadded `2024/12/3`, validates game name, official ID, date chronology, six distinct main numbers in 1–49, and enforces strictly increasing period IDs. The special number is never read into the dataset.

```powershell
cd D:\my-github\lotto649

# Regenerate the canonical BIN from all 20 original CSV files.
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset

# Verify the exact file format and byte-for-byte equality to original CSVs.
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset `
    --verify data/bin/lotto649_images_v2.bin

git diff --exit-code -- data/bin/lotto649_images_v2.bin

# New image-focused unit and integration tests.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The writer uses an atomic temporary-file replacement and verifies the file after writing. Invalid masks (anything other than exactly six `1` pixels), duplicate or non-monotonic IDs, incorrect header version, truncated records, and unsupported widths are rejected; no malformed record is silently fixed.

### Model input

The default value of `lotto7x7.research.DATA` is now `data/bin/lotto649_images_v2.bin`.

```python
from lotto7x7.research import load_history
images, dates, draw_ids = load_history()
assert images.dtype == 'uint8'
assert images.shape[1:] == (7, 7)
assert not dates.any()  # Exact dates were not stored in this BIN.
```

`lotto7x7 learn`, `replay`, `check` and `gallery` consume the same loaded six-pixel masks. The old numerical `.npy` archive remains an **optional explicit comparison** input, not the default training dataset. The BIN contains no information that was not already contained in the six main numbers: format simplification cannot, on its own, improve prediction or establish any lottery edge.

## Version isolation

BIN v1 (`L649BIN1`, 11-byte records) and v2 (`L649IMG2`, 53-byte records) are **not compatible**. The decoder checks the magic and record width; it rejects v1 instead of guessing. Only v2 is included in the new feature branch. The 20 original CSV files remain unmodified as the authoritative backup and provenance.
