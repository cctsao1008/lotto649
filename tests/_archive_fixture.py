"""Reconstruct the pinned 2,498-draw source used by the M2/M4 tests."""
import json
from pathlib import Path
import struct

from scripts.export_lotto649_csv import HEAD, HISTORY, IMAGE, generate_yearly_csv

ROOT = Path(__file__).resolve().parent


def pre_p5a_archive(directory):
    bin_root = directory / "bin"
    csv_root = directory / "csv"
    bin_root.mkdir()
    csv_root.mkdir()
    manifest = (ROOT / "fixtures/archive_115000104/manifest.json").read_bytes()
    count = json.loads(manifest)["total_draws"]
    (bin_root / "lotto649_source_manifest.json").write_bytes(manifest)
    for filename, record in (
        ("lotto649_history_v1.bin", HISTORY),
        ("lotto649_images_v2.bin", IMAGE),
    ):
        raw = bytearray((ROOT.parent / "data/bin" / filename).read_bytes()
                        [:HEAD.size + count * record.size])
        struct.pack_into("<I", raw, 8, count)
        (bin_root / filename).write_bytes(raw)
    generated, _ = generate_yearly_csv(bin_root)
    for filename, raw in generated.items():
        (csv_root / filename).write_bytes(raw)
    return csv_root, bin_root
