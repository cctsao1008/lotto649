from pathlib import Path
from zipfile import ZipFile

RAW = Path("data/raw")
OUT = Path("data/csv")

for zip_path in sorted(RAW.glob("*.zip")):
    year = zip_path.stem.split("_")[-1]
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".csv"):
                continue

            name = info.filename
            if not (info.flag_bits & 0x800):
                try:
                    encoded = name.encode("cp437")
                    for enc in ("utf-8", "cp950", "big5"):
                        try:
                            decoded = encoded.decode(enc)
                            if any(c in decoded for c in ("樂透", "彩", "539")):
                                name = decoded
                                break
                        except UnicodeDecodeError:
                            continue
                except UnicodeEncodeError:
                    pass

            target = OUT / year / Path(name.replace("\\", "/")).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            print(f"[PASS] {target}")