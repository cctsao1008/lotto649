#!/usr/bin/env python3
"""AUZO-only 2004–2026 Lotto649 fetch and BIN rebuild; standard library only."""
from __future__ import annotations
import argparse, hashlib, json, re, struct, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://lotto.auzo.com.tw/biglotto/list_{year}_all.html"
HEAD = struct.Struct("<8sII")
HISTORY_ROW = struct.Struct("<II7B")
IMAGE_ROW = struct.Struct("<I49B")
PAT = re.compile(
    r"(?<!\d)(?P<issue>\d{6}|\d{9})(?!\d)\s+"
    r"(?P<day>20\d{2}-\d{2}-\d{2})\s+\(星期[一二三四五六日天]\)\s*"
    r"落球順序\s*[:：]\s*(?P<order>(?:\d{1,2}\s+){5}\d{1,2})\s+"
    r"大小順序\s*[:：]\s*(?P<main>(?:\d{1,2}\s+){5}\d{1,2})\s+"
    r"(?P<bonus>\d{1,2})\s*\$\s*(?P<sales>[\d,]+)"
)

class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"): self.hidden += 1
        if tag in ("td", "tr", "th", "br", "div", "p", "li"): self.parts.append(" ")
    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"): self.hidden -= 1
        if tag in ("td", "tr", "th", "br", "div", "p", "li"): self.parts.append(" ")
    def handle_data(self, data):
        if not self.hidden: self.parts.extend((data, " "))
    def text(self):
        return " ".join("".join(self.parts).split())

@dataclass(frozen=True)
class Draw:
    original_issue: str
    id: int
    day: date
    main: tuple[int, ...]
    bonus: int
    sales: int

def parse_year(year: int, raw: bytes) -> list[Draw]:
    decoded = None
    for codec in ("utf-8-sig", "big5", "cp950"):
        try: candidate = raw.decode(codec)
        except UnicodeError: continue
        if "落球順序" in candidate and "大小順序" in candidate:
            decoded = candidate
            break
    if decoded is None: raise ValueError(f"{year}: invalid encoding / missing expected labels")
    page = Text()
    page.feed(decoded)
    matches = list(PAT.finditer(page.text()))
    if not matches: raise ValueError(f"{year}: no draws parsed (HTML structure changed?)")
    draws = []
    roc = year - 1911
    for match in matches:
        issue = match["issue"]
        if not re.fullmatch(rf"{roc:03d}(?:\d{{3}}|\d{{6}})", issue):
            raise ValueError(f"{year}: invalid issue {issue}")
        seq = int(issue[3:])
        day = date.fromisoformat(match["day"])
        main = tuple(map(int, match["main"].split()))
        order = tuple(map(int, match["order"].split()))
        bonus = int(match["bonus"])
        if seq == 0 or day.year != year: raise ValueError(f"{year}: invalid issue/date {issue}")
        if len(main) != 6 or tuple(sorted(set(main))) != main or sorted(order) != list(main):
            raise ValueError(f"{year}: malformed numbers in {issue}")
        if not all(1 <= n <= 49 for n in (*main, bonus)) or bonus in main:
            raise ValueError(f"{year}: invalid special number in {issue}")
        draws.append(Draw(issue, roc * 1000000 + seq, day, main, bonus,
                          int(match["sales"].replace(",", ""))))
    draws.sort(key=lambda d: d.id)
    if [d.id % 1000000 for d in draws] != list(range(1, len(draws) + 1)):
        raise ValueError(f"{year}: missing or repeated draw IDs")
    if any(a.day >= b.day for a, b in zip(draws, draws[1:])):
        raise ValueError(f"{year}: nonascending dates")
    return draws

def fetch(year: int):
    url = BASE.format(year=year)
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Lotto649Research/1.0)",
                                             "Accept": "text/html"})
            with urlopen(request, timeout=30) as response: raw = response.read()
            return year, raw, parse_year(year, raw)
        except Exception:
            if attempt == 2: raise
            time.sleep(1.5 * (attempt + 1))

def binary(draws: list[Draw], kind: str) -> bytes:
    if kind == "history":
        rows = b"".join(HISTORY_ROW.pack(d.id, int(d.day.strftime("%Y%m%d")),
                                         *d.main, d.bonus) for d in draws)
        magic, width = b"L649HIS1", HISTORY_ROW.size
    else:
        def image(d):
            bits = bytearray(49)
            for n in d.main: bits[n - 1] = 1
            return IMAGE_ROW.pack(d.id, *bits)
        rows = b"".join(image(d) for d in draws)
        magic, width = b"L649IMG2", IMAGE_ROW.size
    result = HEAD.pack(magic, len(draws), width) + rows
    assert len(result) == HEAD.size + len(draws) * width
    assert HEAD.unpack_from(result) == (magic, len(draws), width)
    return result

def rebuild(out: Path, workers: int, cutoff: date, count: int):
    years, origins = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(fetch, year): year for year in range(2004, 2027)}
        for task in as_completed(tasks):
            year, raw, draws = task.result()
            years[year] = draws
            origins[str(year)] = {
                "url": BASE.format(year=year), "page_sha256": hashlib.sha256(raw).hexdigest(),
                "page_count": len(draws), "first_issue": draws[0].original_issue,
                "last_issue": draws[-1].original_issue,
                "suspicious_sales_2147483647": sum(d.sales == 2147483647 for d in draws)
            }
            print(f"[OK] {year}: {len(draws)} records", flush=True)
    if set(years) != set(range(2004, 2027)): raise ValueError("missing a year")
    for year, draws in years.items():
        if len(draws) < 100: raise ValueError(f"{year} has too few records: {len(draws)}")
        if year < 2007 and len(draws) != 104: raise ValueError(f"{year} should contain 104 draws")
    draws = sorted((d for records in years.values() for d in records if d.day <= cutoff),
                   key=lambda d: d.id)
    if len(draws) != count: raise ValueError(f"expected {count} draws, parsed {len(draws)}")
    if draws[0].day != date(2004, 1, 5) or draws[-1].day != cutoff:
        raise ValueError("date anchors failed")
    if any(a.id >= b.id or a.day >= b.day for a, b in zip(draws, draws[1:])):
        raise ValueError("duplicate or unordered draw ID/date")
    files = {
        "lotto649_history_v1.bin": binary(draws, "history"),
        "lotto649_images_v2.bin": binary(draws, "image")
    }
    manifest = {
        "format": "lotto649.auzo.single-source.v1",
        "source": "https://lotto.auzo.com.tw/",
        "as_of": cutoff.isoformat(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "draw_count": len(draws),
        "first_draw_id": draws[0].id, "last_draw_id": draws[-1].id,
        "history_record_format": "<II7B: normalized ID, YYYYMMDD, 6 sorted numbers, bonus",
        "legacy_image_record_format": "<I49B: normalized ID, 49 pixels, six set",
        "note": "Six-digit 2004–2006 issues map to 9-digit ROC*1000000 + seq; "
                "sales excluded because source contains suspected saturated values. "
                "Single third-party source; not official/independently verified.",
        "files": {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                  for name, value in files.items()},
        "years": dict(sorted(origins.items()))
    }
    files["lotto649_source_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    for filename, contents in files.items():
        temp = out / (filename + ".tmp")
        temp.write_bytes(contents)
        temp.replace(out / filename)
    print(f"[DONE] {len(draws)} validated draws; BIN files saved under {out}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/bin"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--until", type=date.fromisoformat, default=date(2026, 11, 6))
    ap.add_argument("--expect-count", type=int, default=2497)
    args = ap.parse_args()
    if not 1 <= args.workers <= 6: ap.error("--workers must be between 1 and 6")
    rebuild(args.out_dir, args.workers, args.until, args.expect_count)
