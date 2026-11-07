#!/usr/bin/env python3
"""Biga-only complete Lotto649 (2004..2026) -> BIN v1 + legacy 7x7 BIN v2.

Crawls 25 paginated result pages (one site, one data source).
Fails without writing files if page count, draw continuity or ball integrity fail.
Uses only Python standard library.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import time
from urllib.request import Request, urlopen

BASE = "https://2019.biga.com.tw/ARCHIVE/DRAWDATA/PAGINATION/ZP/biglottoresultlist?page={page}"
PAGES = range(1, 26)
HEAD = struct.Struct("<8sII")
FULL = struct.Struct("<II7B")
IMAGE = struct.Struct("<I49B")

# Each Biga record: date, weekday, ROC draw ID, calendar week, 1..20 lunar
# month ordinal, two Chinese characters, drawn six, bonus, sorted six.
PATTERN = re.compile(
    r"(?<!\d)(?P<day>20\d{2}/\d{2}/\d{2})\s+"
    r"星期[一二三四五六日天]\s+"
    r"(?P<issue>\d{6}|\d{9})\s+\d{6}\s+\d{1,2}\s+[\u4e00-\u9fff]{2}\s+"
    r"(?P<order>(?:\d{1,2}\s+){6})"
    r"(?P<bonus>\d{1,2})\s+"
    r"(?P<main>(?:\d{1,2}\s+){5}\d{1,2})(?!\d)"
)

class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"): self.hidden += 1
        if tag in ("td", "th", "tr", "br", "div", "p", "li"): self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"): self.hidden -= 1
        if tag in ("td", "th", "tr", "br", "div", "p", "li"): self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden: self.parts.extend((data, " "))

    def text(self):
        return " ".join("".join(self.parts).split())

@dataclass(frozen=True)
class Draw:
    source_issue: str
    draw_id: int
    day: date
    main: tuple[int, ...]
    bonus: int

def parse_page(raw: bytes) -> list[Draw]:
    decoded = None
    for enc in ("utf-8-sig", "big5", "cp950"):
        try: candidate = raw.decode(enc)
        except UnicodeError: continue
        if "大樂透開獎號碼一覽表" in candidate and "落球順序" in candidate:
            decoded = candidate
            break
    if decoded is None:
        raise ValueError("unexpected encoding, error page, or source HTML structure")
    page = Text()
    page.feed(decoded)
    matches = list(PATTERN.finditer(page.text()))
    if not matches:
        raise ValueError("zero valid Biga lottery table rows; HTML format changed")
    draws = []
    for match in matches:
        day = date.fromisoformat(match["day"].replace("/", "-"))
        issue = match["issue"]
        roc = day.year - 1911
        if not re.fullmatch(rf"{roc:03d}(?:\d{{3}}|\d{{6}})", issue):
            raise ValueError(f"draw/year mismatch: {issue} on {day}")
        seq = int(issue[3:])
        main = tuple(map(int, match["main"].split()))
        order = tuple(map(int, match["order"].split()))
        bonus = int(match["bonus"])
        if seq == 0 or len(main) != 6 or tuple(sorted(set(main))) != main:
            raise ValueError(f"invalid main numbers in {issue}")
        if sorted(order) != list(main):
            raise ValueError(f"ball order disagrees with sorted numbers in {issue}")
        if not all(1 <= x <= 49 for x in (*main, bonus)) or bonus in main:
            raise ValueError(f"invalid main/bonus range in {issue}")
        draws.append(Draw(issue, roc * 1000000 + seq, day, main, bonus))
    return draws

def fetch_page(page: int) -> tuple[int, bytes, list[Draw]]:
    url = BASE.format(page=page)
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Lotto649Research/1.0)",
                                         "Accept": "text/html"})
            with urlopen(req, timeout=30) as response: raw = response.read()
            result = parse_page(raw)
            return page, raw, result
        except Exception:
            if attempt == 2: raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")

def build_binary(draws: list[Draw], which: str) -> bytes:
    if which == "history":
        rows = b"".join(FULL.pack(d.draw_id, int(d.day.strftime("%Y%m%d")),
                                  *d.main, d.bonus) for d in draws)
        magic, width = b"L649HIS1", FULL.size
    else:
        def image_row(d):
            bits = bytearray(49)
            for number in d.main: bits[number - 1] = 1
            return IMAGE.pack(d.draw_id, *bits)
        rows = b"".join(image_row(d) for d in draws)
        magic, width = b"L649IMG2", IMAGE.size
    encoded = HEAD.pack(magic, len(draws), width) + rows
    assert HEAD.unpack_from(encoded) == (magic, len(draws), width)
    assert len(encoded) == HEAD.size + len(draws) * width
    return encoded

def rebuild(out: Path, workers: int, cutoff: date, expected_count: int):
    records = {}
    provenance = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(fetch_page, n): n for n in PAGES}
        for task in as_completed(tasks):
            page, raw, items = task.result()
            records[page] = items
            provenance[str(page)] = {
                "url": BASE.format(page=page),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(items)
            }
            print(f"[OK] page {page:02d}/25 -> {len(items)} rows", flush=True)
    if set(records) != set(PAGES):
        raise ValueError(f"pages missing: {sorted(set(PAGES) - set(records))}")
    all_draws = [d for items in records.values() for d in items if d.day <= cutoff]
    unique = {}
    for d in all_draws:
        if d.draw_id in unique and unique[d.draw_id] != d:
            raise ValueError(f"conflicting duplicates for draw {d.draw_id}")
        unique[d.draw_id] = d
    if len(all_draws) != len(unique):
        raise ValueError(f"duplicate draw IDs across pages: {len(all_draws) - len(unique)}")
    draws = sorted(unique.values(), key=lambda d: d.draw_id)
    if len(draws) != expected_count:
        raise ValueError(f"expected {expected_count} draws, got {len(draws)}")
    if draws[0].day != date(2004, 1, 5) or draws[-1].day != cutoff:
        raise ValueError(f"first/last date wrong: {draws[0].day}, {draws[-1].day}")
    if any(a.draw_id >= b.draw_id or a.day >= b.day for a, b in zip(draws, draws[1:])):
        raise ValueError("global IDs or dates are not chronologically increasing")
    by_year = {}
    for d in draws: by_year.setdefault(d.day.year, []).append(d)
    if set(by_year) != set(range(2004, 2027)):
        raise ValueError("year coverage must be exactly 2004 through 2026")
    for year, items in sorted(by_year.items()):
        seq = [d.draw_id % 1000000 for d in items]
        if seq != list(range(1, len(items) + 1)):
            raise ValueError(f"missing/duplicate draw IDs in year {year}")
        print(f"[YEAR] {year}: {len(items)} draws", flush=True)

    payloads = {
        "lotto649_history_v1.bin": build_binary(draws, "history"),
        "lotto649_images_v2.bin": build_binary(draws, "image")
    }
    manifest = {
        "dataset": "lotto649.single-source.biga.v1",
        "source_site": "https://2019.biga.com.tw/",
        "cutoff": str(cutoff),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_draws": len(draws),
        "first_draw_id": draws[0].draw_id,
        "last_draw_id": draws[-1].draw_id,
        "id_rule": "ROC year * 1,000,000 + annual sequence; raw 6-digit source IDs expanded",
        "history_format": "L649HIS1, <II7B: id, YYYYMMDD, 6 sorted numbers, bonus",
        "image_format": "L649IMG2, <I49B: id and six-hot 7x7 bitmap",
        "source_pages": dict(sorted(provenance.items(), key=lambda pair: int(pair[0]))),
        "by_year": {str(y): len(v) for y, v in sorted(by_year.items())},
        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in payloads.items()},
        "limitations": "One third-party source, not independently cross-checked against official historical draws."
    }
    payloads["lotto649_source_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        temp = out / (name + ".tmp")
        temp.write_bytes(content)
        temp.replace(out / name)
    print(f"[DONE] {len(draws)} validated Lotto649 draws -> {out}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/bin"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--until", type=date.fromisoformat, default=date(2026, 11, 6))
    ap.add_argument("--expect-count", type=int, default=2497)
    args = ap.parse_args()
    if not 1 <= args.workers <= 6: ap.error("--workers must be 1..6")
    rebuild(args.out_dir, args.workers, args.until, args.expect_count)
