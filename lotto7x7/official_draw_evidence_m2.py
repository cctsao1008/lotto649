"""M2 evidence gate: parse two archived source pages for the SAME Lotto649 draw.

Source pages are local immutable HTTP snapshots. Parsing matching content,
TLS capture, and SHA checks are useful audit controls but are not independent
proof that a remote publisher originated the file. The exact page URLs and
retrieval metadata must be preserved for manual review.

Never fetch untrusted URLs, alter original draw files or infer future results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import ssl
from urllib.request import Request, urlopen

from .draw_ingestion_m2 import normalize_draw

SCHEMA = "lotto649.m2.saved_publisher_pages.v1"
TRUSTED = {
    "ctbc": "https://lotto.ctbcbank.com/result_all.htm",
    "independent": "https://api.lottery.com.tw/bl",
}
MAX_PAGE = 2_000_000
HEX = re.compile(r"[0-9a-f]{64}\Z")
TICKET = r"(\d{1,2}(?:\s+\d{1,2}){5})"


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag in ("td", "th", "tr", "br", "div", "p", "li", "span"):
            self.parts.append(" ")

    def handle_endtag(self, tag) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden -= 1
        if tag in ("td", "th", "tr", "br", "div", "p", "li", "span"):
            self.parts.append(" ")

    def handle_data(self, data) -> None:
        if self.hidden == 0:
            self.parts.append(data)


def _text(raw: bytes) -> str:
    if len(raw) > MAX_PAGE or len(raw) < 80:
        raise ValueError("source page exceeds size limit or is too small")
    # Both providers publish Chinese webpages. Reject binary/obfuscated files.
    content = None
    for codec in ("utf-8-sig", "big5", "cp950"):
        try:
            candidate = raw.decode(codec, "strict")
        except UnicodeError:
            continue
        if "大樂透" in candidate:
            content = candidate
            break
    if content is None:
        raise ValueError("source page is not a recognized Chinese Lotto649 publication")
    parser = _VisibleText()
    parser.feed(content)
    return " ".join(" ".join(parser.parts).split())


def parse_ctbc(raw: bytes) -> dict:
    """Only the 大樂透 section, never nearby 威力彩 or 今彩539 numbers."""
    visible = _text(raw)
    marker = re.search(r"遊\s*戲\s*大樂透\s*期\s*別\s*第\s*(\d{9})\s*期", visible)
    if marker is None:
        raise ValueError("CTBC page missing clearly identified 大樂透 issue")
    section = visible[marker.start():]
    stop = re.search(r"遊\s*戲\s*今彩539", section)
    if stop:
        section = section[:stop.start()]
    date_match = re.search(
        r"開獎日期\s*民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        section,
    )
    numbers = re.search(
        r"依大小順序排列\s*[：:]?\s*" + TICKET
        + r"\s*特別號\s*[：:]?\s*(\d{1,2})(?!\d)",
        section,
    )
    if date_match is None or numbers is None:
        raise ValueError("CTBC Lotto649 date, sorted six balls, or bonus unavailable")
    roc, month, day = map(int, date_match.groups())
    return normalize_draw({
        "draw_id": int(marker.group(1)),
        "date": f"{roc + 1911:04d}-{month:02d}-{day:02d}",
        "main": list(map(int, numbers.group(1).split())),
        "bonus": int(numbers.group(2)),
    })


def parse_independent(raw: bytes) -> dict:
    """Parse the first latest draw on the independent lottery.com.tw /bl page."""
    visible = _text(raw)
    # Text format: 115/11/10 115000104頭獎:0 11 23 30 32 40 41 28
    match = re.search(
        r"(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})\s+"
        r"(\d{9})\s*頭獎\s*[:：]\s*\d+\s+"
        + TICKET + r"\s+(\d{1,2})(?!\d)", visible,
    )
    if match is None:
        raise ValueError("independent page missing date/issue/sorted six/bonus row")
    roc, month, day, draw_id = map(int, match.groups()[:4])
    return normalize_draw({
        "draw_id": draw_id,
        "date": f"{roc + 1911:04d}-{month:02d}-{day:02d}",
        "main": list(map(int, match.group(5).split())),
        "bonus": int(match.group(6)),
    })


def _read_one(directory: Path, item: object, provider: str) -> tuple[dict, str]:
    if not isinstance(item, dict) or set(item) != {
        "provider", "url", "sha256", "captured_at_utc"
    }:
        raise ValueError("source snapshot metadata fields do not match contract")
    if item["provider"] != provider or item["url"] != TRUSTED[provider]:
        raise ValueError("source provider URL does not match the pinned HTTPS origin")
    digest = item["sha256"]
    if not isinstance(digest, str) or HEX.fullmatch(digest) is None:
        raise ValueError("source SHA256 is invalid")
    try:
        parsed = datetime.fromisoformat(item["captured_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid capture timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source capture timestamp must specify UTC offset")
    expected_file = directory / (digest + ".html")
    if expected_file.is_symlink():
        raise ValueError("source HTML snapshot cannot be a symlink")
    content = expected_file.read_bytes()
    if sha256(content).hexdigest() != digest:
        raise ValueError("local source snapshot bytes disagree with SHA256")
    draw = parse_ctbc(content) if provider == "ctbc" else parse_independent(content)
    return draw, digest


def verify_local_evidence(draw_id: int, directory: Path) -> dict:
    """Require two different, parseable, exactly matching cached publisher pages."""
    if type(draw_id) is not int or draw_id < 1:
        raise ValueError("invalid requested draw ID")
    directory = Path(directory)
    path = directory / f"{draw_id}.json"
    if path.is_symlink():
        raise ValueError("source index cannot be symlink")
    item = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(item, dict) or
            set(item) != {"schema", "draw", "pages"} or
            item["schema"] != SCHEMA):
        raise ValueError("invalid independent source snapshot index")
    official = normalize_draw(item["draw"])
    if official["draw_id"] != draw_id:
        raise ValueError("snapshot index period does not match expected draw ID")
    pages = item["pages"]
    if not isinstance(pages, list) or len(pages) != 2:
        raise ValueError("exactly two independently published source pages required")
    source_by_provider = {p.get("provider"): p for p in pages if isinstance(p, dict)}
    if set(source_by_provider) != set(TRUSTED):
        raise ValueError("both specific CTBC and independent provider snapshots required")
    if len({p["sha256"] for p in source_by_provider.values()}) != 2:
        raise ValueError("independent source snapshots must have different digests")
    parsed = []
    for provider in TRUSTED:
        data, digest = _read_one(directory, source_by_provider[provider], provider)
        if data != official:
            raise ValueError(f"{provider} source content contradicts indexed draw")
        parsed.append(data)
    if parsed[0] != parsed[1]:
        raise ValueError("two publisher pages do not show the same draw")
    return {
        "status": "PASS_TWO_SAVED_SOURCE_PAGES_CONTENT_AGREES",
        "scope_warning": (
            "Local page SHA and parsing cannot cryptographically prove publisher "
            "origin; provenance still depends on authentic TLS download and archive custody."
        ),
        "draw": official,
        "source_index_sha256": sha256(path.read_bytes()).hexdigest(),
        "page_sha256": {
            provider: source_by_provider[provider]["sha256"] for provider in TRUSTED
        },
    }


def capture_latest(directory: Path) -> dict:
    """Fetch only two pinned HTTPS URLs; write raw immutable pages; do not infer draws."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for provider, url in TRUSTED.items():
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (lotto649-source-audit/1.0)",
                                    "Accept": "text/html"})
        with urlopen(req, timeout=20, context=ssl.create_default_context()) as response:
            if response.geturl() != url:
                raise ValueError("redirect is prohibited for pinned source URL")
            payload = response.read(MAX_PAGE + 1)
        if len(payload) > MAX_PAGE:
            raise ValueError("source page exceeds maximum accepted length")
        digest = sha256(payload).hexdigest()
        name = directory / (digest + ".html")
        with name.open("xb") as f:
            f.write(payload)
        data = parse_ctbc(payload) if provider == "ctbc" else parse_independent(payload)
        saved.append({"provider": provider, "url": url, "sha256": digest,
                      "captured_at_utc": datetime.now(timezone.utc).isoformat(
                          timespec="seconds"), "_draw": data})
    draw = saved[0].pop("_draw")
    if draw != saved[1].pop("_draw"):
        raise ValueError("publishers show different draws; no index created")
    index_path = directory / f"{draw['draw_id']}.json"
    index = {"schema": SCHEMA, "draw": draw, "pages": saved}
    with index_path.open("x", encoding="ascii", newline="\n") as f:
        json.dump(index, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    return verify_local_evidence(draw["draw_id"], directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/private/lotto649-source"))
    parser.add_argument("--capture-latest", action="store_true",
                        help="Perform explicit HTTPS fetch of both pinned source pages")
    parser.add_argument("--verify-draw", type=int, help="Verify existing local source snapshots")
    args = parser.parse_args()
    if args.capture_latest == (args.verify_draw is not None):
        parser.error("choose exactly one of --capture-latest or --verify-draw")
    output = (capture_latest(args.directory) if args.capture_latest else
              verify_local_evidence(args.verify_draw, args.directory))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
