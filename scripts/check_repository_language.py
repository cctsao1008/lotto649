"""Reject CJK ideographs in tracked text and file paths."""
from pathlib import Path
import re
import subprocess
import sys

IDEOGRAPH = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]"
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    listing = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    failed = False
    for name in listing.decode("utf-8").split("\0"):
        if not name:
            continue
        if IDEOGRAPH.search(name):
            print(f"Non-English path: {ascii(name)}", file=sys.stderr)
            failed = True
        raw = (root / name).read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        elif b"\0" not in raw:
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if IDEOGRAPH.search(line):
                print(f"{ascii(name)}:{line_number}: CJK ideograph in tracked text", file=sys.stderr)
                failed = True
                break
    if not failed:
        print("PASS: no CJK ideographs in tracked paths or text")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
