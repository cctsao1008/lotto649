"""Reject CJK ideographs and punctuation in tracked text and file paths."""
from pathlib import Path
import re
import subprocess
import sys

CJK_TEXT = re.compile(
    r"[\u3000-\u303f\uff01-\uff60\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]"
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    listing = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    failed = False
    for name in listing.decode("utf-8").split("\0"):
        if not name:
            continue
        if CJK_TEXT.search(name):
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
            if CJK_TEXT.search(line):
                print(f"{ascii(name)}:{line_number}: CJK character in tracked text", file=sys.stderr)
                failed = True
                break
    if not failed:
        print("PASS: no CJK ideographs or punctuation in tracked paths or text")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
