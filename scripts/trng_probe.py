#!/usr/bin/env python3
"""Capture or replay a trng.info research sample using the standard library."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen


BYTE_COUNT = 64
SOURCE_URL = f"https://api.trng.info/random/bytes?length={BYTE_COUNT}"
SCHEMA = "lotto649.trng_probe.v1"


def decode_bytes(value):
    if (not isinstance(value, str) or len(value) != 2 * BYTE_COUNT
            or not re.fullmatch(r"[0-9a-fA-F]+", value)):
        raise ValueError(f"expected exactly {BYTE_COUNT} bytes encoded as hex")
    return bytes.fromhex(value)


def derive(raw):
    # Read the first 18 bits MSB-first. Skip the six unused bits in byte 3.
    coin_bits = "".join(f"{byte:08b}" for byte in raw[:3])[:18]
    coins = [[2 + int(bit) for bit in coin_bits[i:i + 3]] for i in range(0, 18, 3)]
    lines = [sum(throw) for throw in coins]
    moving = [i for i, line in enumerate(lines, start=1) if line in (6, 9)]

    # Partial Fisher-Yates: discard bytes that would bias the modulo mapping.
    balls = list(range(1, 50))
    picked = 0
    bytes_used = 3
    for byte in raw[3:]:
        bytes_used += 1
        remaining = len(balls) - picked
        if byte >= 256 - 256 % remaining:
            continue
        index = picked + byte % remaining
        balls[picked], balls[index] = balls[index], balls[picked]
        picked += 1
        if picked == 24:
            break
    if picked != 24:
        raise ValueError("random byte buffer exhausted during rejection sampling")

    return {
        "coins_bottom_to_top": coins,
        "lines_bottom_to_top": lines,
        "moving_lines": moving,
        "base_yang_bits_bottom_to_top": [line % 2 for line in lines],
        "changed_yang_bits_bottom_to_top": [int(line in (6, 7)) for line in lines],
        "control_tickets": [sorted(balls[i:i + 6]) for i in range(0, 24, 6)],
        "bytes_consumed": bytes_used,
    }


def capture():
    request = Request(SOURCE_URL, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        if response.geturl() != SOURCE_URL:
            raise ValueError("unexpected redirect from the trng.info endpoint")
        payload = json.load(response)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object from trng.info")
    raw = decode_bytes(payload.get("data"))
    return {
        "schema": SCHEMA,
        "source_url": SOURCE_URL,
        "captured_at_utc": captured_at,
        "raw_hex": raw.hex(),
        "raw_sha256": sha256(raw).hexdigest(),
        "result": derive(raw),
    }


def replay(record):
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("unsupported capture format")
    if record.get("source_url") != SOURCE_URL:
        raise ValueError("unexpected capture source URL")
    raw = decode_bytes(record.get("raw_hex"))
    if sha256(raw).hexdigest() != record.get("raw_sha256"):
        raise ValueError("raw bytes do not match the recorded SHA-256")
    if derive(raw) != record.get("result"):
        raise ValueError("replayed result does not match the capture")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capture", help="fetch once and create a new JSON capture").add_argument(
        "path", type=Path, help="output JSON path (must not exist)"
    )
    commands.add_parser("replay", help="verify and replay a saved capture offline").add_argument(
        "path", type=Path, help="saved JSON path"
    )
    args = parser.parse_args()

    try:
        if args.command == "capture":
            if args.path.exists():
                raise FileExistsError(f"capture already exists: {args.path}")
            record = capture()
            report = json.dumps(record, indent=2)
            with args.path.open("x", encoding="utf-8", newline="\n") as output:
                output.write(report + "\n")
        else:
            record = replay(json.loads(args.path.read_text(encoding="utf-8")))
            report = json.dumps(record, indent=2)
    except (OSError, ValueError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
