"""Run the selection pipeline for one target date."""
import argparse
from datetime import date
import json
from pathlib import Path

from .pipeline import ROOT, SelectionConfig, load_history, select


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m selection select", description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="target date (YYYY-MM-DD)")
    parser.add_argument("--config", type=Path, default=ROOT / "config/strategy.json")
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "data/bin")
    parser.add_argument("--out", type=Path, help="optional new JSON report")
    args = parser.parse_args(argv)
    try:
        config = SelectionConfig.load(args.config)
        report = select(load_history(args.bin_dir), args.date, config)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="\n") as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
