"""Lotto649 v0.2 synthetic controls: IID, biased, noisy periodic, exact cycle, drift.

Synthetic truth is stored for laboratory testing only. Never confuse this
with historical data or allow ground truth into a blind model's inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from lotto_probability import FixedCardinalityDistribution

VERSION = "0.1.0"
MODES = ("iid", "fixed_bias", "periodic_bias", "deterministic_cycle", "change_point")


def generate(*, count, mode, seed, period=52, strength=2.0):
    if (count < 2 or period < 2 or not np.isfinite(strength) or strength < 0):
        raise ValueError("require count>=2, period>=2, finite strength>=0")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    rng = np.random.default_rng(seed)
    n, k = 49, 6
    if mode == "deterministic_cycle":
        if period > count:
            raise ValueError("for deterministic_cycle, period must be <= count")
        unique = set()
        base = []
        for _ in range(period):
            while True:
                s = tuple(sorted(map(int, rng.choice(np.arange(1, n + 1), k, replace=False))))
                if s not in unique:
                    unique.add(s)
                    base.append(s)
                    break
        phases = np.arange(count, dtype=np.int32) % period
        return np.asarray([base[int(p)] for p in phases], dtype=np.uint8), phases, np.zeros((period, n))

    if mode == "iid":
        phases = np.zeros(count, dtype=np.int32)
        logits = np.zeros((1, n))
    elif mode == "fixed_bias":
        phases = np.zeros(count, dtype=np.int32)
        logits = np.zeros((1, n))
        logits[0, rng.choice(n, 12, replace=False)] = strength
    elif mode == "periodic_bias":
        phases = np.arange(count, dtype=np.int32) % period
        logits = np.zeros((period, n))
        for phase in range(period):
            logits[phase, rng.choice(n, 12, replace=False)] = strength
    else:
        phases = (np.arange(count) >= count // 2).astype(np.int32)
        logits = np.zeros((2, n))
        for phase in range(2):
            logits[phase, rng.choice(n, 12, replace=False)] = strength

    distributions = [FixedCardinalityDistribution(weights, k) for weights in logits]
    numbers = np.asarray([distributions[int(p)].sample(rng) for p in phases], dtype=np.uint8)
    return numbers, phases, logits


def main():
    parser = argparse.ArgumentParser(description="Generate 6/49 artificial positive and negative controls")
    parser.add_argument("--mode", choices=MODES, default="iid")
    parser.add_argument("--count", type=int, default=2183)
    parser.add_argument("--period", type=int, default=52)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    numbers, phases, logits = generate(count=args.count, mode=args.mode,
                                       seed=args.seed, period=args.period, strength=args.strength)
    path = args.out or Path("data/synthetic") / f"{args.mode}_n{args.count}_seed{args.seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"version": VERSION, "mode": args.mode, "count": args.count,
                "period": args.period, "strength": args.strength, "seed": args.seed,
                "warning": "Synthetic ground truth only; NEVER use phases/logits as blind-model inputs"}
    np.savez_compressed(path, numbers=numbers, phases=phases, logits=logits,
                        metadata=np.asarray(json.dumps(metadata)))
    unique = len(set(map(tuple, numbers.tolist())))
    print(f"Lotto649 Synthetic Generator v{VERSION}")
    print(f"[PASS] {args.mode}: {len(numbers):,} legal draws, {unique:,} distinct sets")
    print(f"[DONE] {path.resolve()}")
    print("[INFO] Artificial laboratory data; official records unchanged")


if __name__ == "__main__":
    main()
