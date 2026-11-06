"""Phase 7C: within-calendar-year time-order permutation audit for 7x7 images.

The primary predeclared statistic is lag-one cosine correlation of standardized
27-dimensional image signatures. Null samples shuffle order *within each
calendar year and evaluation segment*, preserving yearly image frequencies.
Only a strictly historical training prefix is used for normalization.

Historical draws have already been inspected, so empirical permutation tails
are exploratory diagnostics, not independent prospective significance tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_replay import (
    SOURCE, TRAIN_END, VALIDATION_END, load_archive, split_indices,
)
from lotto_pattern_similarity import signature

VERSION = "0.1.0"
LAGS = (1, 2, 3, 4, 5, 10)
DEFAULT_PERMUTATIONS = 99


def normalized_signatures(images: np.ndarray, *, train_end: int) -> np.ndarray:
    """Fit standardization exclusively to initial training images."""
    arr = np.asarray(images)
    if (arr.ndim != 3 or arr.shape[1:] != (7, 7)
            or not isinstance(train_end, int)
            or not 12 <= train_end <= len(arr)):
        raise ValueError("invalid image matrix or train boundary")
    feats = np.stack([signature(img) for img in arr])
    train = feats[:train_end]
    scale = np.maximum(train.std(axis=0), 0.05)
    z = (feats - train.mean(axis=0)) / scale
    norms = np.linalg.norm(z, axis=1)
    return z / np.maximum(norms[:, None], 1e-12)


def lag_correlation(z: np.ndarray, years: np.ndarray, *,
                    start: int, end: int, lag: int) -> tuple[float, int]:
    """Compare target and past signatures within one year and one segment."""
    if (z.ndim != 2 or years.shape != (len(z),)
            or not isinstance(start, int) or not isinstance(end, int)
            or not isinstance(lag, int) or not 0 <= start < end <= len(z)
            or not 1 <= lag < end - start):
        raise ValueError("invalid time-lag correlation range")
    positions = np.arange(start + lag, end)
    positions = positions[years[positions] == years[positions - lag]]
    if not len(positions):
        raise ValueError("no within-year pairs for requested lag")
    cosines = np.sum(z[positions] * z[positions - lag], axis=1)
    return float(cosines.mean()), len(positions)


def shuffle_within_year(z: np.ndarray, years: np.ndarray, *,
                        start: int, end: int, rng: np.random.Generator) -> np.ndarray:
    """Randomize time ordering without changing any year's multiset of images."""
    if (z.ndim != 2 or years.shape != (len(z),)
            or not 0 <= start < end <= len(z)):
        raise ValueError("invalid shuffle boundary")
    shuffled = z.copy()
    for year in np.unique(years[start:end]):
        positions = np.flatnonzero(years[start:end] == year) + start
        shuffled[positions] = z[positions[rng.permutation(len(positions))]]
    return shuffled


def permutation_audit(images: np.ndarray, dates: np.ndarray, *,
                      train_end: int, segments: dict[str, tuple[int, int]],
                      lags: tuple[int, ...] = LAGS,
                      permutations: int = DEFAULT_PERMUTATIONS,
                      seed: int = 649) -> dict:
    """Return observed lag correlations against within-year shuffled controls."""
    if (permutations < 2 or seed < 0 or not lags
            or any(not isinstance(x, int) or x < 1 for x in lags)):
        raise ValueError("invalid permutation configuration")
    years = np.asarray(dates, dtype=np.int64) // 10000
    if years.shape != (len(images),):
        raise ValueError("one date per image is required")
    z = normalized_signatures(images, train_end=train_end)
    rng = np.random.default_rng(seed)
    result = {}
    for name, (start, end) in segments.items():
        observed = {}
        for lag in lags:
            observed[lag] = lag_correlation(z, years, start=start, end=end, lag=lag)
        null = {lag: [] for lag in lags}
        for _ in range(permutations):
            shuffled = shuffle_within_year(
                z, years, start=start, end=end, rng=rng)
            for lag in lags:
                score, _ = lag_correlation(
                    shuffled, years, start=start, end=end, lag=lag)
                null[lag].append(score)
        result[name] = {
            "indices": [start, end - 1],
            "years": sorted(set(years[start:end].astype(int).tolist())),
            "lags": {
                str(lag): {
                    "pairs": observed[lag][1],
                    "observed_mean_cosine": observed[lag][0],
                    "shuffled_mean_cosine": float(np.mean(null[lag])),
                    "shuffled_std": float(np.std(null[lag], ddof=1)),
                    "shuffled_95_percent_range": [
                        float(x) for x in np.quantile(null[lag], [0.025, 0.975])
                    ],
                    "observed_minus_shuffled_mean": float(
                        observed[lag][0] - np.mean(null[lag])),
                    "empirical_upper_tail": float((
                        1 + np.count_nonzero(
                            np.asarray(null[lag]) >= observed[lag][0]))
                        / (permutations + 1)),
                    "empirical_two_sided_tail": float((
                        1 + np.count_nonzero(
                            np.abs(np.asarray(null[lag]) - np.mean(null[lag]))
                            >= abs(observed[lag][0] - np.mean(null[lag]))))
                        / (permutations + 1)),
                } for lag in lags
            },
        }
    return {
        "schema": "lotto649.phase7c.temporal_permutation.v1",
        "version": VERSION,
        "status": "EXPLORATORY_HISTORICAL_NOT_PROSPECTIVE",
        "primary_statistic": "two-sided departure of train-standardized lag1 cosine from within-year shuffle null",
        "null": "independent_within_calendar_year_order_permutations",
        "seed": seed, "permutations": permutations,
        "lags": list(lags),
        "train_normalization_end_exclusive": train_end,
        "segments": result,
        "warnings": [
            "Within-year shuffle assumes exchangeability within each calendar year.",
            "Other lags are multiple exploratory comparisons, not independent discoveries.",
            "The original archive was already examined, so empirical tail estimates are not blind evidence.",
            "Signature correlation is not itself an image-forecasting improvement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7C image-sequence time-order permutation audit")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--mapping", choices=("sequential", "snake", "permuted"),
                        default="sequential")
    parser.add_argument("--mapping-seed", type=int, default=649)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--validation-end", type=int, default=VALIDATION_END)
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase7c_temporal_audit.json"))
    args = parser.parse_args()
    numbers, dates, _ = load_archive(args.source)
    train, validation, test = split_indices(
        len(numbers), window=5, train_end=args.train_end,
        validation_end=args.validation_end)
    images = encode_draws(numbers, mapping=args.mapping, seed=args.mapping_seed)
    # Use the entire training prefix, including its first five historical draws.
    report = permutation_audit(
        images, dates, train_end=args.train_end,
        segments={
            "training": (0, args.train_end),
            "validation": (int(validation[0]), int(validation[-1]) + 1),
            "historical_test": (int(test[0]), int(test[-1]) + 1),
        },
        permutations=args.permutations, seed=args.seed)
    report["source"] = str(args.source)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    report["mapping"] = args.mapping
    report["mapping_seed"] = args.mapping_seed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Lotto649 Phase 7C Temporal Permutation Audit v{VERSION}")
    print("Segment             Pairs    Lag1 cosine    Shuffled mean    Two-sided tail")
    for name, block in report["segments"].items():
        r = block["lags"]["1"]
        print(f"{name:20s} {r['pairs']:5d} "
              f"{r['observed_mean_cosine']:+13.6f} "
              f"{r['shuffled_mean_cosine']:+16.6f} "
              f"{r['empirical_two_sided_tail']:15.4f}")
    print(f"[DONE] JSON: {args.report.resolve()}")
    print("[CAUTION] Previously explored historical data; tails are exploratory only")


if __name__ == "__main__":
    main()
