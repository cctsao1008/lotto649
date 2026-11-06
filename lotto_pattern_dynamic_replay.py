"""Phase 7C: causal dynamic-image replay, temporal shuffles and IID controls.

Main objective: six-point 7x7 pattern similarity above the conditional IID
expectation for each proposed image. Six-number exact matches are not an
optimization target. Prior neural/logit models are absent from this pipeline.

The original historical archive was inspected in earlier phases: all scores
are retrospective diagnostics, never new prospective evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lotto_image_repr import decode_images, encode_draws
from lotto_pattern_dynamics import (
    METHODS, DynamicPatternLibrary, diversity,
)
from lotto_pattern_replay import (
    SOURCE, TRAIN_END, VALIDATION_END, load_archive, split_indices,
)
from lotto_pattern_similarity import (
    WEIGHTS, conditional_chance_similarity, image_key,
    random_six_images, similarity,
)
from lotto_pattern_temporal_audit import permutation_audit

VERSION = "0.1.0"
PRIMARY = "mean_chance_adjusted_composite"
DEFAULT_SEED = 649


def image_rows(image: np.ndarray) -> list[str]:
    """Seven printable rows, each with exactly seven raster cells."""
    return ["".join("#" if v else "." for v in row) for row in image]


def bootstrap_mean_ci(values: np.ndarray, *, seed: int,
                      repeats: int = 400, block: int = 8) -> list[float]:
    """Circular moving-block bootstrap interval for serially dependent scores."""
    v = np.asarray(values, dtype=np.float64)
    if (v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v))
            or repeats < 10 or block < 1 or seed < 0):
        raise ValueError("invalid block-bootstrap settings")
    rng = np.random.default_rng(seed)
    blocks_needed = (len(v) + block - 1) // block
    starts = rng.integers(0, len(v), size=(repeats, blocks_needed))
    offsets = np.arange(block, dtype=np.int64)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % len(v)
    flat = indexes.reshape(repeats, -1)[:, :len(v)]
    draws = v[flat].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def shuffle_training_by_year(images: np.ndarray, dates: np.ndarray, *,
                             train_end: int, seed: int) -> np.ndarray:
    """Shuffle only the already-observed training images within each year."""
    if (images.ndim != 3 or images.shape[1:] != (7, 7)
            or dates.shape != (len(images),)
            or not 2 <= train_end <= len(images) or seed < 0):
        raise ValueError("invalid training-shuffle input")
    rng = np.random.default_rng(seed)
    shuffled = images[:train_end].copy()
    years = dates[:train_end] // 10000
    for year in np.unique(years):
        positions = np.flatnonzero(years == year)
        shuffled[positions] = images[positions[rng.permutation(len(positions))]]
    return shuffled


def score_period(model: DynamicPatternLibrary, images: np.ndarray,
                 targets: np.ndarray, *, methods: tuple[str, ...],
                 seed: int, null_samples: int,
                 scramble_context: bool = False,
                 dates: np.ndarray | None = None,
                 ids: np.ndarray | None = None,
                 mapping: str = "sequential",
                 mapping_seed: int = 649,
                 include_draws: bool = False) -> tuple[dict, list[dict], dict]:
    """Score pre-result image outputs against targets and matched IID outcomes."""
    if (not methods or any(m not in METHODS for m in methods)
            or null_samples < 2 or seed < 0):
        raise ValueError("invalid score-period settings")
    indices = np.asarray(targets, dtype=np.int64)
    if (indices.ndim != 1 or not len(indices)
            or indices.min() < model.train_end
            or indices.max() >= len(images)):
        raise ValueError("targets outside model replay range")
    records = {name: [] for name in methods}
    predictions = {name: [] for name in methods}
    audit = []
    for t in indices:
        t = int(t)
        actual = images[t]
        controls = random_six_images(count=null_samples,
                                     seed=seed + t * 1000003)
        row = {}
        if include_draws:
            row = {
                "index": t,
                "date": int(dates[t]) if dates is not None else None,
                "draw_id": int(ids[t]) if ids is not None else None,
                "actual_main_six": decode_images(
                    actual, mapping=mapping, seed=mapping_seed).astype(int).tolist(),
                "actual_pattern": image_rows(actual),
                "methods": {},
            }
        for method in methods:
            predicted = model.predict(
                t, method=method, scramble_context=scramble_context,
                scramble_seed=seed + 51)
            observed = similarity(predicted, actual)
            iid_expected = conditional_chance_similarity(predicted, controls)
            adjusted = {
                metric: float(observed[metric] - iid_expected[metric])
                for metric in (*WEIGHTS, "composite")
            }
            records[method].append({
                "raw": observed, "iid": iid_expected, "adjusted": adjusted,
            })
            predictions[method].append(predicted)
            if include_draws:
                row["methods"][method] = {
                    "predicted_main_six": decode_images(
                        predicted, mapping=mapping, seed=mapping_seed).astype(int).tolist(),
                    "predicted_pattern": image_rows(predicted),
                    "raw_similarity": observed,
                    "iid_expected_for_prediction": iid_expected,
                    "adjusted_similarity": adjusted,
                    "exact_six_match_secondary": bool(
                        image_key(predicted) == image_key(actual)),
                }
        if include_draws:
            audit.append(row)
    summary = {}
    residuals = {}
    for method in methods:
        items = records[method]
        residuals[method] = np.asarray(
            [x["adjusted"]["composite"] for x in items], dtype=np.float64)
        summary[method] = {
            "draws": len(items),
            "mean_raw": {metric: float(np.mean(
                [x["raw"][metric] for x in items]))
                for metric in (*WEIGHTS, "composite")},
            "mean_iid_expected": {metric: float(np.mean(
                [x["iid"][metric] for x in items]))
                for metric in (*WEIGHTS, "composite")},
            "mean_chance_adjusted": {metric: float(np.mean(
                [x["adjusted"][metric] for x in items]))
                for metric in (*WEIGHTS, "composite")},
            "diversity": diversity(np.stack(predictions[method])),
        }
        if include_draws:
            summary[method]["exact_six_matches_secondary"] = int(sum(
                entry["methods"][method]["exact_six_match_secondary"]
                for entry in audit))
    return summary, audit, residuals


def run_dynamic_replay(*, source: Path = SOURCE,
                       year: int | None = None, seed: int = DEFAULT_SEED,
                       mapping: str = "sequential", mapping_seed: int = 649,
                       window: int = 5, neighbors: int = 8,
                       last_weight: float = 0.20,
                       null_samples: int = 32, validation_null_samples: int = 16,
                       permutations: int = 99, bank_shuffles: int = 4,
                       bootstrap_repeats: int = 400, bootstrap_block: int = 8,
                       train_end: int = TRAIN_END,
                       validation_end: int = VALIDATION_END) -> dict:
    """Evaluate fixed image-only methods with chronological and shuffled controls."""
    if (seed < 0 or mapping_seed < 0 or bank_shuffles < 0
            or permutations < 2 or null_samples < 2
            or validation_null_samples < 2 or bootstrap_repeats < 10
            or bootstrap_block < 1
            or year is not None and not 1900 <= year <= 2200):
        raise ValueError("invalid Phase 7C configuration")
    source = Path(source)
    numbers, dates, ids = load_archive(source)
    train, val, test = split_indices(
        len(numbers), window=window,
        train_end=train_end, validation_end=validation_end)
    selected_test = test if year is None else test[dates[test] // 10000 == year]
    if not len(selected_test):
        raise ValueError("requested year is absent from frozen evaluation period")
    if dates[val[-1]] >= dates[selected_test[0]]:
        raise ValueError("validation segment must precede evaluation")
    images = encode_draws(numbers, mapping=mapping, seed=mapping_seed)
    # 1. Predeclared temporal-dependence test, not used to retune methods.
    temporal = permutation_audit(
        images, dates, train_end=train_end,
        segments={
            "training": (0, train_end),
            "validation": (train_end, validation_end),
            "historical_test": (validation_end, len(images)),
        }, permutations=permutations, seed=seed)
    # 2. Fit visual analog memory only on training observations.
    model = DynamicPatternLibrary(
        images, train_end=train_end, window=window,
        neighbors=neighbors, last_weight=last_weight)
    valid, _, _ = score_period(
        model, images, val, methods=METHODS,
        seed=seed + 101, null_samples=validation_null_samples)
    chosen = max(METHODS, key=lambda m: valid[m]["mean_chance_adjusted"]["composite"])
    actual, per_draw, adjusted = score_period(
        model, images, selected_test, methods=METHODS,
        seed=seed + 202, null_samples=null_samples,
        dates=dates, ids=ids, mapping=mapping,
        mapping_seed=mapping_seed, include_draws=True)
    # 3. Confidence bands for per-target chance-adjusted score and paired lift.
    for j, method in enumerate(METHODS):
        actual[method]["bootstrap_95_percent_interval"] = bootstrap_mean_ci(
            adjusted[method], seed=seed + 501 + j,
            repeats=bootstrap_repeats, block=bootstrap_block)
        if method != "last_pattern":
            difference = adjusted[method] - adjusted["last_pattern"]
            actual[method]["mean_adjusted_gain_over_last_pattern"] = float(difference.mean())
            actual[method]["gain_over_last_pattern_bootstrap_95_percent_interval"] = (
                bootstrap_mean_ci(
                    difference, seed=seed + 601 + j,
                    repeats=bootstrap_repeats, block=bootstrap_block))
    # 4. Break the *order* of previously known input frames, not future target data.
    scrambled, _, _ = score_period(
        model, images, selected_test,
        methods=("centroid_flow", "analog_fusion"),
        seed=seed + 202, null_samples=null_samples, scramble_context=True)
    # 5. Destroy training image transitions within years, without ever
    # manipulating the actual validation or evaluation observations.
    shuffled_analogs = []
    for k in range(bank_shuffles):
        bank = shuffle_training_by_year(
            images, dates, train_end=train_end,
            seed=seed + 1301 + k)
        null_model = DynamicPatternLibrary(
            images, train_end=train_end, window=window,
            neighbors=neighbors, last_weight=last_weight,
            bank_override=bank)
        random_summary, _, _ = score_period(
            null_model, images, selected_test, methods=("analog_fusion",),
            seed=seed + 202, null_samples=null_samples)
        shuffled_analogs.append(
            random_summary["analog_fusion"]["mean_chance_adjusted"]["composite"])
    analog_real = actual["analog_fusion"]["mean_chance_adjusted"]["composite"]
    shuffle_control = {
        "count": bank_shuffles,
        "scores": shuffled_analogs,
        "real_analog_adjusted_score": analog_real,
        "shuffled_train_mean_adjusted_score": (
            float(np.mean(shuffled_analogs)) if shuffled_analogs else None),
        "empirical_upper_tail_diagnostic": (
            float((1 + np.count_nonzero(np.asarray(shuffled_analogs) >= analog_real))
                  / (bank_shuffles + 1)) if shuffled_analogs else None),
        "qualification": (
            "Training bank order shuffled within years, evaluated against "
            "unaltered real outcomes. Small shuffle counts are diagnostic only."),
    }
    return {
        "schema": "lotto649.phase7c.dynamic_pattern_replay.v1",
        "version": VERSION,
        "status": "EXPLORATORY_PREVIOUSLY_INSPECTED_HISTORICAL_ARCHIVE",
        "primary_objective": "7x7_pattern_similarity_above_conditional_iid",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "year": year if year is not None else "all_test_years",
        "evaluated_draws": len(selected_test),
        "methods": list(METHODS),
        "validation_selected": chosen,
        "split": {
            "train": [int(train[0]), int(train[-1])],
            "validation": [int(val[0]), int(val[-1])],
            "test": [int(selected_test[0]), int(selected_test[-1])],
            "test_start_date": int(dates[selected_test[0]]),
            "test_end_date": int(dates[selected_test[-1]]),
        },
        "config": {
            "seed": seed, "mapping": mapping, "mapping_seed": mapping_seed,
            "window": window, "neighbors": neighbors,
            "last_weight": last_weight, "null_samples": null_samples,
            "validation_null_samples": validation_null_samples,
            "temporal_permutations": permutations,
            "bank_shuffles": bank_shuffles,
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_block": bootstrap_block,
            "metric_weights": WEIGHTS,
        },
        "temporal_order_audit": temporal,
        "validation_summary": valid,
        "evaluation_summary": actual,
        "past_frame_order_scramble": {
            name: {
                "mean_adjusted_composite": values["mean_chance_adjusted"]["composite"],
                "original_minus_scrambled": float(
                    actual[name]["mean_chance_adjusted"]["composite"]
                    - values["mean_chance_adjusted"]["composite"]),
                "diversity": values["diversity"],
            } for name, values in scrambled.items()
        },
        "training_transition_shuffle": shuffle_control,
        "individual_draws": per_draw,
        "limitations": [
            "No claim of a statistically validated advantage over fair draws.",
            "Retrospective 2023-2026 data were previously examined.",
            "Different pattern shapes can score highly without any common numbers.",
            "Diversity by itself does not predict image similarity.",
            "Conditional IID expectations use finite Monte Carlo samples.",
            "Bootstrap confidence intervals do not include all Monte Carlo uncertainty.",
            "Year-stratified shuffling tests temporal order under exchangeability within years.",
            "Time-shuffle diagnostics are not a post-hoc reason to retune weights.",
            "Phase 6 forecast seals and old model artifacts are never changed.",
        ],
    }


def export_csv(report: dict, path: Path) -> None:
    """Flat actual-vs-predicted shape and conditional IID audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "date", "draw_id", "method", "predicted_main_six",
            "actual_main_six", "predicted_pattern", "actual_pattern",
            "raw_composite", "iid_expected_composite", "adjusted_composite",
            "raw_shape_d4", "adjusted_shape_d4", "adjusted_low_dct",
            "exact_six_match_secondary",
        ])
        for draw in report["individual_draws"]:
            for name in report["methods"]:
                pred = draw["methods"][name]
                writer.writerow([
                    draw["date"], draw["draw_id"], name,
                    " ".join(f"{n:02d}" for n in pred["predicted_main_six"]),
                    " ".join(f"{n:02d}" for n in draw["actual_main_six"]),
                    "/".join(pred["predicted_pattern"]),
                    "/".join(draw["actual_pattern"]),
                    pred["raw_similarity"]["composite"],
                    pred["iid_expected_for_prediction"]["composite"],
                    pred["adjusted_similarity"]["composite"],
                    pred["raw_similarity"]["shape_d4"],
                    pred["adjusted_similarity"]["shape_d4"],
                    pred["adjusted_similarity"]["low_dct"],
                    pred["exact_six_match_secondary"],
                ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7C dynamic visual pattern and temporal-shuffle replay")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--year", default="all",
                        help="Calendar year or 'all' historical evaluation targets")
    parser.add_argument("--mapping", choices=("sequential", "snake", "permuted"),
                        default="sequential")
    parser.add_argument("--mapping-seed", type=int, default=649)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--last-weight", type=float, default=0.20)
    parser.add_argument("--null-samples", type=int, default=32)
    parser.add_argument("--validation-null-samples", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--bank-shuffles", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--bootstrap-block", type=int, default=8)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--validation-end", type=int, default=VALIDATION_END)
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase7c_dynamic_replay.json"))
    parser.add_argument("--csv", type=Path,
                        default=Path("data/neural/phase7c_dynamic_replay.csv"))
    args = parser.parse_args()
    try:
        year = None if args.year.lower() == "all" else int(args.year)
    except ValueError:
        parser.error("--year must be calendar year or 'all'")
    report = run_dynamic_replay(
        source=args.source, year=year, seed=args.seed,
        mapping=args.mapping, mapping_seed=args.mapping_seed,
        window=args.window, neighbors=args.neighbors,
        last_weight=args.last_weight, null_samples=args.null_samples,
        validation_null_samples=args.validation_null_samples,
        permutations=args.permutations, bank_shuffles=args.bank_shuffles,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_block=args.bootstrap_block,
        train_end=args.train_end, validation_end=args.validation_end)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    export_csv(report, args.csv)
    print(f"Lotto649 Phase 7C Dynamic Pattern Replay v{VERSION}")
    print(f"Evaluation={report['year']} draws={report['evaluated_draws']}")
    print(f"Validation-only selected={report['validation_selected']}")
    print("Method               Raw pattern     Expected IID    Adjusted     Unique    Top share")
    for name, row in report["evaluation_summary"].items():
        print(f"{name:20s} {row['mean_raw']['composite']:11.6f} "
              f"{row['mean_iid_expected']['composite']:16.6f} "
              f"{row['mean_chance_adjusted']['composite']:+11.6f} "
              f"{row['diversity']['unique_pixel_patterns']:9d} "
              f"{row['diversity']['top_pattern_fraction']:11.2%}")
    print("Lag-1 temporal order (real vs within-year shuffled):")
    for segment, item in report["temporal_order_audit"]["segments"].items():
        lag = item["lags"]["1"]
        print(f"  {segment}: real={lag['observed_mean_cosine']:+.5f} "
              f"shuffle={lag['shuffled_mean_cosine']:+.5f} "
              f"tail={lag['empirical_upper_tail']:.4f}")
    for name, item in report["past_frame_order_scramble"].items():
        print(f"[CONTEXT SHUFFLE] {name}: real-minus-shuffled="
              f"{item['original_minus_scrambled']:+.6f}")
    bank = report["training_transition_shuffle"]
    if bank["count"]:
        print(f"[TRAIN SHUFFLE] analog_fusion real={bank['real_analog_adjusted_score']:+.6f} "
              f"shuffle mean={bank['shuffled_train_mean_adjusted_score']:+.6f} "
              f"trials={bank['count']}")
    print(f"[DONE] JSON: {args.report.resolve()}")
    print(f"[DONE] CSV: {args.csv.resolve()}")
    print("[CAUTION] Historical sequence evidence is exploratory; no wagering edge shown")


if __name__ == "__main__":
    main()
