"""Phase 7B: historical replay where image similarity is the primary outcome.

No legacy MLP, logit, CNN classification or exact-six optimizer is used.
Methods produce one legal 7x7 pattern from earlier images only. Every score
is compared against random IID targets for that exact proposed pattern.
The historical archive is previously examined, so all results are exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from lotto_image_repr import decode_images, encode_draws
from lotto_pattern_models import METHODS, PatternLibrary
from lotto_pattern_similarity import (
    VERSION as SIMILARITY_VERSION, WEIGHTS,
    conditional_chance_similarity, image_key, random_six_images,
    similarity,
)

VERSION = "0.1.0"
SOURCE = Path("data/normalized/lotto649_base.npy")
TRAIN_END, VALIDATION_END = 1309, 1746
DEFAULT_SEED = 649


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently read the normalized archive; ignore special for prediction."""
    source = np.load(path, allow_pickle=False)
    if (source.dtype.names is None
            or not {"date", "draw_id", "numbers", "special"}.issubset(source.dtype.names)):
        raise ValueError("expected structured normalized Lotto649 archive")
    numbers = np.asarray(source["numbers"])
    dates = np.asarray(source["date"], dtype=np.int64)
    draw_ids = np.asarray(source["draw_id"], dtype=np.int64)
    special = np.asarray(source["special"], dtype=np.int64)
    if (numbers.ndim != 2 or numbers.shape[1] != 6 or len(numbers) < 40
            or not np.issubdtype(numbers.dtype, np.integer)):
        raise ValueError("invalid archive draw matrix")
    numbers = np.sort(numbers.astype(np.int64), axis=1)
    if (np.any(numbers < 1) or np.any(numbers > 49)
            or np.any(np.diff(numbers, axis=1) == 0)
            or not np.all(np.diff(dates) > 0)
            or len(set(draw_ids.tolist())) != len(draw_ids)
            or np.any(draw_ids <= 0)
            or np.any(special < 1) or np.any(special > 49)
            or np.any(np.any(numbers == special[:, None], axis=1))):
        raise ValueError("invalid chronology, draw identity or numbers")
    for d in dates:
        datetime.strptime(str(int(d)), "%Y%m%d")
    return numbers, dates, draw_ids


def split_indices(n: int, *, window: int, train_end: int,
                  validation_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use fixed original Phase 5 absolute boundaries, never re-split an archive."""
    if (not all(isinstance(x, int) for x in (n, window, train_end, validation_end))
            or not 1 <= window < train_end
            or train_end-window < 10 or validation_end-train_end < 10
            or n-validation_end < 10):
        raise ValueError("invalid fixed chronological boundaries")
    return (np.arange(window, train_end, dtype=np.int64),
            np.arange(train_end, validation_end, dtype=np.int64),
            np.arange(validation_end, n, dtype=np.int64))


def _image_view(image: np.ndarray) -> list[str]:
    return ["".join("#" if cell else "." for cell in row) for row in image]


def _score_targets(lib: PatternLibrary, images: np.ndarray,
                   indices: np.ndarray, *, seed: int,
                   null_samples: int, include_draws: bool,
                   dates: np.ndarray | None = None,
                   ids: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """Score each immutable candidate against outcome and conditional IID null."""
    if null_samples < 2:
        raise ValueError("at least two IID reference samples required")
    per_method: dict[str, list[dict]] = {name: [] for name in METHODS}
    audit = []
    for t in indices:
        t = int(t)
        actual = images[t]
        # These control outcomes are generated without observing the actual
        # target and shared across methods for paired comparisons.
        controls = random_six_images(
            count=null_samples, seed=seed + 1000003 * t)
        row = {}
        if include_draws:
            row = {
                "index": t,
                "date": int(dates[t]) if dates is not None else None,
                "draw_id": int(ids[t]) if ids is not None else None,
                "actual_main_six": decode_images(actual).astype(int).tolist(),
                "actual_pattern": _image_view(actual),
                "methods": {},
            }
        for method in METHODS:
            prediction = lib.predict(t, method=method)
            observed = similarity(prediction, actual)
            chance = conditional_chance_similarity(prediction, controls)
            delta = observed["composite"] - chance["composite"]
            per_method[method].append({
                "similarity": observed, "iid_expected": chance,
                "chance_adjusted": float(delta),
                "exact_six_match": bool(image_key(prediction) == image_key(actual)),
            })
            if include_draws:
                row["methods"][method] = {
                    "predicted_main_six": decode_images(prediction).astype(int).tolist(),
                    "predicted_pattern": _image_view(prediction),
                    "similarity": observed,
                    "iid_expected_for_prediction": chance,
                    "chance_adjusted_composite": float(delta),
                    "exact_six_match": bool(image_key(prediction) == image_key(actual)),
                }
        if include_draws:
            audit.append(row)
    summary = {}
    for name, rows in per_method.items():
        summary[name] = {
            "draws": len(rows),
            "mean_similarity": {
                metric: float(np.mean([r["similarity"][metric] for r in rows]))
                for metric in (*WEIGHTS, "composite")
            },
            "mean_iid_expected_for_predictions": {
                metric: float(np.mean([r["iid_expected"][metric] for r in rows]))
                for metric in (*WEIGHTS, "composite")
            },
            "mean_chance_adjusted_composite": float(np.mean([
                r["chance_adjusted"] for r in rows])),
            "exact_six_matches_secondary": int(sum(r["exact_six_match"] for r in rows)),
        }
    return summary, audit


def run_replay(*, source: Path = SOURCE, year: int | None = 2025,
               seed: int = DEFAULT_SEED, window: int = 5,
               clusters: int = 8, mapping: str = "sequential",
               mapping_seed: int = 649, null_samples: int = 32,
               validation_null_samples: int = 16,
               train_end: int = TRAIN_END,
               validation_end: int = VALIDATION_END) -> dict:
    """Predict shape only, select on validation, replay chronological test rows."""
    if (not isinstance(seed, int) or seed < 0
            or not isinstance(mapping_seed, int) or mapping_seed < 0
            or year is not None and not 1900 <= year <= 2200
            or not isinstance(window, int) or window < 1
            or not isinstance(clusters, int) or clusters < 2
            or null_samples < 2 or validation_null_samples < 2):
        raise ValueError("invalid experiment settings")
    numbers, dates, ids = load_archive(source)
    train, val, test = split_indices(
        len(numbers), window=window,
        train_end=train_end, validation_end=validation_end)
    indices = test if year is None else test[dates[test] // 10000 == year]
    if not len(indices):
        raise ValueError("requested year absent from frozen test segment")
    if int(dates[val[-1]]) >= int(dates[indices[0]]):
        raise ValueError("validation must precede chosen test period")
    images = encode_draws(numbers, mapping=mapping, seed=mapping_seed)
    lib = PatternLibrary(images, train_end=train_end,
                         window=window, clusters=clusters)
    valid, _ = _score_targets(
        lib, images, val, seed=seed + 13,
        null_samples=validation_null_samples, include_draws=False)
    selected = max(METHODS, key=lambda k: valid[k]["mean_chance_adjusted_composite"])
    test_summary, per_draw = _score_targets(
        lib, images, indices, seed=seed + 29,
        null_samples=null_samples, include_draws=True,
        dates=dates, ids=ids)
    return {
        "schema": "lotto649.phase7b.pattern_similarity.v1",
        "version": VERSION,
        "similarity_version": SIMILARITY_VERSION,
        "status": "EXPLORATORY_HISTORICAL_REUSE_NOT_PROSPECTIVE_BLIND",
        "objective": "Predict a close 7x7 binary pattern, calibrated against IID targets",
        "primary_score": "mean_chance_adjusted_composite",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "year": year if year is not None else "all_test_years",
        "evaluated_draws": len(indices),
        "methods": list(METHODS),
        "validation_selected": selected,
        "config": {
            "seed": seed, "window": window, "clusters": clusters,
            "mapping": mapping, "mapping_seed": mapping_seed,
            "null_samples_per_test_draw": null_samples,
            "null_samples_per_validation_draw": validation_null_samples,
            "metric_weights": WEIGHTS,
        },
        "split": {
            "train": [int(train[0]), int(train[-1])],
            "validation": [int(val[0]), int(val[-1])],
            "evaluation": [int(indices[0]), int(indices[-1])],
            "test_first_date": int(dates[indices[0]]),
            "test_last_date": int(dates[indices[-1]]),
        },
        "validation_summary": valid,
        "evaluation_summary": test_summary,
        "train_only_pattern_families": lib.cluster_report(),
        "individual_draws": per_draw,
        "limitations": [
            "A high raw shape score does not imply exact-number matching.",
            "Every prediction pattern is calibrated against simulated IID outcomes.",
            "Conditional IID expectation is Monte Carlo and has sampling error.",
            "The composite metric and its weights are a design choice, not an official prize.",
            "Only the training window is used to fit clusters and analog memory.",
            "Each target t uses realized observations prior to t as context; no target leakage.",
            "Test archive was explored in earlier experiments; results are not blind.",
            "No older neural, logit or six-subset probability model is used.",
            "The official Phase 6 prospective forecast seal is left unchanged.",
        ],
    }


def export_csv(report: dict, path: Path) -> None:
    """Flat audit table: primary shape similarity, chance adjustment and components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "date", "draw_id", "method", "predicted_main_six",
            "actual_main_six", "predicted_pattern", "actual_pattern",
            "raw_similarity", "iid_expected_similarity",
            "chance_adjusted_similarity", *WEIGHTS.keys(),
            "exact_six_match_secondary",
        ])
        for draw in report["individual_draws"]:
            for name in report["methods"]:
                row = draw["methods"][name]
                writer.writerow([
                    draw["date"], draw["draw_id"], name,
                    " ".join(f"{n:02d}" for n in row["predicted_main_six"]),
                    " ".join(f"{n:02d}" for n in draw["actual_main_six"]),
                    "/".join(row["predicted_pattern"]),
                    "/".join(draw["actual_pattern"]),
                    row["similarity"]["composite"],
                    row["iid_expected_for_prediction"]["composite"],
                    row["chance_adjusted_composite"],
                    *[row["similarity"][metric] for metric in WEIGHTS],
                    row["exact_six_match"],
                ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7B pattern similarity replay without legacy models")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--year", default="2025",
                        help="Calendar year (e.g. 2025) or all")
    parser.add_argument("--mapping", choices=("sequential", "snake", "permuted"),
                        default="sequential")
    parser.add_argument("--mapping-seed", type=int, default=649)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--null-samples", type=int, default=32)
    parser.add_argument("--validation-null-samples", type=int, default=16)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--validation-end", type=int, default=VALIDATION_END)
    parser.add_argument("--report", type=Path, default=Path(
        "data/neural/phase7b_pattern_similarity.json"))
    parser.add_argument("--csv", type=Path, default=Path(
        "data/neural/phase7b_pattern_similarity.csv"))
    parser.add_argument("--show-draws", action="store_true")
    args = parser.parse_args()
    try:
        year = None if args.year.lower() == "all" else int(args.year)
    except ValueError as exc:
        parser.error("--year must be a calendar year or 'all'")
    result = run_replay(
        source=args.source, year=year, seed=args.seed,
        window=args.window, clusters=args.clusters,
        mapping=args.mapping, mapping_seed=args.mapping_seed,
        null_samples=args.null_samples,
        validation_null_samples=args.validation_null_samples,
        train_end=args.train_end, validation_end=args.validation_end)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    export_csv(result, args.csv)
    print(f"Lotto649 Phase 7B Pattern Similarity v{VERSION}")
    print(f"Evaluation={result['year']} draws={result['evaluated_draws']}")
    print(f"Validation-only selected={result['validation_selected']}")
    print("Method                Raw similarity    Expected IID    Chance adjusted")
    for name, record in result["evaluation_summary"].items():
        print(f"{name:21s} "
              f"{record['mean_similarity']['composite']:14.6f} "
              f"{record['mean_iid_expected_for_predictions']['composite']:15.6f} "
              f"{record['mean_chance_adjusted_composite']:+18.6f}")
    if args.show_draws:
        for row in result["individual_draws"]:
            print(f"[DRAW] {row['date']} {row['draw_id']}")
            print("  Actual:")
            for line in row["actual_pattern"]:
                print("   " + line)
            for method in result["methods"]:
                p = row["methods"][method]
                print(f"  {method}: raw={p['similarity']['composite']:.4f} "
                      f"chance_adjusted={p['chance_adjusted_composite']:+.4f}")
                for line in p["predicted_pattern"]:
                    print("   " + line)
    print(f"[DONE] JSON: {args.report.resolve()}")
    print(f"[DONE] CSV: {args.csv.resolve()}")
    print("[CAUTION] Historical shape similarity is not evidence of a lottery advantage")


if __name__ == "__main__":
    main()
