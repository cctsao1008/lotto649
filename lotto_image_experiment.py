"""Phase 7: controlled one-step learning on reversible 7x7 representations.

Each model returns canonical 49-number logits and is trained with the existing
exact normalized six-subset likelihood. No output is a wagering recommendation.
Historical 2025 data have already been inspected; this is exploratory replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lotto_historical import SOURCE, load_official_main_draws
from lotto_image_repr import temporal_windows
from lotto_neural import (IID_LOGP, K, N, SmallMLP, fit_model,
                          make_observations)
from lotto_replay import (TRAIN_END, VALIDATION_END,
                          top_subset, validate_replay_split)
from lotto_torch_decoder import FixedCardinalityDecoder

VERSION = "0.1.0"
MODEL_NAMES = ("iid", "flat_mlp", "cnn_sequential", "cnn_snake",
               "cnn_permuted", "cnn_dct", "cnn_fft", "cnn_temporal")


class SpatialCNN(nn.Module):
    """Small spatial encoder; 49 outputs are always canonical number logits."""

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        if channels < 1 or hidden < 1:
            raise ValueError("channels and hidden size must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.Tanh(),
            nn.Flatten(), nn.Linear(8 * 7 * 7, hidden), nn.Tanh(),
            nn.Linear(hidden, N),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-2:] != (7, 7):
            raise ValueError("CNN features must be [batch,channels,7,7]")
        return self.net(x)


class TemporalCNN(nn.Module):
    """3D-convolutional encoder across past-time and two pixel dimensions."""

    def __init__(self, window: int, hidden: int = 16):
        super().__init__()
        if window < 1 or hidden < 1:
            raise ValueError("window and hidden must be positive")
        self.net = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv3d(4, 4, kernel_size=3, padding=1), nn.Tanh(),
            nn.Flatten(), nn.Linear(4 * window * 7 * 7, hidden), nn.Tanh(),
            nn.Linear(hidden, N),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[-2:] != (7, 7):
            raise ValueError("temporal features must be [batch,1,time,7,7]")
        return self.net(x)


def make_model_and_features(draws: np.ndarray, *, name: str,
                            window: int, hidden: int,
                            mapping_seed: int) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Build a causal feature matrix; do not use the prediction target in x."""
    if name == "flat_mlp":
        features, targets, _ = make_observations(draws, window)
        return SmallMLP(window, hidden), features, targets
    if name not in MODEL_NAMES or name == "iid":
        raise ValueError("unknown trainable representation")
    if name == "cnn_temporal":
        features, targets, _ = temporal_windows(
            draws, window=window, mapping="sequential", as_volume=True)
        model: nn.Module = TemporalCNN(window, hidden)
    else:
        mapping = {"cnn_sequential": "sequential", "cnn_snake": "snake",
                   "cnn_permuted": "permuted"}.get(name, "sequential")
        domain = {"cnn_dct": "dct", "cnn_fft": "fft"}.get(name, "spatial")
        features, targets, _ = temporal_windows(
            draws, window=window, mapping=mapping,
            domain=domain, seed=mapping_seed)
        channels = window * (2 if domain == "fft" else 1)
        model = SpatialCNN(channels, hidden)
    return model, torch.from_numpy(features), torch.from_numpy(targets.copy())


def score_fitted(model: nn.Module, x: torch.Tensor,
                 targets: torch.Tensor, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full-subset log scores and pre-result logits, with no gradients."""
    decoder = FixedCardinalityDecoder(N, K)
    probabilities = []
    weights = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = model(x[start:start + batch_size])
            weights.append(pred.cpu().numpy().astype(np.float64))
            probabilities.append(
                decoder(pred, targets[start:start + batch_size]).cpu().numpy().astype(np.float64))
    return np.concatenate(weights), np.concatenate(probabilities)


def run_image_experiment(*, source: Path = SOURCE, year: int = 2025,
                         models: tuple[str, ...] = MODEL_NAMES,
                         seed: int = 649, mapping_seed: int = 649,
                         window: int = 5, hidden: int = 16,
                         epochs: int = 40, patience: int = 6,
                         batch_size: int = 128, lr: float = 0.005,
                         weight_decay: float = 0.02,
                         train_end: int = TRAIN_END,
                         validation_end: int = VALIDATION_END) -> dict:
    """Fit once with fixed train/validation indices; replay an observed year."""
    if (not models or len(set(models)) != len(models)
            or "iid" not in models or any(x not in MODEL_NAMES for x in models)
            or seed < 0 or mapping_seed < 0 or year < 1900 or year > 2200
            or hidden < 1 or epochs < 1 or patience < 1 or batch_size < 1):
        raise ValueError("invalid experiment configuration")
    draws, dates, ids = load_official_main_draws(source)
    train, validation, test = validate_replay_split(
        len(draws), window=window, train_end=train_end,
        validation_end=validation_end)
    evaluated = test[dates[test] // 10000 == year]
    if not len(evaluated):
        raise ValueError("requested year is absent from the frozen evaluation segment")
    if int(dates[validation[-1]]) >= int(dates[evaluated[0]]):
        raise ValueError("validation must precede selected evaluation year")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    samples = evaluated - window
    trained = {}
    validation_log_scores = {"iid": float(IID_LOGP)}
    eval_logits = {"iid": np.zeros((len(evaluated), N), dtype=np.float64)}
    eval_log_scores = {"iid": np.full(len(evaluated), IID_LOGP, dtype=np.float64)}
    target_draws = draws[evaluated]
    for name in models:
        if name == "iid":
            continue
        torch.manual_seed(seed)
        model, features, targets = make_model_and_features(
            draws, name=name, window=window, hidden=hidden,
            mapping_seed=mapping_seed)
        if (len(features) != len(draws) - window
                or not np.array_equal(targets[samples].numpy(), target_draws)):
            raise AssertionError("causal target indexing mismatch")
        trained_model, trace = fit_model(
            model, x_train=features[torch.as_tensor(train - window)],
            y_train=targets[torch.as_tensor(train - window)],
            x_val=features[torch.as_tensor(validation - window)],
            y_val=targets[torch.as_tensor(validation - window)],
            seed=seed, epochs=epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay)
        logits, logps = score_fitted(
            trained_model, features[torch.as_tensor(samples)],
            targets[torch.as_tensor(samples)], batch_size=batch_size)
        if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(logps)):
            raise ValueError("non-finite out-of-sample image predictions")
        trained[name] = trace
        validation_log_scores[name] = float(trace["best_val_log_score"])
        eval_logits[name] = logits
        eval_log_scores[name] = logps
    validation_selected = max(
        (name for name in MODEL_NAMES if name in models),
        key=lambda name: validation_log_scores[name])
    per_draw = []
    summary = {}
    for model_index, name in enumerate(MODEL_NAMES):
        if name not in models:
            continue
        full_matches = []
        for i, index in enumerate(evaluated):
            actual = target_draws[i].astype(int).tolist()
            top = top_subset(eval_logits[name][i], seed=seed,
                             draw_id=int(ids[index]), model_index=model_index)
            exact = top == actual
            full_matches.append(exact)
            if len(per_draw) <= i:
                per_draw.append({"date": int(dates[index]),
                                 "draw_id": int(ids[index]),
                                 "actual_main_six": actual,
                                 "models": {}})
            per_draw[i]["models"][name] = {
                "predicted_main_six": top,
                "exact_six_match": bool(exact),
                "result": "PASS" if exact else "FAIL",
                "full_set_log_probability": float(eval_log_scores[name][i]),
                "log_score_lift_vs_iid_nats": float(eval_log_scores[name][i] - IID_LOGP),
            }
        summary[name] = {
            "exact_six_matches": int(sum(full_matches)),
            "evaluated_draws": len(evaluated),
            "exact_six_success_rate": float(np.mean(full_matches)),
            "mean_log_score_lift_vs_iid_nats": float(np.mean(eval_log_scores[name] - IID_LOGP)),
            "mean_full_set_log_probability": float(np.mean(eval_log_scores[name])),
        }
    return {
        "schema": "lotto649.phase7.image_experiment.v1", "version": VERSION,
        "status": "EXPLORATORY_HISTORICAL_REUSE_NOT_PROSPECTIVE_BLIND",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "year": year, "evaluated_draws": len(evaluated),
        "models": list(models), "default_model": "iid",
        "validation_selected": validation_selected,
        "validation_log_scores": validation_log_scores,
        "training": trained,
        "split": {
            "train": [int(train[0]), int(train[-1])],
            "validation": [int(validation[0]), int(validation[-1])],
            "evaluation": [int(evaluated[0]), int(evaluated[-1])],
            "year_first_date": int(dates[evaluated[0]]),
            "year_last_date": int(dates[evaluated[-1]]),
        },
        "config": {
            "seed": seed, "mapping_seed": mapping_seed,
            "window": window, "hidden": hidden, "epochs": epochs,
            "patience": patience, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
        },
        "iid_theory": {
            "single_full_set_probability": 1 / math.comb(N, K),
            "expected_full_set_matches": len(evaluated) / math.comb(N, K),
        },
        "summary_by_model": summary,
        "individual_draws": per_draw,
        "limitations": [
            "Historical draw outcomes were inspected previously; this is not new blind evidence.",
            "No phase-6 seal or source is changed by this experiment.",
            "Only a complete six-of-six match counts as PASS.",
            "Models share original fixed train/validation targets and evaluation year.",
            "Inputs contain earlier draws only; lagged test histories may contain earlier observed test draws.",
            "Image transforms retain all input information but cannot create predictive signal.",
            "The default model remains IID until independent future evidence supports a change.",
        ],
    }


def export_csv(report: dict, path: Path) -> None:
    """Produce a flat, one-row-per-model-per-draw exact-match audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("date", "draw_id", "model", "predicted_main_six",
                         "actual_main_six", "exact_six_match", "result",
                         "full_set_log_probability", "log_score_lift_vs_iid_nats"))
        for row in report["individual_draws"]:
            for name in report["models"]:
                p = row["models"][name]
                writer.writerow((row["date"], row["draw_id"], name,
                                 " ".join(f"{x:02d}" for x in p["predicted_main_six"]),
                                 " ".join(f"{x:02d}" for x in row["actual_main_six"]),
                                 p["exact_six_match"], p["result"],
                                 p["full_set_log_probability"],
                                 p["log_score_lift_vs_iid_nats"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 historical image-representation six-of-six experiment")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--mapping-seed", type=int, default=649)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--validation-end", type=int, default=VALIDATION_END)
    parser.add_argument("--report", type=Path, default=Path("data/neural/phase7_2025.json"))
    parser.add_argument("--csv", type=Path, default=Path("data/neural/phase7_2025.csv"))
    parser.add_argument("--show-draws", action="store_true")
    args = parser.parse_args()
    result = run_image_experiment(
        source=args.source, year=args.year, models=tuple(args.models),
        seed=args.seed, mapping_seed=args.mapping_seed,
        window=args.window, hidden=args.hidden,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        train_end=args.train_end, validation_end=args.validation_end)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    export_csv(result, args.csv)
    print(f"Lotto649 Phase 7 Image Experiment v{VERSION}")
    print(f"Year={result['year']} evaluated draws={result['evaluated_draws']}")
    print(f"Validation-only selected={result['validation_selected']}")
    print("Model               Exact 6/6      Tested     Mean log-score lift")
    for name, item in result["summary_by_model"].items():
        print(f"{name:20s} {item['exact_six_matches']:6d} "
              f"{item['evaluated_draws']:11d} "
              f"{item['mean_log_score_lift_vs_iid_nats']:+21.6f}")
    if args.show_draws:
        for row in result["individual_draws"]:
            actual = " ".join(f"{n:02d}" for n in row["actual_main_six"])
            for name in result["models"]:
                p = row["models"][name]
                chosen = " ".join(f"{n:02d}" for n in p["predicted_main_six"])
                print(f"{row['date']} {row['draw_id']} {name:20s} "
                      f"predicted={chosen} actual={actual} {p['result']}")
    print(f"[DONE] JSON: {args.report.resolve()}")
    print(f"[DONE] CSV: {args.csv.resolve()}")
    print("[CAUTION] This is previously explored historical data, not prospective evidence")


if __name__ == "__main__":
    main()
