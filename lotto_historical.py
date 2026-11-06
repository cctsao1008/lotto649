"""Lotto649 Phase 5: exploratory historical one-step probability backtest.

This tool uses official main-six draw records in chronological order.
It compares fair IID, a static six-subset distribution, a small MLP,
and a train-only lag-overlap model. Validation chooses a single model;
test data are scored only after training and validation decisions freeze.

IMPORTANT: the historical archive has been analyzed in earlier research.
Its final 20% is therefore an exploratory chronological holdout, not a
pristine prospective blind test. This program produces NO lottery tickets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from lotto_neural import (IID_LOGP, K, N, SmallMLP, StaticLogitModel,
                          chronological_split, fit_model, make_observations)
from lotto_temporal import fit_beta, lag_mean_log_score, overlap_distribution
from lotto_torch_decoder import FixedCardinalityDecoder

VERSION = "0.1.0"
SOURCE = Path("data/normalized/lotto649_base.npy")
REPORT = Path("data/neural/phase5_historical.json")
CANDIDATES = ("iid", "static", "lag", "mlp")


def load_official_main_draws(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate historical main numbers, date, and draw identifiers."""
    archive = np.load(path, allow_pickle=False)
    required = {"date", "draw_id", "numbers", "special"}
    if archive.dtype.names is None or not required.issubset(archive.dtype.names):
        raise ValueError("expected structured normalized archive with date/draw_id/numbers/special")
    if len(archive) < 40:
        raise ValueError("historical archive is too short")
    dates = np.asarray(archive["date"], dtype=np.int64)
    ids = np.asarray(archive["draw_id"], dtype=np.int64)
    raw = np.asarray(archive["numbers"])
    if raw.ndim != 2 or raw.shape != (len(archive), K):
        raise ValueError("numbers field must have six values per draw")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("draw numbers must be integers")
    numbers = np.sort(raw.astype(np.int64), axis=1)
    if (not np.all((numbers >= 1) & (numbers <= N))
            or not np.all(np.diff(numbers, axis=1) > 0)):
        raise ValueError("invalid main draw numbers")
    special = np.asarray(archive["special"], dtype=np.int64)
    if (not np.all((special >= 1) & (special <= N))
            or np.any(np.any(numbers == special[:, None], axis=1))):
        raise ValueError("invalid special number")
    if (not np.all(np.diff(dates) > 0) or len(np.unique(ids)) != len(ids)
            or np.any(ids < 1)):
        raise ValueError("dates must be increasing and draw IDs unique")
    for item in dates:
        datetime.strptime(str(int(item)), "%Y%m%d")
    return numbers, dates, ids


def choose_from_validation(scores: dict[str, float]) -> str:
    """Pre-specified selection: highest validation log score; ties favor IID."""
    if set(scores) != set(CANDIDATES):
        raise ValueError("all four candidate validation scores are required")
    if not all(math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("validation scores must be finite")
    return max(CANDIDATES, key=lambda key: scores[key])


def moving_block_ci(values: np.ndarray, *, block_size: int = 13,
                    samples: int = 2000, seed: int = 20261107) -> list[float]:
    """Descriptive circular moving-block bootstrap interval for a mean.

    Uses one draw-aligned series of paired score differences. It does NOT
    correct for model search, prior historical exploration, or future drift.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.all(np.isfinite(arr)):
        raise ValueError("values must be a nonempty finite 1D series")
    if block_size < 1 or samples < 1:
        raise ValueError("block_size and samples must be positive")
    rng = np.random.default_rng(seed)
    n = len(arr)
    block = min(block_size, n)
    positions = np.arange(block, dtype=np.int64)
    blocks_needed = math.ceil(n / block)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        starts = rng.integers(0, n, size=blocks_needed)
        selections = ((starts[:, None] + positions) % n).reshape(-1)[:n]
        means[i] = arr[selections].mean()
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


@torch.no_grad()
def model_log_probs(model: torch.nn.Module, decoder: FixedCardinalityDecoder,
                    x: torch.Tensor, y: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    return np.concatenate([
        decoder(model(x[start:start + batch_size]),
                y[start:start + batch_size]).cpu().numpy().astype(np.float64)
        for start in range(0, len(x), batch_size)
    ])


def lag_log_probs(draws: np.ndarray, indices: np.ndarray, beta: float) -> np.ndarray:
    """Log probabilities for each target from the PREVIOUS observed main draw."""
    _, log_z = overlap_distribution(beta)
    previous = draws[indices - 1]
    current = draws[indices]
    shared = np.any(current[:, :, None] == previous[:, None, :], axis=2).sum(axis=1)
    return beta * shared.astype(np.float64) - log_z


def run_historical(*, source: Path = SOURCE, seed: int = 649, window: int = 5,
                   hidden: int = 16, epochs: int = 40, patience: int = 6,
                   batch_size: int = 128, lr: float = 0.005,
                   weight_decay: float = 0.02, bootstrap_samples: int = 2000,
                   block_size: int = 13) -> dict:
    """Freeze all model parameters using train and validation, then score test."""
    if bootstrap_samples < 1 or block_size < 1:
        raise ValueError("invalid bootstrap settings")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    numbers, dates, ids = load_official_main_draws(source)
    train, val, test = chronological_split(len(numbers), window)
    history, targets, origins = make_observations(numbers, window)
    def select(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if not np.array_equal(origins[indices - window], indices):
            raise AssertionError("causal history indexing mismatch")
        chosen = torch.from_numpy(indices - window)
        return history[chosen], targets[chosen]

    train_x, train_y = select(train)
    val_x, val_y = select(val)
    test_x, test_y = select(test)
    decoder = FixedCardinalityDecoder(N, K)
    fitted = {}
    training = {}
    for name, model in (
        ("static", StaticLogitModel()),
        ("mlp", SmallMLP(window, hidden)),
    ):
        trained, trace = fit_model(
            model, x_train=train_x, y_train=train_y,
            x_val=val_x, y_val=val_y, seed=seed,
            epochs=epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay)
        fitted[name] = trained
        training[name] = trace

    # The echo coefficient is fitted on TRAIN ONLY, not on the test tail.
    echo_beta = fit_beta(numbers, train_end=int(train[-1]) + 1)
    val_scores = {
        "iid": float(IID_LOGP),
        "static": float(training["static"]["best_val_log_score"]),
        "mlp": float(training["mlp"]["best_val_log_score"]),
        "lag": float(lag_log_probs(numbers, val, echo_beta).mean()),
    }
    selected = choose_from_validation(val_scores)

    test_logps = {
        "iid": np.full(len(test), IID_LOGP),
        "lag": lag_log_probs(numbers, test, echo_beta),
        "static": model_log_probs(fitted["static"], decoder, test_x, test_y, batch_size),
        "mlp": model_log_probs(fitted["mlp"], decoder, test_x, test_y, batch_size),
    }
    for name, values in test_logps.items():
        if len(values) != len(test) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid test scores for {name}")
    metrics = {}
    for name in CANDIDATES:
        per_draw_lift = test_logps[name] - IID_LOGP
        metrics[name] = {
            "val_log_score": val_scores[name],
            "test_log_score": float(test_logps[name].mean()),
            "test_lift_nats": float(per_draw_lift.mean()),
            "test_block_bootstrap_95pct_ci_lift": moving_block_ci(
                per_draw_lift, block_size=block_size,
                samples=bootstrap_samples, seed=20261107),
            "positive_individual_draw_fraction": float(np.mean(per_draw_lift > 0)),
        }

    calendar_years = dates[test] // 10000
    years = []
    for year in np.unique(calendar_years):
        mask = calendar_years == year
        years.append({
            "year": int(year),
            "draws": int(mask.sum()),
            "selected_model": selected,
            "selected_lift_nats": float(
                (test_logps[selected][mask] - IID_LOGP).mean()),
            "model_lift_nats": {
                name: float((scores[mask] - IID_LOGP).mean())
                for name, scores in test_logps.items()
            },
        })

    return {
        "schema": "lotto649.neural.phase5.historical.v1",
        "version": VERSION,
        "research_status": "EXPLORATORY_HISTORICAL_REUSE_NOT_PROSPECTIVE_BLIND",
        "cautions": [
            "Historical draw patterns have been analyzed in prior experiments.",
            "The chronological test period is NOT a pristine prospective blind holdout.",
            "Model selection uses validation only; all test candidates are reported.",
            "One-step sequential evaluation can use earlier observed test draws as context.",
            "Model parameters are frozen throughout test; no within-test retraining.",
            "Block bootstrap intervals are descriptive and not multiple-testing corrected.",
            "Full-set log-score lift is not jackpot hit-rate or three-month hit probability.",
            "No tickets are selected and no financial advantage is inferred.",
        ],
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": int(len(numbers)),
        "first_draw": {"date": int(dates[0]), "draw_id": int(ids[0])},
        "last_draw": {"date": int(dates[-1]), "draw_id": int(ids[-1])},
        "config": {
            "seed": seed, "window": window, "hidden": hidden,
            "epochs": epochs, "patience": patience, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "bootstrap_samples": bootstrap_samples, "block_size": block_size,
        },
        "splits": {
            "train": {"indices": [int(train[0]), int(train[-1])],
                      "dates": [int(dates[train[0]]), int(dates[train[-1]])]},
            "validation": {"indices": [int(val[0]), int(val[-1])],
                           "dates": [int(dates[val[0]]), int(dates[val[-1]])]},
            "test": {"indices": [int(test[0]), int(test[-1])],
                     "dates": [int(dates[test[0]]), int(dates[test[-1]])]},
        },
        "train_only_echo_beta": float(echo_beta),
        "training": training,
        "validation_selected": selected,
        "validation_scores": val_scores,
        "test_models": metrics,
        "test_by_year": years,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only exploratory real-draw log-score backtest")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=13)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"historical source missing: {args.source}")
    result = run_historical(
        source=args.source, seed=args.seed, window=args.window,
        hidden=args.hidden, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples,
        block_size=args.block_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Lotto649 Historical One-Step Backtest v{VERSION}")
    print(f"Archive: {result['rows']} draws, {result['first_draw']['date']} to "
          f"{result['last_draw']['date']}")
    for stage, item in result["splits"].items():
        print(f"{stage:10s} {item['dates'][0]}..{item['dates'][1]} "
              f"indices={item['indices']}")
    print(f"Train-only echo beta: {result['train_only_echo_beta']:+.5f}")
    print(f"Validation-selected model: {result['validation_selected']}")
    for name in CANDIDATES:
        result_for_model = result["test_models"][name]
        interval = result_for_model["test_block_bootstrap_95pct_ci_lift"]
        print(f"  {name:6s} val={result_for_model['val_log_score']:+.6f} "
              f"test lift={result_for_model['test_lift_nats']:+.6f} "
              f"block CI95=[{interval[0]:+.6f},{interval[1]:+.6f}]")
    print(f"[DONE] {args.report.resolve()}")
    print("[CAUTION] Historical replay is exploratory, not a new blinded prediction")


if __name__ == "__main__":
    main()
