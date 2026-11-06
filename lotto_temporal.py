"""Lotto649 v0.2 Phase 4: controlled temporal-learning experiment.

Synthetic lag-echo generator follows an EXACT six-number conditional distribution:
  P(S_t | S_{t-1}) proportional to exp(beta * |S_t intersect S_{t-1}|).

Here the last draw affects the next draw; unlike a fixed frequency bias,
this is truly time-dependent and learnable from observed history. We compare
IID, a static model, a small MLP and an explicit 1-parameter lag baseline.

No oracle synthetic parameters are supplied to learning/evaluation; fixed
candidate models and one-step proper log scoring only. Not a lottery forecast.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from lotto_neural import IID_LOGP, chronological_split, evaluate
from lotto_robustness import summarize
from lotto_synthetic import generate

VERSION = "0.1.0"
N, K = 49, 6
MODES = ("iid", "lag_echo")
MODELS = ("static", "mlp", "lag_model")
METRICS = ("static", "mlp", "lag_model", "mlp_minus_static", "mlp_minus_lag")
NUMBER_SPACE = np.arange(1, N + 1, dtype=np.uint8)
J = np.arange(K + 1, dtype=np.float64)
# Number of legal target sets with exactly j of the six previous numbers.
LOG_MULTIPLICITY = np.log(np.asarray(
    [math.comb(K, j) * math.comb(N - K, K - j) for j in range(K + 1)],
    dtype=np.float64,
))


def overlap_distribution(beta: float) -> tuple[np.ndarray, float]:
    """Exact probability distribution of next-draw overlap J in {0,...,6}."""
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    values = LOG_MULTIPLICITY + beta * J
    log_z = float(np.logaddexp.reduce(values))
    probabilities = np.exp(values - log_z)
    return probabilities, log_z


def expected_overlap(beta: float) -> float:
    p, _ = overlap_distribution(beta)
    return float(np.dot(J, p))


def generate_lag_echo(*, count: int, seed: int, strength: float = 2.0) -> np.ndarray:
    """Exact Markov sampling: boost every number drawn in the preceding draw.

    Generate overlap count j from its exact combinatorial distribution;
    then sample j numbers from the previous set and 6-j from its complement.
    Each complete subset at given overlap j is equiprobable, so the resulting
    conditional probability is exactly exp(beta*j)/Z(beta).
    """
    if count < 40 or seed < 0 or not math.isfinite(strength) or strength < 0:
        raise ValueError("count>=40, seed>=0, and finite strength>=0 required")
    rng = np.random.default_rng(seed)
    distribution, _ = overlap_distribution(strength)
    numbers = np.empty((count, K), dtype=np.uint8)
    numbers[0] = np.sort(rng.choice(NUMBER_SPACE, K, replace=False))
    for t in range(1, count):
        previous = numbers[t - 1]
        complement = np.setdiff1d(NUMBER_SPACE, previous, assume_unique=True)
        overlap = int(rng.choice(K + 1, p=distribution))
        preserved = rng.choice(previous, overlap, replace=False)
        fresh = rng.choice(complement, K - overlap, replace=False)
        numbers[t] = np.sort(np.concatenate((preserved, fresh)))
    return numbers


def overlaps(draws: np.ndarray) -> np.ndarray:
    if draws.ndim != 2 or draws.shape[1] != K or len(draws) < 2:
        raise ValueError("expected at least two six-number draws")
    return np.sum(
        np.any(draws[1:, :, None] == draws[:-1, None, :], axis=2),
        axis=1,
        dtype=np.int64,
    )


def fit_beta(draws: np.ndarray, train_end: int) -> float:
    """Train-only MLE of the 1-parameter lag-echo model.

    Maximize sum(beta * observed_overlap - log Z(beta)).
    The unique (interior) MLE matches E_beta[overlap] to training overlap mean.
    Bounds are fixed in advance to prevent extreme parameter estimates.
    """
    if train_end < 10 or train_end > len(draws):
        raise ValueError("train_end must lie between 10 and number of draws")
    target = float(np.mean(overlaps(draws[:train_end])))
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if expected_overlap(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def lag_mean_log_score(draws: np.ndarray, target_positions: np.ndarray,
                       beta: float) -> float:
    """One-step log score using the actually observed previous draw only."""
    ids = np.asarray(target_positions, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or ids.min() < 1 or ids.max() >= len(draws):
        raise ValueError("test positions must be within 1..len(draws)-1")
    _, log_z = overlap_distribution(beta)
    overlap_test = overlaps(draws)[ids - 1]
    return float(np.mean(beta * overlap_test - log_z))


def temporal_experiment(*, seeds: int = 10, start_seed: int = 10000,
                        count: int = 2183, strength: float = 2.0,
                        modes: tuple[str, ...] = MODES, window: int = 5,
                        hidden: int = 16, epochs: int = 40,
                        patience: int = 6, batch_size: int = 128,
                        lr: float = 0.005, weight_decay: float = 0.02,
                        bootstrap_samples: int = 2000) -> dict:
    if seeds < 1 or start_seed < 0 or count < 40:
        raise ValueError("invalid seeds, start_seed, or count")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("modes must be a unique nonempty subset of iid, lag_echo")
    if not math.isfinite(strength) or strength < 0 or bootstrap_samples < 1:
        raise ValueError("invalid strength or bootstrap samples")

    rows = []
    with tempfile.TemporaryDirectory(prefix="lotto649_phase4_") as temp:
        path = Path(temp) / "observed_numbers_only.npz"
        for i in range(seeds):
            seed = start_seed + i
            for mode in modes:
                if mode == "iid":
                    draws = generate(count=count, mode="iid", seed=seed)[0]
                else:
                    draws = generate_lag_echo(count=count, seed=seed, strength=strength)
                # Only observed numbers enter the neural pipeline.
                np.savez_compressed(path, numbers=draws)
                result = evaluate(data=path, seed=seed, window=window, hidden=hidden,
                                  epochs=epochs, patience=patience, batch_size=batch_size,
                                  lr=lr, weight_decay=weight_decay)
                train, _val, test = chronological_split(count, window)
                beta_hat = fit_beta(draws, train_end=int(train[-1]) + 1)
                lag_lift = lag_mean_log_score(draws, test, beta_hat) - IID_LOGP
                static_lift = result["models"]["static"]["test_lift_nats"]
                mlp_lift = result["models"]["mlp"]["test_lift_nats"]
                row = {
                    "mode": mode, "seed": seed,
                    "draws_sha256": result["dataset_sha256"],
                    "train_range": result["train_range"],
                    "val_range": result["val_range"],
                    "test_range": result["test_range"],
                    "fitted_lag_beta": float(beta_hat),
                    "empirical_train_overlap": float(np.mean(overlaps(draws[:train[-1] + 1]))),
                    "static": float(static_lift),
                    "mlp": float(mlp_lift),
                    "lag_model": float(lag_lift),
                    "mlp_minus_static": float(mlp_lift - static_lift),
                    "mlp_minus_lag": float(mlp_lift - lag_lift),
                    "best_epoch_static": result["models"]["static"]["best_epoch"],
                    "best_epoch_mlp": result["models"]["mlp"]["best_epoch"],
                }
                rows.append(row)
                print(f"[{i + 1:02d}/{seeds}] {mode:8s} seed={seed} "
                      f"static={static_lift:+.5f} mlp={mlp_lift:+.5f} "
                      f"lag={lag_lift:+.5f} fitted_beta={beta_hat:+.3f}", flush=True)

    summary = {}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            metric: summarize([row[metric] for row in subset],
                              bootstrap_seed=20261106 + j,
                              bootstrap_samples=bootstrap_samples)
            for j, metric in enumerate(METRICS)
        }
    return {
        "schema": "lotto649.neural.phase4.temporal.v1",
        "version": VERSION,
        "objective": "Detect genuine one-step temporal dependencies on synthetic data",
        "cautions": [
            "Lag-echo artificial process differs from fair independent lottery draws.",
            "Oracle strength is for generation only; model sees observed numbers.",
            "Known one-parameter lag baseline matches synthetic family by design.",
            "Past observed test draws are allowed as one-step context, not as training targets.",
            "No future multi-month forecasts, betting, or lottery advantage demonstrated.",
        ],
        "config": {
            "seeds": seeds, "start_seed": start_seed, "count": count,
            "strength_generation_only": strength, "modes": list(modes),
            "window": window, "hidden": hidden,
            "epochs": epochs, "patience": patience, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "bootstrap_samples": bootstrap_samples,
        },
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 temporal dependency positive control")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--draws", type=int, default=2183)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=Path("data/neural/phase4_temporal.json"))
    args = parser.parse_args()
    print(f"Lotto649 Phase 4 Temporal Control v{VERSION}")
    report = temporal_experiment(
        seeds=args.seeds, start_seed=args.start_seed, count=args.draws,
        strength=args.strength, modes=tuple(args.modes),
        window=args.window, hidden=args.hidden, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for mode, results in report["summary"].items():
        print(f"\n[{mode}] summary")
        for metric in METRICS:
            statistics = results[metric]
            ci = statistics["bootstrap_95pct_ci_mean_nats"]
            ci_text = "N/A" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
            print(f"  {metric:20s} mean={statistics['mean_nats']:+.6f} "
                  f"CI95={ci_text} positive={statistics['positive_fraction']:.1%}")
    print(f"\n[DONE] {args.report.resolve()}")
    print("[INFO] Synthetic one-step temporal evaluation only; no ticket selection")


if __name__ == "__main__":
    main()
