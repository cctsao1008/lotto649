"""Lotto649 Architecture v0.2 Phase 3: pre-specified synthetic robustness study.

Run the SAME training/validation/test procedure over independent seeded
synthetic datasets. Study controls: fair IID and known fixed-weight bias.
No oracle synthetic parameters are exposed to the learner, no real lottery
predictions are made, and test outcomes never select training epochs.

Example:
    python lotto_robustness.py --seeds 10 --draws 2183
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from lotto_neural import evaluate
from lotto_synthetic import generate

VERSION = "0.1.0"
MODELS = ("static", "mlp")
MODES = ("iid", "fixed_bias")
METRICS = ("static", "mlp", "mlp_minus_static", "best_of_two_test_diagnostic")


def summarize(values: list[float], *, bootstrap_seed: int = 20261106,
              bootstrap_samples: int = 2000) -> dict:
    """Seed-level descriptive statistics and percentile bootstrap CI.

    Each seed represents one independently generated full experiment.
    Bootstrap draws *whole seeds*, not individual overlapping forecasts.
    The interval describes an experiment distribution, not jackpot odds.
    """
    if not values or bootstrap_samples < 1:
        raise ValueError("at least one value and one bootstrap sample required")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite scalars")
    n = len(array)
    result = {
        "replicates": n,
        "mean_nats": float(array.mean()),
        "median_nats": float(np.median(array)),
        "std_nats": float(array.std(ddof=1)) if n > 1 else None,
        "positive_fraction": float(np.mean(array > 0)),
        "min_nats": float(array.min()),
        "max_nats": float(array.max()),
        "bootstrap_95pct_ci_mean_nats": None,
    }
    if n > 1:
        rng = np.random.default_rng(bootstrap_seed)
        means = np.empty(bootstrap_samples, dtype=np.float64)
        for b in range(bootstrap_samples):
            means[b] = array[rng.integers(0, n, n)].mean()
        low, high = np.percentile(means, [2.5, 97.5])
        result["bootstrap_95pct_ci_mean_nats"] = [float(low), float(high)]
    return result


def row_scores(report: dict) -> dict:
    """Return per-seed test performance. The max is DIAGNOSTIC ONLY.

    Selecting a model by whichever has the best *test* score leaks
    holdout information. Never use max diagnostic as a deployable estimate.
    """
    lifts = {name: float(report["models"][name]["test_lift_nats"]) for name in MODELS}
    if not all(math.isfinite(v) for v in lifts.values()):
        raise ValueError("non-finite test lift")
    return {
        **lifts,
        "mlp_minus_static": lifts["mlp"] - lifts["static"],
        "best_of_two_test_diagnostic": max(lifts.values()),
    }


def aggregate(rows: list[dict], *, bootstrap_samples: int = 2000) -> dict:
    """Summarize each control using all seeds, including losing runs."""
    if not rows:
        raise ValueError("no experiment rows")
    result = {}
    for mode in MODES:
        part = [r for r in rows if r["mode"] == mode]
        if not part:
            continue
        result[mode] = {
            metric: summarize([float(row[metric]) for row in part],
                              bootstrap_seed=20261106 + i,
                              bootstrap_samples=bootstrap_samples)
            for i, metric in enumerate(METRICS)
        }
    return result


def run(*, seeds: int = 10, start_seed: int = 649, draws: int = 2183,
        modes: tuple[str, ...] = MODES, strength: float = 2.0,
        window: int = 5, hidden: int = 16, epochs: int = 40,
        patience: int = 6, batch_size: int = 128, lr: float = 0.005,
        weight_decay: float = 0.02, bootstrap_samples: int = 2000) -> dict:
    if seeds < 1 or start_seed < 0 or draws < 40:
        raise ValueError("require seeds>=1, start_seed>=0 and draws>=40")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("modes must be a nonempty unique subset of iid, fixed_bias")
    if not math.isfinite(strength) or strength < 0 or bootstrap_samples < 1:
        raise ValueError("invalid strength or bootstrap_samples")
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="lotto649_phase3_") as temp:
        file = Path(temp) / "draws_only.npz"
        for index in range(seeds):
            seed = start_seed + index
            for mode in modes:
                numbers, _oracle_phase, _oracle_logits = generate(
                    count=draws, mode=mode, seed=seed, strength=strength)
                # Crucial isolation: train/evaluate can open ONLY numbers.
                # Oracles stay in Python local variables, NEVER the NPZ payload.
                np.savez_compressed(file, numbers=numbers)
                report = evaluate(
                    data=file, seed=seed, window=window, hidden=hidden,
                    epochs=epochs, patience=patience, batch_size=batch_size,
                    lr=lr, weight_decay=weight_decay)
                measures = row_scores(report)
                row = {
                    "mode": mode, "seed": seed,
                    "test_range": report["test_range"],
                    "train_range": report["train_range"],
                    "val_range": report["val_range"],
                    "draws_sha256": report["dataset_sha256"],
                    **measures,
                    "best_epoch_static": report["models"]["static"]["best_epoch"],
                    "best_epoch_mlp": report["models"]["mlp"]["best_epoch"],
                }
                rows.append(row)
                print(f"[{index + 1:02d}/{seeds}] {mode:10s} seed={seed} "
                      f"static={measures['static']:+.6f} "
                      f"mlp={measures['mlp']:+.6f} nats", flush=True)
    return {
        "schema": "lotto649.neural.phase3.v1",
        "version": VERSION,
        "purpose": "Pre-specified synthetic IID negative / fixed-bias positive controls",
        "cautions": [
            "Only synthetic one-step complete-set log scores; no real-draw prediction.",
            "All selected hyperparameters and seeds are recorded; further exploratory runs require multiplicity caution.",
            "The best-of-two test maximum is an intentionally optimistic diagnostic, NOT an unbiased selected-model score.",
            "Positive lifts in an individual IID test can occur by chance.",
            "Bootstrap across independently generated experiment seeds; not across temporally overlapping forecasts.",
        ],
        "config": {
            "seeds": seeds, "start_seed": start_seed, "draws": draws,
            "modes": list(modes), "strength": strength,
            "window": window, "hidden": hidden, "epochs": epochs,
            "patience": patience, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "bootstrap_samples": bootstrap_samples,
        },
        "rows": rows,
        "summary": aggregate(rows, bootstrap_samples=bootstrap_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 multi-seed IID and bias controls")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=649)
    parser.add_argument("--draws", type=int, default=2183)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=Path("data/neural/phase3_robustness.json"))
    args = parser.parse_args()
    print(f"Lotto649 Phase 3 Robustness v{VERSION}")
    result = run(
        seeds=args.seeds, start_seed=args.start_seed, draws=args.draws,
        modes=tuple(args.modes), strength=args.strength, window=args.window,
        hidden=args.hidden, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for mode, measures in result["summary"].items():
        print(f"\n[{mode}] seeded experiment summary")
        for metric in METRICS:
            data = measures[metric]
            interval = data["bootstrap_95pct_ci_mean_nats"]
            ci = (f"[{interval[0]:+.6f}, {interval[1]:+.6f}]" if interval else "N/A")
            print(f"  {metric:29s} mean={data['mean_nats']:+.6f} "
                  f"CI95={ci} positive={data['positive_fraction']:.1%}")
    print(f"\n[DONE] {args.report.resolve()}")
    print("[INFO] The test-max metric is DIAGNOSTIC, not model-selection evidence")
    print("[INFO] No real lottery forecasts or ticket selection performed")


if __name__ == "__main__":
    main()
