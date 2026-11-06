"""Lotto649 Phase 4.2: optimizer-convergence and safe-residual audit.

Why: on the Phase 4.1 synthetic lag-echo generator, one-scalar EchoOnly
approached the exact train-only lag MLE, but 40 AdamW epochs left its gain
below the optimum. A jointly trained Echo+MLP severely underperformed.

This experiment compares (A) original AdamW EchoOnly, (B) train-only
maximum-likelihood echo, and (C) an MLE-initialized, FROZEN echo connection
plus a residual neural network. The residual is accepted only when its
validation log score improves over the zero-residual baseline.

All results are synthetic one-step forecasts. No full-six picks, monetary
decisions or claim of real-draw predictability.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from lotto_echo import EchoOnlyModel, EchoResidualMLP
from lotto_neural import (IID_LOGP, K, N, chronological_split, fit_model,
                          make_observations, mean_log_score)
from lotto_robustness import summarize
from lotto_synthetic import generate
from lotto_temporal import fit_beta, generate_lag_echo, lag_mean_log_score
from lotto_torch_decoder import FixedCardinalityDecoder

VERSION = "0.1.0"
MODES = ("iid", "lag_echo")
METRICS = ("legacy_echo", "mle_echo", "staged_residual",
           "mle_minus_legacy", "staged_minus_mle")


def fit_staged_residual(
    beta: float, *, window: int, hidden: int,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    seed: int, epochs: int = 40, patience: int = 6,
    batch_size: int = 128, lr: float = 0.005,
    weight_decay: float = 0.02, min_delta: float = 1e-4,
) -> tuple[EchoResidualMLP, dict]:
    """Fit only residual, keep beta fixed; epoch zero is a valid baseline.

    Model selection happens only on chronological validation data.
    The residual can be reverted to zero when it offers no validated gain.
    This safeguard is NOT a mathematical guarantee on unseen test data.
    """
    if not math.isfinite(beta) or window < 1 or hidden < 1:
        raise ValueError("invalid beta/window/hidden")
    if epochs < 1 or patience < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0 or min_delta < 0:
        raise ValueError("invalid optimization settings")
    if min(len(x_train), len(y_train), len(x_val), len(y_val)) < 1:
        raise ValueError("train and validation must be nonempty")
    if not (len(x_train) == len(y_train) and len(x_val) == len(y_val)):
        raise ValueError("features and targets must align")
    torch.manual_seed(seed)
    model = EchoResidualMLP(window, hidden)
    with torch.no_grad():
        model.echo_gain.fill_(beta)
    model.echo_gain.requires_grad_(False)
    decoder = FixedCardinalityDecoder(N, K)
    baseline_val = mean_log_score(model, decoder, x_val, y_val, batch_size)
    if not math.isfinite(baseline_val):
        raise RuntimeError("non-finite baseline log score")
    best_val, best_epoch = baseline_val, 0
    best_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.residual.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    sampler = torch.Generator().manual_seed(seed)
    stale = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        for ids in torch.randperm(len(x_train), generator=sampler).split(batch_size):
            loss = -decoder(model(x_train[ids]), y_train[ids]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual.parameters(), max_norm=5)
            optimizer.step()
        val = mean_log_score(model, decoder, x_val, y_val, batch_size)
        if not math.isfinite(val):
            raise RuntimeError("non-finite validation log score")
        if val > best_val + min_delta:
            best_val, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    assert float(model.echo_gain.item()) == float(torch.tensor(beta).item())
    return model, {
        "baseline_val_log_score": float(baseline_val),
        "best_val_log_score": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_run": int(last_epoch),
        "selected_residual": bool(best_epoch > 0),
        "validation_improvement_nats": float(best_val - baseline_val),
    }


def trial(*, draws: np.ndarray, seed: int,
          window: int = 5, hidden: int = 16,
          epochs: int = 40, patience: int = 6,
          batch_size: int = 128, lr: float = 0.005,
          weight_decay: float = 0.02, min_delta: float = 1e-4) -> dict:
    """Evaluate frozen candidates on a pre-defined holdout.

    Synthetic oracle beta is never an input here.
    """
    # Reuse the same strict 49-bit historical encoding as Phase 2.
    x, y, _ = make_observations(draws.astype(np.int64), window)
    train_idx, val_idx, test_idx = chronological_split(len(draws), window)
    def select(ids):
        positions = torch.from_numpy(ids - window)
        return x[positions], y[positions]
    x_train, y_train = select(train_idx)
    x_val, y_val = select(val_idx)
    x_test, y_test = select(test_idx)

    torch.manual_seed(seed)
    legacy, legacy_info = fit_model(
        EchoOnlyModel(window),
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val, seed=seed, epochs=epochs,
        patience=patience, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )
    decoder = FixedCardinalityDecoder(N, K)
    legacy_lift = mean_log_score(legacy, decoder, x_test, y_test, batch_size) - IID_LOGP

    beta = fit_beta(draws, int(train_idx[-1]) + 1)
    mle_lift = lag_mean_log_score(draws, test_idx, beta) - IID_LOGP
    # Cross-implementation equality: exact same subset distribution.
    mle_model = EchoOnlyModel(window)
    with torch.no_grad():
        mle_model.echo_gain.fill_(beta)
    torch_equivalent = mean_log_score(mle_model, decoder, x_test, y_test, batch_size) - IID_LOGP
    if abs(torch_equivalent - mle_lift) > 1e-5:
        raise AssertionError("PyTorch MLE echo and analytical lag probability disagree")

    staged, staged_info = fit_staged_residual(
        beta, window=window, hidden=hidden,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        seed=seed, epochs=epochs, patience=patience,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        min_delta=min_delta,
    )
    staged_lift = mean_log_score(staged, decoder, x_test, y_test, batch_size) - IID_LOGP
    return {
        "seed": int(seed),
        "train_range": [int(train_idx[0]), int(train_idx[-1])],
        "val_range": [int(val_idx[0]), int(val_idx[-1])],
        "test_range": [int(test_idx[0]), int(test_idx[-1])],
        "fitted_beta": float(beta),
        "legacy_echo_gain": float(legacy.echo_gain.item()),
        "legacy_best_epoch": int(legacy_info["best_epoch"]),
        "legacy_echo": float(legacy_lift),
        "mle_echo": float(mle_lift),
        "staged_residual": float(staged_lift),
        "mle_minus_legacy": float(mle_lift - legacy_lift),
        "staged_minus_mle": float(staged_lift - mle_lift),
        "staged_training": staged_info,
    }


def run(*, seeds: int = 10, start_seed: int = 30000, count: int = 2183,
        strength: float = 2.0, modes: tuple[str, ...] = MODES,
        window: int = 5, hidden: int = 16, epochs: int = 40,
        patience: int = 6, batch_size: int = 128,
        lr: float = 0.005, weight_decay: float = 0.02,
        min_delta: float = 1e-4, bootstrap_samples: int = 2000) -> dict:
    if seeds < 1 or start_seed < 0 or count < 40:
        raise ValueError("invalid seeds/start_seed/count")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("modes must be unique subset of iid,lag_echo")
    if not math.isfinite(strength) or strength < 0 or bootstrap_samples < 1:
        raise ValueError("invalid strength/bootstrap_samples")
    rows = []
    for i in range(seeds):
        seed = start_seed + i
        for mode in modes:
            draws = (generate(count=count, mode="iid", seed=seed)[0]
                     if mode == "iid"
                     else generate_lag_echo(count=count, seed=seed, strength=strength))
            item = trial(
                draws=draws, seed=seed, window=window, hidden=hidden,
                epochs=epochs, patience=patience,
                batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                min_delta=min_delta)
            rows.append({"mode": mode, **item})
            print(f"[{i + 1:02d}/{seeds}] {mode:8s} seed={seed} "
                  f"legacy={item['legacy_echo']:+.5f} "
                  f"MLE={item['mle_echo']:+.5f} "
                  f"staged={item['staged_residual']:+.5f} "
                  f"beta={item['fitted_beta']:+.3f} "
                  f"residual={'ON' if item['staged_training']['selected_residual'] else 'OFF'}",
                  flush=True)
    summary = {}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            metric: summarize([r[metric] for r in subset],
                              bootstrap_seed=20261106 + j,
                              bootstrap_samples=bootstrap_samples)
            for j, metric in enumerate(METRICS)
        }
        summary[mode]["residual_selected_fraction"] = float(np.mean(
            [r["staged_training"]["selected_residual"] for r in subset]))
    return {
        "schema": "lotto649.neural.phase4.staged.v1",
        "version": VERSION,
        "cautions": [
            "Experiments are synthetic and forecast only one step ahead.",
            "Echo MLE is the correctly specified reference for artificial lag_echo.",
            "Validation-based residual selection is not a test-set guarantee.",
            "No real lottery predictability or shorter jackpot waiting time is established.",
        ],
        "config": {
            "seeds": seeds, "start_seed": start_seed, "count": count,
            "strength_generation_only": strength, "modes": list(modes),
            "window": window, "hidden": hidden, "epochs": epochs,
            "patience": patience, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay, "min_delta": min_delta,
        },
        "rows": rows, "summary": summary,
    }


def main():
    p = argparse.ArgumentParser(description="Phase 4.2 echo convergence and staged residual")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=30000)
    p.add_argument("--draws", type=int, default=2183)
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--report", type=Path,
                   default=Path("data/neural/phase4_staged_residual.json"))
    args = p.parse_args()
    print(f"Lotto649 Phase 4.2 Staged Residual v{VERSION}")
    result = run(
        seeds=args.seeds, start_seed=args.start_seed, count=args.draws,
        strength=args.strength, modes=tuple(args.modes), window=args.window,
        hidden=args.hidden, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        min_delta=args.min_delta, bootstrap_samples=args.bootstrap_samples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for mode, statistics in result["summary"].items():
        print(f"\n[{mode}] summary "
              f"residual-selected={statistics['residual_selected_fraction']:.0%}")
        for metric in METRICS:
            data = statistics[metric]
            ci = data["bootstrap_95pct_ci_mean_nats"]
            label = "N/A" if ci is None else f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"
            print(f"  {metric:20s} mean={data['mean_nats']:+.5f} "
                  f"CI95={label} positive={data['positive_fraction']:.0%}")
    print(f"\n[DONE] {args.report.resolve()}")
    print("[INFO] Synthetic one-step audit; no ticket selection")


if __name__ == "__main__":
    main()
