"""Lotto649 Phase 4.1: test temporal inductive bias, not larger networks.

An echo-aware neural distribution adds a trainable gain to the previous
draw's 49-bit indicator before the exact 49-choose-6 probability decoder.

  Echo-only: logits = alpha * last_draw_indicator
  Echo+MLP:  logits = alpha * last_draw_indicator + MLP(history)

The synthetic lag_echo process can be represented exactly by Echo-only.
This intentionally reveals the benefit of choosing a correct architecture;
it does NOT establish that actual lottery draws have a lag-echo effect.

Every model sees only previously observed draws. Artificial generator
strength is never given to a fitted model. One-step tests only.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lotto_neural import (
    IID_LOGP, N, K, SmallMLP, chronological_split, evaluate, fit_model,
    load_draws, make_observations, mean_log_score,
)
from lotto_robustness import summarize
from lotto_synthetic import generate
from lotto_temporal import fit_beta, generate_lag_echo, lag_mean_log_score
from lotto_torch_decoder import FixedCardinalityDecoder

VERSION = "0.1.0"
MODES = ("iid", "lag_echo")
METRICS = ("static", "mlp", "lag_model", "echo_only", "echo_residual",
           "echo_minus_mlp", "residual_minus_echo", "echo_minus_lag")


class EchoOnlyModel(nn.Module):
    """One learned coefficient multiplied by the last *observed* draw."""

    def __init__(self, window: int):
        super().__init__()
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.echo_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2 or history.shape[1] != self.window * N:
            raise ValueError("expected flattened [batch, window*49] history")
        return self.echo_gain * history[:, -N:]


class EchoResidualMLP(EchoOnlyModel):
    """Neural residual plus a direct echo connection.

    SmallMLP starts with all-zero output. Both the scalar and neural
    residual are fitted via joint complete-set log likelihood.
    """

    def __init__(self, window: int, hidden: int = 16):
        super().__init__(window)
        self.residual = SmallMLP(window, hidden)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return super().forward(history) + self.residual(history)


def run(*, seeds: int = 10, start_seed: int = 20000, draws: int = 2183,
        strength: float = 2.0, modes: tuple[str, ...] = MODES,
        window: int = 5, hidden: int = 16, epochs: int = 40,
        patience: int = 6, batch_size: int = 128, lr: float = 0.005,
        weight_decay: float = 0.02, bootstrap_samples: int = 2000) -> dict:
    """Paired fair-IID and lag-echo control over fresh independent seeds."""
    if seeds < 1 or start_seed < 0 or draws < 40:
        raise ValueError("invalid seeds, start_seed, or draws")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("modes must be a unique subset of iid, lag_echo")
    if not math.isfinite(strength) or strength < 0 or bootstrap_samples < 1:
        raise ValueError("invalid generator strength or bootstrap samples")

    rows = []
    decoder = FixedCardinalityDecoder(N, K)
    with tempfile.TemporaryDirectory(prefix="lotto649_echo_") as tmp:
        path = Path(tmp) / "observed_numbers_only.npz"
        for index in range(seeds):
            seed = start_seed + index
            for mode in modes:
                numbers = (generate(count=draws, mode="iid", seed=seed)[0]
                           if mode == "iid"
                           else generate_lag_echo(count=draws, seed=seed, strength=strength))
                np.savez_compressed(path, numbers=numbers)  # no oracle fields
                base = evaluate(data=path, seed=seed, window=window,
                                hidden=hidden, epochs=epochs, patience=patience,
                                batch_size=batch_size, lr=lr,
                                weight_decay=weight_decay)
                observed = load_draws(path)
                features, targets, _ = make_observations(observed, window)
                train_idx, val_idx, test_idx = chronological_split(len(observed), window)

                def pick(indices):
                    ids = torch.from_numpy(indices - window)
                    return features[ids], targets[ids]

                x_train, y_train = pick(train_idx)
                x_val, y_val = pick(val_idx)
                x_test, y_test = pick(test_idx)

                results = {}
                for name, model in (
                    ("echo_only", EchoOnlyModel(window)),
                    ("echo_residual", EchoResidualMLP(window, hidden)),
                ):
                    torch.manual_seed(seed)
                    fitted, info = fit_model(
                        model, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, seed=seed,
                        epochs=epochs, patience=patience,
                        batch_size=batch_size, lr=lr,
                        weight_decay=weight_decay)
                    logp = mean_log_score(fitted, decoder, x_test, y_test, batch_size)
                    results[name] = {
                        "lift": float(logp - IID_LOGP),
                        "best_epoch": int(info["best_epoch"]),
                        "fitted_echo_gain": float(fitted.echo_gain.detach().item()),
                    }

                fitted_beta = fit_beta(observed, int(train_idx[-1]) + 1)
                lag_lift = lag_mean_log_score(observed, test_idx, fitted_beta) - IID_LOGP
                mlp = float(base["models"]["mlp"]["test_lift_nats"])
                echo = results["echo_only"]["lift"]
                residual = results["echo_residual"]["lift"]
                row = {
                    "mode": mode, "seed": seed,
                    "draws_sha256": base["dataset_sha256"],
                    "train_range": base["train_range"],
                    "val_range": base["val_range"],
                    "test_range": base["test_range"],
                    "static": float(base["models"]["static"]["test_lift_nats"]),
                    "mlp": mlp,
                    "lag_model": float(lag_lift),
                    "fitted_lag_beta": float(fitted_beta),
                    "echo_only": echo,
                    "echo_residual": residual,
                    "echo_minus_mlp": float(echo - mlp),
                    "residual_minus_echo": float(residual - echo),
                    "echo_minus_lag": float(echo - lag_lift),
                    "echo_only_info": results["echo_only"],
                    "echo_residual_info": results["echo_residual"],
                }
                rows.append(row)
                print(
                    f"[{index + 1:02d}/{seeds}] {mode:8s} seed={seed} "
                    f"MLP={mlp:+.4f} echo={echo:+.4f} "
                    f"echo+MLP={residual:+.4f} lag={lag_lift:+.4f} "
                    f"alpha={results['echo_only']['fitted_echo_gain']:+.3f}",
                    flush=True,
                )

    summary = {}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            metric: summarize([r[metric] for r in subset],
                              bootstrap_seed=20261106 + j,
                              bootstrap_samples=bootstrap_samples)
            for j, metric in enumerate(METRICS)
        }
    return {
        "schema": "lotto649.neural.phase4.echo.v1",
        "version": VERSION,
        "purpose": "Inductive bias vs MLP capacity for an artificial Markov mechanism",
        "cautions": [
            "Artificial lag_echo generator is intentionally structurally matched to EchoOnly.",
            "Fitted coefficients see train labels only; model selection uses validation only.",
            "Comparisons of echo+MLP and echo-only on test are descriptive diagnostics.",
            "No claim of real draw predictability, first-hit improvement or wagering advantage.",
        ],
        "config": {
            "seeds": seeds, "start_seed": start_seed, "draws": draws,
            "modes": list(modes), "strength_generation_only": strength,
            "window": window, "hidden": hidden,
            "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "bootstrap_samples": bootstrap_samples,
        },
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.1 echo-aware inductive bias benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=20000)
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
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase4_echo_inductive_bias.json"))
    args = parser.parse_args()
    print(f"Lotto649 Phase 4.1 Inductive Bias v{VERSION}")
    result = run(
        seeds=args.seeds, start_seed=args.start_seed, draws=args.draws,
        strength=args.strength, modes=tuple(args.modes),
        window=args.window, hidden=args.hidden, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for mode, items in result["summary"].items():
        print(f"\n[{mode}] summary")
        for metric in METRICS:
            stats = items[metric]
            ci = stats["bootstrap_95pct_ci_mean_nats"]
            ci_label = "N/A" if ci is None else f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"
            print(f"  {metric:22s} mean={stats['mean_nats']:+.5f} "
                  f"CI95={ci_label} positive={stats['positive_fraction']:.0%}")
    print(f"\n[DONE] {args.report.resolve()}")
    print("[INFO] Synthetic one-step benchmark only; no ticket selection")


if __name__ == "__main__":
    main()
