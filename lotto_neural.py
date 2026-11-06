"""Lotto649 Architecture v0.2 Phase 2: reproducible small neural learning.

Train and evaluate static 49-logit and small MLP models against IID on
synthetic six-number draws. Data are split in chronological order; each
target's features contain only previous draws. Synthetic oracle phases/logits
are intentionally never loaded. This is ONE-STEP distribution forecasting,
not a 3-month jackpot-hit forecast or ticket-selection algorithm.

Install separately: python -m pip install torch
Example: python lotto_neural.py --data data/synthetic/fixed_bias_n2183_seed649.npz
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lotto_torch_decoder import FixedCardinalityDecoder

VERSION = "0.1.0"
N, K = 49, 6
IID_LOGP = -math.log(math.comb(N, K))


def load_draws(path: Path) -> np.ndarray:
    """Load ONLY the observed numbers, never synthetic phase/parameter oracles."""
    with np.load(path, allow_pickle=False) as archive:
        if "numbers" not in archive:
            raise ValueError("expected synthetic .npz with a 'numbers' array")
        draws = archive["numbers"]
    if draws.ndim != 2 or draws.shape[1] != K or draws.shape[0] < 40:
        raise ValueError("require at least 40 six-number draws")
    if not np.issubdtype(draws.dtype, np.integer):
        raise ValueError("numbers must be integers")
    if not np.all((draws >= 1) & (draws <= N)) or not np.all(np.diff(draws, axis=1) > 0):
        raise ValueError("each draw must be sorted, unique, and in 1..49")
    return draws.astype(np.int64, copy=True)


def make_observations(draws: np.ndarray, window: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """At target t: x[t] consists *exclusively* of draws [t-window, t)."""
    if not isinstance(window, int) or window < 1 or window >= len(draws):
        raise ValueError("window must be 1..(draw count - 1)")
    binary = np.zeros((len(draws), N), dtype=np.float32)
    binary[np.arange(len(draws))[:, None], draws - 1] = 1.0
    origins = np.arange(window, len(draws), dtype=np.int64)
    history = np.stack([binary[t - window:t].reshape(-1) for t in origins])
    targets = draws[origins]
    return torch.from_numpy(history), torch.from_numpy(targets.copy()), origins


def chronological_split(n: int, window: int, train_fraction: float = 0.6,
                        val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absolute draw positions, not a random split of overlapping sequences."""
    if n < 40 or window < 1 or window >= n:
        raise ValueError("invalid draw count or window")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("train/validation fractions must leave a test section")
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    if train_end - window < 10 or val_end - train_end < 10 or n - val_end < 10:
        raise ValueError("each split needs at least 10 targets")
    return (np.arange(window, train_end, dtype=np.int64),
            np.arange(train_end, val_end, dtype=np.int64),
            np.arange(val_end, n, dtype=np.int64))


class StaticLogitModel(nn.Module):
    """Trainable time-invariant marginal log-weights (statistical reference)."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(len(features), -1)


class SmallMLP(nn.Module):
    """A deliberately small causal feature model, with an IID initial output."""

    def __init__(self, window: int, hidden: int = 16):
        super().__init__()
        if window < 1 or hidden < 1:
            raise ValueError("window/hidden must be positive")
        self.layers = nn.Sequential(
            nn.Linear(window * N, hidden),
            nn.Tanh(),
            nn.Linear(hidden, N),
        )
        # Beginning at the IID baseline makes early-training behavior auditable.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


@torch.no_grad()
def mean_log_score(model: nn.Module, decoder: FixedCardinalityDecoder,
                   x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        total += decoder(logits, y[i:i + batch_size]).sum().item()
    return total / len(x)


def fit_model(model: nn.Module, *, x_train: torch.Tensor, y_train: torch.Tensor,
              x_val: torch.Tensor, y_val: torch.Tensor, seed: int,
              epochs: int, batch_size: int, lr: float, weight_decay: float,
              patience: int) -> tuple[nn.Module, dict]:
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("invalid optimization settings")
    decoder = FixedCardinalityDecoder(N, K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_validation = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = torch.randperm(len(x_train), generator=generator)
        for ids in shuffled.split(batch_size):
            logits = model(x_train[ids])
            loss = -decoder(logits, y_train[ids]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        validation = mean_log_score(model, decoder, x_val, y_val, batch_size)
        if not math.isfinite(validation):
            raise RuntimeError("non-finite validation log score")
        if validation > best_validation + 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no valid training epoch")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch,
                   "best_val_log_score": best_validation,
                   "best_val_lift_nats": best_validation - IID_LOGP}


def evaluate(*, data: Path, seed: int = 649, window: int = 5,
             hidden: int = 16, epochs: int = 40, patience: int = 6,
             batch_size: int = 128, lr: float = 0.005,
             weight_decay: float = 0.02) -> dict:
    if hidden < 1:
        raise ValueError("hidden size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small CPU workloads can become slower with excessive thread oversubscription.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    draws = load_draws(data)
    train_ids, val_ids, test_ids = chronological_split(len(draws), window)
    features, labels, origins = make_observations(draws, window)
    # Sample index equals target absolute index minus the history window.
    train_sel = torch.as_tensor(train_ids - window)
    val_sel = torch.as_tensor(val_ids - window)
    test_sel = torch.as_tensor(test_ids - window)
    assert np.array_equal(origins[train_sel.numpy()], train_ids)
    x_train, y_train = features[train_sel], labels[train_sel]
    x_val, y_val = features[val_sel], labels[val_sel]
    x_test, y_test = features[test_sel], labels[test_sel]

    results: dict = {}
    decoder = FixedCardinalityDecoder(N, K)
    for name, model in (("static", StaticLogitModel()), ("mlp", SmallMLP(window, hidden))):
        fitted, training = fit_model(
            model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
            seed=seed, epochs=epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay)
        test_log_score = mean_log_score(fitted, decoder, x_test, y_test, batch_size)
        results[name] = {**training, "test_log_score": test_log_score,
                         "test_lift_nats": test_log_score - IID_LOGP}
    return {
        "schema": "lotto649.neural.phase2.v1",
        "dataset": str(data),
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "note": ("One-step synthetic log score only. Oracle parameters/phases not loaded. "
                 "A single positive test score is not evidence of future lottery predictability."),
        "seed": seed, "draws": len(draws), "window": window,
        "hidden": hidden, "epochs_max": epochs, "patience": patience,
        "train_range": [int(train_ids[0]), int(train_ids[-1])],
        "val_range": [int(val_ids[0]), int(val_ids[-1])],
        "test_range": [int(test_ids[0]), int(test_ids[-1])],
        "iid": {"test_log_score": IID_LOGP, "test_lift_nats": 0.0},
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: static vs small MLP vs fair IID")
    parser.add_argument("--data", type=Path,
                        default=Path("data/synthetic/fixed_bias_n2183_seed649.npz"))
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"data not found: {args.data} (generate it with lotto_synthetic.py)")
    result = evaluate(data=args.data, seed=args.seed, window=args.window,
                      hidden=args.hidden, epochs=args.epochs, patience=args.patience,
                      batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    report = args.report or Path("data/neural") / f"{args.data.stem}_phase2_seed{args.seed}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"Lotto649 Neural Learning v{VERSION}")
    print(f"Dataset: {args.data} | draws={result['draws']} | window={result['window']}")
    print(f"Splits: train={result['train_range']} val={result['val_range']} test={result['test_range']}")
    print(f"IID  test LogScore={IID_LOGP:+.6f} | Lift={0.0:+.6f} nats")
    for name, scores in result["models"].items():
        print(f"{name:6s} test LogScore={scores['test_log_score']:+.6f} "
              f"| Lift={scores['test_lift_nats']:+.6f} nats "
              f"| best epoch={scores['best_epoch']}")
    print(f"[DONE] {report.resolve()}")
    print("[INFO] A positive lift on a single dataset does not establish statistical significance")


if __name__ == "__main__":
    main()
