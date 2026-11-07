"""Pre-registered image loss vs. six-pixel decoder consistency study (v0.3).

The original v0.2 training code and its repository commit are frozen.
This module adds two loss variants without changing the CNN, GRU, dataset,
chronological split, or hard six-pixel decoder. Real-draw observations are
exploratory because the historical evaluation segment was inspected earlier.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import decode, encode, iid_expected, iid_images, similarity
from .diagnostics import infer_logits, logit_diagnostics
from .learning import (ImageSequenceLearner, autocorrelation, image_windows,
                       shape_loss, train_model)
from .models import diversity
from .research import DATA, calendar_years, load_history, split

SCHEMA = "lotto7x7.loss_study.v0.3"
V02_BASELINE_COMMIT = "0102c561530522dd0abae4fd4aa0b8fccef81a9e"
METHODS = ("original_v02", "balanced_bce", "projected_six_mass")


def projected_six_probabilities(logits: torch.Tensor, *, temperature: float = 1.0
                                ) -> torch.Tensor:
    """Differentiable mass-six sigmoid projection, preserving logit ranks.

    Solve sum(sigmoid((logit + bias) / temperature)) == 6 separately for each
    sample. A one-step implicit correction after detached root finding supplies
    the derivative of the bias, so changing one logit cannot change total mass
    to first order. The final hard decoder still takes the same six top logits.
    """
    if (logits.ndim != 3 or logits.shape[1:] != (7, 7)
            or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("expected finite floating [batch,7,7] logits and temperature > 0")
    z = logits.reshape(-1, 49)
    with torch.no_grad():
        lower = -z.max(dim=1).values - 24.0 * temperature
        upper = -z.min(dim=1).values + 24.0 * temperature
        for _ in range(42):
            midpoint = (lower + upper) / 2.0
            mass = torch.sigmoid((z + midpoint[:, None]) / temperature).sum(dim=1)
            lower = torch.where(mass < 6.0, midpoint, lower)
            upper = torch.where(mass >= 6.0, midpoint, upper)
        center = ((lower + upper) / 2.0)[:, None]
    # Differentiate the constrained bias via the implicit-function theorem.
    p0 = torch.sigmoid((z + center) / temperature)
    denominator = (p0 * (1.0 - p0)).sum(dim=1, keepdim=True).detach().clamp(min=1e-8)
    corrected_center = center + temperature * (6.0 - p0.sum(dim=1, keepdim=True)) / denominator
    return torch.sigmoid((z + corrected_center) / temperature).reshape(-1, 7, 7)


def shape_term(probabilities: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Use exactly the v0.2 D4/translation-aware autocorrelation geometry."""
    predicted = autocorrelation(probabilities)
    with torch.no_grad():
        actual = autocorrelation(observed)
        variants = torch.stack([
            variant
            for turns in range(4)
            for variant in (
                torch.rot90(actual, turns, dims=(-2, -1)),
                torch.flip(torch.rot90(actual, turns, dims=(-2, -1)), dims=(-1,)),
            )
        ], dim=1)
    return 169.0 * (predicted[:, None] - variants).square().mean(
        dim=(-2, -1)).min(dim=1).values.mean()


def loss_v03(logits: torch.Tensor, observed: torch.Tensor,
             *, method: str) -> torch.Tensor:
    """Two treatment arms; the original arm delegates to frozen v0.2 code.

    original_v02: exact v0.2 weighted BCE + shape + weak mass penalty.
    balanced_bce: no positive class weighting; mass penalty retained.
    projected_six_mass: calibrated soft six-point mass; hard Top-6 is unchanged.
    """
    if method not in METHODS:
        raise ValueError("unsupported loss variant")
    if (logits.ndim != 3 or logits.shape[1:] != (7, 7)
            or observed.shape != logits.shape
            or not bool(torch.isfinite(logits).all())
            or not bool(torch.isfinite(observed).all())
            or not bool(torch.all((observed == 0) | (observed == 1)))
            or not bool(torch.all(observed.sum(dim=(1, 2)) == 6))):
        raise ValueError("expected finite logits and exactly-six binary labels")
    if method == "original_v02":
        return shape_loss(logits, observed)
    if method == "balanced_bce":
        probabilities = torch.sigmoid(logits)
        mass_penalty = ((probabilities.sum(dim=(1, 2)) - 6.0) / 7.0).square().mean()
        return (0.35 * shape_term(probabilities, observed)
                + 0.60 * F.binary_cross_entropy_with_logits(logits, observed)
                + 0.05 * mass_penalty)
    probabilities = projected_six_probabilities(logits)
    # A small spatial MSE replaces the now-redundant six-mass penalty.
    return (0.35 * shape_term(probabilities, observed)
            + 0.60 * F.binary_cross_entropy(
                probabilities.clamp(1e-7, 1.0 - 1e-7), observed)
            + 0.05 * F.mse_loss(probabilities, observed))


def train_treatment(images: np.ndarray, *, method: str,
                    train_end: int, validation_end: int,
                    window: int = 5, embedding: int = 24, hidden: int = 32,
                    epochs: int = 45, patience: int = 8, batch_size: int = 64,
                    learning_rate: float = 0.003, seed: int = 649
                    ) -> tuple[ImageSequenceLearner, dict]:
    """Match original v0.2 architecture, initialization, optimizer and split."""
    if method not in METHODS[1:]:
        raise ValueError("treatment must be balanced_bce or projected_six_mass")
    if (not 2 <= window < train_end - 10
            or not train_end + 5 <= validation_end < len(images)
            or epochs < 1 or patience < 1 or batch_size < 1 or seed < 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid treatment training parameters")
    x, y = image_windows(images, window=window)
    train_count = train_end - window
    validation_count = validation_end - window
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ImageSequenceLearner(embedding=embedding, hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(train_count, generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_v03(model(x[indices]), y[indices], method=method)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [float(loss_v03(model(a), b, method=method).item())
                      for a, b in zip(x[train_count:validation_count].split(batch_size),
                                      y[train_count:validation_count].split(batch_size))]
        validation_loss = float(np.mean(losses))
        if not np.isfinite(validation_loss):
            raise RuntimeError("non-finite validation image loss")
        history.append({"epoch": epoch, "validation_image_shape_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("image treatment did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "architecture": "frozen_v02_cnn_encoder_and_gru",
        "objective": method,
        "best_epoch": best_epoch,
        "best_validation_image_shape_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "training_targets": [window, train_end-1],
        "validation_targets": [train_end, validation_end-1],
        "window": window,
    }


def _train_arms(images: np.ndarray, *, train_end: int, validation_end: int,
                window: int, embedding: int, hidden: int, epochs: int,
                patience: int, batch_size: int,
                learning_rate: float, seed: int) -> tuple[dict, dict]:
    args = dict(train_end=train_end, validation_end=validation_end,
                window=window, embedding=embedding, hidden=hidden,
                epochs=epochs, patience=patience, batch_size=batch_size,
                learning_rate=learning_rate, seed=seed)
    # This is the original, unchanged v0.2 function, not a reimplementation.
    original, original_fit = train_model(images, **args)
    balanced, balanced_fit = train_treatment(images, method=METHODS[1], **args)
    projected, projected_fit = train_treatment(images, method=METHODS[2], **args)
    return (dict(zip(METHODS, (original, balanced, projected))),
            dict(zip(METHODS, (original_fit, balanced_fit, projected_fit))))


def _score_images(images: np.ndarray, targets: np.ndarray,
                  logits: dict[str, np.ndarray], *, baseline: int,
                  seed: int, ids: np.ndarray | None = None,
                  dates: np.ndarray | None = None) -> tuple[dict, dict, list]:
    """Score with identical per-target IID images for all three arms."""
    if baseline < 2 or any(k not in logits for k in METHODS):
        raise ValueError("missing predictions or invalid baseline")
    from .diagnostics import decode_logits
    predictions = {key: decode_logits(logits[key]) for key in METHODS}
    raw = {key: [] for key in METHODS}
    residual = {key: [] for key in METHODS}
    rows = []
    for row, t in enumerate(targets):
        t = int(t)
        chance = iid_images(baseline, seed=seed + t*1000003)
        entry = {"index": t,
                 "draw_id": int(ids[t]) if ids is not None else None,
                 "year": int(calendar_years(ids[[t]])[0]) if ids is not None else None,
                 "date": int(dates[t]) if dates is not None and dates[t] else None,
                 "actual": decode(images[t]), "methods": {}}
        for method in METHODS:
            pred = predictions[method][row]
            sim = similarity(pred, images[t])
            chance_score = iid_expected(pred, chance)
            raw[method].append(float(sim["composite"]))
            residual[method].append(float(sim["composite"] - chance_score))
            entry["methods"][method] = {
                "predicted": decode(pred),
                "raw_similarity": sim,
                "iid_expected": chance_score,
                "chance_adjusted": residual[method][-1],
            }
        rows.append(entry)
    summary = {method: {
        "mean_raw_similarity": float(np.mean(raw[method])),
        "mean_iid_adjusted_similarity": float(np.mean(residual[method])),
        "unique_patterns": diversity(predictions[method]),
        "per_draw_adjusted": residual[method],
    } for method in METHODS}
    return summary, residual, rows


def paired_intervals(scores: dict[str, list[float]], *, iterations: int = 400,
                     block: int = 8, seed: int = 649) -> dict:
    """Exploratory paired moving-block confidence intervals, not blind tests."""
    if (iterations < 2 or block < 1 or seed < 0
            or any(k not in scores for k in METHODS)):
        raise ValueError("invalid interval arguments")
    n = len(scores[METHODS[0]])
    if not n or any(len(scores[k]) != n for k in METHODS):
        raise ValueError("paired score lengths disagree")
    rng = np.random.default_rng(seed)
    result = {}
    for alt in METHODS[1:]:
        d = np.asarray(scores[alt]) - np.asarray(scores[METHODS[0]])
        sampled = []
        for _ in range(iterations):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            positions = np.concatenate([((start + np.arange(block)) % n)
                                        for start in starts])[:n]
            sampled.append(float(d[positions].mean()))
        result[f"{alt}_minus_{METHODS[0]}"] = {
            "mean": float(d.mean()),
            "interval_95": [float(v) for v in np.quantile(sampled, [0.025, 0.975])],
            "iterations": iterations,
            "block_length": block,
            "note": "exploratory_not_prospective_blind",
        }
    return result


def _evaluate_soft_mass(logits: np.ndarray, *, projected: bool) -> float:
    """Soft mass is measured before the six-pixel hard decoder."""
    values = torch.as_tensor(logits, dtype=torch.float32)
    with torch.no_grad():
        probability = (projected_six_probabilities(values) if projected
                       else values.sigmoid())
    return float(probability.sum(dim=(1, 2)).mean().item())


def study(path: Path = DATA, *, year: int | None = None,
          window: int = 5, epochs: int = 45, patience: int = 8,
          batch_size: int = 64, learning_rate: float = 0.003,
          embedding: int = 24, hidden: int = 32,
          baseline: int = 32, bootstrap: int = 400,
          seed: int = 649) -> dict:
    """Freeze baseline v0.2 then compare losses on the same chronological data."""
    if baseline < 2 or bootstrap < 2 or seed < 0:
        raise ValueError("invalid loss-study configuration")
    path = Path(path)
    images, dates, draw_ids = load_history(path)
    train, val, test = split(len(images), window)
    targets = test if year is None else test[calendar_years(draw_ids[test]) == year]
    if not len(targets):
        raise ValueError("no evaluation images in requested year")
    arms, fits = _train_arms(
        images, train_end=int(val[0]), validation_end=int(test[0]),
        window=window, embedding=embedding, hidden=hidden,
        epochs=epochs, patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed)
    logits = {name: infer_logits(model, images, indices=targets,
                                window=window, batch_size=batch_size)
              for name, model in arms.items()}
    summary, per_draw, rows = _score_images(
        images, targets, logits, baseline=baseline, seed=seed,
        ids=draw_ids, dates=dates)
    diag = {name: logit_diagnostics(data) for name, data in logits.items()}
    for method in METHODS:
        diag[method]["effective_soft_mass"] = _evaluate_soft_mass(
            logits[method], projected=(method == "projected_six_mass"))
    return {
        "schema": SCHEMA,
        "status": "EXPLORATORY_PREVIOUSLY_INSPECTED_HISTORICAL_EVALUATION",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_v02_commit": V02_BASELINE_COMMIT,
        "baseline_v02_untouched": True,
        "comparison": "same_encoder_gru_and_top_six_decoder_only_loss_changes",
        "total_draws": len(images),
        "evaluation_draws": len(targets),
        "split": {"train_targets": [int(train[0]), int(train[-1])],
                  "validation_targets": [int(val[0]), int(val[-1])],
                  "evaluation_targets": [int(targets[0]), int(targets[-1])]},
        "config": {"year": year if year is not None else "all",
                   "window": window, "epochs": epochs, "patience": patience,
                   "batch_size": batch_size, "learning_rate": learning_rate,
                   "embedding": embedding, "hidden": hidden,
                   "baseline_per_draw": baseline,
                   "bootstrap_iterations": bootstrap, "seed": seed},
        "training": fits,
        "summary": summary,
        "logit_diagnostics": diag,
        "paired_intervals": paired_intervals(per_draw, iterations=bootstrap,
                                              seed=seed + 77),
        "rows": rows,
        "limitations": [
            "The test segment has been inspected and is not a blind forecast.",
            "A preserved v0.2 code path provides the original loss and decoder.",
            "The soft mass projection does not guarantee new hard patterns.",
            "Unweighted BCE and projected soft probabilities change the loss landscape.",
            "More unique outputs do not establish predictive skill.",
        ],
    }


def synthetic_control(*, kind: str, n: int = 120, seed: int = 649,
                      epochs: int = 45, window: int = 3,
                      embedding: int = 8, hidden: int = 12) -> dict:
    """Train all loss arms on seeded IID or explicitly learnable periodic masks.

    This is a capacity/sanity control, not a real-lottery edge experiment.
    """
    if kind not in ("iid", "periodic") or n < 100 or epochs < 1 or window < 2:
        raise ValueError("invalid synthetic control")
    if kind == "periodic":
        samples = ([1, 2, 3, 4, 5, 6],
                   [1, 8, 15, 22, 29, 36],
                   [1, 9, 17, 25, 33, 41],
                   [2, 11, 14, 23, 37, 49])
        patterns = np.stack([encode(np.asarray(a)) for a in samples])
        images = patterns[np.arange(n) % len(patterns)]
    else:
        images = iid_images(n, seed=seed + 100)
    train, val, test = split(n, window)
    arms, fits = _train_arms(images, train_end=int(val[0]),
                             validation_end=int(test[0]),
                             window=window, embedding=embedding, hidden=hidden,
                             epochs=epochs, patience=12, batch_size=16,
                             learning_rate=0.003, seed=seed)
    logits = {name: infer_logits(model, images, indices=test,
                                window=window, batch_size=16)
              for name, model in arms.items()}
    metrics, _scores, scored_rows = _score_images(images, test, logits, baseline=16, seed=seed)
    return {
        "kind": kind,
        "seed": seed, "samples": n, "evaluation_draws": len(test),
        "best_epochs": {name: fit["best_epoch"] for name, fit in fits.items()},
        "summary": {name: {
            "adjusted_mean": metrics[name]["mean_iid_adjusted_similarity"],
            "unique_patterns": metrics[name]["unique_patterns"]["unique"],
            "mean_image_shape_similarity": float(np.mean([
                similarity(encode(np.asarray(row["methods"][name]["predicted"])),
                           images[row["index"]])["shape"]
                for row in scored_rows
            ])),
        } for name in METHODS},
        "note": ("known_periodic_image_relation" if kind == "periodic"
                 else "independent_fair_six_pixel_draws"),
    }
