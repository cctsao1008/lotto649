"""Image-learning v0.2 diagnostics: output collapse and temporal ablations.

All comparison arms use the same predeclared loss, chronology boundaries,
validation-only early stopping, random seeds, and paired conditional-IID
controls. The 2023-2026 evaluation archive is exploratory, not blind.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import random

import numpy as np
import torch
from torch import nn

from .core import decode, iid_expected, iid_images, similarity
from .learning import (ImageSequenceLearner, decode_image, image_windows,
                       permute_history_windows, shape_loss, train_model)
from .models import diversity
from .research import DATA, calendar_years, load_history, split

METHODS = ("ordered_cnn_gru", "shuffled_cnn_gru", "constant_template")
SCHEMA = "lotto7x7.image_diagnostics.v0.2"


class ConstantTemplate(nn.Module):
    """Learn one 7x7 logit field; never inspect any historical input pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.full((7, 7), -2.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] < 2 or images.shape[2:] != (7, 7):
            raise ValueError("input shape must be [batch,time>=2,7,7]")
        return self.logits.expand(images.shape[0], -1, -1)


def train_constant_template(images: np.ndarray, *, train_end: int,
                            validation_end: int, window: int = 5,
                            epochs: int = 45, patience: int = 8,
                            batch_size: int = 64, learning_rate: float = 0.003,
                            seed: int = 649) -> tuple[ConstantTemplate, dict]:
    """Same optimizer/loss/early-stop protocol as the sequence model."""
    if (not 2 <= window < train_end - 10
            or not train_end + 5 <= validation_end < len(images)
            or epochs < 1 or patience < 1 or batch_size < 1 or seed < 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid constant-control training parameters")
    x, y = image_windows(images, window=window)
    tr_end = train_end - window
    va_end = validation_end - window
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ConstantTemplate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(tr_end, generator=torch.Generator().manual_seed(seed + epoch))
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = shape_loss(model(x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [shape_loss(model(a), b).item()
                      for a, b in zip(x[tr_end:va_end].split(batch_size),
                                      y[tr_end:va_end].split(batch_size))]
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "validation_image_shape_loss": mean_loss})
        if mean_loss < best_loss - 1e-8:
            best_loss, best_epoch = mean_loss, epoch
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("constant control did not produce a valid epoch")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "architecture": "trainable_constant_7x7_logits_without_history",
        "history_order": "no_history",
        "best_epoch": best_epoch,
        "best_validation_image_shape_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "training_targets": [window, train_end - 1],
        "validation_targets": [train_end, validation_end - 1],
        "window": window,
        "objective": "same_shape_loss_as_cnn_gru",
    }


def infer_logits(model: nn.Module, images: np.ndarray, *, indices: np.ndarray,
                 window: int, batch_size: int = 64,
                 history_order: str = "chronological",
                 permutation_seed: int = 649) -> np.ndarray:
    """Infer logits using only past frames; never include target pixels."""
    idx = np.asarray(indices, dtype=np.int64)
    if (idx.ndim != 1 or len(idx) < 1 or idx.min() < window
            or idx.max() >= len(images) or batch_size < 1
            or history_order not in ("chronological", "permuted")):
        raise ValueError("invalid diagnostic inference arguments")
    frames = torch.as_tensor(
        np.stack([images[int(t)-window:int(t)] for t in idx]),
        dtype=torch.float32,
    )
    if history_order == "permuted":
        frames = permute_history_windows(frames, idx, seed=permutation_seed)
    out = []
    model.eval()
    with torch.no_grad():
        for part in frames.split(batch_size):
            out.append(model(part).detach().cpu().numpy())
    result = np.concatenate(out)
    if result.shape != (len(idx), 7, 7) or not np.all(np.isfinite(result)):
        raise ValueError("invalid trained image logits")
    return result


def decode_logits(logits: np.ndarray) -> np.ndarray:
    """Convert [n,7,7] logits to legal six-pixel images by stable top-six."""
    x = np.asarray(logits)
    if x.ndim != 3 or x.shape[1:] != (7, 7) or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("invalid prediction logits")
    return np.stack([decode_image(torch.as_tensor(z)) for z in x])


def logit_diagnostics(logits: np.ndarray) -> dict:
    """Measure continuous-output variation independently of six-pixel decoding."""
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (7, 7) or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("logits must be finite [n,7,7] images")
    values = array.reshape(len(array), 49)
    sorted_values = np.sort(values, axis=1)
    gaps = sorted_values[:, -6] - sorted_values[:, -7]
    masks = decode_logits(array)
    selected = masks.reshape(len(array), 49).sum(axis=0)
    stats = diversity(masks)
    standard_deviation = values.std(axis=0)
    patterns = Counter(tuple(int(n) for n in np.flatnonzero(mask) + 1)
                       for mask in masks.reshape(len(array), 49))
    return {
        "mean_pixel_logit_std": float(standard_deviation.mean()),
        "max_pixel_logit_std": float(standard_deviation.max()),
        "mean_top6_top7_margin": float(gaps.mean()),
        "min_top6_top7_margin": float(gaps.min()),
        "median_top6_top7_margin": float(np.median(gaps)),
        "mean_sigmoid_mass": float((1.0 / (1.0 + np.exp(-np.clip(array, -80, 80)))).sum(axis=(1, 2)).mean()),
        "unique_patterns": stats,
        "always_selected_numbers": (np.flatnonzero(selected == len(array)) + 1).astype(int).tolist(),
        "selection_counts_by_number": selected.astype(int).tolist(),
        "consecutive_top6_change_rate": (float(np.mean(np.any(masks[1:] != masks[:-1], axis=(1, 2))))
                                        if len(array) > 1 else 0.0),
        "top_patterns": [{"numbers": list(pattern), "count": count}
                         for pattern, count in patterns.most_common(10)],
    }


def latent_diagnostics(model: nn.Module, images: np.ndarray, *, indices: np.ndarray,
                       window: int, batch_size: int = 64,
                       history_order: str = "chronological",
                       permutation_seed: int = 649) -> dict:
    """Compare CNN embeddings and GRU hidden-state variation across contexts."""
    if not isinstance(model, ImageSequenceLearner):
        return {"available": False, "reason": "no_sequence_encoder"}
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1 or not len(idx) or idx.min() < window or idx.max() >= len(images):
        raise ValueError("invalid latent index range")
    frames = torch.as_tensor(
        np.stack([images[int(t)-window:int(t)] for t in idx]),
        dtype=torch.float32,
    )
    if history_order == "permuted":
        frames = permute_history_windows(frames, idx, seed=permutation_seed)
    features, states = [], []
    with torch.no_grad():
        model.eval()
        for batch in frames.split(batch_size):
            n, w = batch.shape[:2]
            emb = model.encoder(batch.reshape(n*w, 1, 7, 7)).reshape(n, w, -1)
            output, _ = model.temporal(emb)
            features.append(emb[:, -1].cpu().numpy())
            states.append(output[:, -1].cpu().numpy())
    embs = np.concatenate(features)
    hidden = np.concatenate(states)
    return {
        "available": True,
        "final_frame_embedding_mean_component_std": float(embs.std(axis=0).mean()),
        "gru_final_hidden_mean_component_std": float(hidden.std(axis=0).mean()),
        "gru_final_hidden_max_component_std": float(hidden.std(axis=0).max()),
    }


def inference_order_control(model: ImageSequenceLearner, images: np.ndarray,
                            *, indices: np.ndarray, window: int,
                            batch_size: int, seed: int) -> dict:
    """Hold model weights fixed; shuffle only previously observed context order."""
    original = infer_logits(model, images, indices=indices, window=window,
                            batch_size=batch_size)
    shuffled = infer_logits(model, images, indices=indices, window=window,
                            batch_size=batch_size, history_order="permuted",
                            permutation_seed=seed + 9001)
    a = decode_logits(original)
    b = decode_logits(shuffled)
    return {
        "mean_absolute_logit_delta": float(np.abs(original - shuffled).mean()),
        "changed_top6_fraction": float(np.mean(np.any(a != b, axis=(1, 2)))),
        "mean_common_selected_pixels": float(np.mean((a * b).sum(axis=(1, 2)))),
        "control": "same_trained_weights_and_history_multisets_shuffled_order_only",
    }


def evaluate_predictions(images: np.ndarray, ids: np.ndarray,
                         logits_by_method: dict[str, np.ndarray], *,
                         baseline: int, seed: int, include_rows: bool,
                         draw_ids: np.ndarray | None = None,
                         dates: np.ndarray | None = None) -> tuple[dict, list, dict]:
    """Shared per-target IID controls make all three methods directly paired."""
    indices = np.asarray(ids, dtype=np.int64)
    if (baseline < 2 or len(indices) == 0
            or any(name not in logits_by_method for name in METHODS)
            or any(np.asarray(logits_by_method[name]).shape != (len(indices), 7, 7)
                   for name in METHODS)):
        raise ValueError("invalid diagnostic evaluation arguments")
    predicted = {name: decode_logits(logits_by_method[name]) for name in METHODS}
    adjusted = {name: [] for name in METHODS}
    raw = {name: [] for name in METHODS}
    rows = []
    for i, target in enumerate(indices):
        chance = iid_images(baseline, seed=seed + int(target)*1000003)
        actual = images[int(target)]
        row = None
        if include_rows:
            row = {"index": int(target), "draw_id": int(draw_ids[target]) if draw_ids is not None else None,
                   "date": int(dates[target]) if dates is not None and dates[target] else None,
                   "year": int(calendar_years(draw_ids[[target]])[0]) if draw_ids is not None else None,
                   "actual": decode(actual), "methods": {}}
        for name in METHODS:
            guess = predicted[name][i]
            score = similarity(guess, actual)
            conditional_iid = iid_expected(guess, chance)
            effect = float(score["composite"] - conditional_iid)
            adjusted[name].append(effect)
            raw[name].append(score["composite"])
            if row is not None:
                row["methods"][name] = {
                    "predicted": decode(guess),
                    "raw_similarity": score,
                    "iid_expected": conditional_iid,
                    "chance_adjusted": effect,
                }
        if row is not None:
            rows.append(row)
    metrics = {
        name: {
            "mean_raw_similarity": float(np.mean(raw[name])),
            "mean_iid_adjusted_similarity": float(np.mean(adjusted[name])),
            "unique_patterns": diversity(predicted[name]),
            "per_draw_adjusted": adjusted[name],
        } for name in METHODS
    }
    return metrics, rows, adjusted


def paired_block_intervals(values: dict[str, list], *,
                           iterations: int = 400, block: int = 8,
                           seed: int = 649) -> dict:
    """Exploratory circular moving-block intervals for paired score differences."""
    if (iterations < 2 or block < 1 or seed < 0 or not all(name in values for name in METHODS)):
        raise ValueError("invalid paired interval arguments")
    length = len(values[METHODS[0]])
    if not length or any(len(values[name]) != length for name in METHODS):
        raise ValueError("nonmatching per-draw score lengths")
    rng = np.random.default_rng(seed)
    out = {}
    for other in METHODS[1:]:
        delta = np.asarray(values[METHODS[0]]) - np.asarray(values[other])
        sample = []
        blocks = int(np.ceil(length/block))
        for _ in range(iterations):
            starts = rng.integers(0, length, size=blocks)
            positions = np.concatenate([(start + np.arange(block)) % length
                                        for start in starts])[:length]
            sample.append(float(delta[positions].mean()))
        out[f"{METHODS[0]}_minus_{other}"] = {
            "mean": float(delta.mean()),
            "interval_95": [float(x) for x in np.quantile(sample, [0.025, 0.975])],
            "iterations": iterations,
            "block_length": block,
            "note": "exploratory_paired_block_interval_not_blind_significance",
        }
    return out


def diagnose(path: Path = DATA, *, year: int | None = None,
             window: int = 5, epochs: int = 45, patience: int = 8,
             batch_size: int = 64, learning_rate: float = 0.003,
             embedding: int = 24, hidden: int = 32,
             baseline: int = 32, bootstrap: int = 400,
             seed: int = 649) -> dict:
    """Compare ordered, order-shuffled, and history-free image learning."""
    if baseline < 2 or bootstrap < 2 or batch_size < 1 or seed < 0:
        raise ValueError("invalid diagnostic experiment parameters")
    path = Path(path)
    images, dates, draw_ids = load_history(path)
    train, validation, test = split(len(images), window)
    targets = test if year is None else test[calendar_years(draw_ids[test]) == year]
    if not len(targets):
        raise ValueError("no test images in requested year")
    settings = dict(train_end=int(validation[0]), validation_end=int(test[0]),
                    window=window, epochs=epochs, patience=patience,
                    batch_size=batch_size, learning_rate=learning_rate,
                    seed=seed)
    ordered, fit_ordered = train_model(
        images, embedding=embedding, hidden=hidden, **settings)
    shuffled, fit_shuffled = train_model(
        images, embedding=embedding, hidden=hidden, history_order="permuted", **settings)
    constant, fit_constant = train_constant_template(images, **settings)
    arms = {"ordered_cnn_gru": ordered,
            "shuffled_cnn_gru": shuffled,
            "constant_template": constant}
    fitting = {"ordered_cnn_gru": fit_ordered,
               "shuffled_cnn_gru": fit_shuffled,
               "constant_template": fit_constant}

    def get_logits(indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: infer_logits(
                model, images, indices=indices, window=window,
                batch_size=batch_size,
                history_order="permuted" if name == "shuffled_cnn_gru" else "chronological",
                permutation_seed=seed + 9001)
            for name, model in arms.items()
        }

    validation_logits = get_logits(validation)
    val, _, _ = evaluate_predictions(
        images, validation, validation_logits, baseline=max(8, baseline),
        seed=seed + 17, include_rows=False)
    test_logits = get_logits(targets)
    summary, rows, scores = evaluate_predictions(
        images, targets, test_logits, baseline=baseline,
        seed=seed, include_rows=True, draw_ids=draw_ids, dates=dates)
    logits = {name: logit_diagnostics(values)
              for name, values in test_logits.items()}
    latents = {
        name: latent_diagnostics(
            model, images, indices=targets, window=window,
            batch_size=batch_size,
            history_order="permuted" if name == "shuffled_cnn_gru" else "chronological",
            permutation_seed=seed + 9001)
        for name, model in arms.items()
    }
    return {
        "schema": SCHEMA,
        "status": "EXPLORATORY_ALREADY_INSPECTED_HISTORICAL_DATA",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_draws": len(images),
        "evaluation_draws": len(targets),
        "year": year if year is not None else "all",
        "split": {"train_targets": [int(train[0]), int(train[-1])],
                  "validation_targets": [int(validation[0]), int(validation[-1])],
                  "test_targets": [int(targets[0]), int(targets[-1])]},
        "config": {"window": window, "epochs": epochs, "patience": patience,
                   "batch_size": batch_size, "learning_rate": learning_rate,
                   "embedding": embedding, "hidden": hidden,
                   "baseline_per_draw": baseline, "bootstrap_iterations": bootstrap,
                   "seed": seed},
        "training": fitting,
        "validation": val,
        "summary": summary,
        "logit_diagnostics": logits,
        "latent_diagnostics": latents,
        "fixed_weights_history_order_control": inference_order_control(
            ordered, images, indices=targets, window=window,
            batch_size=batch_size, seed=seed),
        "paired_intervals": paired_block_intervals(scores, iterations=bootstrap,
                                                    seed=seed+77),
        "rows": rows,
        "limitations": [
            "Previously inspected evaluation segment cannot support a blind predictive claim.",
            "The model-selection metric is validation shape loss, not test adjusted similarity.",
            "Template, shuffled sequence and ordered sequence have different capacities.",
            "Shuffled-context ablation preserves each target's historical images but destroys their order.",
            "Higher image diversity alone is not evidence of forecasting skill.",
            "Pixel position geometry is arbitrary; no physical lottery spatial process is assumed.",
        ],
    }
