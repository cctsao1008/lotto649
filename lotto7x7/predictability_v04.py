"""Version 0.4 predictability audit: convergence and temporal information.

This is an exploratory historical diagnostic, NOT a prospective lottery
forecast. The archived v0.2/v0.3 implementations are imported unchanged.
The primary loss is v0.3 balanced BCE, with one shared train/validation/test
chronology and paired IID controls. There is no test-set model selection.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .core import decode, encode, iid_expected, iid_images, similarity
from .diagnostics import ConstantTemplate, infer_logits
from .learning import (ImageSequenceLearner, decode_image, image_windows,
                       permute_history_windows)
from .loss_study_v03 import loss_v03
from .research import DATA, calendar_years, load_history, split

SCHEMA = "lotto7x7.predictability_audit.v0.4"
METHODS = ("ordered_cnn_gru", "shuffled_cnn_gru", "constant_template")
LOSS = "balanced_bce"


def uniform_brier() -> float:
    """Exact per-pixel Brier score of a fair uniform six-of-49 draw."""
    p = 6.0 / 49.0
    return p * (1.0 - p)


def _initialize(method: str, *, seed: int, embedding: int, hidden: int) -> nn.Module:
    if method not in METHODS or seed < 0:
        raise ValueError("invalid audit method or seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return (ConstantTemplate() if method == "constant_template"
            else ImageSequenceLearner(embedding=embedding, hidden=hidden))


def _snapshot(model: nn.Module) -> dict:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _mean_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               *, batch_size: int) -> float:
    model.eval()
    with torch.no_grad():
        losses = [float(loss_v03(model(a), b, method=LOSS).item())
                  for a, b in zip(x.split(batch_size), y.split(batch_size))]
    if not losses:
        raise ValueError("audit loss requires nonempty windows")
    return float(np.mean(losses))


def train_audit_model(images: np.ndarray, *, method: str,
                      seed: int, train_end: int, validation_end: int,
                      window: int = 5, embedding: int = 24, hidden: int = 32,
                      epochs: int = 45, patience: int = 8,
                      batch_size: int = 64, learning_rate: float = 0.003
                      ) -> tuple[nn.Module, nn.Module, dict]:
    """Record epoch zero and per-epoch train/validation loss, then early-stop.

    Shuffling changes only the order of strictly preceding images in each
    context window; it does not move any training targets or split boundaries.
    All methods share the identical balanced-BCE objective and optimizer.
    """
    if (method not in METHODS or seed < 0 or not 2 <= window < train_end - 10
            or not train_end + 5 <= validation_end < len(images)
            or epochs < 1 or patience < 1 or batch_size < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid predictability training configuration")
    x, y = image_windows(images, window=window)
    if method == "shuffled_cnn_gru":
        x = permute_history_windows(
            x, np.arange(window, len(images)), seed=seed + 9001)
    tr_end, va_end = train_end - window, validation_end - window
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = _initialize(method, seed=seed, embedding=embedding, hidden=hidden)
    initial = _initialize(method, seed=seed, embedding=embedding, hidden=hidden)
    training_x, training_y = x[:tr_end], y[:tr_end]
    validation_x, validation_y = x[tr_end:va_end], y[tr_end:va_end]

    def observe(epoch: int) -> dict:
        return {
            "epoch": epoch,
            "training_loss": _mean_loss(model, training_x, training_y,
                                        batch_size=batch_size),
            "validation_loss": _mean_loss(model, validation_x, validation_y,
                                          batch_size=batch_size),
        }

    trace = [observe(0)]
    best_loss = trace[0]["validation_loss"]
    best_epoch = 0
    best_state = _snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(
            tr_end, generator=torch.Generator().manual_seed(seed + epoch))
        for positions in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_v03(model(training_x[positions]), training_y[positions],
                            method=LOSS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        point = observe(epoch)
        if not np.isfinite(point["training_loss"]) or not np.isfinite(point["validation_loss"]):
            raise RuntimeError("non-finite convergence audit loss")
        trace.append(point)
        if point["validation_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = point["validation_loss"], epoch
            best_state = _snapshot(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    initial.eval()
    return model, initial, {
        "objective": LOSS,
        "method": method,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": len(trace) - 1,
        "history": trace,
        "training_targets": [window, train_end - 1],
        "validation_targets": [train_end, validation_end - 1],
        "epoch_zero_included": True,
        "history_order": ("permuted" if method == "shuffled_cnn_gru"
                          else "no_history" if method == "constant_template"
                          else "chronological"),
    }


def _predict(model: nn.Module, images: np.ndarray, *, targets: np.ndarray,
             method: str, seed: int, window: int, batch_size: int) -> np.ndarray:
    return infer_logits(
        model, images, indices=targets, window=window,
        batch_size=batch_size,
        history_order="permuted" if method == "shuffled_cnn_gru" else "chronological",
        permutation_seed=seed + 9001,
    )


def _score(logits: np.ndarray, actual: np.ndarray, *,
           chance: list[np.ndarray] | None = None,
           score_cache: dict | None = None
           ) -> tuple[dict, np.ndarray | None, np.ndarray]:
    """Pixel Brier and optional paired conditional-IID-adjusted image score."""
    if (logits.ndim != 3 or logits.shape != actual.shape
            or logits.shape[1:] != (7, 7) or not len(logits)
            or not np.isfinite(logits).all()):
        raise ValueError("invalid audit prediction or target shape")
    probabilities = torch.sigmoid(torch.as_tensor(logits)).numpy()
    brier = ((probabilities - actual) ** 2).mean(axis=(1, 2))
    masks = np.stack([decode_image(torch.as_tensor(z)) for z in logits])
    patterns = Counter(mask.tobytes() for mask in masks)
    metric = {
        "brier_mean": float(np.mean(brier)),
        "brier_skill_vs_uniform": float(1.0 - np.mean(brier) / uniform_brier()),
        "unique_patterns": len(patterns),
        "top_pattern_share": float(max(patterns.values()) / len(masks)),
        "effective_soft_mass": float(probabilities.sum(axis=(1, 2)).mean()),
    }
    adjusted = None
    if chance is not None:
        if len(chance) != len(masks):
            raise ValueError("IID target count mismatch")
        values = []
        for t, (guess, observation, controls) in enumerate(zip(masks, actual, chance)):
            cache_key = (t, guess.tobytes())
            if score_cache is not None and cache_key in score_cache:
                value = score_cache[cache_key]
            else:
                value = (similarity(guess, observation)["composite"] -
                         iid_expected(guess, controls))
                if score_cache is not None:
                    score_cache[cache_key] = value
            values.append(value)
        adjusted = np.asarray(values, dtype=np.float64)
        metric["mean_iid_adjusted_similarity"] = float(adjusted.mean())
    return metric, adjusted, brier.astype(np.float64)


def paired_block_interval(delta: np.ndarray, *, iterations: int = 400,
                          block_size: int = 8, seed: int = 649) -> dict:
    """Moving-block interval over target dates, after averaging model seeds."""
    x = np.asarray(delta, dtype=np.float64)
    if (x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all()
            or iterations < 2 or block_size < 1 or seed < 0):
        raise ValueError("invalid paired bootstrap data")
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(x) / block_size))
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        starts = rng.integers(0, len(x), size=k)
        positions = (starts[:, None] + np.arange(block_size)[None, :]) % len(x)
        means[i] = x[positions.ravel()[:len(x)]].mean()
    return {
        "mean": float(x.mean()),
        "interval_95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "iterations": iterations,
        "block_length": block_size,
        "unit": "target_date_after_averaging_seeds",
        "interpretation": "exploratory_retrospective_not_blind_significance",
    }


def _run_experiment(images: np.ndarray, targets: np.ndarray, *,
                    seed_list: list[int], window: int, epochs: int,
                    patience: int, batch_size: int, learning_rate: float,
                    embedding: int, hidden: int, baseline: int,
                    null_seed: int, on_progress: Callable[[str], None] | None = None
                    ) -> tuple[list[dict], dict]:
    n = len(images)
    train, validation, test = split(n, window)
    if not len(targets) or targets.min() < test[0] or targets.max() > test[-1]:
        raise ValueError("evaluation targets must lie in the last chronological split")
    # Paired IID controls are fixed by absolute target index, not model/seed.
    chance = [iid_images(baseline, seed=null_seed + int(t)*1000003)
              for t in targets]
    actual = images[targets]
    run_results = []
    adjusted = {method: [] for method in METHODS}
    initial_adjusted = {method: [] for method in METHODS}
    briers = {method: [] for method in METHODS}
    score_cache = {}
    for seed in seed_list:
        methods = {}
        for method in METHODS:
            final, epoch0, fit = train_audit_model(
                images, method=method, seed=seed,
                train_end=int(validation[0]), validation_end=int(test[0]),
                window=window, embedding=embedding, hidden=hidden,
                epochs=epochs, patience=patience, batch_size=batch_size,
                learning_rate=learning_rate)
            initial_logits = _predict(epoch0, images, targets=targets,
                                      method=method, seed=seed, window=window,
                                      batch_size=batch_size)
            initial_metric, initial_residual, _ = _score(
                initial_logits, actual, chance=chance, score_cache=score_cache)
            logits = _predict(final, images, targets=targets,
                              method=method, seed=seed, window=window,
                              batch_size=batch_size)
            metric, residual, per_target_brier = _score(
                logits, actual, chance=chance, score_cache=score_cache)
            assert residual is not None and initial_residual is not None
            adjusted[method].append(residual)
            initial_adjusted[method].append(initial_residual)
            briers[method].append(per_target_brier)
            methods[method] = {
                "convergence": fit,
                "epoch0": initial_metric,
                "selected_checkpoint": metric,
            }
        run_results.append({"seed": seed, "methods": methods})
        if on_progress is not None:
            on_progress(f"[SEED] {seed}: ordered best epoch "
                        f"{methods['ordered_cnn_gru']['convergence']['best_epoch']}; "
                        f"shuffled {methods['shuffled_cnn_gru']['convergence']['best_epoch']}; "
                        f"constant {methods['constant_template']['convergence']['best_epoch']}")
    scores = {method: np.stack(adjusted[method]) for method in METHODS}
    initial_scores = {method: np.stack(initial_adjusted[method]) for method in METHODS}
    pixel = {method: np.stack(briers[method]) for method in METHODS}
    summary = {}
    for method in METHODS:
        metrics = [entry["methods"][method]["selected_checkpoint"]
                   for entry in run_results]
        summary[method] = {
            "mean_iid_adjusted_similarity": float(scores[method].mean()),
            "mean_pixel_brier": float(pixel[method].mean()),
            "mean_brier_skill_vs_uniform": float(1.0 - pixel[method].mean() / uniform_brier()),
            "mean_unique_patterns": float(np.mean([m["unique_patterns"] for m in metrics])),
            "mean_top_pattern_share": float(np.mean([m["top_pattern_share"] for m in metrics])),
            "best_epoch_by_seed": [entry["methods"][method]["convergence"]["best_epoch"]
                                   for entry in run_results],
            "epoch0_mean_brier": float(np.mean([
                entry["methods"][method]["epoch0"]["brier_mean"]
                for entry in run_results])),
            "epoch0_mean_iid_adjusted_similarity": float(initial_scores[method].mean()),
            "training_gain_over_epoch0_adjusted": float(
                (scores[method] - initial_scores[method]).mean()),
        }
    series = {
        "adjusted": {method: scores[method].mean(axis=0) for method in METHODS},
        "epoch0_adjusted": {method: initial_scores[method].mean(axis=0)
                            for method in METHODS},
        "brier": {method: pixel[method].mean(axis=0) for method in METHODS},
    }
    return run_results, {"summary": summary, "series": series,
                         "evaluation_targets": int(len(targets))}


def _synthetic_images(kind: str, n: int, seed: int,
                      *, window: int = 5) -> np.ndarray:
    if kind == "iid":
        return iid_images(n, seed=seed + 101)
    if kind == "periodic":
        if not 2 <= window <= 20:
            raise ValueError("periodic control needs a window between 2 and 20")
        # Match the cycle length to the causal window. Each target then has
        # exactly the SAME unordered multiset of prior images, regardless of
        # phase. Only frame order can identify the next periodic image.
        examples = ([1, 2, 3, 4, 5, 6], [1, 8, 15, 22, 29, 36],
                    [1, 9, 17, 25, 33, 41], [2, 11, 14, 23, 37, 49],
                    [6, 12, 18, 24, 30, 36], [5, 10, 15, 20, 25, 30],
                    [7, 14, 21, 28, 35, 42], [3, 13, 23, 33, 43, 48])
        distinct = [encode(np.asarray(row)) for row in examples]
        if window > len(distinct):
            for image in iid_images(50, seed=seed + 202):
                if image.tobytes() not in {m.tobytes() for m in distinct}:
                    distinct.append(image)
                if len(distinct) >= window:
                    break
        patterns = np.stack(distinct[:window])
        return patterns[np.arange(n) % window]
    raise ValueError("synthetic type must be iid or periodic")


def audit(path: Path = DATA, *, year: int | None = None,
          seeds: int = 5, seed: int = 649, window: int = 5,
          epochs: int = 45, patience: int = 8,
          batch_size: int = 64, learning_rate: float = 0.003,
          embedding: int = 24, hidden: int = 32,
          baseline: int = 32, bootstrap: int = 400,
          control_seeds: int = 2, control_samples: int = 120,
          control_epochs: int = 45, include_controls: bool = True,
          on_progress: Callable[[str], None] | None = None) -> dict:
    """Run predeclared CPU-only multi-seed chronological predictability audit."""
    if (seeds < 1 or control_seeds < 1 or control_samples < 100
            or control_epochs < 1 or baseline < 2 or bootstrap < 2
            or seed < 0 or batch_size < 1):
        raise ValueError("invalid predictability audit settings")
    path = Path(path)
    images, dates, draw_ids = load_history(path)
    train, val, test = split(len(images), window)
    targets = (test if year is None
               else test[calendar_years(draw_ids[test]) == year])
    if not len(targets):
        raise ValueError("requested year has no test data")
    seed_list = list(range(seed, seed + seeds))
    results, outcome = _run_experiment(
        images, targets, seed_list=seed_list, window=window,
        epochs=epochs, patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, embedding=embedding, hidden=hidden,
        baseline=baseline, null_seed=seed + 100003,
        on_progress=on_progress)
    main_series = outcome.pop("series")
    intervals = {
        "ordered_minus_shuffled_adjusted": paired_block_interval(
            main_series["adjusted"]["ordered_cnn_gru"] -
            main_series["adjusted"]["shuffled_cnn_gru"],
            iterations=bootstrap, seed=seed + 77),
        "ordered_minus_constant_adjusted": paired_block_interval(
            main_series["adjusted"]["ordered_cnn_gru"] -
            main_series["adjusted"]["constant_template"],
            iterations=bootstrap, seed=seed + 78),
        "ordered_minus_uniform_brier": paired_block_interval(
            main_series["brier"]["ordered_cnn_gru"] - uniform_brier(),
            iterations=bootstrap, seed=seed + 79),
        "ordered_training_gain_over_epoch0_adjusted": paired_block_interval(
            main_series["adjusted"]["ordered_cnn_gru"] -
            main_series["epoch0_adjusted"]["ordered_cnn_gru"],
            iterations=bootstrap, seed=seed + 80),
    }
    controls = {}
    if include_controls:
        for kind in ("periodic", "iid"):
            synthetic = _synthetic_images(kind, control_samples, seed, window=window)
            _, _, synthetic_test = split(len(synthetic), window)
            reps, result = _run_experiment(
                synthetic, synthetic_test,
                seed_list=seed_list[:control_seeds] if control_seeds <= seeds
                else list(range(seed, seed + control_seeds)),
                window=window, epochs=control_epochs, patience=patience,
                batch_size=batch_size, learning_rate=learning_rate,
                embedding=embedding, hidden=hidden,
                baseline=min(baseline, 16), null_seed=seed + 200003,
                on_progress=None)
            result.pop("series")
            controls[kind] = {
                "n_images": int(len(synthetic)),
                "n_evaluation": int(len(synthetic_test)),
                "seed_count": len(reps),
                "summary": result["summary"],
                "per_seed": reps,
                "description": ("phase_dependent_cycle_equal_to_history_window"
                                if kind == "periodic" else "independent_uniform_six_of_49"),
            }
            if on_progress is not None:
                on_progress(f"[CONTROL] {kind}: {len(reps)} seeds complete")
    return {
        "schema": SCHEMA,
        "status": "EXPLORATORY_HISTORICAL_ALREADY_INSPECTED_NOT_BLIND",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_draws": int(len(images)),
        "evaluation_draws": int(len(targets)),
        "year": year if year is not None else "all",
        "split": {
            "train_targets": [int(train[0]), int(train[-1])],
            "validation_targets": [int(val[0]), int(val[-1])],
            "evaluation_targets": [int(targets[0]), int(targets[-1])],
        },
        "config": {
            "seeds": seed_list, "window": window, "epochs": epochs,
            "patience": patience, "batch_size": batch_size,
            "learning_rate": learning_rate, "embedding": embedding,
            "hidden": hidden, "baseline_per_target": baseline,
            "bootstrap_iterations": bootstrap, "control_seeds": control_seeds,
            "control_samples": control_samples, "control_epochs": control_epochs,
            "include_controls": include_controls,
            "device": "cpu", "loss": LOSS,
        },
        "uniform_no_history_baseline": {
            "probability_per_pixel": 6.0 / 49.0,
            "pixel_brier": uniform_brier(),
            "time_information": False,
        },
        "summary": outcome["summary"],
        "per_seed": results,
        "paired_intervals": intervals,
        "synthetic_controls": controls,
        "limitations": [
            "Historical evaluation draws have been inspected in earlier phases.",
            "Confidence intervals are descriptive and do not establish blind significance.",
            "Different model capacities: constant logits have fewer parameters.",
            "Temporal shuffle preserves the multiset of causal past frames per target.",
            "Seed averages share the same evaluation dates and are not independent experiments.",
            "A random mapping and rolling-origin study are not included in this version.",
            "Image geometry is arbitrary; positive shape similarity is not lottery evidence.",
        ],
    }
