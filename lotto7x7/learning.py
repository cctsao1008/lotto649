"""Learn 7x7 image-sequence dynamics with a CNN encoder and recurrent decoder.

Unlike the retired jackpot-logit models, this network is trained on binary
images and a differentiable image-shape objective. It outputs a six-pixel
bitmap. A differentiable image-shape term and supervised pixel reconstruction
are trained jointly; shape similarity relative to IID is the main evaluation. No six-subset likelihood is used.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import decode, iid_expected, iid_images, similarity, validate
from .models import diversity
from .research import DATA, calendar_years, load_history, split

VERSION = "0.1.0"


class ImageSequenceLearner(nn.Module):
    """Per-frame CNN followed by a temporal GRU and 7x7 image output head."""

    def __init__(self, *, embedding: int = 24, hidden: int = 32) -> None:
        super().__init__()
        if embedding < 2 or hidden < 2:
            raise ValueError("embedding and hidden dimensions must be >= 2")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 12, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(12 * 7 * 7, embedding), nn.Tanh(),
        )
        self.temporal = nn.GRU(embedding, hidden, batch_first=True)
        self.image_head = nn.Linear(hidden, 7 * 7)
        nn.init.xavier_uniform_(self.image_head.weight)
        nn.init.constant_(self.image_head.bias, -2.0)

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:
        """Accept [batch,time,7,7] floats and return [batch,7,7] image logits."""
        x = image_sequence
        if x.ndim != 4 or x.shape[1] < 2 or x.shape[2:] != (7, 7):
            raise ValueError("image input must be [batch,time>=2,7,7]")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("input image sequence must contain finite pixels")
        b, t, _, _ = x.shape
        embeddings = self.encoder(x.reshape(b * t, 1, 7, 7)).reshape(b, t, -1)
        response, _ = self.temporal(embeddings)
        return self.image_head(response[:, -1]).reshape(b, 7, 7)


def autocorrelation(image: torch.Tensor) -> torch.Tensor:
    """Normalized non-wrapping translation-invariant pair displacements.

    Zero-padding to 13x13 prevents original 7x7 boundaries from wrapping.
    The autocorrelation distribution is invariant to translation and
    differentiable through the supplied floating-point image.
    """
    if image.ndim != 3 or image.shape[1:] != (7, 7):
        raise ValueError("images must be [batch,7,7]")
    padded = F.pad(image, (0, 6, 0, 6))
    fft = torch.fft.fft2(padded)
    power = fft.real.square() + fft.imag.square()
    distribution = torch.fft.fftshift(torch.fft.ifft2(power).real, dim=(-2, -1))
    mass = image.sum(dim=(1, 2)).square().clamp(min=1e-6)
    return distribution / mass[:, None, None]


def shape_loss(logits: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Differentiable image objective: reconstruct structure, not jackpot prizes.

    Compare autocorrelation under the eight rigid D4 transformations; this
    component is invariant to image translation, quarter-turns and reflection.
    Include spatial reconstruction and six-pixel mass constraints to
    keep predictions identifiable and suitable for a 7x7 output bitmap.
    """
    if logits.ndim != 3 or logits.shape[1:] != (7, 7) or observed.shape != logits.shape:
        raise ValueError("expected matching [batch,7,7] image tensors")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(observed).all()):
        raise ValueError("loss inputs must be finite")
    probability = torch.sigmoid(logits)
    predicted_pairs = autocorrelation(probability)
    with torch.no_grad():
        actual_pairs = autocorrelation(observed)
        transforms = torch.stack([
            variant
            for rotations in range(4)
            for variant in (
                torch.rot90(actual_pairs, rotations, dims=(-2, -1)),
                torch.flip(torch.rot90(actual_pairs, rotations,
                                       dims=(-2, -1)), dims=(-1,)),
            )
        ], dim=1)
    squared = (predicted_pairs[:, None] - transforms).square().mean(dim=(-2, -1))
    # Comparing autocorrelations is a shape objective; the alignment is D4-invariant.
    structural = 169.0 * torch.min(squared, dim=1).values.mean()
    spatial = F.mse_loss(probability, observed)
    cardinality = ((probability.sum(dim=(1, 2)) - 6.0) / 7.0).square().mean()
    return 0.35 * structural + 0.60 * F.binary_cross_entropy_with_logits(logits, observed, pos_weight=logits.new_tensor(7.0)) + 0.05 * cardinality


def image_windows(images: np.ndarray, *, window: int = 5
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build strictly causal windows, with no target image among its inputs."""
    source = np.asarray(images)
    if source.ndim != 3 or source.shape[1:] != (7, 7) or not 2 <= window < len(source):
        raise ValueError("invalid window size or image history")
    for entry in source:
        validate(entry)
    x = np.stack([source[t-window:t] for t in range(window, len(source))])
    y = source[window:].copy()
    return (torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32))


def permute_history_windows(windows: torch.Tensor, indices: np.ndarray,
                            *, seed: int) -> torch.Tensor:
    """Fixed per-target history-order control without moving target images.

    A stable seed plus absolute target index gives identical permutations in
    train, validation and inference, independent of batch composition.
    Every target retains exactly the same previous images as the ordered input.
    """
    ids = np.asarray(indices, dtype=np.int64)
    if (windows.ndim != 4 or windows.shape[2:] != (7, 7)
            or windows.shape[1] < 2 or ids.shape != (len(windows),)
            or np.any(ids < windows.shape[1]) or seed < 0):
        raise ValueError("invalid context permutation configuration")
    rows = []
    for i, target in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(target)]))
        perm = rng.permutation(windows.shape[1])
        # Disallow a no-op shuffle so every input is a genuine ablation.
        if np.array_equal(perm, np.arange(windows.shape[1])):
            perm = np.roll(perm, 1)
        rows.append(windows[i, perm])
    return torch.stack(rows, dim=0)


def decode_image(logits: torch.Tensor) -> np.ndarray:
    """Convert one predicted image into six filled pixels without probability LUTs."""
    prediction = logits.detach().cpu().numpy()
    if prediction.shape != (7, 7) or not np.all(np.isfinite(prediction)):
        raise ValueError("expected a finite 7x7 predicted image")
    selected = np.argsort(-prediction.ravel(), kind="stable")[:6]
    image = np.zeros(49, dtype=np.uint8)
    image[selected] = 1
    return validate(image.reshape(7, 7))


def train_model(images: np.ndarray, *, train_end: int, validation_end: int,
                window: int = 5, embedding: int = 24, hidden: int = 32,
                epochs: int = 30, patience: int = 6, batch_size: int = 64,
                learning_rate: float = 0.003, seed: int = 649,
                history_order: str = "chronological"
                ) -> tuple[ImageSequenceLearner, dict]:
    """Fit only pre-validation images and early-stop only on validation images."""
    if (not 2 <= window < train_end - 10
            or not train_end + 5 <= validation_end < len(images)
            or epochs < 1 or patience < 1 or batch_size < 1 or seed < 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid chronological training hyperparameters")
    if history_order not in ("chronological", "permuted"):
        raise ValueError("history_order must be chronological or permuted")
    x, y = image_windows(images, window=window)
    if history_order == "permuted":
        x = permute_history_windows(
            x, np.arange(window, len(images)), seed=seed + 9001)
    # x[k] predicts draw t=k+window; t is excluded from x[k] by construction.
    tr_end = train_end - window
    va_end = validation_end - window
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ImageSequenceLearner(embedding=embedding, hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(tr_end, generator=gen)
        for ids in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = shape_loss(model(x[ids]), y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [shape_loss(model(a), b).item()
                      for a, b in zip(x[tr_end:va_end].split(batch_size),
                                      y[tr_end:va_end].split(batch_size))]
        val_loss = float(np.mean(losses))
        trace.append({"epoch": epoch, "validation_image_shape_loss": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().clone()
                          for name, tensor in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_image_shape_loss": best_loss,
        "epochs_run": len(trace),
        "history": trace,
        "training_targets": [window, train_end - 1],
        "validation_targets": [train_end, validation_end - 1],
        "window": window,
        "architecture": "cnn_frame_encoder_plus_gru_next_7x7_image",
        "history_order": history_order,
        "history_permutation_seed": seed + 9001 if history_order == "permuted" else None,
        "objective": "image_shape_autocorrelation_plus_pixel_reconstruction_and_six_pixel_mass",
        "loss_weights": {"shape_autocorrelation": 0.35,
                         "pixel_reconstruction": 0.60, "pixel_mass": 0.05},
    }


def predict_images(model: ImageSequenceLearner, images: np.ndarray,
                   *, indices: np.ndarray, window: int,
                   batch_size: int = 64) -> np.ndarray:
    """Batch inference without reading the target images into input windows."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1 or not len(idx) or idx.min() < window or idx.max() >= len(images):
        raise ValueError("invalid prediction target index")
    inputs = np.stack([images[t-window:t] for t in idx])
    outputs = []
    with torch.no_grad():
        model.eval()
        for group in torch.as_tensor(inputs, dtype=torch.float32).split(batch_size):
            logits = model(group)
            outputs.extend(decode_image(item) for item in logits)
    return np.stack(outputs)


def learned_replay(path: Path = DATA, *, year: int | None = None,
                   window: int = 5, epochs: int = 30, patience: int = 6,
                   batch_size: int = 64, learning_rate: float = 0.003,
                   embedding: int = 24, hidden: int = 32,
                   baseline: int = 32, seed: int = 649) -> tuple[dict, ImageSequenceLearner]:
    """Evaluate genuinely trained image models, not hand-built fixed prototypes."""
    if baseline < 2:
        raise ValueError("IID baseline sample count must be >= 2")
    path = Path(path)
    images, dates, ids = load_history(path)
    train, val, test = split(len(images), window)
    chosen = test if year is None else test[calendar_years(ids[test]) == year]
    if not len(chosen):
        raise ValueError("year has no observations in chronological test range")
    model, training = train_model(
        images, train_end=int(val[0]), validation_end=int(test[0]),
        window=window, embedding=embedding, hidden=hidden, epochs=epochs,
        patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed)
    predictions = predict_images(model, images, indices=chosen,
                                 window=window, batch_size=batch_size)
    records = []
    sample = {"learned_image": [], "previous_image": []}
    raws = {"learned_image": [], "previous_image": []}
    for i, t in enumerate(chosen):
        current = images[int(t)]
        chance = iid_images(baseline, seed=seed + int(t)*1000003)
        compare = {"learned_image": predictions[i],
                   "previous_image": images[int(t)-1]}
        item = {"date": int(dates[t]) if dates[t] else None,
                "year": int(calendar_years(ids[[t]])[0]),
                "draw_id": int(ids[t]),
                "actual": decode(current), "methods": {}}
        for name, guess in compare.items():
            observed = similarity(guess, current)
            iid = iid_expected(guess, chance)
            score = observed["composite"] - iid
            sample[name].append(float(score))
            raws[name].append(observed["composite"])
            item["methods"][name] = {
                "predicted": decode(guess),
                "raw_similarity": observed,
                "iid_expected": iid,
                "chance_adjusted": float(score),
            }
        records.append(item)
    summary = {
        name: {
            "mean_raw_similarity": float(np.mean(raws[name])),
            "mean_iid_adjusted_similarity": float(np.mean(sample[name])),
            "unique_patterns": diversity(
                predictions if name == "learned_image" else images[chosen-1]),
        } for name in sample
    }
    return ({
        "schema": "lotto7x7.learned_image.v1",
        "status": "EXPLORATORY_HISTORICAL_NOT_PROSPECTIVE_BLIND",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_draws": len(images),
        "evaluation_draws": len(chosen),
        "year": year if year is not None else "all",
        "trained_image_model": training,
        "config": {
            "window": window, "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "embedding": embedding, "hidden": hidden,
            "baseline_per_draw": baseline, "seed": seed,
        },
        "summary": summary,
        "rows": records,
        "limitations": [
            "Actual images were inspected in prior phases; retrospective only.",
            "CNN plus GRU learns image features, not six-subset probabilities.",
            "Image-shape loss includes an auxiliary spatial reconstruction term.",
            "A shape score above zero must be compared to chance-conditioned IID.",
            "Image diversity does not by itself prove predictive value.",
        ],
    }, model)
