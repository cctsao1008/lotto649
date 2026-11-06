"""Phase 7C: causal, dynamically generated 7x7 pattern forecasting methods.

No old logits, classifier, neural network, or six-of-six likelihood are used.
The analog bank is frozen at train_end. Each prediction t uses only observed
images with index < t. Dynamic results are legal binary six-point images.
"""
from __future__ import annotations

from collections import Counter
from math import exp

import numpy as np

from lotto_pattern_similarity import image_key, signature

VERSION = "0.1.0"
METHODS = ("last_pattern", "centroid_flow", "analog_fusion")


def _centroid(image: np.ndarray) -> np.ndarray:
    return np.argwhere(image == 1).mean(axis=0)


def translate_in_bounds(image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Translate all six pixels together and clamp the displacement to bounds.

    Unlike np.roll, this does not wrap points from one border to the other.
    It never clips or adds points, preserving the full six-point shape.
    """
    key = image_key(image)
    points = np.asarray(np.unravel_index(np.asarray(key), (7, 7))).T
    shift = np.rint(np.asarray(displacement, dtype=np.float64)).astype(np.int64)
    if shift.shape != (2,) or not np.all(np.isfinite(displacement)):
        raise ValueError("expected a finite two-component shift")
    lower = -np.min(points, axis=0)
    upper = 6 - np.max(points, axis=0)
    shift = np.clip(shift, lower, upper)
    moved = points + shift
    result = np.zeros((7, 7), dtype=np.uint8)
    result[moved[:, 0], moved[:, 1]] = 1
    if int(result.sum()) != 6:
        raise AssertionError("translation changed active-pixel count")
    return result


def shape_class(image: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Canonical six-point shape modulo D4 transforms and translation."""
    image_key(image)
    variants = []
    for rotations in range(4):
        base = np.rot90(image, rotations)
        for candidate in (base, np.fliplr(base)):
            points = np.argwhere(candidate == 1)
            points = points - points.min(axis=0)
            variants.append(tuple(sorted(tuple(map(int, p)) for p in points)))
    return min(variants)


def diversity(images: np.ndarray) -> dict:
    """Summarize distinct image outputs, independent of predictive accuracy."""
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (7, 7) or not len(arr):
        raise ValueError("expected nonempty [draws,7,7] array")
    keys = [image_key(item) for item in arr]
    counts = Counter(keys)
    classes = Counter(shape_class(item) for item in arr)
    shares = np.asarray(list(counts.values()), dtype=np.float64) / len(arr)
    entropy = float(-np.sum(shares * np.log(shares)))
    return {
        "draws": len(arr),
        "unique_pixel_patterns": len(counts),
        "unique_shape_classes_d4_translate": len(classes),
        "unique_fraction": len(counts) / len(arr),
        "top_pattern_fraction": max(counts.values()) / len(arr),
        "entropy_nats": entropy,
        "effective_pattern_count": float(np.exp(entropy)),
        "top_pattern_raster_indices": list(counts.most_common(1)[0][0]),
        "top_pattern_occurrences": counts.most_common(1)[0][1],
    }


class DynamicPatternLibrary:
    """One-step visual forecasts with train-only analog contexts and targets."""

    def __init__(self, images: np.ndarray, *, train_end: int,
                 window: int = 5, neighbors: int = 8,
                 last_weight: float = 0.20,
                 bank_override: np.ndarray | None = None):
        raw = np.asarray(images)
        if (raw.ndim != 3 or raw.shape[1:] != (7, 7)
                or not isinstance(train_end, int)
                or not isinstance(window, int)
                or not isinstance(neighbors, int)
                or not (2 <= window < train_end - 10)
                or not (train_end < len(raw) - 10)
                or not 1 <= neighbors <= train_end - window
                or not 0.0 <= last_weight < 1.0):
            raise ValueError("invalid dynamic pattern configuration")
        for img in raw:
            image_key(img)
        self.images = raw.astype(np.uint8, copy=True)
        self.train_end = train_end
        self.window = window
        self.neighbors = neighbors
        self.last_weight = float(last_weight)
        self.signatures = np.stack([signature(x) for x in self.images])
        if bank_override is None:
            self.bank_images = self.images[:train_end].copy()
            bank_signatures = self.signatures[:train_end]
        else:
            override = np.asarray(bank_override)
            if override.shape != (train_end, 7, 7):
                raise ValueError("shuffled train bank must have shape [train_end,7,7]")
            for img in override:
                image_key(img)
            self.bank_images = override.astype(np.uint8, copy=True)
            bank_signatures = np.stack([signature(x) for x in self.bank_images])
        self.bank_targets = np.arange(window, train_end, dtype=np.int64)
        self.bank_contexts = np.stack([
            bank_signatures[t-window:t] for t in self.bank_targets
        ])
        self.frame_weights = np.linspace(0.5, 1.0, window)
        self.frame_weights /= self.frame_weights.sum()

    def _previous_images(self, t: int, *, scrambled: bool,
                         scramble_seed: int) -> np.ndarray:
        history = self.images[t-self.window:t].copy()
        if scrambled:
            # Only rearrange images already observed prior to t.
            rng = np.random.default_rng(scramble_seed + t * 1000003)
            order = rng.permutation(self.window)
            if np.array_equal(order, np.arange(self.window)):
                order = np.roll(order, 1)
            history = history[order]
        return history

    def predict(self, t: int, *, method: str,
                scramble_context: bool = False,
                scramble_seed: int = 649) -> np.ndarray:
        """Forecast t without seeing its actual image; return six active pixels."""
        if (method not in METHODS or not isinstance(t, int)
                or not self.train_end <= t < len(self.images)
                or scramble_seed < 0):
            raise ValueError("invalid forecast index, method or seed")
        if method == "last_pattern":
            return self.images[t-1].copy()
        history = self._previous_images(
            t, scrambled=scramble_context, scramble_seed=scramble_seed)
        if method == "centroid_flow":
            change = _centroid(history[-1]) - _centroid(history[-2])
            return translate_in_bounds(history[-1], change)
        query = np.stack([signature(x) for x in history])
        levels = np.mean((self.bank_contexts - query[None, :, :]) ** 2, axis=2)
        distance = np.sum(levels * self.frame_weights[None, :], axis=1)
        deltas = np.diff(self.bank_contexts, axis=1) - np.diff(query, axis=0)[None, :, :]
        dynamics = np.mean(deltas * deltas, axis=(1, 2))
        distance = 0.70 * distance + 0.30 * dynamics
        ranking = np.argsort(distance, kind="stable")[:self.neighbors]
        selected = self.bank_targets[ranking]
        near_distance = distance[ranking]
        scale = max(float(np.median(near_distance)), 1e-8)
        weights = np.exp(-(near_distance - float(np.min(near_distance))) / scale)
        weights /= weights.sum()
        heat = self.last_weight * history[-1].astype(np.float64)
        for ref_target, importance in zip(selected, weights):
            train_prev = self.bank_images[ref_target - 1]
            reference_target = self.bank_images[ref_target]
            movement = _centroid(history[-1]) - _centroid(train_prev)
            transported = translate_in_bounds(reference_target, movement)
            heat += (1.0 - self.last_weight) * importance * transported
        # Stable top six pixels, without training outcome leakage or sampling.
        selected_cells = np.argsort(-heat.ravel(), kind="stable")[:6]
        forecast = np.zeros(49, dtype=np.uint8)
        forecast[selected_cells] = 1
        result = forecast.reshape(7, 7)
        image_key(result)
        return result
