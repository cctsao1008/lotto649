"""Image-only causal analog and motion forecasters; no legacy algorithms."""
from __future__ import annotations

from collections import Counter
import numpy as np

from .core import _key, signature, validate

METHODS = ("previous", "centroid_motion", "temporal_analog")


def translate(image: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate six pixels together without wrapping or dropping any point."""
    source = validate(image)
    change = np.asarray(offset, dtype=np.float64)
    if change.shape != (2,) or not np.all(np.isfinite(change)):
        raise ValueError("expected finite 2D displacement")
    points = np.argwhere(source == 1)
    shift = np.clip(np.rint(change).astype(int), -points.min(axis=0), 6-points.max(axis=0))
    dest = np.zeros((7, 7), dtype=np.uint8)
    moved = points + shift
    dest[moved[:, 0], moved[:, 1]] = 1
    return validate(dest)


def _center(image: np.ndarray) -> np.ndarray:
    return np.argwhere(image == 1).mean(axis=0)


class ImageForecaster:
    """Freeze all learned analog pairs at train_end; earlier test images are context only."""

    def __init__(self, images: np.ndarray, *, train_end: int, window: int = 5,
                 train_order: np.ndarray | None = None):
        all_images = np.asarray(images)
        if (all_images.ndim != 3 or all_images.shape[1:] != (7, 7)
                or not isinstance(window, int) or not isinstance(train_end, int)
                or window < 2 or train_end < window + 12 or len(all_images) <= train_end):
            raise ValueError("invalid image bank configuration")
        for image in all_images:
            validate(image)
        self.images = all_images.copy()
        self.train_end = train_end
        self.window = window
        bank = all_images[:train_end] if train_order is None else np.asarray(train_order)
        if bank.shape != (train_end, 7, 7):
            raise ValueError("training bank override shape mismatch")
        self.bank = bank.copy()
        bank_feat = np.stack([signature(im) for im in self.bank])
        self.following = np.arange(window, train_end)
        self.contexts = np.stack([bank_feat[i-window:i] for i in self.following])
        self.weights = np.linspace(0.5, 1.0, window)
        self.weights /= self.weights.sum()

    def predict(self, t: int, method: str) -> np.ndarray:
        if method not in METHODS or not self.train_end <= t < len(self.images):
            raise ValueError("invalid forecast target or method")
        previous = self.images[t-1]
        if method == "previous":
            return previous.copy()
        if method == "centroid_motion":
            return translate(previous, _center(previous)-_center(self.images[t-2]))
        latest = self.images[t-self.window:t]
        context = np.stack([signature(x) for x in latest])
        distance = ((self.contexts-context[None])**2).mean(axis=2) @ self.weights
        past_target = int(self.following[np.argmin(distance)])
        historical_next = self.bank[past_target]
        # Move the old next-frame shape near the present image's centroid.
        delta = _center(previous) - _center(self.bank[past_target-1])
        return translate(historical_next, delta)


def diversity(images: np.ndarray) -> dict:
    """Measure output collapse; diversity alone is not forecasting skill."""
    masks = np.asarray(images)
    if masks.ndim != 3 or masks.shape[1:] != (7, 7) or len(masks) == 0:
        raise ValueError("expected a nonempty image sequence")
    counts = Counter(_key(x) for x in masks)
    shares = np.array(list(counts.values()), dtype=float)/len(masks)
    return {"unique": len(counts), "top_fraction": float(shares.max()),
            "entropy_nats": float(-np.sum(shares*np.log(shares)))}
