"""Phase 7B image-pattern forecasters, without old logit or MLP models.

These are predeclared deterministic pattern estimators, not claims of lottery
predictability. They output legal six-pixel 7x7 masks; the historical bank,
clusters, and transitions are constructed using train-only observations.
"""
from __future__ import annotations

import numpy as np

from lotto_pattern_similarity import image_key, signature

METHODS = ("last_pattern", "temporal_analog", "cluster_transition")
VERSION = "0.1.0"


def _centers_and_representatives(features: np.ndarray, clusters: int,
                                 iterations: int = 16
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic farthest-seeded centroid clustering and member prototypes."""
    n, dim = features.shape
    if not 2 <= clusters <= n:
        raise ValueError("clusters must be between two and training sample count")
    first = int(np.argmin(np.sum((features - features.mean(axis=0)) ** 2, axis=1)))
    selected = [first]
    minimum = np.sum((features - features[first]) ** 2, axis=1)
    for _ in range(1, clusters):
        next_idx = int(np.argmax(minimum))
        selected.append(next_idx)
        minimum = np.minimum(minimum, np.sum(
            (features - features[next_idx]) ** 2, axis=1))
    centers = features[selected].copy()
    for _ in range(iterations):
        d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        updated = centers.copy()
        for k in range(clusters):
            members = features[labels == k]
            if len(members):
                updated[k] = members.mean(axis=0)
        if np.allclose(updated, centers, rtol=0, atol=1e-11):
            centers = updated
            break
        centers = updated
    d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    representatives = np.empty(clusters, dtype=np.int64)
    for k in range(clusters):
        members = np.flatnonzero(labels == k)
        if len(members):
            representatives[k] = int(members[np.argmin(d2[members, k])])
        else:
            representatives[k] = int(np.argmin(d2[:, k]))
    return centers, labels, representatives


class PatternLibrary:
    """Frozen reference patterns from indices [0,train_end); never train on eval."""

    def __init__(self, images: np.ndarray, *, train_end: int,
                 window: int = 5, clusters: int = 8):
        arr = np.asarray(images)
        if (arr.ndim != 3 or arr.shape[1:] != (7, 7)
                or not isinstance(window, int) or not isinstance(train_end, int)
                or not (1 <= window < train_end - 10 <= len(arr) - 10)
                or not 2 <= clusters <= train_end):
            raise ValueError("invalid images or fixed train boundary")
        for im in arr:
            image_key(im)
        self.images = arr.astype(np.uint8, copy=True)
        self.train_end = train_end
        self.window = window
        self.signatures = np.stack([signature(im) for im in self.images])
        train_sig = self.signatures[:train_end]
        self.cluster_centers, train_labels, self.representatives = (
            _centers_and_representatives(train_sig, clusters))
        self.cluster_sizes = np.bincount(train_labels, minlength=clusters)
        # Only one-step transitions with both previous and next in training.
        self.transitions = np.zeros((clusters, clusters), dtype=np.int64)
        for a, b in zip(train_labels[:-1], train_labels[1:]):
            self.transitions[int(a), int(b)] += 1
        # Each bank item is a known training target and has a fully known
        # window of observations directly preceding it.
        self.bank_targets = np.arange(window, train_end)
        self.context_bank = np.stack([
            self.signatures[t-window:t] for t in self.bank_targets
        ])
        # Recent image receives the largest similarity weight.
        self.frame_weights = np.linspace(0.5, 1.0, window, dtype=np.float64)

    def cluster_label(self, image: np.ndarray) -> int:
        z = signature(image)
        return int(np.argmin(np.sum((self.cluster_centers - z) ** 2, axis=1)))

    def predict(self, t: int, *, method: str) -> np.ndarray:
        """Predict t using current earlier images and a frozen train-only bank."""
        if method not in METHODS:
            raise ValueError("unknown pattern method")
        if not isinstance(t, int) or not self.train_end <= t < len(self.images):
            raise ValueError("target must belong to validation or test range")
        if method == "last_pattern":
            return self.images[t-1].copy()
        if method == "temporal_analog":
            query = self.signatures[t-self.window:t]
            errors = np.mean((self.context_bank - query) ** 2, axis=2)
            distances = np.sum(errors * self.frame_weights[None, :], axis=1)
            best = int(self.bank_targets[int(np.argmin(distances))])
            return self.images[best].copy()
        last_label = self.cluster_label(self.images[t - 1])
        # Laplace smoothing and most common next cluster for this state;
        # equivalent to argmax of transition counts with equal prior.
        next_label = int(np.argmax(self.transitions[last_label] + 1))
        pattern_index = int(self.representatives[next_label])
        return self.images[pattern_index].copy()

    def cluster_report(self) -> list[dict]:
        """Human-readable fixed image families based on train-only statistics."""
        answer = []
        for k, count in enumerate(self.cluster_sizes):
            representative = self.images[int(self.representatives[k])]
            answer.append({
                "cluster_id": k,
                "train_members": int(count),
                "train_representative_raster_indices": list(
                    image_key(representative)),
                "representative_image": representative.astype(int).tolist(),
                "transition_counts": self.transitions[k].astype(int).tolist(),
            })
        return answer
