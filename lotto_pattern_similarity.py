"""Phase 7B: fixed, auditable 7x7 binary-image similarity metrics.

All six-pixel images are reversible masks. Similar-looking patterns do not
necessarily represent shared lottery numbers or a predictable drawing process.
Spatial and shape similarities must be calibrated against IID six-pixel targets
conditioned on the specific forecast pattern.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import comb, sqrt

import numpy as np
from scipy.fft import dctn

VERSION = "0.1.0"
N, K, SIDE = 49, 6, 7

# Fixed before reviewing Phase 7B scores; never tune on test outcomes.
WEIGHTS = {
    "pixel_jaccard": 0.15,
    "shape_d4": 0.35,
    "row_column": 0.15,
    "centroid": 0.10,
    "adjacency": 0.10,
    "low_dct": 0.15,
}


def image_key(image: np.ndarray) -> tuple[int, ...]:
    """Validate one legal six-point binary image; return its raster indices."""
    x = np.asarray(image)
    if x.shape != (SIDE, SIDE) or not np.issubdtype(x.dtype, np.number):
        raise ValueError("expected one numeric 7x7 binary image")
    if (not np.isrealobj(x) or not np.all(np.isfinite(x))
            or not np.all((x == 0) | (x == 1)) or int(x.sum()) != K):
        raise ValueError("image must contain exactly six active binary pixels")
    return tuple(np.flatnonzero(x).astype(int).tolist())


@lru_cache(maxsize=65536)
def _describe(key: tuple[int, ...]) -> dict:
    x = np.zeros((SIDE, SIDE), dtype=np.float64)
    x.ravel()[list(key)] = 1
    coords = np.argwhere(x == 1).astype(np.int16)
    low = dctn(x, type=2, norm="ortho")[:3, :3].copy()
    low[0, 0] = 0  # Constant DC coefficient contains no information for fixed-six masks.
    adj = np.asarray([
        (x[:, :-1] * x[:, 1:]).sum(),
        (x[:-1, :] * x[1:, :]).sum(),
        (x[:-1, :-1] * x[1:, 1:]).sum()
        + (x[:-1, 1:] * x[1:, :-1]).sum(),
    ], dtype=np.float64)
    rotated = tuple(
        np.argwhere(y == 1).astype(np.int16)
        for k in range(4)
        for y in (np.rot90(x, k), np.fliplr(np.rot90(x, k)))
    )
    return {
        "image": x, "points": coords, "rotated": rotated,
        "rows": x.sum(axis=1), "cols": x.sum(axis=0),
        "centroid": coords.mean(axis=0), "adj": adj,
        "low_dct": low,
    }


def _max_alignment_overlap(points_a: np.ndarray, variants_b: tuple) -> int:
    """Maximum overlap after any D4 reflection/rotation and integer translation.

    The entire six-point set is translated on an unbounded integer plane;
    points are never cropped, wrapped, or duplicated.
    """
    best = 0
    for points_b in variants_b:
        offsets = Counter(
            (int(a[0] - b[0]), int(a[1] - b[1]))
            for a in points_a for b in points_b
        )
        best = max(best, max(offsets.values()))
        if best == K:
            break
    return best


def _pair_descriptors(a: dict, b: dict) -> dict[str, float]:
    shared = int(np.sum(a["image"] * b["image"]))
    exact_jaccard = shared / (2 * K - shared)
    aligned = _max_alignment_overlap(a["points"], b["rotated"])
    shape_jaccard = aligned / (2 * K - aligned)
    rc_error = float(np.abs(a["rows"] - b["rows"]).sum()
                     + np.abs(a["cols"] - b["cols"]).sum())
    # Two 6-count 7-bin histograms differ by at most 12 L1 each.
    rc = 1.0 - rc_error / (4 * K)
    centroid = 1.0 - float(np.linalg.norm(
        a["centroid"] - b["centroid"])) / (6 * sqrt(2))
    # Any two active points contribute to at most one adjacency direction.
    adj = 1.0 - float(np.abs(a["adj"] - b["adj"]).sum()) / (2 * comb(K, 2))
    # Orthonormal DCT and projection imply distance <= sqrt(2 * K).
    dct = 1.0 - float(np.linalg.norm(a["low_dct"] - b["low_dct"])) / sqrt(2 * K)
    result = {
        "pixel_jaccard": exact_jaccard,
        "shape_d4": shape_jaccard,
        "row_column": rc,
        "centroid": centroid,
        "adjacency": adj,
        "low_dct": dct,
    }
    result = {name: float(np.clip(value, 0.0, 1.0))
              for name, value in result.items()}
    result["composite"] = sum(WEIGHTS[name] * value
                              for name, value in result.items())
    return result


def similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Return position, rigid-shape, geometry and low-band DCT similarity.

    Full 7x7 DCT Euclidean distance is intentionally not included: Parseval's
    identity makes it identical to spatial Euclidean distance. Low-frequency
    DCT is a lossy feature for scoring only; the original mask is preserved.
    """
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("predeclared similarity weights must sum to one")
    return _pair_descriptors(_describe(image_key(a)), _describe(image_key(b)))


def signature(image: np.ndarray) -> np.ndarray:
    """Fixed 27-dimensional image signature for clustering and analog lookup.

    No outcome labels, fitted coefficients, or future-period information enter
    this representation.
    """
    d = _describe(image_key(image))
    return np.concatenate((
        d["low_dct"].ravel()[1:] / sqrt(K),
        d["rows"] / K, d["cols"] / K,
        d["centroid"] / 6,
        d["adj"] / comb(K, 2),
    )).astype(np.float64)


def random_six_images(*, count: int, seed: int) -> np.ndarray:
    """Generate independent legal IID 7x7 masks without neural models."""
    if not isinstance(count, int) or count < 1 or seed < 0:
        raise ValueError("count must be positive and seed nonnegative")
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.choice(N, K, replace=False) for _ in range(count)])
    result = np.zeros((count, N), dtype=np.uint8)
    result[np.arange(count)[:, None], draws] = 1
    return result.reshape(count, SIDE, SIDE)


def conditional_chance_similarity(predicted: np.ndarray,
                                  controls: np.ndarray) -> dict[str, float]:
    """Expected similarity to an IID target for this exact forecast mask.

    Controls must be sampled independently of the true evaluated target.
    This helps prevent central or repetitive shapes being mistaken for
    predictive evidence merely because they score well under a shape metric.
    """
    p = _describe(image_key(predicted))
    x = np.asarray(controls)
    if x.ndim != 3 or x.shape[1:] != (SIDE, SIDE) or not len(x):
        raise ValueError("controls must have shape [samples,7,7]")
    rows = [_pair_descriptors(p, _describe(image_key(item))) for item in x]
    return {name: float(np.mean([item[name] for item in rows]))
            for name in (*WEIGHTS, "composite")}
