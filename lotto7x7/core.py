"""Lossless Lotto649 image representation and image-first similarity metrics."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.fft import dctn, idctn

SIDE, NUMBERS, PIXELS = 7, 49, 6
# A preset research metric, not a physical model of the drawing machine.
WEIGHTS = {"shape": 0.50, "occupancy": 0.20, "dct": 0.20, "pixels": 0.10}


def layout(kind: str = "sequential", seed: int = 649) -> np.ndarray:
    """Raster-cell to lottery number bijection, recorded with every experiment."""
    ids = np.arange(1, NUMBERS + 1).reshape(SIDE, SIDE)
    if kind == "sequential":
        return ids
    if kind == "snake":
        ids[1::2] = ids[1::2, ::-1]
        return ids
    if kind == "permuted" and isinstance(seed, int) and seed >= 0:
        return np.random.default_rng(seed).permutation(ids.ravel()).reshape(7, 7)
    raise ValueError("unknown layout or invalid permutation seed")


def validate(image: np.ndarray) -> np.ndarray:
    """Validate one 7x7 grid with exactly six binary active pixels."""
    x = np.asarray(image)
    if (x.shape != (7, 7) or not np.issubdtype(x.dtype, np.number)
            or not np.isrealobj(x) or not np.all(np.isfinite(x))
            or not np.all((x == 0) | (x == 1)) or int(x.sum()) != PIXELS):
        raise ValueError("image must be a 7x7 binary grid with exactly six active pixels")
    return x.astype(np.uint8)


def encode(numbers: np.ndarray, *, mapping: str = "sequential", seed: int = 649) -> np.ndarray:
    """Encode one six-number draw to a lossless image; ignore bonus number."""
    ids = np.asarray(numbers)
    if (ids.shape != (6,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 1) or np.any(ids > 49) or len(np.unique(ids)) != 6):
        raise ValueError("draw must contain six distinct integers in 1..49")
    return np.isin(layout(mapping, seed), ids).astype(np.uint8)


def decode(image: np.ndarray, *, mapping: str = "sequential", seed: int = 649) -> list[int]:
    """Recover the six numbers from the bitmap under the same layout."""
    return sorted(layout(mapping, seed)[validate(image) == 1].astype(int).tolist())


def spectrum(image: np.ndarray, domain: str) -> np.ndarray:
    """Full 2D FFT or orthonormal DCT; no coefficients are discarded."""
    x = validate(image).astype(np.float64)
    if domain == "dct":
        return dctn(x, norm="ortho")
    if domain == "fft":
        return np.fft.fft2(x, norm="ortho")
    raise ValueError("domain must be 'dct' or 'fft'")


def invert_spectrum(coefficients: np.ndarray, domain: str) -> np.ndarray:
    """Inverse transform returns near-binary floats; verify before rounding."""
    c = np.asarray(coefficients)
    if c.shape != (7, 7) or not np.all(np.isfinite(c)):
        raise ValueError("expected finite 7x7 coefficients")
    if domain == "dct":
        if not np.isrealobj(c):
            raise ValueError("DCT coefficients must be real")
        return idctn(c, norm="ortho")
    if domain == "fft":
        restored = np.fft.ifft2(c, norm="ortho")
        if np.max(np.abs(restored.imag)) > 1e-9:
            raise ValueError("FFT spectrum is not conjugate symmetric")
        return restored.real
    raise ValueError("domain must be 'dct' or 'fft'")


@lru_cache(maxsize=40000)
def _descriptor(key: tuple[int, ...]) -> dict:
    x = np.zeros((7, 7), dtype=np.float64)
    x.ravel()[list(key)] = 1
    dct = dctn(x, norm="ortho")[:3, :3].ravel()[1:]
    points = np.argwhere(x == 1)
    variants = [np.argwhere(y == 1) for rot in range(4)
                for y in (np.rot90(x, rot), np.fliplr(np.rot90(x, rot)))]
    return {"x": x, "points": points, "variants": variants,
            "rows": x.sum(axis=1), "cols": x.sum(axis=0),
            "centroid": points.mean(axis=0), "low_dct": dct}


def _key(image: np.ndarray) -> tuple[int, ...]:
    return tuple(np.flatnonzero(validate(image)).astype(int).tolist())


def signature(image: np.ndarray) -> np.ndarray:
    """Image-only feature vector for temporal analog matching."""
    a = _descriptor(_key(image))
    return np.concatenate((a["low_dct"], a["rows"] / 6,
                           a["cols"] / 6, a["centroid"] / 6))


def _shape_overlap(a: dict, b: dict) -> int:
    """Max six-point overlap under D4 rotations/reflections and integer shifts."""
    best = 0
    for rotated in b["variants"]:
        offsets = Counter((int(p[0]-q[0]), int(p[1]-q[1]))
                          for p in a["points"] for q in rotated)
        best = max(best, max(offsets.values()))
        if best == 6:
            return 6
    return best


def similarity(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Geometric resemblance independent of exact six-number jackpot hits."""
    a, b = _descriptor(_key(predicted)), _descriptor(_key(observed))
    overlap = _shape_overlap(a, b)
    pixel_overlap = int(np.sum(a["x"] * b["x"]))
    # Non-negative, normalized [0,1] descriptors, identical images score 1.
    fields = {
        "shape": overlap / (12-overlap),
        "pixels": pixel_overlap / (12-pixel_overlap),
        "occupancy": 1-float(np.abs(a["rows"]-b["rows"]).sum()
                            +np.abs(a["cols"]-b["cols"]).sum())/24,
        "dct": 1-float(np.linalg.norm(a["low_dct"]-b["low_dct"])) / sqrt(12),
    }
    fields = {k: float(np.clip(v, 0, 1)) for k, v in fields.items()}
    fields["composite"] = sum(WEIGHTS[k]*fields[k] for k in WEIGHTS)
    return fields


def iid_images(count: int, seed: int) -> np.ndarray:
    """Independent fair random six-active-pixel masks for conditional controls."""
    if not isinstance(count, int) or count < 1 or seed < 0:
        raise ValueError("invalid IID sample count or seed")
    rng = np.random.default_rng(seed)
    result = np.zeros((count, 49), dtype=np.uint8)
    for row in result:
        row[rng.choice(49, 6, replace=False)] = 1
    return result.reshape(count, 7, 7)


def iid_expected(prediction: np.ndarray, controls: np.ndarray) -> float:
    """Chance expectation conditional on the proposed forecast shape."""
    if np.asarray(controls).ndim != 3 or np.asarray(controls).shape[1:] != (7, 7):
        raise ValueError("control images must be [n,7,7]")
    return float(np.mean([similarity(prediction, x)["composite"] for x in controls]))
