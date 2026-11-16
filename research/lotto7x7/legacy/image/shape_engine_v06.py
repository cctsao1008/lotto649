"""Lotto7x7 v0.6: canonical six-pixel shapes and exact D4/translation orbits.

A family is an equivalence class under *non-wrapping* translations,
90-degree rotations, and reflections on a 7x7 lattice. A family is an
abstract image shape, not a forecast of lottery numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import comb
from typing import Iterable

import numpy as np

SIDE = 7
PIXELS = 6
UNIVERSE_SIZE = comb(49, 6)
Points = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Placement:
    """One distinct legal placement of a canonical shape on the 7x7 board."""
    numbers: tuple[int, ...]
    mirrored: bool
    rotations_ccw: int
    row_shift: int
    col_shift: int

    @property
    def points(self) -> Points:
        return tuple(sorted(((n - 1) // SIDE, (n - 1) % SIDE)
                            for n in self.numbers))


def validate_images(images: np.ndarray) -> np.ndarray:
    """Strictly validate a chronological array of exact-six binary bitmaps."""
    x = np.asarray(images)
    if (x.ndim != 3 or x.shape[1:] != (SIDE, SIDE) or len(x) < 1
            or not np.issubdtype(x.dtype, np.number) or not np.isrealobj(x)
            or not np.all(np.isfinite(x)) or not np.all((x == 0) | (x == 1))
            or not np.all(x.sum(axis=(1, 2)) == PIXELS)):
        raise ValueError("expected [draws,7,7] binary images with exactly six pixels")
    return np.asarray(x, dtype=np.uint8)


def _points(points: Iterable[tuple[int, int]]) -> Points:
    """Validate lattice points before any normalization or transformation."""
    try:
        values = tuple(points)
        if len(values) != PIXELS:
            raise ValueError("a shape must contain exactly six points")
        for p in values:
            if (not isinstance(p, (tuple, list)) or len(p) != 2
                    or any(not isinstance(v, (int, np.integer)) for v in p)):
                raise ValueError("shape coordinates must be integer pairs")
        pts = tuple(sorted((int(r), int(c)) for r, c in values))
    except TypeError as exc:
        raise ValueError("shape coordinates must be integer pairs") from exc
    if len(set(pts)) != PIXELS:
        raise ValueError("shape points must be distinct")
    return pts


def image_points(image: np.ndarray) -> Points:
    """Extract six points from a 7x7 bitmap, keeping pixel coordinates."""
    x = np.asarray(image)
    if (x.shape != (SIDE, SIDE) or not np.issubdtype(x.dtype, np.number)
            or not np.isrealobj(x) or not np.all(np.isfinite(x))
            or not np.all((x == 0) | (x == 1)) or int(x.sum()) != PIXELS):
        raise ValueError("image must be a 7x7 binary bitmap with exactly six pixels")
    return _points(tuple(zip(*np.nonzero(x))))


def bitmap(numbers: Iterable[int]) -> np.ndarray:
    """Convert exactly six distinct labels 1..49 to a sequential 7x7 image."""
    values = tuple(numbers)
    if (len(values) != PIXELS
            or any(not isinstance(n, (int, np.integer)) or n < 1 or n > 49
                   for n in values)
            or len(set(values)) != PIXELS):
        raise ValueError("numbers must be exactly six distinct integers in 1..49")
    x = np.zeros((SIDE, SIDE), dtype=np.uint8)
    x.ravel()[np.array(values, dtype=np.int64) - 1] = 1
    return x


def _normalize(points: Points) -> Points:
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _transform(points: Points, *, mirrored: bool, rotations: int) -> Points:
    if not isinstance(rotations, int) or rotations not in (0, 1, 2, 3):
        raise ValueError("rotations must be a quarter turn count 0..3")
    out = []
    for row, col in points:
        r, c = row, -col if mirrored else col
        for _ in range(rotations):
            r, c = -c, r
        out.append((r, c))
    return _normalize(_points(out))


def orientations(points: Iterable[tuple[int, int]]) -> tuple[tuple[Points, bool, int], ...]:
    """Unique normalized D4 orientations with deterministic transform metadata."""
    base = _normalize(_points(points))
    unique: dict[Points, tuple[bool, int]] = {}
    for mirrored in (False, True):
        for rotation in range(4):
            transformed = _transform(base, mirrored=mirrored, rotations=rotation)
            unique.setdefault(transformed, (mirrored, rotation))
    return tuple((p, *unique[p]) for p in sorted(unique))


def canonical_points(image_or_points: np.ndarray | Iterable[tuple[int, int]]) -> Points:
    """D4+translation-invariant canonical point set, anchored at top-left."""
    x = np.asarray(image_or_points)
    pts = image_points(x) if x.shape == (SIDE, SIDE) else _points(image_or_points)
    return min(p for p, _, _ in orientations(pts))


def shape_key(image_or_points: np.ndarray | Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{r},{c}" for r, c in canonical_points(image_or_points))


def parse_shape_key(key: str) -> Points:
    """Accept only keys already in unique canonical representation."""
    if not isinstance(key, str):
        raise ValueError("invalid shape key")
    try:
        pts = _points(tuple(tuple(int(v) for v in item.split(","))
                            for item in key.split(";")))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid shape key") from exc
    if (any(not 0 <= r < SIDE or not 0 <= c < SIDE for r, c in pts)
            or shape_key(pts) != key):
        raise ValueError("shape key must be normalized and canonical")
    return pts


def shape_id(key: str) -> str:
    """Compact stable ID; full canonical key is retained to detect collisions."""
    parse_shape_key(key)
    return "S-" + hashlib.sha256((f"7x7:6:{key}").encode("ascii")).hexdigest()[:16]


def orbit_size(key: str) -> int:
    """Exact number of distinct legal positions, without enumeration."""
    origin = parse_shape_key(key)
    total = 0
    for pts, _, _ in orientations(origin):
        width = max(c for _, c in pts) + 1
        height = max(r for r, _ in pts) + 1
        total += (SIDE - height + 1) * (SIDE - width + 1)
    if not 1 <= total <= UNIVERSE_SIZE:
        raise AssertionError("invalid orbit cardinality")
    return total


def enumerate_placements(key: str) -> tuple[Placement, ...]:
    """Enumerate all distinct legal, non-wrapping images in one D4 orbit."""
    origin = parse_shape_key(key)
    placements: list[Placement] = []
    observed: set[tuple[int, ...]] = set()
    for pts, mirrored, rotations in orientations(origin):
        width = max(c for _, c in pts) + 1
        height = max(r for r, _ in pts) + 1
        for row_shift in range(SIDE - height + 1):
            for col_shift in range(SIDE - width + 1):
                numbers = tuple(sorted((r + row_shift) * SIDE + c + col_shift + 1
                                       for r, c in pts))
                if numbers in observed:
                    raise AssertionError("duplicate placement in distinct orientations")
                observed.add(numbers)
                placements.append(Placement(numbers, mirrored, rotations,
                                            row_shift, col_shift))
    if len(placements) != orbit_size(key):
        raise AssertionError("computed orbit does not match enumerated placements")
    return tuple(placements)


def history_shape_families(images: np.ndarray, draw_ids: np.ndarray) -> list[dict]:
    """Classify only observed historical images, without test or target leakage."""
    x = validate_images(images)
    ids = np.asarray(draw_ids)
    if (ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("draw ids must be positive strictly increasing integers")
    items: dict[str, dict] = {}
    for image, draw_id in zip(x, ids):
        key = shape_key(image)
        if key not in items:
            items[key] = {
                "shape_key": key, "shape_id": shape_id(key),
                "historical_count": 0, "first_draw_id": int(draw_id),
                "last_draw_id": int(draw_id), "orbit_size": orbit_size(key),
            }
        items[key]["historical_count"] += 1
        items[key]["last_draw_id"] = int(draw_id)
    return [items[key] for key in sorted(items)]
