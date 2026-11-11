"""v0.29 P0: deterministic, outcome-blind geometric features of six-hot 7x7 images.

Regions are fixed lattice-cell sets, not continuous area probabilities.
Sequential mapping is canonical; snake and seeded random are sensitivity controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

SCHEMA = "lotto649.geometry.v0.29.p0.v1"
CATALOG_SCHEMA = "lotto649.geometry.catalog.v0.29.p0.v1"
MAPPING_METHODS = ("sequential", "snake", "random")
GRID = 7
POINTS = tuple((r, c) for r in range(GRID) for c in range(GRID))
INVARIANT_REGIONS = (
    "disk_r1", "disk_r2", "disk_r3", "annulus_r2_r3", "diamond_r3",
    "center_square_3", "border", "diagonal_union",
)
ORIENTED_REGIONS = ("ellipse_horizontal", "ellipse_vertical")


def _region_cells():
    cells = {}
    for r in (1, 2, 3):
        cells[f"disk_r{r}"] = tuple((y, x) for y, x in POINTS
                                    if (y - 3) ** 2 + (x - 3) ** 2 <= r * r)
    cells["annulus_r2_r3"] = tuple((y, x) for y, x in POINTS
                                    if 4 < (y - 3) ** 2 + (x - 3) ** 2 <= 9)
    cells["diamond_r3"] = tuple((y, x) for y, x in POINTS
                                if abs(y - 3) + abs(x - 3) <= 3)
    cells["center_square_3"] = tuple((y, x) for y, x in POINTS
                                     if abs(y - 3) <= 1 and abs(x - 3) <= 1)
    cells["border"] = tuple((y, x) for y, x in POINTS
                            if y in (0, 6) or x in (0, 6))
    cells["diagonal_union"] = tuple((y, x) for y, x in POINTS
                                    if y == x or y + x == 6)
    cells["ellipse_horizontal"] = tuple((y, x) for y, x in POINTS
                                         if 4 * (x - 3) ** 2 + 9 * (y - 3) ** 2 <= 36)
    cells["ellipse_vertical"] = tuple((y, x) for y, x in POINTS
                                       if 9 * (x - 3) ** 2 + 4 * (y - 3) ** 2 <= 36)
    return cells


REGIONS = _region_cells()
REGION_SETS = {name: frozenset(cells) for name, cells in REGIONS.items()}
DIST2_SUPPORT = tuple(sorted({dy * dy + dx * dx
                              for dy in range(7) for dx in range(7) if dy or dx}))


def mapping_positions(method: str = "sequential", seed: int = 649) -> tuple[tuple[int, int], ...]:
    if method not in MAPPING_METHODS:
        raise ValueError("unsupported image mapping")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if method == "sequential":
        mapping = POINTS
    elif method == "snake":
        mapping = tuple((r, c if r % 2 == 0 else 6 - c) for r, c in POINTS)
    else:
        indices = list(range(49))
        Random(seed).shuffle(indices)
        mapping = tuple(POINTS[i] for i in indices)
    if len(set(mapping)) != 49:
        raise AssertionError("not a bijective six-of-49 layout")
    return mapping


def validate_ticket(numbers: object) -> tuple[int, ...]:
    if (not isinstance(numbers, (tuple, list)) or len(numbers) != 6 or
            any(type(n) is not int or not 1 <= n <= 49 for n in numbers) or
            len(set(numbers)) != 6):
        raise ValueError("expected exactly six unique integers in 1..49")
    return tuple(sorted(numbers))


def _graph_features(cells: tuple[tuple[int, int], ...]):
    adjacency = {p: set() for p in cells}
    cardinal = diagonal = 0
    d2 = Counter()
    for a, b in combinations(cells, 2):
        dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
        d2[dy * dy + dx * dx] += 1
        if max(dy, dx) == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
            if dy + dx == 1:
                cardinal += 1
            else:
                diagonal += 1
    remaining = set(cells)
    component_sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)
                size += 1
        component_sizes.append(size)
    component_sizes.sort(reverse=True)
    components = len(component_sizes)
    edge_count = cardinal + diagonal
    return {
        "connected_components": components,
        "component_partition": component_sizes,
        "cardinal_edges": cardinal,
        "diagonal_edges": diagonal,
        "graph_cycle_rank": edge_count - 6 + components,
        "pair_distance_squared_histogram": [d2.get(value, 0) for value in DIST2_SUPPORT],
    }


def geometry_features(numbers: object, *, mapping: str = "sequential",
                      seed: int = 649) -> dict:
    ticket = validate_ticket(numbers)
    index = mapping_positions(mapping, seed)
    cells = tuple(sorted(index[n - 1] for n in ticket))
    bitmap = tuple("".join("1" if (r, c) in cells else "0" for c in range(7))
                   for r in range(7))
    graph = _graph_features(cells)
    row_span = max(y for y, _ in cells) - min(y for y, _ in cells)
    col_span = max(x for _, x in cells) - min(x for _, x in cells)
    return {
        "schema": SCHEMA,
        "mapping_id": mapping if mapping != "random" else f"random:{seed}",
        "ticket": list(ticket),
        "bitmap": list(bitmap),
        "region_occupancies": {name: sum(cell in region for cell in cells)
                               for name, region in REGION_SETS.items()},
        "graph": graph,
        "bounding_spans_sorted": sorted([row_span, col_span]),
        "center_squared_radius_sum": sum((r - 3) ** 2 + (c - 3) ** 2 for r, c in cells),
    }


def catalog() -> dict:
    common = {
        "schema": CATALOG_SCHEMA,
        "coordinate_convention": "zero-based(row,col), row-major canonical numbers 1..49",
        "boundary_rule": "inclusive lattice cell center; no fractional cell areas",
        "invariant_regions_under_D4": list(INVARIANT_REGIONS),
        "oriented_regions_under_D4": list(ORIENTED_REGIONS),
        "pair_distance_squared_support": list(DIST2_SUPPORT),
        "regions": {
            name: {"lattice_cell_count": len(cells),
                   "cells": [list(p) for p in cells]}
            for name, cells in REGIONS.items()
        },
    }
    payload = json.dumps(common, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return {**common, "definition_sha256": sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true", help="Report fixed region catalog")
    parser.add_argument("--numbers", nargs=6, type=int, metavar="N")
    parser.add_argument("--mapping", choices=MAPPING_METHODS, default="sequential")
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path, help="Write one new JSON file without overwriting")
    args = parser.parse_args()
    if args.catalog == (args.numbers is not None):
        parser.error("choose exactly one of --catalog or --numbers with six values")
    output = catalog() if args.catalog else geometry_features(args.numbers, mapping=args.mapping, seed=args.seed)
    text = json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(text)
        print(f"PASS: wrote new geometry record to {args.out}")


if __name__ == "__main__":
    main()
