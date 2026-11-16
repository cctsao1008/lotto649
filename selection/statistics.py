"""Score candidate combinations against exact fair 6/49 distributions."""
from collections import Counter

import numpy as np

from .combinations import TOTAL, exact_sum_counts, uniform_candidates
from .shapes import classify_numbers


def _deficits(observed, probabilities, count):
    # A variance floor of one draw limits the influence of rare categories.
    expected = count * probabilities
    scale = np.sqrt(expected * (1 - probabilities) + 1)
    return np.tanh((expected - observed) / scale)


def score_candidates(history, *, pool_size, seed, weights, shape_counts):
    """Return candidates, three bounded component scores and their weighted mean."""
    pool = uniform_candidates(pool_size, seed)
    # Use a signed type so totals above 255 cannot wrap.
    history = np.asarray(history, dtype=np.int64)
    totals = np.asarray(exact_sum_counts(), dtype=float) / TOTAL
    sums = _deficits(np.bincount(history.sum(axis=1), minlength=len(totals)),
                     totals, len(history))
    numbers = _deficits(np.bincount(history.ravel(), minlength=50),
                        np.full(50, 6 / 49), len(history))

    history_shapes = Counter(map(int, classify_numbers(history)["edge_code"]))
    pool_shapes = classify_numbers(pool)["edge_code"]
    if history_shapes.keys() - shape_counts.keys() or set(pool_shapes) - shape_counts.keys():
        raise ValueError("shape reference does not cover the observed classes")
    codes = list(shape_counts)
    shapes = dict(zip(codes, _deficits(
        np.array([history_shapes[code] for code in codes]),
        np.array([shape_counts[code] / TOTAL for code in codes]), len(history),
    )))

    components = np.column_stack((
        sums[pool.sum(axis=1)], numbers[pool].mean(axis=1),
        [shapes[code] for code in pool_shapes],
    ))
    combined = (components * weights).sum(axis=1)
    return pool, components, combined
