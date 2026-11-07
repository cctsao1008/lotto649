"""Fresh chronological image research with null controls and no old test suite."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np

from .core import encode, decode, iid_images, iid_expected, signature, similarity
from .models import METHODS, ImageForecaster, diversity

DATA = Path("data/bin/lotto649_images_v2.bin")


def load_history(path: Path = DATA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read verified 7x7 BIN v2 images as the primary model input.

    A .npy source remains available only for existing archived-data parity
    experiments. BIN v2 stores no date: missing dates are represented by 0
    internally and never shown as a fictitious calendar date. Calendar year
    is instead recovered from the official draw ID.
    """
    path = Path(path)
    if path.suffix.lower() == ".bin":
        from .bin_dataset import decode_bin

        records = decode_bin(path.read_bytes())
        images = records["image"].copy()
        ids = records["draw_id"].astype(np.int64)
        dates = np.zeros(len(ids), dtype=np.int64)
        if len(ids) < 50:
            raise ValueError("too few BIN images for chronological analysis")
        return images, dates, ids
    if path.suffix.lower() != ".npy":
        raise ValueError("image input must be a BIN v2 or normalized NPY file")
    archive = np.load(path, allow_pickle=False)
    if (archive.dtype.names is None or not
            {"date", "draw_id", "numbers", "special"}.issubset(archive.dtype.names)):
        raise ValueError("expected normalized archive with date/draw_id/numbers/special")
    dates = archive["date"].astype(np.int64)
    draws = archive["numbers"].astype(np.int64)
    ids = archive["draw_id"].astype(np.int64)
    special = archive["special"].astype(np.int64)
    if (len(draws) < 50 or draws.shape != (len(dates), 6)
            or np.any(np.diff(dates) <= 0)
            or len(set(ids.tolist())) != len(ids)
            or np.any(ids <= 0) or np.any(special < 1) or np.any(special > 49)
            or np.any(np.any(draws == special[:, None], axis=1))):
        raise ValueError("invalid archive identity or chronology")
    for date in dates:
        datetime.strptime(str(date), "%Y%m%d")
    images = np.stack([encode(row) for row in draws])
    return images, dates, ids


def calendar_years(draw_ids: np.ndarray) -> np.ndarray:
    """Derive Gregorian calendar year from ROC-year official draw IDs."""
    ids = np.asarray(draw_ids, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 1_000_000):
        raise ValueError("invalid draw IDs for ROC-year conversion")
    return ids // 1000000 + 1911


def split(n: int, window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 60/20/20 by targets; never shuffle train/validation/test."""
    a, b = int(0.6*n), int(0.8*n)
    if window < 2 or a - window < 20 or b-a < 10 or n-b < 10:
        raise ValueError("insufficient observations for chronological split")
    return np.arange(window, a), np.arange(a, b), np.arange(b, n)


def normalized_signatures(images: np.ndarray, train_end: int) -> np.ndarray:
    """Stable training-only centering; zero-variance patterns have zero direction."""
    feats = np.stack([signature(im) for im in images])
    reference = feats[0]
    centered = feats[:train_end] - reference
    mean = reference + centered.mean(axis=0)
    scale = np.maximum(centered.std(axis=0), 0.05)
    z = (feats - mean)/scale
    norm = np.linalg.norm(z, axis=1)
    result = np.zeros_like(z)
    good = norm > 1e-10
    result[good] = z[good]/norm[good, None]
    return result


def year_shuffle(images: np.ndarray, years: np.ndarray, *, start: int,
                 end: int, rng: np.random.Generator) -> np.ndarray:
    """Permute images only within a single calendar year and chosen segment."""
    result = images.copy()
    for year in np.unique(years[start:end]):
        ids = np.flatnonzero(years[start:end] == year)+start
        result[ids] = images[ids[rng.permutation(len(ids))]]
    return result


def temporal_audit(images: np.ndarray, dates: np.ndarray, *, train_end: int,
                   start: int, end: int, shuffles: int = 99, seed: int = 649) -> dict:
    """Lag-one signature correlation versus same-year order permutations."""
    if shuffles < 2 or not 0 <= start < end <= len(images) or end-start < 5:
        raise ValueError("invalid temporal permutation parameters")
    years = dates//10000
    z = normalized_signatures(images, train_end)
    targets = np.arange(start+1, end)
    targets = targets[years[targets] == years[targets-1]]
    if len(targets) < 2:
        raise ValueError("too few same-year pairs")
    def score(array):
        return float((array[targets]*array[targets-1]).sum(axis=1).mean())
    observed = score(z)
    rng = np.random.default_rng(seed)
    null = np.array([score(year_shuffle(z, years, start=start, end=end, rng=rng))
                     for _ in range(shuffles)])
    center = null.mean()
    tail = (1+np.count_nonzero(np.abs(null-center) >= abs(observed-center)))/(shuffles+1)
    return {"lag1_observed": observed, "lag1_shuffled_mean": float(center),
            "lag1_delta": float(observed-center), "two_sided_tail": float(tail),
            "within_year_pairs": int(len(targets)), "permutations": shuffles}


def _evaluate(model: ImageForecaster, images: np.ndarray, ids: np.ndarray,
              *, baseline: int, seed: int, details: bool) -> tuple[dict, list]:
    samples = defaultdict(list)
    predicted = defaultdict(list)
    rows = []
    for idx in ids:
        idx = int(idx)
        iid = iid_images(baseline, seed=seed+1000003*idx)
        item = {"index": idx, "actual": decode(images[idx]), "methods": {}} if details else None
        for method in METHODS:
            guess = model.predict(idx, method)
            raw = similarity(guess, images[idx])
            expected = iid_expected(guess, iid)
            residual = raw["composite"]-expected
            predicted[method].append(guess)
            samples[method].append(residual)
            if details:
                item["methods"][method] = {
                    "predicted": decode(guess), "raw_similarity": raw,
                    "iid_expected": expected, "adjusted": float(residual)}
        if details:
            rows.append(item)
    metrics = {}
    for name in METHODS:
        values = np.asarray(samples[name])
        metrics[name] = {"adjusted_mean": float(values.mean()),
                         "raw_mean": float(np.mean([r["methods"][name]["raw_similarity"]["composite"] for r in rows])) if details else None,
                         "unique_patterns": diversity(np.stack(predicted[name])),
                         "per_draw_adjusted": values.tolist() if details else None}
    return metrics, rows


def research(path: Path = DATA, *, year: int | None = None,
             window: int = 5, baseline: int = 32, shuffles: int = 99,
             seed: int = 649) -> dict:
    """Frozen training analogs, validation selection and exploratory test replay."""
    images, dates, ids = load_history(path)
    train, val, test = split(len(images), window=window)
    selected = test if year is None else test[calendar_years(ids[test]) == year]
    if not len(selected) or baseline < 2 or shuffles < 2:
        raise ValueError("no evaluation draws or invalid control count")
    model = ImageForecaster(images, train_end=int(val[0]), window=window)
    validation, _ = _evaluate(model, images, val, baseline=max(8, baseline//2),
                               seed=seed+11, details=False)
    chosen = max(METHODS, key=lambda m: validation[m]["adjusted_mean"])
    result, rows = _evaluate(model, images, selected, baseline=baseline,
                             seed=seed+23, details=True)
    # BIN lacks exact dates; use ROC-year labels for year-stratified shuffles.
    chronology_for_audit = np.where(dates > 0, dates,
                                    calendar_years(ids) * 10000)
    audit = temporal_audit(images, chronology_for_audit, train_end=int(val[0]),
                           start=int(selected[0]), end=int(selected[-1])+1,
                           shuffles=shuffles, seed=seed)
    return {"schema": "lotto7x7.visual.v1", "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exploratory_only": True,
            "metric": "composite shape similarity minus conditional IID expectation",
            "model_selected_on_validation_only": chosen,
            "total_draws": len(images), "evaluated_draws": len(selected),
            "split": {"train": [int(train[0]), int(train[-1])],
                      "validation": [int(val[0]), int(val[-1])],
                      "test": [int(selected[0]), int(selected[-1])]},
            "config": {"window": window, "baseline": baseline,
                       "shuffles": shuffles, "seed": seed},
            "temporal_order": audit, "validation": validation,
            "evaluation": result,
            "rows": [{**row, "date": (int(dates[row["index"]])
                                        if dates[row["index"]] else None),
                      "year": int(calendar_years(ids[[row["index"]]])[0]),
                      "draw_id": int(ids[row["index"]])} for row in rows]}
