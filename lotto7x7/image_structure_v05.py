"""v0.5 image-first morphological, spectral and temporal structure research.

All descriptors are computed from immutable six-pixel images. Geometry and
spectra describe a chosen 7x7 numbering layout, not physical lottery geometry.
The null is independent *six-of-49* images, never independent Bernoulli pixels.
"""
from __future__ import annotations

from html import escape
import hashlib
from math import log, sqrt
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from .core import decode, layout, validate, iid_images
from .research import DATA, calendar_years, load_history

SCHEMA = "lotto7x7.image_structure.v0.5"
LAYOUTS = ("sequential", "snake", "permuted")
MORPHOLOGY_NAMES = (
    "centroid_row", "centroid_col", "radial_spread", "pair_distance_mean",
    "pair_distance_std", "bounding_box_fraction", "neighbor_edges_8",
    "components_8", "row_entropy", "column_entropy",
    *(f"row_count_{i}" for i in range(7)),
    *(f"column_count_{i}" for i in range(7)),
)
SPECTRAL_NAMES = (
    *(f"dct_{r}_{c}" for r in range(3) for c in range(3) if r or c),
    "dct_low_energy_ratio", "fft_low_energy_ratio",
    "fft_radial_band_1", "fft_radial_band_2", "fft_radial_band_3",
)
METRICS = ("pixel_overlap", "low_dct_cosine")


def check_images(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images)
    if (x.ndim != 3 or x.shape[1:] != (7, 7) or len(x) < 2 or
            not np.issubdtype(x.dtype, np.number) or not np.isrealobj(x) or
            not np.all(np.isfinite(x)) or not np.all((x == 0) | (x == 1)) or
            not np.all(x.sum(axis=(1, 2)) == 6)):
        raise ValueError("expected [n,7,7] binary images with exactly six points")
    return x.astype(np.uint8, copy=False)


def map_images(images: np.ndarray, *, kind: str = "sequential", seed: int = 649
               ) -> np.ndarray:
    """Lossless rerasterization; the six number labels remain unchanged."""
    x = check_images(images)
    numbers_in_raster_order = layout(kind, seed).reshape(-1) - 1
    return x.reshape(-1, 49)[:, numbers_in_raster_order].reshape(-1, 7, 7).copy()


def _components(points: np.ndarray) -> int:
    remaining = {tuple(map(int, p)) for p in points}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            adjacent = {(r + dr, c + dc) for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1) if dr or dc}
            connected = remaining.intersection(adjacent)
            remaining.difference_update(connected)
            stack.extend(connected)
    return count


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts != 0] / 6.0
    return float(-np.sum(p * np.log(p)) / log(7.0))


def describe_images(images: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic per-image features; NO features of a future image are used.

    DCT uses orthonormal coefficients. FFT low-frequency energies omit DC;
    fractions use non-DC spectral energy as their denominator. Both are
    measures of the ARBITRARY 7x7 numbering arrangement.
    """
    masks = check_images(images)
    n = len(masks)
    morphology = np.empty((n, len(MORPHOLOGY_NAMES)), dtype=np.float64)
    for k, image in enumerate(masks):
        pts = np.argwhere(image).astype(np.float64)
        centroid = pts.mean(axis=0)
        diff = pts - centroid
        pairs = pts[:, None, :] - pts[None, :, :]
        indices = np.triu_indices(6, 1)
        distances = np.sqrt(np.sum(pairs[indices] ** 2, axis=1))
        neighbor_edges = np.logical_and(
            np.max(np.abs(pairs), axis=2) == 1,
            np.triu(np.ones((6, 6), dtype=bool), k=1),
        ).sum()
        extent = (pts.max(axis=0) - pts.min(axis=0) + 1).prod()
        rows = image.sum(axis=1).astype(np.float64)
        cols = image.sum(axis=0).astype(np.float64)
        morphology[k] = [
            *(centroid / 6),
            float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))) / sqrt(72)),
            float(distances.mean() / sqrt(72)),
            float(distances.std() / sqrt(72)),
            float(extent / 49.0), float(neighbor_edges / 15.0),
            float(_components(pts) / 6.0), _entropy(rows), _entropy(cols),
            *(rows / 6), *(cols / 6),
        ]
    floats = masks.astype(np.float64)
    dct = dctn(floats, axes=(-2, -1), norm="ortho")
    fft = np.fft.fft2(floats, axes=(-2, -1), norm="ortho")
    fft_energy = np.abs(fft) ** 2
    full_non_dc = np.maximum(np.sum(dct ** 2, axis=(-2, -1)) - dct[:, 0, 0] ** 2,
                             1e-12)
    low = dct[:, :3, :3].reshape(n, 9)[:, 1:]
    low_energy = np.sum(low ** 2, axis=1) / full_non_dc
    frequencies = np.fft.fftfreq(7) * 7
    radii = np.sqrt(frequencies[:, None] ** 2 + frequencies[None, :] ** 2)
    low_mask = (radii > 0) & (radii <= 2.0)
    fft_low = fft_energy[:, low_mask].sum(axis=1) / full_non_dc
    bands = [fft_energy[:, (radii > lower) & (radii <= upper)].sum(axis=1)
             / full_non_dc for lower, upper in [(0, 1.5), (1.5, 2.5), (2.5, 4.3)]]
    spectral = np.column_stack([low, low_energy, fft_low, *bands])
    if not (np.all(np.isfinite(morphology)) and np.all(np.isfinite(spectral))):
        raise RuntimeError("non-finite image descriptors")
    return {"morphology": morphology, "spectral": spectral,
            "low_dct": low, "dct": dct}


def _time_pair_means(images: np.ndarray, low: np.ndarray,
                     years: np.ndarray, lags: tuple[int, ...]) -> dict[int, dict]:
    y = np.asarray(years)
    results = {}
    norm = np.linalg.norm(low, axis=1)
    unit = low / np.maximum(norm[:, None], 1e-10)
    for lag in lags:
        positions = np.arange(lag, len(images))
        positions = positions[y[positions] == y[positions - lag]]
        if len(positions) < 3:
            raise ValueError("too few within-year temporal pairs for a lag")
        current = images[positions]
        past = images[positions-lag]
        common = (current * past).sum(axis=(1, 2)).astype(np.float64)
        cosine = np.sum(unit[positions] * unit[positions-lag], axis=1)
        results[lag] = {"pairs": len(positions), "pixel_overlap": float(common.mean()),
                        "low_dct_cosine": float(cosine.mean())}
    return results


def temporal_null(images: np.ndarray, years: np.ndarray, *,
                  lags: tuple[int, ...] = (1, 2, 3, 5, 10),
                  permutations: int = 499, seed: int = 649) -> dict:
    """Within-year order null preserves individual images and yearly marginals.

    Exact 6-of-49 pixel overlap is numbering-layout invariant; low-band DCT
    cosine is intentionally sensitive to arbitrary spatial layout.
    Monte Carlo p values are two-sided deviations from the permutation center.
    Bonferroni adjustment covers (number_of_lags * two_metrics) tests.
    """
    x = check_images(images)
    y = np.asarray(years)
    if (y.shape != (len(x),) or permutations < 9 or seed < 0 or
            not lags or any(not isinstance(lag, int) or lag < 1 or lag >= len(x)
                            for lag in lags) or len(set(lags)) != len(lags)):
        raise ValueError("invalid temporal-null configuration")
    low = describe_images(x)["low_dct"]
    observed = _time_pair_means(x, low, y, lags)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == year) for year in np.unique(y)]
    null = {lag: {name: np.empty(permutations, dtype=np.float64)
                  for name in METRICS} for lag in lags}
    for iteration in range(permutations):
        perm = np.arange(len(x))
        for group in groups:
            perm[group] = rng.permutation(group)
        score = _time_pair_means(x[perm], low[perm], y, lags)
        for lag in lags:
            for metric in METRICS:
                null[lag][metric][iteration] = score[lag][metric]
    result = {}
    m = len(lags) * len(METRICS)
    for lag in lags:
        result[str(lag)] = {"pairs": observed[lag]["pairs"]}
        for metric in METRICS:
            values = null[lag][metric]
            center = float(values.mean())
            difference = abs(observed[lag][metric] - center)
            tail = (1 + int(np.count_nonzero(np.abs(values-center) >= difference-1e-12))) / (permutations+1)
            result[str(lag)][metric] = {
                "observed": observed[lag][metric], "null_mean": center,
                "null_95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
                "delta": float(observed[lag][metric]-center),
                "permutation_p_two_sided": float(tail),
                "bonferroni_p": float(min(1.0, m*tail)),
            }
    return result


def _summary_stats(array: np.ndarray, names: tuple[str, ...]) -> dict:
    return {name: {"mean": float(array[:, idx].mean()),
                   "std": float(array[:, idx].std())}
            for idx, name in enumerate(names)}


def image_study(path: Path = DATA, *, permutations: int = 499,
                null_images: int = 4096, seed: int = 649,
                lags: tuple[int, ...] = (1, 2, 3, 5, 10),
                mapping_seeds: int = 2) -> dict:
    if (permutations < 9 or null_images < 100 or mapping_seeds < 0 or
            mapping_seeds > 8 or seed < 0):
        raise ValueError("invalid image research configuration")
    images, dates, ids = load_history(path)
    images = check_images(images)
    if len(ids) != len(images) or np.any(np.diff(ids) <= 0):
        raise ValueError("invalid chronological image draw ids")
    years = calendar_years(ids)
    fixed_iid = iid_images(null_images, seed+4103)
    mappings = [("sequential", seed), ("snake", seed)] + [
        ("permuted", seed + i + 1) for i in range(mapping_seeds)]
    mapping_results = []
    for kind, value in mappings:
        mapped = map_images(images, kind=kind, seed=value)
        iid_mapped = map_images(fixed_iid, kind=kind, seed=value)
        feats = describe_images(mapped)
        iid_feats = describe_images(iid_mapped)
        metrics = temporal_null(mapped, years, lags=lags,
                                permutations=permutations, seed=seed+120)
        def section(k, names):
            obs = _summary_stats(feats[k], names)
            base = _summary_stats(iid_feats[k], names)
            return {name: {
                **obs[name], "iid_mean": base[name]["mean"],
                "iid_std": base[name]["std"],
                "delta_vs_iid": obs[name]["mean"] - base[name]["mean"],
            } for name in names}
        mapping_results.append({
            "mapping": kind, "mapping_seed": value,
            "morphology": section("morphology", MORPHOLOGY_NAMES),
            "spectral": section("spectral", SPECTRAL_NAMES),
            "temporal": metrics,
            "pixel_occupancy": mapped.mean(axis=0).tolist(),
            "iid_pixel_occupancy": iid_mapped.mean(axis=0).tolist(),
            "mean_dct_power": np.mean(feats["dct"]**2,axis=0).tolist(),
            "iid_mean_dct_power": np.mean(iid_feats["dct"]**2,axis=0).tolist(),
        })
    latest_features = describe_images(images)
    return {
        "schema": SCHEMA, "draws": len(images), "first_draw_id": int(ids[0]),
        "last_draw_id": int(ids[-1]), "year_range": [int(years[0]), int(years[-1])],
        "input_sha256": hashlib.sha256(images.tobytes()+ids.tobytes()).hexdigest(),
        "configuration": {"permutations": permutations,
                          "null_images": null_images, "seed": seed,
                          "lags": list(lags), "mapping_seeds": mapping_seeds},
        "exact_iid_expected_overlap": 36.0/49.0,
        "mapping_comparisons": mapping_results,
        "latest_images": [{"draw_id": int(ids[i]), "year": int(years[i]),
                           "numbers": decode(images[i]), "bitmap": images[i].tolist(),
                           "geometry": {"centroid_row": float(latest_features["morphology"][i,0]*6),
                                        "centroid_col": float(latest_features["morphology"][i,1]*6),
                                        "neighbor_edges": int(round(latest_features["morphology"][i,6]*15)),
                                        "components": int(round(latest_features["morphology"][i,7]*6)),
                                        "dct_low_energy_fraction": float(latest_features["spectral"][i,8])}}
                          for i in range(max(0,len(images)-20),len(images))],
        "limitations": [
            "All real-draw analysis is exploratory; these observations are not blind forecasts.",
            "7x7 adjacency, FFT and DCT geometry are artifacts of an arbitrary numbering layout.",
            "Within-year permutations preserve yearly compositions but assume exchangeability within each year.",
            "Multiple lag and mapping inspections require caution even after within-mapping correction.",
            "A non-random descriptive feature does not establish out-of-sample predictability.",
        ],
    }


def _svg(bitmap: np.ndarray, *, size: int = 18) -> str:
    x = validate(bitmap)
    rect = []
    for i, cell in enumerate(x.ravel()):
        r, c = divmod(i, 7)
        rect.append(f'<rect x="{c*size+1}" y="{r*size+1}" width="{size-2}" height="{size-2}" rx="2" fill="{("#1b5b9f" if cell else "#e5ebf2")}"/>')
    return f'<svg role="img" aria-label="7x7 image" viewBox="0 0 {7*size} {7*size}">'+''.join(rect)+'</svg>'


def _heatmap(data: list[list[float]]) -> str:
    arr = np.asarray(data, dtype=float)
    maxv = float(max(0.3,arr.max()))
    cells = []
    for i,p in enumerate(arr.ravel()):
        r,c = divmod(i,7)
        alpha = 0.10 + 0.90 * float(p) / maxv
        cells.append(f'<rect x="{c*20}" y="{r*20}" width="19" height="19" rx="2" fill="#2367aa" fill-opacity="{alpha:.3f}"/>')
    return '<svg role="img" aria-label="7x7 occupancy frequency" viewBox="0 0 140 140">'+''.join(cells)+'</svg>'


def image_gallery(report: dict) -> str:
    """Offline visual image-research report; pure HTML/SVG, no CDN."""
    if report.get("schema") != SCHEMA or not report.get("mapping_comparisons"):
        raise ValueError("expected valid v0.5 image study report")
    rows=[]
    for mapping in report["mapping_comparisons"]:
        name = escape(mapping["mapping"] + " (seed " + str(mapping["mapping_seed"]) + ")")
        table=[]
        for lag, metrics in mapping["temporal"].items():
            for key in METRICS:
                val=metrics[key]
                table.append(f'<tr><td>{escape(lag)}</td><td>{escape(key)}</td>'
                             f'<td>{val["observed"]:.5f}</td><td>{val["null_mean"]:.5f}</td>'
                             f'<td>{val["bonferroni_p"]:.4f}</td></tr>')
        rows.append(f'<section><h2>{name}</h2><div class="heat">{_heatmap(mapping["pixel_occupancy"])}'
                    f'{_heatmap(mapping["iid_pixel_occupancy"])}</div>'
                    '<p>Pixel occupancy: actual (left) / IID control (right).</p>'
                    f'<div class="heat">{_heatmap(mapping["mean_dct_power"])}'
                    f'{_heatmap(mapping["iid_mean_dct_power"])}</div>'
                    '<p>Mean squared DCT coefficient magnitude: actual (left) / IID control (right).</p>'
                    '<table><tr><th>Lag</th><th>Metric</th><th>Observed</th><th>Null</th><th>Adjusted p</th></tr>'+
                    ''.join(table)+'</table></section>')
    rec=[]
    for item in report["latest_images"]:
        rec.append(f'<div class="sample"><h4>{item["year"]} / {item["draw_id"]}</h4>'
                   f'{_svg(np.asarray(item["bitmap"]))}<p>{escape(str(item["numbers"]))}</p>'
                   f'<p>8-connectivity: {item["geometry"]["components"]} components; '+
                   f'{item["geometry"]["neighbor_edges"]} adjacent pairs</p></div>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Lotto7x7 v0.5 Image Structure</title>'
            '<style>body{font:14px system-ui;background:#f4f7fb;color:#1b2c3f;'
            'max-width:1100px;margin:25px auto;padding:12px}section{background:white;'
            'padding:18px;border-radius:12px;margin:15px 0}table{border-collapse:collapse;'
            'width:100%}th,td{padding:7px;border-bottom:1px solid #e3ebf3;text-align:left}'
            '.heat{display:flex;gap:25px}svg{width:140px;max-width:100%}'
            '.samples{display:flex;gap:12px;flex-wrap:wrap}.sample{background:white;'
            'padding:12px;border-radius:8px}.sample svg{width:126px}'
            '@media(max-width:600px){body{margin:0}.heat{gap:8px}}'
            '</style><h1>Lotto7x7 v0.5 Image Structure &amp; Dynamics</h1>'
            '<p>Exploratory analysis only; image geometry is imposed by number layout, '
            'not physical lottery geometry. Corrected p-values do not establish a '
            'forecasting edge.</p>' + ''.join(rows) +
            '<section><h2>Latest historical six-pixel images</h2><div class="samples">'+
            ''.join(rec)+'</div></section></html>')
