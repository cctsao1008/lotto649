"""v0.11 descriptive 7x7 lottery image classification and counting.

Three deliberately distinct notions of a class:
  1. exact D4 + non-wrapping translation shape equivalence;
  2. invariant, interpretable 8-neighbor morphology;
  3. CUDA-assisted prototype clustering with an explicit fixed number of classes.
The 5-of-6 overlap relation is a *network*, not a transitive shape class.
No temporal forecasting or winning advantage is inferred.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from math import comb, sqrt
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

SCHEMA = "lotto7x7.image_classification.v0.11"
BLOCKS = ((0, 15), (15, 17), (17, 20), (20, 34), (34, 40))
BLOCK_NAMES = ("pairwise_distances", "bounding_box", "connectivity", "axis_occupancy", "component_sizes")
UNIVERSE = comb(49, 6)


def validate(images: np.ndarray, draw_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(images)
    ids = np.asarray(draw_ids)
    if (x.ndim != 3 or x.shape[1:] != (7, 7) or len(x) < 2
            or not np.issubdtype(x.dtype, np.number)
            or not np.all(np.isfinite(x)) or not np.all((x == 0) | (x == 1))
            or not np.all(x.sum(axis=(1, 2)) == 6)
            or ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("need chronological [N,7,7] legal six-hot images and draw IDs")
    return np.asarray(x, dtype=np.uint8), np.asarray(ids, dtype=np.int64)


def numbers(image: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.asarray(image).reshape(-1)) + 1).astype(int).tolist()


def canonical(points: np.ndarray) -> str:
    """Canonical D4+translation key; all transformed positions remain on 7x7."""
    arr = np.asarray(points, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) not in (5, 6):
        raise ValueError("canonical shape requires five or six points")
    if len({tuple(p) for p in arr}) != len(arr):
        raise ValueError("duplicate points")
    best = None
    for reflected in (False, True):
        p = arr.copy()
        if reflected:
            p[:, 1] = -p[:, 1]
        for rot in range(4):
            q = p.copy()
            for _ in range(rot):
                q = np.column_stack((-q[:, 1], q[:, 0]))
            q -= q.min(axis=0)
            q = q[np.lexsort((q[:, 1], q[:, 0]))]
            key = ";".join(f"{r},{c}" for r, c in q)
            if best is None or key < best:
                best = key
    return best


def orbit_size(key: str) -> int:
    pts = np.asarray([list(map(int, pair.split(","))) for pair in key.split(";")], dtype=np.int64)
    if len(pts) != 6 or canonical(pts) != key:
        raise ValueError("need canonical six-point shape key")
    seen = set()
    total = 0
    for reflected in (False, True):
        p = pts.copy()
        if reflected:
            p[:, 1] = -p[:, 1]
        for rot in range(4):
            q = p.copy()
            for _ in range(rot):
                q = np.column_stack((-q[:, 1], q[:, 0]))
            q -= q.min(axis=0)
            form = tuple(sorted(map(tuple, q.tolist())))
            if form in seen:
                continue
            seen.add(form)
            height, width = q.max(axis=0) + 1
            total += (8 - int(height)) * (8 - int(width))
    return total


def morphology(images: np.ndarray):
    """Return 40 rotation/reflection/translation invariant descriptors and counts."""
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (7, 7) or not np.all(arr.sum(axis=(1, 2)) == 6):
        raise ValueError("need [N,7,7] six-hot images")
    features = np.empty((len(arr), 40), dtype=np.float32)
    descriptors = []
    for t, image in enumerate(arr):
        pts = np.argwhere(image).astype(int)
        dr = np.abs(pts[:, None, 0] - pts[None, :, 0])
        dc = np.abs(pts[:, None, 1] - pts[None, :, 1])
        ii, jj = np.triu_indices(6, k=1)
        d2 = (dr[ii, jj] ** 2 + dc[ii, jj] ** 2).astype(float)
        adj8 = (np.maximum(dr, dc) == 1)
        adj4 = (dr + dc == 1)
        edges8, edges4 = int(np.sum(adj8[ii, jj])), int(np.sum(adj4[ii, jj]))
        remaining, component_sizes = set(range(6)), []
        while remaining:
            seed = remaining.pop()
            queue, size = [seed], 1
            while queue:
                u = queue.pop()
                for v in list(remaining):
                    if adj8[u, v]:
                        remaining.remove(v)
                        queue.append(v)
                        size += 1
            component_sizes.append(size)
        component_sizes.sort(reverse=True)
        c8 = len(component_sizes)
        h, w = (pts.max(axis=0) - pts.min(axis=0) + 1).tolist()
        occupancy = [np.sort(np.bincount(pts[:, axis], minlength=7)) for axis in (0, 1)]
        occupancy.sort(key=lambda a: tuple(a))
        vec = np.concatenate((
            np.sort(d2) / 72.,
            np.array(sorted((h, w)), dtype=float) / 7.,
            [edges8 / 15., edges4 / 15., c8 / 6.],
            np.concatenate(occupancy).astype(float) / 6.,
            np.array(component_sizes + [0] * (6 - c8), dtype=float) / 6.,
        ))
        assert vec.shape == (40,)
        features[t] = vec
        descriptors.append({"components_8": c8, "adjacent_pairs_8": edges8,
                            "adjacent_pairs_4": edges4, "bounding_box": sorted((h, w)),
                            "largest_component": max(component_sizes)})
    return features, descriptors


def classify_exact(images, ids):
    shapes = {}
    exact = Counter()
    for image, draw_id in zip(images, ids):
        pts = np.argwhere(image)
        key = canonical(pts)
        row = shapes.setdefault(key, {"shape_key": key, "count": 0,
                                     "first_draw_id": int(draw_id), "last_draw_id": int(draw_id),
                                     "example_numbers": numbers(image)})
        row["count"] += 1
        row["last_draw_id"] = int(draw_id)
        exact[tuple(numbers(image))] += 1
    repeats = [dict(v, orbit_size=orbit_size(k), fair_expected=len(images) * orbit_size(k) / UNIVERSE)
               for k, v in shapes.items() if v["count"] > 1]
    repeats.sort(key=lambda r: (-r["count"], r["shape_key"]))
    histogram = Counter(r["count"] for r in shapes.values())
    return {"exact_ticket_unique": len(exact),
            "exact_ticket_repeated": sum(v > 1 for v in exact.values()),
            "shape_class_count": len(shapes),
            "size_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "repeated_shape_classes": repeats}


def similarity_five_graph(images):
    """Compute pair links via shared canonical 5-point subshapes, not O(N^2) pixels."""
    n = len(images)
    index = defaultdict(list)
    for t, image in enumerate(images):
        pts = np.argwhere(image)
        for key in {canonical(np.delete(pts, k, axis=0)) for k in range(6)}:
            index[key].append(t)
    parent = list(range(n))
    sizes = [1] * n
    degree = [0] * n
    pairs = set()

    def find(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for bucket in index.values():
        for i, a in enumerate(bucket):
            for b in bucket[i+1:]:
                x, y = min(a, b), max(a, b)
                pid = x * n + y
                if pid in pairs:
                    continue
                pairs.add(pid)
                degree[x] += 1
                degree[y] += 1
                rx, ry = find(x), find(y)
                if rx != ry:
                    if sizes[rx] < sizes[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    sizes[rx] += sizes[ry]
    group_counts = Counter(find(i) for i in range(n))
    histogram = Counter(group_counts.values())
    return {"similarity_definition": "at_least_5_of_6_after_D4_and_nonwrapping_translation",
            "pair_count": len(pairs), "images_with_similar_neighbor": sum(d > 0 for d in degree),
            "linked_components": len(group_counts), "largest_linked_component": max(group_counts.values()),
            "component_size_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "warning": "Connected-component chains are NOT pairwise mutually similar; do not treat network as prototype classes."}


def _normalize_features(raw: np.ndarray, reference: np.ndarray | None = None):
    if reference is None:
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
    else:
        mean, std = reference
    x = (raw - mean) / np.maximum(std, 1.e-5)
    for a, b in BLOCKS:
        x[:, a:b] /= sqrt(b-a)
    return x.astype(np.float32), (mean, std)


def _embeddings(images, checkpoint: Path, *, device: torch.device, batch_size: int):
    from .deep_feature_v010 import DeepFeatureNet
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema") != "lotto7x7.deep_feature_checkpoint.v0.10":
        raise ValueError("v0.10 checkpoint required")
    emb = int(saved["architecture"]["embedding"])
    hid = int(saved["architecture"]["hidden"])
    net = DeepFeatureNet(emb, hid).to(device)
    net.load_state_dict(saved["state_dict"], strict=True)
    net.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(images), batch_size):
            x = torch.as_tensor(images[k:k+batch_size], device=device, dtype=torch.float32).unsqueeze(1)
            outs.append(net.encode(x)[:, 0].cpu().numpy())
    return np.concatenate(outs, axis=0)


def kmeans_gpu(x: np.ndarray, clusters: int, *, device: str, seed: int, iterations: int = 100):
    if not 2 <= clusters <= len(x):
        raise ValueError("cluster count out of range")
    if device not in ("auto", "cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("invalid/unavailable device")
    dev = torch.device("cuda" if device == "cuda" or device == "auto" and torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(x, dtype=torch.float32, device=dev)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("nonfinite features")
    rng = np.random.default_rng(seed)
    indices = [int(rng.integers(0, len(x)))]
    distances = torch.cdist(tensor, tensor[indices]).square().min(dim=1).values
    for _ in range(1, clusters):
        weights = distances.cpu().numpy().astype(float)
        weights[indices] = 0
        if weights.sum() <= 1.e-12:
            possibilities = list(set(range(len(x))) - set(indices))
            index = possibilities[int(rng.integers(len(possibilities)))]
        else:
            index = int(rng.choice(len(x), p=weights / weights.sum()))
        indices.append(index)
        distances = torch.minimum(distances, torch.cdist(tensor, tensor[index:index+1]).square().reshape(-1))
    centroids = tensor[indices].clone()
    previous = None
    for _ in range(iterations):
        d2 = torch.cdist(tensor, centroids).square()
        labels = d2.argmin(dim=1)
        if previous is not None and torch.equal(labels, previous):
            break
        previous = labels.clone()
        new = centroids.clone()
        for k in range(clusters):
            members = tensor[labels == k]
            if len(members):
                new[k] = members.mean(dim=0)
            else:
                farthest = d2.min(dim=1).values.argmax()
                new[k] = tensor[farthest]
        centroids = new
    d2 = torch.cdist(tensor, centroids).square()
    labels = d2.argmin(dim=1)
    # Stable IDs: descending class size, then centroid's representative index.
    sizes = [(k, int((labels == k).sum())) for k in range(clusters)]
    ordered = [k for k, _ in sorted(sizes, key=lambda v: (-v[1], v[0]))]
    reorder = torch.empty(clusters, dtype=torch.long, device=dev)
    for i, k in enumerate(ordered):
        reorder[k] = i
    return reorder[labels].cpu().numpy(), centroids[ordered].cpu().numpy(), str(dev)


def iid_legal_images(count: int, rng: np.random.Generator) -> np.ndarray:
    """Each row is an independent legal 6/49 draw; never sample duplicate pixels."""
    values = rng.random((count, 49), dtype=np.float32)
    indices = np.argpartition(values, 6, axis=1)[:, :6]
    result = np.zeros((count, 49), dtype=np.uint8)
    result[np.arange(count)[:, None], indices] = 1
    return result.reshape(-1, 7, 7)


def report(images, draw_ids, *, clusters=12, feature_mode="geometry",
           checkpoint: Path | None = None, device="auto", seed=649,
           iid_samples=10000, batch_size=2048):
    x, ids = validate(images, draw_ids)
    if (feature_mode not in ("geometry", "cnn", "fused")
            or feature_mode != "geometry" and checkpoint is None
            or clusters < 2 or clusters > len(x) or iid_samples < 100
            or batch_size < 1 or seed < 0):
        raise ValueError("invalid image classification configuration")
    geometry, info = morphology(x)
    geometry_z, reference = _normalize_features(geometry)
    dev = torch.device("cuda" if device == "cuda" or device == "auto" and torch.cuda.is_available() else "cpu")
    deep_z, deep_ref = None, None
    if feature_mode != "geometry":
        deep_raw = _embeddings(x, Path(checkpoint), device=dev, batch_size=batch_size)
        deep_mean, deep_std = deep_raw.mean(axis=0), np.maximum(deep_raw.std(axis=0), 1.e-5)
        deep_ref = (deep_mean, deep_std)
        deep_z = ((deep_raw - deep_mean) / deep_std) / sqrt(deep_raw.shape[1])
    features = geometry_z if feature_mode == "geometry" else deep_z if feature_mode == "cnn" else np.column_stack((geometry_z, deep_z))
    labels, centers, used_device = kmeans_gpu(features, clusters, device=device, seed=seed)
    rng = np.random.default_rng(seed + 999)
    reference_counts = np.zeros(clusters, dtype=np.int64)
    # Freeze centroids and normalization for fair IID placement controls.
    # This is DESCRIPTIVE only because clusters were fitted using real draws.
    centroids = torch.as_tensor(centers, device=dev, dtype=torch.float32)
    for start in range(0, iid_samples, batch_size):
        synth = iid_legal_images(min(batch_size, iid_samples-start), rng)
        synthetic_geometry, _ = morphology(synth)
        synth_z, _ = _normalize_features(synthetic_geometry, reference)
        if feature_mode != "geometry":
            raw = _embeddings(synth, Path(checkpoint), device=dev, batch_size=batch_size)
            syn_deep = ((raw - deep_ref[0]) / deep_ref[1]) / sqrt(raw.shape[1])
            synth_z = syn_deep if feature_mode == "cnn" else np.column_stack((synth_z, syn_deep))
        with torch.no_grad():
            z = torch.as_tensor(synth_z, dtype=torch.float32, device=dev)
            members = torch.cdist(z, centroids).argmin(dim=1).cpu().numpy()
        reference_counts += np.bincount(members, minlength=clusters)
    classes = []
    for k in range(clusters):
        items = np.flatnonzero(labels == k)
        d2 = np.square(features[items] - centers[k]).sum(axis=1)
        representative = int(items[np.argmin(d2)])
        count = len(items)
        expected = float(reference_counts[k]) / iid_samples
        classes.append({"class_id": f"C{k+1:02d}", "count": count,
                        "historical_share": count/len(x), "iid_share_estimate": expected,
                        "observed_to_iid_share_ratio": (count/len(x))/expected if expected else None,
                        "representative_draw_id": int(ids[representative]),
                        "representative_numbers": numbers(x[representative]),
                        "representative_bitmap": x[representative].tolist(),
                        "example_draw_ids": ids[items[:8]].astype(int).tolist(),
                        "mean_distance_to_centroid": float(np.sqrt(d2).mean()),
                        "mean_connected_components8": float(np.mean([info[i]["components_8"] for i in items]))})
    exact = classify_exact(x, ids)
    network = similarity_five_graph(x)
    cc = Counter(d["components_8"] for d in info)
    adjacency = Counter(d["adjacent_pairs_8"] for d in info)
    returns = {"schema": SCHEMA, "research_only": True, "draw_count":len(x),
            "first_draw_id":int(ids[0]), "last_draw_id":int(ids[-1]),
            "source_sha256": hashlib.sha256(x.tobytes()+ids.tobytes()).hexdigest(),
            "cluster_method":"kmeans_plus_plus_on_D4_invariant_geometry_or_frozen_v010_encoder",
            "feature_mode":feature_mode, "device":used_device, "seed":seed,
            "checkpoint_sha256": (hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                                  if checkpoint is not None and feature_mode != "geometry" else None),
            "number_of_classes":clusters, "iid_reference_samples":iid_samples,
            "exact_shape_classes":exact,
            "eight_neighbor_morphology":{
                "components_histogram":{str(k):v for k,v in sorted(cc.items())},
                "adjacent_pair_histogram":{str(k):v for k,v in sorted(adjacency.items())}},
            "five_of_six_similarity_network":network,
            "cluster_summary": classes,
            "assignments":[{"draw_id":int(ids[i]),"class_id":f"C{int(labels[i])+1:02d}",
                            "numbers":numbers(x[i]),"components_8":info[i]["components_8"]}
                           for i in range(len(x))],
            "warnings":[
                "Exploratory descriptive clustering. Class labels are not forecasts or winning probabilities.",
                "The 7x7 raster mapping is arbitrary; spatial geometry is not a physical lottery mechanism.",
                "K-means assignments and classes change with class count, feature mode and seed.",
                "IID reference uses fixed observed-trained centroids and is descriptive, NOT a selection-adjusted p-value.",
                "Five-of-six similarity graph chains do not constitute mutually similar classes.",
                "The v0.10 CNN encoder was trained on a historical prefix; its features do not demonstrate predictive utility.",
            ]}
    return returns


def html_gallery(item:dict) -> str:
    def bitmap(row):
        squares=[]
        for r in range(7):
            for c in range(7):
                active=row[r][c]==1
                squares.append(f'<rect x="{c*21+2}" y="{r*21+2}" width="19" height="19" rx="2" fill="{("#137e78" if active else "#e5e7eb")}"/>')
        return '<svg viewBox="0 0 149 149" aria-label="7x7 six-point image">'+''.join(squares)+'</svg>'
    cells=[]
    for cl in item["cluster_summary"]:
        cells.append(f'<article><h2>{escape(cl["class_id"])} · {cl["count"]} draws</h2>'
                     f'<p>History: {cl["historical_share"]:.2%} · Fair-IID: {cl["iid_share_estimate"]:.2%}</p>'
                     f'{bitmap(cl["representative_bitmap"])}'
                     f'<p>Example: {escape(str(cl["representative_numbers"]))}</p>'
                     f'<p>Mean 8-neighbor components: {cl["mean_connected_components8"]:.2f}</p></article>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Lotto649 7×7 Shape Classification</title>'
            '<style>body{font-family:system-ui;margin:2rem;background:#f7f9fb;color:#26323a}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}'
            'article{background:#fff;border:1px solid #dee3e7;border-radius:12px;padding:1rem}'
            'svg{width:180px;max-width:100%}h2{font-size:1.08rem}p{font-size:.88rem}</style>'
            '<h1>7×7 Image Classification &amp; Quantification</h1>'
            f'<p>{item["draw_count"]} actual drawings · {item["number_of_classes"]} prototype classes · '
            f'features: {escape(item["feature_mode"])} · device: {escape(item["device"])}</p>'
            '<p>Descriptive shape classes, not predicted winning probabilities.</p>'
            '<div class="grid">'+''.join(cells)+'</div></html>')
