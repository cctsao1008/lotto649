"""Phase 7 exploratory 7x7 geometry diagnostics with an IID control.

Geometry is a chosen representation; this tool tests descriptive statistics
and does not assert a predictive mechanism or select any future six-number set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lotto_historical import SOURCE, load_official_main_draws
from lotto_image_repr import encode_draws, cell_to_number

VERSION = "0.1.0"


def image_features(binary_image: np.ndarray) -> dict:
    """Analyze one valid six-pixel mask using purely deterministic geometry."""
    x = np.asarray(binary_image)
    if (x.shape != (7, 7) or not np.issubdtype(x.dtype, np.number)
            or not np.all(np.isfinite(x))
            or not np.all((x == 0) | (x == 1)) or int(x.sum()) != 6):
        raise ValueError("expected one binary 7x7 image with six active pixels")
    rows, cols = np.nonzero(x)
    adjacent_h = int((x[:, :-1] * x[:, 1:]).sum())
    adjacent_v = int((x[:-1, :] * x[1:, :]).sum())
    adjacent_diag = int((x[:-1, :-1] * x[1:, 1:]).sum() +
                        (x[:-1, 1:] * x[1:, :-1]).sum())
    return {
        "row_occupancy": x.sum(axis=1).astype(int).tolist(),
        "column_occupancy": x.sum(axis=0).astype(int).tolist(),
        "centroid_row": float(rows.mean()),
        "centroid_column": float(cols.mean()),
        "four_neighbor_pairs": adjacent_h + adjacent_v,
        "diagonal_neighbor_pairs": adjacent_diag,
        "horizontal_mirror_symmetric": bool(np.array_equal(x, x[:, ::-1])),
        "vertical_mirror_symmetric": bool(np.array_equal(x, x[::-1, :])),
    }


def analyze_draws(draws: np.ndarray, *, mapping: str = "sequential",
                  seed: int = 649, control_samples: int = 2000) -> dict:
    """Compare observed image structure to IID six-of-49 simulated masks."""
    if not isinstance(control_samples, int) or control_samples < 1:
        raise ValueError("control_samples must be positive")
    images = encode_draws(draws, mapping=mapping, seed=seed)
    if images.ndim != 3 or len(images) < 2:
        raise ValueError("analysis requires at least two observed draws")
    feats = [image_features(img) for img in images]
    overlap = np.sum(images[1:] * images[:-1], axis=(1, 2))
    generator = np.random.default_rng(seed + 7)
    fake = np.stack([np.sort(generator.choice(49, 6, replace=False) + 1)
                     for _ in range(control_samples)])
    controls = [image_features(img) for img in encode_draws(
        fake, mapping=mapping, seed=seed)]
    observed_4 = np.asarray([f["four_neighbor_pairs"] for f in feats])
    reference_4 = np.asarray([f["four_neighbor_pairs"] for f in controls])
    return {
        "schema": "lotto649.phase7.image_geometry.v1",
        "mapping": mapping,
        "mapping_seed": seed,
        "cell_to_number": cell_to_number(mapping, seed=seed).tolist(),
        "draws": len(images),
        "iid_control_draws": control_samples,
        "observed_mean_four_neighbor_pairs": float(observed_4.mean()),
        "iid_mean_four_neighbor_pairs": float(reference_4.mean()),
        "observed_mean_diagonal_pairs": float(np.mean([
            f["diagonal_neighbor_pairs"] for f in feats])),
        "iid_mean_diagonal_pairs": float(np.mean([
            f["diagonal_neighbor_pairs"] for f in controls])),
        "observed_mean_consecutive_overlap": float(overlap.mean()),
        "iid_theoretical_mean_consecutive_overlap": 36 / 49,
        "limitations": [
            "Descriptive results depend on the artificial pixel mapping.",
            "IID simulation is an illustrative control, not a significance test.",
            "Full spectral energy is constant for every six-pixel binary image.",
            "Any apparent anomaly requires independent prospective replication.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 read-only image geometry audit")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--mapping", choices=("sequential", "snake", "permuted"),
                        default="sequential")
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--control-samples", type=int, default=2000)
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase7_geometry.json"))
    args = parser.parse_args()
    draws, dates, periods = load_official_main_draws(args.source)
    result = analyze_draws(draws, mapping=args.mapping, seed=args.seed,
                           control_samples=args.control_samples)
    result["source"] = str(args.source)
    result["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    result["first_date"] = int(dates[0])
    result["last_date"] = int(dates[-1])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Lotto649 Phase 7 Geometry Audit v{VERSION}")
    print(f"Mapping={result['mapping']}, draws={result['draws']}")
    print(f"Observed adjacent-pair mean: {result['observed_mean_four_neighbor_pairs']:.6f}")
    print(f"IID-control adjacent-pair mean: {result['iid_mean_four_neighbor_pairs']:.6f}")
    print(f"Observed successive overlap mean: {result['observed_mean_consecutive_overlap']:.6f}")
    print(f"IID expected successive overlap: {36 / 49:.6f}")
    print(f"[DONE] {args.report.resolve()}")
    print("[CAUTION] These are descriptive artificial-grid features, not forecast skill")


if __name__ == "__main__":
    main()
