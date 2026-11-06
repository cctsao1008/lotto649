"""Lotto649 correlation diagnostics v0.1.0.

Read-only input: data/normalized/lotto649_base.npy.
Same-draw phi has the theoretical baseline -1/48 under fair 6/49 drawing.
Lagged correlations are exploratory. A max-statistic permutation test controls
the search across all 49x49 cells and selected lags under exchangeability.
This is a diagnostic, not evidence of prediction or a betting recommendation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
SOURCE = Path("data/normalized/lotto649_base.npy")
OUT = Path("data/correlation")
LAGS = (1, 2, 3, 5, 10)
BASE_P = 6 / 49
SAME_DRAW_PHI = -1 / 48


def load_presence(path: Path):
    draws = np.load(path, allow_pickle=False)
    if draws.dtype.names is None or not {"date", "draw_id", "numbers"}.issubset(draws.dtype.names):
        raise ValueError("Expected structured NPY fields date/draw_id/numbers")
    numbers = np.asarray(draws["numbers"], dtype=np.int16)
    if numbers.ndim != 2 or numbers.shape[1] != 6 or len(numbers) <= max(LAGS):
        raise ValueError("Expected at least 11 draws of six numbers")
    if np.any((numbers < 1) | (numbers > 49)):
        raise ValueError("Number out of range 1..49")
    if np.any(np.diff(np.sort(numbers, axis=1), axis=1) == 0):
        raise ValueError("Duplicate numbers in a draw")
    dates = draws["date"].astype(np.int64)
    if np.any(np.diff(dates) <= 0) or len(set(draws["draw_id"].tolist())) != len(draws):
        raise ValueError("Draws must be ordered with unique dates and IDs")
    x = np.zeros((len(numbers), 49), dtype=np.int32)
    x[np.arange(len(numbers))[:, None], numbers - 1] = 1
    return x, int(dates[0]), int(dates[-1])


def phi_matrix(a: np.ndarray, b: np.ndarray):
    n = len(a)
    pa = a.sum(axis=0, dtype=np.float64) / n
    pb = b.sum(axis=0, dtype=np.float64) / n
    joint = (a.T @ b).astype(np.float64) / n
    denom = np.sqrt((pa * (1 - pa))[:, None] * (pb * (1 - pb))[None, :])
    phi = np.divide(joint - pa[:, None] * pb[None, :],
                    denom, out=np.zeros_like(joint), where=denom > 0)
    return phi, (a.T @ b)


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def same_draw(x):
    phi, co = phi_matrix(x, x)
    rows = []
    for i in range(49):
        for j in range(i + 1, 49):
            rows.append({
                "n1": i + 1, "n2": j + 1,
                "cooccurrences": int(co[i, j]),
                "phi": round(float(phi[i, j]), 7),
                "theoretical_phi": round(SAME_DRAW_PHI, 7),
                "excess_phi": round(float(phi[i, j] - SAME_DRAW_PHI), 7)
            })
    rows.sort(key=lambda r: -abs(r["excess_phi"]))
    return rows


def lag_data(x, lag):
    a, b = x[:-lag], x[lag:]
    phi, co = phi_matrix(a, b)
    return phi, co


def lag_rows(lag, phi, co):
    rows = []
    for i in range(49):
        for j in range(49):
            rows.append({
                "lag": lag, "from_number": i + 1, "to_number": j + 1,
                "cooccurrences": int(co[i, j]),
                "phi": round(float(phi[i, j]), 7)
            })
    rows.sort(key=lambda r: -abs(r["phi"]))
    return rows


def permutation_global_test(x, lags, observed_max, repeats, seed):
    """Max-|phi| across selected lags/cells; permute target rows per lag.

    This tests exchangeability of the target draw order conditional on its
    observed contents, not physical RNG fairness. Only exploratory evidence.
    """
    rng = np.random.default_rng(seed)
    extremes = np.empty(repeats, dtype=np.float64)
    for r in range(repeats):
        largest = 0.0
        for lag in lags:
            a, b = x[:-lag], x[lag:]
            shuffled = b[rng.permutation(len(b))]
            phi, _ = phi_matrix(a, shuffled)
            largest = max(largest, float(np.max(np.abs(phi))))
        extremes[r] = largest
    exceed = int(np.count_nonzero(extremes >= observed_max))
    return {
        "permutations": repeats, "seed": seed,
        "observed_max_abs_phi": round(observed_max, 8),
        "global_p_value": (exceed + 1) / (repeats + 1),
        "null_max_mean": float(np.mean(extremes)),
        "interpretation": "Exploratory global max-statistic; a small p-value needs independent out-of-sample verification."
    }


def main():
    parser = argparse.ArgumentParser(description="Lotto649 same-draw and lagged association diagnostics")
    parser.add_argument("--permutations", type=int, default=199, help=">= 19; increase to 999 for stronger resolution")
    parser.add_argument("--seed", type=int, default=649)
    args = parser.parse_args()
    if args.permutations < 19:
        parser.error("--permutations must be >= 19")

    print(f"Lotto649 Correlation Analyzer v{VERSION}")
    x, start, end = load_presence(SOURCE)
    print(f"[PASS] base draws: {len(x):,}; range: {start}–{end}")

    OUT.mkdir(parents=True, exist_ok=True)
    same = same_draw(x)
    save_csv(OUT / "same_draw_pairs.csv", same)
    top = {}
    observed_max = 0.0
    winner = None
    for lag in LAGS:
        phi, co = lag_data(x, lag)
        rows = lag_rows(lag, phi, co)
        save_csv(OUT / f"lag_{lag}.csv", rows)
        top[str(lag)] = rows[:10]
        if abs(rows[0]["phi"]) > observed_max:
            observed_max = abs(rows[0]["phi"])
            winner = rows[0]
        print(f"[PASS] Lag {lag}: {len(rows)} ordered pairs")
    print(f"[INFO] Permutation max-statistic: {args.permutations} repeats")
    perm = permutation_global_test(x, LAGS, observed_max, args.permutations, args.seed)
    report = {
        "version": VERSION, "draws": len(x),
        "start_date": start, "end_date": end, "lags": list(LAGS),
        "base_probability": BASE_P,
        "same_draw_theoretical_phi_off_diagonal": SAME_DRAW_PHI,
        "top_same_draw_excess_phi": same[:10],
        "top_lagged_abs_phi": top,
        "strongest_lagged_cell": winner,
        "multiple_testing": perm,
        "cautions": [
            "Same-draw off-diagonal phi is not zero under 6/49 without replacement.",
            "Lagged association is not demonstrated forecasting skill.",
            "Bonus combinations are not independent draws and are excluded.",
            "A high historical score from searching many cells is subject to selection bias."
        ]
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"Lotto649 Correlation Analyzer v{VERSION}",
        f"Draws: {len(x):,} | {start} - {end}",
        f"Theoretical same-draw off-diagonal phi: {SAME_DRAW_PHI:.6f}",
        "", "Top 5 same-draw excess associations:"
    ]
    for row in same[:5]:
        lines.append(f"  {row['n1']:02d}+{row['n2']:02d}: co={row['cooccurrences']}, excess_phi={row['excess_phi']:+.4f}")
    lines += ["", "Top lagged association per lag:"]
    for lag in LAGS:
        item = top[str(lag)][0]
        lines.append(f"  lag {lag:2d}: {item['from_number']:02d}->{item['to_number']:02d} phi={item['phi']:+.4f}")
    lines += [
        "", f"Global max-|phi| permutation p: {perm['global_p_value']:.6f} ({args.permutations} repeats)",
        "CAUTION: This is an exploratory dependence test, not a prediction model.",
        "CAUTION: An apparent association does not increase any fixed ticket's fair-draw jackpot probability."
    ]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[DONE]", OUT.resolve())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
