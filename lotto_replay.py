"""Historical one-step six-number replay for Lotto649 research.

Recreates Phase 5's chronological training and validation boundaries, freezes
parameters, and compares diagnostic top-six sets and full-subset log scores
with realized draws in an already-observed calendar year.

This is exploratory historical replay, not a prospective blind experiment,
and does not recommend lottery tickets. A top-six diagnostic is not evidence
of financial advantage, especially with 13,983,816 possible combinations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from lotto_historical import (CANDIDATES, SOURCE, choose_from_validation,
                              lag_log_probs, load_official_main_draws)
from lotto_neural import (IID_LOGP, K, N, SmallMLP, StaticLogitModel,
                          fit_model, make_observations)
from lotto_probability import FixedCardinalityDistribution
from lotto_temporal import fit_beta

VERSION = "0.1.0"
TOTAL_SUBSETS = math.comb(N, K)

# Phase 5 boundaries from the 2,183-draw frozen research protocol.
# These indices must not be shifted when more draw records are appended.
TRAIN_END = 1309
VALIDATION_END = 1746
DEFAULT_YEAR = 2025
DEFAULT_RANK_SAMPLES = 50000


def validate_replay_split(n: int, *, window: int, train_end: int,
                          validation_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed absolute indices; no observed test outcome defines splits."""
    if not (isinstance(window, int) and isinstance(train_end, int)
            and isinstance(validation_end, int)):
        raise ValueError("window and split boundaries must be integers")
    if not (1 <= window < train_end
            and train_end - window >= 10
            and validation_end - train_end >= 10
            and n - validation_end >= 10):
        raise ValueError("invalid fixed chronological replay boundaries")
    return (np.arange(window, train_end, dtype=np.int64),
            np.arange(train_end, validation_end, dtype=np.int64),
            np.arange(validation_end, n, dtype=np.int64))


def uniform_subset_pool(*, n: int = N, k: int = K,
                        samples: int, seed: int, chunk_size: int = 4096) -> np.ndarray:
    """Sample uniform unordered k-subsets using exchangeable random priorities."""
    if not (isinstance(n, int) and isinstance(k, int)
            and 1 <= k <= n
            and isinstance(samples, int) and samples >= 1
            and isinstance(chunk_size, int) and chunk_size >= 1):
        raise ValueError("invalid uniform subset pool parameters")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    result = np.empty((samples, k), dtype=np.int16)
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        priorities = rng.random((count, n))
        chosen = np.argpartition(priorities, kth=k - 1, axis=1)[:, :k]
        result[start:start + count] = np.sort(chosen, axis=1)
    return result


def top_subset(logits: np.ndarray, *, seed: int, draw_id: int,
               model_index: int) -> list[int]:
    """Select a retrospective MAP subset; independent seeded tie-breaking."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (N,) or not np.all(np.isfinite(values)):
        raise ValueError("expected 49 finite log weights")
    if seed < 0 or draw_id < 0 or model_index < 0:
        raise ValueError("invalid tie-breaking seed, period or model")
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, draw_id, model_index, 649]))
    # The secondary key is randomized BEFORE observing the target numbers.
    order = np.lexsort((rng.random(N), -values))
    return sorted((order[:K] + 1).astype(int).tolist())


def midrank_fraction_mc(logits: np.ndarray, realized: np.ndarray,
                        sampled_subsets: np.ndarray) -> tuple[float, float]:
    """Estimate upper-tail midrank fraction under uniform combinations.

    Lower values represent better model ranks. Ties receive half credit.
    The second output is a distribution-free per-draw 95% Hoeffding bound
    for Monte Carlo sampling error, not an inference about model accuracy.
    """
    w = np.asarray(logits, dtype=np.float64)
    actual = np.asarray(realized, dtype=np.int64)
    pool = np.asarray(sampled_subsets)
    if (w.shape != (N,) or not np.all(np.isfinite(w))
            or actual.shape != (K,) or len(set(actual.tolist())) != K
            or np.any((actual < 1) | (actual > N))
            or pool.ndim != 2 or pool.shape[1] != K or len(pool) == 0
            or not np.issubdtype(pool.dtype, np.integer)
            or np.any(pool < 0) or np.any(pool >= N)
            or np.any(np.diff(np.sort(pool, axis=1), axis=1) == 0)):
        raise ValueError("invalid ranking inputs")
    target_weight = float(np.sum(w[actual - 1]))
    sampled_weights = np.sum(w[pool], axis=1)
    more_probable = sampled_weights > target_weight + 1e-10
    tied = np.abs(sampled_weights - target_weight) <= 1e-10
    fraction = float(np.mean(more_probable + 0.5 * tied))
    bound = math.sqrt(math.log(40.0) / (2.0 * len(pool)))
    return fraction, min(1.0, bound)


def lag_midrank_exact(*, beta: float, overlap: int) -> float:
    """Exact uniform-combination midrank of a lag model by overlap class."""
    if not math.isfinite(beta) or not (isinstance(overlap, int) and 0 <= overlap <= K):
        raise ValueError("invalid lag beta or overlap")
    sizes = [math.comb(K, j) * math.comb(N - K, K - j)
             for j in range(K + 1)]
    if sum(sizes) != TOTAL_SUBSETS:
        raise AssertionError("lag overlap multiplicities do not normalize")
    if beta == 0.0:
        return 0.5
    greater = sum(sizes[j] for j in range(K + 1)
                  if (j > overlap if beta > 0 else j < overlap))
    return (greater + 0.5 * sizes[overlap]) / TOTAL_SUBSETS


def expected_iid_overlap_distribution() -> dict[str, float]:
    """Exact hypergeometric law for six independent chosen vs drawn numbers."""
    return {str(j): (math.comb(K, j) * math.comb(N-K, K-j)) / TOTAL_SUBSETS
            for j in range(K + 1)}


def prepare_frozen_models(numbers: np.ndarray, *,
                          train: np.ndarray, validation: np.ndarray,
                          history: torch.Tensor, targets: torch.Tensor,
                          window: int, hidden: int, seed: int,
                          epochs: int, patience: int,
                          batch_size: int, lr: float,
                          weight_decay: float) -> tuple[dict, float, dict]:
    """Fit the Phase 5 challengers without touching evaluation observations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    def features_at(indices: np.ndarray):
        positions = torch.from_numpy(indices - window)
        return history[positions], targets[positions]
    x_train, y_train = features_at(train)
    x_val, y_val = features_at(validation)
    models = {}
    traces = {}
    for name, model in (("static", StaticLogitModel()),
                        ("mlp", SmallMLP(window, hidden))):
        fitted, trace = fit_model(
            model, x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val, seed=seed, epochs=epochs,
            patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay)
        fitted.eval()
        models[name] = fitted
        traces[name] = trace
    beta = float(fit_beta(numbers, train_end=int(train[-1]) + 1))
    validation_scores = {
        "iid": float(IID_LOGP),
        "static": float(traces["static"]["best_val_log_score"]),
        "mlp": float(traces["mlp"]["best_val_log_score"]),
        "lag": float(lag_log_probs(numbers, validation, beta).mean()),
    }
    return models, beta, {
        "traces": traces,
        "validation_scores": validation_scores,
        "validation_selected": choose_from_validation(validation_scores),
    }


def run_replay(*, source: Path = SOURCE, year: int = DEFAULT_YEAR,
               seed: int = 649, window: int = 5, hidden: int = 16,
               epochs: int = 40, patience: int = 6,
               batch_size: int = 128, lr: float = 0.005,
               weight_decay: float = 0.02,
               train_end: int = TRAIN_END,
               validation_end: int = VALIDATION_END,
               rank_samples: int = DEFAULT_RANK_SAMPLES) -> dict:
    """Reproduce the frozen Phase 5 fit, then replay observed test-year draws."""
    if (not isinstance(year, int) or year < 1900 or year > 2200
            or not isinstance(seed, int) or seed < 0
            or not isinstance(rank_samples, int) or rank_samples < 1
            or hidden < 1):
        raise ValueError("invalid replay year, seed, hidden size or ranking samples")
    numbers, dates, periods = load_official_main_draws(source)
    train, validation, test = validate_replay_split(
        len(numbers), window=window,
        train_end=train_end, validation_end=validation_end)
    year_positions = test[dates[test] // 10000 == year]
    if not len(year_positions):
        raise ValueError("requested year has no observations in the fixed test period")
    if np.any(dates[train] >= dates[year_positions[0]]) or np.any(
            dates[validation] >= dates[year_positions[0]]):
        raise ValueError("training and validation must precede evaluation year")
    history, targets, origins = make_observations(numbers, window)
    if not np.array_equal(origins, np.arange(window, len(numbers))):
        raise AssertionError("causal history alignment changed")
    models, beta, training = prepare_frozen_models(
        numbers, train=train, validation=validation, history=history,
        targets=targets, window=window, hidden=hidden, seed=seed,
        epochs=epochs, patience=patience, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay)
    selected_x = history[torch.from_numpy(year_positions - window)]
    with torch.no_grad():
        model_logits = {
            name: model(selected_x).detach().cpu().numpy().astype(np.float64)
            for name, model in models.items()
        }
    pool = uniform_subset_pool(samples=rank_samples, seed=seed + 649)
    draw_rows = []
    overlap_hist = {name: [0] * (K + 1) for name in CANDIDATES}
    mean_hits = {name: [] for name in CANDIDATES}
    log_lifts = {name: [] for name in CANDIDATES}
    rank_fractions = {name: [] for name in CANDIDATES}
    exact_hits = {name: 0 for name in CANDIDATES}

    for local_idx, pos in enumerate(year_positions):
        realized = numbers[pos]
        previous = numbers[pos - 1]
        lag_logits = np.zeros(N, dtype=np.float64)
        lag_logits[previous - 1] = beta
        scores = {
            "iid": np.zeros(N, dtype=np.float64),
            "static": model_logits["static"][local_idx],
            "lag": lag_logits,
            "mlp": model_logits["mlp"][local_idx],
        }
        by_model = {}
        for model_index, name in enumerate(CANDIDATES):
            weights = scores[name]
            top = top_subset(weights, seed=seed, draw_id=int(periods[pos]),
                             model_index=model_index)
            hits = len(set(top).intersection(realized.tolist()))
            probability = (IID_LOGP if name == "iid"
                           else FixedCardinalityDistribution(weights, K).log_prob(realized))
            if name == "iid":
                rank_fraction, error_bound, rank_type = 0.5, 0.0, "exact_uniform_tie"
            elif name == "lag":
                intersection = int(np.isin(previous, realized).sum())
                rank_fraction = lag_midrank_exact(beta=beta, overlap=intersection)
                error_bound, rank_type = 0.0, "exact_overlap_class_midrank"
            else:
                rank_fraction, error_bound = midrank_fraction_mc(
                    weights, realized, pool)
                rank_type = "monte_carlo_uniform_subset_midrank"
            midrank = 1.0 + (TOTAL_SUBSETS - 1) * rank_fraction
            actual_score = float(probability)
            by_model[name] = {
                "diagnostic_top_six": top,
                "matched_main_numbers": hits,
                "exact_six_match": hits == K,
                "observed_set_log_probability": actual_score,
                "log_score_lift_vs_iid_nats": float(actual_score - IID_LOGP),
                "observed_set_midrank": float(midrank),
                "rank_fraction_lower_is_better": float(rank_fraction),
                "rank_fraction_mc_95pct_halfwidth": float(error_bound),
                "rank_method": rank_type,
            }
            mean_hits[name].append(hits)
            log_lifts[name].append(actual_score - IID_LOGP)
            rank_fractions[name].append(rank_fraction)
            overlap_hist[name][hits] += 1
            exact_hits[name] += int(hits == K)
        draw_rows.append({
            "date": int(dates[pos]),
            "draw_id": int(periods[pos]),
            "absolute_index": int(pos),
            "actual_main_six": realized.astype(int).tolist(),
            "previous_main_six": previous.astype(int).tolist(),
            "models": by_model,
        })

    summary = {
        name: {
            "draws": len(draw_rows),
            "mean_matched_main_numbers": float(np.mean(mean_hits[name])),
            "matched_main_number_histogram": {
                str(j): count for j, count in enumerate(overlap_hist[name])
            },
            "exact_six_matches": exact_hits[name],
            "mean_log_score_lift_vs_iid_nats": float(np.mean(log_lifts[name])),
            "mean_observed_set_rank_fraction": float(np.mean(rank_fractions[name])),
            "mean_observed_set_midrank": float(
                1.0 + (TOTAL_SUBSETS-1) * np.mean(rank_fractions[name])),
        } for name in CANDIDATES
    }
    return {
        "schema": "lotto649.neural.phase5.historical_top_six_replay.v1",
        "version": VERSION,
        "status": "EXPLORATORY_HISTORICAL_REPLAY_NOT_PROSPECTIVE_BLIND",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "year": year,
        "records": len(numbers),
        "evaluated_draws": len(draw_rows),
        "split_boundaries": {
            "train": [int(train[0]), int(train[-1])],
            "validation": [int(validation[0]), int(validation[-1])],
            "test_start": int(test[0]),
            "first_evaluated": {"date": int(dates[year_positions[0]]),
                                "draw_id": int(periods[year_positions[0]])},
            "last_evaluated": {"date": int(dates[year_positions[-1]]),
                               "draw_id": int(periods[year_positions[-1]])},
        },
        "config": {
            "seed": seed, "window": window, "hidden": hidden,
            "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "rank_samples": rank_samples,
            "rank_sampling_seed": seed + 649,
        },
        "training": training,
        "train_only_lag_beta": beta,
        "iid_theory": {
            "total_six_subsets": TOTAL_SUBSETS,
            "exact_six_match_probability_per_draw": 1.0 / TOTAL_SUBSETS,
            "expected_main_numbers_matched_per_draw": float(K*K/N),
            "expected_exact_six_matches_for_this_year": float(
                len(draw_rows) / TOTAL_SUBSETS),
            "matched_main_number_distribution": expected_iid_overlap_distribution(),
        },
        "summary_by_model": summary,
        "individual_draws": draw_rows,
        "limitations": [
            "Historical outcomes were studied before; this is not a pristine blind holdout.",
            "Training and validation boundaries are fixed to Phase 5 positions.",
            "Test targets are used only after their diagnostic forecast is formed.",
            "Earlier observed test draws are allowed as causal one-step context.",
            "The top six are retrospective diagnostic MAP representatives, not bets.",
            "IID and lag model midranks are exact; static and MLP ranks are Monte Carlo approximations.",
            "Rank Monte Carlo intervals describe sampling uncertainty, not predictability.",
            "Full six-number exact hits are extremely rare under fair IID; zero hits are expected.",
            "Do not optimize hyperparameters using these previously examined outcomes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrospective historical six-number replay; not a ticket tool")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--validation-end", type=int, default=VALIDATION_END)
    parser.add_argument("--rank-samples", type=int, default=DEFAULT_RANK_SAMPLES)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source not found: {args.source}")
    result = run_replay(
        source=args.source, year=args.year, seed=args.seed, window=args.window,
        hidden=args.hidden, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, train_end=args.train_end,
        validation_end=args.validation_end, rank_samples=args.rank_samples)
    output = args.report or Path("data/neural") / f"historical_replay_{args.year}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8")
    print(f"Lotto649 Historical Six-Number Replay v{VERSION}")
    print(f"Year: {result['year']}, draws: {result['evaluated_draws']}")
    print(f"Validation-selected model: {result['training']['validation_selected']}")
    print(f"IID expected matches/draw: {result['iid_theory']['expected_main_numbers_matched_per_draw']:.6f}")
    print("Model     Mean hits     Exact-six     Mean log lift     Mean observed rank fraction")
    for name in CANDIDATES:
        stats = result["summary_by_model"][name]
        print(f"{name:7s}  {stats['mean_matched_main_numbers']:9.4f} "
              f"{stats['exact_six_matches']:12d} "
              f"{stats['mean_log_score_lift_vs_iid_nats']:+16.6f} "
              f"{stats['mean_observed_set_rank_fraction']:26.4f}")
    print(f"[DONE] {output.resolve()}")
    print("[CAUTION] Historical top-six picks are retrospective diagnostics, not prospective tickets")


if __name__ == "__main__":
    main()
