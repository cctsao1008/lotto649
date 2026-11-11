"""v0.30 P1 — exactly uniform Sum-conditional six-of-49 sampler.

Construct fair random six-number tickets conditioned on their EXACT sum
using combinatorial completion counts. Further filter by the frozen v0.28
L3 graph class and exact-integer Log Product bin: rejection from a uniform
Sum-conditional proposal is unbiased in that fully matched stratum.

This is a structural distribution study, NOT next-draw conditional
forecasting, as the ticket's own Sum/L3/LogBin are unknown before drawing.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .geometry_convergence_v029 import _check_config, audit_repository_inputs
from .geometry_redundancy_v029 import _certificate, _descriptors
from .geometry_structure_v029 import PAIR_FEATURES
from .geometry_v029 import catalog
from .geometry_repair_v029 import P0_SHA256

SCHEMA = "lotto649.v030.exact_sum_conditional_mc.v1"
DEFAULT_PROPOSALS = 40000
DEFAULT_SEED = 649
MAX_PROPOSALS = 250000


@lru_cache(maxsize=None)
def completion_count(lo: int, k: int, target: int, universe: int = 49) -> int:
    """Exact count of increasing k-subsets in [lo..universe] with given Sum.

    Correctness invariant: f(lo,k,s)=f(lo+1,k,s)+f(lo+1,k-1,s-lo),
    with tight min/max-sum pruning and f(lo,0,0)=1.
    This is an integer count, NOT a Gaussian approximation.
    """
    if not all(type(x) is int for x in (lo, k, target, universe)):
        raise ValueError("completion arguments must be integers")
    if not (1 <= lo <= universe + 1 and 0 <= k <= universe
            and 1 <= universe <= 49):
        raise ValueError("invalid finite-population completion parameters")
    if k == 0:
        return int(target == 0)
    if universe - lo + 1 < k:
        return 0
    minimum = k * lo + k * (k - 1) // 2
    maximum = k * universe - k * (k - 1) // 2
    if target < minimum or target > maximum:
        return 0
    return (
        completion_count(lo + 1, k, target, universe)
        + completion_count(lo + 1, k - 1, target - lo, universe)
    )


def sample_uniform_fixed_sum(
    target_sum: int, count: int, *,
    rng: np.random.Generator, universe: int = 49, pick: int = 6,
) -> np.ndarray:
    """Draw iid uniform legal subsets from {C:sum(C)==target_sum} exactly.

    At each step, candidate next ball j has weight equal to the exact
    number of completions after j. This yields 1/F(1,k,sum) for every
    valid increasing combination without rejection or approximated PMF.
    """
    if type(target_sum) is not int or type(count) is not int or type(universe) is not int \
            or type(pick) is not int or not (1 <= count <= MAX_PROPOSALS) \
            or not (2 <= pick < universe <= 49):
        raise ValueError("invalid fixed-Sum sampling configuration")
    total = completion_count(1, pick, target_sum, universe)
    if total < 1:
        raise ValueError("requested exact Sum has no legal combinations")
    rows = np.empty((count, pick), dtype=np.int16)
    for row in rows:
        lo, left, remainder = 1, pick, target_sum
        for col in range(pick):
            total_left = completion_count(lo, left, remainder, universe)
            choice = int(rng.integers(total_left))
            for ball in range(lo, universe - left + 2):
                ways = completion_count(ball + 1, left - 1,
                                        remainder - ball, universe)
                if choice < ways:
                    row[col] = ball
                    lo, left, remainder = ball + 1, left - 1, remainder - ball
                    break
                choice -= ways
            else:
                raise AssertionError("exact uniform conditional sampler lost a completion")
        if remainder != 0:
            raise AssertionError("selected subset violates fixed Sum")
    return rows


def conditional_ablation(
    historical: object, *, proposals: int = DEFAULT_PROPOSALS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Analyze a pre-existing certificate stratum with independent fair draws.

    The certificate stratum was discovered during earlier research; any
    empirical tail rank for its two tickets is a descriptive *post-selection*
    rank, not an exact test of unexpected past performance.
    """
    if type(proposals) is not int or not (1000 <= proposals <= MAX_PROPOSALS):
        raise ValueError("proposals must be an integer in 1000..250000")
    _check_config(1, seed)
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("P0 geometry definition has changed")
    # Validate historical source, but do NOT fit conditional probabilities to it.
    from .geometry_convergence_v029 import _validate_draws
    history = _validate_draws(historical)
    certificate = _certificate()
    target = certificate["same_v028_baseline"]
    target_l3, target_sum, target_log = (target["l3"], target["sum"], target["log_bin"])
    total = completion_count(1, 6, target_sum, 49)
    if total <= 0:
        raise AssertionError("fixed certificate Sum has no legal combinations")
    rng = np.random.default_rng(seed)
    proposed = sample_uniform_fixed_sum(target_sum, proposals, rng=rng)
    keys, geometry, draws = _descriptors(proposed)
    if not np.all(keys[:, 1] == target_sum):
        raise AssertionError("conditional proposal violated fixed sum")
    l3_ok = keys[:, 0] == target_l3
    log_ok = keys[:, 2] == target_log
    accepted_mask = l3_ok & log_ok
    accepted = geometry[accepted_mask]
    chosen = draws[accepted_mask]
    min_accepted = 25
    if len(accepted) < min_accepted:
        raise ValueError(
            f"conditional sample insufficient ({len(accepted)} < {min_accepted}); "
            "increase --proposals without modifying matching conditions"
        )
    # The results below describe the probability distribution given *known*
    # Sum, LogBin and L3. The next random ticket has unknown classification.
    metrics = {}
    for i, name in enumerate(PAIR_FEATURES):
        values = accepted[:, i].astype(float)
        metrics[name] = {
            "conditional_mean_fair_mc": float(np.mean(values)),
            "conditional_variance_fair_mc": float(np.var(values, ddof=1)),
            "conditional_min": int(np.min(values)),
            "conditional_max": int(np.max(values)),
            "conditional_quantiles_05_25_50_75_95": np.quantile(
                values, [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
        }
    ranks = []
    for row in certificate["tickets"]:
        details = {}
        for name in PAIR_FEATURES:
            i = PAIR_FEATURES.index(name)
            observed = row["pair_geometry"][name]
            values = accepted[:, i]
            details[name] = {
                "observed": observed,
                "descriptive_fraction_conditional_le": float(np.mean(values <= observed)),
                "post_selected_not_a_hypothesis_p": True,
            }
        ranks.append({"ticket": row["numbers"], "conditional_feature_ranks": details})
    return {
        "schema": SCHEMA,
        "decision": "UNBIASED_CONDITIONAL_STRUCTURE_DESCRIPTIVE_NOT_PREDICTIVE",
        "history_draws_read_only": len(history),
        "fixed_baseline": {
            "l3": target_l3, "sum": target_sum, "log_bin": target_log,
            "v028_baseline_interpretation": "exact L3 + exact Sum + integer Log Product bin; Joint derived",
        },
        "sampler": {
            "seed": seed, "total_legal_combinations_with_exact_sum": total,
            "exact_sum_proposals": proposals,
            "accepted_full_condition": int(len(accepted)),
            "accepted_distinct_combinations": int(len({
                tuple(map(int, row)) for row in chosen
            })),
            "l3_only_accept_count": int(np.count_nonzero(l3_ok)),
            "logbin_only_accept_count": int(np.count_nonzero(log_ok)),
            "full_condition_accept_fraction": float(len(accepted) / proposals),
            "sampling_proof": (
                "Each ticket with exact Sum receives probability 1/F(1,6,Sum), "
                "and filtering by original L3+LogBin therefore produces iid uniform "
                "samples within the full finite stratum (duplicates are valid)."
            ),
            "confidence_note": (
                "Monte Carlo quantiles are descriptive approximations with sampling "
                "error. This is not exact enumeration of the L3/Sum/LogBin stratum."
            ),
        },
        "conditional_fair_structural_statistics": metrics,
        "prior_exploratory_certificate_rank": ranks,
        "limits": [
            "The next draw's Sum/L3/LogBin are unknown: posterior-looking conditioning "
            "on the outcome's own features is not next-draw prediction.",
            "This reference stratum and certificate were investigated before P1; "
            "do not assign their descriptive conditional ranks confirmatory p-values.",
            "Independence of output draws holds conditional on the exact baseline "
            "event by rejection sampling from an exact uniform Sum-conditional proposal.",
            "No interpretation of predictive probability, forced compensation, "
            "new ticket selection or real wager benefit is supported.",
            "Historical data are only validated and counted; all reference probability "
            "statistics come from independently simulated legal combinations.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--proposals", type=int, default=DEFAULT_PROPOSALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    path = args.out.resolve()
    if not path.is_relative_to(Path("results").resolve()):
        raise ValueError("v0.30 P1 report must be created under ignored results/")
    historical, gate = audit_repository_inputs(args.csv_root, args.bin_root)
    report = conditional_ablation(
        historical, proposals=args.proposals, seed=args.seed
    )
    report["source_gate"] = gate
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        f.write("\n")
    print(f"PASS P1: {len(historical)} audited historical draws, "
          f"{report['sampler']['exact_sum_proposals']} exact-Sum proposals")
    print(f"CONDITIONAL: {report['sampler']['accepted_full_condition']} accepted "
          f"for exact Sum={report['fixed_baseline']['sum']}, "
          f"L3={report['fixed_baseline']['l3']}, "
          f"LogBin={report['fixed_baseline']['log_bin']}")
    print("DECISION: UNBIASED_CONDITIONAL_STRUCTURE_DESCRIPTIVE_NOT_PREDICTIVE")
    print(f"REPORT: {path}")


if __name__ == "__main__":
    main()
