"""v0.7.1 paired inference, validation, multipliers and CLI tests."""
from __future__ import annotations

import argparse
import copy
import json
from math import comb
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lotto7x7.baseline_v07 import ARMS, SCHEMA as BASELINE_SCHEMA, STRATEGIES
from lotto7x7.coverage_v06 import exact_coverage
from lotto7x7.robustness_v071 import (
    AUDIT_SCHEMA, SWEEP_SCHEMA, audit_rolling, holm, mcnemar_exact,
    sweep_rolling, _positive_tuple,
)
from lotto7x7.robustness_cli_v071 import configure_parsers, run

ACTUAL = (1, 2, 3, 4, 5, 6)
# Exactly three matching numbers in SUCCESS; no matches in other tickets.
SUCCESS = (1, 2, 3, 10, 11, 12)
FAILURE = (
    (7, 8, 9, 10, 11, 12),
    (13, 14, 15, 16, 17, 18),
    (19, 20, 21, 22, 23, 24),
    (25, 26, 27, 28, 29, 30),
    (31, 32, 33, 34, 35, 36),
)


def portfolio(n, success=False):
    bets = [SUCCESS] + list(FAILURE[:n-1]) if success else list(FAILURE[:n])
    result = {"bets": [{"numbers": list(ticket)} for ticket in bets]}
    cvg = exact_coverage(bets)
    hits = [len(set(ticket).intersection(ACTUAL)) for ticket in bets]
    highest = max(hits)
    score = {
        "ticket_hits": hits, "max_hits": highest,
        **{f"any_ge{k}": int(highest >= k) for k in range(3, 7)},
        "exact_fair_coverage": cvg["at_least"],
    }
    return result, score


def fake_report(*, count=5, patterns=None, seed=649):
    if patterns is None:
        patterns = [(1, 0)]*9 + [(0, 1)]*5 + [(1, 1)]*2 + [(0, 0)]*84
    targets = []
    for i, (s1_hit, b1_hit) in enumerate(patterns):
        year = 111 + i//35
        draw_id = year*1000000 + 70 + i
        scores, committed = {}, {}
        for arm in ARMS:
            strategies = ("random",) if arm == ARMS[0] else (
                ("disjoint",) if arm == ARMS[3] else STRATEGIES)
            scores[arm] = {}
            committed[arm] = {"pool_sha256": None, "portfolios": {}}
            for strategy in strategies:
                on = s1_hit if arm == ARMS[2] else b1_hit if arm == ARMS[1] else False
                chosen, sc = portfolio(count, bool(on))
                scores[arm][strategy] = sc
                committed[arm]["portfolios"][strategy] = [v["numbers"] for v in chosen["bets"]]
        targets.append({"target_draw_id": draw_id,
                        "cutoff_draw_id": draw_id - 1,
                        "history_sha256": "f"*64, "actual": list(ACTUAL),
                        "selection_commitments": committed,
                        "scores": scores})
    return {"schema": BASELINE_SCHEMA, "parameters": {"count": count, "seed": seed},
            "target_count": len(targets), "targets": targets}


class StatisticalTests(unittest.TestCase):
    def test_exact_mcnemar_anchor(self):
        self.assertAlmostEqual(mcnemar_exact(9, 5), 0.4239501953125)
        self.assertAlmostEqual(mcnemar_exact(10, 5), 0.3017578125)
        self.assertAlmostEqual(mcnemar_exact(8, 6), 0.79052734375)
        self.assertEqual(mcnemar_exact(0, 0), 1.)
        self.assertEqual(mcnemar_exact(1, 1), 1.)
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2/1024)
        for x, y in ((-1, 2), (1, -1), (1.5, 2), (False, 1)):
            with self.subTest(pair=(x, y)), self.assertRaises(ValueError):
                mcnemar_exact(x, y)

    def test_holm_anchors_and_monotonicity(self):
        result = holm({"r": 0.4239501953125, "s": 0.3017578125,
                       "c": 0.79052734375})
        self.assertAlmostEqual(result["r"], 0.9052734375)
        self.assertAlmostEqual(result["s"], 0.9052734375)
        self.assertAlmostEqual(result["c"], 0.9052734375)
        self.assertEqual(holm({"a": 0.001, "b": 0.02, "c": 0.5}),
                         {"a": 0.003, "b": 0.04, "c": 0.5})
        with self.assertRaises(ValueError):
            holm({})

    def test_replays_preserve_paired_outcomes(self):
        report = fake_report()
        first = audit_rolling(report)
        second = audit_rolling(report)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], AUDIT_SCHEMA)
        self.assertEqual(first["target_count"], 100)
        for method in STRATEGIES:
            item = first["comparisons"][method]
            self.assertEqual((item["both_ge3"], item["s1_only_ge3"],
                              item["b1_only_ge3"], item["neither_ge3"]),
                             (2, 9, 5, 84))
            self.assertAlmostEqual(item["observed_difference"], 0.04)
            self.assertAlmostEqual(item["mcnemar_two_sided_exact_p"],
                                   0.4239501953125)
            self.assertEqual(sum(row["draws"] for row in item["year"].values()), 100)
            self.assertLessEqual(item["observed_paired_bootstrap_95"][0],
                                 item["observed_paired_bootstrap_95"][1])

    def test_count_1_through_5_and_no_extra_draws(self):
        for n in range(1, 6):
            rep = fake_report(count=n, patterns=[(1, 0), (0, 1), (0, 0)])
            out = audit_rolling(rep)
            self.assertEqual(out["ticket_count_per_portfolio"], n)
            self.assertEqual(out["target_count"], 3)
            self.assertEqual(out["comparisons"]["random"]["s1_only_ge3"], 1)

    def test_detect_tampered_scores_and_portfolios(self):
        original = fake_report(patterns=[(1, 0), (0, 1)])
        probes = []
        a = copy.deepcopy(original)
        a["targets"][0]["scores"][ARMS[2]]["random"]["any_ge3"] = 0
        probes.append(a)
        a = copy.deepcopy(original)
        a["targets"][0]["selection_commitments"][ARMS[2]]["portfolios"]["random"][0] = [1,2,3,4,5,6]
        probes.append(a)
        a = copy.deepcopy(original)
        a["targets"][1]["target_draw_id"] = a["targets"][0]["target_draw_id"]
        probes.append(a)
        a = copy.deepcopy(original)
        a["targets"][0]["cutoff_draw_id"] = a["targets"][0]["target_draw_id"]
        probes.append(a)
        a = copy.deepcopy(original)
        a["parameters"]["count"] = 6
        probes.append(a)
        a = copy.deepcopy(original)
        a["targets"][0]["scores"][ARMS[2]]["random"]["exact_fair_coverage"]["3"] = 0.5
        probes.append(a)
        for i, item in enumerate(probes):
            with self.subTest(case=i), self.assertRaises((ValueError, KeyError)):
                audit_rolling(item)

    def test_parameter_list_validation(self):
        self.assertEqual(_positive_tuple("1, 2,3", low=1, high=5), (1,2,3))
        for s in ("1,1", "0", "6", "1,", "1,x", "", "-1", "1,2.0"):
            with self.subTest(spec=s), self.assertRaises(ValueError):
                _positive_tuple(s, low=1, high=5)

    def test_same_draws_clustered_across_seeds_and_counts(self):
        calls = []
        def fake_rolling(images, ids, **kwargs):
            count, seed = kwargs["count"], kwargs["seed"]
            calls.append((count, seed))
            patterns = [(1, 0), (0, 1), (1, 1), (0, 0)] if seed == 649 else [
                (0, 1), (1, 0), (0, 0), (1, 1)]
            return fake_report(count=count, seed=seed, patterns=patterns)
        with patch("lotto7x7.robustness_v071.rolling_origin", side_effect=fake_rolling):
            out = sweep_rolling(np.zeros((1,)), np.zeros((1,)),
                                seeds="649,650", counts="1,5", eval_draws=4)
        self.assertEqual(out["schema"], SWEEP_SCHEMA)
        self.assertEqual(out["independent_target_draws"], 4)
        self.assertEqual(out["multiple_comparisons"]["tests"], 12)
        self.assertEqual(calls, [(1,649), (1,650), (5,649), (5,650)])
        for n in ("N1", "N5"):
            for method in STRATEGIES:
                item = out["draw_clustered_aggregate"][n][method]
                self.assertEqual(item["independent_target_draws"], 4)
                self.assertEqual(item["seed_replicates_per_draw"], 2)
                self.assertAlmostEqual(item["mean_observed_difference"], 0)

    def test_sweep_rejects_different_target_sets(self):
        calls = []
        def fake_rolling(images, ids, **kwargs):
            calls.append(kwargs["seed"])
            value = fake_report(count=kwargs["count"], patterns=[(1, 0), (0, 1)])
            if kwargs["seed"] == 650:
                value["targets"][1]["target_draw_id"] += 1
            return value
        with patch("lotto7x7.robustness_v071.rolling_origin", side_effect=fake_rolling):
            with self.assertRaises(AssertionError):
                sweep_rolling(None, None, seeds="649,650", counts="5", eval_draws=2)

    def test_cli_works_from_existing_json_without_full_recompute(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)
        configure_parsers(sub)
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp)/"baseline.json"
            out = Path(tmp)/"audit.json"
            inp.write_text(json.dumps(fake_report(patterns=[(1,0), (0,1)])), encoding="utf-8")
            args = parser.parse_args(["robustness-audit", "--input", str(inp), "--out", str(out)])
            run(args)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], AUDIT_SCHEMA)
            self.assertIn("random", result["comparisons"])
            sweep = parser.parse_args(["robustness-sweep", "--seeds", "649,650",
                                       "--counts", "1,2,3,4,5"])
            self.assertEqual(sweep.seeds, "649,650")
            self.assertEqual(sweep.counts, "1,2,3,4,5")


if __name__ == "__main__":
    unittest.main()
