"""Issue #38: adversarial, no-network tests of evidence integrity and stage gates."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lotto7x7 import research_governance as g


class ResearchGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "docs").mkdir()
        (self.root / "docs" / "original.md").write_bytes(b"Frozen original evidence\n")
        self.base = {
            "id": "test-experiment-0001", "issue": 38, "domain": "geometry",
            "research_question": "Does this test have a valid evidence claim?",
            "hypothesis": "A frozen reproducible hypothesis.",
            "evidence_stage": "EXPLORATORY_RETROSPECTIVE",
            "research_decision": "NO_EVIDENCE", "engineering_status": "PASS",
            "source_cutoff": "2010-01-01, frozen fixture", "sample_size": 6,
            "planned_future_draws": 0,
            "model_version": "synthetic-v1", "null_model": "fair independent 6/49",
            "multiple_testing": "Bonferroni, single hypothesis",
            "primary_endpoint": "unbiased contrast", "method": "frozen synthetic test",
            "result_summary": "Synthetic null result, not a forecast",
            "limitations": ["Test fixture, not a real observed drawing."],
            "next_action": "Retain this record", "evidence_commit": None,
            "ci_run": None, "evidence_as_of": "2026-11-14",
            "artifacts": [{"role": "report", "path": "docs/original.md",
                           "git_blob_sha1": g.git_blob_sha1(b"Frozen original evidence\n")}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def validated(self, entries=None, *, blobs=True):
        return g.validate_registry({"schema": g.SCHEMA, "entries": entries or [self.base]},
                                   self.root, verify_blobs=blobs)

    def test_valid_fixture_and_deterministic_render(self):
        entries = self.validated()
        a = g.render_index(entries)
        self.assertEqual(a, g.render_index(entries))
        self.assertIn("NO_EVIDENCE", a)
        self.assertIn("original.md", a)

    def test_git_blob_hash_detects_mutation(self):
        self.validated()
        (self.root / "docs/original.md").write_bytes(b"Silently altered!\n")
        with self.assertRaisesRegex(g.GovernanceError, "pinned original Git blob changed"):
            self.validated()

    def test_allow_checkout_crlf_without_allowing_mixed_cr(self):
        (self.root / "docs/original.md").write_bytes(b"Frozen original evidence\r\n")
        self.validated()
        (self.root / "docs/original.md").write_bytes(b"Frozen original evidence\rX\n")
        with self.assertRaises(g.GovernanceError):
            self.validated()

    def test_reject_path_traversal_private_paths_and_symlinks(self):
        for path in ["../secret.md", "/tmp/secret", "data/private/bets.json", "docs/../../x", "docs\\original.md"]:
            with self.subTest(path=path):
                item = deepcopy(self.base)
                item["artifacts"][0]["path"] = path
                with self.assertRaises(g.GovernanceError):
                    self.validated([item])
        (self.root / "docs/link.md").symlink_to(self.root.parent)
        item = deepcopy(self.base)
        item["artifacts"][0]["path"] = "docs/link.md"
        with self.assertRaises(g.GovernanceError):
            self.validated([item])

    def test_no_retroactive_promotion_from_a_historical_null(self):
        bad = deepcopy(self.base)
        bad["research_decision"] = "VALIDATED_GO"
        with self.assertRaises(g.GovernanceError):
            self.validated([bad], blobs=False)

    def test_ci_pass_does_not_establish_predictive_go(self):
        bad = deepcopy(self.base)
        bad["evidence_stage"] = "PROSPECTIVE_EVALUATED"
        bad["research_decision"] = "VALIDATED_GO"
        bad["ci_run"] = "https://github.com/cctsao1008/lotto649/actions/runs/12345"
        with self.assertRaisesRegex(g.GovernanceError, "never automatically declares predictive GO"):
            self.validated([bad], blobs=False)

    def test_preseal_pending_require_zero_outcomes_and_seal(self):
        p = deepcopy(self.base)
        p.update(evidence_stage="PROSPECTIVE_PRESEALED",research_decision="PENDING",
                 sample_size=0,planned_future_draws=100)
        p["artifacts"][0]["role"] = "seal"
        self.validated([p])
        for field, value in [("sample_size", 1), ("planned_future_draws", 0),
                             ("research_decision", "NO_EVIDENCE")]:
            bad = deepcopy(p)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(g.GovernanceError):
                self.validated([bad], blobs=False)
        bad = deepcopy(p)
        bad["artifacts"][0]["role"] = "report"
        with self.assertRaises(g.GovernanceError):
            self.validated([bad], blobs=False)

    def test_p1a_native_result_guard(self):
        item = deepcopy(self.base)
        item["id"] = "iching-p1a-20261113"
        item["artifacts"][0]["path"] = "docs/p1a.json"
        payload = {"decision": "EXPLORATORY_ASSOCIATION_SCREEN_NO_EVIDENCE",
                   "analysis_draws": 2476, "bonferroni_0_05_significant_endpoints": []}
        raw = (json.dumps(payload) + "\n").encode()
        (self.root / "docs/p1a.json").write_bytes(raw)
        item["artifacts"][0]["git_blob_sha1"] = g.git_blob_sha1(raw)
        self.validated([item])
        # A fully pinned, self-consistent new byte stream cannot silently rewrite
        # a scientific decision when the original native report schema is known.
        payload["decision"] = "EXPLORATORY_ASSOCIATION_SCREEN_GO"
        raw = (json.dumps(payload) + "\n").encode()
        (self.root / "docs/p1a.json").write_bytes(raw)
        item["artifacts"][0]["git_blob_sha1"] = g.git_blob_sha1(raw)
        with self.assertRaisesRegex(g.GovernanceError,"P1A original"):
            self.validated([item])

    def test_p2a_native_six_test_multiplicity_guard(self):
        item = deepcopy(self.base)
        item["id"] = "iching-p2a-20261114"
        item["artifacts"][0]["path"] = "docs/p2a.json"
        payload = {"decision": "EXPLORATORY_ASSOCIATION_NO_EVIDENCE",
                   "analysis_draws": 2476, "bonferroni_significant_endpoints": [],
                   "endpoints": {str(k): {"bonferroni_six_global_p": 0.3345 + k/100}
                                 for k in range(6)}}
        raw = (json.dumps(payload) + "\n").encode()
        (self.root / "docs/p2a.json").write_bytes(raw)
        item["artifacts"][0]["git_blob_sha1"] = g.git_blob_sha1(raw)
        self.validated([item])
        payload["bonferroni_significant_endpoints"] = ["test"]
        raw = (json.dumps(payload) + "\n").encode()
        (self.root / "docs/p2a.json").write_bytes(raw)
        item["artifacts"][0]["git_blob_sha1"] = g.git_blob_sha1(raw)
        with self.assertRaises(g.GovernanceError):
            self.validated([item])

    def test_duplicate_invalid_and_unsorted_ids(self):
        a = deepcopy(self.base)
        b = deepcopy(self.base)
        with self.assertRaises(g.GovernanceError):
            self.validated([a,b],blobs=False)
        b["id"] = "bad/experiment"
        with self.assertRaises(g.GovernanceError):
            self.validated([b],blobs=False)
        b["id"] = "aaa"
        with self.assertRaisesRegex(g.GovernanceError,"lexicographically sorted"):
            self.validated([a,b],blobs=False)

    def test_missing_fields_and_unknown_evidence_stage(self):
        item=deepcopy(self.base)
        del item["null_model"]
        with self.assertRaises(g.GovernanceError):
            self.validated([item],blobs=False)
        item=deepcopy(self.base)
        item["evidence_stage"]="FAKE_PROSPECTIVE_SUCCESS"
        with self.assertRaises(g.GovernanceError):
            self.validated([item],blobs=False)

    def test_theorem_never_assumes_forecast(self):
        item=deepcopy(self.base)
        item.update(evidence_stage="STRUCTURAL_THEOREM",research_decision="NO_EVIDENCE")
        with self.assertRaises(g.GovernanceError):
            self.validated([item],blobs=False)
        item["research_decision"]="PROVEN_MATHEMATICALLY"
        self.validated([item],blobs=False)

    def test_index_verification_detects_stale_render(self):
        entries=self.validated()
        (self.root/g.INDEX).write_text("stale",encoding="utf-8")
        (self.root/g.REGISTRY).parent.mkdir(parents=True)
        (self.root/g.REGISTRY).write_text(json.dumps({"schema":g.SCHEMA,"entries":entries}),encoding="utf-8")
        self.assertEqual(g.main(["verify", "--root", str(self.root)]),2)
        (self.root/g.INDEX).write_text(g.render_index(entries),encoding="utf-8")
        self.assertEqual(g.main(["verify", "--root", str(self.root)]),0)


if __name__ == "__main__":
    unittest.main()
