"""Adversarial source/seal-only tests for v0.32 prospective readiness."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path

from lotto7x7 import prospective_readiness_v032 as p


ROOT = Path(__file__).resolve().parents[2]


class ProspectiveReadinessTests(unittest.TestCase):
    def test_all_four_original_seals_and_sky_archive(self):
        meta = [p.check_study(ROOT, s) for s in p.STUDIES]
        self.assertEqual(len(meta), 4)
        self.assertEqual([x["horizon"] for x in meta], [10, 9, 200, 100])
        p.verify_cross_study(meta)

    def test_original_publication_sha_not_a_current_file_hash(self):
        for s in p.STUDIES:
            self.assertEqual(p.blob_sha1((ROOT/s.path).read_bytes()), s.blob)
        self.assertEqual(len({s.blob for s in p.STUDIES}), 4)

    def test_git_windows_lf_crlf_checkout_equivalence(self):
        raw=b"abc\nxyz\n"
        self.assertEqual(p.blob_sha1(raw), p.blob_sha1(raw.replace(b"\n",b"\r\n")))
        with self.assertRaises(ValueError):
            p.blob_sha1(b"abc\rxyz\n")

    def test_original_seal_tampering_refused(self):
        one=p.STUDIES[0]
        raw=(ROOT/one.path).read_bytes()
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/one.path
            dest.parent.mkdir(parents=True)
            dest.write_bytes(raw.replace(b"SHADOW_PRE_REGISTERED_NO_PURCHASE",
                                         b"SHADOW_PRE_REGISTERED_YES_PURCHASE", 1))
            with self.assertRaisesRegex(ValueError,"seal drift"):
                p.safe_seal(Path(d),one)

    def test_fixed_preseal_schedule_must_be_exact_prefix(self):
        one=p.STUDIES[0]
        schedule=p.check_study(ROOT,one)["fixed_schedule"]
        self.assertEqual(schedule[0],{"draw_id":115000104,"date":"2026-11-10"})
        broken=copy.deepcopy(schedule)
        broken[2]["draw_id"]=115000108
        with self.assertRaisesRegex(ValueError,"nonconsecutive"):
            p.check_fixed_schedule(broken,one)
        with self.assertRaises(ValueError):
            p.check_fixed_schedule(schedule[:-1],one)

    def test_six_sorted_legal_disjoint_numbers(self):
        good={"A":{"tickets":[[1,2,3,4,5,6],[7,8,9,10,11,12],
                              [13,14,15,16,17,18],[19,20,21,22,23,24]]}}
        p.check_ticket_arms(good,1)
        bad=copy.deepcopy(good)
        bad["A"]["tickets"][1][-1]=13
        with self.assertRaisesRegex(ValueError,"disjoint"):
            p.check_ticket_arms(bad,1)
        bad=copy.deepcopy(good)
        bad["A"]["tickets"][0][0]=0
        with self.assertRaises(ValueError):
            p.check_ticket_arms(bad,1)

    def test_fixed_archived_not_same_as_admitted(self):
        meta=p.check_study(ROOT,p.STUDIES[0])
        archive=[{"draw_id":115000104,"date":"2026-11-10"},
                 {"draw_id":115000105,"date":"2026-11-13"}]
        self.assertEqual(len(p.archived_prefix(meta,archive)),2)
        self.assertNotIn("authenticated",str(meta))
        broken=[{"draw_id":115000105,"date":"2026-11-13"}]
        with self.assertRaises(ValueError):
            p.archived_prefix(meta,broken)

    def test_rolling_starts_only_at_first_prespecified_future_event(self):
        meta=p.check_study(ROOT,p.STUDIES[2])
        self.assertEqual(p.archived_prefix(meta,[{"draw_id":115000105,"date":"2026-11-13"}]),[])
        good=[{"draw_id":115000106,"date":"2026-11-17"},
              {"draw_id":115000107,"date":"2026-11-20"}]
        self.assertEqual(len(p.archived_prefix(meta,good)),2)
        with self.assertRaises(ValueError):
            p.archived_prefix(meta,good[1:])
        with self.assertRaises(ValueError):
            p.archived_prefix(meta,good+[{"draw_id":115000109,"date":"2026-11-27"}])

    def test_p3_sky_tampering_rejected(self):
        study=p.STUDIES[3]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for rel in (study.path,"prospective/v24_p3_sky_calendar_20261114.csv"):
                src=ROOT/rel
                dest=root/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(src.read_bytes())
            p.check_study(root,study)
            sky=root/"prospective/v24_p3_sky_calendar_20261114.csv"
            sky.write_bytes(sky.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA256 drift"):
                p.check_study(root,study)

    def test_cross_study_overlap_is_not_independent_replication(self):
        meta=[p.check_study(ROOT,s) for s in p.STUDIES]
        p.verify_cross_study(meta)
        self.assertEqual(meta[1]["fixed_schedule"],meta[0]["fixed_schedule"][1:])
        self.assertEqual(meta[2]["first_eligible_draw"],meta[3]["first_eligible_draw"])
        self.assertEqual(meta[0]["fixed_schedule"][2],meta[2]["first_eligible_draw"])

    def test_no_one_study_threshold_proves_familywise_go(self):
        self.assertGreater(2 * (1 / 20), 0.05)
        self.assertEqual(2 * (1 / 20), 0.10)

    def test_fixed_schedule_roc_date_mismatch_rejected(self):
        one=p.STUDIES[0]
        sch=copy.deepcopy(p.check_study(ROOT,one)["fixed_schedule"])
        sch[3]["date"]="2025-11-20"
        with self.assertRaises(ValueError):
            p.check_fixed_schedule(sch,one)


if __name__=="__main__":
    unittest.main()
