"""P2 solar8/lunar8 proof obligations, independent synthetic edge cases."""
import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
import json
from zipfile import ZipFile

from lotto7x7 import iching_gua_qi_v24_p1 as p1
from lotto7x7 import iching_solar_lunar_v24_p2 as p2

TZ = timezone(timedelta(hours=8))


def at(year, month, day, hour=0, minute=0, microsecond=0):
    return datetime(year, month, day, hour, minute, tzinfo=TZ, microsecond=microsecond)


class TestP2Math(unittest.TestCase):
    def test_gray_cycle_and_bijection(self):
        seq = [p2.gray3(i) for i in range(8)]
        self.assertEqual(seq, [0, 1, 3, 2, 6, 7, 5, 4])
        self.assertEqual(len(set(seq)), 8)
        self.assertTrue(all((seq[i] ^ seq[(i+1) % 8]).bit_count() == 1 for i in range(8)))
        for bad in (-1, 8, 2.0, "1", None):
            with self.assertRaises(ValueError):
                p2.gray3(bad)

    def test_exact_sector_boundaries(self):
        for i in range(8):
            self.assertEqual(p2.sector8(100*i, 800), i)
            if i:
                self.assertEqual(p2.sector8(100*i - 1, 800), i-1)
        self.assertEqual(p2.sector8(799, 800), 7)
        for a, b in [(-1, 800), (800, 800), (0, 0), (1.0, 8), (0, 8.0)]:
            with self.assertRaises(ValueError):
                p2.sector8(a, b)

    def test_64_reachable_hexagrams_and_bit_order(self):
        seen = set()
        for s, l in itertools.product(range(8), repeat=2):
            r = p2.hexagram_from_sectors(s, l)
            code = int(r["astronomical_hexagram_code"])
            self.assertEqual(code & 7, p2.gray3(l))
            self.assertEqual(code >> 3, p2.gray3(s))
            self.assertEqual(r["astronomical_hexagram_binary_bottom_to_top"],
                             "".join(str((code >> j) & 1) for j in range(6)))
            self.assertEqual(int(r["astronomical_yang_count"]), code.bit_count())
            seen.add(code)
        self.assertEqual(seen, set(range(64)))

    def test_trigram_chinese_codes(self):
        self.assertEqual(p2.TRIGRAMS_ZH, ("坤", "震", "坎", "兌", "艮", "離", "巽", "乾"))
        self.assertEqual(p2.hexagram_from_sectors(0, 0)["astronomical_hexagram_code"], "0")
        self.assertEqual(p2.hexagram_from_sectors(5, 5)["astronomical_hexagram_code"], "63")

    def test_winter_spring_summer_autumn_anchors(self):
        a = [(at(2024, 12, 21, 12), "winter_solstice"),
             (at(2025, 3, 21, 12), "spring_equinox"),
             (at(2025, 6, 21, 12), "summer_solstice"),
             (at(2025, 9, 21, 12), "autumn_equinox"),
             (at(2025, 12, 21, 12), "winter_solstice")]
        for when, expected in zip((x[0] for x in a[:-1]), (0, 2, 4, 6)):
            r = p2.solar_segment(when, a)
            self.assertEqual(int(r["solar_sector_8"]), expected)
        self.assertEqual(p2.solar_segment(a[0][0] - timedelta(microseconds=1), a), {})
        self.assertEqual(p2.solar_segment(a[-1][0], a), {})

    def test_solar_half_open_boundary_microsecond(self):
        # A 96-day quarter / 3 months gives exact 1/8-year at quarter midpoint.
        winter = at(2024, 12, 1, 12)
        spring = winter + timedelta(days=96)
        anchors = [(winter, "winter_solstice"), (spring, "spring_equinox")]
        middle = winter + timedelta(days=48)
        self.assertEqual(p2.solar_segment(middle-timedelta(microseconds=1), anchors)["solar_sector_8"], "0")
        self.assertEqual(p2.solar_segment(middle, anchors)["solar_sector_8"], "1")
        self.assertEqual(p2.solar_segment(spring-timedelta(microseconds=1), anchors)["solar_sector_8"], "1")

    def test_solar_leap_year_date_arithmetic(self):
        a = [(at(2024, 2, 28), "winter_solstice"),
             (at(2024, 3, 1), "spring_equinox")]
        r = p2.solar_segment(at(2024, 2, 29), a)
        self.assertEqual(r["solar_sector_8"], "1")
        self.assertEqual(r["solar_phase_fraction"], "0.1250000000")

    def test_solar_misordered_anchors(self):
        with self.assertRaises(ValueError):
            p2.solar_segment(at(2025, 1, 1), [
                (at(2024, 12, 21), "winter_solstice"),
                (at(2025, 3, 21), "summer_solstice")])

    def test_lunar_exact_new_moon_and_end_boundary(self):
        moons = [at(2024, 1, 1, 12), at(2024, 1, 25, 12), at(2024, 2, 18, 12)]
        self.assertEqual(p2.lunar_segment(moons[0], moons)["lunar_sector_8"], "0")
        self.assertEqual(p2.lunar_segment(moons[0] + timedelta(days=3), moons)["lunar_sector_8"], "1")
        self.assertEqual(p2.lunar_segment(moons[1]-timedelta(microseconds=1), moons)["lunar_sector_8"], "7")
        self.assertEqual(p2.lunar_segment(moons[1], moons)["lunar_sector_8"], "0")
        self.assertEqual(p2.lunar_segment(moons[-1], moons), {})
        self.assertEqual(p2.lunar_segment(moons[0]-timedelta(microseconds=1), moons), {})

    def test_lunar_half_moon_is_proxy_not_observed_full_moon(self):
        moons = [at(2024, 1, 1), at(2024, 1, 25)]
        self.assertEqual(p2.lunar_segment(at(2024, 1, 13), moons)["lunar_phase_fraction"], "0.5000000000")
        self.assertEqual(p2.lunar_segment(at(2024, 1, 13), moons)["lunar_sector_8"], "4")

    def test_missing_flags_and_outcome_separation(self):
        prior = {"reference_at_taipei": "2004-01-05T12:00+08:00",
                 "seasonal_state_12": "", "seasonal_hexagram_code": ""}
        draw = {"date": "2004-01-05", "draw_id": "93000001",
                "reference_at_taipei": "2004-01-05T12:00+08:00",
                "observed_code": 58,
                "numbers": {**{f"n{i}": f"{2*i:02d}" for i in range(1,7)}, "bonus": "13"}}
        row = p2.make_row(draw, prior, [(at(2004, 3, 20), "spring_equinox"),
                                         (at(2004, 6, 20), "summer_solstice")],
                          [at(2004, 1, 8), at(2004, 2, 7)])
        self.assertEqual(row["astronomy_pair_mapped"], "false")
        self.assertEqual(row["missing_reason"], "solar_boundary_unavailable+lunar_boundary_unavailable")
        self.assertEqual(row["astronomical_hexagram_code"], "")
        self.assertEqual(row["same_draw_hexagram_code_OUTCOME_ONLY"], "58")
        self.assertEqual(row["hamming_distance_OUTCOME_ONLY"], "")

    def test_same_astronomy_with_different_draw_outcomes(self):
        when = at(2025, 2, 1, 12)
        sun = [(at(2024, 12, 20), "winter_solstice"),
               (at(2025, 3, 20), "spring_equinox")]
        moon = [at(2025, 1, 30), at(2025, 2, 28)]
        prior = {"reference_at_taipei": when.isoformat(timespec="minutes"),
                 "seasonal_state_12": "0", "seasonal_hexagram_code": "1"}
        base = {"date": when.date().isoformat(), "draw_id": "1",
                "reference_at_taipei": when.isoformat(timespec="minutes"),
                "numbers": {**{f"n{i}": str(i) for i in range(1,7)}, "bonus": "49"}}
        a = p2.make_row({**base, "observed_code": 0}, prior, sun, moon)
        b = p2.make_row({**base, "observed_code": 63}, prior, sun, moon)
        for k in ("astronomical_hexagram_code", "solar_sector_8", "lunar_sector_8",
                  "astronomical_hexagram_binary_bottom_to_top", "source_provenance"):
            self.assertEqual(a[k], b[k], k)
        self.assertNotEqual(a["hamming_distance_OUTCOME_ONLY"], b["hamming_distance_OUTCOME_ONLY"])

    def test_reference_requires_taipei_offset(self):
        with self.assertRaises(ValueError):
            p1.parse_local("2025-11-13T12:00:00+00:00")
        with self.assertRaises(ValueError):
            p1.parse_local("2025-11-13T12:00:00")


class TestP2OfficialSources(unittest.TestCase):
    def test_source_raws_are_pinned_when_available(self):
        root = Path("data/astronomy/usno_official")
        if not (root / "manifest_2004_2026.json").exists():
            self.skipTest("official USNO raw archive not present in isolated development tests")
        events, manifest, audit = p1.verify_source_archive(
            root / "usno_events_2004_2026.csv", root / "manifest_2004_2026.json", root / "raw")
        self.assertEqual(len(events), 1276)
        self.assertEqual(audit["usno_original_json_files"], 46)
        self.assertEqual(manifest["events_csv_sha256"], p1.EXPECTED_USNO_EVENTS_SHA256)
        anchors, moons = p1.parse_anchors(events)
        self.assertEqual((len(anchors), len(moons)), (92, 284))
        self.assertFalse(p2.solar_segment(at(2004, 1, 5, 12), anchors))
        self.assertFalse(p2.lunar_segment(at(2004, 1, 5, 12), moons))

    def test_readonly_sources_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/p2.OUT_CSV
            p.write_text("preexisting immutable data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                p2.generate(events_file=Path(d)/"missing_events",
                            manifest_file=Path(d)/"missing_manifest", raw_dir=Path(d)/"raw",
                            draw_file=Path(d)/"missing_draw", postdraw_file=Path(d)/"missing_post",
                            postdraw_report=Path(d)/"missing_post_report",
                            p1_file=Path(d)/"missing_p1", p1_report=Path(d)/"missing_p1_report",
                            out_dir=Path(d))
            self.assertEqual(p.read_text(encoding="utf-8"), "preexisting immutable data")


if __name__ == "__main__":
    unittest.main()
