"""v2.4 P1 USNO astronomy, seasonal state and outcome-isolation tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from lotto7x7.iching_gua_qi_v24_p1 import (
    GUA_CODES, GUA_NAMES, GUA_CHINESE, RULE, TZ, ANCHORS,
    checked_bytes, digest, lunar_phase, make_row, micros, parse_anchors,
    parse_local, six_ball_hexagram, solar_state, verify_source_archive,
)


def at(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


class TestSeasonalStates(unittest.TestCase):
    def setUp(self):
        self.a = [(at("2025-12-21T23:00+08:00"), "winter_solstice"),
                  (at("2026-03-20T23:00+08:00"), "spring_equinox"),
                  (at("2026-06-21T23:00+08:00"), "summer_solstice"),
                  (at("2026-09-23T23:00+08:00"), "autumn_equinox"),
                  (at("2026-12-22T23:00+08:00"), "winter_solstice"),
                  (at("2027-03-20T23:00+08:00"), "spring_equinox")]

    def test_all_twelve_canonical_codes(self):
        self.assertEqual(GUA_CODES, (1,3,7,15,31,63,62,60,56,48,32,0))
        self.assertEqual(len(set(GUA_CODES)), 12)
        self.assertEqual(GUA_NAMES[0], "Fu")
        self.assertEqual(GUA_CHINESE[-1], "坤")
        self.assertEqual(sum(i.bit_count() for i in GUA_CODES), 36)

    def test_solar_anchor_centers(self):
        for t,n in self.a[:-1]:
            row=solar_state(t,self.a)
            self.assertEqual(int(row["seasonal_state_12"]), ANCHORS[n])
            self.assertEqual(row["seasonal_hexagram_code"],str(GUA_CODES[ANCHORS[n]]))
        self.assertEqual(solar_state(self.a[-1][0],self.a), {})

    def test_exact_half_state_boundary_advances(self):
        start,end=self.a[0][0],self.a[1][0]
        width=micros(end-start)
        self.assertEqual(width % 6, 0)
        boundary=start + timedelta(microseconds=width//6)
        prev=boundary-timedelta(microseconds=1)
        self.assertEqual(solar_state(prev,self.a)["seasonal_state_12"],"0")
        self.assertEqual(solar_state(boundary,self.a)["seasonal_state_12"],"1")

    def test_full_year_all_twelve_states_are_reachable(self):
        seen={solar_state(self.a[0][0]+timedelta(hours=4*i),self.a).get("seasonal_state_12")
              for i in range(24*365//4)}
        self.assertEqual(seen-{None},{str(i) for i in range(12)})

    def test_leap_year_in_interval(self):
        arr=[(at("2023-12-22T12:00+08:00"),"winter_solstice"),
             (at("2024-03-20T12:00+08:00"),"spring_equinox")]
        r=solar_state(at("2024-02-29T12:00+08:00"),arr)
        self.assertEqual(r["solar_anchor_start"],"winter_solstice")
        self.assertTrue(0<float(r["solar_phase_fraction"])<0.25)

    def test_out_of_archive_boundary_blank(self):
        self.assertEqual(solar_state(self.a[0][0]-timedelta(days=1),self.a),{})
        self.assertEqual(solar_state(self.a[-1][0]+timedelta(days=1),self.a),{})

    def test_boolean_policy_at_noon(self):
        boundary=at("2026-03-20T23:00+08:00")
        row=solar_state(at("2026-03-20T12:00+08:00"),self.a)
        self.assertEqual(row["solar_anchor_start"],"winter_solstice")
        self.assertEqual(row["solar_anchor_end"],"spring_equinox")
        row2=solar_state(boundary,self.a)
        self.assertEqual(row2["solar_anchor_start"],"spring_equinox")

    def test_solar_phase_fraction_monotonic_per_interval(self):
        start,end=self.a[1][0],self.a[2][0]
        samples=[solar_state(start+(end-start)*f,self.a) for f in (0,.2,.4,.6,.8)]
        phases=[float(x["solar_phase_fraction"]) for x in samples]
        self.assertEqual(phases,sorted(phases))
        self.assertAlmostEqual(phases[0],0.25)


class TestMoonPhase(unittest.TestCase):
    def test_new_moon_exact_and_half(self):
        start=at("2026-01-01T12:00+08:00")
        moons=[start,start+timedelta(days=29),start+timedelta(days=58)]
        self.assertEqual(lunar_phase(start,moons)["moon_phase_fraction"],"0.0000000000")
        self.assertEqual(lunar_phase(start+timedelta(days=14,hours=12),moons)["moon_phase_fraction"],"0.5000000000")
        self.assertEqual(lunar_phase(moons[-1],moons),{})

    def test_early_missing_does_not_guess(self):
        moons=[at("2004-01-22T05:05+08:00"),at("2004-02-20T17:18+08:00")]
        self.assertEqual(lunar_phase(at("2004-01-05T12:00+08:00"),moons),{})


class TestOutcomeIsolation(unittest.TestCase):
    def test_parity_order_matches_legacy_p0(self):
        self.assertEqual(six_ball_hexagram([6,9,12,13,33,39]),58)
        self.assertEqual(six_ball_hexagram([11,23,30,32,40,41]),35)

    def test_invalid_ticket_rejected(self):
        for numbers in ([1,1,2,3,4,5],[0,2,3,4,5,6],[1,2,3,4,5,50],[6,5,4,3,2,1]):
            with self.subTest(numbers=numbers),self.assertRaises(ValueError):
                six_ball_hexagram(numbers)

    def test_outcome_changes_only_outcome_columns(self):
        observed={"date":"2026-03-21","draw_id":"12345678",
            "reference_at_taipei":"2026-03-21T12:00+08:00",
            "numbers":{**{f"n{i}":f"{i:02d}" for i in range(1,7)},"bonus":"12"},
            "observed_code":63}
        seasons=[(at("2025-12-21T23:00+08:00"),"winter_solstice"),
                 (at("2026-03-20T23:00+08:00"),"spring_equinox"),
                 (at("2026-06-21T23:00+08:00"),"summer_solstice")]
        moons=[at("2026-03-01T12:00+08:00"),at("2026-03-30T12:00+08:00")]
        x=make_row(observed,seasons,moons)
        y=make_row({**observed,"observed_code":0},seasons,moons)
        for f in ("seasonal_state_12","seasonal_hexagram_code","moon_phase_fraction", "solar_phase_fraction"):
            self.assertEqual(x[f],y[f])
        self.assertNotEqual(x["same_draw_hexagram_code_OUTCOME_ONLY"],y["same_draw_hexagram_code_OUTCOME_ONLY"])
        self.assertNotEqual(x["hamming_distance_OUTCOME_ONLY"],y["hamming_distance_OUTCOME_ONLY"])
        self.assertEqual(x["mapping_version"],RULE)


class TestRealUsnoArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zip_path=Path(__file__).parent.parent / "_usno_official_test.zip"
        if not cls.zip_path.exists():
            raise unittest.SkipTest("real user USNO ZIP not provided to this environment")

    def test_46_raw_json_provenance_and_event_chronology(self):
        with tempfile.TemporaryDirectory() as d:
            with ZipFile(self.zip_path) as z:
                z.extractall(d)
            root=Path(d)/"usno_official"
            events,_,audit=verify_source_archive(root/"usno_events_2004_2026.csv",
                    root/"manifest_2004_2026.json",root/"raw")
            self.assertEqual(len(events),1276)
            self.assertEqual(audit["usno_original_json_files"],46)
            seasons,moons=parse_anchors(events)
            self.assertEqual(len(seasons),92)
            self.assertEqual(len(moons),284)
            self.assertEqual(solar_state(at("2004-01-05T12:00+08:00"),seasons),{})
            self.assertEqual(lunar_phase(at("2004-01-05T12:00+08:00"),moons),{})
            self.assertTrue(solar_state(at("2026-11-10T12:00+08:00"),seasons))
            self.assertTrue(lunar_phase(at("2026-11-10T12:00+08:00"),moons))

    def test_usno_hash_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with ZipFile(self.zip_path) as z:
                z.extractall(d)
            root=Path(d)/"usno_official"
            path=root/"raw"/"moon_phases"/"2004.json"
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                verify_source_archive(root/"usno_events_2004_2026.csv",
                    root/"manifest_2004_2026.json",root/"raw")


if __name__ == "__main__":
    unittest.main()
