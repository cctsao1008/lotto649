"""v2.4 mathematical, source-provenance and temporal-leakage regression tests."""
from collections import Counter
from datetime import date,timedelta
from itertools import combinations
from math import comb
import unittest

from lotto7x7.astronomy_research_v24 import (
    UNIVERSE,HEX_ENCODING,COLUMNS,hex_code,exact_hexagram_universe,
    parse_cwa_calendar,join_data,
)


def mk(day,ident,balls=(4,9,17,24,29,42),bonus=1):
    return {'date':day,'draw_id':ident,'main':list(balls),'bonus':bonus}


class AstronomyResearchV24Tests(unittest.TestCase):
    def test_01_universe_count_exact(self):
        x=exact_hexagram_universe()
        self.assertEqual(len(x),64)
        self.assertEqual(sum(x.values()),comb(49,6))
        self.assertEqual(UNIVERSE,13983816)
        self.assertEqual(x[0],134596)
        self.assertEqual(max(x.values()),296010)
        self.assertNotEqual(min(x.values()),max(x.values()))

    def test_02_reduced_universe_independent_oracle(self):
        n=9;k=3
        brute=Counter(sum((v&1)<<i for i,v in enumerate(t)) for t in combinations(range(1,n+1),k))
        self.assertEqual(exact_hexagram_universe(ball_count=n,ticket_size=k),dict(sorted(brute.items())))

    def test_03_parity_is_bottom_up(self):
        self.assertEqual(hex_code([4,9,17,24,29,42]),0b010110)
        self.assertEqual(hex_code([1,2,3,4,5,6]),21)
        self.assertEqual(hex_code([2,4,6,8,10,12]),0)

    def test_04_bad_numbers_rejected(self):
        for row in ([1,2,3,4,5,5],[0,1,2,3,4,5],[1,2,3,4,5,50],[9,2,3,4,5,6],[1,2,3],[True,2,3,4,5,6]):
            with self.assertRaises(ValueError):hex_code(row)

    def test_05_cwa_documented_english_schema(self):
        src='GregorianCalendar,RepublicOfChinaCalendar,AgriculturalCalendar,AgriculturalCalendarMonth,AgriculturalCalendarDay,Week\n2026-11-10,115,丙午,10,2,2\n'
        v=parse_cwa_calendar(src.encode('utf-8-sig'))
        self.assertEqual(v['2026-11-10'],{'cwa_lunar_year_label':'丙午','cwa_lunar_month':'10','cwa_lunar_leap_month':'0','cwa_lunar_day':'2'})

    def test_06_cwa_leap_month_and_chinese_schema(self):
        src='格里曆日期,民國曆年,農曆年,農曆月,農曆日,星期\n2026-11-10,115,丙午,閏6,2,2\n'
        v=parse_cwa_calendar(src.encode('utf-8'))
        self.assertEqual(v['2026-11-10']['cwa_lunar_month'],'6')
        self.assertEqual(v['2026-11-10']['cwa_lunar_leap_month'],'1')

    def test_07_cwa_invalid_duplicate_and_wrong_weekday_rejected(self):
        head='GregorianCalendar,RepublicOfChinaCalendar,AgriculturalCalendar,AgriculturalCalendarMonth,AgriculturalCalendarDay,Week\n'
        for text in (head+'2026-11-10,115,丙午,10,2,1\n',
                     head+'2026-11-10,115,丙午,10,2,2\n2026-11-10,115,丙午,10,2,2\n',
                     head+'2026-11-10,114,丙午,10,2,2\n',
                     head+'2026-11-10,115,丙午,13,2,2\n'):
            with self.assertRaises(ValueError):parse_cwa_calendar(text.encode())

    def test_08_one_to_one_draw_join_and_no_cwa_imputation(self):
        f=lambda d:{'solar_ecliptic_longitude_deg':'100.000','astronomy_provenance':'TEST_EPHEMERIS'}
        ds=[mk('2026-11-10',115000104),mk('2026-11-13',115000105,(1,3,5,7,9,11))]
        out=join_data(ds,astro_func=f)
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]['cwa_calendar_provenance'],'NOT_AVAILABLE')
        self.assertEqual(out[0]['cwa_lunar_month'],'')
        self.assertEqual(out[0]['previous_hexagram_code'],'')
        self.assertEqual(out[1]['previous_hexagram_code'],out[0]['hexagram_code'])
        self.assertEqual(out[1]['changed_lines_from_previous_draw'],str((int(out[0]['hexagram_code'])^int(out[1]['hexagram_code'])).bit_count()))
        self.assertEqual(out[1]['draw_gap_days'],'3')
        self.assertEqual(tuple(out[0]),COLUMNS)

    def test_09_cwa_join_only_present_day(self):
        dates={'2026-11-10':{'cwa_lunar_year_label':'丙午','cwa_lunar_month':'10','cwa_lunar_leap_month':'0','cwa_lunar_day':'2'}}
        f=lambda d:{'astronomy_provenance':'MOCK'}
        a=join_data([mk('2026-11-10',115000104),mk('2026-11-13',115000105)],dates,astro_func=f)
        self.assertEqual(a[0]['cwa_lunar_day'],'2')
        self.assertEqual(a[1]['cwa_lunar_day'],'')

    def test_10_out_of_order_rejected(self):
        f=lambda d:{}
        with self.assertRaises(ValueError):join_data([mk('2026-11-13',115000105),mk('2026-11-10',115000104)],astro_func=f)

    def test_11_no_mix_inferred_and_official_fields(self):
        f=lambda d:{'moon_illumination_percent':'50.0','astronomy_provenance':'EPHEMERIS'}
        a=join_data([mk('2026-11-10',115000104)],astro_func=f)[0]
        self.assertEqual(a['astronomy_provenance'],'EPHEMERIS')
        self.assertEqual(a['cwa_calendar_provenance'],'NOT_AVAILABLE')
        self.assertEqual(a['cwa_lunar_year_label'],'')

    def test_12_encoding_contract_is_stable(self):
        self.assertEqual(HEX_ENCODING,'ascending_6_numbers_bottom_to_top_odd_yang_even_yin_v1')


if __name__=='__main__':unittest.main()
