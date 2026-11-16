"""v0.25 P0 full exact covariance adversarial tests; no forecast data."""
from __future__ import annotations
from itertools import combinations
from hashlib import sha1,sha256
from math import comb
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from research.lotto7x7 import exact_covariance_v025 as m

ROOT=Path(__file__).resolve().parents[3]
FIX=ROOT/"data/reference/shape_class_counts.json"
CPP=ROOT/"research/lotto7x7/portfolio/exact_covariance_v025.cpp"
PINNED_FIXTURE_GIT_BLOB="7d301f16937d985ce5fb7cef590bb1b1a109b0b6"

def graph_reference(numbers):
    # Independent Python vertex BFS, not the new C++ implementation.
    pts=[divmod(n-1,7) for n in numbers]
    adj=[{i} for i in range(6)]
    card=diag=0
    for i,j in combinations(range(6),2):
        dr=abs(pts[i][0]-pts[j][0]);dc=abs(pts[i][1]-pts[j][1])
        if dr<=1 and dc<=1:adj[i].add(j);adj[j].add(i)
        card+=int(dr+dc==1);diag+=int(dr==1 and dc==1)
    left=set(range(6));size=[]
    while left:
        seen=set();pending=[min(left)]
        while pending:
            p=pending.pop()
            if p in seen:continue
            seen.add(p);pending.extend(adj[p]-seen)
        left-=seen;size.append(len(seen))
    part=sum(v<<(3*(5-i)) for i,v in enumerate(sorted(size,reverse=True)))
    return (part<<10)|(card<<5)|diag

class ExactFullCovarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.exe=Path(cls.tmp.name)/"census"
        cls.tsv=Path(cls.tmp.name)/"census.tsv"
        subprocess.run(["g++","-O3","-std=c++17","-Wall","-Wextra","-Werror",str(CPP),"-o",str(cls.exe)],check=True)
        with cls.tsv.open("wb") as fp:
            subprocess.run([str(cls.exe)],check=True,stdout=fp)
        raw=FIX.read_bytes()
        if b"\r\n" in raw:raw=raw.replace(b"\r\n",b"\n")
        actual=sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        if actual!=PINNED_FIXTURE_GIT_BLOB:raise AssertionError("original v0.16 source fixture drift")
        cls.baseline=json.loads(FIX.read_text("utf-8"))
        cls.data,cls.digest=m.read_census(cls.tsv,cls.baseline)
        cls.report=m.build_exact_moments(cls.data,cls.digest,cls.baseline)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_full_exact_enumeration_and_taxonomy_counts(self):
        self.assertEqual(sum(v["count"] for v in self.data.values()),13983816)
        self.assertEqual(len(self.data),94)
        self.assertEqual(self.report["feature_count"],163)
        self.assertEqual(sum(map(len,self.report["raw_second_moment_upper"])),13366)
        self.assertEqual([len(self.baseline["counts"][x]) for x in ("component","partition","edge")],[6,11,94])

    def test_frozen_golden_census_checksum(self):
        self.assertEqual(self.digest,"0a7efd559b25bff00ee978213e398113649e7702519d7e7da5556a2036471a72")

    def test_scalar_cpp_classifier_matches_independent_python_graph(self):
        samples=[
            (1,2,3,4,5,6),(1,3,5,7,9,11),(1,8,15,22,29,36),
            (4,12,19,20,44,48),(10,17,24,31,38,45),(2,9,16,23,30,49)]
        for ticket in samples:
            x=subprocess.check_output([str(self.exe),"--classify",*map(str,ticket)],text=True)
            self.assertEqual(int(x.strip()),graph_reference(ticket))
        bad=subprocess.run([str(self.exe),"--classify","1","2","3","4","4","6"],capture_output=True)
        self.assertNotEqual(bad.returncode,0)

    def test_exact_numeric_linear_constraints(self):
        r=self.report;u=m.U
        self.assertEqual(r["sums"][:49],[comb(48,5)]*49)
        self.assertEqual(r["sums"][49],150*u)
        self.assertEqual(r["sums"][50],25*comb(48,5))
        self.assertEqual(m.cov_numerator(r,49,49),1075*u*u)
        for j in [0,25,48,49,50,51,52,58,69,162]:
            raw=lambda i,k:r["raw_second_moment_upper"][min(i,k)][abs(k-i)]
            self.assertEqual(sum(raw(i,j) for i in range(49)),6*r["sums"][j])
            self.assertEqual(raw(49,j),sum((i+1)*raw(i,j) for i in range(49)))
            self.assertEqual(raw(50,j),sum(raw(i,j) for i in range(0,49,2)))

    def test_span_independent_extrema_combinatorics(self):
        s1,s2,n=m.span_moments()
        r=self.report
        self.assertEqual(s1,m.U*250//7)
        self.assertEqual(s1,r["sums"][51])
        self.assertEqual(s2,r["raw_second_moment_upper"][51][0])
        self.assertEqual(sum(n),6*s1)
        self.assertEqual(n[0],n[-1])

    def test_nested_shape_onehot_zero_covariance_constraints(self):
        r=self.report
        for start,end in ((52,58),(58,69),(69,163)):
            self.assertEqual(sum(r["sums"][start:end]),m.U)
            for j in (0,1,49,50,51,52,58,80,162):
                self.assertEqual(sum(m.cov_numerator(r,i,j) for i in range(start,end)),0)
        for j in (0,1,49,50,51,69,162):
            self.assertEqual(sum(m.cov_numerator(r,i,j) for i in range(49)),0)

    def test_fail_closed_census_or_fixture_manipulation(self):
        fake=json.loads(json.dumps(self.baseline));fake["schema"]="wrong"
        with self.assertRaises(ValueError):m.read_census(self.tsv,fake)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"bad.tsv"
            path.write_bytes(self.tsv.read_bytes().replace(b"38347776\t1220298",b"38347776\t1220299",1))
            with self.assertRaises(ValueError):m.read_census(path,self.baseline)

    def test_output_reproducible_and_overwrite_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"result.json"
            cmd=["python","-m","research.lotto7x7.exact_covariance_v025","--census-tsv",str(self.tsv),"--fixture",str(FIX),"--out",str(p)]
            subprocess.run(cmd,check=True,cwd=ROOT,capture_output=True)
            self.assertEqual(json.loads(p.read_text("utf-8")),self.report)
            self.assertNotEqual(subprocess.run(cmd,cwd=ROOT,capture_output=True).returncode,0)

if __name__=="__main__":unittest.main()
