"""Exact complete 163x163 Lotto649 fair 6/49 population covariance.

A standalone C++ census emits only 94 L3 sufficient-statistic rows, which
are independently checked against the original v0.16 full-universe fixture.
All upper-triangle raw second moments and means are stored as Python ints:
Cov(i,j) = (U * M2(i,j) - S(i)*S(j)) / U**2, exactly.
No historical outcomes or future lottery numbers enter any calculation.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path

U=comb(49,6)
RULE="shape8-graph-partition-cardinal-diagonal-v1"
SCHEMA="lotto649.v025.full_163_exact_covariance.v1"

def l1_from_partition(partition:int)->int:
    sizes=[(partition>>(3*(5-i)))&7 for i in range(6)]
    if sum(sizes)!=6 or sizes!=sorted(sizes,reverse=True):
        raise ValueError("malformed v0.16 partition")
    return sum(bool(x) for x in sizes)

def read_census(path:Path,baseline:dict)->tuple[dict,str]:
    if (baseline.get("schema")!=RULE or baseline.get("complete") is not True
            or baseline.get("total")!=U
            or set(baseline.get("counts",{}))!={"component","partition","edge"}):
        raise ValueError("invalid original exact Shape fixture")
    for values in baseline["counts"].values():
        if sum(int(v) for v in values.values())!=U:
            raise ValueError("nonconserving reference Shape masses")
    raw=path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("invalid canonical L3 census line endings")
    lines=raw.decode("ascii").splitlines()
    if lines[0]!=f"#exact_covariance_v025.tsv.v1\t{U}\t94":
        raise ValueError("invalid full-universe header")
    data={}
    for line in lines[1:]:
        cols=line.split("\t")
        if len(cols)!=52 or any(not x.isdecimal() for x in cols):
            raise ValueError("invalid L3 sufficient statistic")
        code,count,span_sum,*numbers=map(int,cols)
        if (code in data or not count or not count*5<=span_sum<=count*48
                or sum(numbers)!=6*count):
            raise ValueError("inconsistent L3 class counts")
        data[code]={"count":count,"span_sum":span_sum,"number":numbers}
    expected={int(k):int(v) for k,v in baseline["counts"]["edge"].items()}
    if len(data)!=94 or list(data)!=sorted(data) or set(data)!=set(expected):
        raise ValueError("incomplete/unordered L3 taxonomy")
    if any(data[k]["count"]!=v for k,v in expected.items()):
        raise ValueError("census differs from independent v0.16 full-universe L3")
    parent={"component":{},"partition":{}}
    numbers=[0]*49
    for code,info in data.items():
        p=code>>10
        for level,key in (("component",l1_from_partition(p)),("partition",p)):
            totals=parent[level]; totals[key]=totals.get(key,0)+info["count"]
        for i,v in enumerate(info["number"]): numbers[i]+=v
    for level,totals in parent.items():
        if totals!={int(k):int(v) for k,v in baseline["counts"][level].items()}:
            raise ValueError("L3-to-"+level+" mapping differs from original v0.16")
    if any(v!=comb(48,5) for v in numbers):
        raise ValueError("number marginals differ from exact C(48,5)")
    return data,sha256(raw).hexdigest()

def span_moments()->tuple[int,int,list[int]]:
    """Independent exact min/max counting by C(max-min-1,4)."""
    total=total_sq=0
    number=[0]*49
    for lo in range(1,50):
        for hi in range(lo+5,50):
            gap=hi-lo;cases=comb(gap-1,4)
            total+=gap*cases
            total_sq+=gap*gap*cases
            for n in range(lo,hi+1):
                number[n-1]+=gap*(cases if n==lo or n==hi else comb(gap-2,3))
    return total,total_sq,number

def build_exact_moments(data:dict,census_hash:str,baseline:dict)->dict:
    c1,c2=comb(48,5),comb(47,4)
    components=sorted(int(k) for k in baseline["counts"]["component"])
    partitions=sorted(int(k) for k in baseline["counts"]["partition"])
    edges=sorted(data)
    names=[f"number_{n:02d}" for n in range(1,50)]+["sum","odd_count","span"]
    names += [f"component_{v}" for v in components]
    names += [f"partition_{v}" for v in partitions]
    names += [f"edge_{v}" for v in edges]
    dim=len(names)
    if dim!=163 or [len(components),len(partitions),len(edges)]!=[6,11,94]:
        raise ValueError("wrong exact feature registry dimensions")
    lookup={name:i for i,name in enumerate(names)}
    sums=[0]*dim
    second=[[0]*dim for _ in range(dim)]
    def add(i,j,x):
        if i>j:i,j=j,i
        second[i][j]+=x
    for i in range(49):sums[i]=c1
    sums[49]=U*150;sums[50]=c1*25
    s1,s2,range_number=span_moments()
    sums[51]=s1
    if s1*7!=U*250:raise AssertionError("exact mean span identity")
    for i in range(49):
        for j in range(i,49):add(i,j,c1 if i==j else c2)
        add(i,49,c2*1225+(c1-c2)*(i+1))
        add(i,50,c2*25+(c1-c2)*((i+1)%2))
        add(i,51,range_number[i])
    add(49,49,c2*1225**2+(c1-c2)*sum(k*k for k in range(1,50)))
    add(49,50,c2*1225*25+(c1-c2)*625)
    add(50,50,c2*25*25+(c1-c2)*25)
    add(49,51,sum((i+1)*v for i,v in enumerate(range_number)))
    add(50,51,sum(v for i,v in enumerate(range_number) if (i+1)%2))
    add(51,51,s2)
    observed_span=0
    for code,stat in data.items():
        count=stat["count"]
        observed_span+=stat["span_sum"]
        cls=[lookup[f"component_{l1_from_partition(code>>10)}"],
             lookup[f"partition_{code>>10}"],lookup[f"edge_{code}"]]
        for category in cls:
            sums[category]+=count
            for i,v in enumerate(stat["number"]):add(i,category,v)
            add(49,category,sum((i+1)*v for i,v in enumerate(stat["number"])))
            add(50,category,sum(v for i,v in enumerate(stat["number"]) if (i+1)%2))
            add(51,category,stat["span_sum"])
        for i in range(3):
            for j in range(i,3):add(cls[i],cls[j],count)
    if observed_span!=s1:
        raise AssertionError("exhaustive C++ spans disagree with independent min/max combinatorics")
    for name,i in lookup.items():
        if name.startswith(("component_","partition_","edge_")):
            level,code=name.rsplit("_",1)
            if sums[i]!=int(baseline["counts"][level][code]):
                raise AssertionError("exact category mean drift")
    return {
        "schema":SCHEMA,"universe":U,"feature_count":dim,
        "feature_registry":names,"sums":sums,
        "raw_second_moment_upper":[second[i][i:] for i in range(dim)],
        "covariance_exact_formula":"(universe * raw_second_moment[i,j] - sums[i]*sums[j]) / universe^2",
        "reference_taxonomy_rule":RULE,"census_tsv_sha256":census_hash,
        "sources":"full exact universe; independent frozen v0.16 L1/L2/L3 fixture; analytic combinatorial number and span moments",
        "research_decision":"EXACT_THEORETICAL_COVARIANCE_ONLY_NO_PREDICTIVE_CLAIM",
        "notes":[
          "All 163x163 covariance entries are exact rational population moments, not Monte Carlo estimates.",
          "The 163x163 matrix is rank deficient due to deterministic ticket-size, Sum/OddCount and nested Shape constraints.",
          "This produces a new exact reference; it does not silently replace legacy v0.25 finite-sample MC selection settings.",
          "No inference about prospective prediction, payout or the next winning combination is permitted."
        ]
    }

def cov_numerator(report:dict,i:int,j:int)->int:
    if i>j:i,j=j,i
    return (report["universe"]*report["raw_second_moment_upper"][i][j-i]
            -report["sums"][i]*report["sums"][j])

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census-tsv",type=Path,required=True)
    p.add_argument("--fixture",type=Path,default=Path("data/reference/shape_class_counts.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():p.error("write-once result path must not already exist")
    baseline=json.loads(a.fixture.read_text("utf-8"))
    data,digest=read_census(a.census_tsv,baseline)
    result=build_exact_moments(data,digest,baseline)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,ensure_ascii=True)
        f.write("\n")
    print(f"[PASS] universe={U} features=163 L1/L2/L3=6/11/94; exact rational covariance")
    print("[SHA256] census="+digest)
    print("[FILE]",a.out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
