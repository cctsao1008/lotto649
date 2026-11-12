"""v1.5 independent exact-integer exhaustive audit of frozen v0.28 M0.

First prove the v1.4 lossless reduction from 2,000 original candidates to
893 necessary primary-optimal candidates, then enumerate every four-partition
combination under all original six-ball exclusivity and rank constraints.

The C++17 helper is a standalone exact integer enumerator, NOT a solver call,
and small-pool results are cross-checked against an independently written
Python reference. No modification of any historical input or frozen ticket.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v13 import validate_pool, is_feasible, quality
from .strategy_verification_v14 import (
    EXPECTED_POOL_SHA, SHAPE_BOUND, ORIGINAL_TOTAL, ORIGINAL_JOINT,
    exact_primary_optimal_reduction,
)

SCHEMA = "lotto649.strategy_verification.v1.5"
SOURCE_CPP = Path(__file__).resolve().parents[1] / "tools" / "lexicographic_enumerator_v15.cpp"
EXPECTED_REDUCED_COUNT = 893
KNOWN_NUMERIC_OPTIMUM_TOTAL = 192715
KNOWN_NUMERIC_OPTIMUM_JOINT = 52895


def input_stream(pool: list[dict], *, primary_bound: int) -> str:
    """Strictly validate and canonically serialize all integer candidate fields."""
    validate_pool(pool)
    if type(primary_bound) is not int or primary_bound < 0:
        raise ValueError("invalid primary bound")
    parts: dict[int,int] = {}
    for row in pool:
        p = row['part']; shape = row['rank'][3]
        if p in parts and parts[p] != shape:
            raise ValueError("reduced Shape partition is not constant")
        parts[p] = shape
    if len(parts) != 4 or sum(parts.values()) != primary_bound:
        raise ValueError("four Shape class maxima must sum to proven primary bound")
    lines = [f"{len(pool)} {primary_bound}"]
    for i, row in enumerate(pool):
        lines.append(' '.join(map(str,[i,row['part'],*row['ticket'],*row['rank']])))
    return '\n'.join(lines)+'\n'


def compile_enumerator(cxx: str, dest: Path, *, source: Path = SOURCE_CPP) -> None:
    compiler = shutil.which(cxx) if not Path(cxx).is_file() else cxx
    if not compiler:
        raise RuntimeError(f"C++17 compiler not found: {cxx}")
    cmd = [str(compiler),'-std=c++17','-O3','-Wall','-Wextra','-Werror',
           '-o',str(dest),str(source)]
    run = subprocess.run(cmd,capture_output=True,text=True,timeout=90,check=False)
    if run.returncode:
        raise RuntimeError(f"C++17 compilation failed: {run.stderr[-2000:]}")


def enumerate_exact(pool: list[dict], *, primary_bound: int, executable: Path,
                    timeout: int = 240) -> dict:
    raw = input_stream(pool,primary_bound=primary_bound)
    proc = subprocess.run([str(executable)],input=raw,text=True,
                          capture_output=True,timeout=timeout,check=False)
    if proc.returncode:
        raise RuntimeError(f"C++ exhaustive audit failed: {proc.stderr[-2000:]}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("C++ audit did not produce one JSON object") from exc
    if result['schema']!='lotto649.v15.exhaustive_int.v1' or result['candidates']!=len(pool):
        raise AssertionError("auditor identity or candidate count mismatch")
    if result['primary_bound']!=primary_bound or result['best_rank_sums'][3]!=primary_bound:
        raise AssertionError("auditor primary shape bound mismatch")
    if result['cartesian_count']!=__import__('math').prod(result['partition_sizes']):
        raise AssertionError("auditor Cartesian-product count mismatch")
    if not (0 < result['primary_feasible_portfolios'] <= result['disjoint_four_portfolios'] <= result['cartesian_count']):
        raise AssertionError("auditor enumeration counts inconsistent")
    ids = result['selected_candidate_ids']
    if type(ids) is not list or len(ids)!=4 or len(set(ids))!=4 or any(type(x)!=int or not 0<=x<len(pool) for x in ids):
        raise AssertionError("auditor witness invalid indices")
    ranks,tickets,parts=validate_pool(pool)
    if not is_feasible(ids,tickets,parts):
        raise AssertionError("auditor witness violates official hard constraints")
    totals=[sum(int(ranks[i,k]) for i in ids) for k in range(4)]
    if (totals != result['best_rank_sums'] or min(totals)!=primary_bound or
        sum(totals)!=result['best_total_rank'] or totals[2]!=result['best_joint_rank']):
        raise AssertionError("auditor output incorrect against Python integer recomputation")
    result['selected_tickets']=[list(tickets[i]) for i in ids]
    result['enumerator_input_sha256']=sha256(raw.encode('ascii')).hexdigest()
    return result


def python_reference(pool: list[dict], *, primary_bound: int) -> dict:
    """Independent tiny-pool oracle. Not intended for 893 real candidates."""
    input_stream(pool,primary_bound=primary_bound)
    ranks,tickets,parts=validate_pool(pool)
    groups=[[i for i,p in enumerate(parts) if p==part] for part in sorted(set(parts))]
    best=(-1,-1); witnesses=[]; considered=0
    for ids in product(*groups):
        if not is_feasible(list(ids),tickets,parts):
            continue
        totals=[sum(int(ranks[i,k]) for i in ids) for k in range(4)]
        if min(totals)<primary_bound:
            continue
        considered+=1
        key=(sum(totals),totals[2])
        if key>best:
            best=key;witnesses=[tuple(ids)]
        elif key==best:
            witnesses.append(tuple(ids))
    if not witnesses:
        raise ValueError('no feasible primary-optimal portfolio')
    return {'best_total_rank':best[0], 'best_joint_rank':best[1],
            'primary_feasible_portfolios':considered,
            'tied_best_lexicographic_portfolios':len(witnesses),
            'selected_candidate_ids':list(min(witnesses))}


def verify_exhaustive_m0(*, csv_root:Path,bin_root:Path,manifest_path:Path,
                         golden_path:Path,cxx:str='g++') -> dict:
    audited = verify_full_m0(csv_root=csv_root,bin_root=bin_root,
        manifest_path=manifest_path,golden_path=golden_path,include_verified_pool=True)
    if audited['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or audited['failed_checks']:
        raise AssertionError("v1.2 frozen M0 source and score gate did not pass")
    pool=audited.pop('verified_pool')
    digest=sha256(json.dumps(pool,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()
    if digest!=EXPECTED_POOL_SHA:raise AssertionError("M0 fixed 2000-candidate pool SHA drift")
    original=audited['replayed']['primary']['tickets']
    reduced,reduction=exact_primary_optimal_reduction(pool,original)
    if len(reduced)!=EXPECTED_REDUCED_COUNT:raise AssertionError("reduced primary-optimal slice drift")
    reduced_digest=sha256(json.dumps(reduced,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()
    with TemporaryDirectory(prefix='lotto649_v15_') as directory:
        executable=Path(directory)/'exhaustive_v15'
        compile_enumerator(cxx,executable)
        result=enumerate_exact(reduced,primary_bound=SHAPE_BOUND,executable=executable)
    totals=result['best_rank_sums']
    if result['best_total_rank']<KNOWN_NUMERIC_OPTIMUM_TOTAL or (result['best_total_rank']==KNOWN_NUMERIC_OPTIMUM_TOTAL and result['best_joint_rank']<KNOWN_NUMERIC_OPTIMUM_JOINT):
        raise AssertionError("exact enumeration found less than the independently verified v1.4 feasible MIP witness")
    improved=result['best_total_rank']>ORIGINAL_TOTAL or (result['best_total_rank']==ORIGINAL_TOTAL and result['best_joint_rank']>ORIGINAL_JOINT)
    return {
        'schema':SCHEMA,
        'decision':('EXACT_INTEGER_FINITE_POOL_LEXICOGRAPHIC_OPTIMUM_CONFIRMED'
                    if result['best_total_rank']==KNOWN_NUMERIC_OPTIMUM_TOTAL and result['best_joint_rank']==KNOWN_NUMERIC_OPTIMUM_JOINT
                    else 'EXACT_INTEGER_FINITE_POOL_LEXICOGRAPHIC_OPTIMUM_DIFFERS_FROM_V14_MIP'),
        'original_primary_is_full_lexicographic_optimum':not improved,
        'original_primary_secondary_suboptimal':improved,
        'gate':{'v12_check_count':len(audited['checks']), 'draws':audited['source']['historical_prefix_draws'],
                'historical_prefix_sha256':audited['source']['historical_prefix_sha256'],
                'frozen_pool_sha256':digest, 'reduced_pool_sha256':reduced_digest,
                'cpp_source_sha256':sha256(SOURCE_CPP.read_bytes()).hexdigest()},
        'original':{'tickets':original,'rank_sums':reduction['original_block_rank_sums'],
                    'maximin':SHAPE_BOUND,'total_rank':ORIGINAL_TOTAL,'joint_rank':ORIGINAL_JOINT},
        'proven_reduction':reduction,
        'exact_exhaustive_evidence':result,
        'limitations':[
            'Exhaustiveness is relative to the original 2000 qualified candidate pool only.',
            'Independent C++ enumerator exhausts all four-class portfolios with integer arithmetic and no MIP.',
            'A reproducible computational proof depends on the enumerator, input hashes, compiler and test correctness.',
            'The original frozen heuristic followed its documented procedure even though its final secondary score was suboptimal.',
            'No global optimum among 80000 sampled or 13983816 possible individual tickets is claimed.',
            'No forecast or winning-probability advantage follows from an improved rank tuple.',
            'No source data, frozen M0, purchased bets, or P3f/P5A seals were modified.',
        ],
    }


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--cxx',default='g++')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    dest=args.out.resolve()
    if not dest.is_relative_to(Path('results').resolve()) or dest.exists():
        p.error('new write-once v1.5 audit file required under results/')
    report=verify_exhaustive_m0(csv_root=args.csv_root,bin_root=args.bin_root,
        manifest_path=args.manifest,golden_path=args.golden,cxx=args.cxx)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='ascii',newline='\n') as handle:
        json.dump(report,handle,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        handle.write('\n')
    found=report['exact_exhaustive_evidence']
    print(f"STRATEGY V15: {report['decision']}")
    print(f"EXACT TOTAL: {found['best_total_rank']} JOINT: {found['best_joint_rank']} M0 ORIGINAL TOTAL: {ORIGINAL_TOTAL}")
    print(f"SEARCH: cartesian={found['cartesian_count']} disjoint={found['disjoint_four_portfolios']} primary_feasible={found['primary_feasible_portfolios']} ties={found['tied_best_lexicographic_portfolios']}")
    print(f"RESEARCH ONLY TICKETS: {found['selected_tickets']}")
    print('REPORT:',dest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
