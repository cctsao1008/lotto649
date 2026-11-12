"""Exact full original 13,704-qualified M0 candidate-pool lexicographic audit.

If the entire qualified pool has the same exact optimum as the original 2,000
subset, nesting proves every intermediate pool has that same optimum, too.
This is a deterministic finite-sample strategy certificate, NOT prediction.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v14 import exact_primary_optimal_reduction
from .strategy_verification_v15 import compile_enumerator, enumerate_exact, SOURCE_CPP
from .strategy_verification_v16 import (
    SCHEMA as SCALING_SCHEMA, nested_pools, canonical_sha, ORIGINAL_POOL_SHA,
    ORIGINAL_SAMPLE_SHA, ORIGINAL_PREFIX_SHA, V15_REFERENCE, QUALIFIED_COUNT,
    BASELINE_MAXIMIN, BASELINE_TOTAL, BASELINE_JOINT, portfolio_audit,
)

SCHEMA = 'lotto649.strategy_verification.v1.6.exact_full_qualified'


def nested_sandwich_certificate(subset_quality: tuple[int,...],
                                full_quality: tuple[int,...]) -> dict:
    """If subset⊆B⊆full and opt(subset)=opt(full), opt(B) equals both.

    Qualities are lexicographic integer tuples. Caller separately proves
    subset inclusion and exactly enumerates full candidate pool optimum.
    """
    if len(subset_quality)!=4 or len(full_quality)!=4 or any(type(x)!=int for x in subset_quality+full_quality):
        raise ValueError('four exact integer objective components required')
    if full_quality < subset_quality:
        raise AssertionError('full superset cannot have worse optimum than subset')
    return {'subset_quality':list(subset_quality),'full_quality':list(full_quality),
            'all_nested_intermediate_optima_proven_equal':full_quality==subset_quality,
            'proof':'max over subset <= max over nested intermediate <= max over full; equality of endpoints proves middle equality'}


def verify_full_qualified(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                          cxx:str='g++',timeout:int=360) -> dict:
    if type(timeout) is not int or not 30 <= timeout <= 600:
        raise ValueError('reasonable bounded exhaustive enumeration timeout required')
    v12=verify_full_m0(csv_root=csv_root,bin_root=bin_root,
       manifest_path=manifest,golden_path=golden,
       include_verified_pool=True, include_qualified_candidates=True)
    if v12['failed_checks'] or v12['decision'] != 'M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or len(v12['checks']) !=152:
        raise AssertionError('original independent 152 checks did not pass')
    original=v12.pop('verified_pool'); qualified=v12.pop('qualified_candidates')
    if (len(qualified)!=QUALIFIED_COUNT or len(original)!=2000
        or canonical_sha(original)!=ORIGINAL_POOL_SHA
        or v12['source']['historical_prefix_sha256']!=ORIGINAL_PREFIX_SHA
        or v12['sample']['sha256']!=ORIGINAL_SAMPLE_SHA):
        raise AssertionError('historical/qualified candidate source changed')
    stages=nested_pools(original,qualified,(2000,QUALIFIED_COUNT))
    full=stages[-1]['pool']
    original_primary=v12['replayed']['primary']['tickets']
    baseline=portfolio_audit(original,[list(x) for x in V15_REFERENCE])
    if baseline['quality']!=[BASELINE_MAXIMIN,BASELINE_TOTAL,BASELINE_MAXIMIN,BASELINE_JOINT]:
        raise AssertionError('v1.5 exact original baseline drift')
    reduced,proof=exact_primary_optimal_reduction(full,original_primary)
    with TemporaryDirectory(prefix='lotto649_v16_full_') as directory:
        exe=Path(directory)/'enumerator'
        compile_enumerator(cxx,exe)
        result=enumerate_exact(reduced,primary_bound=BASELINE_MAXIMIN,executable=exe,timeout=timeout)
    qual=[int(result['primary_bound']),int(result['best_total_rank']),BASELINE_MAXIMIN,int(result['best_joint_rank'])]
    sandwich=nested_sandwich_certificate(tuple(baseline['quality']),tuple(qual))
    out={
      'schema':SCHEMA,
      'decision':'EXACT_FULL_QUALIFIED_OPTIMUM_MATCHES_ORIGINAL_2K' if sandwich['all_nested_intermediate_optima_proven_equal'] else 'EXACT_FULL_QUALIFIED_OPTIMUM_IMPROVES_BEYOND_ORIGINAL_2K',
      'source_gate':{'original_v12_checks':len(v12['checks']), 'draws':v12['source']['historical_prefix_draws'],
                     'historical_prefix_sha256':ORIGINAL_PREFIX_SHA,'original_sample_sha256':ORIGINAL_SAMPLE_SHA,
                     'original_2000_sha256':ORIGINAL_POOL_SHA,'all_qualified':len(qualified),
                     'full_qualified_pool_sha256':stages[-1]['pool_sha256'],
                     'reduced_pool_sha256':canonical_sha(reduced),
                     'standalone_cpp_source_sha256':sha256(SOURCE_CPP.read_bytes()).hexdigest()},
      'primary_optimum_reduction':proof,
      'exact_full_qualified_enumeration':result,
      'nested_sandwich_proof':sandwich,
      'research_only_no_new_bets':True,
      'scope_limits':['The 13,704 candidates were originally sampled and qualified; the other legal lottery combinations are not covered.',
                      'The four-objective ranking reference is frozen at the original qualified 13,704 sample.',
                      'End-to-end source gate uses historical results only through the original 2026-11-06 cutoff.',
                      'Computational exhaustive proof is contingent on correctly implemented tested C++ integer enumerator.',
                      'No prediction, future fairness, or monetary advantage follows from this certificate.',
                      'No paid tickets or original M0/P3f/P5A sealed files changed.']}
    return out


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    parser.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    parser.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    parser.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    parser.add_argument('--cxx',default='g++')
    parser.add_argument('--timeout',type=int,default=360)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    out=args.out.resolve()
    if not out.is_relative_to(Path('results').resolve()) or out.exists():
        parser.error('fresh write-once report under results/ required')
    result=verify_full_qualified(csv_root=args.csv_root,bin_root=args.bin_root,
            manifest=args.manifest,golden=args.golden,cxx=args.cxx,timeout=args.timeout)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='ascii',newline='\n') as f:
        json.dump(result,f,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        f.write('\n')
    proof=result['nested_sandwich_proof']
    e=result['exact_full_qualified_enumeration']
    print('V16 EXACT:',result['decision'])
    print('FULL REDUCED:',e['candidates'],'CARTESIAN:',e['cartesian_count'],'LEGAL:',e['primary_feasible_portfolios'])
    print('FULL QUALITY:',proof['full_quality'],'BASELINE:',proof['subset_quality'])
    print('REPORT:',out)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
