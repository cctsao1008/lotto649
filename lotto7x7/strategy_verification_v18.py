"""v1.8: independently certify v1.7 new-seed screened-pool lexicographic optima.

This only evaluates the *frozen finite candidate pools*, not real-world draw
predictability. The v1.5 C++ enumerator performs exact integer enumeration after
strictly proved lossless Shape-class reduction. v1.7 score reference is frozen.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v13 import shape_partition_bound, validate_pool, is_feasible, quality
from .strategy_verification_v15 import compile_enumerator, enumerate_exact, SOURCE_CPP
from .strategy_verification_v16 import canonical_sha, portfolio_audit
from .strategy_verification_v17 import sampled_unique, score_uniform_stream, screen_pool, greedy_multistart

SCHEMA = 'lotto649.strategy_verification.v1.8.new_seed_exact.v1'
REFERENCE_RANK_AXES_SHA = '18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b'
SOURCE_PREFIX_SHA = 'd865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8'
EXPECTED = (
    # (seed, raw sample count, screening policy, exact v1.7 pool SHA256)
    (649, 80000, 'ranked', '632cbace8abe991e6b61588c908b2d9b6f1c45f75e3647f852d4dff261cd7106'),
    (649, 80000, 'shape_round_robin', '4c4589564c9adc2ed75f0e8e1b9a795813de7350847d68f39a67b364c7d9b02e'),
    (20261111, 160000, 'ranked', 'a54b38bc6513494dc00ec0e3b077427889ab575c2912708bccfdd06cd87552fe'),
    (20261112, 160000, 'ranked', 'ba04bb5d1dab97dc58474737f1d23c8c8f40d91257e1487e89ffa49a09b41d50'),
)
EXPECTED_V17_WITNESS_QUALITY = (43549,195617,43549,54184)


def reduce_primary_optimal_candidates(pool: list[dict], witness: list[list[int]]):
    """PROOF: maximin<=ShapeSum<=sum of top four distinct Shape class caps.

    If all four caps are strictly above the fifth, any portfolio attaining that
    value MUST select exactly one cap-attaining ticket in each top class. This
    reduction retains every possible primary-optimal portfolio, so exhaustive
    enumeration of these candidates proves full lexicographic optimality.
    """
    ranks, tickets, parts = validate_pool(pool)
    inc=portfolio_audit(pool,witness)
    maxima=shape_partition_bound(pool)['all_partition_caps']
    if len(maxima)<5 or maxima[3]['max_shape_rank']<=maxima[4]['max_shape_rank']:
        raise ValueError('FAIL_CLOSED_SHAPE_PARTITION_CAP_TIE_OR_INSUFFICIENT_CLASSES')
    caps={int(x['partition_code']):int(x['max_shape_rank']) for x in maxima[:4]}
    upper=sum(caps.values())
    if inc['min_rank']!=upper or inc['rank_sums'][3]!=upper:
        raise ValueError('FAIL_CLOSED_NO_PRIMARY_OPTIMAL_WITNESS')
    reduced=[r for r in pool if r['part'] in caps and r['rank'][3]==caps[r['part']]]
    # Crosscheck that the selected optimal witness actually lies in this slice.
    ref={tuple(r['ticket']) for r in reduced}
    if not all(tuple(t) in ref for t in witness):
        raise AssertionError('primary-optimal witness omitted by supposedly necessary slice')
    group_counts={str(k):sum(x['part']==k for x in reduced) for k in sorted(caps)}
    if any(n<1 for n in group_counts.values()):
        raise AssertionError('empty required partition')
    return reduced, {'primary_upper':upper,
                     'strict_fourth_vs_fifth_cap_gap':maxima[3]['max_shape_rank']-maxima[4]['max_shape_rank'],
                     'top_shape_caps':{str(k):v for k,v in sorted(caps.items())},
                     'primary_optimal_reduced_count':len(reduced),
                     'per_partition_count':group_counts,
                     'reduced_sha256':canonical_sha(reduced),
                     'proof':'Any four-line portfolio has min(block ranks)<=ShapeSum<=top-four class caps. Equality requires one rank-cap ticket in each of the uniquely top four classes.'}


def certify_pool(pool:list[dict], *, expected_sha:str, executable:Path,
                 restarts:int=6,timeout:int=240) -> dict:
    """Fully exact independent re-evaluation with original v1.7 scorer fixed."""
    if len(pool)!=2000 or canonical_sha(pool)!=expected_sha:
        raise ValueError('FAIL_CLOSED_V17_SCREENED_POOL_SHA_DRIFT')
    observed=greedy_multistart(pool,restarts=restarts)
    greedy_q=tuple(observed['quality'])
    if greedy_q!=EXPECTED_V17_WITNESS_QUALITY:
        raise AssertionError('FAIL_CLOSED_V17_REPLAY_WITNESS_QUALITY_DRIFT')
    reduced,proof=reduce_primary_optimal_candidates(pool,observed['tickets'])
    enumerated=enumerate_exact(reduced,primary_bound=proof['primary_upper'],
                               executable=executable,timeout=timeout)
    exact_q=(proof['primary_upper'],enumerated['best_total_rank'],
             proof['primary_upper'],enumerated['best_joint_rank'])
    if exact_q<greedy_q:
        raise AssertionError('Exhaustive optimum below feasible independently validated witness')
    # Verify entire independent exact witness against ORIGINAL unreduced pool.
    final=portfolio_audit(pool,enumerated['selected_tickets'])
    if tuple(final['quality'])!=exact_q:
        raise AssertionError('C++ exact witness does not agree with Python pool scorer')
    if enumerated['tied_best_lexicographic_portfolios']<1:
        raise AssertionError('invalid exact optimum tie count')
    return {'pool_sha256':expected_sha,
            'greedy_quality':list(greedy_q),
            'exhaustive_quality':list(exact_q),
            'greedy_was_optimal':greedy_q==exact_q,
            'difference_exact_minus_greedy':[int(b-a) for a,b in zip(greedy_q,exact_q)],
            'certification':'EXACT_FINITE_2000_POOL_LEXICOGRAPHIC_CERTIFIED',
            'lossless_reduction':proof,
            'exhaustive_counts':{
                k:enumerated[k] for k in ('cartesian_count','disjoint_four_portfolios',
                 'primary_feasible_portfolios','tied_best_lexicographic_portfolios','partition_sizes')},
            'exact_best_candidate_ids':enumerated['selected_candidate_ids'],
            'exact_best_witness':enumerated['selected_tickets'],
            'exact_rank_sums':enumerated['best_rank_sums'],
            'enumerator_input_sha256':enumerated['enumerator_input_sha256']}


def verify_new_seed_exact(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                          cxx:str='g++',cases:tuple=EXPECTED,timeout:int=240) -> dict:
    if len(cases)<1 or len(set((s,n,p) for s,n,p,_ in cases))!=len(cases):
        raise ValueError('duplicate or empty exact-case schedule')
    verified=verify_full_m0(csv_root=csv_root,bin_root=bin_root,
         manifest_path=manifest,golden_path=golden,
         include_qualified_candidates=True,include_frozen_score_reference=True)
    if verified['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or \
       len(verified['checks'])!=152 or verified['failed_checks'] or \
       verified['source']['historical_prefix_sha256']!=SOURCE_PREFIX_SHA:
        raise AssertionError('independent source provenance or original scoring failed')
    frozen=verified.pop('frozen_score_reference')
    original=verified.pop('qualified_candidates')
    original_scores=score_uniform_stream([tuple(c['ticket']) for c in original],frozen)[0]
    if {(tuple(x['ticket']),x['part'],tuple(x['rank'])) for x in original_scores} != \
       {(tuple(x['ticket']),x['part'],tuple(x['rank'])) for x in original}:
        raise AssertionError('FAIL_CLOSED_REFERENCE_RESCORING_DRIFT')
    axes_sha=sha256(json.dumps(frozen['sorted_qualified_gains'],separators=(',',':')).encode('ascii')).hexdigest()
    if axes_sha!=REFERENCE_RANK_AXES_SHA:raise AssertionError('reference axes sha mismatch')
    inputs={}
    for seed,n,policy,sha in cases:
        inputs.setdefault(seed,set()).add(n)
    pool_map={}
    for seed,sizes in inputs.items():
        stream,checkpoints=sampled_unique(seed,tuple(sorted(sizes)))
        scored,_=score_uniform_stream(stream,frozen)
        for n in sizes:
            subset=[x for x in scored if x['appearance']<n]
            if len(subset)<2000:raise ValueError('not enough positive candidates')
            for s,count,policy,sha in cases:
                if s==seed and count==n:
                    pool_map[(s,n,policy)]=screen_pool(subset,policy=policy,cap=2000,seed=s,sample_size=n)
    reports=[]
    with TemporaryDirectory(prefix='lotto649_v18_') as tmp:
        binary=Path(tmp)/'exact_int'
        compile_enumerator(cxx,binary,source=SOURCE_CPP)
        for seed,n,policy,sha in cases:
            row=certify_pool(pool_map[(seed,n,policy)],expected_sha=sha,
                             executable=binary,timeout=timeout)
            row.update({'seed':seed,'raw_sample_size':n,'policy':policy})
            reports.append(row)
    return {'schema':SCHEMA,
            'decision':'NEW_SEED_2000_POOL_EXACT_CERTIFICATES_COMPLETE',
            'source_gate':{'v12_checks':len(verified['checks']),
                'historical_prefix_sha256':SOURCE_PREFIX_SHA,
                'frozen_reference_candidates':len(original),
                'reference_rank_axes_sha256':axes_sha,
                'reference_rescoring':'PASS_ALL_13704_FOUR_RANK_VECTORS',
                'cpp_source_sha256':sha256(SOURCE_CPP.read_bytes()).hexdigest()},
            'cases':reports,
            'limitations':[
                'Exhaustive proof is only for each SHA-frozen 2000-screened candidate pool.',
                'A pool-optimal score is not a certificate across all newly qualified candidates or all possible 6/49 tickets.',
                'Scoring and ranks remain frozen to the original historical cutoff and empirical 13704-item reference sample.',
                'The C++ enumerator uses exact integers; mathematical reduction requires strict top-four partition maxima.',
                'No next-draw or prize-probability advantage is established.',
            ]}


def main(argv:list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--cxx',default='g++')
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    dest=args.out.resolve()
    if not dest.is_relative_to(Path('results').resolve()) or dest.exists():
        p.error('exclusive-create report path under results required')
    report=verify_new_seed_exact(csv_root=args.csv_root,bin_root=args.bin_root,
       manifest=args.manifest,golden=args.golden,cxx=args.cxx,timeout=args.timeout)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='ascii',newline='\n') as fp:
        json.dump(report,fp,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False);fp.write('\n')
    print('V18',report['decision'])
    for row in report['cases']:
        print('CASE',row['seed'],row['raw_sample_size'],row['policy'],
            'GREEDY',row['greedy_quality'],'EXACT',row['exhaustive_quality'],
            'PRIMARY_ELIGIBLE',row['exhaustive_counts']['primary_feasible_portfolios'],
            'TIES',row['exhaustive_counts']['tied_best_lexicographic_portfolios'],
            'STATUS','GREEDY_OPTIMAL' if row['greedy_was_optimal'] else 'GREEDY_SUBOPTIMAL')
    print('REPORT',dest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
