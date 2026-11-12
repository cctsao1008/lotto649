"""v1.4: independent lexicographic M0 tie-break audit on the immutable pool.

The v1.3 Shape partition argument proves the primary maximin bound is 43549.
Any feasible portfolio attaining this bound MUST pick one maximum-Shape-rank
candidate from each of the four uniquely highest Shape partitions. With the
Shape sum fixed, rank-total is the second criterion and Joint sum the final
nonconstant criterion. Report exact-integer Lagrangian upper bounds for both
objectives and integer-verified HiGHS MIP incumbents without conflating
algorithmic execution, bounded optimality, or real-world prediction.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import floor
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, coo_matrix

from .strategy_verification_v12 import verify_full_m0
from .strategy_verification_v13 import (
    validate_pool, is_feasible, quality, shape_partition_bound,
)

SCHEMA = 'lotto649.strategy_verification.v1.4'
EXPECTED_POOL_SHA = '366af606a178173d159198fef784a779b8ed430d52babc89d526f1df575f216e'
SHAPE_BOUND = 43549
ORIGINAL_TOTAL = 189817
ORIGINAL_JOINT = 49829
SCALE = 1_000_000_000


def exact_primary_optimal_reduction(pool: list[dict], original: list[list[int]], *, expected_bound: int = SHAPE_BOUND, expected_total: int = ORIGINAL_TOTAL, expected_joint: int = ORIGINAL_JOINT) -> tuple[list[dict], dict]:
    """Exactly characterize every portfolio achieving the fixed-pool primary optimum.

    Requires a unique choice of the four highest partition classes by their
    maximum Shape rank (strict 4th-vs-5th cap separation). A candidate in a
    top class can appear in a primary-optimum portfolio only when it attains
    the class's maximum Shape rank. This is an EXACT necessary condition,
    not a heuristic candidate pruning.
    """
    ranks,tickets,parts = validate_pool(pool)
    caps = shape_partition_bound(pool)['all_partition_caps']
    if len(caps)<5 or caps[3]['max_shape_rank'] <= caps[4]['max_shape_rank']:
        raise ValueError('unique-top-four-partition proof does not apply')
    bound = sum(int(c['max_shape_rank']) for c in caps[:4])
    if bound != expected_bound:
        raise ValueError('original Shape upper bound has drifted')
    cap_by_part = {int(c['partition_code']):int(c['max_shape_rank']) for c in caps[:4]}
    selected = [p for p in pool if p['part'] in cap_by_part and int(p['rank'][3])==cap_by_part[p['part']]]
    groups = defaultdict(int)
    for p in selected:
        groups[int(p['part'])]+=1
    origin = {tuple(p['ticket']):i for i,p in enumerate(pool)}
    try:
        chosen = [origin[tuple(t)] for t in original]
    except KeyError as e:
        raise ValueError('original ticket missing from verified pool') from e
    if not is_feasible(chosen,tickets,parts):
        raise ValueError('original portfolio violates hard constraints')
    totals = ranks[chosen].sum(axis=0).tolist()
    if min(totals) != bound or totals[3]!=bound or sum(totals)!=expected_total or totals[2]!=expected_joint:
        raise ValueError('original frozen primary objective changed')
    if any(tuple(t) not in {tuple(x['ticket']) for x in selected} for t in original):
        raise AssertionError('original optimum not in mathematically necessary slice')
    return selected, {
        'frozen_primary_maximin':int(bound),
        'original_block_rank_sums':[int(x) for x in totals],
        'original_lexicographic_quality':list(quality(totals,'balanced')),
        'four_highest_partition_caps':[{'part':int(c['partition_code']), 'cap':int(c['max_shape_rank'])} for c in caps[:4]],
        'strict_partition_gap_to_fifth':int(caps[3]['max_shape_rank']-caps[4]['max_shape_rank']),
        'top_partition_candidate_counts':{str(k):v for k,v in sorted(groups.items())},
        'reduced_candidates':len(selected),
        'reduction_is_necessary_for_every_primary_optimum':True,
    }


def model_matrix(pool: list[dict], primary_bound: int = SHAPE_BOUND, total_floor: int | None = None):
    """Two-stage MILP: exactly one per winning Shape partition, 24 disjoint numbers.

    All candidate Shape ranks sum to the proved fixed primary maximum.
    Enforce other three block totals >= primary_bound. Stage two additionally
    enforces a certified maximum total-rank floor to maximize Joint rank.
    """
    ranks,tickets,parts=validate_pool(pool)
    groups=sorted(set(parts))
    if len(groups)!=4:raise ValueError('the reduced optimal slice must contain four partitions')
    shape_caps={p:{int(ranks[i,3]) for i,part in enumerate(parts) if part==p} for p in groups}
    if any(len(v)!=1 for v in shape_caps.values()) or sum(next(iter(v)) for v in shape_caps.values())!=primary_bound:
        raise ValueError('Shape cap is not constant and tight in reduced slice')
    n=len(pool)
    groups_i={part:j for j,part in enumerate(groups)}
    # 49 ball <= 1, three score lower bounds, optional second-stage total floor.
    row_bounds=49+3+(total_floor is not None)
    rows=[];cols=[];values=[]
    for i,(ticket,part) in enumerate(zip(tickets,parts)):
        for ball in ticket:
            rows.append(ball-1);cols.append(i);values.append(1.)
        for k in range(3):
            rows.append(49+k);cols.append(i);values.append(-float(ranks[i,k]))
        if total_floor is not None:
            rows.append(52);cols.append(i);values.append(-float(ranks[i].sum()))
    A_ub=coo_matrix((values,(rows,cols)),shape=(row_bounds,n)).tocsr()
    b_ub=np.array([1.]*49+[-float(primary_bound)]*3+([-float(total_floor)] if total_floor is not None else []))
    er=[];ec=[]
    for i,part in enumerate(parts):er.append(groups_i[part]);ec.append(i)
    A_eq=coo_matrix((np.ones(n),(er,ec)),shape=(4,n)).tocsr()
    return ranks,tickets,parts,groups,A_ub,b_ub,A_eq


def _exact_lagrangian_bound(ranks:np.ndarray,tickets:list[tuple[int,...]],parts:list[int],
                            groups:list[int],score:np.ndarray,thresholds:list[tuple[np.ndarray,int]],
                            multipliers_ub:list[int], multipliers_eq:list[int],scale:int=SCALE) -> dict:
    """Exactly valid upper bound: relax ball exclusivity, score floors and group equalities.

    For x in [0,1]^n, group equalities Gx=1, ball Nx<=1 and score-floor
    -R_j x<=-L_j, choose arbitrary mu>=0 and unrestricted nu:
    score*x <= sum(mu_ball)-sum(mu_floor*L)+sum(nu_group)
      + sum_i max(0,score_i - mu_ball(i) + sum(mu_floor*R_ji)-nu_group(i)).
    Integer scaled multipliers make the verification pure exact arithmetic.
    """
    n=len(tickets)
    if (len(multipliers_ub)!=49+len(thresholds) or len(multipliers_eq)!=len(groups)
        or any(type(x)!=int or x<0 for x in multipliers_ub)
        or any(type(x)!=int for x in multipliers_eq) or scale<1):
        raise ValueError('bad scaled dual certificate')
    group_to_i={p:i for i,p in enumerate(groups)}
    num=sum(multipliers_ub[:49])+sum(multipliers_eq)
    for j,(_,lower) in enumerate(thresholds):num-=multipliers_ub[49+j]*lower
    for i,(ticket,part) in enumerate(zip(tickets,parts)):
        term=int(score[i])*scale-sum(multipliers_ub[ball-1] for ball in ticket)-multipliers_eq[group_to_i[part]]
        for j,(vals,_) in enumerate(thresholds):
            term+=multipliers_ub[49+j]*int(vals[i])
        num+=max(0,term)
    return {'method':'EXACT_SCALED_INTEGER_LAGRANGIAN_BOUND',
            'integer_upper_bound':num//scale,
            'numerator':int(num),'denominator':int(scale),
            'multipliers_nonzero':sum(bool(x) for x in multipliers_ub+multipliers_eq),
            'numerically_suggested_exactly_integer_verified':True}


def exact_lp_dual_certificate(pool:list[dict], *, score_kind:str, primary_bound:int=SHAPE_BOUND,
                              total_floor:int|None=None) -> dict:
    ranks,tickets,parts,groups,A,b,E=model_matrix(pool,primary_bound,total_floor)
    score = ranks.sum(axis=1) if score_kind=='total' else ranks[:,2] if score_kind=='joint' else None
    if score is None:raise ValueError('unsupported objective')
    lp=linprog(-score.astype(float),A_ub=A,b_ub=b,A_eq=E,b_eq=np.ones(4),bounds=(0,1),method='highs')
    if not lp.success:raise RuntimeError('LP relaxation failed: '+str(lp.message))
    mu=[max(0,int(round(-v*SCALE))) for v in lp.ineqlin.marginals]
    nu=[int(round(-v*SCALE)) for v in lp.eqlin.marginals]
    thresholds=[(ranks[:,k],primary_bound) for k in range(3)]
    if total_floor is not None:thresholds.append((ranks.sum(axis=1),total_floor))
    result=_exact_lagrangian_bound(ranks,tickets,parts,groups,score,thresholds,mu,nu)
    result['lp_relaxation_float_upper']=-float(lp.fun)
    result['objective']=score_kind
    return result


def certified_mip_search(pool:list[dict], *, score_kind:str,primary_bound:int=SHAPE_BOUND,
                         total_floor:int|None=None, seconds:float=45.) -> dict:
    ranks,tickets,parts,groups,A,b,E=model_matrix(pool,primary_bound,total_floor)
    score=ranks.sum(axis=1) if score_kind=='total' else ranks[:,2] if score_kind=='joint' else None
    if score is None or not 1<=seconds<=180:raise ValueError('invalid objective or time limit')
    res=milp(-score.astype(float),integrality=np.ones(len(pool),dtype=np.int8),
             bounds=Bounds(np.zeros(len(pool)),np.ones(len(pool))),
             constraints=[LinearConstraint(A,-np.inf,b),LinearConstraint(E,np.ones(4),np.ones(4))],
             options={'time_limit':float(seconds),'mip_rel_gap':0.,'disp':False})
    info={'objective':score_kind,'solver_status':int(res.status),'solver_message':str(res.message),
          'seconds_limit':seconds,'solver_mip_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None,
          'solver_numeric_bound':float(-res.mip_dual_bound) if getattr(res,'mip_dual_bound',None) is not None else None}
    if res.x is None:
        info['feasible']=False
        return info
    chosen=[i for i,v in enumerate(res.x) if v>0.5]
    if not is_feasible(chosen,tickets,parts):raise AssertionError('numeric solver returned illegal tickets')
    totals=ranks[chosen].sum(axis=0)
    if totals.min()<primary_bound or (total_floor is not None and totals.sum()<total_floor):
        raise AssertionError('numeric solver violates integral hard score constraints')
    objective=int(totals.sum() if score_kind=='total' else totals[2])
    info.update({'feasible':True,'objective_integer':objective,
                 'tickets':[list(tickets[i]) for i in chosen],
                 'block_rank_sums':[int(v) for v in totals],
                 'lexicographic_quality':list(quality(totals.tolist(),'balanced')),
                 'solver_certifies_integer_optimum':int(res.status)==0 and
                     res.mip_gap is not None and float(res.mip_gap)<=1e-10})
    # Strictly check objective and best dual bound, not only a solver success code.
    if info['solver_certifies_integer_optimum']:
        ub=info['solver_numeric_bound']
        if ub is None or ub+1e-4<objective or ub-objective>1e-4:
            info['solver_certifies_integer_optimum']=False
    return info


def audit_lexicographic(pool:list[dict],original:list[list[int]], *,primary_bound:int=SHAPE_BOUND,
                        mip_seconds:float=45.,expected_total:int=ORIGINAL_TOTAL,expected_joint:int=ORIGINAL_JOINT) -> dict:
    reduced,reduction=exact_primary_optimal_reduction(pool,original,expected_bound=primary_bound,expected_total=expected_total,expected_joint=expected_joint)
    original_total=int(sum(reduction['original_block_rank_sums']))
    original_joint=int(reduction['original_block_rank_sums'][2])
    bound_total=exact_lp_dual_certificate(reduced,score_kind='total',primary_bound=primary_bound)
    stage_total=certified_mip_search(reduced,score_kind='total',primary_bound=primary_bound,seconds=mip_seconds)
    if stage_total.get('feasible') and stage_total['objective_integer']>bound_total['integer_upper_bound']:
        raise AssertionError('feasible secondary witness exceeds exact LP dual bound')
    complete_total=stage_total.get('solver_certifies_integer_optimum',False) or (
        stage_total.get('feasible') and stage_total['objective_integer']==bound_total['integer_upper_bound'])
    info={'reduction':reduction, 'original_total_rank_sum':original_total,'original_joint_rank_sum':original_joint,
          'secondary_total_rank':{'exact_upper_bound':bound_total,'solver':stage_total,
                                 'certified':bool(complete_total),
                                 'strict_improvement_found':bool(stage_total.get('feasible') and stage_total['objective_integer']>original_total)}}
    if not stage_total.get('feasible'):
        info['decision']='SECONDARY_SEARCH_NO_FEASIBLE_WITNESS'
        return info
    optimum_total=stage_total['objective_integer']
    if not complete_total:
        info['decision']='SECONDARY_GAP_UNRESOLVED_EXPLORATORY'
        return info
    # Since the third lexicographic metric is Shape (fixed by the primary optimum),
    # the remaining tie-break is Joint after rank-total is maximized.
    bound_joint=exact_lp_dual_certificate(reduced,score_kind='joint',primary_bound=primary_bound,total_floor=optimum_total)
    stage_joint=certified_mip_search(reduced,score_kind='joint',primary_bound=primary_bound,
                                    total_floor=optimum_total,seconds=mip_seconds)
    if stage_joint.get('feasible') and stage_joint['objective_integer']>bound_joint['integer_upper_bound']:
        raise AssertionError('feasible joint witness exceeds exact LP dual bound')
    complete_joint=stage_joint.get('solver_certifies_integer_optimum',False) or (
        stage_joint.get('feasible') and stage_joint['objective_integer']==bound_joint['integer_upper_bound'])
    info['tertiary_joint_rank']={'exact_upper_bound':bound_joint,'solver':stage_joint,'certified':bool(complete_joint)}
    if complete_joint and stage_joint.get('feasible'):
        witness=stage_joint['lexicographic_quality']
        prior=reduction['original_lexicographic_quality']
        info['decision']='ORIGINAL_FULL_LEXICOGRAPHIC_OPTIMUM_CERTIFIED' if witness<=prior else 'ORIGINAL_PRIMARY_MAXIMIN_OPTIMAL_SECONDARY_SUBOPTIMAL'
        info['complete_lexicographic_optimum_quality']=witness
    else:
        info['decision']='SECONDARY_OPTIMAL_BUT_JOINT_TIE_BREAK_UNRESOLVED'
    return info


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    p.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    p.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    p.add_argument('--mip-seconds',type=float,default=45.)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    target=args.out.resolve()
    if not target.is_relative_to(Path('results').resolve()) or target.exists():
        p.error('exclusive-create audit report required under results/')
    verified=verify_full_m0(csv_root=args.csv_root,bin_root=args.bin_root,
                            manifest_path=args.manifest,golden_path=args.golden,
                            include_verified_pool=True)
    if verified['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED' or verified['failed_checks']:
        raise AssertionError('original v1.2 verifier gate did not pass')
    pool=verified.pop('verified_pool')
    pool_digest=sha256(json.dumps(pool,separators=(',',':'),sort_keys=True).encode('ascii')).hexdigest()
    if pool_digest!=EXPECTED_POOL_SHA:
        raise ValueError('original bounded 2,000-candidate pool hash drifted')
    study=audit_lexicographic(pool,verified['replayed']['primary']['tickets'],mip_seconds=args.mip_seconds)
    output={'schema':SCHEMA,'decision':study['decision'],'historical_draws':2497,
            'frozen_pool_sha256':pool_digest,'v12_checks':len(verified['checks']),
            'source_sha256':verified['source']['historical_prefix_sha256'],
            'audit':study,
            'limitations':['Only original 2,000-candidate M0 pool and its fixed historical cutoff.',
                           'Original Primary portfolio, original pre-draw seals and purchased tickets were not modified.',
                           'Solver-based optimum certificates are subject to integer feasibility and solver tolerances unless exact LP certificate matches.',
                           'Strategy selection compliance is not evidence of increased next-draw probability.']}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='ascii',newline='\n') as fp:
        json.dump(output,fp,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False);fp.write('\n')
    z=study['secondary_total_rank']
    print('STRATEGY V14:',study['decision'])
    print('PRIMARY:',SHAPE_BOUND,'ORIGINAL SECONDARY:',study['original_total_rank_sum'])
    print('SECONDARY: candidate',z['solver'].get('objective_integer'), 'exact upper',z['exact_upper_bound']['integer_upper_bound'], 'MIP', z['solver']['solver_status'])
    if 'tertiary_joint_rank' in study:
        j=study['tertiary_joint_rank'];print('JOINT: candidate',j['solver'].get('objective_integer'), 'exact upper',j['exact_upper_bound']['integer_upper_bound'],'MIP',j['solver']['solver_status'])
    print('REPORT:',target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
