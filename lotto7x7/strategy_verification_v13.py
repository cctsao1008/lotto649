"""v1.3: integer-valid finite-pool M0 maximin upper bound + decision evidence.

The mathematical dual certificate is valid for the original 2,000-ticket
qualified pool, not all C(49,6) tickets. v1.2 must PASS before this module
uses its optionally exposed read-only audited pool.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


SCHEMA = 'lotto649.strategy_verification.v1.3'
CERT_SCALE = 100_000_000
BLOCKS = ('sum', 'log', 'joint', 'shape')


def validate_pool(pool: list[dict]) -> tuple[np.ndarray, list[tuple[int,...]], list[int]]:
    if not isinstance(pool, list) or len(pool)<4:
        raise ValueError('expected 4+ candidate rows')
    rows=[]; ranks=[]; parts=[]
    for c in pool:
        if set(c) != {'ticket','part','rank'}:
            raise ValueError('candidate schema mismatch')
        t=c['ticket']; r=c['rank']; part=c['part']
        if (not isinstance(t,list) or len(t)!=6 or
            any(type(v)!=int or not 1<=v<=49 for v in t) or t!=sorted(set(t))):
            raise ValueError('illegal six-number candidate')
        if not isinstance(r,list) or len(r)!=4 or any(type(v)!=int or v<0 for v in r):
            raise ValueError('invalid integer rank vector')
        if type(part)!=int or part<0:
            raise ValueError('invalid partition code')
        rows.append(tuple(t));ranks.append(r);parts.append(part)
    if len(set(rows))!=len(rows):
        raise ValueError('duplicate candidate in audited pool')
    return np.asarray(ranks,dtype=np.int64),rows,parts


def constraints(pool: list[dict]):
    """Max z, where z <= each block rank-sum of FOUR hard-feasible tickets."""
    ranks,tickets,parts=validate_pool(pool)
    n=len(pool)
    shape_keys=sorted(set(parts))
    part_to_row={p:49+i for i,p in enumerate(shape_keys)}
    block_start=49+len(shape_keys)
    m=block_start+4
    ai=[];aj=[];av=[]
    for j,(ticket,p) in enumerate(zip(tickets,parts)):
        for ball in ticket:
            ai.append(ball-1); aj.append(j); av.append(1.0)
        ai.append(part_to_row[p]);aj.append(j);av.append(1.0)
        for k in range(4):
            ai.append(block_start+k);aj.append(j);av.append(-float(ranks[j,k]))
    for k in range(4):
        ai.append(block_start+k);aj.append(n);av.append(1.0)
    a_ub=coo_matrix((av,(ai,aj)),shape=(m,n+1)).tocsr()
    b_ub=np.r_[np.ones(block_start,dtype=float),np.zeros(4)]
    a_eq=coo_matrix(([1.0]*n,([0]*n,list(range(n)))),shape=(1,n+1)).tocsr()
    zmax=4*int(ranks.max())
    if zmax<=0:
        raise ValueError('no positive rank potential')
    return ranks,tickets,parts,shape_keys,a_ub,b_ub,a_eq,zmax


def exact_dual_upper_bound(*,ranks:np.ndarray,tickets:list[tuple[int,...]],
                           parts:list[int],shape_keys:list[int],
                           multiplier_int:list[int],lambda_int:int,
                           zmax:int,scale:int=CERT_SCALE) -> dict:
    """Prove max-min upper bound by an EXACT integer Lagrangian calculation.

    For any nonnegative mu on the <= rows and any real lambda on sum(x)=4:
      max z <= sum(mu over number+partition constraints) + 4*lambda
        + sum_i max(0,sum_k mu_block[k]*rank[i,k]-sum_ball mu_ball
                         -mu_partition[i]-lambda)
        + zmax*max(0,1-sum_k mu_block[k]),
    where x in [0,1]^n, z in [0,zmax]. Integers encode all multipliers / scale.
    This bound applies to every integer feasible 4-ticket subset regardless
    of numerical LP solver tolerances (the LP only suggests multipliers).
    """
    n=len(tickets)
    m=49+len(shape_keys)+4
    if len(multiplier_int)!=m or any(type(v)!=int or v<0 for v in multiplier_int):
        raise ValueError('dual multipliers must be nonnegative integers')
    if type(lambda_int)!=int or type(zmax)!=int or type(scale)!=int or scale<=0 or zmax<0:
        raise ValueError('invalid exact certificate denominator or bounds')
    if ranks.shape!=(n,4) or len(parts)!=n:
        raise ValueError('rank/pool mismatch')
    pmap={p:49+i for i,p in enumerate(shape_keys)}
    b0=49+len(shape_keys)
    positive_mu=sum(multiplier_int[:b0])
    sum_mu_blocks=sum(multiplier_int[b0:])
    extra_x=0
    for i,(ticket,p) in enumerate(zip(tickets,parts)):
        coefficient=sum(multiplier_int[b0+k]*int(ranks[i,k]) for k in range(4))
        coefficient-=sum(multiplier_int[ball-1] for ball in ticket)
        coefficient-=multiplier_int[pmap[p]]+lambda_int
        extra_x+=max(0,coefficient)
    extra_z=zmax*max(0,scale-sum_mu_blocks)
    numerator=positive_mu+4*lambda_int+extra_x+extra_z
    upper=Fraction(numerator,scale)
    if upper<0:
        raise AssertionError('invalid negative upper bound for nonnegative objective')
    return {
        'exact_upper_bound_numerator':int(numerator),
        'exact_upper_bound_denominator':scale,
        'integer_upper_bound':numerator//scale,
        'upper_bound_float':float(upper),
        'sparse_dual_mu_integer_scaled':{str(j):v for j,v in enumerate(multiplier_int) if v},
        'dual_lambda_integer_scaled':lambda_int,
        'z_box_upper':zmax,
        'certificate': 'EXACT_INTEGER_LAGRANGIAN_UPPER_BOUND_FOR_FINITE_POOL',
        'proof_uses_only_hard_4_tickets_49_number_and_partition_and_4_block_constraints':True,
    }


def optimization_certificate(pool:list[dict], *, test_milp_seconds:float=0.) -> dict:
    ranks,tickets,parts,shape_keys,a_ub,b_ub,a_eq,zmax=constraints(pool)
    n=len(pool)
    c=np.zeros(n+1,dtype=float);c[n]=-1.
    lb=np.zeros(n+1,dtype=float)
    ub=np.r_[np.ones(n),float(zmax)]
    result=linprog(c,A_ub=a_ub,b_ub=b_ub,A_eq=a_eq,b_eq=np.array([4.]),
                   bounds=list(zip(lb,ub)),method='highs')
    if not result.success or result.x is None:
        raise RuntimeError(f'LP relaxation failed: {result.message}')
    mu=np.maximum(0.,-np.asarray(result.ineqlin.marginals,dtype=float))
    lam=-float(result.eqlin.marginals[0])
    multiplier=[max(0,int(round(float(x)*CERT_SCALE))) for x in mu]
    lambda_int=int(round(lam*CERT_SCALE))
    exact=exact_dual_upper_bound(ranks=ranks,tickets=tickets,parts=parts,
              shape_keys=shape_keys,multiplier_int=multiplier,lambda_int=lambda_int,
              zmax=zmax)
    exact.update({'lp_solution_bound_float':float(result.x[n]),
                  'pool_count':n,'distinct_shape_partitions':len(shape_keys),
                  'nonzero_dual_multipliers':sum(bool(x) for x in multiplier)})
    if test_milp_seconds>0:
        integral=np.r_[np.ones(n,dtype=int),0]
        lp_constraints=[LinearConstraint(a_ub,-np.inf,b_ub),
                        LinearConstraint(a_eq,[4.],[4.])]
        sol=milp(c,integrality=integral,bounds=Bounds(lb,ub),
                 constraints=lp_constraints,options={'time_limit':float(test_milp_seconds),
                  'mip_rel_gap':0.001,'disp':False})
        info={'status':int(sol.status),'message':str(sol.message),
              'time_limit_seconds':float(test_milp_seconds),
              'mip_gap':float(sol.mip_gap) if getattr(sol,'mip_gap',None) is not None else None}
        if sol.x is not None:
            choice=[i for i in range(n) if float(sol.x[i])>0.5]
            if len(choice)==4 and is_feasible(choice,tickets,parts):
                measured=ranks[choice].sum(axis=0).tolist()
                z=int(min(measured))
                if z>exact['integer_upper_bound']:
                    raise AssertionError('independent integer solution defeats exact dual certificate')
                info.update({'feasible':True,'main_maximin':z,
                             'rank_vector_sum':[int(x) for x in measured],
                             'tickets':[list(tickets[i]) for i in choice]})
            else:info['feasible']=False
        exact['mip_exploration']=info
    return exact


def is_feasible(indices:list[int],tickets:list[tuple[int,...]],parts:list[int])->bool:
    if len(indices)!=4 or len(set(indices))!=4:return False
    try:
        rows=[tickets[i] for i in indices]
        part_codes=[parts[i] for i in indices]
    except (IndexError,TypeError):return False
    return len(set(part_codes))==4 and len(set(v for t in rows for v in t))==24


def quality(rank_vector:list[int],style:str) -> tuple[int,...]:
    low=min(rank_vector)
    total=sum(rank_vector)
    if style=='balanced':return low,total,rank_vector[3],rank_vector[2]
    if style=='shape':return rank_vector[3],low,total,rank_vector[2]
    if style=='joint':return rank_vector[2],low,total,rank_vector[3]
    raise ValueError('invalid style')


def explain_selection(pool:list[dict],selection:list[list[int]],*,style:str='balanced') -> dict:
    ranks,tickets,parts=validate_pool(pool)
    table={t:i for i,t in enumerate(tickets)}
    try: chosen=[table[tuple(line)] for line in selection]
    except KeyError as e:raise ValueError('selected ticket absent from candidate pool') from e
    if not is_feasible(chosen,tickets,parts):raise ValueError('selected four tickets break hard constraints')
    sums=ranks[chosen].sum(axis=0).tolist()
    baseline=quality(sums,style)
    alternatives=[]
    for slot in range(4):
        remaining=[j for i,j in enumerate(chosen) if i!=slot]
        options=[]
        for candidate in range(len(pool)):
            if candidate==chosen[slot]:
                continue  # Report a genuine alternative, not the selected ticket.
            proposal=remaining+[candidate]
            if not is_feasible(proposal,tickets,parts):continue
            vec=ranks[proposal].sum(axis=0).tolist()
            options.append((quality(vec,style),candidate,vec))
        if not options:
            alternatives.append({'ticket_position':slot+1,'current_ticket':list(tickets[chosen[slot]]),
             'rank_contribution':[int(x) for x in ranks[chosen[slot]].tolist()],
             'feasible_one_ticket_replacements':0,'best_replacement_ticket':None,
             'best_replacement_quality':None,'best_replacement_rank_sum':None,
             'strict_improvement_exists':False})
            continue
        top_score=max(k for k,_,_ in options)
        best=min((j,v) for k,j,v in options if k==top_score)
        alternative=top_score>baseline
        alternatives.append({'ticket_position':slot+1,
              'current_ticket':list(tickets[chosen[slot]]),
              'rank_contribution':[int(x) for x in ranks[chosen[slot]].tolist()],
              'feasible_one_ticket_replacements':len(options),
              'best_replacement_ticket':list(tickets[best[0]]),
              'best_replacement_quality':list(top_score),
              'best_replacement_rank_sum':[int(v) for v in best[1]],
              'strict_improvement_exists':bool(alternative)})
    return {'criterion':style,'selected':selection,'rank_vector_sum':[int(x) for x in sums],
            'quality_tuple':list(baseline),
            'bottleneck_blocks':[BLOCKS[i] for i,v in enumerate(sums) if v==min(sums)],
            'one_swap_stationary':not any(x['strict_improvement_exists'] for x in alternatives),
            'each_ticket_and_best_alternative':alternatives}


def verify_m0_gap(*,csv_root:Path,bin_root:Path,manifest:Path,golden:Path,
                  mip_seconds:float=0.) -> dict:
    from .strategy_verification_v12 import verify_full_m0
    verified=verify_full_m0(csv_root=csv_root,bin_root=bin_root,manifest_path=manifest,
                            golden_path=golden,include_verified_pool=True)
    if verified['failed_checks'] or verified['decision']!='M0_BOUNDED_STRATEGY_REPLAY_PASS_GLOBAL_OPTIMUM_UNCERTIFIED':
        raise ValueError('v1.2 original independent audit is not PASS')
    pool=verified.pop('verified_pool')
    if len(pool)!=2000:raise ValueError('expected exactly 2,000 pre-validated candidates')
    pool_sha=sha256(json.dumps(pool,separators=(',',':'),sort_keys=True).encode('ascii')).hexdigest()
    cert=optimization_certificate(pool,test_milp_seconds=mip_seconds)
    explanations={}
    for style,name in [('balanced','primary'),('shape','shape_first'),('joint','joint_first')]:
        explanation=explain_selection(pool,verified['replayed'][name]['tickets'],style=style)
        if not explanation['one_swap_stationary']:
            raise AssertionError(name+' violates 1-swap stationarity')
        explanations[name]=explanation
    initial=explanations['primary']['rank_vector_sum']
    incumbent=min(initial)
    upper=cert['integer_upper_bound']
    if upper<incumbent:raise AssertionError('rigorous bound falls below feasible original objective')
    better=cert.get('mip_exploration',{}).get('main_maximin')
    if better is not None and better>incumbent:
        primary_status='ORIGINAL_HEURISTIC_STRICTLY_SUBOPTIMAL_IN_FIXED_POOL'
    elif upper==incumbent:
        primary_status='EXACT_MAXIMIN_OPTIMUM_CERTIFIED_IN_FIXED_POOL'
    else:primary_status='BOUNDED_GAP_UNRESOLVED_IN_FIXED_POOL'
    return {
        'schema':SCHEMA,'decision':'M0_STRATEGY_EXPLANATION_AND_GAP_AUDIT_PASS',
        'primary_maximin_status':primary_status,
        'scope':'EXACT original 2,000 candidate finite pool only; no full 6/49 portfolio optimum certificate',
        'source_verification':{'original_v12_checks':len(verified['checks']),
              'original_v12_decision':verified['decision'],
              'prefix_sha256':verified['source']['historical_prefix_sha256']},
        'pool_sha256':pool_sha,'pool_size':len(pool),
        'bounds':{'original_primary_main_maximin_lower':incumbent,
                  'upper':upper,
                  'absolute_gap_at_most':upper-incumbent,
                  'relative_gap_at_most_percent':100*(upper-incumbent)/upper if upper else 0,
                  'certificate':cert},
        'explanations':explanations,
        'limits':['Certified bound applies only to the frozen 2,000-candidate pool, not the full 13,983,816-ticket universe.',
                  'Original M0 multicriteria lexicographic tie-break is separately reported; upper bound addresses primary maximin only.',
                  'No modification to purchased tickets or old M0/P3f/P5A seals.',
                  'Strategy correctness and optimization quality do not establish next-draw predictive advantage.']
    }


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    parser.add_argument('--bin-root',type=Path,default=Path('data/bin'))
    parser.add_argument('--manifest',type=Path,default=Path('experiments/v028_m0_four_block_candidates_20261110.json'))
    parser.add_argument('--golden',type=Path,default=Path('tests/fixtures/v028_m0_original_ci_golden_v12.json'))
    parser.add_argument('--mip-seconds',type=float,default=0.,help='optional bounded feasible integer search; no guarantee of optimality')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.mip_seconds<0 or args.mip_seconds>120:parser.error('mip seconds must be 0..120')
    target=args.out.resolve()
    if not target.is_relative_to(Path('results').resolve()) or target.exists():
        parser.error('new report required under results/')
    result=verify_m0_gap(csv_root=args.csv_root,bin_root=args.bin_root,manifest=args.manifest,
                         golden=args.golden,mip_seconds=args.mip_seconds)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='ascii',newline='\n') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        handle.write('\n')
    q=result['bounds']
    print(f"{result['decision']}: original={q['original_primary_main_maximin_lower']}, "
          f"certified bounded upper={q['upper']}, gap <= {q['absolute_gap_at_most']}")
    print(f"PRIMARY: {result['primary_maximin_status']}")
    for name,detail in result['explanations'].items():
        print(f"{name.upper()}: rank_sum={detail['rank_vector_sum']}; bottleneck={detail['bottleneck_blocks']}; one_swap_stationary={detail['one_swap_stationary']}")
    print(f"REPORT: {target}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
