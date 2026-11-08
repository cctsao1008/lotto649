"""v0.21 — uniform candidate selection, exact portfolio coverage, and honest shape-scoring comparison.

The only *inference* in the experimental arm is the frozen P1 L3
Dirichlet class ratio used in v0.17. Fair six-number ticket odds remain
1/C(49,6). Exact coverage optimization changes portfolio-level threshold
coverage, not full six-number jackpot probability. No future results used.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .coverage_v06 import exact_coverage, normalized_bets
from .shape_prospective_v019 import check_freeze, file_sha256, verify_seal
from .shape_taxonomy_v015 import classify_numbers, edge_name
from .shape_theory_v016 import UNIVERSE, classification_keys

SCHEMA = 'lotto7x7.unified_selection.v0.21'
OBJECTIVES = ('3plus', '4plus', '5plus')
STRATEGIES = ('uniform', 'coverage', 'shape_score')


def uniform_candidates(size: int, seed: int) -> np.ndarray:
    """IID six-of-49 *tickets*, rejecting duplicates until `size` distinct.

    The resulting unordered pool is an exactly uniform sample *without*
    replacement from all C(49,6) six-sets; no historical shapes or class
    probabilities are used in pool generation.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or not 5 <= size <= 5000
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('candidate pool must be 5..5000 and seed a nonnegative integer')
    rng = np.random.default_rng(seed)
    seen: set[tuple[int,...]] = set()
    out: list[tuple[int,...]] = []
    while len(out) < size:
        # Keys: 49 iid continuous uniforms, select six smallest without replacement.
        keys = rng.random((max(32, size-len(out)), 49))
        selected = np.argpartition(keys, kth=5, axis=1)[:, :6] + 1
        for row in selected:
            ticket = tuple(sorted(int(v) for v in row))
            if ticket not in seen:
                seen.add(ticket)
                out.append(ticket)
                if len(out) == size:
                    break
    return np.asarray(out, dtype=np.uint8)


def _digest_candidates(candidates: np.ndarray) -> str:
    canonical = [list(map(int, row)) for row in candidates]
    return hashlib.sha256(json.dumps(canonical, separators=(',', ':'),
                                     ensure_ascii=True).encode('ascii')).hexdigest()


def frozen_shape_scores(candidates: np.ndarray, freeze: dict,
                        tables: dict) -> tuple[np.ndarray, dict]:
    """v0.17 frozen L3 q/p, not a probability advantage for six-number tickets."""
    verify_seal(freeze)
    hist = freeze['frozen_rule']['historical_counts']['edge']
    tau = float(freeze['frozen_rule']['tau'])
    n = int(freeze['source']['historical_draw_count'])
    classes = classification_keys(classify_numbers(candidates))
    table = tables['edge']
    codes = table['codes'].astype(np.int64)
    ix = np.searchsorted(codes, classes['edge'])
    if np.any(ix >= len(codes)) or not np.array_equal(codes[ix], classes['edge'].astype(np.int64)):
        raise ValueError('candidate category absent from frozen fair universe')
    counts = np.array([int(hist[str(int(c))]) for c in codes], dtype=np.int64)
    if counts.sum() != n:
        raise ValueError('frozen category totals differ from historical count')
    fair_p = table['p']
    q = (counts + tau * fair_p) / (n + tau)
    if not np.isclose(q.sum(), 1., atol=1e-12, rtol=0):
        raise AssertionError('normalized shape class posterior missing mass')
    ratios = q[ix] / fair_p[ix]
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise AssertionError('nonpositive shape ratio')
    return ratios, {
        'codes': classes,
        'edge_name': [edge_name(int(c)) for c in classes['edge']],
        'fair_class_probability': fair_p[ix],
        'estimated_class_probability': q[ix],
    }


def _objective_key(coverage: dict, objective: str) -> tuple[int, int, int]:
    if objective not in OBJECTIVES:
        raise ValueError('objective must be 3plus, 4plus or 5plus')
    c = coverage['at_least_counts']
    priority = {
        '3plus': ('3', '4', '5'),
        '4plus': ('4', '3', '5'),
        '5plus': ('5', '4', '3'),
    }[objective]
    return tuple(int(c[k]) for k in priority)


def coverage_select(candidates: np.ndarray, count: int, *, objective: str,
                    max_evaluations: int = 10000,
                    on_progress: Callable[[str],None] | None = None) -> tuple[list[int],dict]:
    """Exact-oracle greedy plus budgeted first-improvement 1-swap.

    Only a *bounded* search of the candidate pool; never claimed to solve
    the global set-cover problem across all C(49,6) legal tickets.
    Starting with the uniform baseline and accepting only strict exact-score
    improvements guarantees selected objective >= uniform within same pool.
    """
    x=np.asarray(candidates)
    if x.ndim != 2 or x.shape[1] != 6 or not 1 <= count <= 5 or count > len(x):
        raise ValueError('invalid candidate pool and ticket count')
    if objective not in OBJECTIVES or max_evaluations<1:
        raise ValueError('invalid portfolio objective or evaluation budget')
    if len(set(map(tuple,x.tolist()))) != len(x):
        raise ValueError('duplicate candidate tickets')
    # Exact oracle returns full universe counts. Use only legal 1..5 tickets.
    cache: dict[tuple[int,...],dict] = {}
    evaluations = 0
    def score(indices: list[int]) -> tuple[int,int,int]:
        nonlocal evaluations
        key=tuple(sorted(indices))
        if key not in cache:
            if evaluations >= max_evaluations:
                raise RuntimeError('exact objective evaluation budget exhausted')
            cache[key] = exact_coverage([list(map(int,x[i])) for i in key])
            evaluations += 1
        return _objective_key(cache[key],objective)
    baseline=list(range(count))
    best_baseline=score(baseline)
    # Initialize at first uniform candidate for reproducible greedy growth.
    chosen=[0]
    for slot in range(1,count):
        best_index=None
        best_score=None
        for i in range(len(x)):
            if i in chosen:continue
            candidate=score(chosen+[i])
            if best_score is None or candidate > best_score:
                best_score=candidate;best_index=i
        if best_index is None:raise AssertionError('failed to extend portfolio')
        chosen.append(best_index)
        if on_progress: on_progress(f'[COVERAGE] greedy {slot+1}/{count}: {best_score} exact objective counts')
    if score(chosen)<best_baseline:chosen=baseline.copy()
    # One bounded deterministic improving swap pass. Every accepted move is
    # evaluated against the *exact* probability target, never a proxy metric.
    accepted=0
    for slot in range(count):
        prev=score(chosen)
        for i in range(len(x)):
            if i in chosen:continue
            trial=chosen.copy();trial[slot]=i
            if score(trial)>prev:
                chosen=trial;accepted+=1;break
    if score(chosen)<best_baseline:raise AssertionError('coverage optimized below baseline')
    diagnostics={
      'method':'exact_coverage_oracle_greedy_plus_one_pass_first_improvement_swap',
      'objective':objective,
      'objective_order':{'3plus':['3','4','5'],'4plus':['4','3','5'],'5plus':['5','4','3']}[objective],
      'oracle_evaluations':evaluations,
      'one_swap_improvements':accepted,
      'globally_optimal':False,
      'search_scope':'finite freshly uniform candidate pool, not all 13,983,816 tickets',
      'baseline_objective':list(best_baseline),
      'selected_objective':list(score(chosen)),
    }
    return chosen,diagnostics


def _grid_html(numbers: list[int]) -> str:
    on=set(numbers)
    cells=[]
    for i in range(49):
        x=(i%7)*22; y=(i//7)*22
        fill='#167b72' if i+1 in on else '#eaf0f1'
        cells.append(f'<rect x="{x}" y="{y}" width="19" height="19" rx="3" fill="{fill}" />')
    return '<svg width="154" height="154" viewBox="0 0 154 154" role="img" aria-label="7x7 six-pixel lotto shape">'+''.join(cells)+'</svg>'


def build_selection(catalog: Path, universe: Path, freeze_path: Path, *,
                    pool_size: int = 100, tickets: int = 5,
                    objective: str = '3plus', seed: int = 649,
                    max_evaluations: int = 10000,
                    on_progress: Callable[[str],None] | None = None) -> tuple[dict,list[dict],list[dict]]:
    """Read-only integration of prevalidated P1/P0 freeze, no live outcomes."""
    if not isinstance(tickets,int) or isinstance(tickets,bool) or not 1<=tickets<=5:
        raise ValueError('tickets must be 1..5')
    if pool_size<tickets or objective not in OBJECTIVES:
        raise ValueError('invalid pool size or coverage objective')
    freeze_path=Path(freeze_path)
    with freeze_path.open(encoding='utf-8') as stream:
        freeze=json.load(stream)
    ids,nums,tables,_=check_freeze(freeze,Path(catalog),Path(universe))
    if len(ids)!=int(freeze['source']['historical_draw_count']):
        raise AssertionError('historical count mismatch')
    pool=uniform_candidates(pool_size,seed)
    ratios,details=frozen_shape_scores(pool,freeze,tables)
    cls=details['codes']
    graph_codes=classify_numbers(pool)
    scorer_order=np.argsort(-ratios,kind='stable')
    selections={
        'uniform':list(range(tickets)),
        'shape_score':[int(i) for i in scorer_order[:tickets]],
    }
    coverage_indices,optimization=coverage_select(pool,tickets,objective=objective,
                    max_evaluations=max_evaluations,on_progress=on_progress)
    selections['coverage']=coverage_indices
    # Strict per-strategy exact audit and pooled output, no ticket 6/6 uplift.
    strategies={}
    bet_rows=[]
    for method in STRATEGIES:
        indexes=selections[method]
        selected=normalized_bets([pool[i].tolist() for i in indexes])
        exact=exact_coverage([list(t) for t in selected])
        expected_full=tickets/UNIVERSE
        if exact['at_least_counts']['6']!=tickets or abs(exact['at_least']['6']-expected_full)>1e-14:
            raise AssertionError('six-hit fair odds invariant violated')
        strategies[method]={
            'indices':indexes,
            'tickets':[list(t) for t in selected],
            'exact_fair_portfolio_coverage':exact,
            'experimental_shape_score_mean':float(np.mean(ratios[indexes])),
            'description':{
                'uniform':'First N tickets from an independent uniform candidate pool',
                'coverage':'Exact >=h fair-universe probability local optimization over same pool',
                'shape_score':'Top N historical L3 smoothed q/p class scores; exploratory only',
            }[method]
        }
        for rank,i in enumerate(indexes,1):
            bet_rows.append({'strategy':method,'rank':rank,'pool_index':int(i),
                 'numbers':' '.join(f'{int(v):02d}' for v in pool[i]),
                 'edge_class':details['edge_name'][i],
                 'shape_ratio':float(ratios[i]),
                 'class_fair_p':float(details['fair_class_probability'][i])})
    pool_rows=[]
    for i,row in enumerate(pool):
        pool_rows.append({'pool_index':i,
             'numbers':' '.join(f'{int(v):02d}' for v in row),
             'component_class':int(cls['component'][i]),
             'partition_code':int(cls['partition'][i]),
             'edge_class':details['edge_name'][i],
             'cardinal_edges':int(graph_codes['cardinal_edges'][i]),
             'diagonal_edges':int(graph_codes['diagonal_edges'][i]),
             'class_fair_p':float(details['fair_class_probability'][i]),
             'class_model_q':float(details['estimated_class_probability'][i]),
             'shape_ratio':float(ratios[i])})
    report={
      'schema':SCHEMA,'research_only':True,
      'settings':{'pool_size':pool_size,'tickets':tickets,'objective':objective,
                  'seed':seed,'max_evaluations':max_evaluations,
                  'candidate_sampler':'uniform legal 6/49 ticket sets without replacement',
                  'shape_model':'frozen P1 L3 Dirichlet q/p (tau=384), no model refit'},
      'data':{'catalog_sha256':file_sha256(Path(catalog)),
              'universe_sha256':file_sha256(Path(universe)),
              'freeze_sha256':file_sha256(freeze_path),
              'freeze_rule_sha256':freeze['frozen_rule']['sha256'],
              'history_count':len(ids),'last_historical_draw_id':int(ids[-1]),
              'candidate_pool_sha256':_digest_candidates(pool),
              'candidate_count':len(pool),
              'fair_universe':UNIVERSE},
      'strategies':strategies,'coverage_optimization':optimization,
      'comparison':{
         'coverage_vs_uniform_delta_3plus':strategies['coverage']['exact_fair_portfolio_coverage']['at_least']['3'] - strategies['uniform']['exact_fair_portfolio_coverage']['at_least']['3'],
         'coverage_vs_uniform_delta_4plus':strategies['coverage']['exact_fair_portfolio_coverage']['at_least']['4'] - strategies['uniform']['exact_fair_portfolio_coverage']['at_least']['4'],
         'shape_score_vs_uniform_delta_3plus':strategies['shape_score']['exact_fair_portfolio_coverage']['at_least']['3'] - strategies['uniform']['exact_fair_portfolio_coverage']['at_least']['3'],
         'shape_score_vs_uniform_delta_4plus':strategies['shape_score']['exact_fair_portfolio_coverage']['at_least']['4'] - strategies['uniform']['exact_fair_portfolio_coverage']['at_least']['4'],
      },
      'limitations':[
         'No evidence of higher six-number jackpot odds: any distinct N tickets have N / C(49,6) exact fair chance.',
         'Fair portfolio >=3/4/5 coverage is not a learned next-draw prediction.',
         'Shape scorer is historical category frequency, not a trained/validated neural network selector.',
         'P2 prequential class log-gains were negative on the 1,997 historical evaluation draws.',
         'All strategy comparisons use exactly the SAME freshly uniform candidate pool and ticket count.',
         'Coverage search is a finite-pool greedy+swap heuristic with exact evaluations, not a global optimum.',
         'No new future outcome or post-freeze data enters the selection or scoring.',
         'This command does not submit a lottery ticket or make a purchase.'
      ]
    }
    return report,pool_rows,bet_rows


def selection_html(report: dict) -> str:
    st=report['strategies']
    def pct(v):return f'{100*float(v):.5f}%'
    rows=[]
    for name in STRATEGIES:
        s=st[name];p=s['exact_fair_portfolio_coverage']['at_least']
        tr=''.join(f'<td>{pct(p[h])}</td>' for h in ('3','4','5','6'))
        rows.append(f'<tr><th>{escape(name)}</th>{tr}</tr>')
    blocks=[]
    for name in STRATEGIES:
        s=st[name]
        tickets=[]
        for nums in s['tickets']:
            tickets.append('<div class="pick">'+_grid_html(nums)+'<div class="nums">'+escape(' '.join(f'{x:02d}' for x in nums))+'</div></div>')
        blocks.append(f'<section><h2>{escape(name)} — {len(s["tickets"])} tickets</h2>' +
            f'<p>{escape(s["description"])}</p><div class="picks">'+''.join(tickets)+'</div></section>')
    json_meta=escape(json.dumps(report['data'],ensure_ascii=False,indent=2))
    return ('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Lotto649 v0.21 Unified Selection</title>
<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#213a40;background:#f4f7f8}h1{color:#087e75}section{background:white;border-radius:12px;border:1px solid #dce5e6;padding:1.3rem;margin:1rem 0}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #e0e5e6;padding:.6rem;text-align:left}th{font-weight:600}.picks{display:flex;flex-wrap:wrap;gap:14px}.pick{border:1px solid #d7e3e1;padding:.5rem;border-radius:8px}.nums{font:12px ui-monospace,monospace;text-align:center}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{border-left:4px solid #ba8846;padding:.8rem;background:#faf4e5}</style>
<h1>Lotto649 — v0.21 Unified Selection</h1><p>同一批合法、均勻抽樣的候選池；固定注數；三種不同選號策略。所有覆蓋機率均為公平 6/49 的精確組合計數。</p>''' +
      f'<section><h2>精確機率比較</h2><p>候選 {report["settings"]["pool_size"]} 組，選 {report["settings"]["tickets"]} 注，主要最佳化 >= {report["settings"]["objective"][0]} 號。</p><table><thead><tr><th>策略</th><th>至少 3 號</th><th>至少 4 號</th><th>至少 5 號</th><th>六個主號全中</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></section>' +
      ''.join(blocks) +
      '<section class="warning"><strong>研究限制</strong><p>歷史形狀分數並未證明具有預測優勢。任何相異的 N 注，公平六個主號全中率都相同為 N/13,983,816；最佳化只改變多注「至少命中幾號」的聯集機率，沒有提高單注開獎機率。沒有串接下注功能。</p></section>' +
      '<section><h2>輸入來源與重現資訊</h2><pre>'+json_meta+'</pre></section></html>')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:raise ValueError('empty output rows')
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='raise')
        writer.writeheader();writer.writerows(rows)
