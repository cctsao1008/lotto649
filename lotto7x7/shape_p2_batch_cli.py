"""P2 Batch 3 single CLI: leakage-safe v0.17 + offline v0.18 Explorer."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .shape_predictive_v017 import analyze, per_draw_rows, last_candidate_rows
from .shape_explorer_v018 import offline_shape_explorer
from .shape_theory_v016 import RULE_VERSION,UNIVERSE


def configure_parsers(action):
    p=action.add_parser('shape-p2-batch',help='P2 Batch 3: walk-forward shape validation, fair-IID controls and offline interactive Shape Explorer')
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--min-train',type=int,default=500)
    p.add_argument('--tau',type=float,default=384.)
    p.add_argument('--iid-replicates',type=int,default=1000)
    p.add_argument('--ticket-null-replicates',type=int,default=128)
    p.add_argument('--candidate-pool',type=int,default=64)
    p.add_argument('--tickets-per-draw',type=int,default=5)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v017_v018_p2_batch.json'))


def output_paths(path:Path)->dict[str,Path]:
    p=Path(path)
    if p.suffix.lower()!='.json':raise ValueError('--out must end with .json')
    stem=str(p.with_suffix(''))
    return {'JSON':p,'BACKTEST_CSV':Path(stem+'_backtest.csv'),
        'TICKETS_CSV':Path(stem+'_tickets.csv'),
        'REPORT_HTML':Path(stem+'_report.html'),
        'EXPLORER_HTML':Path(stem+'_explorer.html'),
        'NPZ':Path(stem+'.npz'),
        'PREREGISTRATION_JSON':Path(stem+'_future_watchlist.json')}


def _hash(path:Path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def _read_p1(catalog:Path,universe:Path):
    b=json.loads(universe.read_text(encoding='utf-8'))
    if (b.get('schema')!=RULE_VERSION or b.get('rule_hash_sha256')!=sha256(RULE_VERSION.encode()).hexdigest()
        or not b.get('complete') or int(b.get('total',0))!=UNIVERSE or
        any(sum(map(int,b['counts'][k].values()))!=UNIVERSE for k in ('component','partition','edge'))):
        raise ValueError('P1 exact-universe schema / category masses / rule hash are inconsistent')
    with np.load(catalog,allow_pickle=False) as z:
        if not {'draw_ids','numbers','component','partition','edge','density','bandwidths','mode_labels'}<=set(z.files):
            raise ValueError('missing required P1 cached features')
        ids=z['draw_ids'].astype(np.int64)
        numbers=z['numbers'].astype(np.uint8)
        density=z['density'].astype(np.float32)
        bws=z['bandwidths']
        col=int(np.argmin(abs(bws-.55)))
        modes=z['mode_labels'].astype(np.int32)
        code_arrays={k:z[k] for k in ('component','partition','edge')}
    from .shape_theory_v016 import classification_keys
    from .shape_taxonomy_v015 import classify_numbers
    actual=classification_keys(classify_numbers(numbers))
    if any(not np.array_equal(code_arrays[k],actual[k]) for k in actual):
        raise ValueError('P1 cached class codes fail independently recalculated taxonomy')
    if density.shape!=(len(ids),len(bws)) or modes.shape!=(len(ids),):
        raise ValueError('P1 KDE / mode arrays out of alignment')
    return ids,numbers,b,density[:,col],modes


def _write_csv(path:Path,rows):
    it=iter(rows)
    first=next(it,None)
    if first is None:raise ValueError('empty CSV rows')
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(first.keys()),extrasaction='raise')
        w.writeheader();w.writerow(first)
        for x in it:w.writerow(x)


def report_html(report:dict)->str:
    from html import escape
    p=['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"/>',
       '<title>Lotto649 v0.17 — Walk-forward Validation</title>',
       '<style>body{font:15px system-ui;max-width:1080px;margin:2rem auto;padding:0 1rem;background:#f5f8f9;color:#283a45}section{background:#fff;border:1px solid #e3e9eb;border-radius:11px;padding:1rem;margin:1rem 0}table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:.55rem;border-bottom:1px solid #e4eaed}th:first-child,td:first-child{text-align:left}h1{color:#137d72}.mute{color:#627682}</style>',
       '<h1>Batch 3 / P2 — Leakage-safe Walk-forward Validation</h1>',
       '<p>Historical data are revealed one draw at a time. Exact uniform 6/49 class priors, no future features, independent IID sequence refits. No claim of increased winning probability.</p>',
       '<section><h2>Sequential categorical log scores against fair IID</h2>',
       '<table><tr><th>Level</th><th>Mean gain (bits / draw)</th><th>Sum LLR (nats)</th><th>Independent IID p (one-sided)</th><th>3-level adjusted p</th></tr>']
    for level,v in report['prequential_log_scores'].items():
        p.append(f'<tr><td>{escape(level)}</td><td>{v["mean_log_gain_per_draw_bits"]:.6f}</td>'
           f'<td>{v["cumulative_log_likelihood_ratio"]:.5f}</td>'
           f'<td>{v["iid_refit_upper_tail_p"]:.5f}</td>'
           f'<td>{v["iid_refit_upper_tail_bonferroni_across_three_levels"]:.5f}</td></tr>')
    p.extend(['</table><p class="mute">Positive cumulative log gain supports the online class model; non-positive means exact IID predicted the observed classes better. A small Monte Carlo p is not prospective proof after exploratory model development.</p></section>',
        '<section><h2>Equal-budget five-ticket backtests</h2>',
        '<table><tr><th>Strategy</th><th>Total bets</th><th>3+ ticket hits</th><th>4+ ticket hits</th><th>5+ hits</th><th>6 hits</th><th>Conditional IID tail p (3+)</th></tr>'])
    for name,v in report['ticket_backtest']['strategies'].items():
        p.append(f'<tr><td>{escape(name)}</td><td>{v["total_bets"]:,}</td>'
           f'<td>{v["total_match_at_least_3"]}</td><td>{v["total_match_at_least_4"]}</td>'
           f'<td>{v["total_match_at_least_5"]}</td><td>{v["total_match_6"]}</td>'
           f'<td>{v["iid_conditional_p_ge_3plus"]:.5f}</td></tr>')
    rare=report['rare_shape_watchlist']
    p.extend(['</table><p class="mute">Candidate pools and model predictions are generated before each target draw. IID ticket null keeps selected tickets fixed; it does not independently refit entire adaptive models.</p></section>',
        '<section><h2>Rare shape watchlist — post-selected, NOT confirmed</h2>',
        f'<p>{escape(rare["class"])}: {rare["observed_total"]} history occurrences; {rare["observed_in_retrospective_evaluation"]} after chosen warmup. ',
        'It was discovered by inspecting the complete history; historical evaluation is not independent confirmation.</p></section>',
        '<section><h2>Research decision</h2><p>Shape scores are descriptive until sequential independent evidence exceeds IID controls. '
        'Every valid 6/49 combination has the same fair future probability: 1/13,983,816.</p>',
        '<p>Open the companion <code>_explorer.html</code> to query any six numbers entirely offline.</p></section></html>'])
    return '\n'.join(p)


def run(args):
    if args.command!='shape-p2-batch':raise ValueError('wrong P2 subcommand')
    paths=output_paths(args.out)
    sources={args.catalog.resolve(),args.universe.resolve()}
    if len(set(p.resolve() for p in paths.values()))!=len(paths) or any(p.exists() or p.resolve() in sources for p in paths.values()):
        raise FileExistsError('Output paths must be new, unique and must never overwrite inputs')
    ids,nums,base,density,modes=_read_p1(args.catalog,args.universe)
    report,arrays=analyze(ids,nums,base,min_train=args.min_train,tau=args.tau,
       iid_replicates=args.iid_replicates,ticket_null_replicates=args.ticket_null_replicates,
       seed=args.seed,pool_size=args.candidate_pool,bets=args.tickets_per_draw,
       on_progress=lambda s:print(s,flush=True))
    report['source_checksums']={'catalog_sha256':_hash(args.catalog),'universe_sha256':_hash(args.universe),
      'rule_hash_sha256':base['rule_hash_sha256']}
    watch={'schema':'lotto7x7.p2_future_only_watchlist.v0.17',
      'source_draw_id_last':int(ids[-1]),'recorded_from_historical_exploration':True,
      'eligibility':'Future draws strictly after source_draw_id_last; historical draws cannot be used as prospective validation',
      'fixed_category':report['rare_shape_watchlist'],
      'frozen_rule_hash_sha256':base['rule_hash_sha256'],
      'fair_model':'Uniform independent legal 6/49 main numbers',
      'status':'WATCHLIST ONLY: no monitoring automation configured'}
    explorer=offline_shape_explorer(ids,nums,base,density=density,mode_labels=modes)
    for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
    for name,p in paths.items():
        if name=='JSON':
            with p.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        elif name=='BACKTEST_CSV':_write_csv(p,per_draw_rows(arrays,args.min_train))
        elif name=='TICKETS_CSV':_write_csv(p,last_candidate_rows(arrays,report))
        elif name=='REPORT_HTML':
            with p.open('x',encoding='utf-8') as f:f.write(report_html(report))
        elif name=='EXPLORER_HTML':
            with p.open('x',encoding='utf-8') as f:f.write(explorer)
        elif name=='NPZ':
            with p.open('xb') as f:np.savez_compressed(f,**arrays)
        else:
            with p.open('x',encoding='utf-8') as f:json.dump(watch,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        print(f'[DONE] {name}: {p.resolve()}',flush=True)
    print('[CAUTION] Backtests and shape visualization cannot imply greater next-draw odds under fair IID',flush=True)
