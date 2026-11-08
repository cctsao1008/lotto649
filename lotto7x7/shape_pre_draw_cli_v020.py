"""CLI for v0.20 pre-draw class-forecast sealing and post-draw reconciliation."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .shape_prospective_v019 import check_freeze
from .shape_pre_draw_v020 import make_commitment, audit_commitment


def configure_parsers(action):
    p=action.add_parser('shape-prospective-seal',help='Publish a full next-draw shape-class forecast BEFORE draw time')
    _common(p)
    p.add_argument('--report',type=Path,required=True,help='latest v0.19 report with all prior scored draws')
    p.add_argument('--issue-id',type=int,required=True)
    p.add_argument('--issue-date',type=str,required=True,help='Gregorian YYYY-MM-DD, normally Tuesday or Friday')
    p.add_argument('--out',type=Path,required=True)

    p=action.add_parser('shape-prospective-audit',help='Reconcile pre-draw sealed category probabilities with new v0.19 event')
    _common(p)
    p.add_argument('--seal',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True,help='v0.19 report AFTER importing the new draw')
    p.add_argument('--out',type=Path,required=True)


def _common(p):
    p.add_argument('--freeze',type=Path,default=Path('docs/prospective_freeze_20261108.json'))
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))


def _reserve(out:Path, inputs:list[Path]):
    if out.suffix.lower()!='.json':raise ValueError('--out must be a .json path')
    target_html=out.with_suffix('.html')
    if target_html==out or out.resolve() in {p.resolve() for p in inputs}:
        raise ValueError('output path conflicts with input')
    if out.exists() or target_html.exists():
        raise FileExistsError('refusing to overwrite an existing pre-draw record')
    out.parent.mkdir(parents=True,exist_ok=True)


def _render(data:dict)->str:
    tag=data.get('decision','PRE_DRAW_CLASS_FORECAST')
    rows=[]
    if 'distributions' in data:
        for lev,entries in data['distributions'].items():
            for item in entries:
                rows.append(f'<tr><td>{escape(lev)}</td><td>{escape(item["class"])}</td>'
                            f'<td>{item["p_fair"]:.9g}</td><td>{item["q_before_draw"]:.9g}</td>'
                            f'<td>{item["relative_ratio"]:.7g}</td></tr>')
    label=escape(tag)
    return ('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Lotto649 v0.20</title>'''
            '''<style>body{font:15px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#243543}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:.5rem;text-align:left}th{position:sticky;top:0;background:white}h1{color:#168073}.note{padding:.8rem;border:1px solid #aaa}</style>'''
            f'<h1>Lotto649 v0.20 — {label}</h1>'
            f'<p>期別：{data["target"]["draw_id"]}；日期：{escape(data["target"]["date"])}</p>'
            f'<p>SHA256 seal：<code>{escape(data.get("self_checksum_sha256",data.get("commitment_sha256","")))}</code></p>'
            '<p class="note">形狀分類機率不是下一期六碼預測。必須於開獎前公開 Git 提交，並獨立核對官方開獎結果。</p>'
            '<table><tr><th>層級</th><th>分類</th><th>公平機率</th><th>模型機率</th><th>比值</th></tr>'
            +''.join(rows)+'</table></html>')


def run(args):
    ids,nums,tables,_=check_freeze(json.loads(args.freeze.read_text(encoding='utf-8')),args.catalog,args.universe)
    frozen=json.loads(args.freeze.read_text(encoding='utf-8'))
    previous=json.loads(args.report.read_text(encoding='utf-8'))
    inputs=[args.freeze,args.catalog,args.universe,args.report]
    if args.command=='shape-prospective-seal':
        record=make_commitment(frozen,tables,previous,issue_id=args.issue_id,issue_date=args.issue_date,
                               freeze_path=args.freeze,report_path=args.report)
        print(f'[PRE-DRAW] issue={args.issue_id} date={args.issue_date} previous={record["previous_events_count"]}',flush=True)
    elif args.command=='shape-prospective-audit':
        inputs.append(args.seal)
        seal=json.loads(args.seal.read_text(encoding='utf-8'))
        record=audit_commitment(seal,frozen,tables,previous)
        print(f'[AUDIT] issue={record["target"]["draw_id"]} status={record["decision"]}',flush=True)
    else:
        raise ValueError('unknown v0.20 action')
    _reserve(args.out, inputs)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    with args.out.with_suffix('.html').open('x',encoding='utf-8') as f:f.write(_render(record))
    print(f'[DONE] JSON: {args.out.resolve()}',flush=True)
    print(f'[DONE] HTML: {args.out.with_suffix(".html").resolve()}',flush=True)
