"""Small command line for the rebuilt image-only Lotto649 research project."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from .core import decode, encode, invert_spectrum, spectrum
from .research import DATA, load_history, research
from .learning import learned_replay
import torch


def gallery(report: dict, count: int = 40) -> str:
    """Standalone visual audit with no external code or legacy dependencies."""
    def image_svg(numbers):
        image = encode(numbers)
        rects = []
        for r in range(7):
            for c in range(7):
                fill = "#173e72" if image[r, c] else "#edf2f8"
                rects.append(f'<rect x="{c*23+5}" y="{r*23+5}" width="21" height="21" rx="3" fill="{fill}"/>')
        return '<svg viewBox="0 0 171 171" role="img" aria-label="7 by 7 six-point image">'+''.join(rects)+'</svg>'
    blocks = []
    for entry in report['rows'][:count]:
        figures = ['<section><h4>Actual</h4>'+image_svg(entry['actual'])+'</section>']
        for name, item in entry['methods'].items():
            adjusted = item.get('adjusted', item.get('chance_adjusted'))
            figures.append('<section><h4>'+escape(name)+'</h4>'+image_svg(item['predicted'])+
                           '<p>Adjusted '+f"{adjusted:+.4f}"+'</p></section>')
        blocks.append('<article><h3>'+str(entry['date'])+' | '+str(entry['draw_id'])+'</h3><div class="grid">'+''.join(figures)+'</div></article>')
    summary = report['evaluation'] if report.get('schema') == 'lotto7x7.visual.v1' else report['summary']
    scores = ''.join('<tr><th>'+escape(name)+'</th><td>'+
                     f"{item.get('adjusted_mean', item.get('mean_iid_adjusted_similarity')):+.6f}"+
                     '</td><td>'+str(item['unique_patterns']['unique'])+'</td></tr>'
                     for name,item in summary.items())
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Lotto7x7 Visual Research</title>'
            '<style>body{background:#f5f7fa;color:#182b40;font:15px system-ui;margin:30px auto;max-width:1150px}'
            'article{background:white;border:1px solid #d9e2eb;border-radius:12px;padding:15px;margin:12px 0}'
            '.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}'
            'svg{width:100%;max-width:155px}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}'
            'p{color:#456}h4{font-size:13px}@media(max-width:650px){.grid{grid-template-columns:repeat(2,1fr)}}</style>'
            '<h1>Lotto7x7: image pattern similarity</h1><p>Exploratory history, not a prospective forecast or ticket suggestion.</p>'
            '<table><tr><th>Image method</th><th>Adjusted similarity</th><th>Unique predicted patterns</th></tr>'+scores+'</table>'
            '<h2>Observed and predicted images</h2>'+''.join(blocks)+'</html>')


def main() -> None:
    parser = argparse.ArgumentParser(description="7x7 image-first lottery research")
    action = parser.add_subparsers(dest="command", required=True)
    a = action.add_parser("check", help="Check archive and image/transform round-trip")
    a.add_argument("--data", type=Path, default=DATA)
    r = action.add_parser("replay", help="Research graphical image similarity")
    r.add_argument("--data", type=Path, default=DATA)
    r.add_argument("--year", default="all")
    r.add_argument("--window", type=int, default=5)
    r.add_argument("--baseline", type=int, default=32)
    r.add_argument("--shuffles", type=int, default=99)
    r.add_argument("--seed", type=int, default=649)
    r.add_argument("--out", type=Path, default=Path("results/visual_replay.json"))
    learn = action.add_parser("learn", help="Train CNN + temporal GRU on 7x7 image sequences")
    learn.add_argument("--data", type=Path, default=DATA)
    learn.add_argument("--year", default="all")
    learn.add_argument("--window", type=int, default=5)
    learn.add_argument("--epochs", type=int, default=45)
    learn.add_argument("--patience", type=int, default=8)
    learn.add_argument("--batch-size", type=int, default=64)
    learn.add_argument("--learning-rate", type=float, default=0.003)
    learn.add_argument("--embedding", type=int, default=24)
    learn.add_argument("--hidden", type=int, default=32)
    learn.add_argument("--baseline", type=int, default=32)
    learn.add_argument("--seed", type=int, default=649)
    learn.add_argument("--out", type=Path, default=Path("results/learned_replay.json"))
    learn.add_argument("--weights", type=Path, default=Path("results/learned_image_weights.pt"))
    g = action.add_parser("gallery", help="Render 7x7 SVG report from replay JSON")
    g.add_argument("--report", type=Path, default=Path("results/visual_replay.json"))
    g.add_argument("--out", type=Path, default=Path("results/visual_gallery.html"))
    g.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    if args.command == "check":
        images, dates, ids = load_history(args.data)
        first = images[0]
        for domain in ("dct", "fft"):
            if abs(invert_spectrum(spectrum(first, domain), domain)-first).max() > 1e-10:
                raise RuntimeError("image spectral reconstruction failed")
        print(f"[PASS] Data records: {len(images)}, date range: {dates[0]}..{dates[-1]}")
        print(f"[PASS] First image six numbers: {decode(first)}")
        print("[PASS] 7x7 encoding and full DCT/FFT reconstruction")
    elif args.command == "replay":
        year = None if args.year.lower() == "all" else int(args.year)
        report = research(args.data, year=year, window=args.window,
                          baseline=args.baseline, shuffles=args.shuffles, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        print(f"[EVALUATED] {report['evaluated_draws']} historical draws; selection: {report['model_selected_on_validation_only']}")
        print("Method                 Adjusted similarity    Unique patterns    Top pattern share")
        for name, item in report['evaluation'].items():
            d = item['unique_patterns']
            print(f"{name:22s} {item['adjusted_mean']:+17.6f} {d['unique']:16d} {d['top_fraction']:18.2%}")
        t = report['temporal_order']
        print(f"[ORDER] Lag-1 real={t['lag1_observed']:+.6f} shuffled={t['lag1_shuffled_mean']:+.6f} two-sided tail={t['two_sided_tail']:.4f}")
        print(f"[DONE] {args.out.resolve()}")
        print("[CAUTION] Previously explored historical data; no demonstrated forecasting edge")
    elif args.command == "learn":
        year = None if args.year.lower() == "all" else int(args.year)
        report, trained = learned_replay(
            args.data, year=year, window=args.window, epochs=args.epochs,
            patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, embedding=args.embedding,
            hidden=args.hidden, baseline=args.baseline, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        torch.save(trained.state_dict(), args.weights)
        print("Lotto7x7 Image Learning v0.1.0")
        print(f"[TRAIN] CNN+GRU image learner: best epoch {report['trained_image_model']['best_epoch']}")
        print(f"[EVALUATED] {report['evaluation_draws']} image targets")
        for name, row in report['summary'].items():
            print(f"{name:18s} adjusted={row['mean_iid_adjusted_similarity']:+.6f} "
                  f"unique_images={row['unique_patterns']['unique']}")
        print(f"[DONE] Report: {args.out.resolve()}")
        print(f"[DONE] Weights: {args.weights.resolve()}")
        print("[CAUTION] Retrospective image similarity is not proof of a lottery edge")
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report.get('schema') not in ('lotto7x7.visual.v1', 'lotto7x7.learned_image.v1') or args.count < 1:
            raise ValueError('unsupported report schema or count')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(gallery(report, args.count), encoding='utf-8')
        print(f"[DONE] HTML: {args.out.resolve()}")


if __name__ == '__main__':
    main()
