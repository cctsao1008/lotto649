"""Small command line for the rebuilt image-only Lotto649 research project."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from .core import decode, encode, invert_spectrum, spectrum
from .research import DATA, load_history, research
from .learning import learned_replay
from .diagnostics import SCHEMA as DIAGNOSTICS_SCHEMA, diagnose
from .loss_study_v03 import study as loss_study, synthetic_control, SCHEMA as LOSS_STUDY_SCHEMA
from .predictability_v04 import audit as predictability_audit, METHODS as PREDICTABILITY_METHODS
import torch
import numpy as np


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
        date_label = (str(entry['date']) if entry.get('date') is not None
                      else f"Year {entry['year']}")
        blocks.append('<article><h3>'+date_label+' | '+str(entry['draw_id'])+'</h3><div class="grid">'+''.join(figures)+'</div></article>')
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
    diag = action.add_parser("diagnose", help="v0.2 CNN/GRU collapse and time-order ablations")
    diag.add_argument("--data", type=Path, default=DATA)
    diag.add_argument("--year", default="all")
    diag.add_argument("--window", type=int, default=5)
    diag.add_argument("--epochs", type=int, default=45)
    diag.add_argument("--patience", type=int, default=8)
    diag.add_argument("--batch-size", type=int, default=64)
    diag.add_argument("--learning-rate", type=float, default=0.003)
    diag.add_argument("--embedding", type=int, default=24)
    diag.add_argument("--hidden", type=int, default=32)
    diag.add_argument("--baseline", type=int, default=32)
    diag.add_argument("--bootstrap", type=int, default=400)
    diag.add_argument("--seed", type=int, default=649)
    diag.add_argument("--out", type=Path, default=Path("results/image_diagnostics_v02.json"))
    loss_parser = action.add_parser("loss-study", help="v0.3 frozen-v0.2 loss/decoder comparison")
    loss_parser.add_argument("--data", type=Path, default=DATA)
    loss_parser.add_argument("--year", default="all")
    loss_parser.add_argument("--window", type=int, default=5)
    loss_parser.add_argument("--epochs", type=int, default=45)
    loss_parser.add_argument("--patience", type=int, default=8)
    loss_parser.add_argument("--batch-size", type=int, default=64)
    loss_parser.add_argument("--learning-rate", type=float, default=0.003)
    loss_parser.add_argument("--embedding", type=int, default=24)
    loss_parser.add_argument("--hidden", type=int, default=32)
    loss_parser.add_argument("--baseline", type=int, default=32)
    loss_parser.add_argument("--bootstrap", type=int, default=400)
    loss_parser.add_argument("--seed", type=int, default=649)
    loss_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                             help="Training compute device; CPU keeps frozen v0.3 behavior")
    loss_parser.add_argument("--out", type=Path, default=Path("results/loss_study_v03.json"))
    synthetic = action.add_parser("loss-synthetic", help="v0.3 periodic/IID controls")
    synthetic.add_argument("--epochs", type=int, default=45)
    synthetic.add_argument("--seed", type=int, default=649)
    synthetic.add_argument("--out", type=Path, default=Path("results/loss_synthetic_v03.json"))
    pa = action.add_parser("predictability", help="v0.4 multi-seed temporal predictability audit")
    pa.add_argument("--data", type=Path, default=DATA)
    pa.add_argument("--year", default="all")
    pa.add_argument("--window", type=int, default=5)
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--seed", type=int, default=649)
    pa.add_argument("--epochs", type=int, default=45)
    pa.add_argument("--patience", type=int, default=8)
    pa.add_argument("--batch-size", type=int, default=64)
    pa.add_argument("--learning-rate", type=float, default=0.003)
    pa.add_argument("--embedding", type=int, default=24)
    pa.add_argument("--hidden", type=int, default=32)
    pa.add_argument("--baseline", type=int, default=32)
    pa.add_argument("--bootstrap", type=int, default=400)
    pa.add_argument("--control-seeds", type=int, default=2)
    pa.add_argument("--control-samples", type=int, default=120)
    pa.add_argument("--control-epochs", type=int, default=45)
    pa.add_argument("--no-controls", action="store_true")
    pa.add_argument("--out", type=Path, default=Path("results/predictability_audit_v04.json"))
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
        from .research import calendar_years
        if np.any(dates):
            print(f"[PASS] Data records: {len(images)}, date range: {dates[0]}..{dates[-1]}")
        else:
            years = calendar_years(ids)
            print(f"[PASS] Data records: {len(images)}, years: {years[0]}..{years[-1]}")
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
    elif args.command == "diagnose":
        year = None if args.year.lower() == "all" else int(args.year)
        report = diagnose(
            args.data, year=year, window=args.window, epochs=args.epochs,
            patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, embedding=args.embedding,
            hidden=args.hidden, baseline=args.baseline,
            bootstrap=args.bootstrap, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Lotto7x7 Image Learning Diagnostics v0.2")
        print(f"[EVALUATED] {report['evaluation_draws']} historical images")
        print("Method                    Adjusted      Unique   Top share    Pixel logit SD")
        for name, result in report["summary"].items():
            diversity = result["unique_patterns"]
            score = report["logit_diagnostics"][name]
            print(f"{name:25s} {result['mean_iid_adjusted_similarity']:+10.6f} "
                  f"{diversity['unique']:10d} {diversity['top_fraction']:10.2%} "
                  f"{score['mean_pixel_logit_std']:17.6f}")
        control = report["fixed_weights_history_order_control"]
        print(f"[ORDER] Fixed-weight shuffled-context top-six change: "
              f"{control['changed_top6_fraction']:.2%}")
        for name, item in report["paired_intervals"].items():
            bounds = item["interval_95"]
            print(f"[PAIRED] {name}: {item['mean']:+.6f} "
                  f"interval=[{bounds[0]:+.6f}, {bounds[1]:+.6f}]")
        print(f"[DONE] Diagnostic report: {args.out.resolve()}")
        print("[CAUTION] Previously inspected historical results are exploratory only")
    elif args.command == "loss-study":
        year = None if args.year.lower() == "all" else int(args.year)
        report = loss_study(
            args.data, year=year, window=args.window, epochs=args.epochs,
            patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, embedding=args.embedding,
            hidden=args.hidden, baseline=args.baseline,
            bootstrap=args.bootstrap, seed=args.seed, device=args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Lotto7x7 Image Loss Alignment v0.3")
        device_info = report["execution"]
        print(f"[DEVICE] {device_info['actual_device']} "
              f"(PyTorch {device_info['torch_version']}); "
              f"training_seconds={device_info['training_seconds']:.3f}")
        print(f"[V02] Frozen baseline commit: {report['baseline_v02_commit']}")
        print(f"[EVALUATED] {report['evaluation_draws']} historical image targets")
        print("Method                 Adjusted     Unique  Top share   Soft mass   Pixel logit SD")
        for method, item in report["summary"].items():
            d = item["unique_patterns"]
            m = report["logit_diagnostics"][method]
            print(f"{method:22s} {item['mean_iid_adjusted_similarity']:+10.6f} "
                  f"{d['unique']:9d} {d['top_fraction']:10.2%} "
                  f"{m['effective_soft_mass']:11.5f} {m['mean_pixel_logit_std']:15.6f}")
        for name, item in report["paired_intervals"].items():
            low, high = item["interval_95"]
            print(f"[PAIRED] {name}: {item['mean']:+.6f} "
                  f"interval=[{low:+.6f}, {high:+.6f}]")
        print(f"[DONE] Report: {args.out.resolve()}")
        print("[CAUTION] Retrospective evaluation; diversity does not imply predictive skill")
    elif args.command == "predictability":
        year = None if args.year.lower() == "all" else int(args.year)
        print("Lotto7x7 Predictability Audit v0.4 (CPU)", flush=True)
        report = predictability_audit(
            args.data, year=year, window=args.window, seeds=args.seeds,
            seed=args.seed, epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            embedding=args.embedding, hidden=args.hidden,
            baseline=args.baseline, bootstrap=args.bootstrap,
            control_seeds=args.control_seeds, control_samples=args.control_samples,
            control_epochs=args.control_epochs, include_controls=not args.no_controls,
            on_progress=lambda line: print(line, flush=True))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[EVALUATED] {report['evaluation_draws']} historical targets, {args.seeds} seeds")
        print(f"[UNIFORM] Pixel Brier baseline: {report['uniform_no_history_baseline']['pixel_brier']:.6f}")
        print("Method                  Adjusted    Pixel Brier   Brier Skill   Epoch-0 Gain  Mean Unique")
        for name in PREDICTABILITY_METHODS:
            value = report["summary"][name]
            print(f"{name:24s} {value['mean_iid_adjusted_similarity']:+10.6f} "
                  f"{value['mean_pixel_brier']:13.6f} "
                  f"{value['mean_brier_skill_vs_uniform']:+12.4%} "
                  f"{value['training_gain_over_epoch0_adjusted']:+13.6f} "
                  f"{value['mean_unique_patterns']:12.2f}")
        for name, entry in report["paired_intervals"].items():
            low, high = entry["interval_95"]
            print(f"[PAIRED] {name}: {entry['mean']:+.6f} "
                  f"interval=[{low:+.6f}, {high:+.6f}]")
        for kind, control in report["synthetic_controls"].items():
            ordered = control["summary"]["ordered_cnn_gru"]
            shuffled = control["summary"]["shuffled_cnn_gru"]
            print(f"[CONTROL] {kind}: ordered Brier={ordered['mean_pixel_brier']:.6f}; "
                  f"shuffled Brier={shuffled['mean_pixel_brier']:.6f}")
        print(f"[DONE] Audit report: {args.out.resolve()}")
        print("[CAUTION] Previously inspected historical results are exploratory, not blind")
    elif args.command == "loss-synthetic":
        report = {"schema": "lotto7x7.loss_synthetic.v0.3",
                  "controls": [synthetic_control(kind=name, epochs=args.epochs,
                                                 seed=args.seed)
                               for name in ("periodic", "iid")]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Lotto7x7 Image Loss Synthetic Controls v0.3")
        for control in report["controls"]:
            print(f"[CONTROL] {control['kind']} image series")
            for method, item in control["summary"].items():
                print(f"  {method:22s} adjusted={item['adjusted_mean']:+.6f} "
                      f"shape={item['mean_image_shape_similarity']:.5f} "
                      f"unique={item['unique_patterns']}")
        print(f"[DONE] Report: {args.out.resolve()}")
        print("[CAUTION] A synthetic positive control is not lottery evidence")
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report.get('schema') not in ('lotto7x7.visual.v1', 'lotto7x7.learned_image.v1', DIAGNOSTICS_SCHEMA, LOSS_STUDY_SCHEMA) or args.count < 1:
            raise ValueError('unsupported report schema or count')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(gallery(report, args.count), encoding='utf-8')
        print(f"[DONE] HTML: {args.out.resolve()}")


if __name__ == '__main__':
    main()
