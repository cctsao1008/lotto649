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
from .image_structure_v05 import image_study, image_gallery as image_structure_gallery
from .image_compare_v05 import image_compare, image_compare_gallery, MODELS as IMAGE_MODELS
from .image_dynamics_v051 import image_dynamics, image_dynamics_gallery
from .candidate_pool_v06 import create_shape_pool, shape_pool_gallery, POLICIES as SHAPE_POLICIES
from .bet_selection_v06 import select_bets
from .coverage_audit_v06 import audit_selection
from .baseline_cli_v07 import configure_parsers as configure_v07_parsers, run as run_v07
from .robustness_cli_v071 import configure_parsers as configure_v071_parsers, run as run_v071
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
    structure = action.add_parser("image-study", help="v0.5 image morphology, DCT/FFT, temporal and layout controls")
    structure.add_argument("--data", type=Path, default=DATA)
    structure.add_argument("--permutations", type=int, default=499)
    structure.add_argument("--null-images", type=int, default=4096)
    structure.add_argument("--mapping-seeds", type=int, default=2)
    structure.add_argument("--lags", default="1,2,3,5,10")
    structure.add_argument("--seed", type=int, default=649)
    structure.add_argument("--out", type=Path, default=Path("results/image_structure_v05.json"))
    structure.add_argument("--html-out", type=Path, default=Path("results/image_structure_v05.html"))
    comparison = action.add_parser("image-compare", help="v0.5 four image-derived sequence representations")
    comparison.add_argument("--data", type=Path, default=DATA)
    comparison.add_argument("--window", type=int, default=5)
    comparison.add_argument("--seeds", type=int, default=3)
    comparison.add_argument("--seed", type=int, default=649)
    comparison.add_argument("--epochs", type=int, default=45)
    comparison.add_argument("--patience", type=int, default=8)
    comparison.add_argument("--batch-size", type=int, default=64)
    comparison.add_argument("--learning-rate", type=float, default=0.003)
    comparison.add_argument("--embedding", type=int, default=24)
    comparison.add_argument("--hidden", type=int, default=32)
    comparison.add_argument("--bootstrap", type=int, default=400)
    comparison.add_argument("--out", type=Path, default=Path("results/image_compare_v05.json"))
    comparison.add_argument("--html-out", type=Path, default=Path("results/image_compare_v05.html"))
    dyn = action.add_parser("image-dynamics", help="v0.5.1 transitions, morphology distributions, non-DC spectra, similarity matrices")
    dyn.add_argument("--data", type=Path, default=DATA)
    dyn.add_argument("--seed", type=int, default=649)
    dyn.add_argument("--null-images", type=int, default=4096)
    dyn.add_argument("--mapping-seeds", type=int, default=2)
    dyn.add_argument("--transitions", type=int, default=12)
    dyn.add_argument("--matrix-count", type=int, default=24)
    dyn.add_argument("--null-shape-pairs", type=int, default=128)
    dyn.add_argument("--compare-report", type=Path, default=None,
                     help="Read-only historical image-compare JSON; rejects mismatched dataset")
    dyn.add_argument("--out", type=Path, default=Path("results/image_dynamics_v051.json"))
    dyn.add_argument("--html-out", type=Path, default=Path("results/image_dynamics_v051.html"))
    sp = action.add_parser("shape-pool", help="v0.6 Phase 1 image shape families and candidate pool")
    sp.add_argument("--data", type=Path, default=DATA)
    sp.add_argument("--size", type=int, default=100)
    sp.add_argument("--families", type=int, default=20)
    sp.add_argument("--family-policy", choices=SHAPE_POLICIES, default="uniform-family")
    sp.add_argument("--seed", type=int, default=649)
    sp.add_argument("--sample-count", type=int, default=5)
    sp.add_argument("--out", type=Path, default=Path("results/shape_pool_v06.json"))
    sp.add_argument("--html-out", type=Path, default=Path("results/shape_pool_v06.html"))
    bs = action.add_parser("bet-select", help="v0.6 select 1..5 tickets from one frozen shape pool")
    bs.add_argument("--pool", type=Path, required=True)
    bs.add_argument("--count", type=int, default=5)
    bs.add_argument("--strategy", choices=("all", "random", "shape-diverse", "coverage-optimized"), default="all")
    bs.add_argument("--seed", type=int, default=1658)
    bs.add_argument("--out", type=Path, default=Path("results/bet_selection_v06.json"))
    ca = action.add_parser("coverage-audit", help="v0.6 exact 1..5 ticket coverage; MC for research pool")
    ca.add_argument("--bets", type=Path, required=True)
    ca.add_argument("--samples", type=int, default=50000,
                    help="IID sample count for pool-wide partial-hit estimates only")
    ca.add_argument("--seed", type=int, default=20261108)
    ca.add_argument("--draw", type=str, default=None,
                    help="Optional comma-separated known six-number draw, post-draw scoring only")
    ca.add_argument("--out", type=Path, default=Path("results/coverage_audit_v06.json"))
    configure_v07_parsers(action)
    configure_v071_parsers(action)
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
    elif args.command == "shape-pool":
        print("Lotto7x7 v0.6 Shape Pool — Phase 1 (CPU)", flush=True)
        images, _, draw_ids = load_history(args.data)
        report = create_shape_pool(
            images, draw_ids, pool_size=args.size, family_count=args.families,
            family_policy=args.family_policy, seed=args.seed,
            sample_count=args.sample_count)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(shape_pool_gallery(report), encoding="utf-8")
        print(f"[DATA] {report['source']['draw_count']} historical images, "
              f"cutoff={report['source']['cutoff_draw_id']}")
        print(f"[SHAPES] {report['shape_library_family_count']} historical shape families; "
              f"{len(report['selected_families'])} selected")
        print(f"[POOL] {report['candidate_count']} distinct six-number combinations")
        for i, item in enumerate(report["random_preview"], start=1):
            numbers = " ".join(f"{n:02d}" for n in item["numbers"])
            print(f"[PREVIEW {i}] Pool #{item['pool_index']+1:03d}: {numbers}")
        chance = report["fair_independent_reference"]
        print(f"[FAIR] Pool 6/6 coverage={chance['candidate_pool_exact_six_coverage']:.8%}; "
              f"preview={chance['preview_exact_six_coverage']:.8%}")
        print(f"[HASH] {report['candidate_sha256']}")
        print(f"[DONE] JSON: {args.out.resolve()}")
        print(f"[DONE] HTML: {args.html_out.resolve()}")
        print("[CAUTION] Unoptimized random preview, not a forecast or an instruction to purchase")
    elif args.command == "bet-select":
        pool = json.loads(args.pool.read_text(encoding="utf-8"))
        report = select_bets(pool, count=args.count, strategy=args.strategy, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[POOL] {report['pool_candidate_count']} frozen candidates; "
              f"SHA256={report['pool_sha256']}")
        for name, item in report["strategies"].items():
            print(f"[STRATEGY] {name} — {len(item['bets'])} alternative bets")
            for j, bet in enumerate(item["bets"], start=1):
                numbers = " ".join(f"{n:02d}" for n in bet["numbers"])
                print(f"  {j}. Pool #{bet['pool_index']+1:03d}: {numbers}")
            exact = item["exact_fair_coverage"]["at_least"]
            print(f"  [EXACT] >=3={exact['3']:.6%} >=4={exact['4']:.6%} "
                  f">=5={exact['5']:.6%} 6/6={exact['6']:.8%}")
        print(f"[DONE] Selection JSON: {args.out.resolve()}")
        print("[CAUTION] These are alternative research portfolios; do not sum their ticket counts")
    elif args.command == "coverage-audit":
        selections = json.loads(args.bets.read_text(encoding="utf-8"))
        known_draw = None if args.draw is None else tuple(int(v.strip()) for v in args.draw.split(","))
        report = audit_selection(selections, samples=args.samples, seed=args.seed, actual=known_draw)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[TICKETS] {report['ticket_count_per_strategy']} per strategy "
              f"(research candidate pool={report['research_pool_count']})")
        for name, item in report["strategies"].items():
            p = item["exact_fair_coverage"]["at_least"]
            print(f"[EXACT] {name}: >=3={p['3']:.6%} >=4={p['4']:.6%} "
                  f">=5={p['5']:.6%} 6/6={p['6']:.8%}")
        print("[PARETO] " + ", ".join(report["pareto_nondominated"]))
        print("[POOL] Partial-hit coverage: IID Monte Carlo, with CI (not purchased tickets)")
        print(f"[DONE] Coverage audit JSON: {args.out.resolve()}")
    elif args.command in ("baseline-study", "shadow-snapshot", "shadow-score"):
        run_v07(args)
    elif args.command in ("robustness-audit", "robustness-sweep"):
        run_v071(args)
    elif args.command == "image-dynamics":
        print("Lotto7x7 v0.5.1 Visual Image Dynamics (CPU)", flush=True)
        report = image_dynamics(
            args.data, seed=args.seed, null_images=args.null_images,
            mapping_seeds=args.mapping_seeds, transitions=args.transitions,
            matrix_count=args.matrix_count, null_shape_pairs=args.null_shape_pairs,
            comparison_report=args.compare_report,
            on_progress=lambda line: print(line, flush=True))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(image_dynamics_gallery(report), encoding="utf-8")
        print(f"[IMAGES] {report['draws']} historical six-point bitmaps")
        chance = report["iid_baselines"]
        print(f"[IID] Exact common pixels={chance['expected_common_pixels']:.6f}; "
              f"non-DC mean power={chance['non_dc_dct_expected_power']:.6f}")
        for item in report["recent_transitions"][-3:]:
            print(f"[TRANSITION] {item['previous_draw_id']} -> {item['current_draw_id']}: "
                  f"retained={item['retained']}, exact IoU={item['exact_pixel_jaccard']:.4f}, "
                  f"D4 shape={item['d4_translation_shape_similarity']:.4f}")
        if report["prediction_gallery"] is not None:
            print(f"[GALLERY] {len(report['prediction_gallery']['examples'])} saved historical comparison examples")
        print(f"[DONE] JSON: {args.out.resolve()}")
        print(f"[DONE] HTML: {args.html_out.resolve()}")
        print("[CAUTION] Descriptive image structure is not evidence of a prediction advantage")
    elif args.command == "image-study":
        print("Lotto7x7 Image Structure & Dynamics v0.5 (CPU)", flush=True)
        lags = tuple(int(value.strip()) for value in args.lags.split(","))
        report = image_study(
            args.data, permutations=args.permutations,
            null_images=args.null_images, mapping_seeds=args.mapping_seeds,
            lags=lags, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(image_structure_gallery(report), encoding="utf-8")
        print(f"[IMAGES] {report['draws']} historical six-pixel images")
        print(f"[IID] Expected common pixels for two fair six-subsets: {report['exact_iid_expected_overlap']:.6f}")
        for mapped in report["mapping_comparisons"]:
            key = f"{mapped['mapping']}({mapped['mapping_seed']})"
            lag = mapped["temporal"][str(lags[0])]
            overlap = lag["pixel_overlap"]
            dct = lag["low_dct_cosine"]
            print(f"[LAYOUT] {key}: lag-{lags[0]} overlap={overlap['observed']:.4f} "
                  f"null={overlap['null_mean']:.4f} p_adj={overlap['bonferroni_p']:.4f}; "
                  f"DCT-cosine={dct['observed']:.4f} "
                  f"null={dct['null_mean']:.4f} p_adj={dct['bonferroni_p']:.4f}")
        print(f"[DONE] JSON: {args.out.resolve()}")
        print(f"[DONE] HTML: {args.html_out.resolve()}")
        print("[CAUTION] Arbitrary image layouts; retrospective patterns are not predictive evidence")
    elif args.command == "image-compare":
        print("Lotto7x7 Image Representation Comparison v0.5 (CPU)", flush=True)
        report = image_compare(
            args.data, seeds=args.seeds, seed=args.seed, window=args.window,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            embedding=args.embedding, hidden=args.hidden,
            bootstrap=args.bootstrap,
            on_progress=lambda message: print(message, flush=True))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(image_compare_gallery(report), encoding="utf-8")
        print(f"[EVALUATED] {report['evaluation_draws']} historical image targets")
        null = report["chance_baselines"]
        print(f"[IID] Pixel Brier={null['pixel_brier_uniform']:.6f}; mean hits={null['expected_hits_iid']:.6f}")
        print("Method                   Pixel Brier    Mean Hits    Shape Sim    Parameters")
        for name in IMAGE_MODELS:
            value = report["summary"][name]
            print(f"{name:24s} {value['mean_pixel_brier']:12.6f} "
                  f"{value['mean_hits']:12.4f} {value['mean_shape_similarity']:12.4f} "
                  f"{value['parameter_count']:12d}")
        for name, item in report["paired_intervals"].items():
            low, high = item["interval_95"]
            print(f"[PAIRED] {name}: {item['mean']:+.6f} "
                  f"interval=[{low:+.6f}, {high:+.6f}]")
        print(f"[DONE] JSON: {args.out.resolve()}")
        print(f"[DONE] HTML: {args.html_out.resolve()}")
        print("[CAUTION] Historical tests have been inspected and are exploratory only")
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
