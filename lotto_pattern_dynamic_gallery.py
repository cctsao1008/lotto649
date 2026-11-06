"""Phase 7C static HTML view for changing 7x7 patterns and their controls."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from lotto_pattern_gallery import svg_mask

VERSION = "0.1.0"


def render_gallery(report: dict, *, max_draws: int = 40) -> str:
    """Visualize the observed and predicted image sequence in one HTML file."""
    if (report.get("schema") != "lotto649.phase7c.dynamic_pattern_replay.v1"
            or not isinstance(max_draws, int) or max_draws < 1):
        raise ValueError("requires Phase 7C replay JSON and positive max_draws")
    methods = report["methods"]
    styles = """
    *{box-sizing:border-box}body{font:14px system-ui,-apple-system,sans-serif;
    background:#f5f7fb;color:#172a42;margin:0}main{max-width:1240px;
    padding:28px 18px 70px;margin:auto}
    h1{font-size:26px}h2{margin:32px 0 14px;font-size:19px}
    p{line-height:1.55}small{color:#52667c}.sub{color:#516680}
    .banner,.card,.gallery-card{background:white;border:1px solid #dbe5f0;
    border-radius:12px;padding:16px}
    .banner{margin:18px 0}
    table{background:white;width:100%;border-collapse:collapse}
    th,td{padding:10px;border-bottom:1px solid #e2eaf4;text-align:left}
    .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}
    .pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px}
    .gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:12px}
    .img-cell{min-width:0}.img-cell svg{width:100%;max-width:136px}
    .gallery-card h3{margin:0 0 11px}.img-cell h4{margin:0 0 8px;font-size:13px}
    .img-cell p{font-size:12px;margin:5px 0;color:#52667c}
    .metric{font-variant-numeric:tabular-nums;font-weight:650}
    @media(max-width:600px){.gallery{grid-template-columns:1fr}.pair{grid-template-columns:repeat(2,1fr)}}
    @media print{body{background:white}.gallery-card,.card{break-inside:avoid}}
    """
    table_rows = []
    for method in methods:
        r = report["evaluation_summary"][method]
        table_rows.append(
            "<tr>"
            f"<th>{html.escape(method)}</th>"
            f"<td>{r['mean_raw']['composite']:.4f}</td>"
            f"<td>{r['mean_iid_expected']['composite']:.4f}</td>"
            f"<td>{r['mean_chance_adjusted']['composite']:+.4f}</td>"
            f"<td>{r['diversity']['unique_pixel_patterns']}</td>"
            f"<td>{r['diversity']['unique_shape_classes_d4_translate']}</td>"
            f"<td>{r['diversity']['top_pattern_fraction']:.1%}</td>"
            "</tr>"
        )
    audit_rows = []
    for name, group in report["temporal_order_audit"]["segments"].items():
        r = group["lags"]["1"]
        audit_rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{r['observed_mean_cosine']:+.4f}</td>"
            f"<td>{r['shuffled_mean_cosine']:+.4f}</td>"
            f"<td>{r['empirical_two_sided_tail']:.3f}</td>"
            "</tr>"
        )
    comparisons = []
    for row in report["individual_draws"][:max_draws]:
        cells = [
            '<section class="img-cell"><h4>Observed</h4>'
            + svg_mask(row["actual_pattern"], "Observed six-pixel mask")
            + "<p>" + " ".join(f"{x:02d}" for x in row["actual_main_six"]) + "</p>"
            + "</section>"
        ]
        for method in methods:
            p = row["methods"][method]
            cells.append(
                f'<section class="img-cell"><h4>{html.escape(method)}</h4>'
                + svg_mask(p["predicted_pattern"], f"Predicted by {method}")
                + "<p>" + " ".join(f"{x:02d}" for x in p["predicted_main_six"]) + "</p>"
                + '<p class="metric">Adjusted '
                + f"{p['adjusted_similarity']['composite']:+.3f}</p>"
                + "<p>Shape D4 "
                + f"{p['raw_similarity']['shape_d4']:.3f}</p>"
                + "</section>"
            )
        comparisons.append(
            '<article class="gallery-card">'
            f'<h3>Draw {int(row["draw_id"])} · {int(row["date"])}</h3>'
            + '<div class="pair">' + "".join(cells) + "</div></article>"
        )
    content = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Lotto649 Phase 7C Dynamic Image Gallery</title>",
        f"<style>{styles}</style></head><body><main>",
        "<h1>Phase 7C · Dynamic 7×7 pattern analysis</h1>",
        "<div class='banner'>",
        f"<p>Draws: <b>{int(report['evaluated_draws'])}</b> · "
        f"Historical period: <b>{html.escape(str(report['year']))}</b> · "
        f"Validation-selected method: <b>{html.escape(report['validation_selected'])}</b></p>",
        "<p>Unlike the older fixed-cluster prototypes, the dynamic approaches "
        "adapt their six-pixel output to recently observed image sequences.</p>",
        "<small>Exploratory previously analyzed history; similarity is not "
        "evidence of improved prize odds.</small></div>",
        "<h2>Forecast similarity and pattern diversity</h2>",
        "<table><thead><tr><th>Method</th><th>Raw</th><th>IID expectation</th>"
        "<th>Adjusted</th><th>Unique grids</th><th>Unique shapes</th>"
        "<th>Top grid share</th></tr></thead><tbody>",
        *table_rows, "</tbody></table>",
        "<h2>Time-order permutation control</h2>",
        "<p class='sub'>Within each year, retain the six-pixel images but shuffle "
        "their chronological order. Values are cosine correlations of train-"
        "normalized image features, not the main shape score.</p>",
        "<table><thead><tr><th>Segment</th><th>Observed lag 1</th>"
        "<th>Shuffled mean</th><th>Exploratory two-sided tail</th>"
        "</tr></thead><tbody>", *audit_rows, "</tbody></table>",
        "<h2>Predicted versus observed changing shapes</h2>",
        '<div class="gallery">', *comparisons, "</div>",
        "<p class='sub'>A similar-looking shape may have zero matching numbers. "
        "All pattern-score gains must exceed the conditional IID expectation. "
        "No previous neural or full-six-match model is used.</p>",
        "</main></body></html>",
    ]
    return "".join(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render dynamic 7x7 forecasts and temporal audit as HTML")
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase7c_dynamic_replay.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/neural/phase7c_dynamic_gallery.html"))
    parser.add_argument("--max-draws", type=int, default=40)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_gallery(report, max_draws=args.max_draws),
                        encoding="utf-8")
    print(f"Lotto649 Phase 7C Dynamic Image Gallery v{VERSION}")
    print(f"[DONE] HTML: {args.out.resolve()}")


if __name__ == "__main__":
    main()
