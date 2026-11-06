"""Phase 7B static HTML gallery for inspecting forecast and observed 7x7 masks.

No JavaScript, external resources, image models, or wagering suggestions.
The visual report reads only existing Phase 7B replay JSON.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

VERSION = "0.1.0"


def svg_mask(pattern: list[str], label: str) -> str:
    """Render a 7x7 mask as small accessible inline SVG without interpolation."""
    if (len(pattern) != 7 or any(len(row) != 7
            or set(row) - {"#", "."} for row in pattern)
            or sum(row.count("#") for row in pattern) != 6):
        raise ValueError("invalid six-point seven-row image")
    cells = []
    for r, row in enumerate(pattern):
        for c, char in enumerate(row):
            x, y = 5 + c * 20, 5 + r * 20
            fill = "#173a63" if char == "#" else "#eef3f8"
            cells.append(f'<rect x="{x}" y="{y}" width="18" height="18" '
                         f'rx="3" fill="{fill}"/>')
            if char == "#":
                cells.append(f'<circle cx="{x+9}" cy="{y+9}" r="3" fill="#fff"/>')
    title = html.escape(label)
    return (f'<svg viewBox="0 0 150 150" role="img" '
            f'aria-label="{title}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(cells) + "</svg>")


def render_gallery(report: dict, *, method: str, max_draws: int = 40) -> str:
    """Return a standalone document summarizing shape similarity and prototypes."""
    if report.get("schema") != "lotto649.phase7b.pattern_similarity.v1":
        raise ValueError("requires Phase 7B similarity JSON, not old Phase 7 score JSON")
    if method not in report["methods"] or max_draws < 1:
        raise ValueError("method must exist and max_draws must be positive")
    summary = report["evaluation_summary"]
    row_html = []
    for name in report["methods"]:
        s = summary[name]
        row_html.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{s['mean_similarity']['composite']:.4f}</td>"
            f"<td>{s['mean_iid_expected_for_predictions']['composite']:.4f}</td>"
            f"<td class='delta'>{s['mean_chance_adjusted_composite']:+.4f}</td>"
            "</tr>"
        )
    families = []
    for p in report["train_only_pattern_families"]:
        mask = ["".join("#" if v else "." for v in row)
                for row in p["representative_image"]]
        families.append(
            '<section class="tile">'
            f'<h3>Cluster {int(p["cluster_id"])}</h3>'
            + svg_mask(mask, f"Pattern family {p['cluster_id']}")
            + f'<small>{int(p["train_members"])} training draws</small>'
            + '</section>'
        )
    examples = []
    for row in report["individual_draws"][:max_draws]:
        s = row["methods"][method]
        actual = " ".join(f"{x:02d}" for x in row["actual_main_six"])
        predicted = " ".join(f"{x:02d}" for x in s["predicted_main_six"])
        examples.append(
            '<article class="example">'
            f'<h3>Draw {int(row["draw_id"])} | {int(row["date"])}</h3>'
            '<div class="pair">'
            '<section><h4>Observed image</h4>'
            + svg_mask(row["actual_pattern"], "Actual six-point image")
            + f'<small>{html.escape(actual)}</small></section>'
            '<section><h4>Predicted image</h4>'
            + svg_mask(s["predicted_pattern"], "Predicted six-point image")
            + f'<small>{html.escape(predicted)}</small></section>'
            '</div>'
            f'<p>Raw similarity <b>{s["similarity"]["composite"]:.3f}</b> · '
            f'Expected vs IID <b>{s["iid_expected_for_prediction"]["composite"]:.3f}</b> · '
            f'Adjusted <b>{s["chance_adjusted_composite"]:+.3f}</b></p>'
            f'<p class="sub">D4 shape: {s["similarity"]["shape_d4"]:.3f} | '
            f'Pixel: {s["similarity"]["pixel_jaccard"]:.3f} | '
            f'DCT low-band: {s["similarity"]["low_dct"]:.3f}</p>'
            '</article>'
        )
    css = """
    *{box-sizing:border-box}body{font:15px system-ui,-apple-system,sans-serif;
    color:#192b3d;background:#f5f8fc;margin:0}
    main{max-width:1090px;margin:auto;padding:30px 18px 65px}
    h1{font-size:28px;line-height:1.2}h2{margin-top:36px}
    p{line-height:1.5}.sub,small{color:#53667e}
    .banner{background:white;border:1px solid #dce5f0;border-radius:12px;
    padding:16px 22px;margin:18px 0}
    table{border-collapse:collapse;width:100%;background:white;
    border-radius:12px;overflow:hidden}
    td,th{text-align:left;padding:10px 14px;border-bottom:1px solid #e6edf5}
    .delta{font-variant-numeric:tabular-nums;font-weight:700}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:14px}
    .tile,.example{background:white;border:1px solid #dce5f0;
    padding:16px;border-radius:12px}
    .tile svg{max-width:145px;width:100%;display:block}
    .tile small{display:block}
    .examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}
    .pair{display:flex;gap:14px;align-items:flex-start}
    .pair section{width:50%}.pair svg{max-width:150px;width:100%;display:block}
    h3{font-size:16px;margin:0 0 10px}h4{font-size:13px;margin:5px 0}
    .pair small{font-variant-numeric:tabular-nums}
    @media print{body{background:white}.tile,.example{break-inside:avoid}}
    """
    year = report["year"]
    doc = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Lotto649 Phase 7B Image Similarity Gallery</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>Phase 7B · Image-pattern similarity</h1>",
        "<div class='banner'>",
        f"<p>Period: <b>{html.escape(str(year))}</b> · "
        f"Observed draws: <b>{int(report['evaluated_draws'])}</b> · "
        f"Displayed method: <b>{html.escape(method)}</b></p>",
        "<p>Primary outcome: image similarity adjusted for the same predicted "
        "pattern against IID random targets. Scores do not imply lottery winnings.</p>",
        "<p class='sub'>Experimental replay of previously seen history; "
        "not a prospective blind study.</p></div>",
        "<h2>Model-free image pattern approaches</h2>",
        "<table><thead><tr><th>Method</th><th>Raw</th><th>Chance expectation</th>"
        "<th>Adjusted</th></tr></thead><tbody>",
        *row_html, "</tbody></table>",
        "<h2>Training image families</h2>",
        "<div class='grid'>", *families, "</div>",
        f"<h2>Historical pattern comparisons (first {min(max_draws, len(report['individual_draws']))})</h2>",
        "<div class='examples'>", *examples, "</div>",
        "<p class='sub'># = active pixel. Same-shape patterns may contain no "
        "identical lottery numbers. No legacy logits, MLP or exact-match "
        "objective is part of Phase 7B.</p>",
        "</main></body></html>",
    ]
    return "".join(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Phase 7B visual similarity gallery")
    parser.add_argument("--report", type=Path,
                        default=Path("data/neural/phase7b_pattern_similarity.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/neural/phase7b_pattern_gallery.html"))
    parser.add_argument("--method", default=None)
    parser.add_argument("--max-draws", type=int, default=40)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    method = args.method or report["validation_selected"]
    document = render_gallery(report, method=method, max_draws=args.max_draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    print(f"Lotto649 Phase 7B Image Gallery v{VERSION}")
    print(f"[METHOD] {method}")
    print(f"[DONE] HTML: {args.out.resolve()}")


if __name__ == "__main__":
    main()
