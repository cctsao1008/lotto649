"""Lotto7x7 v0.5.1 — image-first visual dynamics and IID controls.

This is a *descriptive* audit, not a forecasting model. Every input is a fixed
six-point image; geometry depends on an explicitly recorded number-to-pixel map.
The model-gallery extension only displays previously saved historical predictions.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .core import decode, iid_images, similarity, validate
from .image_structure_v05 import (
    MORPHOLOGY_NAMES, check_images, describe_images, map_images,
)
from .research import DATA, calendar_years, load_history

SCHEMA = "lotto7x7.visual_image_dynamics.v0.5.1"
COMPARE_SCHEMA = "lotto7x7.image_compare.v0.5"
SELECTED = (
    ("radial_spread", "Radial spread"),
    ("pair_distance_mean", "Mean pair distance"),
    ("bounding_box_fraction", "Bounding box area"),
    ("neighbor_edges_8", "8-neighbor edge pairs"),
    ("components_8", "8-connected components"),
    ("row_entropy", "Row occupancy entropy"),
)
IID_OCCUPANCY = 6.0 / 49.0
IID_OVERLAP = 36.0 / 49.0
IID_NON_DC_POWER = (6.0 * 43.0) / (49.0 * 48.0)


def transition(previous: np.ndarray, current: np.ndarray) -> dict:
    """Position-aware and position-free comparison for two real 7x7 masks."""
    before, after = check_images(np.stack([previous, current]))
    kept = before & after
    removed = before & (~after.astype(bool))
    added = after & (~before.astype(bool))
    overlap = int(kept.sum())
    if int(removed.sum()) != 6-overlap or int(added.sum()) != 6-overlap:
        raise RuntimeError("non-conserving six-pixel transition")
    shape = float(similarity(before, after)["shape"])
    return {
        "retained": overlap, "removed": 6-overlap, "added": 6-overlap,
        "exact_pixel_jaccard": overlap/(12.0-overlap),
        "d4_translation_shape_similarity": shape,
        "previous_mask": before.tolist(), "current_mask": after.tolist(),
        "transition_code": (2*kept + 3*removed + 4*added).astype(np.uint8).tolist(),
    }


def _distribution(values: np.ndarray, iid_values: np.ndarray, *, bins: int = 14) -> dict:
    """Normalize actual and IID histograms independently to probabilities."""
    a, b = np.asarray(values, dtype=np.float64), np.asarray(iid_values, dtype=np.float64)
    if (a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b) or
            not np.isfinite(a).all() or not np.isfinite(b).all() or bins < 2):
        raise ValueError("expected finite, non-empty feature samples")
    low, high = min(a.min(),b.min()), max(a.max(),b.max())
    if abs(high-low) < 1e-12:
        low -= 0.5
        high += 0.5
    padding = (high-low) * 0.002
    edges = np.linspace(low-padding,high+padding,bins+1)
    observed = np.histogram(a,bins=edges)[0]/len(a)
    control = np.histogram(b,bins=edges)[0]/len(b)
    null_std = float(b.std(ddof=1)) if len(b)>1 else 0.0
    # Descriptive reference only; z does not correct multiple feature/map looks.
    error = null_std*np.sqrt(1.0/len(a)+1.0/len(b))
    return {
        "observed_mean": float(a.mean()), "iid_mean": float(b.mean()),
        "delta": float(a.mean()-b.mean()),
        "descriptive_z_mc": float((a.mean()-b.mean())/error) if error>1e-12 else None,
        "iid_std": null_std,
        "edges": edges.tolist(), "observed_hist": observed.tolist(),
        "iid_hist": control.tolist(),
    }


def _layouts(images: np.ndarray, iid: np.ndarray, *, seed: int,
             mapping_seeds: int, on_progress: Callable[[str],None] | None) -> list[dict]:
    mappings = [("sequential", seed), ("snake", seed)] + [
        ("permuted",seed+i+1) for i in range(mapping_seeds)]
    output = []
    for kind,value in mappings:
        a = map_images(images,kind=kind,seed=value)
        b = map_images(iid,kind=kind,seed=value)
        a_features = describe_images(a)
        b_features = describe_images(b)
        morphological = {}
        for key,title in SELECTED:
            i=MORPHOLOGY_NAMES.index(key)
            morphological[key] = {"title":title,**_distribution(
                a_features["morphology"][:,i], b_features["morphology"][:,i])}
        occupancy = a.mean(axis=0)
        # Use the *exact* fair marginal expectation, rather than the noise in
        # a finite independently sampled IID occupancy heatmap.
        occupancy_z = (occupancy-IID_OCCUPANCY) / np.sqrt(
            IID_OCCUPANCY*(1-IID_OCCUPANCY)/len(a))
        result = {
            "mapping":kind, "seed":int(value),
            "occupancy":occupancy.tolist(),
            "iid_exact_occupancy":IID_OCCUPANCY,
            "occupancy_delta":(occupancy-IID_OCCUPANCY).tolist(),
            "occupancy_z_unadjusted":occupancy_z.tolist(),
            "morphology":morphological,
            "spectral":_spectral_from_dct(a_features["dct"],b_features["dct"]),
        }
        output.append(result)
        if on_progress: on_progress(f"[LAYOUT] {kind}({value}): descriptors and non-DC spectrum complete")
    return output


def _spectral_from_dct(actual_dct: np.ndarray, iid_dct: np.ndarray) -> dict:
    observed = np.mean(actual_dct**2,axis=0)
    random = np.mean(iid_dct**2,axis=0)
    diff = observed-random
    observed[0,0] = random[0,0] = diff[0,0] = 0.0
    return {"observed_non_dc_power":observed.tolist(),
            "iid_non_dc_power":random.tolist(),
            "delta_non_dc_power":diff.tolist(),
            "exact_iid_expected_power_each_non_dc":IID_NON_DC_POWER,
            "excluded_dc":True}


def similarity_matrices(images: np.ndarray, ids: np.ndarray, *, count: int = 24,
                        seed: int = 649, null_pairs: int = 128) -> dict:
    """Recent pairwise exact overlaps and D4/translation tolerant shapes."""
    x=check_images(images)
    draws=np.asarray(ids,dtype=np.int64)
    if (len(draws)!=len(x) or count<2 or count>len(x) or
            null_pairs<16 or seed<0):
        raise ValueError("invalid similarity matrix arguments")
    subset=x[-count:]
    flat=subset.reshape(count,49).astype(np.int16)
    exact=(flat@flat.T).astype(np.uint8)
    invariant=np.eye(count,dtype=np.float64)
    for i in range(count):
        for j in range(i):
            value=float(similarity(subset[i],subset[j])["shape"])
            invariant[i,j]=invariant[j,i]=value
    controls=iid_images(2*null_pairs,seed+7491)
    null_shape=float(np.mean([
        similarity(controls[2*k], controls[2*k+1])["shape"]
        for k in range(null_pairs)
    ]))
    return {"draw_ids":draws[-count:].astype(int).tolist(),
            "exact_common_pixels":exact.tolist(),
            "shape_similarity_d4_translation":invariant.tolist(),
            "iid_expected_exact_common_pixels":IID_OVERLAP,
            "iid_sample_mean_shape_similarity":null_shape,
            "iid_null_shape_pairs":null_pairs,
            "note":"Shape comparison tolerates D4 rotations and translations; exact pixels do not."}


def _model_gallery(previous: dict, images: np.ndarray, ids: np.ndarray,
                   input_hash: str) -> dict:
    """Read-only link to prior v0.5 predictions; do not retrain or select."""
    if previous.get("schema") != COMPARE_SCHEMA or previous.get("data_sha256") != input_hash:
        raise ValueError("prediction report schema or historical dataset hash differs")
    actual_by_id = {int(k):im for k,im in zip(ids,images)}
    result=[]
    for row in previous.get("examples",[]):
        draw_id=int(row["draw_id"])
        if draw_id not in actual_by_id:
            raise ValueError("prediction gallery draw ID does not exist")
        expected=actual_by_id[draw_id]
        if not np.array_equal(np.asarray(row["bitmap"]),expected):
            raise ValueError("prediction gallery actual image differs from dataset")
        forecasts={}
        for name,info in row["predictions"].items():
            prediction=validate(np.asarray(info["bitmap"]))
            t=transition(prediction,expected)
            if sorted(info["numbers"])!=decode(prediction) or int(info["hits"])!=t["retained"]:
                raise ValueError("prediction gallery stored results are inconsistent")
            forecasts[name]={"mask":prediction.tolist(),"numbers":decode(prediction),
                             "exact_hits":t["retained"],
                             "exact_jaccard":t["exact_pixel_jaccard"],
                             "shape_similarity":t["d4_translation_shape_similarity"],
                             "transition_code":t["transition_code"]}
        result.append({"draw_id":draw_id,"actual":expected.tolist(),
                       "numbers":decode(expected),"methods":forecasts})
    if not result: raise ValueError("no valid prediction examples")
    return {"schema":COMPARE_SCHEMA,"examples":result,
            "evaluation_draws":previous.get("evaluation_draws"),
            "note":"Archived v0.5 historical forecast examples, not prospective predictions."}


def image_dynamics(path: Path = DATA, *, seed: int = 649,
                   null_images: int = 4096, mapping_seeds: int = 2,
                   transitions: int = 12, matrix_count: int = 24,
                   null_shape_pairs: int = 128,
                   comparison_report: Path | None = None,
                   on_progress: Callable[[str],None] | None = None) -> dict:
    if (seed<0 or null_images<200 or not 0<=mapping_seeds<=8 or
            not 1<=transitions<=100 or not 2<=matrix_count<=80 or
            null_shape_pairs<16):
        raise ValueError("invalid v0.5.1 configuration")
    image_set, dates, ids = load_history(path)
    images=check_images(image_set)
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)!=len(images) or len(images)<100 or np.any(np.diff(ids)<=0):
        raise ValueError("invalid chronological draw records")
    digest=hashlib.sha256(images.tobytes()+ids.tobytes()).hexdigest()
    iid=iid_images(null_images,seed+5801)
    mapped=_layouts(images,iid,seed=seed,mapping_seeds=mapping_seeds,
                    on_progress=on_progress)
    recent=[]
    start=max(1,len(images)-transitions)
    for i in range(start,len(images)):
        item=transition(images[i-1],images[i])
        recent.append({"previous_draw_id":int(ids[i-1]),
                       "current_draw_id":int(ids[i]),
                       "previous_numbers":decode(images[i-1]),
                       "current_numbers":decode(images[i]),**item})
    mat=similarity_matrices(images,ids,count=min(matrix_count,len(images)),
                            seed=seed,null_pairs=null_shape_pairs)
    result={
        "schema":SCHEMA,"draws":int(len(images)),
        "data_sha256":digest,"first_draw_id":int(ids[0]),
        "last_draw_id":int(ids[-1]),
        "year_range":[int(v) for v in (calendar_years(ids)[0],calendar_years(ids)[-1])],
        "configuration":{"seed":seed,"null_images":null_images,
                         "mapping_seeds":mapping_seeds,"transitions":transitions,
                         "matrix_count":matrix_count,"null_shape_pairs":null_shape_pairs},
        "iid_baselines":{"pixel_probability":IID_OCCUPANCY,
                         "expected_common_pixels":IID_OVERLAP,
                         "non_dc_dct_expected_power":IID_NON_DC_POWER},
        "layouts":mapped,"recent_transitions":recent,
        "similarity_matrices":mat,
        "prediction_gallery":None,
        "limitations":[
            "All historical analysis is exploratory, not a future blind prediction.",
            "7x7 geometry, adjacency and spectrum depend on arbitrary number-to-cell mapping.",
            "IID reference samples follow uniform six-without-replacement, not independent pixels.",
            "Morphology deviations and occupancy z values are descriptive and unadjusted for multiple looks.",
            "D4/translation-invariant shape similarity cannot substitute for exact-number hits.",
            "A visually interesting transition or frequency difference does not establish forecastability.",
        ],
    }
    if comparison_report is not None:
        path=Path(comparison_report)
        result["prediction_gallery"]=_model_gallery(
            json.loads(path.read_text(encoding="utf-8")),images,ids,digest)
        if on_progress: on_progress(f"[GALLERY] Linked saved v0.5 forecast examples: {len(result['prediction_gallery']['examples'])}")
    return result


# Offline SVG and HTML. There are no remote requests, external scripts or fonts.
def _bitmap_svg(mask: list[list[int]], *, code: bool = False, cell: int = 18) -> str:
    arr=np.asarray(mask)
    if arr.shape!=(7,7):raise ValueError("invalid 7x7 visualization")
    colors={0:"#edf2f7", 1:"#346db2", 2:"#27886a",3:"#d18a47",4:"#3678b7"} if code else {0:"#edf2f7",1:"#346db2"}
    if not set(np.unique(arr)).issubset(colors):raise ValueError("invalid visualization pixels")
    squares=[]
    for r in range(7):
        for c in range(7):
            v=int(arr[r,c]);squares.append(f'<rect x="{c*cell+1}" y="{r*cell+1}" width="{cell-2}" height="{cell-2}" rx="3" fill="{colors[v]}"/>')
    size=7*cell
    return f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="7 by 7 image">'+''.join(squares)+'</svg>'


def _color(z:float, *, signed:bool, limit:float) -> str:
    t=min(abs(z)/max(limit,1e-12),1.0)
    if signed:
        # Orange for positive; blue for negative; off-white near zero.
        base=(207,109,69) if z>0 else (50,110,169)
    else:
        base=(41,100,159)
    rgb=tuple(int(246*(1-t)+v*t) for v in base)
    return "#"+"".join(f"{v:02x}" for v in rgb)


def _heatmap(values:list[list[float]], *, title:str, signed:bool,
             limit:float,dc:bool=False,cell:int=18) -> str:
    arr=np.asarray(values,dtype=np.float64)
    if arr.shape!=(7,7) or not np.isfinite(arr).all():
        raise ValueError("expected finite 7x7 heatmap")
    entries=[]
    for r in range(7):
        for c in range(7):
            z=float(arr[r,c]);not_dc=(r,c)!=(0,0)
            fill=_color(z,signed=signed,limit=limit) if not_dc or not dc else "#c4cad3"
            entries.append(f'<rect x="{c*cell+1}" y="{r*cell+1}" width="{cell-2}" height="{cell-2}" rx="3" fill="{fill}"><title>row {r+1}, col {c+1}: {z:+.5f}</title></rect>')
    if dc:entries.append('<path d="M4 4 L15 15 M15 4 L4 15" stroke="#66717e" stroke-width="1.7"/>')
    size=7*cell
    return f'<div class="hunit"><h4>{escape(title)}</h4><svg viewBox="0 0 {size} {size}" role="img">'+''.join(entries)+'</svg></div>'


def _histogram(entry:dict) -> str:
    actual=np.array(entry["observed_hist"]);iid=np.array(entry["iid_hist"])
    if len(actual)!=len(iid) or not len(actual):raise ValueError("invalid histogram")
    height=90;w=340;left=22;bottom=12;chart_h=height-bottom-5
    top=max(1e-8,actual.max(),iid.max())
    result=[]
    # Light bars for actual values and an orange outline for the IID reference.
    width=(w-2*left)/len(actual)
    for k,(a,b) in enumerate(zip(actual,iid)):
        x=left+k*width;y=5+(top-a)/top*chart_h
        result.append(f'<rect x="{x+1:.2f}" y="{y:.2f}" width="{width-2:.2f}" height="{5+chart_h-y:.2f}" fill="#4b80b6" opacity=".64"/>')
    pts=" ".join(f"{left+(k+.5)*width:.1f},{5+(top-b)/top*chart_h:.1f}" for k,b in enumerate(iid))
    return f'<svg viewBox="0 0 {w} {height}" aria-label="Histogram of actual versus IID shapes"><polyline points="{pts}" fill="none" stroke="#cf7546" stroke-width="2.5"/>'+''.join(result)+'</svg>'


def _matrix_heat(data:list[list[float]],ids:list[int], *, shape:bool) -> str:
    arr=np.asarray(data,dtype=np.float64)
    n=len(ids)
    if arr.shape!=(n,n) or n>80 or not np.isfinite(arr).all():raise ValueError("invalid matrix")
    cell=12 if n>=28 else 15
    parts=[]
    for r in range(n):
        for c in range(n):
            value=float(arr[r,c]);fill=_color(value,signed=False,limit=1 if shape else 6)
            parts.append(f'<rect x="{c*cell}" y="{r*cell}" width="{cell-1}" height="{cell-1}" fill="{fill}"><title>Draw {ids[r]} vs {ids[c]}: {value:.4f}</title></rect>')
    size=n*cell
    return f'<svg class="matrix" viewBox="0 0 {size} {size}" role="img" aria-label="Recent draw similarity matrix">'+''.join(parts)+'</svg>'


def image_dynamics_gallery(report:dict) -> str:
    """Produce a compact, print-friendly full offline report from audited data."""
    if report.get("schema")!=SCHEMA or not report.get("recent_transitions") or not report.get("layouts"):
        raise ValueError("invalid v0.5.1 report")
    layout=report["layouts"][0]
    if layout["mapping"]!="sequential":raise ValueError("first map must be sequential")
    occupancy=layout["occupancy_z_unadjusted"]
    spectra=layout["spectral"]
    maximum=max(abs(float(v)) for row in spectra["delta_non_dc_power"] for v in row)
    hmaps=(
        _heatmap(layout["occupancy"],title="Observed pixel frequency",signed=False,limit=.22)+
        _heatmap(layout["occupancy_delta"],title="Observed − fair 6/49",signed=True,limit=.04)+
        _heatmap(occupancy,title="Occupancy deviation (unadjusted z)",signed=True,limit=4)+
        _heatmap(spectra["observed_non_dc_power"],title="Observed non-DC DCT power",signed=False,limit=.20,dc=True)+
        _heatmap(spectra["iid_non_dc_power"],title="IID non-DC DCT power",signed=False,limit=.20,dc=True)+
        _heatmap(spectra["delta_non_dc_power"],title="Observed − IID DCT power",signed=True,limit=max(maximum,.01),dc=True)
    )
    transitions=[]
    for event in report["recent_transitions"]:
        transitions.append('<article class="transition"><h3>'+str(event["previous_draw_id"])+
                           ' → '+str(event["current_draw_id"])+
                           '</h3><div class="triple"><div><span>Previous</span>'+_bitmap_svg(event["previous_mask"])+
                           '</div><div><span>Changed pixels</span>'+_bitmap_svg(event["transition_code"],code=True)+
                           '</div><div><span>Current</span>'+_bitmap_svg(event["current_mask"])+
                           '</div></div><p class="numbers">'+escape(str(event["previous_numbers"]))+
                           ' → '+escape(str(event["current_numbers"]))+
                           f'</p><p><b>{event["retained"]}</b> retained · <b>{event["removed"]}</b> removed · '
                           f'<b>{event["added"]}</b> added</p>'
                           f'<p class="muted">Exact IoU {event["exact_pixel_jaccard"]:.3f} · '
                           f'D4 shape {event["d4_translation_shape_similarity"]:.3f}</p></article>')
    dists=[]
    for key,title in SELECTED:
        item=layout["morphology"][key]
        z=item["descriptive_z_mc"]
        ztext="n/a" if z is None else f"{z:+.2f}"
        dists.append('<article class="feature"><h3>'+escape(title)+'</h3>'+
                     _histogram(item)+
                     f'<p>Actual {item["observed_mean"]:.4f} · IID {item["iid_mean"]:.4f}</p>'+
                     f'<p class="muted">Difference {item["delta"]:+.4f} · MC z {ztext} (unadjusted)</p></article>')
    comparisons=[]
    for item in report["layouts"]:
        desc=item["morphology"]
        fmt=lambda key: desc[key]["delta"]
        comparisons.append('<tr><td>'+escape(item["mapping"])+f' / {item["seed"]}</td>'+
                           f'<td>{fmt("neighbor_edges_8"):+.4f}</td>'+
                           f'<td>{fmt("components_8"):+.4f}</td>'+
                           f'<td>{fmt("radial_spread"):+.4f}</td></tr>')
    matr=report["similarity_matrices"]
    matrix_html=('<div class="matrixrow"><div><h3>Exact shared pixel positions (0–6)</h3>'+
                 _matrix_heat(matr["exact_common_pixels"],matr["draw_ids"],shape=False)+
                 '</div><div><h3>Shape similarity (D4 + shift, 0–1)</h3>'+
                 _matrix_heat(matr["shape_similarity_d4_translation"],matr["draw_ids"],shape=True)+
                 '</div></div>')
    gallery=""
    if report.get("prediction_gallery") is not None:
        cases=[]
        for example in report["prediction_gallery"]["examples"]:
            cards=['<div><h4>Actual</h4>'+_bitmap_svg(example["actual"])+
                   '<p>'+escape(str(example["numbers"]))+'</p></div>']
            for name,item in example["methods"].items():
                cards.append('<div><h4>'+escape(name)+'</h4>'+_bitmap_svg(item["mask"])+
                             f'<p>{item["exact_hits"]} exact hits · shape {item["shape_similarity"]:.3f}</p></div>')
            cases.append(f'<article class="sample"><h3>Historical draw {example["draw_id"]}</h3>'+
                         '<div class="predictioncards">'+''.join(cards)+'</div></article>')
        gallery='<section><h2>Previously saved v0.5 image predictions</h2><p class="muted">'
        gallery+='Historical test examples, not new prospective forecasts. Exact hits are distinct from shape resemblance.</p>'
        gallery+=''.join(cases)+'</section>'
    css='''<style>
*{box-sizing:border-box}body{font:14px/1.5 system-ui,Arial,sans-serif;color:#1d3148;background:#f2f5f9;margin:0}main{max-width:1260px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}h3{font-size:13px;margin:0 0 10px}h4{font-size:12px;margin:0 0 7px}p{margin:8px 0} .muted{color:#65758a}.subtitle{color:#4e6279}section{background:#fff;border:1px solid #dae4ed;border-radius:15px;padding:20px;margin:18px 0;box-shadow:0 3px 12px #142d4810}.badge{display:inline-block;color:#345f89;background:#e9f1fa;padding:3px 9px;border-radius:20px;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:12px}.heatgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.hunit{min-width:0}.hunit svg{width:180px;max-width:100%;height:auto}.transition,.feature,.sample{background:#f9fbfd;border:1px solid #e3ebf2;border-radius:10px;padding:14px}.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.triple svg{width:100%;max-width:126px}.triple span{display:block;font-size:11px;color:#6c7b90}.numbers{font-size:12px;white-space:normal;overflow-wrap:anywhere}.feature svg{width:100%;max-width:340px}.legend{display:flex;flex-wrap:wrap;gap:13px;margin-bottom:15px}.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:middle}.matrixrow{display:flex;gap:36px;flex-wrap:wrap}.matrix{max-width:100%;width:330px}.predictioncards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.predictioncards svg{max-width:100%;width:115px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #dfe7ef;padding:8px;text-align:left}th{font-size:12px;color:#4d637c}.small{font-size:12px}@media(max-width:850px){.heatgrid{grid-template-columns:repeat(2,1fr)}.predictioncards{grid-template-columns:repeat(3,1fr)}}@media(max-width:580px){main{padding:12px}.heatgrid{grid-template-columns:repeat(2,1fr);gap:8px}.grid{grid-template-columns:1fr}.predictioncards{grid-template-columns:repeat(2,1fr)}section{padding:12px}}@media print{body{background:white}section,.transition,.feature,.sample{box-shadow:none;break-inside:avoid}.grid{grid-template-columns:repeat(2,1fr)}.heatgrid{grid-template-columns:repeat(3,1fr)}}
</style>'''
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Lotto7x7 v0.5.1 Visual Image Dynamics</title>'+css+'<main>'
            '<h1>Lotto7x7 · Visual Image Dynamics</h1>'
            '<p class="subtitle">Image-first morphology, temporal transitions, spectral structure and exact position vs tolerant shape comparisons.</p>'
            f'<p><span class="badge">v0.5.1</span> {report["draws"]} verified images · '
            f'{report["first_draw_id"]}–{report["last_draw_id"]} · '
            f'{report["configuration"]["null_images"]} IID controls</p>'
            '<section><h2>1 · Consecutive image transitions</h2>'
            '<p class="muted">Retained = same number/pixel position. Added/removed count equally because every image has exactly six points. Shape similarity allows rotations, reflections and translations; it is not a hit count.</p>'
            '<div class="legend"><span><i class="swatch" style="background:#27886a"></i>Retained</span>'
            '<span><i class="swatch" style="background:#d18a47"></i>Removed</span>'
            '<span><i class="swatch" style="background:#3678b7"></i>Added</span></div><div class="grid">'+''.join(transitions)+'</div></section>'
            '<section><h2>2 · Pixel occupancy and non-DC spectral energy</h2>'
            '<p class="muted">Top row: actual occupancy, actual minus exact IID 6/49, and unadjusted standardized deviation. Bottom row: mean squared DCT power with DC removed; control and differences. Color scales are separate and fixed within each comparison type; hover for values.</p>'
            '<div class="heatgrid">'+hmaps+'</div><p class="small muted">'
            f'Fair non-DC DCT coefficient mean square: {IID_NON_DC_POWER:.6f}. '
            'DC is excluded because every image has six active pixels. Heatmap extremes are descriptive, not corrected significance tests.</p></section>'
            '<section><h2>3 · Morphology versus independent random six-point images</h2>'
            '<div class="legend"><span><i class="swatch" style="background:#4b80b6"></i>Historical images</span>'
            '<span><i class="swatch" style="background:#cf7546"></i>IID controls (line)</span></div>'
            '<div class="grid">'+''.join(dists)+'</div><p class="muted">'
            'Morphology values follow the v0.5 normalized feature definitions. Per-feature MC z values are exploratory and not adjusted for looking at multiple features or mappings.</p>'
            '<h3>Spatial-mapping sensitivity: actual minus IID morphology feature means</h3>'
            '<table><tr><th>Layout / seed</th><th>8-neighbor edges</th><th>8-connected components</th><th>Radial spread</th></tr>'+''.join(comparisons)+'</table></section>'
            '<section><h2>4 · Recent cross-draw similarity matrices</h2><p class="muted">'
            'Both matrices compare the same most-recent chronological draws (oldest at top/left). The exact-position matrix ignores image shape; the second permits arbitrary D4 rotation/reflection and translation.</p>'+matrix_html+
            f'<p class="small muted">IID expected exact shared pixels: {IID_OVERLAP:.4f}. '
            f'IID sample expected shape similarity: {matr["iid_sample_mean_shape_similarity"]:.4f} '
            f'({matr["iid_null_shape_pairs"]} independent random pairs). The diagonal is self-similarity, not evidence.</p></section>'+gallery+
            '<section><h2>Methodology & limitations</h2><p>This is a retrospective exploratory visual analysis. Six-pixel images are a lossless encoding of numbers, not the physical geometry of the lottery machine. Changing spatial mapping changes some shape and spectral descriptors but not exact-number overlaps. No graph in this report establishes a forecasting edge.</p></section></main></html>')
