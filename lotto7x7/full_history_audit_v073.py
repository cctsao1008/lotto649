"""v0.7.3 independent CPU score audit and draw-clustered multi-seed analysis.

Read BIN wire formats without using the project image decoder; recompute actual
six-number draws, every ticket's hit count, and complete v0.7.2 summaries on
CPU. Fair-coverage verification uses an independent disjoint combinatorial
anchor and, for overlapping portfolios, the existing exact DP (explicitly NOT
independent verification of that algorithm). This is historical exploratory
research; shared source formats are NOT an official data cross-check.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import hashlib
import json
from math import comb
from pathlib import Path
import struct

import numpy as np

# Freeze the v0.7.2 wire/report contract locally; do not depend on the GPU
# scorer's module, constants or imports for independent report validation.
SOURCE_SCHEMA = "lotto7x7.full_history_gpu.v0.7.2"
NAMES = (
    "B0-global-random/random", "B1-random-pool/random",
    "B1-random-pool/scenario-coverage", "S1-shape-pool/random",
    "S1-shape-pool/scenario-coverage", "B2-disjoint/disjoint",
)

AUDIT_SCHEMA = "lotto7x7.full_history_independent_audit.v0.7.3"
MULTISEED_SCHEMA = "lotto7x7.full_history_multiseed.v0.7.3"
UNIVERSE = comb(49, 6)
HEADER = struct.Struct("<8sII")
IMAGE = struct.Struct("<I49B")
HISTORY = struct.Struct("<II7B")
METHODS = ("random", "scenario-coverage")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _close(actual, declared, *, label: str, atol=2e-12):
    if (not isinstance(declared, (int, float)) or isinstance(declared, bool)
            or not np.isfinite(declared) or abs(float(actual)-float(declared)) > atol):
        raise ValueError(f"{label}: reported={declared!r}, recomputed={actual!r}")


def _ticket_set(items, count: int, *, context: str):
    if not isinstance(items, list) or len(items) != count:
        raise ValueError(f"{context}: ticket count mismatch")
    tickets = []
    for item in items:
        if (not isinstance(item, list) or len(item) != 6
                or any(type(n) is not int or not 1 <= n <= 49 for n in item)
                or len(set(item)) != 6 or item != sorted(item)):
            raise ValueError(f"{context}: invalid or unsorted six-number ticket")
        tickets.append(tuple(item))
    if len(set(tickets)) != count:
        raise ValueError(f"{context}: duplicate ticket")
    return tuple(tickets)


def _pair_p(s_only: int, b_only: int) -> float:
    """Exact two-sided conditional McNemar using Python integers, no SciPy."""
    n = s_only + b_only
    if n == 0:
        return 1.0
    k = min(s_only, b_only)
    return min(1.0, (2 * sum(comb(n, j) for j in range(k+1))) / (2**n))


def _holm(values: dict[str, float]):
    ranked = sorted(values, key=lambda name: (values[name], name))
    out, previous = {}, 0.0
    for j, name in enumerate(ranked):
        previous = max(previous, (len(ranked)-j)*values[name])
        out[name] = min(1.0, previous)
    return out


def _interval(values, *, seed: int, replicates: int = 4000):
    """Draw-level percentile bootstrap; never treat seed replications as draws."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or not len(observations) or not np.isfinite(observations).all():
        raise ValueError("bootstrap requires finite nonempty draw-level values")
    rng = np.random.default_rng(seed)
    out = np.empty(replicates)
    for offset in range(0, replicates, 128):
        upper = min(replicates, offset+128)
        indices = rng.integers(0, len(observations), (upper-offset, len(observations)))
        out[offset:upper] = observations[indices].mean(axis=1)
    return [float(v) for v in np.quantile(out, (0.025, 0.975))]


def _disjoint_exact(count: int) -> dict[str, float]:
    """Independent closed form for N pairwise-disjoint 6/49 tickets.

    For >=3 only two distinct tickets can both qualify (3+3 exactly), each
    intersection contains C(6,3)^2=400 possible winning draws; for >=4,5,6,
    even two qualifying tickets cannot fit into a 6-number draw.
    """
    if not 1 <= count <= 5:
        raise ValueError("N must be 1..5")
    result = {}
    for threshold in (3, 4, 5, 6):
        single = sum(comb(6, k)*comb(43, 6-k) for k in range(threshold, 7))
        intersections = comb(count, 2) * 400 if threshold == 3 else 0
        result[str(threshold)] = (count*single-intersections)/UNIVERSE
    return result


@lru_cache(maxsize=4096)
def _expected_fair(tickets):
    if len({n for row in tickets for n in row}) == 6*len(tickets):
        return _disjoint_exact(len(tickets))
    # Same tested exact-DP mathematical engine as v0.7.2; CPU scoring is separate.
    from .coverage_v06 import exact_coverage
    return exact_coverage(tickets)["at_least"]


@dataclass(frozen=True)
class VerifiedHistory:
    ids: tuple[int, ...]
    numbers: tuple[tuple[int, ...], ...]
    archive_sha256: str
    images_sha256: str
    history_sha256: str
    manifest_sha256: str
    source_site: str
    manifest_source: str


def _unpack_records(path: Path, *, magic: bytes, row: struct.Struct):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: incomplete binary header")
    stored_magic, count, width = HEADER.unpack_from(raw)
    if (stored_magic, width) != (magic, row.size) or count < 1:
        raise ValueError(f"{path}: incompatible BIN magic, record width, or count")
    if len(raw) != HEADER.size+count*row.size:
        raise ValueError(f"{path}: inconsistent BIN byte length")
    return raw, [row.unpack_from(raw, HEADER.size+i*row.size) for i in range(count)]


def read_verified_history(data: Path, manifest_path: Path, *, history_path: Path | None = None):
    """Wire-parse both independent BIN layouts and check the source manifest.

    Important: both files originate from the same third-party archive, so this
    is an integrity/format cross-check, not independent official ground truth.
    """
    data, manifest_path = Path(data), Path(manifest_path)
    history_path = Path(history_path) if history_path is not None else data.with_name("lotto649_history_v1.bin")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if (manifest.get("dataset") != "lotto649.single-source.biga.v1"
            or not isinstance(manifest.get("source_site"), str)):
        raise ValueError("unsupported or missing source manifest schema")
    img_bytes, image_rows = _unpack_records(data, magic=b"L649IMG2", row=IMAGE)
    his_bytes, history_rows = _unpack_records(history_path, magic=b"L649HIS1", row=HISTORY)
    if len(image_rows) != len(history_rows) or len(image_rows) != manifest.get("total_draws"):
        raise ValueError("manifest/source row-count mismatch")
    for name, raw in ((data.name, img_bytes), (history_path.name, his_bytes)):
        entry = manifest.get("files", {}).get(name)
        if not isinstance(entry, dict) or entry.get("bytes") != len(raw) or entry.get("sha256") != _digest(raw):
            raise ValueError(f"manifest SHA256/byte-count mismatch: {name}")
    ids, numbers, bitmaps = [], [], bytearray()
    last_date, last_id = None, 0
    for i, ((draw_id, *pixels), (hid, day, *vals)) in enumerate(zip(image_rows, history_rows)):
        main, bonus = tuple(vals[:6]), vals[6]
        if draw_id != hid or draw_id <= last_id:
            raise ValueError(f"row {i}: image/history draw IDs mismatch or unordered")
        if len(main) != 6 or len(set(main)) != 6 or main != tuple(sorted(main)) or any(n < 1 or n > 49 for n in main):
            raise ValueError(f"row {i}: invalid sorted main numbers")
        if not 1 <= bonus <= 49 or bonus in main:
            raise ValueError(f"row {i}: invalid bonus number")
        if any(p not in (0,1) for p in pixels) or sum(pixels) != 6:
            raise ValueError(f"row {i}: image must be six-hot binary")
        active = tuple(j+1 for j,p in enumerate(pixels) if p)
        if active != main:
            raise ValueError(f"row {i}: image and history main numbers differ")
        try:
            stamp = date.fromisoformat(f"{day//10000:04d}-{day//100%100:02d}-{day%100:02d}")
        except ValueError as err:
            raise ValueError(f"row {i}: invalid YYYYMMDD") from err
        if last_date is not None and stamp <= last_date:
            raise ValueError(f"row {i}: non-increasing dates")
        if draw_id//1000000 != stamp.year-1911 or draw_id%1000000 <= 0:
            raise ValueError(f"row {i}: draw ID/year mismatch")
        ids.append(draw_id)
        numbers.append(main)
        bitmaps.extend(pixels)
        last_id, last_date = draw_id, stamp
    if (ids[0] != manifest.get("first_draw_id") or ids[-1] != manifest.get("last_draw_id")):
        raise ValueError("manifest first/last draw anchors mismatch")
    id_raw = b"".join(struct.pack("<q", draw_id) for draw_id in ids)
    archive_sha = _digest(bytes(bitmaps)+id_raw)
    return VerifiedHistory(
        ids=tuple(ids), numbers=tuple(numbers), archive_sha256=archive_sha,
        images_sha256=_digest(img_bytes), history_sha256=_digest(his_bytes),
        manifest_sha256=_digest(manifest_raw), source_site=manifest["source_site"],
        manifest_source=manifest.get("dataset", "unknown"),
    )


def _year_of(draw_id: int):
    return str(draw_id//1_000_000 + 1911)


def audit_report(report: dict, history: VerifiedHistory, *, allow_partial=False,
                 audit_seed=649) -> dict:
    """Independently rescore all 6 portfolios at each target from raw BIN IDs.

    Every exact fair-coverage value is checked: closed-form for disjoint sets,
    shared existing project DP for overlaps. No CUDA, GPU tensors or existing
    v0.7.2 score_batch() are used for scoring and verifying summaries.
    """
    if not isinstance(report, dict) or report.get("schema") != SOURCE_SCHEMA:
        raise ValueError("expected v0.7.2 full-history-gpu JSON")
    parameters = report.get("parameters", {})
    warmup, count = parameters.get("warmup"), parameters.get("count")
    if (type(warmup) is not int or not 1 <= warmup < len(history.ids)
            or type(count) is not int or not 1 <= count <= 5):
        raise ValueError("invalid warmup or ticket count in report")
    if parameters.get("archive_sha256") != history.archive_sha256:
        raise ValueError("report/archive SHA256 mismatch: dataset may have changed")
    if (report.get("source_draws") != len(history.ids) or
            report.get("eligible_draws") != len(history.ids)-warmup or
            report.get("warmup_only_draws") != warmup):
        raise ValueError("report/source draw counts inconsistent")
    rows = report.get("targets")
    skip = report.get("skipped_eligible_targets")
    if (not isinstance(rows, list) or not rows or type(skip) is not int
            or skip < 0 or skip+len(rows) != len(history.ids)-warmup
            or report.get("evaluated_draws") != len(rows)):
        raise ValueError("target count, skipped targets or chronology inconsistent")
    if not allow_partial and skip:
        raise ValueError("expected full historical range; use --allow-partial for smoke report")
    if set(report.get("summary", {})) != set(NAMES):
        raise ValueError("report summary arms differ from v0.7.2 schema")
    all_hits = {name: [] for name in NAMES}
    all_fair = {name: [] for name in NAMES}
    years = defaultdict(list)
    base = warmup+skip
    for i, row in enumerate(rows):
        t = base+i
        draw_id = history.ids[t]
        if row.get("draw_id") != draw_id or row.get("cutoff_draw_id") != history.ids[t-1]:
            raise ValueError(f"target {i}: draw ID or cutoff mismatch")
        if row.get("actual") != list(history.numbers[t]):
            raise ValueError(f"target {draw_id}: actual numbers differ from verified BIN")
        bets_by_arm = row.get("bets")
        fair_by_arm = row.get("fair_coverage")
        if not isinstance(bets_by_arm, dict) or set(bets_by_arm) != set(NAMES):
            raise ValueError(f"target {draw_id}: incomplete bet arms")
        if not isinstance(fair_by_arm, dict) or set(fair_by_arm) != set(NAMES):
            raise ValueError(f"target {draw_id}: incomplete fair coverage arms")
        actual = set(history.numbers[t])
        for name in NAMES:
            tickets = _ticket_set(bets_by_arm[name], count, context=f"{draw_id}/{name}")
            is_disjoint = len({n for bet in tickets for n in bet}) == 6*count
            if name == "B2-disjoint/disjoint" and not is_disjoint:
                raise ValueError(f"target {draw_id}: B2 tickets overlap")
            max_hits = max(len(actual.intersection(bet)) for bet in tickets)
            all_hits[name].append(max_hits)
            fair_dict = fair_by_arm[name]
            if not isinstance(fair_dict, dict) or set(fair_dict) != {"3","4","5","6"}:
                raise ValueError(f"target {draw_id}: invalid fair threshold dictionary")
            expected = _expected_fair(tickets)
            for level in (3,4,5,6):
                _close(expected[str(level)], fair_dict[str(level)],
                       label=f"{draw_id}/{name}/fair>={level}")
            _close(count/UNIVERSE, fair_dict["6"], label=f"{draw_id}/{name}/jackpot")
            all_fair[name].append(fair_dict["3"])
        years[_year_of(draw_id)].append(i)
    by_arm = {}
    for name in NAMES:
        vals = np.asarray(all_hits[name], dtype=np.int64)
        ps = np.asarray(all_fair[name], dtype=np.float64)
        expected = report["summary"][name]
        hits = int(np.count_nonzero(vals >= 3))
        hist = [int(np.count_nonzero(vals == j)) for j in range(7)]
        if expected.get("draws") != len(rows) or expected.get("ge3_counts") != hits or expected.get("max_hits_histogram") != hist:
            raise ValueError(f"summary mismatch: {name} count/histogram")
        _close(vals.mean(), expected.get("mean_max_hits"), label=f"summary/{name}/mean_hits")
        _close(hits/len(rows), expected.get("realized_ge3"), label=f"summary/{name}/ge3")
        _close(ps.mean(), expected.get("fair_expected_ge3"), label=f"summary/{name}/fair_ge3")
        reported_years = expected.get("year")
        if not isinstance(reported_years, dict) or set(reported_years) != set(years):
            raise ValueError(f"summary/{name}/years inconsistent")
        for year, idx in years.items():
            summary = reported_years[year]
            ii=np.asarray(idx,dtype=int)
            if summary.get("draws") != len(ii) or summary.get("ge3_count") != int(np.count_nonzero(vals[ii]>=3)):
                raise ValueError(f"summary/{name}/{year} count mismatch")
            _close(ps[ii].mean(), summary.get("fair_ge3"), label=f"summary/{name}/{year}/fair")
        by_arm[name] = {
            "ge3_count": hits, "ge3_rate": hits/len(rows),
            "fair_ge3_mean": float(ps.mean()), "ge3_residual": float((vals>=3).mean()-ps.mean()),
            "max_hits_histogram_0_to_6": hist, "mean_max_hits": float(vals.mean()),
        }
    pairs={}
    for j,method in enumerate(METHODS):
        na=f"S1-shape-pool/{method}"
        nb=f"B1-random-pool/{method}"
        a=(np.asarray(all_hits[na])>=3).astype(np.int8)
        b=(np.asarray(all_hits[nb])>=3).astype(np.int8)
        fa=np.asarray(all_fair[na],dtype=float)
        fb=np.asarray(all_fair[nb],dtype=float)
        s_only=int(np.count_nonzero((a==1)&(b==0)))
        b_only=int(np.count_nonzero((a==0)&(b==1)))
        diff=a.astype(float)-b.astype(float)
        excess=diff-(fa-fb)
        expected=report.get("s1_vs_b1_paired", {}).get(method)
        if not isinstance(expected, dict):
            raise ValueError("paired result missing")
        for field, value in (("s1_only",s_only),("b1_only",b_only),
                             ("both",int(np.count_nonzero((a==1)&(b==1)))),
                             ("neither",int(np.count_nonzero((a==0)&(b==0))))):
            if expected.get(field)!=value:
                raise ValueError(f"paired {method}/{field} mismatch")
        for field, value in (("difference",diff.mean()),("fair_difference",(fa-fb).mean()),
                             ("excess_difference",excess.mean()),
                             ("exact_mcnemar_p",_pair_p(s_only,b_only))):
            _close(value,expected.get(field),label=f"paired/{method}/{field}")
        if set(expected.get("year", {})) != set(years):
            raise ValueError(f"paired/{method} year group mismatch")
        for year,idx in years.items():
            ii=np.asarray(idx,dtype=int)
            y=expected["year"][year]
            if y.get("targets") != len(ii):
                raise ValueError(f"paired/{method}/{year} targets mismatch")
            _close(diff[ii].mean(),y.get("observed_s1_minus_b1"),
                   label=f"paired/{method}/{year}/observed")
            _close((fa[ii]-fb[ii]).mean(),y.get("fair_s1_minus_b1"),
                   label=f"paired/{method}/{year}/fair")
        pairs[method]={
            "s1_only":s_only,"b1_only":b_only,"difference":float(diff.mean()),
            "fair_difference":float((fa-fb).mean()),
            "excess_difference":float(excess.mean()),
            "exact_mcnemar_p":_pair_p(s_only,b_only),
            "draw_bootstrap_ci95":_interval(diff,seed=audit_seed+101*j),
            "draw_excess_bootstrap_ci95":_interval(excess,seed=audit_seed+101*j+1),
        }
    corrected=_holm({key:value["exact_mcnemar_p"] for key,value in pairs.items()})
    for name,value in pairs.items():
        _close(corrected[name],report["s1_vs_b1_paired"][name].get("holm_adjusted_p"),
               label=f"paired/{name}/holm")
        value["holm_adjusted_p"]=corrected[name]
    source_sha=_digest(json.dumps(report,sort_keys=True,separators=(",",":"),
                                  ensure_ascii=True,allow_nan=False).encode("utf-8"))
    return {
        "schema":AUDIT_SCHEMA,"research_only":True,
        "source_report_sha256":source_sha,
        "dataset":{
            "draws":len(history.ids),"first_draw_id":history.ids[0],
            "last_draw_id":history.ids[-1],"source_site":history.source_site,
            "manifest_sha256":history.manifest_sha256,
            "image_sha256":history.images_sha256,
            "history_sha256":history.history_sha256,
            "archive_sha256":history.archive_sha256,
        },
        "parameters":parameters,
        "target_count":len(rows),"validated_full_history":not bool(skip),
        "independent_cpu_scored_ticket_sets":len(rows)*len(NAMES),
        "summary":by_arm,"s1_vs_b1_paired":pairs,
        "math_checks":"independent disjoint hypergeometric formula; shared project exact DP for overlaps",
        "limitations":[
            "CPU scorer does not reuse GPU score_batch or CUDA, but does not reconstruct pre-draw selections.",
            "Biga history/images BIN files share a single third-party source, not an official independent cross-check.",
            "Overlapping-ticket fair probabilities are verified with existing exact DP, not an independent mathematical implementation.",
            "Historical data and strategy choices have been inspected; all inference remains exploratory.",
        ],
    }


def multiseed_report(reports: list[dict], audits: list[dict], *, seed=649) -> dict:
    """Aggregate seed-specific hits within each historical target BEFORE resampling.

    Explicitly reject duplicate seeds, mismatched draw IDs, different archive,
    or differing configurations. Per-seed McNemar p-values are descriptive and
    adjusted across 2*K comparisons; no pooled-seed McNemar pseudo-replication.
    """
    if not isinstance(reports,list) or len(reports)<2 or len(reports)!=len(audits):
        raise ValueError("requires at least two corresponding full reports and audits")
    original=reports[0]
    src_params=original["parameters"]
    names = set(NAMES)
    ids = [r["draw_id"] for r in original["targets"]]
    seeds=set()
    for report,audit in zip(reports,audits):
        prm=report.get("parameters",{})
        this_seed=prm.get("seed")
        if type(this_seed) is not int or this_seed in seeds:
            raise ValueError("multi-seed reports must have unique integer seeds")
        seeds.add(this_seed)
        if (prm.get("archive_sha256")!=src_params.get("archive_sha256")
                or any(prm.get(k)!=src_params.get(k) for k in
                       ("warmup","count","size","families","scenarios","tail_targets"))
                or [r["draw_id"] for r in report["targets"]]!=ids):
            raise ValueError("seed reports have different inputs, parameters or target IDs")
        if (audit.get("schema")!=AUDIT_SCHEMA
                or audit["source_report_sha256"]!=_digest(json.dumps(report,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8"))
                or audit["target_count"]!=len(ids)):
            raise ValueError("audited input report fingerprint or target count mismatch")
        if set(report.get("summary",{}))!=names:
            raise ValueError("inconsistent strategy names")
        if any(row["actual"]!=original["targets"][i]["actual"] or
               row["cutoff_draw_id"]!=original["targets"][i]["cutoff_draw_id"]
               for i,row in enumerate(report["targets"])):
            raise ValueError("seed reports differ on official target numbers or cutoffs")
    ordered=sorted(zip(reports,audits),key=lambda pair: pair[0]["parameters"]["seed"])
    run_summary={}
    for report,audit in ordered:
        ss=report["parameters"]["seed"]
        run_summary[str(ss)]={
            "source_report_sha256":audit["source_report_sha256"],
            "arms":audit["summary"],"paired":audit["s1_vs_b1_paired"],
        }
    aggregated={}
    for j,method in enumerate(METHODS):
        s1=f"S1-shape-pool/{method}"
        b1=f"B1-random-pool/{method}"
        observed=[];fair=[];per_seed={}
        for report,audit in ordered:
            rows=report["targets"]
            ob=np.asarray([
                int(max(len(set(row["actual"]).intersection(ticket)) for ticket in row["bets"][s1])>=3)-
                int(max(len(set(row["actual"]).intersection(ticket)) for ticket in row["bets"][b1])>=3)
                for row in rows],dtype=np.float64)
            fo=np.asarray([row["fair_coverage"][s1]["3"]-row["fair_coverage"][b1]["3"]
                           for row in rows],dtype=np.float64)
            observed.append(ob); fair.append(fo)
            per_seed[str(report["parameters"]["seed"])]=float(ob.mean())
        draw_observed=np.mean(np.stack(observed),axis=0)
        draw_fair=np.mean(np.stack(fair),axis=0)
        draw_excess=draw_observed-draw_fair
        years=defaultdict(list)
        for i,draw_id in enumerate(ids): years[_year_of(draw_id)].append(i)
        aggregated[method]={
            "independent_draw_targets":len(ids),"seed_replicates_per_target":len(ordered),
            "per_seed_observed_differences":per_seed,
            "mean_observed_difference":float(draw_observed.mean()),
            "mean_fair_coverage_difference":float(draw_fair.mean()),
            "mean_excess_over_fair":float(draw_excess.mean()),
            "draw_level_ci95":_interval(draw_observed,seed=seed+107*j),
            "draw_level_excess_ci95":_interval(draw_excess,seed=seed+107*j+1),
            "year":{year:{"draws":len(index),
                           "observed_difference":float(draw_observed[index].mean()),
                           "fair_coverage_difference":float(draw_fair[index].mean()),
                           "excess_over_fair":float(draw_excess[index].mean())}
                    for year,index in sorted(years.items())},
        }
    calibrations={}
    for j,name in enumerate(NAMES):
        by_seed=[];fair_by_seed=[]
        for report,audit in ordered:
            values=np.asarray([int(max(len(set(r["actual"]).intersection(ticket))
                                          for ticket in r["bets"][name])>=3)
                               for r in report["targets"]],dtype=np.float64)
            expected=np.asarray([r["fair_coverage"][name]["3"] for r in report["targets"]],dtype=np.float64)
            by_seed.append(values);fair_by_seed.append(expected)
        rate=np.mean(np.stack(by_seed),axis=0)
        fair_rate=np.mean(np.stack(fair_by_seed),axis=0)
        calibrations[name]={
            "mean_realized_rate":float(rate.mean()),
            "mean_fair_rate":float(fair_rate.mean()),
            "mean_residual":float((rate-fair_rate).mean()),
            "draw_level_residual_ci95":_interval(rate-fair_rate,seed=seed+313+j),
        }
    tests={f"seed{report['parameters']['seed']}/{method}":
           audit['s1_vs_b1_paired'][method]['exact_mcnemar_p']
           for report,audit in ordered for method in METHODS}
    corrected=_holm(tests)
    return {
        "schema":MULTISEED_SCHEMA,"research_only":True,
        "archive_sha256":src_params["archive_sha256"],
        "draw_targets":len(ids),"source_draws":original["source_draws"],
        "seeds":[report["parameters"]["seed"] for report,_ in ordered],
        "common_parameters":{k:v for k,v in src_params.items() if k!="seed"},
        "per_seed":run_summary,"seed_paired_holm_pvalues":corrected,
        "draw_clustered_s1_minus_b1":aggregated,
        "draw_clustered_fair_calibration":calibrations,
        "limitations":[
            "Each seed reuses identical historical target draws. No seed counts as independent evidence.",
            "Only target draws are bootstrap units. Aggregate results do not use pseudo-replicated McNemar tests.",
            "Holm covers two comparisons per provided seed only; it does not correct prior hindsight strategy selection.",
            "No independent official lottery source was checked; shared third-party data may contain source errors.",
            "Historical retrospective replay cannot demonstrate prospective forecasting ability.",
        ],
    }
