"""Issue #38: read-only, reproducible evidence registry and scientific status gates.

No lottery draw predictions, purchases, source edits or updates to sealed studies.
"""
from __future__ import annotations
import argparse
from hashlib import sha1
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

SCHEMA = "lotto649.research.evidence_registry.v1"
REGISTRY = Path("research/evidence_registry_v1.json")
INDEX = Path("docs/research_evidence_index.md")
STAGES = {"SOURCE_ARCHIVE", "STRUCTURAL_THEOREM", "DESCRIPTIVE",
          "EXPLORATORY_RETROSPECTIVE", "PROSPECTIVE_PRESEALED", "PROSPECTIVE_EVALUATED"}
DECISIONS = {"VERIFIED_SOURCE", "PROVEN_MATHEMATICALLY", "DESCRIPTIVE_COMPLETE",
             "NO_EVIDENCE", "PENDING", "INCONCLUSIVE", "VALIDATED_GO"}
ENGINEERING = {"PASS", "IN_PROGRESS", "UNVERIFIED"}
DOMAINS = {"probability", "statistics", "geometry", "astronomy", "iching",
           "portfolio", "source_engineering"}
ROLES = {"source", "protocol", "report", "seal", "mathematical_proof"}
REQUIRED = {"id", "issue", "domain", "research_question", "hypothesis", "evidence_stage",
            "research_decision", "engineering_status", "source_cutoff", "sample_size",
            "planned_future_draws", "model_version", "null_model", "multiple_testing",
            "primary_endpoint", "result_summary", "method", "limitations",
            "next_action", "artifacts", "evidence_commit", "ci_run", "evidence_as_of"}
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
CI = re.compile(r"https://github\.com/cctsao1008/lotto649/actions/runs/[0-9]+\Z")

class GovernanceError(ValueError):
    """Invalid study provenance, recorded result or evidence tier."""

def fail(message: str) -> None:
    raise GovernanceError(message)

def git_blob_sha1(contents: bytes) -> str:
    return sha1(b"blob " + str(len(contents)).encode("ascii") + b"\0" + contents).hexdigest()

def safe_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        fail("artifact path must be a relative POSIX path")
    p = PurePosixPath(value)
    if (p.is_absolute() or p.as_posix() != value or value.startswith("./") or
            any(x in ("", ".", "..") for x in value.split("/")) or
            value.startswith("data/private/") or p.parts[0] in ("results", "private")):
        fail("unsafe or private artifact path: " + repr(value))
    return Path(*p.parts)

def checked_file(root: Path, artifact: dict, ident: str) -> Path:
    rel = safe_path(artifact["path"])
    p = root / rel
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
        fail(f"{ident}: missing/symlink/out-of-root artifact: {rel}")
    raw = p.read_bytes()
    if git_blob_sha1(raw) != artifact["git_blob_sha1"]:
        if b"\r\n" in raw and b"\r" not in raw.replace(b"\r\n", b""):
            raw = raw.replace(b"\r\n", b"\n")
        if git_blob_sha1(raw) != artifact["git_blob_sha1"]:
            fail(f"{ident}: pinned original Git blob changed: {rel}")
    return p

def inspect_native(e: dict, root: Path) -> None:
    ident = e["id"]
    reports = [root / safe_path(a["path"]) for a in e["artifacts"]
               if a["role"] in ("report", "seal") and a["path"].endswith(".json")]
    if ident in ("iching-p1a-20261113", "iching-p2a-20261114",
                 "iching-p3-static100-20261114", "m0-v23-static200-20261113"):
        if len(reports) != 1:
            fail(f"{ident}: expected one native JSON evidence artifact")
        j = json.loads(reports[0].read_text(encoding="utf-8"))
        if ident == "iching-p1a-20261113":
            if (j.get("decision") != "EXPLORATORY_ASSOCIATION_SCREEN_NO_EVIDENCE" or
                j.get("analysis_draws") != 2476 or
                j.get("bonferroni_0_05_significant_endpoints") != []):
                fail("P1A original recorded conclusion/coverage inconsistent")
        elif ident == "iching-p2a-20261114":
            if (j.get("decision") != "EXPLORATORY_ASSOCIATION_NO_EVIDENCE" or
                j.get("analysis_draws") != 2476 or
                j.get("bonferroni_significant_endpoints") != [] or
                len(j.get("endpoints", {})) != 6 or
                abs(min(v["bonferroni_six_global_p"] for v in j["endpoints"].values())-0.3345) > 1e-10):
                fail("P2A original six-test conclusion or correction inconsistent")
        elif ident == "iching-p3-static100-20261114":
            if (j.get("study_id") != "v24-p3-usno-sky-parity100-20261114" or
                j.get("current_result") != "PENDING_0_OF_100_NO_PREDICTIVE_GO" or
                j.get("horizon_consecutive_verified_draws") != 100 or
                j.get("first_eligible_draw", {}).get("draw_id") != 115000106 or
                j.get("model", {}).get("never_automatically_conclude_predictive_go") is not True):
                fail("P3 originally sealed future pending status changed")
        else:
            p = j.get("prospective_protocol", {})
            if (j.get("experiment_id") != "v23-m0-full-universe-shadow-20261113" or
                p.get("first_draw_id") != 115000106 or
                p.get("horizon_consecutive_official_draws") != 200):
                fail("v2.3 original 200-draw protocol altered")

def validate_registry(doc: Any, root: Path, *, verify_blobs: bool = True) -> list[dict]:
    if not isinstance(doc, dict) or set(doc) != {"schema", "entries"} or doc.get("schema") != SCHEMA:
        fail("unknown registry schema or root fields")
    entries = doc["entries"]
    if not isinstance(entries, list) or not entries:
        fail("registry must be nonempty")
    ids = set()
    for e in entries:
        if not isinstance(e, dict) or set(e) != REQUIRED:
            fail("unexpected or missing evidence record fields")
        ident = e["id"]
        if not isinstance(ident, str) or not ID.fullmatch(ident) or ident in ids:
            fail("duplicate or unsafe experiment ID")
        ids.add(ident)
        if type(e["issue"]) is not int or e["issue"] <= 0:
            fail(f"{ident}: bad issue number")
        for key in ("research_question", "hypothesis", "model_version", "null_model",
                    "multiple_testing", "primary_endpoint", "result_summary",
                    "method", "next_action", "evidence_as_of"):
            if not isinstance(e[key], str) or not e[key].strip():
                fail(f"{ident}: missing {key}")
        if e["domain"] not in DOMAINS or e["evidence_stage"] not in STAGES or e["research_decision"] not in DECISIONS or e["engineering_status"] not in ENGINEERING:
            fail(f"{ident}: unknown stage, decision, domain or engineering status")
        for key in ("sample_size", "planned_future_draws"):
            if type(e[key]) is not int or e[key] < 0:
                fail(f"{ident}: invalid {key}")
        if e["source_cutoff"] is not None and (not isinstance(e["source_cutoff"], str) or not e["source_cutoff"].strip()):
            fail(f"{ident}: invalid source cutoff")
        if not isinstance(e["limitations"], list) or not e["limitations"] or not all(isinstance(v, str) and v.strip() for v in e["limitations"]):
            fail(f"{ident}: missing limitation statement")
        arts = e["artifacts"]
        if not isinstance(arts, list) or not arts or not all(isinstance(a, dict) for a in arts):
            fail(f"{ident}: missing artifacts")
        if len({a.get("path") for a in arts}) != len(arts):
            fail(f"{ident}: duplicate evidence path")
        roles = {a.get("role") for a in arts}
        if e["evidence_stage"] == "PROSPECTIVE_PRESEALED":
            if (e["research_decision"] != "PENDING" or e["sample_size"] != 0 or
                    e["planned_future_draws"] < 1 or "seal" not in roles):
                fail(f"{ident}: original prospective pre-seal must remain PENDING, 0 evaluated")
        elif e["research_decision"] == "PENDING":
            fail(f"{ident}: PENDING requires original future-only pre-seal")
        if e["research_decision"] == "VALIDATED_GO":
            fail(f"{ident}: governance never automatically declares predictive GO; requires separate review")
        if e["evidence_stage"] == "STRUCTURAL_THEOREM" and e["research_decision"] != "PROVEN_MATHEMATICALLY":
            fail(f"{ident}: exact theorem may not be classified as forecast evidence")
        if e["engineering_status"] != "PASS" and e["research_decision"] in {"VERIFIED_SOURCE","PROVEN_MATHEMATICALLY","DESCRIPTIVE_COMPLETE"}:
            fail(f"{ident}: source/theory claim requires engineering PASS")
        commit = e["evidence_commit"]
        ci = e["ci_run"]
        if commit is not None and (not isinstance(commit, str) or not SHA40.fullmatch(commit)):
            fail(f"{ident}: invalid evidence commit")
        if ci is not None and (not isinstance(ci, str) or not CI.fullmatch(ci)):
            fail(f"{ident}: invalid CI run")
        for a in arts:
            if set(a) != {"role", "path", "git_blob_sha1"} or a["role"] not in ROLES or not isinstance(a["git_blob_sha1"], str) or not SHA40.fullmatch(a["git_blob_sha1"]):
                fail(f"{ident}: invalid artifact metadata")
            safe_path(a["path"])
            if verify_blobs:
                checked_file(root, a, ident)
        if verify_blobs:
            inspect_native(e, root)
    if [e["id"] for e in entries] != sorted(ids):
        fail("registry entries must be lexicographically sorted by stable ID")
    return entries

def render_index(entries: list[dict]) -> str:
    B = chr(96)
    lines = [
        "# Lotto649 — Research Evidence Index", "",
        f"This index is generated from [{B}research/evidence_registry_v1.json{B}](../research/evidence_registry_v1.json).",
        "It describes **recorded evidence at the listed snapshot dates**, not a live status feed or a proof of lottery predictability.",
        "The Git blob hashes pin *original artifacts*, so later source append operations do not silently rewrite historical experiments.",
        "",
        "| Experiment | Domain | Evidence tier | Scientific decision | Engineering | Quantity* | Source |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for e in entries:
        a = e["artifacts"][0]
        values = [e["id"],e["domain"],e["evidence_stage"],e["research_decision"],
                  e["engineering_status"],str(e["sample_size"]),
                  f"[{a['role']}](../{a['path']})"]
        if any("|" in x or "\n" in x for x in values):
            fail("unsafe Markdown table delimiter")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "*Quantity is the original record unit: legal combinations for exact theory, astronomical events for USNO source, historical draws for retrospective studies and evaluated future draws for sealed trials.*",
        "", "## Interpretation", "",
        f"- {B}PROVEN_MATHEMATICALLY{B} describes an exact structural or combinatorial result; it does **not** imply a forecast advantage.",
        f"- {B}NO_EVIDENCE{B} records the scope of a particular exploratory test; it is not a proof that all possible signals are absent.",
        f"- {B}PENDING{B} on a {B}PROSPECTIVE_PRESEALED{B} record means a fixed study exists **before its future outcomes are evaluated**. An issue can be closed for engineering while its statistical result remains pending.",
        f"- {B}PASS{B} means integrity/engineering checks succeeded. It never means {B}VALIDATED_GO{B} or a better chance of winning.",
        "- Source/protocol/result documents retain their original versioned cutoffs; newer official draws must not be silently joined to those older experiments.",
        "", "## How to extend", "",
        "See [Research Knowledge System](research_knowledge_system.md) for domain taxonomy, lifecycle gates, status transitions and examples.",
        "Follow [New Research Hypothesis Template](research_hypothesis_template.md).",
        "Existing [M1 Experiment Ledger](experiment_ledger_m1.md) remains the authority for public seal identities and local private paid registrations.",
        "", "## Validation", "",
        f"{B}{B}{B}powershell",
        "python -m lotto7x7.research_governance verify",
        "python -m unittest discover -s tests -p 'test_research_governance.py' -v",
        f"{B}{B}{B}", "",
    ])
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "render"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        entries = validate_registry(json.loads((root / REGISTRY).read_text(encoding="utf-8")), root)
        expected = render_index(entries)
        if args.command == "render":
            print(expected, end="")
        else:
            actual = (root / INDEX).read_text(encoding="utf-8")
            if actual.replace("\r\n", "\n") != expected:
                fail(f"rendered index out of sync: {INDEX}")
            print(f"PASS: {len(entries)} immutable research entries; scientific status gates, pinned SHA-1 original artifacts and Markdown index")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"RESEARCH GOVERNANCE FAIL: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
