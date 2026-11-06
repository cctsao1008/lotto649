"""Lotto649 Phase 6: prospective, one-step shadow forecast sealing.

Prepare: create immutable-by-policy forecast JSON before the target draw.
Verify: inspect schema and canonical SHA-256 integrity.
Score: compare a Git-committed seal with a subsequently updated draw file.

The executable cannot establish an authoritative pre-draw timestamp.
Publish the seal to a third-party timestamped host before the draw;
a local clock and Git commit metadata alone are not sufficient evidence.
No six-number selections or bets are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from lotto_historical import load_official_main_draws
from lotto_neural import (IID_LOGP, N, K, SmallMLP, StaticLogitModel,
                          chronological_split, fit_model, make_observations)
from lotto_probability import FixedCardinalityDistribution
from lotto_temporal import fit_beta

VERSION = "0.1.0"
SCHEMA = "lotto649.shadow.forecast.v1"
MODELS = ("iid", "static", "lag", "mlp")
TPE = ZoneInfo("Asia/Taipei")
SOURCE = Path("data/normalized/lotto649_base.npy")
SEAL_DIR = Path("data/shadow/seals")
SCORE_DIR = Path("data/shadow/scores")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_bytes(value: dict) -> bytes:
    """Serialize a JSON value deterministically, rejecting NaN and infinity."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value: dict) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_date(value: str) -> date:
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("date must have exact YYYY-MM-DD format")
    return result


def parse_deadline(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("deadline must have an explicit timezone offset")
    return result.astimezone(timezone.utc)


def observed_prefix_hash(numbers: np.ndarray, dates: np.ndarray,
                         ids: np.ndarray) -> str:
    """Hash chronological main-six outcomes and draw identifiers only."""
    if len(numbers) != len(dates) or len(dates) != len(ids):
        raise ValueError("misaligned observed draw arrays")
    payload = {
        "dates": [int(v) for v in dates],
        "draw_ids": [int(v) for v in ids],
        "numbers": numbers.astype(int).tolist(),
    }
    return digest(payload)


def tracked_clean_git_revision() -> str:
    """Record the repository revision and reject modified tracked code."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True).stdout.strip().lower()
    if not HEX40.fullmatch(head):
        raise ValueError("invalid Git HEAD revision")
    for cmd in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise ValueError("tracked repository files must be clean before preparation")
    return head


def validate_forecast_payload(payload: dict) -> None:
    """Check self-consistency without assuming an external trusted clock."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected forecast schema")
    if payload.get("default_model") != "iid":
        raise ValueError("default prospective model must remain iid")
    if set(payload.get("forecasts", {})) != set(MODELS):
        raise ValueError("expected forecasts for iid, static, lag, mlp")
    target = parse_date(payload["target_date"])
    deadline = parse_deadline(payload["deadline_utc"])
    prepared = parse_deadline(payload["prepared_at_utc"])
    cutoff = date.fromisoformat(str(payload["source"]["last_date"]))
    if target <= cutoff or deadline.astimezone(TPE).date() != target:
        raise ValueError("invalid temporal relationship between target and source")
    if prepared >= deadline:
        raise ValueError("forecast preparation is not before declared deadline")
    if payload["source"]["draw_count"] < 40:
        raise ValueError("insufficient historical draw data")
    for key in ("sha256", "observed_prefix_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", payload["source"][key]):
            raise ValueError("invalid source fingerprint")
    if not HEX40.fullmatch(payload["code_commit"]):
        raise ValueError("invalid source-code revision")
    for name in MODELS:
        logits = payload["forecasts"][name]["logits"]
        if not isinstance(logits, list) or len(logits) != N:
            raise ValueError(f"{name}: expected exactly 49 logits")
        values = np.asarray(logits, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: non-finite logits")
        if name == "iid" and not np.all(values == 0):
            raise ValueError("iid logits must be exactly zero")


def make_seal(payload: dict) -> dict:
    validate_forecast_payload(payload)
    return {"payload": payload, "payload_sha256": digest(payload)}


def verify_seal_file(path: Path) -> dict:
    """Verify deterministic integrity, but NOT when GitHub received the seal."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if set(content) != {"payload", "payload_sha256"}:
        raise ValueError("invalid seal envelope")
    if not re.fullmatch(r"[0-9a-f]{64}", content["payload_sha256"]):
        raise ValueError("invalid payload hash")
    if digest(content["payload"]) != content["payload_sha256"]:
        raise ValueError("forecast payload SHA-256 mismatch")
    validate_forecast_payload(content["payload"])
    return content


def write_once(path: Path, obj: dict) -> None:
    """Create a new file without overwriting an existing seal or report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(obj, sort_keys=True, indent=2,
                            ensure_ascii=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def compute_forecasts(numbers: np.ndarray, *, seed: int = 649,
                      window: int = 5, hidden: int = 16,
                      epochs: int = 40, patience: int = 6,
                      batch_size: int = 128, lr: float = 0.005,
                      weight_decay: float = 0.02) -> tuple[dict, dict]:
    """Fit fixed challengers on old draws; forecast only AFTER the last draw."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train, validation, _test = chronological_split(len(numbers), window)
    history, targets, _ = make_observations(numbers, window)
    def fetch(indices: np.ndarray):
        positions = torch.from_numpy(indices - window)
        return history[positions], targets[positions]
    x_train, y_train = fetch(train)
    x_val, y_val = fetch(validation)
    trained = {}
    traces = {}
    for name, model in (("static", StaticLogitModel()),
                        ("mlp", SmallMLP(window, hidden))):
        fitted, trace = fit_model(
            model, x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val, seed=seed,
            epochs=epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay)
        trained[name], traces[name] = fitted, trace
    # A one-step forecast may use all previously observed draws as context.
    final_history = np.zeros((window, N), dtype=np.float32)
    recent = numbers[-window:]
    final_history[np.arange(window)[:, None], recent - 1] = 1.
    x_next = torch.from_numpy(final_history.reshape(1, -1))
    with torch.no_grad():
        static_logits = trained["static"](x_next)[0].detach().numpy().astype(float)
        mlp_logits = trained["mlp"](x_next)[0].detach().numpy().astype(float)
    beta = fit_beta(numbers, train_end=int(train[-1]) + 1)
    lag_logits = np.zeros(N, dtype=np.float64)
    lag_logits[numbers[-1] - 1] = beta
    forecasts = {
        "iid": {"logits": [0.0] * N},
        "static": {"logits": static_logits.tolist()},
        "lag": {"logits": lag_logits.tolist()},
        "mlp": {"logits": mlp_logits.tolist()},
    }
    meta = {
        "method": "frozen_phase5_60_20_training_and_last_observed_context",
        "seed": seed, "window": window, "hidden": hidden,
        "epochs": epochs, "patience": patience,
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "train_range": [int(train[0]), int(train[-1])],
        "validation_range": [int(validation[0]), int(validation[-1])],
        "static_best_epoch": traces["static"]["best_epoch"],
        "mlp_best_epoch": traces["mlp"]["best_epoch"],
        "lag_train_only_beta": float(beta),
    }
    return forecasts, meta


def prepare(*, source: Path, target_date: str, deadline: str, out: Path,
            max_source_age_days: int = 7, seed: int = 649, window: int = 5,
            hidden: int = 16, epochs: int = 40, patience: int = 6,
            batch_size: int = 128, lr: float = 0.005,
            weight_decay: float = 0.02,
            now: datetime | None = None, code_commit: str | None = None) -> dict:
    """Produce a candidate seal, refusing outdated or future-contaminated data."""
    if out.exists():
        raise FileExistsError(f"seal already exists: {out}")
    if max_source_age_days < 0:
        raise ValueError("source-age limit cannot be negative")
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)
    target = parse_date(target_date)
    close = parse_deadline(deadline)
    if target < current.astimezone(TPE).date():
        raise ValueError("target date is already in the past")
    if close.astimezone(TPE).date() != target or current >= close:
        raise ValueError("target deadline must be on target Taipei date and in the future")
    snapshot_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    numbers, dates, ids = load_official_main_draws(source)
    last = datetime.strptime(str(int(dates[-1])), "%Y%m%d").date()
    age = (current.astimezone(TPE).date() - last).days
    if age < 0 or age > max_source_age_days:
        raise ValueError(f"outdated or future-dated source: last draw {last}, age {age} days")
    if target <= last:
        raise ValueError("target must be later than the newest observed draw")
    revision = code_commit if code_commit is not None else tracked_clean_git_revision()
    forecasts, training = compute_forecasts(
        numbers, seed=seed, window=window, hidden=hidden,
        epochs=epochs, patience=patience, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay)
    if hashlib.sha256(source.read_bytes()).hexdigest() != snapshot_sha:
        raise ValueError("source archive changed during forecast preparation")
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "prepared_at_utc": current.isoformat(timespec="seconds"),
        "deadline_utc": close.isoformat(timespec="seconds"),
        "target_date": target.isoformat(),
        "default_model": "iid",
        "model_policy": "IID baseline; static/lag/mlp diagnostic challengers only",
        "status": "PREPARED_TIME_NOT_EXTERNALLY_ATTESTED",
        "code_commit": revision,
        "source": {
            "path": str(source),
            "sha256": snapshot_sha,
            "observed_prefix_sha256": observed_prefix_hash(numbers, dates, ids),
            "draw_count": len(numbers),
            "last_date": last.isoformat(),
            "last_draw_id": int(ids[-1]),
            "max_source_age_days": max_source_age_days,
        },
        "training": training,
        "forecasts": forecasts,
        "limitations": [
            "One-step complete-six-set probability prediction only.",
            "Local preparation time is not independent proof of publication time.",
            "Git commit metadata alone does not prove the seal was uploaded pre-draw.",
            "No selected six-number sets, three-month claims or wagering advice.",
        ],
    }
    seal = make_seal(payload)
    write_once(out, seal)
    return seal


def verify_committed_content(path: Path, commit: str) -> None:
    """Verify exact bytes are committed, not external publication timing."""
    if not HEX40.fullmatch(commit.lower()):
        raise ValueError("expected full 40-character Git commit SHA")
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          check=True, capture_output=True, text=True).stdout.strip()
    root = Path(repo).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("seal path must be within the repository") from exc
    result = subprocess.run(["git", "show", f"{commit}:{relative.as_posix()}"],
                            check=True, capture_output=True)
    if result.stdout != path.read_bytes():
        raise ValueError("working seal differs from committed seal bytes")


def score(*, seal_path: Path, updated_source: Path, seal_commit: str,
          out: Path, now: datetime | None = None,
          skip_git_verification: bool = False) -> dict:
    """Score sealed probabilities against the FIRST new draw after cutoff."""
    if out.exists():
        raise FileExistsError(f"score report already exists: {out}")
    seal = verify_seal_file(seal_path)
    if not skip_git_verification:
        verify_committed_content(seal_path, seal_commit)
    payload = seal["payload"]
    current = (now if now is not None else datetime.now(timezone.utc))
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if current.astimezone(timezone.utc) <= parse_deadline(payload["deadline_utc"]):
        raise ValueError("cannot score before the declared draw deadline")
    numbers, dates, ids = load_official_main_draws(updated_source)
    original_n = payload["source"]["draw_count"]
    if len(numbers) <= original_n:
        raise ValueError("target draw has not been appended")
    if (observed_prefix_hash(numbers[:original_n], dates[:original_n],
                             ids[:original_n])
            != payload["source"]["observed_prefix_sha256"]):
        raise ValueError("historical source prefix differs from sealed snapshot")
    target_date = int(parse_date(payload["target_date"]).strftime("%Y%m%d"))
    if int(dates[original_n]) != target_date:
        raise ValueError("first new observed draw does not match target date")
    result_numbers = numbers[original_n].tolist()
    scored = {}
    for name in MODELS:
        logits = payload["forecasts"][name]["logits"]
        logp = (IID_LOGP if name == "iid"
                else FixedCardinalityDistribution(logits, K).log_prob(result_numbers))
        scored[name] = {
            "log_probability": float(logp),
            "lift_vs_iid_nats": float(logp - IID_LOGP),
        }
    report = {
        "schema": "lotto649.shadow.score.v1",
        "status": ("TEST_ONLY_GIT_VERIFICATION_SKIPPED"
                   if skip_git_verification
                   else "CONTENT_COMMIT_VERIFIED_PRE_DRAW_PUBLICATION_UNVERIFIED"),
        "seal_payload_sha256": seal["payload_sha256"],
        "seal_file_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        "seal_commit": seal_commit,
        "target_date": payload["target_date"],
        "target_draw_id": int(ids[original_n]),
        "source_updated_sha256": hashlib.sha256(updated_source.read_bytes()).hexdigest(),
        "default_model": "iid",
        "scored_models": scored,
        "limitations": [
            "Git commit data does not independently establish pre-draw publication time.",
            "Independent server-side timestamp evidence must be checked separately.",
            "No ticket selections or monetary interpretation.",
        ],
    }
    write_once(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 shadow forecast seal, verify, score")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--target-date", required=True, help="YYYY-MM-DD, future Taipei draw date")
    p.add_argument("--deadline", required=True,
                   help="Official pre-draw cutoff with timezone, e.g. YYYY-MM-DDTHH:MM:SS+08:00")
    p.add_argument("--out", type=Path)
    p.add_argument("--max-source-age-days", type=int, default=7)
    p.add_argument("--seed", type=int, default=649)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.02)
    v = commands.add_parser("verify")
    v.add_argument("--seal", type=Path, required=True)
    v.add_argument("--seal-commit", help="Optionally verify identical Git commit bytes")
    s = commands.add_parser("score")
    s.add_argument("--seal", type=Path, required=True)
    s.add_argument("--updated-source", type=Path, default=SOURCE)
    s.add_argument("--seal-commit", required=True)
    s.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        dst = args.out or SEAL_DIR / f"forecast_{args.target_date.replace('-', '')}.json"
        result = prepare(
            source=args.source, target_date=args.target_date,
            deadline=args.deadline, out=dst,
            max_source_age_days=args.max_source_age_days,
            seed=args.seed, window=args.window, hidden=args.hidden,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay)
        print(f"[PREPARED] {dst.resolve()}")
        print(f"Payload SHA-256: {result['payload_sha256']}")
        print("[NEXT] Review official draw cutoff, then commit and push the seal BEFORE the draw")
        print("[CAUTION] Local creation and Git timestamps are not trusted third-party evidence")
    elif args.command == "verify":
        result = verify_seal_file(args.seal)
        if args.seal_commit:
            verify_committed_content(args.seal, args.seal_commit)
        print(f"[PASS] Payload SHA-256: {result['payload_sha256']}")
        print(f"[INFO] Default model: {result['payload']['default_model']}")
        print("[CAUTION] Publication timing has NOT been verified")
    else:
        dst = args.out or SCORE_DIR / f"score_{verify_seal_file(args.seal)['payload']['target_date'].replace('-', '')}.json"
        result = score(seal_path=args.seal, updated_source=args.updated_source,
                       seal_commit=args.seal_commit, out=dst)
        for name, metrics in result["scored_models"].items():
            print(f"  {name:6s} logP={metrics['log_probability']:+.8f} "
                  f"lift={metrics['lift_vs_iid_nats']:+.8f}")
        print(f"[SCORED] {dst.resolve()}")
        print("[CAUTION] Pre-draw independent publication time remains unverified")


if __name__ == "__main__":
    main()
