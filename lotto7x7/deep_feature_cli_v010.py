"""CLI for v0.10 GPU CNN-autoencoder and multiscale GRU feature discovery."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch

from .deep_feature_v010 import discover_deep
from .research import DATA, load_history


def configure_parsers(action):
    p = action.add_parser(
        "deep-feature-gpu",
        help="v0.10 GPU deep CNN-GRU periodicity/motif discovery with IID + shuffle controls")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--train-draws", type=int, default=1024,
                   help="Prefix for self-supervised representation training; next segment is independent discovery")
    p.add_argument("--holdout-draws", type=int, default=500)
    p.add_argument("--window", type=int, default=64, help="Causal GRU context in draws")
    p.add_argument("--validation", type=int, default=128)
    p.add_argument("--embedding", type=int, default=16)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--reconstruction-weight", type=float, default=0.7)
    p.add_argument("--recent-windows", type=int, nargs="+", default=[256, 512])
    p.add_argument("--z-half-lives", type=int, nargs="+", default=[0, 64, 256])
    p.add_argument("--motif-windows", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--min-period", type=float, default=2.0)
    p.add_argument("--max-period", type=float, default=128.0)
    p.add_argument("--surrogates", type=int, default=99)
    p.add_argument("--surrogate-batch", type=int, default=4)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--seed", type=int, default=649)
    p.add_argument("--out", type=Path,
                   default=Path("results/v010_deep_feature_cuda.json"))
    p.add_argument("--model-out", type=Path, default=None,
                   help="Trained encoder/GRU checkpoint; default same stem as --out with .pt")


def run(args):
    if args.command != "deep-feature-gpu":
        raise ValueError("unexpected v0.10 command")
    model_path = args.model_out if args.model_out is not None else args.out.with_suffix(".pt")
    if args.out.resolve() == model_path.resolve():
        raise ValueError("JSON and model checkpoint outputs must differ")
    if args.out.exists() or model_path.exists():
        raise FileExistsError("Select new --out / --model-out paths; existing results never overwritten")
    images, _, ids = load_history(args.data)
    print(f"[SOURCE] draws={len(images)} train={args.train_draws} "
          f"discovery={len(images)-args.train_draws-args.holdout_draws} "
          f"heldout={args.holdout_draws}", flush=True)
    report, trained = discover_deep(
        images, ids, device=args.device, train_draws=args.train_draws,
        holdout_draws=args.holdout_draws, window=args.window,
        validation=args.validation, embedding=args.embedding, hidden=args.hidden,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, reconstruction_weight=args.reconstruction_weight,
        recent_windows=tuple(args.recent_windows), half_lives=tuple(args.z_half_lives),
        motif_windows=tuple(args.motif_windows), min_period=args.min_period,
        max_period=args.max_period, surrogates=args.surrogates,
        surrogate_batch=args.surrogate_batch, top_count=args.top,
        seed=args.seed, on_progress=lambda msg: print(msg, flush=True),
        return_model=True)
    # Persist fitted weights once, so later candidate-scoring research can use
    # this exact encoder rather than rerunning the same training.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "lotto7x7.deep_feature_checkpoint.v0.10",
              "architecture": {"embedding": args.embedding, "hidden": args.hidden,
                               "window": args.window},
              "trained_on_draws": args.train_draws,
              "source_image_sha256": report["source_image_sha256"],
              "state_dict": {k: v.detach().cpu().clone()
                             for k, v in trained.state_dict().items()}}
    with model_path.open("xb") as f:
        torch.save(record, f)
    report["model_checkpoint"] = {
        "path": str(model_path),
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "trained_draws": args.train_draws,
        "note": "Frozen research model, NOT a validated future lottery predictor."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(f"[DEVICE] {report['device']} deep_features={report['deep_feature_channels']}", flush=True)
    for name, result in report["nulls"].items():
        print(f"[NULL] {name}: period_global_p={result['period_global_p']:.6f} "
              f"motif_global_p={result['motif_global_p']:.6f} "
              f"(both detectors Bonferroni adjusted in JSON)", flush=True)
    for key, label in (("best_period", "PERIOD"), ("best_motif", "MOTIF")):
        item = report[key]
        if item:
            print(f"[{label}] {item.get('feature', item.get('feature_family'))} "
                  f"strength={item['strength']:.6f} "
                  f"p_iid={item['p_two_detector_adjusted_iid_legal_6of49']:.6f} "
                  f"p_shuffle={item['p_two_detector_adjusted_shuffled_historical_discovery']:.6f}", flush=True)
    print(f"[MODEL SAVED] {model_path.resolve()}", flush=True)
    print(f"[DONE] {args.out.resolve()}", flush=True)
    print("[CAUTION] Exploration only; fixed-encoder conditional nulls, not proof of number-selection advantage", flush=True)
