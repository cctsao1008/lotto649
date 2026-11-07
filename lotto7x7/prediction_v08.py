"""v0.8 causal GPU probability research for 49 Lotto649 main-number marginals.

Expanding-history block retraining: every model fit uses only draws strictly
preceding the block; within-block inference reads only past draw contexts.
Outputs per-number marginals summing to six, proper scoring rules, and Top-6
hit rates against uniform, smoothed-frequency, and recency baselines.
No forecasting edge is implied by this implementation.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from math import log, comb
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "lotto7x7.probability_replay.v0.8"
METHODS = ("iid-uniform", "prefix-frequency", "recent-ewma", "image-cnn-gru")
BASE_PROB = 6.0 / 49.0


def _validate(images, draw_ids):
    x = np.asarray(images)
    ids = np.asarray(draw_ids)
    if (x.ndim != 3 or x.shape[1:] != (7, 7) or len(x) < 60
            or not np.issubdtype(x.dtype, np.number)
            or not np.all((x == 0) | (x == 1))
            or not np.all(x.sum(axis=(1, 2)) == 6)
            or ids.shape != (len(x),)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("expected ordered 7x7 six-hot main-number images and draw IDs")
    return np.asarray(x, dtype=np.float32).reshape(-1, 49), ids.astype(np.int64)


def device_for(choice: str):
    if choice not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto, cuda, or cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (choice == "cuda" or
                        (choice == "auto" and torch.cuda.is_available())) else "cpu")


class ProbabilityCNNGRU(nn.Module):
    """Causal 7x7 frame encoder plus temporal GRU, 49 marginal logits."""
    def __init__(self, embedding: int = 24, hidden: int = 32):
        super().__init__()
        if embedding < 4 or hidden < 4:
            raise ValueError("embedding and hidden must be >= 4")
        self.conv = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(8, 12, 3, padding=1), nn.GELU(),
                                  nn.Flatten(), nn.Linear(12 * 7 * 7, embedding), nn.GELU())
        self.rnn = nn.GRU(embedding, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 49)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, log(BASE_PROB / (1.0 - BASE_PROB)))

    def forward(self, window):
        if window.ndim != 4 or window.shape[1] < 2 or window.shape[2:] != (7, 7):
            raise ValueError("expected [batch,window>=2,7,7]")
        b, w, _, _ = window.shape
        z = self.conv(window.reshape(b * w, 1, 7, 7)).reshape(b, w, -1)
        z, _ = self.rnn(z)
        return self.head(z[:, -1, :])


def project_six(logits: torch.Tensor) -> torch.Tensor:
    """Constrain marginal sums to six without claiming empirical calibration.

    This is a constraint on 49 marginal probabilities, NOT an independent
    Bernoulli model for the six numbers or a jackpot-probability estimate.
    """
    if logits.ndim != 2 or logits.shape[1] != 49 or not bool(torch.isfinite(logits).all()):
        raise ValueError("expected finite [batch,49] logits")
    # Center logits before the shared-intercept bisection to handle wide ranges.
    centered = logits - logits.mean(dim=1, keepdim=True)
    lower = centered.new_full((len(centered), 1), -80.)
    upper = centered.new_full((len(centered), 1), 80.)
    for _ in range(38):
        mid = (lower + upper) * 0.5
        greater = torch.sigmoid(centered + mid).sum(dim=1, keepdim=True) > 6.0
        upper = torch.where(greater, mid, upper)
        lower = torch.where(greater, lower, mid)
    return torch.sigmoid(centered + (lower + upper) * 0.5)


def _windows(labels: np.ndarray, indices: np.ndarray, window: int):
    """Window for t only contains labels [t-window, t), never target t."""
    if (indices.ndim != 1 or np.any(indices < window)
            or np.any(indices >= len(labels)) or len(indices) == 0):
        raise ValueError("invalid causal window indices")
    return np.stack([labels[t-window:t].reshape(window, 7, 7)
                     for t in indices]).astype(np.float32)


def _fit_prefix(labels: np.ndarray, cutoff: int, *, window: int, validation: int,
                epochs: int, patience: int, batch_size: int, learning_rate: float,
                seed: int, device: torch.device, embedding: int, hidden: int):
    """Fit on a prefix only; model selection uses validation within prefix."""
    if not window + validation + 16 <= cutoff <= len(labels):
        raise ValueError("insufficient historical training prefix")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_indices = np.arange(window, cutoff - validation, dtype=np.int64)
    val_indices = np.arange(cutoff - validation, cutoff, dtype=np.int64)
    train_x = torch.as_tensor(_windows(labels, train_indices, window), device=device)
    train_y = torch.as_tensor(labels[train_indices], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(_windows(labels, val_indices, window), device=device)
    val_y = torch.as_tensor(labels[val_indices], dtype=torch.float32, device=device)
    net = ProbabilityCNNGRU(embedding, hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    best, best_epoch, no_progress, trace = None, -1, 0, []
    for epoch in range(epochs):
        net.train()
        permutation = torch.randperm(len(train_indices), device=device)
        for ix in permutation.split(batch_size):
            logits = net(train_x[ix])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        net.eval()
        with torch.no_grad():
            projected = project_six(net(val_x))
            val_brier = float((projected - val_y).square().mean().item())
        trace.append(val_brier)
        if val_brier < (min(trace[:-1]) - 1e-7 if len(trace)>1 else float('inf')):
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch + 1
            no_progress = 0
        else:
            no_progress += 1
            if no_progress >= patience:
                break
    if best is None:
        raise AssertionError("no best validation checkpoint")
    net.load_state_dict(best)
    net.eval()
    return net, {"fit_cutoff_index": cutoff,
                 "training_target_indices": [int(train_indices[0]), int(train_indices[-1])],
                 "validation_target_indices": [int(val_indices[0]), int(val_indices[-1])],
                 "best_epoch": best_epoch, "validation_brier": min(trace),
                 "epochs_run": len(trace)}


def _baseline_curves(labels: np.ndarray, *, half_life: float = 32., pseudo: float = 40.):
    """Prefix-only smoothed marginals; both sum to six for every target."""
    n = len(labels)
    global_prob = np.empty((n, 49), dtype=np.float64)
    recent_prob = np.empty((n, 49), dtype=np.float64)
    total = np.zeros(49, dtype=np.float64)
    weighted = np.zeros(49, dtype=np.float64)
    weight = 0.
    decay = 2 ** (-1. / half_life)
    for t in range(n):
        global_prob[t] = (total + pseudo * BASE_PROB) / (t + pseudo)
        recent_prob[t] = (weighted + pseudo * BASE_PROB) / (weight + pseudo)
        total += labels[t]
        weighted = weighted * decay + labels[t]
        weight = weight * decay + 1
    return global_prob, recent_prob


def _top6(prob: np.ndarray, *, seed: int, t: int, method: str):
    if method == "iid-uniform":
        # Uniform marginal probabilities contain no ranking; choose fair seed-only tie-breaking.
        rng = np.random.default_rng(seed + 7919 * t)
        return sorted(map(int, rng.choice(49, 6, replace=False) + 1))
    order = np.lexsort((np.arange(49), -prob))[:6] + 1
    return sorted(map(int, order))


def _boot_mean(values, *, seed: int, reps: int = 1200):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = np.empty(reps)
    for start in range(0, reps, 128):
        stop = min(start + 128, reps)
        ix = rng.integers(0, len(x), size=(stop-start, len(x)))
        sample[start:stop] = x[ix].mean(axis=1)
    return [float(z) for z in np.quantile(sample, [.025,.975])]


def replay(images, draw_ids, *, warmup=200, block_size=100, window=5,
           validation=32, epochs=12, patience=3, batch_size=64,
           learning_rate=.002, embedding=24, hidden=32, seed=649,
           device="auto", tail_targets=None,
           on_progress: Callable[[str], None] | None = None):
    """Full chronology: train only before each block; no in-block weight update.

    Every forecast at t has features solely from [t-window,t). Earlier known
    outcomes inside the block can be used for subsequent forecasts, as online.
    """
    labels, ids = _validate(images, draw_ids)
    if (type(warmup) is not int or type(block_size) is not int
            or type(window) is not int or type(validation) is not int
            or type(epochs) is not int or type(patience) is not int
            or type(batch_size) is not int or type(seed) is not int
            or not 60 <= warmup < len(labels) or block_size < 1 or window < 2
            or validation < 8 or epochs < 1 or patience < 1 or batch_size < 1
            or seed < 0 or not 0 < learning_rate <= .1):
        raise ValueError("invalid chronological GPU training parameters")
    if warmup <= window + validation + 16:
        raise ValueError("warmup leaves insufficient training records")
    if tail_targets is not None and (type(tail_targets) is not int or tail_targets < 1):
        raise ValueError("tail-targets must be positive")
    dev = device_for(device)
    start = warmup if tail_targets is None else max(warmup, len(labels)-tail_targets)
    global_prob, recent_prob = _baseline_curves(labels)
    predicted = np.empty((len(labels)-start, 49), dtype=np.float32)
    blocks = []
    for begin in range(start, len(labels), block_size):
        stop = min(begin+block_size, len(labels))
        model, train_info = _fit_prefix(
            labels, begin, window=window, validation=validation,
            epochs=epochs, patience=patience, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed+begin,
            device=dev, embedding=embedding, hidden=hidden)
        indices = np.arange(begin, stop, dtype=np.int64)
        features = torch.as_tensor(_windows(labels, indices, window), device=dev)
        with torch.no_grad():
            p = torch.cat([project_six(model(chunk))
                          for chunk in features.split(batch_size)], dim=0)
        predicted[begin-start:stop-start] = p.cpu().numpy()
        blocks.append({**train_info, "evaluation_target_indices": [begin,stop-1]})
        if on_progress:
            on_progress(f"[GPU BLOCK {len(blocks)}] fitted_through={int(ids[begin-1])} "
                        f"targets={begin-start+1}..{stop-start}/{len(labels)-start} "
                        f"val_brier={train_info['validation_brier']:.6f}")
    curves = {
        "iid-uniform": np.full((len(labels)-start,49), BASE_PROB, dtype=np.float64),
        "prefix-frequency": global_prob[start:],
        "recent-ewma": recent_prob[start:],
        "image-cnn-gru": predicted.astype(np.float64),
    }
    for name, p in curves.items():
        if not np.all(np.isfinite(p)) or not np.allclose(p.sum(axis=1), 6, atol=2e-4):
            raise AssertionError(f"invalid six-number probability calibration: {name}")
    targets = []
    for pos, t in enumerate(range(start, len(labels))):
        actual = sorted(map(int, np.flatnonzero(labels[t]) + 1))
        pred = {}
        for name, probs in curves.items():
            row = probs[pos]
            picks = _top6(row, seed=seed, t=t, method=name)
            hit = len(set(picks).intersection(actual))
            eps = 1e-7
            clipped = np.clip(row, eps, 1-eps)
            brier = float(np.square(row - labels[t]).mean())
            ll = float(-(labels[t]*np.log(clipped)+(1-labels[t])*np.log(1-clipped)).mean())
            pred[name] = {"probabilities_1_to_49": [round(float(v),9) for v in row],
                          "top6_numbers": picks, "top6_hits": hit,
                          "brier": brier, "binary_log_loss": ll}
        targets.append({"target_draw_id": int(ids[t]),
                        "cutoff_draw_id": int(ids[t-1]),
                        "actual_main_numbers": actual, "predictions": pred})
    summary = {}
    by_year = defaultdict(list)
    for i,r in enumerate(targets):
        by_year[str(r['target_draw_id']//1000000+1911)].append(i)
    baseline_brier = np.asarray([r['predictions']['iid-uniform']['brier'] for r in targets])
    for j, name in enumerate(METHODS):
        p = [r['predictions'][name] for r in targets]
        briers = np.asarray([v['brier'] for v in p])
        hit = np.asarray([v['top6_hits'] for v in p])
        improvement = baseline_brier - briers
        summary[name] = {"draws":len(targets), "mean_brier": float(briers.mean()),
                         "mean_binary_log_loss": float(np.mean([v['binary_log_loss'] for v in p])),
                         "brier_skill_vs_iid": float(improvement.mean()),
                         "draw_bootstrap_skill_ci95": _boot_mean(improvement, seed=seed+123+j),
                         "mean_top6_hits": float(hit.mean()),
                         "top6_ge3_rate": float(np.mean(hit>=3)),
                         "top6_hit_histogram": [int(np.sum(hit==k)) for k in range(7)],
                         "by_year": {year: {"draws":len(ix),
                                             "mean_brier":float(np.mean(briers[ix])),
                                             "mean_hits":float(np.mean(hit[ix]))}
                                     for year,ix in sorted(by_year.items())}}
    # Separate latest-data prospective-ready research forecast; no unknown draw
    # is available and no future result enters fitting or inference.
    latest_model, latest_train = _fit_prefix(
        labels, len(labels), window=window, validation=validation,
        epochs=epochs, patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed+len(labels),
        device=dev, embedding=embedding, hidden=hidden)
    latest_input = torch.as_tensor(labels[-window:].reshape(1,window,7,7),device=dev)
    with torch.no_grad():
        latest_prob = project_six(latest_model(latest_input))[0].cpu().numpy()
    augmented = np.vstack((labels, np.zeros((1,49), dtype=np.float32)))
    next_global, next_recent = _baseline_curves(augmented)
    next_curves = {"iid-uniform": np.full(49, BASE_PROB),
                   "prefix-frequency":next_global[-1],
                   "recent-ewma":next_recent[-1],
                   "image-cnn-gru":latest_prob}
    forecast = {
        "after_last_known_draw_id":int(ids[-1]),
        "actual_next_draw_not_available":True,
        "latest_model_fit":latest_train,
        "models":{name:{"probabilities_1_to_49":[round(float(v),9) for v in prob],
                        "top6_numbers":_top6(prob,seed=seed,t=len(labels),method=name)}
                  for name,prob in next_curves.items()}}
    importances = {name:summary[name]['brier_skill_vs_iid'] for name in METHODS}
    p_ge3 = sum(comb(6, k)*comb(43, 6-k) for k in range(3,7))/comb(49,6)
    return {"schema":SCHEMA, "research_only":True,
            "evaluation_kind":"expanding-history-block-frozen-rolling-origin",
            "source_image_sha256": hashlib.sha256(np.asarray(images,dtype=np.uint8).tobytes()+ids.tobytes()).hexdigest(),
            "source_draws":len(labels), "evaluated_draws":len(targets),
            "source_first_draw_id":int(ids[0]), "source_last_draw_id":int(ids[-1]),
            "compute":{"device":str(dev),"torch_version":torch.__version__,
                       "gpu_name":torch.cuda.get_device_name(dev) if dev.type=='cuda' else None,
                       "model":"CNN+GRU 49 marginal logits; shared-logit probability projection sum=6"},
            "config":{"warmup":warmup,"block_size":block_size,"window":window,
                      "validation":validation,"epochs":epochs,"patience":patience,
                      "batch_size":batch_size,"learning_rate":learning_rate,
                      "embedding":embedding,"hidden":hidden,"seed":seed,
                      "tail_targets":tail_targets},
            "training_blocks":blocks, "summary":summary,
            "iid_reference":{"one_ticket_mean_hits":36/49,
                             "one_ticket_ge3_probability":p_ge3,
                             "full_jackpot_probability":1/comb(49,6)},
            "next_forecast":forecast,
            "retrospective_best_brier_skill_name":max(importances, key=importances.get),
            "targets":targets,
            "warnings":["Retrospective exploration and previously inspected history, not prospective validation.",
                        "Only individual number marginals are modeled. A sum-six projection does not enforce joint probabilities or empirical calibration.",
                        "Retrospective best model is descriptive only; do not select a live model by looking at this evaluation.",
                        "Lottery draws are expected IID; observed improvements need replication and prospective validation.",
                        "Uniform top-six ties use seed-only random selection, not winning numbers.",
                        "Within-block model weights freeze; previous released draws may update input contexts only.",
                        "Model selection (early stopping) uses prefix validation only; no unseen target outcomes.",
                        "This is probability research, NOT a proven winning-number predictor."]}
