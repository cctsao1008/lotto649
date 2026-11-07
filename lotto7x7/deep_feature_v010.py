"""v0.10 GPU deep-feature research: frozen CNN autoencoder + multiscale GRU.

A historical image-only model is fit on an early prefix, then its fixed
representation is examined for spectral peaks and repeated subsequences on a
separate discovery segment. Conditional max-null distributions use legal 6/49
IID images and permutations of the discovery draws. The suffix is reserved for
predeclared next-draw Brier scoring, not for feature selection.

Neither feature recurrence nor image reconstruction implies predictive skill.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .feature_discovery_v09 import (
    _scan_periods, iid_six_hot, resolve_device, shuffled_history,
    validate_images,
)
from .prediction_v08 import project_six

SCHEMA = "lotto7x7.deep_feature.v0.10"
IID_PROB = 6.0 / 49.0


def causal_rolling_mean(z: torch.Tensor, count: int) -> torch.Tensor:
    """Causal trailing mean including the current position, no future context.

    z: [B,T,D], count>=1; previous positions only at time t.
    """
    if z.ndim != 3 or count < 1:
        raise ValueError("expected [batch,time,features] and positive count")
    csum = F.pad(z.cumsum(dim=1), (0, 0, 1, 0))
    t = z.shape[1]
    hi = torch.arange(1, t + 1, device=z.device)
    lo = (hi - count).clamp_min(0)
    counts = (hi - lo).to(z.dtype).view(1, t, 1)
    return (csum[:, hi] - csum[:, lo]) / counts


class DeepFeatureNet(nn.Module):
    """Self-supervised image encoder/decoder + causal multi-scale GRU head.

    The decoder reconstructs all known frames. A GRU reads embeddings,
    embedding differences and causal 4/16/64 lag averages to forecast the
    next six-hot bitmap. Hidden features are then inspected, not presumed useful.
    """

    def __init__(self, embedding: int = 16, hidden: int = 24):
        super().__init__()
        if embedding < 4 or hidden < 4:
            raise ValueError("embedding and hidden must both be >=4")
        self.embedding = embedding
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 12, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(12 * 7 * 7, embedding), nn.Tanh(),
        )
        self.decoder = nn.Linear(embedding, 49)
        self.temporal = nn.GRU(embedding * 5, hidden, batch_first=True)
        self.predictor = nn.Linear(hidden, 49)
        nn.init.zeros_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias, math.log(IID_PROB / (1. - IID_PROB)))
        nn.init.constant_(self.decoder.bias, math.log(IID_PROB / (1. - IID_PROB)))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """[B,T,7,7] -> [B,T,embedding]."""
        if images.ndim != 4 or images.shape[2:] != (7, 7):
            raise ValueError("expected [batch,time,7,7] images")
        b, t = images.shape[:2]
        return self.encoder(images.reshape(b * t, 1, 7, 7)).reshape(b, t, -1)

    @staticmethod
    def multiscale(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError("expected [batch,time,latent]")
        diff = torch.cat((torch.zeros_like(latent[:, :1]), latent[:, 1:] - latent[:, :-1]), dim=1)
        return torch.cat((latent, diff,
                          causal_rolling_mean(latent, 4),
                          causal_rolling_mean(latent, 16),
                          causal_rolling_mean(latent, 64)), dim=-1)

    def temporal_state(self, latent: torch.Tensor) -> torch.Tensor:
        out, _ = self.temporal(self.multiscale(latent))
        return out[:, -1, :]

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(windows)
        decoded = self.decoder(z)
        next_logits = self.predictor(self.temporal_state(z))
        return decoded, next_logits


def next_windows(images: torch.Tensor, indices: torch.Tensor, window: int) -> torch.Tensor:
    """For target t use only images[t-window:t]."""
    if (images.ndim != 3 or images.shape[1:] != (7, 7) or window < 2 or
            indices.ndim != 1 or len(indices) < 1 or
            int(indices.min()) < window or int(indices.max()) >= len(images)):
        raise ValueError("invalid causal next-image windows")
    return images.unfold(0, window, 1).permute(0, 3, 1, 2)[indices - window]


def train_prefix(images: torch.Tensor, *, train_draws: int, validation: int,
                 window: int, embedding: int, hidden: int, epochs: int,
                 patience: int, batch_size: int, learning_rate: float,
                 reconstruction_weight: float, seed: int,
                 on_progress: Callable[[str], None] | None = None
                 ) -> tuple[DeepFeatureNet, dict]:
    """Train exactly once on an early chronological prefix, not on scan/holdout."""
    if not (window >= 2 and validation >= 8 and
            window + validation + 16 <= train_draws <= len(images) and
            epochs >= 1 and patience >= 1 and batch_size >= 1 and
            0 < learning_rate <= 0.1 and 0 <= reconstruction_weight <= 10 and
            seed >= 0):
        raise ValueError("invalid early-prefix training configuration")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = DeepFeatureNet(embedding, hidden).to(images.device)
    train_ids = torch.arange(window, train_draws - validation, device=images.device)
    val_ids = torch.arange(train_draws - validation, train_draws, device=images.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    losses = []

    def evaluate(ids: torch.Tensor) -> tuple[float, float]:
        net.eval()
        brier = 0.
        recon = 0.
        with torch.no_grad():
            for ix in ids.split(batch_size):
                w = next_windows(images, ix, window)
                dec, logits = net(w)
                p = project_six(logits)
                actual = images[ix].reshape(-1, 49)
                brier += float((p - actual).square().mean(dim=-1).sum())
                recon += float(F.binary_cross_entropy_with_logits(
                    dec, w.reshape(len(ix), window, 49), reduction="none"
                ).mean(dim=(1, 2)).sum())
        return brier / len(ids), recon / len(ids)

    for ep in range(epochs):
        net.train()
        shuffled = train_ids[torch.randperm(len(train_ids), device=images.device)]
        for ix in shuffled.split(batch_size):
            windows = next_windows(images, ix, window)
            dec, logits = net(windows)
            targets = images[ix].reshape(-1, 49)
            reconstruction = F.binary_cross_entropy_with_logits(
                dec, windows.reshape(len(ix), window, 49))
            objective = F.binary_cross_entropy_with_logits(logits, targets)
            loss = objective + reconstruction_weight * reconstruction
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        val_brier, val_recon = evaluate(val_ids)
        losses.append({"epoch": ep + 1, "validation_brier": val_brier,
                       "validation_reconstruction_bce": val_recon})
        if val_brier < best_val - 1.e-7:
            best_val = val_brier
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_epoch = ep + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    if on_progress:
        on_progress(f"[TRAIN] examples={len(train_ids)} val={len(val_ids)} "
                    f"best_epoch={best_epoch} val_brier={best_val:.7f} device={images.device}")
    return net, {"training_targets": [window, train_draws - validation - 1],
                 "validation_targets": [train_draws - validation, train_draws - 1],
                 "best_epoch": best_epoch, "epochs_run": len(losses),
                 "best_validation_brier": best_val, "trace": losses}


def deep_bank(net: DeepFeatureNet, series: torch.Tensor, *, window: int,
              chunk: int = 128) -> tuple[torch.Tensor, list[str]]:
    """Embed each completed image, then causal GRU state over past <=current.

    series [B,T,7,7], return [B,T-window+1,embedding+hidden].
    The first window-1 draws are context-only; no target labels are read.
    """
    if series.ndim != 4 or series.shape[2:] != (7, 7) or series.shape[1] < window or chunk < 1:
        raise ValueError("invalid deep feature series/window/chunk")
    b, t = series.shape[:2]
    latent_list = []
    net.eval()
    with torch.no_grad():
        for flat in series.reshape(b * t, 7, 7).split(chunk * window):
            latent_list.append(net.encoder(flat[:, None]))
        latent = torch.cat(latent_list).reshape(b, t, net.embedding)
        # unfold -> [B,T-window+1,embedding,window]
        windows = latent.unfold(1, window, 1).permute(0, 1, 3, 2)
        windows = windows.reshape(-1, window, net.embedding)
        outputs = []
        for w in windows.split(chunk):
            outputs.append(net.temporal_state(w))
        states = torch.cat(outputs).reshape(b, t - window + 1, net.hidden)
    bank = torch.cat((latent[:, window - 1:, :], states), dim=-1)
    names = ([f"cnn_latent/feature_{i:02d}" for i in range(net.embedding)] +
             [f"gru_state/feature_{i:02d}" for i in range(net.hidden)])
    return bank, names


def motif_scan(bank: torch.Tensor, names: list[str], windows: tuple[int, ...],
               *, top_count: int = 12) -> tuple[torch.Tensor, list[dict]]:
    """GPU matrix-profile-style nearest nontrivial subsequence search.

    Euclidean z-normalized subsequence distance corresponds monotonically to
    cosine similarity of per-feature normalized subsequences. Excludes
    overlapping subsequences with offset < motif window length. Scan all
    lengths and each entire CNN or GRU latent family; maxima calibrate null.
    """
    if bank.ndim != 3 or bank.shape[-1] != len(names) or not windows:
        raise ValueError("invalid motif bank or windows")
    batch, n, f = bank.shape
    families = [("cnn_latent", [i for i, name in enumerate(names) if name.startswith("cnn_latent/")]),
                ("gru_state", [i for i, name in enumerate(names) if name.startswith("gru_state/")])]
    if any(not ids for _, ids in families):
        raise ValueError("both deep feature families required")
    best_global = torch.full((batch,), -1.0, device=bank.device)
    candidates: list[dict] = []
    for m in dict.fromkeys(windows):
        if m < 3 or 2 * m >= n:
            raise ValueError("motif length must be >=3 and less than half of discovery sequence")
        starts = n - m + 1
        seq_ix = torch.arange(starts, device=bank.device)
        forbidden = (seq_ix[:, None] - seq_ix[None, :]).abs() < m
        for family, feature_ix in families:
            # [B,starts,m,features] -> z-normalize per subsequence per channel
            source = bank[..., feature_ix]
            views = source.unfold(1, m, 1).permute(0, 1, 3, 2)
            centered = views - views.mean(dim=2, keepdim=True)
            scale = centered.square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.e-4)
            normalized = centered / scale
            flat = normalized.reshape(batch, starts, -1)
            norm = flat.norm(dim=-1, keepdim=True)
            unit = flat / norm.clamp_min(1.e-5)
            cosine = torch.bmm(unit, unit.transpose(1, 2))
            # Degenerate subsequences cannot be claimed as perfect motifs.
            viable = norm.squeeze(-1) > 1.e-3
            cosine = cosine.masked_fill(forbidden[None], -2.)
            cosine = cosine.masked_fill(~viable[:, :, None], -2.)
            cosine = cosine.masked_fill(~viable[:, None, :], -2.)
            maxima = cosine.reshape(batch, -1).max(dim=1).values
            best_global = torch.maximum(best_global, maxima)
            if batch == 1 and top_count > 0:
                flat_idx = int(cosine[0].reshape(-1).argmax())
                i, j = divmod(flat_idx, starts)
                candidates.append({"feature_family": family,
                                   "subsequence_length": int(m),
                                   "start_index": i, "match_index": j,
                                   "strength": float(maxima[0])})
    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return best_global, candidates[:top_count]


def _iid_bootstrap(values: np.ndarray, *, seed: int, count: int = 1200) -> list[float]:
    """Descriptive draw-wise bootstrap; block sensitivity checked separately."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    points = np.empty(count, dtype=float)
    for i in range(0, count, 100):
        group = min(100, count - i)
        ix = rng.integers(0, len(x), size=(group, len(x)))
        points[i:i + group] = x[ix].mean(axis=1)
    return [float(x) for x in np.quantile(points, [.025, .975])]


def _score_future(net: DeepFeatureNet, data: torch.Tensor, *, start: int,
                  window: int, batch_size: int, seed: int) -> dict:
    """Predeclared historical suffix score, no training or hyperparam tuning."""
    ids = torch.arange(start, len(data), device=data.device)
    if len(ids) < 1:
        return {"evaluated_targets": 0}
    score = []
    top6_hits = []
    net.eval()
    with torch.no_grad():
        for ix in ids.split(batch_size):
            _, logits = net(next_windows(data, ix, window))
            prob = project_six(logits)
            actual = data[ix].reshape(-1, 49)
            score.extend((prob - actual).square().mean(dim=-1).cpu().numpy().tolist())
            picks = prob.topk(6, dim=-1).indices
            top6_hits.extend(actual.gather(1, picks).sum(dim=-1).cpu().numpy().tolist())
    values = np.asarray(score)
    iid_brier = IID_PROB * (1. - IID_PROB)
    difference = iid_brier - values
    hits = np.asarray(top6_hits)
    return {"evaluated_targets": len(ids), "iid_brier": float(iid_brier),
            "cnn_gru_brier": float(values.mean()),
            "brier_skill_vs_iid": float(difference.mean()),
            "brier_skill_draw_ci95": _iid_bootstrap(difference, seed=seed + 113),
            "mean_top6_hits": float(hits.mean()),
            "ge3_count": int((hits >= 3).sum()),
            "ge3_rate": float((hits >= 3).mean()),
            "note": "Predeclared suffix evaluation; not prospective blind because prior v0.8 studies used these draws."}


def discover_deep(images: np.ndarray, draw_ids: np.ndarray, *, device: str = "auto",
                  train_draws: int = 1024, holdout_draws: int = 500,
                  window: int = 64, validation: int = 128,
                  embedding: int = 16, hidden: int = 24,
                  epochs: int = 12, patience: int = 3, batch_size: int = 64,
                  learning_rate: float = 0.002, reconstruction_weight: float = 0.7,
                  recent_windows: tuple[int, ...] = (256, 512),
                  half_lives: tuple[int, ...] = (0, 64, 256),
                  motif_windows: tuple[int, ...] = (8, 16, 32),
                  min_period: float = 2., max_period: float = 128.,
                  surrogates: int = 99, surrogate_batch: int = 4,
                  top_count: int = 12, seed: int = 649,
                  on_progress: Callable[[str], None] | None = None,
                  return_model: bool = False) -> dict | tuple[dict, DeepFeatureNet]:
    """One real GPU CNN-GRU fit followed by IID/shuffle-calibrated discovery.

    Discovery uses a separate portion that the model's weights never saw.
    All null replicate feature maps use the SAME frozen encoder. Consequently
    p-values are conditional on this encoder, not retrained-model nulls.
    Two detector families (spectral, motif) receive Bonferroni correction.
    """
    x = validate_images(images)
    ids = np.asarray(draw_ids)
    if (ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("draw IDs must be unique strictly ascending positive integers")
    n = len(x)
    discovery_end = n - holdout_draws
    discovery_size = discovery_end - train_draws
    if not (type(seed) is int and seed >= 0 and
            type(train_draws) is int and type(holdout_draws) is int and
            window + validation + 16 <= train_draws < discovery_end and
            holdout_draws >= 16 and discovery_size >= 2 * max(motif_windows) + 1 and
            surrogates >= 9 and surrogate_batch >= 1 and top_count >= 1 and
            all(int(m) == m and m >= 3 for m in motif_windows) and
            recent_windows and half_lives and
            2 <= min_period < max_period and
            all(int(h) == h and h >= 0 for h in half_lives) and
            all(int(w) == w and w >= 16 for w in recent_windows)):
        raise ValueError("invalid deep-feature discovery configuration")
    dev = resolve_device(device)
    tensor = torch.as_tensor(x, dtype=torch.float32, device=dev)
    net, fit = train_prefix(tensor, train_draws=train_draws, validation=validation,
                            window=window, embedding=embedding, hidden=hidden,
                            epochs=epochs, patience=patience,
                            batch_size=batch_size, learning_rate=learning_rate,
                            reconstruction_weight=reconstruction_weight,
                            seed=seed, on_progress=on_progress)
    # Fixed context consists exclusively of known train-prefix images.
    context = tensor[train_draws - window + 1:train_draws]
    discovery = tensor[train_draws:discovery_end]
    joined = torch.cat((context, discovery), dim=0)[None]
    bank, names = deep_bank(net, joined, window=window, chunk=max(256, batch_size * 8))
    assert bank.shape[1] == discovery_size
    observed_fft, top_periods = _scan_periods(
        bank, names, recent_windows=tuple(recent_windows),
        half_lives=tuple(half_lives), min_period=min_period,
        max_period=max_period, top_count=top_count)
    observed_motif, motifs = motif_scan(bank, names, tuple(motif_windows), top_count=top_count)
    if on_progress:
        on_progress(f"[DISCOVERY] trained={train_draws} scan={discovery_size} "
                    f"held_out={holdout_draws} deep_channels={len(names)}")

    nulls = {
        "iid_legal_6of49": {"period_maxima": [], "motif_maxima": []},
        "shuffled_historical_discovery": {"period_maxima": [], "motif_maxima": []},
    }
    gens = {name: torch.Generator(device=dev).manual_seed(seed + offset)
            for name, offset in (("iid_legal_6of49", 101),
                                 ("shuffled_historical_discovery", 202))}
    with torch.no_grad():
        for name in nulls:
            for start in range(0, surrogates, surrogate_batch):
                b = min(surrogate_batch, surrogates - start)
                if name == "iid_legal_6of49":
                    alternate = iid_six_hot(b, discovery_size, generator=gens[name], device=dev)
                else:
                    alternate = shuffled_history(discovery, b, generator=gens[name])
                artificial = torch.cat((context[None].expand(b, -1, -1, -1), alternate), dim=1)
                surrogate_bank, _ = deep_bank(net, artificial, window=window, chunk=max(256, batch_size * 8))
                spectrum, _ = _scan_periods(surrogate_bank, names,
                                            recent_windows=tuple(recent_windows),
                                            half_lives=tuple(half_lives),
                                            min_period=min_period,
                                            max_period=max_period, top_count=0)
                motif, _ = motif_scan(surrogate_bank, names, tuple(motif_windows), top_count=0)
                nulls[name]["period_maxima"].extend(spectrum.cpu().tolist())
                nulls[name]["motif_maxima"].extend(motif.cpu().tolist())
                if on_progress and (start == 0 or start + b == surrogates or
                                    (start + b) % max(1, surrogates // 3) < b):
                    on_progress(f"[NULL] {name}: {start+b}/{surrogates}")

    def fwer(value: float, null: list[float]) -> tuple[float, float]:
        p = (1 + sum(t >= value - 1.e-7 for t in null)) / (len(null) + 1)
        # Bonferroni accounts for looking at both motif and spectral detector.
        return float(p), float(min(1., 2. * p))

    def annotate(rows: list[dict], family: str):
        for row in rows:
            for name, record in nulls.items():
                p, corrected = fwer(row["strength"], record[family + "_maxima"])
                row["p_" + name] = p
                row["p_two_detector_adjusted_" + name] = corrected

    annotate(top_periods, "period")
    annotate(motifs, "motif")
    for row in motifs:
        row["start_draw_id"] = int(ids[train_draws + row["start_index"]])
        row["match_draw_id"] = int(ids[train_draws + row["match_index"]])
    best_period = top_periods[0] if top_periods else None
    best_motif = motifs[0] if motifs else None
    best_significance = {}
    for name, record in nulls.items():
        best_significance[name] = {
            "period_global_p": fwer(float(observed_fft[0]), record["period_maxima"])[0],
            "motif_global_p": fwer(float(observed_motif[0]), record["motif_maxima"])[0],
            "period_global_p_two_detector_adjusted": fwer(float(observed_fft[0]), record["period_maxima"])[1],
            "motif_global_p_two_detector_adjusted": fwer(float(observed_motif[0]), record["motif_maxima"])[1],
            "period_null_maxima": record["period_maxima"],
            "motif_null_maxima": record["motif_maxima"],
        }
    # First model evaluation on holdout, not used to tune this run.
    heldout = _score_future(net, tensor, start=discovery_end, window=window,
                            batch_size=batch_size, seed=seed)
    if on_progress:
        on_progress(f"[HELDOUT] n={heldout['evaluated_targets']} "
                    f"Brier={heldout['cnn_gru_brier']:.7f} vs IID={heldout['iid_brier']:.7f} "
                    f"skill={heldout['brier_skill_vs_iid']:+.8f}")
    report = {
        "schema": SCHEMA,
        "source_draws": n,
        "first_draw_id": int(ids[0]), "last_draw_id": int(ids[-1]),
        "source_image_sha256": hashlib.sha256(x.astype(np.uint8).tobytes() + ids.astype("<i8").tobytes()).hexdigest(),
        "device": str(dev), "gpu_name": (torch.cuda.get_device_name(dev) if dev.type == "cuda" else None),
        "splits": {"representation_train": [0, train_draws - 1],
                   "discovery_scan": [train_draws, discovery_end - 1],
                   "heldout_evaluation": [discovery_end, n - 1]},
        "config": {"train_draws": train_draws, "holdout_draws": holdout_draws,
                   "window": window, "validation": validation,
                   "embedding": embedding, "hidden": hidden,
                   "epochs": epochs, "patience": patience, "batch_size": batch_size,
                   "learning_rate": learning_rate, "reconstruction_weight": reconstruction_weight,
                   "recent_windows": list(recent_windows), "half_lives": list(half_lives),
                   "motif_windows": list(motif_windows),
                   "min_period": min_period, "max_period": max_period,
                   "surrogates": surrogates, "surrogate_batch": surrogate_batch,
                   "top_count": top_count, "seed": seed},
        "architecture": "CNN image autoencoder (latent) + causal 4/16/64 rolling features + GRU next-image head",
        "fit": fit, "deep_feature_channels": len(names),
        "feature_names": names,
        "best_period": best_period, "top_periods": top_periods,
        "best_motif": best_motif, "top_motifs": motifs,
        "observed_period_global_max": float(observed_fft[0]),
        "observed_motif_global_max": float(observed_motif[0]),
        "nulls": best_significance,
        "heldout_next_draw": heldout,
        "warnings": [
            "Exploratory deep-feature search, not evidence of a winning-number advantage.",
            "Null controls condition on a frozen encoder trained on a separate historical prefix; surrogates are not retrained.",
            "Search p-values are max-calibrated per detector and Bonferroni adjusted across period/motif detectors.",
            "CNN geometry is a human-defined number-to-7x7 raster, not a physical mechanism.",
            "Prior v0.8 work examined the last 500 historical draws, so heldout is not a prospective blind test.",
            "Training target labels never appear in their input image windows; discovered features use completed known draws only.",
            "IID 6/49 lottery implies all exact six-number tickets have equal next-draw probability."
        ]}
    return (report, net) if return_model else report
