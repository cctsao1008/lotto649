"""Opt-in CPU/CUDA training for v0.3 image-loss experiments.

Do not change the historical v0.2 learning or diagnostic modules. The default
CPU study continues to use those exact functions. This module reproduces the
same data windows, architecture, optimizer, seed, training and early-stopping
protocol on an explicitly selected device. CUDA results are a new experimental
execution, not bit-for-bit copies of the frozen CPU baseline.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .learning import ImageSequenceLearner, image_windows, shape_loss

DEVICES = ("cpu", "cuda", "auto")


def select_device(requested: str) -> torch.device:
    """Resolve device explicitly; never silently fall back from requested CUDA."""
    if requested not in DEVICES:
        raise ValueError("device must be cpu, cuda, or auto")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python environment; "
            "check torch.version.cuda and torch.cuda.is_available()"
        )
    return torch.device("cuda" if (requested == "cuda" or
                                   requested == "auto" and available) else "cpu")


def train_device_model(images: np.ndarray, *, method: str, device: torch.device,
                       train_end: int, validation_end: int,
                       window: int = 5, embedding: int = 24, hidden: int = 32,
                       epochs: int = 45, patience: int = 8,
                       batch_size: int = 64, learning_rate: float = 0.003,
                       seed: int = 649) -> tuple[ImageSequenceLearner, dict]:
    """Match v0.2/v0.3 training semantics while placing all tensors on device."""
    # Local import avoids circular dependency: v0.3 study owns its loss variants.
    from .loss_study_v03 import METHODS, loss_v03

    if method not in METHODS or not isinstance(device, torch.device):
        raise ValueError("invalid loss method or torch device")
    if (not 2 <= window < train_end - 10
            or not train_end + 5 <= validation_end < len(images)
            or epochs < 1 or patience < 1 or batch_size < 1 or seed < 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid device training hyperparameters")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    x, y = image_windows(images, window=window)
    x, y = x.to(device), y.to(device)
    tr_end, va_end = train_end - window, validation_end - window
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ImageSequenceLearner(embedding=embedding, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(tr_end, generator=generator)
        for ids in order.split(batch_size):
            batch_ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch_ids])
            loss = (shape_loss(logits, y[batch_ids]) if method == "original_v02"
                    else loss_v03(logits, y[batch_ids], method=method))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for a, b in zip(x[tr_end:va_end].split(batch_size),
                            y[tr_end:va_end].split(batch_size)):
                pred = model(a)
                item = (shape_loss(pred, b) if method == "original_v02"
                        else loss_v03(pred, b, method=method))
                val_losses.append(float(item.item()))
        val_loss = float(np.mean(val_losses))
        if not np.isfinite(val_loss):
            raise RuntimeError("non-finite validation loss")
        history.append({"epoch": epoch, "validation_image_shape_loss": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {name: tensor.detach().clone()
                          for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "architecture": "frozen_v02_cnn_encoder_and_gru",
        "objective": method,
        "best_epoch": best_epoch,
        "best_validation_image_shape_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "training_targets": [window, train_end - 1],
        "validation_targets": [train_end, validation_end - 1],
        "window": window,
        "device": str(device),
        "note": "same_protocol_retrained_on_selected_device_not_bitwise_v02_checkpoint",
    }


def infer_device_logits(model: ImageSequenceLearner, images: np.ndarray,
                        *, indices: np.ndarray, window: int, batch_size: int,
                        device: torch.device) -> np.ndarray:
    """Infer from past context frames only; return numpy logits for shared scoring."""
    ids = np.asarray(indices, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or ids.min() < window
            or ids.max() >= len(images) or batch_size < 1):
        raise ValueError("invalid inference indices or batch size")
    # An unindexed 'cuda' is an alias for the current CUDA device, while
    # .to('cuda') places parameters on a concrete device such as 'cuda:0'.
    # Compare physical devices, not the two unequal torch.device spellings.
    model_device = next(model.parameters()).device
    expected_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda" and device.index is None
        and model_device.type == "cuda"
        else device
    )
    if model_device != expected_device:
        raise ValueError("model and inference device disagree")
    windows = np.stack([images[int(t) - window:int(t)] for t in ids])
    input_tensor = torch.as_tensor(windows, dtype=torch.float32)
    pieces = []
    model.eval()
    with torch.no_grad():
        for group in input_tensor.split(batch_size):
            pieces.append(model(group.to(device)).detach().cpu().numpy())
    result = np.concatenate(pieces)
    if result.shape != (len(ids), 7, 7) or not np.isfinite(result).all():
        raise RuntimeError("non-finite or invalid device predictions")
    return result
