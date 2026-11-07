"""v0.5 image-only representation ablation on causal next-six-pixel forecasts.

All four arms consume only the 7x7 past images (directly or their deterministic
morphology/spectrum). All share a GRU, balanced-BCE v0.3 loss, chronological
split, seeds and image decoder. Real historical data remain exploratory.
"""
from __future__ import annotations

import hashlib
from html import escape
from math import comb
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from .core import decode, similarity
from .learning import ImageSequenceLearner, image_windows
from .loss_study_v03 import loss_v03
from .predictability_v04 import paired_block_interval, uniform_brier
from .research import DATA, calendar_years, load_history, split
from .image_structure_v05 import (SCHEMA as STRUCTURE_SCHEMA,
                                  check_images, describe_images, _svg)

SCHEMA = "lotto7x7.image_compare.v0.5"
MODELS = ("image_cnn_gru", "morphology_gru", "spectral_gru", "image_fusion_gru")
OBJECTIVE = "balanced_bce"


def exact_iid_expected_jaccard() -> float:
    """Exact IoU of two independent uniformly drawn six-subsets of 49."""
    total = comb(49, 6)
    return sum(comb(6, k) * comb(43, 6-k) / total * k / (12-k)
               for k in range(7))


class ImageAblationModel(nn.Module):
    """Ablate encoders, not underlying number sequences or label definition."""

    def __init__(self, *, kind: str, morphology_dim: int, spectral_dim: int,
                 embedding: int = 24, hidden: int = 32):
        super().__init__()
        if kind not in MODELS or min(morphology_dim, spectral_dim, embedding, hidden) < 2:
            raise ValueError("invalid image model configuration")
        self.kind = kind
        if kind == "image_cnn_gru":
            self.raw = ImageSequenceLearner(embedding=embedding, hidden=hidden)
            return
        dim = (morphology_dim if kind == "morphology_gru" else
               spectral_dim if kind == "spectral_gru" else
               morphology_dim + spectral_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(dim, embedding), nn.GELU(),
            nn.Linear(embedding, embedding), nn.Tanh(),
        )
        if kind == "image_fusion_gru":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
                nn.Conv2d(8, 12, 3, padding=1), nn.GELU(),
                nn.Flatten(), nn.Linear(12 * 7 * 7, embedding), nn.Tanh(),
            )
        self.temporal = nn.GRU(2*embedding if kind == "image_fusion_gru"
                               else embedding, hidden, batch_first=True)
        self.image_head = nn.Linear(hidden, 49)
        nn.init.xavier_uniform_(self.image_head.weight)
        nn.init.constant_(self.image_head.bias, -2.0)

    def forward(self, images: torch.Tensor, morphology: torch.Tensor,
                spectral: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[2:] != (7, 7) or images.shape[1] < 2:
            raise ValueError("expected [batch,time>=2,7,7] image histories")
        if self.kind == "image_cnn_gru":
            return self.raw(images)
        if morphology.shape[:2] != images.shape[:2] or spectral.shape[:2] != images.shape[:2]:
            raise ValueError("feature and image history lengths disagree")
        features = (morphology if self.kind == "morphology_gru" else
                    spectral if self.kind == "spectral_gru" else
                    torch.cat([morphology, spectral], dim=-1))
        frame = self.feature_encoder(features)
        if self.kind == "image_fusion_gru":
            batch, window = images.shape[:2]
            pixels = self.image_encoder(images.reshape(-1, 1, 7, 7))
            frame = torch.cat([pixels.reshape(batch, window, -1), frame], dim=-1)
        out, _ = self.temporal(frame)
        return self.image_head(out[:, -1]).reshape(-1, 7, 7)


def standardized_features(images: np.ndarray, *, train_end: int
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit feature normalizers from TRAIN IMAGES ONLY, never validation/test."""
    masks = check_images(images)
    if not 20 <= train_end < len(masks):
        raise ValueError("invalid feature normalization training segment")
    descriptors = describe_images(masks)
    outputs = []
    record = {}
    for key in ("morphology", "spectral"):
        values = descriptors[key]
        mu = values[:train_end].mean(axis=0)
        sd = np.maximum(values[:train_end].std(axis=0), 1e-4)
        scaled = (values - mu) / sd
        if not np.all(np.isfinite(scaled)):
            raise RuntimeError("invalid normalized image descriptors")
        outputs.append(scaled.astype(np.float32))
        record[key] = {"train_only_mean": mu.tolist(), "train_only_scale": sd.tolist(),
                       "features": int(values.shape[1])}
    return outputs[0], outputs[1], record


def prepare_windows(images: np.ndarray, *, window: int, train_end: int):
    masks = check_images(images)
    xx, yy = image_windows(masks, window=window)
    morph, spec, fit = standardized_features(masks, train_end=train_end)
    xm = torch.tensor(np.stack([morph[t-window:t] for t in range(window,len(masks))]))
    xs = torch.tensor(np.stack([spec[t-window:t] for t in range(window,len(masks))]))
    return xx, xm, xs, yy, fit


def train_image_arm(windows: tuple, *, kind: str, seed: int,
                    train_end: int, validation_end: int,
                    window: int = 5, embedding: int = 24, hidden: int = 32,
                    epochs: int = 45, patience: int = 8, batch_size: int = 64,
                    learning_rate: float = 0.003) -> tuple[ImageAblationModel, dict]:
    x, morphology, spectral, y = windows
    n = len(x) + window
    if (kind not in MODELS or seed < 0 or not 2 <= window < train_end-10
            or not train_end+5 <= validation_end < n or epochs < 1
            or patience < 1 or batch_size < 1 or learning_rate <= 0
            or not np.isfinite(learning_rate) or
            any(len(z) != len(x) for z in (morphology, spectral, y))):
        raise ValueError("invalid image ablation training settings")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ImageAblationModel(kind=kind, morphology_dim=morphology.shape[-1],
                               spectral_dim=spectral.shape[-1],
                               embedding=embedding, hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    train_count, val_count = train_end-window, validation_end-window

    def loss_for_indices(ids: torch.Tensor) -> torch.Tensor:
        logits = model(x[ids], morphology[ids], spectral[ids])
        return loss_v03(logits, y[ids], method=OBJECTIVE)

    def mean_loss(start: int, end: int) -> float:
        model.eval()
        with torch.no_grad():
            losses = [float(loss_for_indices(torch.arange(a,min(a+batch_size,end))).item())
                      for a in range(start,end,batch_size)]
        return float(np.mean(losses))

    state = {key: item.detach().clone() for key, item in model.state_dict().items()}
    best_epoch, best_loss = 0, mean_loss(train_count,val_count)
    history = [{"epoch": 0, "training_loss": mean_loss(0,train_count),
                "validation_loss": best_loss}]
    stale = 0
    for epoch in range(1,epochs+1):
        model.train()
        order = torch.randperm(train_count,
                               generator=torch.Generator().manual_seed(seed+epoch))
        for ids in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for_indices(ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        training_loss = mean_loss(0,train_count)
        val_loss = mean_loss(train_count,val_count)
        if not np.isfinite(training_loss+val_loss):
            raise RuntimeError("non-finite image ablation loss")
        history.append({"epoch":epoch,"training_loss":training_loss,
                        "validation_loss":val_loss})
        if val_loss < best_loss - 1e-8:
            best_epoch,best_loss,stale=epoch,val_loss,0
            state = {key:item.detach().clone() for key,item in model.state_dict().items()}
        else:
            stale+=1
            if stale >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model, {"best_epoch":best_epoch,"best_validation_loss":best_loss,
                   "history":history,"epochs_run":len(history)-1,
                   "parameter_count":sum(p.numel() for p in model.parameters()),
                   "objective":OBJECTIVE,"train_end":train_end,
                   "validation_end":validation_end,"seed":seed}


def infer_image_arm(model: ImageAblationModel, windows: tuple, *,
                    targets: np.ndarray, window: int, batch_size: int = 64
                    ) -> np.ndarray:
    x, morphology, spectral, _ = windows
    targets=np.asarray(targets,dtype=np.int64)
    indexes=targets-window
    if (targets.ndim != 1 or not len(targets) or
            indexes.min() < 0 or indexes.max() >= len(x) or batch_size < 1):
        raise ValueError("invalid image forecast indices")
    chunks=[]
    model.eval()
    with torch.no_grad():
        for group in torch.from_numpy(indexes.copy()).split(batch_size):
            chunks.append(model(x[group],morphology[group],spectral[group]).cpu().numpy())
    logits=np.concatenate(chunks)
    if logits.shape != (len(targets),7,7) or not np.isfinite(logits).all():
        raise RuntimeError("invalid image forecast")
    return logits


def score_image_logits(logits: np.ndarray, actual: np.ndarray) -> tuple[dict,dict]:
    true=check_images(actual)
    if logits.shape != true.shape or not np.all(np.isfinite(logits)):
        raise ValueError("invalid forecast or actual image array")
    prob=torch.sigmoid(torch.as_tensor(logits,dtype=torch.float32)).numpy()
    brier=((prob-true)**2).mean(axis=(1,2)).astype(np.float64)
    positions=np.argsort(-logits.reshape(len(logits),49),axis=1,kind="stable")[:,:6]
    masks=np.zeros((len(logits),49),dtype=np.uint8)
    masks[np.arange(len(logits))[:,None],positions]=1
    masks=masks.reshape(-1,7,7)
    hits=np.sum(masks*true,axis=(1,2)).astype(np.float64)
    iou=hits/(12.0-hits)
    shapes=np.array([similarity(pred,obs)["shape"] for pred,obs in zip(masks,true)],
                    dtype=np.float64)
    unique=len({p.tobytes() for p in masks})
    summary={"mean_pixel_brier":float(brier.mean()),
             "brier_skill_vs_uniform":float(1-brier.mean()/uniform_brier()),
             "mean_hits":float(hits.mean()),"mean_pixel_iou":float(iou.mean()),
             "mean_image_shape_similarity":float(shapes.mean()),
             "unique_images":unique,
             "hit_distribution":{str(k):int(np.count_nonzero(hits==k)) for k in range(7)},
             "mean_soft_mass":float(prob.sum(axis=(1,2)).mean())}
    return summary,{"brier":brier,"hits":hits,"iou":iou,"masks":masks,"prob":prob}


def image_compare(path: Path = DATA, *, seeds: int = 3, seed: int = 649,
                  window: int = 5, epochs: int = 45, patience: int = 8,
                  batch_size: int = 64, learning_rate: float = 0.003,
                  embedding: int = 24, hidden: int = 32,
                  bootstrap: int = 400,
                  on_progress: Callable[[str],None] | None = None) -> dict:
    if (seeds < 1 or seeds > 20 or seed < 0 or window < 2 or
            epochs < 1 or patience < 1 or batch_size < 1 or
            not np.isfinite(learning_rate) or learning_rate <= 0 or
            embedding < 2 or hidden < 2 or bootstrap < 2):
        raise ValueError("invalid image study configuration")
    images, dates, ids = load_history(path)
    images = check_images(images)
    tr,va,te=split(len(images),window)
    train_end,validation_end=int(va[0]),int(te[0])
    built=prepare_windows(images,window=window,train_end=train_end)
    window_data=built[:4]
    fits={name:[] for name in MODELS}
    rows={name:[] for name in MODELS}
    examples=[]
    start=time.perf_counter()
    for loop,random_seed in enumerate(range(seed,seed+seeds)):
        results={}
        for name in MODELS:
            model,fit=train_image_arm(window_data,kind=name,seed=random_seed,
                                      train_end=train_end,
                                      validation_end=validation_end,
                                      window=window,embedding=embedding,hidden=hidden,
                                      epochs=epochs,patience=patience,
                                      batch_size=batch_size,learning_rate=learning_rate)
            logits=infer_image_arm(model,window_data,targets=te,window=window,
                                   batch_size=batch_size)
            summary,scoring=score_image_logits(logits,images[te])
            fits[name].append(fit)
            rows[name].append({"seed":random_seed,"summary":summary})
            results[name]=scoring
        if loop==0:
            for position in range(max(0,len(te)-16),len(te)):
                t=int(te[position]); actual=images[t]
                examples.append({"draw_id":int(ids[t]),
                                 "actual":decode(actual),
                                 "bitmap":actual.tolist(),
                                 "predictions":{
                                     name:{"bitmap":results[name]["masks"][position].tolist(),
                                           "numbers":decode(results[name]["masks"][position]),
                                           "hits":int(results[name]["hits"][position])}
                                     for name in MODELS}})
        for name in MODELS:
            rows[name][-1]["brier_per_target"]=results[name]["brier"].tolist()
            rows[name][-1]["hits_per_target"]=results[name]["hits"].tolist()
        if on_progress is not None:
            on_progress(f"[SEED] {random_seed}: " + ", ".join(
                f"{name} e{fits[name][-1]['best_epoch']}" for name in MODELS))
    summary={}
    for name in MODELS:
        scores=[entry["summary"] for entry in rows[name]]
        summary[name]={
            "mean_pixel_brier":float(np.mean([x["mean_pixel_brier"] for x in scores])),
            "brier_skill_vs_uniform":float(np.mean([x["brier_skill_vs_uniform"] for x in scores])),
            "mean_hits":float(np.mean([x["mean_hits"] for x in scores])),
            "mean_pixel_iou":float(np.mean([x["mean_pixel_iou"] for x in scores])),
            "mean_shape_similarity":float(np.mean([x["mean_image_shape_similarity"] for x in scores])),
            "mean_unique_images":float(np.mean([x["unique_images"] for x in scores])),
            "best_epoch_by_seed":[fit["best_epoch"] for fit in fits[name]],
            "parameter_count":fits[name][0]["parameter_count"],
        }
    uncertainty={}
    raw_brier=np.mean([r["brier_per_target"] for r in rows[MODELS[0]]],axis=0)
    raw_hits=np.mean([r["hits_per_target"] for r in rows[MODELS[0]]],axis=0)
    for name in MODELS[1:]:
        brier=np.mean([r["brier_per_target"] for r in rows[name]],axis=0)
        hits=np.mean([r["hits_per_target"] for r in rows[name]],axis=0)
        uncertainty[name+"_brier_minus_image_cnn"]=paired_block_interval(
            brier-raw_brier,iterations=bootstrap,seed=seed+109)
        uncertainty[name+"_hits_minus_image_cnn"]=paired_block_interval(
            hits-raw_hits,iterations=bootstrap,seed=seed+110)
    null_iou=exact_iid_expected_jaccard()
    return {
        "schema":SCHEMA,"data_sha256":hashlib.sha256(images.tobytes()+ids.tobytes()).hexdigest(),
        "draws":len(images),"evaluation_draws":len(te),
        "train_targets":[int(tr[0]),int(tr[-1])],
        "validation_targets":[int(va[0]),int(va[-1])],
        "test_targets":[int(te[0]),int(te[-1])],
        "configuration":{"seeds":seeds,"seed":seed,"window":window,"epochs":epochs,
                         "patience":patience,"batch_size":batch_size,
                         "learning_rate":learning_rate,"embedding":embedding,
                         "hidden":hidden,"bootstrap":bootstrap,
                         "device":"cpu","objective":OBJECTIVE,
                         "image_layout":"sequential"},
        "chance_baselines":{"pixel_brier_uniform":uniform_brier(),
                            "expected_hits_iid":36.0/49.0,
                            "expected_pixel_iou_iid":null_iou},
        "feature_normalization":built[4],
        "summary":summary,"training":fits,"seed_results":rows,
        "paired_intervals":uncertainty,"examples":examples,
        "elapsed_seconds":float(time.perf_counter()-start),
        "limitations":[
            "Previously inspected historical test draws are exploratory, not new blind data.",
            "Image morphology and spectra are computed only from each past image.",
            "Architectures have different parameter counts and inductive biases, not a matched-budget comparison.",
            "All models use the same balanced-BCE v0.3 image objective and chronological split.",
            "The DCT/FFT and geometric features depend on a manually chosen seven-by-seven mapping.",
            "Top-six accuracy and image similarity cannot alone demonstrate positive expected lottery return.",
        ]
    }


def image_compare_gallery(report:dict) -> str:
    if report.get("schema") != SCHEMA or not report.get("examples"):
        raise ValueError("expected v0.5 image comparison report")
    header='<tr><th>Method</th><th>Pixel Brier</th><th>Mean hits</th><th>Pixel IoU</th></tr>'
    body=''.join(f'<tr><th>{escape(name)}</th><td>{entry["mean_pixel_brier"]:.6f}</td>'
                 f'<td>{entry["mean_hits"]:.4f}</td><td>{entry["mean_pixel_iou"]:.5f}</td></tr>'
                 for name,entry in report["summary"].items())
    cases=[]
    for example in report["examples"]:
        cards=[f'<div><h4>Actual: {escape(str(example["actual"]))}</h4>{_svg(np.array(example["bitmap"]))}</div>']
        for method, item in example["predictions"].items():
            cards.append(f'<div><h4>{escape(method)} · {item["hits"]} hits</h4>'
                         f'{_svg(np.array(item["bitmap"]))}'
                         f'<p>{escape(str(item["numbers"]))}</p></div>')
        cases.append('<section><h3>Draw '+str(example["draw_id"])+
                     '</h3><div class="grid">'+''.join(cards)+'</div></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Lotto7x7 v0.5 Image Learning Comparison</title>'
            '<style>body{font:14px system-ui;max-width:1150px;margin:25px auto;padding:15px;'
            'background:#f3f6fb;color:#192d47}section{background:white;padding:18px;'
            'margin:12px 0;border-radius:12px}.grid{display:grid;'
            'grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}svg{max-width:100%;'
            'width:132px}table{width:100%;border-collapse:collapse}td,th{padding:8px;'
            'border-bottom:1px solid #dce3ed;text-align:left}h4{font-size:12px}'
            '@media(max-width:720px){.grid{grid-template-columns:repeat(2,1fr)}}'
            '</style><h1>Lotto7x7 v0.5 Image Representation Ablation</h1>'
            '<p>All methods forecast a six-pixel image from past images only. '
            'Historical comparisons are exploratory, not proof of predictive skill.</p>'
            '<section><table>'+header+body+'</table></section>'+''.join(cases)+'</html>')
