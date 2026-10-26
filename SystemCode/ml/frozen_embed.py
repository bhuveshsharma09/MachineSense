"""Stage R10 (CLAUDE.md §7): a frozen ImageNet ResNet-18 as the familiarity encoder for unfamiliar-fault detection.

No training of any kind: torchvision ResNet-18 (IMAGENET1K_V1, `fc` = identity) in eval mode, no gradients. The 6-channel
cache-C window (already normalised as the matching Stage 3C CNN-6 LOCO run normalised it) is packed with no learned
adapter: the three underhang channels (axial, radial, tangential) as one RGB image, the three overhang channels as another;
each is resized to 224 × 224 and ImageNet-normalised, and the window embedding is the two 512-d outputs concatenated
(1,024-d). The evaluation reuses ml.rmd_eval (arm M = the saved CNN-6 embedding) and adds arm F on these embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ml.models.resnet import IMAGENET_MEAN, IMAGENET_STD, ResNetConfig, verify_weights
from ml.openset import fit_mahalanobis, mahalanobis_distances, mahalanobis_min, recording_mean
from ml.paths import RunPaths, results_dir

EMBED_DIM = 1024
U_CHANNELS = (0, 1, 2)   # under_axial, under_radial, under_tangential (configs/base.yaml order) → R, G, B
O_CHANNELS = (3, 4, 5)   # over_axial, over_radial, over_tangential → R, G, B


def pack_bearings(x6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 6, H, W) → (U, O), each (B, 3, H, W): the underhang and the overhang accelerometer as RGB images, in channel
    order (axial, radial, tangential)."""
    if x6.ndim != 4 or x6.shape[1] != 6:
        raise ValueError(f"expected (B, 6, H, W), got {tuple(x6.shape)}")
    return x6[:, list(U_CHANNELS)], x6[:, list(O_CHANNELS)]


def load_frozen_resnet(cfg: ResNetConfig, pretrained: bool = True) -> nn.Module:
    """torchvision ResNet-18 with `fc` = identity, every parameter frozen, in eval mode (BatchNorm keeps the ImageNet
    running statistics). With `pretrained`, the on-disk IMAGENET1K_V1 weights are sha256-checked first (no download)."""
    from torchvision.models import ResNet18_Weights, resnet18

    if pretrained:
        verify_weights(cfg)
    model = resnet18(weights=getattr(ResNet18_Weights, cfg.weights) if pretrained else None)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def _image(x3: torch.Tensor, size: int) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x3.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x3.device).view(1, 3, 1, 1)
    return (F.interpolate(x3, size=(size, size), mode="bilinear", align_corners=False) - mean) / std


@torch.no_grad()
def frozen_embed(model: nn.Module, x6_batch: torch.Tensor, size: int = 224) -> torch.Tensor:
    """(B, 6, 128, 25) float32, normalised → (B, 1024): concat(f(U), f(O)). The model stays in eval mode."""
    model.eval()
    u, o = pack_bearings(x6_batch.to(torch.float32))
    return torch.cat([model(_image(u, size)), model(_image(o, size))], dim=1)


# ------------------------------------------------------------------------------------------------------------ evaluation
def embedding_path(withheld: str, fold: int):
    """results/scores/r18frozen_w-<c>_loco_fold<k>_embeddings.npz (one per (class, fold): the 3C normalisation and window
    tables do not depend on the seed; CLAUDE.md §7 R10 item 12)."""
    return results_dir() / "scores" / f"r18frozen_w-{withheld}_loco_fold{int(fold)}_embeddings.npz"


def run_scores_f(seed: int, withheld: str, fold: int) -> tuple[pd.DataFrame, dict]:
    """The run's saved recording table with arm M (CNN-6 embedding; ml.rmd_eval.run_scores) and arm F (frozen ResNet-18
    embedding, the same estimator: mahalanobis_min fitted on the FIT windows, recording mean), plus each recording's
    nearest class centroid under M and F (the known-risk diagnostic). Raises if the frozen embeddings are not aligned
    row for row with the saved CNN-6 window table."""
    from ml.rmd_eval import run_scores

    rec, info = run_scores(seed, withheld, fold)
    rp = RunPaths("loco", "cnn6", seed, protocol="loco", withheld=withheld)
    z_m = np.load(rp.embeddings(fold))
    z_f = np.load(embedding_path(withheld, fold))
    win = pd.read_parquet(rp.scores(f"fold{fold}_logits"), columns=["rec_id", "window_idx", "class10", "role"])
    for key, col in (("rec_id", "rec_id"), ("window_idx", "window_idx"), ("role", "role")):
        if not np.array_equal(np.asarray(z_f[key]).astype(str), win[col].to_numpy().astype(str)):
            raise ValueError(f"{seed}/{withheld}/{fold}: frozen embeddings and the CNN-6 window table differ in {key}")
    fit = (z_f["role"] == "fit")
    labels = win.class10.to_numpy()[fit]
    ev_ids = z_f["rec_id"][~fit]
    f_w, f_near, f_info = min_and_nearest(z_f["embedding"][fit], labels, z_f["embedding"][~fit], ev_ids)
    rec["F"] = recording_mean(f_w, ev_ids).loc[rec.rec_id].to_numpy()
    rec["nearest_F"] = f_near.loc[rec.rec_id].to_numpy()
    rec["nearest_M"] = nearest_class(z_m["embedding"][fit], labels, z_m["embedding"][~fit], ev_ids).loc[rec.rec_id].to_numpy()
    return rec, {**info, "shrinkage_f": f_info["shrinkage"]}


def min_and_nearest(fit_emb: np.ndarray, fit_labels, emb: np.ndarray, rec_ids) -> tuple[np.ndarray, pd.Series, dict]:
    """mahalanobis_min's window score and nearest_class's per-recording class from ONE fit and one pass over the class
    distances (the same arithmetic as calling both, bit for bit; the 1,024-d distance pass is the slow step)."""
    params = fit_mahalanobis(fit_emb, fit_labels)
    d2 = np.stack([np.einsum("ij,jk,ik->i", emb - m, params["precision"], emb - m) for m in params["centroids"]], axis=1)
    score = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    d = pd.DataFrame(np.sqrt(np.maximum(d2, 0.0)), columns=params["classes"], index=list(rec_ids))
    return score, d.groupby(level=0, sort=False).mean().idxmin(axis=1), {"classes": params["classes"], "shrinkage": params["shrinkage"]}


def nearest_class(fit_emb: np.ndarray, fit_labels, emb: np.ndarray, rec_ids) -> pd.Series:
    """Per recording: the class whose centroid is nearest, by the mean over its windows of the shared-precision
    Mahalanobis distance to each centroid (descriptive only)."""
    params = fit_mahalanobis(fit_emb, fit_labels)
    d = pd.DataFrame(mahalanobis_distances(params, emb), columns=params["classes"], index=list(rec_ids))
    return d.groupby(level=0, sort=False).mean().idxmin(axis=1)


def nearest_normal_share(per_rec: pd.DataFrame) -> pd.DataFrame:
    """Per seed and withheld class: share of withheld (foldk_withheld) recordings whose nearest centroid is `normal`,
    under M and under F."""
    wh = per_rec[per_rec.role == "foldk_withheld"]
    g = wh.groupby(["seed", "withheld"])
    return pd.DataFrame({"n": g.size(), "share_normal_M": g.nearest_M.apply(lambda s: float((s == "normal").mean())),
                         "share_normal_F": g.nearest_F.apply(lambda s: float((s == "normal").mean()))}).reset_index()
