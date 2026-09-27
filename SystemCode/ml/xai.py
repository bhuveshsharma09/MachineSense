"""Minimal explanations for the app (Stage ML-4, item 5). Labelled "what the model attended to": attribution is not
proof of the fault's location, and a predicted location is according to the dataset labels (CLAUDE.md §3).

CNN: input × gradient of the top-class logit (summed over the recording's windows) gives a channels × order-bin map (the
first convolution mixes channels, so Grad-CAM on the last block cannot separate input channels); Grad-CAM on the last
convolution block adds a coarse order-band profile. LR: the top R1 features by coefficient × transformed value (the
QuantileTransformer output, mean over the recording's windows) for LR's own top class."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ml.paths import results_dir
from ml.spectro import order_axis

NOTE = ("What the model attended to; not proof of the fault's location. A predicted fault location is according to "
        "the dataset labels; per-channel attribution or reconstruction error is not fault-location evidence.")
VIB_NAMES = ["under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential"]


@lru_cache(maxsize=1)
def feature_dictionary() -> pd.DataFrame:
    return pd.read_csv(results_dir() / "feature_dictionary.csv").set_index("name")


def cnn_attribution(model, x_norm, class_idx: int, orders: np.ndarray, channel_names=VIB_NAMES, n_groups: int = 32,
                    top_k: int = 3) -> dict:
    """x_norm: (windows, channels, 128, T) float32 normalised tensor. Returns the coarse heat map (channels × n_groups order
    groups, scaled to max 1), the channels' shares, the top order bands (input × gradient) and the Grad-CAM profile."""
    import torch

    acts = {}
    hook = model.features[-1].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        x = x_norm.detach().clone().requires_grad_(True)
        logits, _ = model(x)
        g_x, g_a = torch.autograd.grad(logits[:, class_idx].sum(), [x, acts["a"]])
    finally:
        hook.remove()
    ixg = (g_x * x).detach().abs().sum(dim=(0, 3)).numpy().astype(np.float64)  # (channels, 128 bins)
    n_bins = ixg.shape[1]
    per = n_bins // n_groups
    heat = ixg.reshape(ixg.shape[0], n_groups, per).sum(axis=2)
    total = heat.sum()
    heat_n = heat / heat.max() if heat.max() > 0 else heat
    edges = [(float(orders[i * per]), float(orders[(i + 1) * per - 1])) for i in range(n_groups)]
    band_mass = heat.sum(axis=0) / (total or 1.0)
    top_bands = [{"orders": list(edges[i]), "share": float(band_mass[i])} for i in np.argsort(-band_mass)[:top_k]]
    ch_share = heat.sum(axis=1) / (total or 1.0)
    a = acts["a"].detach()
    w = g_a.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(dim=1)).mean(dim=(0, 2)).numpy().astype(np.float64)  # (bins of the last block,)
    seg = n_bins // cam.size
    cam_n = cam / cam.max() if cam.max() > 0 else cam
    return {"method": "input × gradient (channels × order bins); Grad-CAM on the last convolution block (order profile)",
            "heat_map": {"channels": list(channel_names), "order_groups": edges,
                         "values": np.round(heat_n, 4).tolist()},
            "top_channels": [{"channel": channel_names[i], "share": float(ch_share[i])} for i in np.argsort(-ch_share)[:top_k]],
            "top_order_bands": top_bands,
            "gradcam_order_profile": {"order_segments": [(float(orders[i * seg]), float(orders[(i + 1) * seg - 1]))
                                                         for i in range(cam.size)],
                                      "values": np.round(cam_n, 4).tolist()}}


def lr_attribution(lr_entry: dict, feats: pd.DataFrame, top_k: int = 5) -> dict:
    """Top R1 features for LR's own top class: coefficient × transformed value, mean over the recording's windows."""
    pipe = lr_entry["model"]
    cols = lr_entry["feature_columns"]
    z = pipe[:-1].transform(feats[cols].to_numpy()).mean(axis=0)
    proba = pipe.predict_proba(feats[cols].to_numpy())
    classes = list(pipe[-1].classes_)
    k = int(np.argmax(np.log(np.clip(proba, lr_entry["prob_floor"], 1)).mean(axis=0)))
    contrib = pipe[-1].coef_[k] * z
    fd = feature_dictionary()
    out = []
    for i in np.argsort(-contrib)[:top_k]:
        name = cols[i]
        row = fd.loc[name] if name in fd.index else None
        out.append({"feature": name, "contribution": float(contrib[i]), "transformed_value": float(z[i]),
                    "definition": None if row is None else str(row["definition"]),
                    "unit": None if row is None else str(row["unit"]),
                    "channel": None if row is None else str(row["channel"])})
    return {"method": "coefficient × QuantileTransformer value (mean over windows), for LR's top class",
            "class": classes[k], "top_features": out}


def explain(bundle, x_norm_input, top_class: str, feats: pd.DataFrame, cfg) -> dict:
    """The xai block of diagnose(): CNN attribution for its top class and LR attribution for LR's top class."""
    import torch

    from ml.train import normalise

    nc = bundle.normalisation["cnn6"]
    xn = normalise(x_norm_input, torch.tensor(nc["mean"], dtype=torch.float32), torch.tensor(nc["std"], dtype=torch.float32))
    cls = bundle.manifest.class10.index(top_class)
    return {"note": NOTE,
            "cnn": {"class": top_class, **cnn_attribution(bundle.cnn, xn, cls, order_axis(cfg.preprocess))},
            "lr": lr_attribution(bundle.lr, feats)}
