"""Frozen final models for the final evaluation (CLAUDE.md §12): the 9 CNN-6 LOCO models (with shared AE-6 and B1s) and
R18-ft, stored in artifacts/final/ with a manifest (sha256 per file, git sha, split hash), verified before loading.

Scoring uses the ML-4.2 deployment path on the CPU, from the raw recording and its metadata speed: the R1 features and
the cache-C representation are computed online (`recording_inputs`), then each model scores them. `score_cnn` reproduces
`ml.inference.score_recording` exactly when given the bundle's models (tested), so the LOCO models are calibrated and
tested on the same path as the bundle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.bundle import BundleIntegrityError, sha256_file
from ml.paths import artifacts_dir

SCHEMA = "final-1.0"
MANIFEST = "manifest.json"
VIB = [0, 1, 2, 3, 4, 5]


def final_dir() -> Path:
    return artifacts_dir() / "final"


# ----------------------------------------------------------------------------------------------------------- scoring
def recording_inputs(x: np.ndarray, fr: float, cfg) -> dict:
    """Online R1 features (9 windows) and cache-C representation (9, 7, 128, 25) of one raw recording — computed once and
    shared by every model that scores it."""
    from ml.features import recording_features
    from ml.spectro import stacked_representation

    pre = cfg.preprocess
    return {"feats": recording_features(x, fr, pre),
            "rep": stacked_representation(x, fr, pre, cfg.representation.leading_candidate)}


def _cnn_forward(model, rep: np.ndarray, channels, mean, std) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from ml.train import normalise
    xr = torch.from_numpy(rep[:, channels].astype(np.float16)).to(torch.float32)
    with torch.no_grad():
        logits, emb = model(normalise(xr, torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)))
    return logits.numpy(), emb.numpy().astype(np.float64)


def per_class_mahalanobis(params: dict, emb: np.ndarray) -> np.ndarray:
    """Per window: distance to every class under its own (Ledoit–Wolf) precision (n × classes)."""
    d2 = np.stack([np.einsum("ij,jk,ik->i", emb - m, p, emb - m)
                   for m, p in zip(params["locations"], params["precisions"])], axis=1)
    return np.sqrt(np.maximum(d2, 0.0))


def score_cnn(inputs: dict, cnn, cnn_norm: dict, classes: list[str], maha: dict, ae, ae_norm: dict, ae_cfg: dict,
              b1s: dict, floor: float, maha_pc: dict | None = None) -> dict:
    """The score_recording computation for any class set: CNN recording probabilities (mean log-probability) and MSP;
    Mahalanobis (shared precision; window minimum over classes, mean over windows) and optionally per-class; AE error
    (per channel, equal-weight mean, mean over windows); B1s (mean over windows)."""
    import torch

    from ml.config import AeConfig
    from ml.openset import mahalanobis_distances, recording_probs
    from ml.train_ae import window_channel_errors

    rep, feats = inputs["rep"], inputs["feats"]
    logits, emb = _cnn_forward(cnn, rep, cnn_norm["channels"], cnn_norm["mean"], cnn_norm["std"])
    probs = recording_probs(logits, ["r"] * len(logits), classes, floor).iloc[0]
    ae_ch = window_channel_errors(ae, torch.from_numpy(rep), np.arange(len(rep)), ae_norm["channels"],
                                  np.asarray(ae_norm["mean"]), np.asarray(ae_norm["std"]), "frame", AeConfig(**ae_cfg),
                                  torch.device("cpu"))
    out = {"probs": probs, "top": str(probs.idxmax()), "msp": float(probs.max()),
           "maha": float(mahalanobis_distances(maha, emb).min(axis=1).mean()),
           "ae": float(ae_ch.mean(axis=1).mean()),
           "b1s": float(b1s["model"].score(feats[b1s["feature_columns"]].to_numpy(), feats.speed_hz.to_numpy()).mean())}
    if maha_pc is not None:
        out["maha_pc"] = float(per_class_mahalanobis(maha_pc, emb).min(axis=1).mean())
    return out


def score_r18(inputs: dict, model, norm: dict, classes: list[str], floor: float) -> dict:
    from ml.openset import recording_probs
    logits, _ = _cnn_forward(model, inputs["rep"], norm["channels"], norm["mean"], norm["std"])
    probs = recording_probs(logits, ["r"] * len(logits), classes, floor).iloc[0]
    return {"probs": probs, "top": str(probs.idxmax()), "msp": float(probs.max())}


# ----------------------------------------------------------------------------------------------------------- manifest
def list_files(directory: Path) -> list[str]:
    d = Path(directory)
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != MANIFEST
                  and not p.name.startswith("."))


def write_manifest(directory: Path, info: dict) -> Path:
    d = Path(directory)
    files = [{"file": f, "sha256": sha256_file(d / f)} for f in list_files(d)]
    p = d / MANIFEST
    p.write_text(json.dumps({"schema": SCHEMA, **info, "files": files}, indent=1, ensure_ascii=False, default=float))
    return p


def verify_final(directory: Path | None = None) -> dict:
    """Refuse a missing, extra or altered file (like ml.bundle.verify_bundle)."""
    d = Path(directory or final_dir())
    if not (d / MANIFEST).exists():
        raise BundleIntegrityError(f"no {MANIFEST} in {d}")
    m = json.loads((d / MANIFEST).read_text())
    if m.get("schema") != SCHEMA:
        raise BundleIntegrityError(f"schema {m.get('schema')!r}, expected {SCHEMA!r}")
    listed = {e["file"] for e in m["files"]}
    present = set(list_files(d))
    if present - listed:
        raise BundleIntegrityError(f"files not in the manifest: {sorted(present - listed)}")
    for e in m["files"]:
        f = d / e["file"]
        if not f.exists():
            raise BundleIntegrityError(f"missing file {e['file']}")
        if sha256_file(f) != e["sha256"]:
            raise BundleIntegrityError(f"sha256 mismatch for {e['file']}; refusing to load")
    return m


@dataclass
class LocoModel:
    withheld: str
    classes: list[str]
    cnn: Any
    cnn_norm: dict
    maha: dict
    maha_pc: dict
    thresholds: dict


@dataclass
class FinalModels:
    directory: Path
    manifest: dict
    manifest_sha256: str
    loco: dict[str, LocoModel]
    ae: Any
    ae_norm: dict
    ae_cfg: dict
    b1s: dict
    r18: Any
    r18_norm: dict


def load_final(directory: Path | None = None) -> FinalModels:
    """Verify every hash, then load (CPU)."""
    import joblib
    from safetensors.numpy import load_file as load_np
    from safetensors.torch import load_file

    from ml.config import AeConfig, CnnConfig
    from ml.models.ae import build_ae
    from ml.models.cnn import ScratchCNN
    from ml.models.resnet import ResNet18Spectrogram, ResNetConfig

    d = Path(directory or final_dir())
    m = verify_final(d)
    mc = json.loads((d / "model_config.json").read_text())
    ae_norm = json.loads((d / "shared" / "normalisation.json").read_text())
    ae = build_ae("frame", len(ae_norm["channels"]), AeConfig(**mc["ae"]))
    ae.load_state_dict(load_file(str(d / "shared" / "ae6.safetensors"), device="cpu"))
    loco = {}
    for c in m["loco_classes"]:
        sub = d / "loco" / c
        norm = json.loads((sub / "normalisation.json").read_text())
        cnn = ScratchCNN(len(norm["channels"]), len(norm["classes"]), CnnConfig(**mc["cnn"]))
        cnn.load_state_dict(load_file(str(sub / "cnn6.safetensors"), device="cpu"))
        mh, pc = load_np(str(sub / "mahalanobis.safetensors")), load_np(str(sub / "mahalanobis_pc.safetensors"))
        loco[c] = LocoModel(c, norm["classes"], cnn.eval(), norm,
                            {"classes": norm["classes"], "centroids": mh["centroids"], "precision": mh["precision"]},
                            {"classes": norm["classes"], "locations": pc["locations"], "precisions": pc["precisions"]},
                            json.loads((sub / "thresholds.json").read_text()))
    r18_norm = json.loads((d / "r18ft" / "normalisation.json").read_text())
    r18 = ResNet18Spectrogram(len(r18_norm["channels"]), len(r18_norm["classes"]), ResNetConfig(**mc["resnet"]), "r18ft",
                              pretrained=False)
    r18.load_state_dict(load_file(str(d / "r18ft" / "r18ft.safetensors"), device="cpu"))
    return FinalModels(d, m, sha256_file(d / MANIFEST), loco, ae.eval(), ae_norm, mc["ae"],
                       joblib.load(d / "shared" / "b1s_vib.joblib"), r18.eval(), r18_norm)


def calibration_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("rec_id").sort_index()
