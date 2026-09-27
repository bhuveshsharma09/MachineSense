"""Train and calibrate the deployable models of bundle v1 (Stage ML-4). Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/14_train_bundle.py

Data (Bhuvesh's decision, CLAUDE.md §7): TRAIN = closed train ∩ development pool (660 recordings, 20 normals);
CALIBRATION = closed val ∩ development pool (144 recordings, 5 normals). No reserved recording is touched.
- CNN-6 (Stage 3A architecture and fixed settings, seed 42, 10 classes, vibration only) on the train windows; normalisation
  by norm_stats in development mode (fit_ids = train; context: calibration = val).
- AE-6 (frame level, Stage 3B settings) on the train normals; normalisation on those normals.
- Mahalanobis: class centroids and one shared Ledoit–Wolf within-class covariance on the train-window embeddings.
- LR (capped R1, all features, baselines.logreg) on the train windows; B1s (SpeedPoly, vibration-only R1) on the train
  normals.
- Thresholds are NOT computed here (Stage ML-4.2): after training, run scripts/15_bundle_manifest.py, then
  scripts/19_recalibrate.py, which scores the calibration recordings with ml.inference.score_recording — the application's
  CPU scoring path — and sets τ_p, τ_d, τ_e, τ_b with the Stage ML-4 formulas. This script writes a placeholder.
Writes the model files into artifacts/bundle/ (gitignored) and results/bundle/training_report.json.
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from ml.anomaly import SpeedPoly
from ml.cache import load_features
from ml.classical import logreg_pipeline
from ml.config import load_config
from ml.datasets import NormContext, dev_window_rows, load_representation, norm_stats
from ml.features import feature_columns
from ml.labels import CLASS10
from ml.openset import fit_mahalanobis
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, data_dir, results_dir
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = artifacts_dir() / "bundle"
REPORT = results_dir() / "bundle"
VIB = [0, 1, 2, 3, 4, 5]
SEED = 42


def bundle_split(cfg, meta):
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    return pool, pool[pool.split_closed == "train"], pool[pool.split_closed == "val"]


def main() -> int:
    import torch
    from safetensors.numpy import save_file as save_np
    from safetensors.torch import save_file

    from ml.train import gpu_job, score_windows, train_model
    from ml.train_ae import train_ae
    from ml.utils import get_device, seed_everything

    cfg = load_config(DEFAULT_CONFIG)
    device = get_device()
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool, tr, va = bundle_split(cfg, meta)
    assert tr.split_severity.ne("test").all() and tr.split_speed_block.ne("test").all()
    assert va.split_severity.ne("test").all() and va.split_speed_block.ne("test").all()
    tr_ids, va_ids = sorted(tr.rec_id), sorted(va.rec_id)
    tr_norm = sorted(tr.loc[tr.class10 == "normal", "rec_id"])
    va_norm = sorted(va.loc[va.class10 == "normal", "rec_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cc = cfg.cnn.model_copy(update={"seed": SEED})
    ac = cfg.ae.model_copy(update={"seed": SEED})
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool.assign(dev_fold=0))
    ctx = NormContext(calibration=frozenset(va_ids))
    st = norm_stats(arr, index, meta, fit_ids=tr_ids, context=ctx)
    st_ae = norm_stats(arr, index, meta, fit_ids=tr_norm, context=ctx)
    mean, std = st["mean"][VIB], st["std"][VIB]
    mean_a, std_a = st_ae["mean"][VIB], st_ae["std"][VIB]
    tr_rows = np.flatnonzero(rows.rec_id.isin(tr_ids).to_numpy())
    trn_rows = np.flatnonzero(rows.rec_id.isin(tr_norm).to_numpy())
    with gpu_job(device, artifacts_dir(), cc.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)
        y = torch.from_numpy(rows.class10.map({c: i for i, c in enumerate(CLASS10)}).to_numpy(np.int64)).to(device)
        seed_everything(SEED)
        t0 = time.perf_counter()
        cnn, hist = train_model(cache, y, tr_rows, VIB, mean, std, len(CLASS10), cc, device)
        t_cnn = time.perf_counter() - t0
        _, emb_tr = score_windows(cnn, cache, tr_rows, VIB, mean, std, cc, device)
        seed_everything(SEED)
        t0 = time.perf_counter()
        ae, hist_ae = train_ae(cache, trn_rows, VIB, mean_a, std_a, "frame", ac, device)
        t_ae = time.perf_counter() - t0
        save_file({k: v.detach().cpu().contiguous() for k, v in cnn.state_dict().items()}, str(OUT / "cnn6.safetensors"))
        save_file({k: v.detach().cpu().contiguous() for k, v in ae.state_dict().items()}, str(OUT / "ae6.safetensors"))
    # ---- Mahalanobis parameters (train-window embeddings)
    maha = fit_mahalanobis(emb_tr.astype(np.float64), rows.class10.to_numpy()[tr_rows])
    save_np({"centroids": maha["centroids"], "precision": maha["precision"]}, str(OUT / "mahalanobis.safetensors"))
    # ---- the classifier thresholds are set by scripts/19_recalibrate.py on the deployment scoring path
    # ---- LR and B1s on the R1 features
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = feature_columns(df)
    vib_cols = [c for c in cols if not c.startswith("f_mic_")]
    dtr = df[df.rec_id.isin(tr_ids)]
    lr = logreg_pipeline(cfg.baselines).fit(dtr[cols].to_numpy(), dtr.class10.to_numpy())
    dn = df[df.rec_id.isin(tr_norm)]
    b1s = SpeedPoly(cfg.anomaly.b1s_degree).fit(dn[vib_cols].to_numpy(), dn.speed_hz.to_numpy())
    joblib.dump({"model": lr, "feature_columns": cols, "classes": list(lr.classes_), "prob_floor": cfg.baselines.prob_floor,
                 "doc": "scikit-learn Pipeline(QuantileTransformer → LogisticRegression) on capped R1 window features "
                        "(all features); fitted on bundle-train windows; recording decision = mean log-probability"},
                OUT / "lr_r1.joblib")
    joblib.dump({"model": b1s, "feature_columns": vib_cols, "degree": cfg.anomaly.b1s_degree,
                 "doc": "ml.anomaly.SpeedPoly (B1s): Prep (log10 + StandardScaler) and a per-feature polynomial in "
                        "log(window speed) fitted on the bundle-train normal windows (vibration-only R1 features); "
                        "window score = mean |residual| / residual sd; recording score = mean over windows"},
                OUT / "b1s_vib.joblib")
    thresholds = {"status": "pending: run scripts/15_bundle_manifest.py, then scripts/19_recalibrate.py (Stage ML-4.2)",
                  "tau_p": None, "tau_d": None, "tau_e": None, "tau_b": None}
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    norm = {"cnn6": {"channels": VIB, "mean": mean.tolist(), "std": std.tolist(), "n_windows": st["n_windows"]},
            "ae6": {"channels": VIB, "mean": mean_a.tolist(), "std": std_a.tolist(), "n_windows": st_ae["n_windows"]},
            "mahalanobis": {"classes": maha["classes"], "shrinkage": maha["shrinkage"]}}
    (OUT / "normalisation.json").write_text(json.dumps(norm, indent=1))
    report = {"git_sha": git_sha(), "split_hash": split_hash(meta), "seed": SEED, "device": str(device),
              "data": {"train_recordings": len(tr_ids), "train_normals": len(tr_norm), "calibration_recordings": len(va_ids),
                       "calibration_normals": len(va_norm),
                       "train_speed_range_hz": [float(tr.fr_measured_hz.min()), float(tr.fr_measured_hz.max())],
                       "calibration_speed_range_hz": [float(va.fr_measured_hz.min()), float(va.fr_measured_hz.max())],
                       "severities": {c: sorted(set(g.severity.round(2))) for c, g in tr.groupby("class10")},
                       "train_windows": int(len(tr_rows)), "train_normal_windows": int(len(trn_rows))},
              "thresholds": thresholds, "training_seconds": {"cnn6": t_cnn, "ae6": t_ae},
              "history": {"cnn6_final_loss": hist[-1]["train_loss"], "ae6_final_loss": hist_ae[-1]["train_loss"]}}
    (REPORT / "training_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("data", "training_seconds")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
