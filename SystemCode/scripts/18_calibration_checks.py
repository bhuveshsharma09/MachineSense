"""Stage ML-4.1 items 3 and 8 on the 144 calibration recordings (closed val ∩ development pool) — no evaluation of the full
closed-validation split (recordings outside the pool are reserved under other splits).

    MACHINESENSE_DEVICE=mps python scripts/18_calibration_checks.py   # threshold-setting path on the training device

Item 8 (sanity report, no new accuracy claim): class10 / class6 macro-F1 and confusion matrices of diagnose() (CPU).
Item 3 (calibration consistency): the threshold-setting path (bundle weights on the representation cache, on the device
used when thresholds were set; B1s from the feature cache) vs the inference path (diagnose() on CPU): per-recording score
differences, joint acceptance on both paths vs the saved rate, and the 5 calibration normals under every rule variant.
Findings are recorded, not corrected: any fix needs Bhuvesh's approval (CLAUDE.md §7).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from ml.arbitrator import VARIANTS, decide
from ml.inference import diagnose, get_bundle
from ml.labels import CLASS6, CLASS10
from ml.paths import SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import development_pool

OUT = results_dir() / "bundle"
LABEL = ("not an independent test — these recordings were used for calibration (not for epoch selection: the epochs were "
         "fixed in advance); the bundle's training and calibration sets together use the full development pool; the "
         "only independent evaluation left is the reserved tests")


def threshold_setting_scores(b, cfg, meta, ids) -> pd.DataFrame:
    import torch

    from ml.cache import load_features
    from ml.config import AeConfig
    from ml.datasets import dev_window_rows, load_representation
    from ml.openset import mahalanobis_distances, recording_mean, recording_msp
    from ml.train import score_windows
    from ml.train_ae import window_channel_errors
    from ml.utils import get_device

    dev = get_device()
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, meta[meta.rec_id.isin(ids)].assign(folder="-", dev_fold=0))
    pos = rows.row.to_numpy()
    o = np.argsort(pos)
    x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
    x[o] = arr[pos[o]]
    cache = torch.from_numpy(x).to(dev)
    nc, na = b.normalisation["cnn6"], b.normalisation["ae6"]
    lg, emb = score_windows(b.cnn.to(dev), cache, np.arange(len(rows)), nc["channels"], np.array(nc["mean"]),
                            np.array(nc["std"]), cfg.cnn, dev)
    t = recording_msp(lg, rows.rec_id, CLASS10, cfg.baselines.prob_floor)
    t["maha"] = recording_mean(mahalanobis_distances(b.mahalanobis, emb.astype(np.float64)).min(axis=1), rows.rec_id).loc[t.index]
    err = window_channel_errors(b.ae.to(dev), cache, np.arange(len(rows)), na["channels"], np.array(na["mean"]),
                                np.array(na["std"]), "frame", AeConfig(**b.model_config["ae"]), dev).mean(axis=1)
    t["ae"] = recording_mean(err, rows.rec_id).loc[t.index]
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess)
    df = df[df.rec_id.isin(ids)].sort_values(["rec_id", "window_idx"])
    bm = b.b1s
    t["b1s"] = pd.Series(bm["model"].score(df[bm["feature_columns"]].to_numpy(), df.speed_hz.to_numpy()),
                         index=df.rec_id.to_numpy()).groupby(level=0).mean().loc[t.index]
    b.cnn.to("cpu"), b.ae.to("cpu")
    return t.loc[ids].assign(device=str(dev))


def main() -> int:
    b = get_bundle()
    cfg = b.config
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
    va = pool[pool.split_closed == "val"].set_index("rec_id").sort_index()
    ids = list(va.index)
    rows = []
    for r in ids:
        d = diagnose(r, with_xai=False)["diagnosis"]
        s = d["scores"]
        rows.append({"rec_id": r, "true10": va.loc[r, "class10"], "pred10": d["top_class"],
                     "pred6": max(d["class6_probabilities"], key=d["class6_probabilities"].get),
                     "true6": d["class6_mapping"][va.loc[r, "class10"]], "msp": s["msp"], "maha": s["maha"],
                     "ae": s["ae_error"], "b1s": s["b1s"]})
    inf = pd.DataFrame(rows).set_index("rec_id")
    T = {k: float(b.thresholds[k]) for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
    # ---- item 8
    present6 = [c for c in CLASS6 if c in set(inf.true6)]
    sanity = {"label": LABEL, "n_recordings": len(inf), "split": "closed val ∩ development pool", "path": "diagnose() on CPU",
              "class10_macro_f1": float(f1_score(inf.true10, inf.pred10, labels=CLASS10, average="macro")),
              "class6_macro_f1": float(f1_score(inf.true6, inf.pred6, labels=present6, average="macro")),
              "class10_confusion": {"labels": list(CLASS10), "matrix": confusion_matrix(inf.true10, inf.pred10, labels=CLASS10).tolist()},
              "class6_confusion": {"labels": list(CLASS6), "matrix": confusion_matrix(inf.true6, inf.pred6, labels=list(CLASS6)).tolist()},
              "used_for_epoch_selection": False}
    (OUT / "sanity_report.json").write_text(json.dumps(sanity, indent=1, ensure_ascii=False))
    # ---- item 3
    ts = threshold_setting_scores(b, cfg, meta, ids)
    cmp_ = pd.DataFrame({f"{k}_{p}": v[k] for k in ("msp", "maha", "ae", "b1s") for p, v in (("threshold_setting", ts), ("inference", inf))})
    joint = {p: ((v.msp >= T["tau_p"]) & (v.maha <= T["tau_d"])) for p, v in (("threshold_setting", ts), ("inference", inf))}
    rep = json.loads((OUT / "training_report.json").read_text())
    saved_rate = rep["calibration_sanity_not_evaluation"]["joint_acceptance"]
    normals = []
    for r in va.index[va.class10 == "normal"]:
        x = inf.loc[r]
        normals.append({"rec_id": r, "measured_speed_hz": float(va.loc[r, "fr_measured_hz"]), "pred10": x.pred10,
                        "msp": x.msp, "msp_ge_tau_p": bool(x.msp >= T["tau_p"]), "maha": x.maha, "maha_le_tau_d": bool(x.maha <= T["tau_d"]),
                        "ae": x.ae, "ae_threshold_setting": float(ts.loc[r, "ae"]), "ae_le_tau_e": bool(x.ae <= T["tau_e"]),
                        "b1s": x.b1s, "b1s_threshold_setting": float(ts.loc[r, "b1s"]), "b1s_le_tau_b": bool(x.b1s <= T["tau_b"]),
                        "verdicts": {v: (lambda o: (o["fault_class"] or o["decision"]) + (f" (caution: {o['caution']})" if o["caution"] else ""))(
                            decide(v, x.pred10, x.msp, x.maha, x.ae, x.b1s, T)) for v in VARIANTS}})
    diffs = {k: float((ts[k] - inf[k]).abs().max()) for k in ("msp", "maha", "ae", "b1s")}
    defining = {"tau_p": list(ts.index[ts.msp == T["tau_p"]]), "tau_d": list(ts.index[ts.maha == T["tau_d"]]),
                "tau_e": list(ts.index[(va.loc[ts.index, "class10"] == "normal") & (ts.ae == ts.ae[va.loc[ts.index, "class10"] == "normal"].max())]),
                "tau_b": list(ts.index[(va.loc[ts.index, "class10"] == "normal") & (ts.b1s == ts.b1s[va.loc[ts.index, "class10"] == "normal"].max())])}
    consistency = {
        "status": "FOUND ISSUE — pending Bhuvesh's decision; no threshold, rule or model changed",
        "thresholds": T, "threshold_setting_device": ts.device.iloc[0],
        "saved_joint_acceptance_rate": saved_rate, "saved_joint_acceptance_count": int(round(saved_rate * len(ids))),
        "joint_acceptance": {p: {"count": int(j.sum()), "rate": float(j.mean())} for p, j in joint.items()},
        "recordings_whose_check_differs": list(joint["inference"].index[joint["inference"] != joint["threshold_setting"]]),
        "threshold_defining_recordings": defining,
        "max_abs_score_difference_threshold_setting_vs_inference": diffs,
        "tau_e_recomputed_on_threshold_setting_path": float(ts.ae[va.loc[ts.index, "class10"] == "normal"].max()),
        "normals": normals,
        "cause": ("each threshold equals one calibration recording's own score (order statistic for τ_p, τ_d; maximum "
                  "for τ_e, τ_b), so that recording sits exactly on the boundary; the inference path (CPU, recording by "
                  "recording, online features) reproduces the threshold-setting scores only to float precision (MSP within "
                  "~1e-8, Mahalanobis within ~2e-5, AE within ~2e-7, B1s within ~2e-15), and a difference in the wrong "
                  "direction flips exactly those boundary recordings")}
    cmp_.to_csv(OUT / "calibration_consistency_scores.csv")
    (OUT / "calibration_consistency.json").write_text(json.dumps(consistency, indent=1, default=float))
    print(json.dumps({k: sanity[k] for k in ("class10_macro_f1", "class6_macro_f1")}))
    print(json.dumps({k: consistency[k] for k in ("joint_acceptance", "recordings_whose_check_differs", "threshold_defining_recordings",
                                                  "max_abs_score_difference_threshold_setting_vs_inference",
                                                  "tau_e_recomputed_on_threshold_setting_path")}, indent=1, default=float))
    print(pd.DataFrame(normals).drop(columns=["verdicts"]).to_string())
    for n in normals:
        print(n["rec_id"], n["verdicts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
