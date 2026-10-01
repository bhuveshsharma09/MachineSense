"""Stage R7: forward passes of SAVED models only (no training) — the reproduction check and the closed-set context for D1.
Protocol: CLAUDE.md §7 (Stage R7), declared before computing anything. Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/37_r7_forward.py

1. Reproduction: the 81 LOCO runs (9 withheld classes × 3 outer folds × seeds 42 / 43 / 44) are run forward on every saved
   window; window logits / embeddings and recording MSP / shared Mahalanobis / top class are compared with the saved Stage
   3C / 3C.1 outputs (tolerance: |Δ MSP| ≤ 1e-5, |Δ maha| ≤ 1e-4 × maha, identical top class), and the seed-42 τ_p, τ_d are
   recomputed from the saved calibration scores and compared with results/loco/decision_thresholds.csv (1e-12).
   → results/research/r7_reproduction.json. On any failure the script exits non-zero (the stage stops).
2. D1 context: the closed-set development CNN-6 (seeds 42 / 43 / 44, artifacts/cnn/cnn6*_dev/fold<k>.pt) on the
   horizontal-misalignment 0.5 mm recordings of its own fold: top class, top probability, and the shared Mahalanobis fitted on
   that fold model's own training windows (forward pass only). → results/research/r7_closed_set_hm.csv.
The notebook (notebooks/06_research_layer/17_diagnostics.ipynb) uses the saved, now verified outputs.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.datasets import dev_window_rows, load_representation
from ml.evaluation import recording_predictions
from ml.labels import CLASS10
from ml.openset import mahalanobis_min, recording_mean, recording_msp
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, data_dir, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEEDS = (42, 43, 44)
VIB = [0, 1, 2, 3, 4, 5]
TOL_MSP, TOL_MAHA_REL, TOL_TAU = 1e-5, 1e-4, 1e-12
OUT = results_dir() / "research"
SC = results_dir() / "scores"


def loco_name(c: str, seed: int) -> str:
    return f"cnn6_w-{c}_s{seed}_loco"


def dev_name(seed: int) -> str:
    return "cnn6_dev" if seed == 42 else f"cnn6_s{seed}_dev"


def main() -> int:
    import torch

    from ml.models.cnn import ScratchCNN
    from ml.train import gpu_job, score_windows
    from ml.utils import get_device

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    pos_of = {(r, int(w)): i for i, (r, w) in enumerate(zip(rows.rec_id, rows.window_idx))}
    device = get_device()
    floor = cfg.baselines.prob_floor
    rep = {"seeds": list(SEEDS), "tolerance": {"msp": TOL_MSP, "maha_rel": TOL_MAHA_REL, "tau": TOL_TAU},
           "device": str(device), "git_sha": git_sha(), "split_hash": split_hash(meta), "runs": [],
           "note": "a first pass cast the embeddings to float64 before the Ledoit-Wolf fit and differed from the saved "
                   "Mahalanobis by up to 2.7e-4 (relative); Stage 3C fitted on the float32 embeddings, and with that path "
                   "the saved values reproduce; window logits and embeddings were bit-identical in both passes"}
    ok_all = True
    with gpu_job(device, artifacts_dir(), cfg.cnn.mps_memory_fraction), torch.no_grad():
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)
        del x
        # ---------------------------------------------------------------- 1. LOCO reproduction
        for seed in SEEDS:
            for c in FAULT_CLASSES:
                known = [k for k in CLASS10 if k != c]
                for k in (0, 1, 2):
                    name = loco_name(c, seed)
                    L = pd.read_parquet(SC / f"{name}_fold{k}_logits.parquet")
                    E = np.load(SC / f"{name}_fold{k}_embeddings.npz")
                    R = pd.read_parquet(SC / f"{name}_fold{k}_recording.parquet").set_index("rec_id")
                    norm = json.loads((results_dir() / "loco" / name / f"fold{k}_norm.json").read_text())
                    model = ScratchCNN(len(VIB), len(known), cfg.cnn)
                    model.load_state_dict(torch.load(artifacts_dir() / "loco" / name / f"fold{k}.pt", map_location="cpu",
                                                     weights_only=True))
                    model.to(device).eval()
                    idx = np.array([pos_of[(r, int(w))] for r, w in zip(L.rec_id, L.window_idx)])
                    logits, emb = score_windows(model, cache, idx, VIB, np.asarray(norm["mean"]), np.asarray(norm["std"]),
                                                cfg.cnn, device)
                    d_logit = float(np.abs(logits - L[[f"logit_{q}" for q in known]].to_numpy()).max())
                    assert (E["rec_id"] == L.rec_id.to_numpy()).all() and (E["window_idx"] == L.window_idx.to_numpy()).all()
                    d_emb = float(np.abs(emb - E["embedding"]).max())
                    fit = (L.role == "fit").to_numpy()
                    ev = ~fit
                    rec = recording_msp(logits[ev], L.rec_id.to_numpy()[ev], known, floor)
                    maha_w, _ = mahalanobis_min(emb[fit], L.class10.to_numpy()[fit], emb[ev])  # float32, as Stage 3C computed it
                    rec["maha"] = recording_mean(maha_w, L.rec_id.to_numpy()[ev]).loc[rec.index]
                    s = R.loc[rec.index]
                    d_msp = float(np.abs(rec.msp - s.msp).max())
                    d_maha_rel = float((np.abs(rec.maha - s.maha) / s.maha).max())
                    same_top = bool((rec.pred == s.pred).all())
                    ok = d_msp <= TOL_MSP and d_maha_rel <= TOL_MAHA_REL and same_top
                    ok_all &= ok
                    rep["runs"].append({"seed": seed, "withheld": c, "fold": k, "windows": int(len(L)), "max_abs_logit": d_logit,
                                        "max_abs_embedding": d_emb, "max_abs_msp": d_msp, "max_rel_maha": d_maha_rel,
                                        "same_top_class": same_top, "reproduced": ok})
                    print(f"  s{seed} {c}/{k}: logit {d_logit:.1e} emb {d_emb:.1e} msp {d_msp:.1e} maha {d_maha_rel:.1e} "
                          f"top {same_top} {'ok' if ok else 'FAIL'}", flush=True)
        # seed-42 thresholds and the saved decisions file
        thr = pd.read_csv(results_dir() / "loco" / "decision_thresholds.csv").set_index(["withheld", "outer_fold"])
        dec = pd.read_parquet(SC / "cnn6ae6_s42_loco_decisions.parquet").set_index(["withheld", "outer_fold", "rec_id"])
        tau_d_diff, tau_p_diff, dec_diff = 0.0, 0.0, 0.0
        for c in FAULT_CLASSES:
            for k in (0, 1, 2):
                R = pd.read_parquet(SC / f"{loco_name(c, 42)}_fold{k}_recording.parquet")
                cal = R[R.role == "cal"]
                tp = -order_statistic_threshold(-cal.msp.to_numpy(), 0.975)["threshold"]
                td = order_statistic_threshold(cal.maha.to_numpy(), 0.975)["threshold"]
                tau_p_diff = max(tau_p_diff, abs(tp - thr.loc[(c, k), "tau_p"]))
                tau_d_diff = max(tau_d_diff, abs(td - thr.loc[(c, k), "tau_d"]))
                d = dec.loc[(c, k)].loc[R.rec_id]
                dec_diff = max(dec_diff, float(np.abs(d.msp.to_numpy() - R.msp.to_numpy()).max()),
                               float(np.abs(d.maha.to_numpy() - R.maha.to_numpy()).max()))
        tau_ok = tau_p_diff <= TOL_TAU and tau_d_diff <= TOL_TAU and dec_diff <= TOL_TAU
        ok_all &= tau_ok
        rep["seed42_thresholds"] = {"max_abs_tau_p": tau_p_diff, "max_abs_tau_d": tau_d_diff,
                                    "decisions_file_vs_run_files_max_abs": dec_diff, "reproduced": tau_ok}
        print(f"  seed-42 thresholds: τ_p {tau_p_diff:.1e}, τ_d {tau_d_diff:.1e}, decisions file {dec_diff:.1e}", flush=True)
        rep["all_reproduced"] = bool(ok_all)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "r7_reproduction.json").write_text(json.dumps(rep, indent=1))
        if not ok_all:
            print("REPRODUCTION FAILED: stopping (Stage R7 item 1)")
            return 3
        # ---------------------------------------------------------------- 2. closed-set CNN-6 on mild HM
        hm = pool[(pool.class10 == "horizontal_misalignment") & np.isclose(pool.severity, 0.5)]
        out = []
        for seed in SEEDS:
            tj = json.loads((results_dir() / "cnn" / f"{dev_name(seed)}_train.json").read_text())
            saved = pd.read_parquet(SC / ("cnn6_dev.parquet" if seed == 42 else f"cnn6_s{seed}_dev_logits.parquet"))
            for f in tj["folds"]:
                k = f["fold"]
                if not (hm.dev_fold == k).any():  # folds are folder-grouped: the 0.5 mm folder sits in one fold
                    continue
                model = ScratchCNN(len(VIB), len(CLASS10), cfg.cnn)
                model.load_state_dict(torch.load(artifacts_dir() / "cnn" / dev_name(seed) / f"fold{k}.pt", map_location="cpu",
                                                 weights_only=True))
                model.to(device).eval()
                mean, std = np.asarray(f["norm_mean"]), np.asarray(f["norm_std"])
                tr = np.flatnonzero((rows.dev_fold != k).to_numpy())
                hmr = np.flatnonzero(rows.rec_id.isin(hm[hm.dev_fold == k].rec_id).to_numpy())
                _, e_tr = score_windows(model, cache, tr, VIB, mean, std, cfg.cnn, device)
                lg, e_hm = score_windows(model, cache, hmr, VIB, mean, std, cfg.cnn, device)
                w = rows.iloc[hmr]
                from scipy.special import softmax
                rp = recording_predictions(softmax(lg.astype(np.float64), axis=1), w, floor)
                sv = saved.set_index(["rec_id", "window_idx"]).loc[list(zip(w.rec_id, w.window_idx))]
                rp_saved = recording_predictions(softmax(sv[[f"logit_{q}" for q in CLASS10]].to_numpy(np.float64), axis=1),
                                                 w, floor)
                mw, _ = mahalanobis_min(e_tr, rows.class10.to_numpy()[tr], e_hm)  # float32, the Stage 3C convention
                maha = recording_mean(mw, w.rec_id.to_numpy())
                probs = pd.DataFrame(np.log(np.clip(softmax(lg.astype(np.float64), axis=1), floor, 1)), columns=CLASS10)
                probs = np.exp(probs.groupby(w.rec_id.to_numpy(), sort=False).mean())
                probs = probs.div(probs.sum(axis=1), axis=0)
                for r in rp.index:
                    out.append({"seed": seed, "fold": k, "rec_id": r, "top_class": rp.at[r, "pred10"],
                                "top_probability": float(probs.loc[r].max()), "p_vertical_misalignment": float(probs.at[r, "vertical_misalignment"]),
                                "maha_closed_set": float(maha.loc[r]), "top_matches_saved": bool(rp.at[r, "pred10"] == rp_saved.at[r, "pred10"])})
        pd.DataFrame(out).to_csv(OUT / "r7_closed_set_hm.csv", index=False)
        print(f"closed-set CNN-6: {len(out)} recording × seed rows; top class matches saved: "
              f"{all(o['top_matches_saved'] for o in out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
