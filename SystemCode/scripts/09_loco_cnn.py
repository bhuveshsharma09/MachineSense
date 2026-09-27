"""Deep unknown-fault test (Q3, Stage 3C): LOCO-retrained CNN-6 plus a per-run healthy-only AE-6. Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/09_loco_cnn.py train      # all 9 × 3 runs; resumable
    python scripts/09_loco_cnn.py auroc                                                # threshold-free AUROC / OSCR
    python scripts/09_loco_cnn.py decide                                               # three-way rule (Stage 3C item 2)

Protocol (CLAUDE.md §7, declared before running): for each withheld class c and outer dev fold k, the SAVED Stage 2D.1
Q3 assignment gives fit / calibration / seen-folder evaluation (4:1:1 within folder) on the known classes of folds ≠ k.
- CNN-6 (Stage 3A architecture and fixed settings, seed 42, 9 outputs) trained on the fit recordings only; norm_stats in
  development mode (fit_ids = fit; context: calibration, seen-folder evaluation, fold k, every class-c recording).
- AE-6 (frame level, Stage 3B settings) trained on the fit NORMALS of the same run only (the saved assignment splits the
  normal folder differently per c, so one AE per (c, k)); norm_stats in development mode on those fit normals.
- Recording scores: MSP (lower = more unknown), Mahalanobis (shared within-class Ledoit–Wolf covariance; min over known
  classes per window; mean over windows), k-NN (cosine distance to the 10th nearest fit window; mean over windows), AE
  error (higher = more anomalous). Stage 3A/3B (Q2-only) outputs are never used.
Outputs (ml.paths.RunPaths, family "loco", protocol "loco", withheld class in every name):
results/scores/cnn6_w-<c>_s42_loco_fold<k>_logits.parquet and _fold<k>_embeddings.npz (windows of every role),
results/scores/cnn6_w-<c>_s42_loco_fold<k>_recording.parquet (recording scores), artifacts/loco/<name>/fold<k>.pt for
cnn6 and ae6, results/loco/<name>/fold<k>_norm.json and fold<k>_train.json, results/loco/*.json|csv, MLflow runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import Config, load_config
from ml.datasets import dev_window_rows, load_representation, norm_stats
from ml.labels import CLASS10
from ml.openset import knn_kth_cosine, loco_run_ids, mahalanobis_min, oscr, recording_mean, recording_msp
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, data_dir, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEED = 42
VIB = [0, 1, 2, 3, 4, 5]
CH_NAMES = ["under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential"]
KNN_K = 10
OUT = results_dir() / "loco"
ASSIGNMENT = results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv"
SCORES = {"msp": "lower MSP = more unknown", "maha": "higher = more unknown", "knn": "higher = more unknown",
          "ae": "higher = more anomalous (departure from healthy, not unfamiliarity among faults)"}
ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")


def paths(c: str, seed: int = SEED) -> tuple[RunPaths, RunPaths]:
    return (RunPaths("loco", "cnn6", seed, protocol="loco", withheld=c),
            RunPaths("loco", "ae6", seed, protocol="loco", withheld=c))


def load_pool(cfg: Config, meta: pd.DataFrame) -> pd.DataFrame:
    dc = cfg.development
    folds = pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]]
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(folds, on="rec_id")
    assert len(pool) == len(folds)
    return pool


def run_train(cfg: Config, only_c: str | None, only_k: int | None, seed: int = SEED) -> None:
    import mlflow
    import torch

    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.train_ae import train_ae, window_channel_errors
    from ml.utils import get_device, mlflow_run, seed_everything

    device = get_device()
    cc = cfg.cnn.model_copy(update={"seed": seed})
    ac = cfg.ae.model_copy(update={"seed": seed})
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = load_pool(cfg, meta)
    assignment = pd.read_csv(ASSIGNMENT)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    cache_key = json.loads(next((data_dir() / "representations").glob(
        f"{cfg.representation.leading_candidate}_*.json")).read_text())["cache_key"]
    sh = split_hash(meta)
    todo = [(c, k) for c in FAULT_CLASSES for k in (0, 1, 2)
            if (only_c in (None, c)) and (only_k in (None, k)) and not paths(c, seed)[0].scores(f"fold{k}_recording").exists()]
    print(f"{len(todo)} LOCO runs to train")
    if not todo:
        return
    with gpu_job(device, artifacts_dir(), cc.mps_memory_fraction), \
            mlflow_run(cfg, run_name=f"loco_cnn6ae6_s{seed}") as parent:
        stage = "3C" if seed == SEED else "3C.1"
        mlflow.set_tags({"stage": stage, "question": "Q3", "family": "loco", "seed": str(seed), "split_hash": sh,
                         "cache_key": cache_key})
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # dev-pool windows only
        del x
        for c, k in todo:
            rp, rpa = paths(c, seed)
            for p in (rp.weights(k), rpa.weights(k), rp.norm_stats(k), rpa.norm_stats(k), rp.scores("x")):
                p.parent.mkdir(parents=True, exist_ok=True)
            ids = loco_run_ids(assignment, pool, c, k)
            known = [x for x in CLASS10 if x != c]
            y = torch.from_numpy(rows.class10.map({x: i for i, x in enumerate(known)}).fillna(0).to_numpy(np.int64)).to(device)
            st = norm_stats(arr, index, meta, fit_ids=ids["fit"], context=ids["context"])
            mean, std = st["mean"][VIB], st["std"][VIB]
            fit_normals = sorted(set(ids["fit"]) & set(pool.loc[pool.class10 == "normal", "rec_id"]))
            sta = norm_stats(arr, index, meta, fit_ids=fit_normals, context=ids["context"])
            mean_a, std_a = sta["mean"][VIB], sta["std"][VIB]
            for r_, s_, f_ in ((rp, st, ids["fit"]), (rpa, sta, fit_normals)):
                r_.norm_stats(k).write_text(json.dumps({"fit_ids": f_, "n_windows": s_["n_windows"], "channels": CH_NAMES,
                                                        "mean": s_["mean"][VIB].tolist(), "std": s_["std"][VIB].tolist()}))
            role_of = {r: role for role in ("fit", *ROLES) for r in ids[role]}
            sel = rows.rec_id.map(role_of)
            fit_rows = np.flatnonzero((sel == "fit").to_numpy())
            score_rows = np.flatnonzero(sel.notna().to_numpy())
            run_id = rp.run_name(k)
            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": stage, "question": "Q3", **rp.tags(k), "ae_run": rpa.run_name(k),
                                 "split_hash": sh, "cache_key": cache_key, "git_sha": git_sha()})
                seed_everything(seed)
                t0 = time.perf_counter()
                model, hist = train_model(cache, y, fit_rows, VIB, mean, std, len(known), cc, device,
                                          on_epoch=lambda ep, r: mlflow.log_metrics(
                                              {"train_loss": r["train_loss"], "train_acc": r["train_acc"]}, step=ep))
                t_cnn = time.perf_counter() - t0
                logits, emb = score_windows(model, cache, score_rows, VIB, mean, std, cc, device)
                torch.save(model.state_dict(), rp.weights(k))
                seed_everything(seed)
                t0 = time.perf_counter()
                fitn_rows = np.flatnonzero(rows.rec_id.isin(fit_normals).to_numpy())
                ae, hist_ae = train_ae(cache, fitn_rows, VIB, mean_a, std_a, "frame", ac, device,
                                       on_epoch=lambda ep, r: mlflow.log_metric("ae_train_loss", r["train_loss"], step=ep))
                t_ae = time.perf_counter() - t0
                eval_rows = np.flatnonzero(sel.isin(ROLES).to_numpy())
                ae_err = window_channel_errors(ae, cache, eval_rows, VIB, mean_a, std_a, "frame", ac, device).mean(axis=1)
                torch.save(ae.state_dict(), rpa.weights(k))
                # ---- window table (logits, embeddings) for every scored window
                w = rows.iloc[score_rows]
                wt = pd.DataFrame({"rec_id": w.rec_id.to_numpy(), "window_idx": w.window_idx.to_numpy(),
                                   "dev_fold": w.dev_fold.to_numpy(), "class10": w.class10.to_numpy(),
                                   "role": sel.iloc[score_rows].to_numpy(), "withheld": c, "outer_fold": k,
                                   "variant": "cnn6", "seed": seed, "run_id": run_id})
                for j, name in enumerate(known):
                    wt[f"logit_{name}"] = logits[:, j]
                wt.to_parquet(rp.scores(f"fold{k}_logits"), index=False)
                np.savez_compressed(rp.embeddings(k), embedding=emb, rec_id=wt.rec_id.to_numpy().astype(str),
                                    window_idx=wt.window_idx.to_numpy(), role=wt.role.to_numpy().astype(str),
                                    withheld=np.array(c), outer_fold=np.array(k), variant=np.array("cnn6"),
                                    seed=np.array(seed), run_id=np.array(run_id))
                # ---- recording scores
                is_fit = (wt.role == "fit").to_numpy()
                ev = ~is_fit
                maha_w, maha_info = mahalanobis_min(emb[is_fit], wt.class10[is_fit].to_numpy(), emb[ev])
                knn_w = knn_kth_cosine(emb[is_fit], emb[ev], KNN_K)
                ev_ids = wt.rec_id[ev].to_numpy()
                rec = recording_msp(logits[ev], ev_ids, known, cfg.baselines.prob_floor)
                rec["maha"] = recording_mean(maha_w, ev_ids).loc[rec.index]
                rec["knn"] = recording_mean(knn_w, ev_ids).loc[rec.index]
                ae_ids = rows.rec_id.to_numpy()[eval_rows]
                rec["ae"] = recording_mean(ae_err, ae_ids).loc[rec.index]
                info = pool.set_index("rec_id").loc[rec.index]
                rec = rec.assign(role=[role_of[r] for r in rec.index], class10=info.class10.to_numpy(),
                                 folder=info.folder.to_numpy(), severity=info.severity.to_numpy(),
                                 dev_fold=info.dev_fold.to_numpy(), withheld=c, outer_fold=k, seed=seed,
                                 run_id=run_id, ae_run_id=rpa.run_name(k)).rename_axis("rec_id").reset_index()
                rec.to_parquet(rp.scores(f"fold{k}_recording"), index=False)
                kn = rec[rec.role == "foldk_known"]
                un = rec[rec.role == "foldk_withheld"]
                auc = roc_auc_score(np.r_[np.zeros(len(kn)), np.ones(len(un))], np.r_[-kn.msp, -un.msp])
                mlflow.log_metrics({"train_seconds_cnn": t_cnn, "train_seconds_ae": t_ae, "auroc_msp": auc,
                                    "n_fit": len(ids["fit"]), "n_fit_normals": len(fit_normals)})
                rp.fold_result(k, "train").write_text(json.dumps(
                    {"withheld": c, "fold": k, "known_classes": known, "n": {r: len(ids[r]) for r in ("fit", *ROLES)},
                     "n_fit_normals": len(fit_normals), "n_fit_windows": int(len(fit_rows)),
                     "train_seconds_cnn": t_cnn, "train_seconds_ae": t_ae, "mahalanobis": maha_info,
                     "history_cnn": hist, "history_ae": hist_ae, "split_hash": sh, "cache_key": cache_key,
                     "git_sha": git_sha()}, indent=1))
                print(f"  {run_id}: fit {len(ids['fit'])} rec ({len(fit_rows)} windows), CNN {t_cnn:.0f} s, AE {t_ae:.0f} s "
                      f"({len(fit_normals)} fit normals); MSP AUROC {auc:.3f}", flush=True)
            release_device_memory(device)


def load_recordings(seed: int = SEED) -> pd.DataFrame:
    parts = [pd.read_parquet(paths(c, seed)[0].scores(f"fold{k}_recording")) for c in FAULT_CLASSES for k in (0, 1, 2)]
    return pd.concat(parts, ignore_index=True)


def unknown_score(df: pd.DataFrame, s: str) -> np.ndarray:
    return (-df.msp if s == "msp" else df[s]).to_numpy()


def run_auroc(cfg: Config) -> None:
    """Threshold-free per class c: AUROC and OSCR, (fold-k known) vs (fold-k class c), per score; fold mean and spread;
    macro over the 9 classes. OSCR uses the CNN's closed-set correctness on the known recordings for every score."""
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    rec = load_recordings()
    rows = []
    for (c, k), g in rec.groupby(["withheld", "outer_fold"]):
        kn, un = g[g.role == "foldk_known"], g[g.role == "foldk_withheld"]
        correct = (kn.pred == kn.class10).to_numpy()
        for s in SCORES:
            u_kn, u_un = unknown_score(kn, s), unknown_score(un, s)
            rows.append({"withheld": c, "outer_fold": int(k), "score": s,
                         "auroc": float(roc_auc_score(np.r_[np.zeros(len(u_kn)), np.ones(len(u_un))], np.r_[u_kn, u_un])),
                         "oscr": oscr(u_kn, correct, u_un), "n_known": len(kn), "n_withheld": len(un),
                         "closed_set_accuracy_known": float(correct.mean())})
    pf = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT / "loco_per_fold_auroc_oscr.csv", index=False)
    per_class = pf.groupby(["score", "withheld"]).agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                                                     oscr_mean=("oscr", "mean"), oscr_std=("oscr", "std")).reset_index()
    per_class.to_csv(OUT / "loco_per_class_auroc_oscr.csv", index=False)
    summary = {s: {"auroc_macro": float(g.auroc_mean.mean()), "auroc_macro_std_over_classes": float(g.auroc_mean.std()),
                   "auroc_all_folds_std": float(pf[pf.score == s].auroc.std()),
                   "auroc_all_folds_range": [float(pf[pf.score == s].auroc.min()), float(pf[pf.score == s].auroc.max())],
                   "oscr_macro": float(g.oscr_mean.mean()), "oscr_macro_std_over_classes": float(g.oscr_mean.std())}
               for s, g in per_class.groupby("score")}
    tr = [json.loads(paths(c)[0].fold_result(k, "train").read_text()) for c in FAULT_CLASSES for k in (0, 1, 2)]
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "seed": SEED, "directions": SCORES,
                          "knn": f"cosine distance to the {KNN_K}th nearest fit window; recording = mean over windows",
                          "mahalanobis": "class centroids from fit windows; one shared Ledoit–Wolf covariance of "
                                         "within-class residuals; min over known classes per window; mean over windows",
                          "assignment": str(ASSIGNMENT.relative_to(SYSTEM_CODE_DIR)),
                          "ae_interpretation": "AE error measures departure from healthy behaviour, not unfamiliarity "
                                               "among fault types; a familiar severe fault can have a larger error than "
                                               "an unfamiliar mild one"},
           "training": {"runs": len(tr), "cnn_seconds_total": float(sum(t["train_seconds_cnn"] for t in tr)),
                        "ae_seconds_total": float(sum(t["train_seconds_ae"] for t in tr)),
                        "fit_recordings_range": [min(t["n"]["fit"] for t in tr), max(t["n"]["fit"] for t in tr)],
                        "fit_normals_range": [min(t["n_fit_normals"] for t in tr), max(t["n_fit_normals"] for t in tr)]},
           "summary": summary}
    (OUT / "loco_auroc_oscr.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["training"], indent=1))
    print(per_class.pivot(index="withheld", columns="score", values="auroc_mean").round(3).to_string())
    for s, v in summary.items():
        print(f"  {s:5s} AUROC macro {v['auroc_macro']:.3f} ± {v['auroc_macro_std_over_classes']:.3f} (folds "
              f"{v['auroc_all_folds_range'][0]:.2f}–{v['auroc_all_folds_range'][1]:.2f}); OSCR macro {v['oscr_macro']:.3f}")


ACCEPT = 0.975  # calibration target for τ_p and τ_d, each (declared before running)
ORDERINGS = ("classifier_first", "ae_first")  # classifier-first is PRIMARY
GROUPS = {"withheld": "foldk_withheld", "seen_eval": "eval", "unseen_foldk": "foldk_known"}


def thresholds(g: pd.DataFrame, acceptance: float = ACCEPT) -> dict:
    """τ_p, τ_d from the calibration known recordings (order statistic, ties accepted); τ_e = maximum calibration-normal
    AE error."""
    cal = g[g.role == "cal"]
    tp = order_statistic_threshold(-cal.msp.to_numpy(), acceptance)  # accept iff −MSP ≤ thr ⇔ MSP ≥ −thr
    td = order_statistic_threshold(cal.maha.to_numpy(), acceptance)
    return {"tau_p": -tp["threshold"], "tau_d": td["threshold"], "tau_e": float(cal[cal.class10 == "normal"].ae.max()),
            "n_cal": tp["n"], "r": tp["r"], "attainable_level": tp["attainable_level"],
            "n_cal_normals": int((cal.class10 == "normal").sum())}


def lr_recordings(cfg: Config, meta: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """The classical LR/MSP of Stage 2D.1 re-run per recording on the SAVED assignment (same features, settings and
    95 % order-statistic threshold); checks its per-fold AUROC / OSCR / unknown recall against q3_per_fold.csv."""
    from ml.cache import load_features
    from ml.classical import logreg_pipeline
    from ml.features import feature_columns
    from ml.openset import unknown_scores

    bc = cfg.baselines
    path = max((data_dir() / "features").glob(f"{bc.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = feature_columns(df)
    assignment = pd.read_csv(ASSIGNMENT)
    ref = pd.read_csv(results_dir() / "q1q3" / "q3_per_fold.csv")
    ref = ref[ref.score == "msp"].set_index(["withheld", "outer_fold"])
    out, mismatches = [], []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            ids = loco_run_ids(assignment, pool, c, k)
            fit = df[df.rec_id.isin(ids["fit"])]
            model = logreg_pipeline(bc).fit(fit[cols].to_numpy(), fit.class10.to_numpy())
            parts = []
            for role in ROLES:
                d = df[df.rec_id.isin(ids[role])]
                sc = unknown_scores(model, d[cols].to_numpy(), d.rec_id.to_numpy(), bc.prob_floor)
                parts.append(sc.assign(role=role))
            r = pd.concat(parts).rename_axis("rec_id").reset_index()
            r["msp"] = -r.msp
            thr = order_statistic_threshold(-r[r.role == "cal"].msp.to_numpy(), cfg.openset.acceptance)["threshold"]
            r = r.assign(withheld=c, outer_fold=k, tau=-thr, class10=pool.set_index("rec_id").class10.loc[r.rec_id].to_numpy())
            kn, un = r[r.role == "foldk_known"], r[r.role == "foldk_withheld"]
            auc = float(roc_auc_score(np.r_[np.zeros(len(kn)), np.ones(len(un))], np.r_[-kn.msp, -un.msp]))
            osc = oscr(-kn.msp.to_numpy(), (kn.pred == kn.class10).to_numpy(), -un.msp.to_numpy())
            rec_ = float(np.mean(-un.msp.to_numpy() > thr))
            e = ref.loc[(c, k)]
            for name, a, b in (("auroc", auc, e.auroc), ("oscr", osc, e.oscr), ("unknown_recall", rec_, e.unknown_recall)):
                if abs(a - b) > 1e-12:  # the saved CSV round-trips floats to ~16 significant digits
                    mismatches.append(f"{c}/{k} {name}: {a!r} vs {b!r}")
            out.append(r)
    return pd.concat(out, ignore_index=True), mismatches


def rates(df: pd.DataFrame, col: str, cats: list[str]) -> dict:
    n = len(df)
    return {**{c: float((df[col] == c).mean()) if n else float("nan") for c in cats}, "n": int(n)}


def outcome_tables(dec: pd.DataFrame, col: str) -> dict:
    """Outcome rates for one decision column: withheld; known faults (a) seen-folder and (b) unseen fold k, per class and
    macro over known fault classes; normal recordings (fold k and seen-folder). Per withheld class, overall, and fold
    spread (std over the 27 runs of the per-run rates)."""
    W, KF, NR = ["healthy", "known_fault", "unknown"], ["healthy", "correct", "wrong_fault", "unknown"], ["healthy", "fault", "unknown"]
    out = {"withheld": {}, "known_faults": {}, "normal": {}}
    wd = dec[dec.role == "foldk_withheld"]
    out["withheld"]["per_class"] = {c: rates(g, col, W) for c, g in wd.groupby("withheld")}
    out["withheld"]["overall_recordings"] = rates(wd, col, W)
    per_run = pd.DataFrame([{**rates(g, col, W), "withheld": c, "fold": k} for (c, k), g in wd.groupby(["withheld", "outer_fold"])])
    out["withheld"]["macro_over_classes"] = {x: float(pd.DataFrame(out["withheld"]["per_class"]).T[x].mean()) for x in W}
    out["withheld"]["fold_spread_std"] = {x: float(per_run[x].std()) for x in W}
    out["withheld"]["faults_called_healthy_count"] = int((wd[col] == "healthy").sum())
    for grp, role in (("a_seen_folder", "eval"), ("b_unseen_foldk", "foldk_known")):
        kf = dec[(dec.role == role) & (dec.class10 != "normal")]
        per_class = {c: rates(g, col, KF) for c, g in kf.groupby("class10")}
        runs = []
        for (c, k), g in kf.groupby(["withheld", "outer_fold"]):
            m = pd.DataFrame({cl: rates(h, col, KF) for cl, h in g.groupby("class10")}).T
            runs.append({x: float(m[x].mean()) for x in KF})
        out["known_faults"][grp] = {"per_class": per_class,
                                   "macro_over_known_fault_classes": {x: float(pd.DataFrame(per_class).T[x].mean()) for x in KF},
                                   "fold_spread_std_of_run_macro": {x: float(pd.DataFrame(runs)[x].std()) for x in KF},
                                   "n": int(len(kf))}
        nr = dec[(dec.role == role) & (dec.class10 == "normal")]
        out["normal"][grp.replace("a_seen_folder", "seen_folder_eval").replace("b_unseen_foldk", "foldk")] = rates(nr, col, NR)
    return out


def run_decide(cfg: Config) -> None:
    """Three-way decision rule (C4) with thresholds from calibration recordings only; joint acceptance; outcome tables
    for both orderings (classifier-first primary); AE override counts; the classical LR/MSP comparison."""
    from ml.openset import outcome_category, three_way

    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = load_pool(cfg, meta)
    rec = load_recordings()
    thr_rows, parts = [], []
    for (c, k), g in rec.groupby(["withheld", "outer_fold"]):
        t = thresholds(g)
        thr_rows.append({"withheld": c, "outer_fold": int(k), **t})
        g = g.copy()
        g["msp_ok"] = g.msp >= t["tau_p"]
        g["maha_ok"] = g.maha <= t["tau_d"]
        g["confident"] = g.msp_ok & g.maha_ok
        for o in ORDERINGS:
            d, ov = three_way(g.pred, g.msp, g.maha, g.ae, t["tau_p"], t["tau_d"], t["tau_e"], o)
            g[f"decision_{o}"] = d
            if o == "classifier_first":
                g["override"] = ov
            wmask = (g.role == "foldk_withheld").to_numpy()
            oc = np.empty(len(g), dtype=object)
            oc[wmask] = outcome_category(d[wmask], g.class10.to_numpy()[wmask], True)
            oc[~wmask] = outcome_category(d[~wmask], g.class10.to_numpy()[~wmask], False)
            g[f"outcome_{o}"] = oc
        parts.append(g)
    dec = pd.concat(parts, ignore_index=True)
    thr = pd.DataFrame(thr_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    thr.to_csv(OUT / "decision_thresholds.csv", index=False)
    rp_all = RunPaths("loco", "cnn6ae6", SEED, protocol="loco")
    dec.to_parquet(rp_all.scores("decisions"), index=False)
    # ---- joint acceptance, measured
    acc_rows = []
    for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
        for grp, role in (("calibration", "cal"), ("seen_folder_eval", "eval"), ("unseen_foldk", "foldk_known")):
            for sub, m in (("all_known", g.role == role), ("known_faults", (g.role == role) & (g.class10 != "normal")),
                           ("normal", (g.role == role) & (g.class10 == "normal"))):
                h = g[m]
                acc_rows.append({"withheld": c, "outer_fold": int(k), "group": grp, "subset": sub, "n": len(h),
                                 "msp_acceptance": float(h.msp_ok.mean()), "maha_acceptance": float(h.maha_ok.mean()),
                                 "joint_acceptance": float(h.confident.mean())})
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT / "joint_acceptance_per_run.csv", index=False)
    acc_sum = acc.groupby(["group", "subset"]).agg(joint_mean=("joint_acceptance", "mean"), joint_std=("joint_acceptance", "std"),
                                                   msp_mean=("msp_acceptance", "mean"), maha_mean=("maha_acceptance", "mean"),
                                                   n_total=("n", "sum")).reset_index()
    pooled = dec.assign(group=dec.role.map({"cal": "calibration", "eval": "seen_folder_eval", "foldk_known": "unseen_foldk"}))
    pooled = pooled[pooled.group.notna()].groupby("group").confident.mean()
    # ---- override counts (classifier-first)
    kinds = {"truly_healthy": dec.class10 == "normal", "known_fault": (dec.class10 != "normal") & (dec.role != "foldk_withheld"),
             "withheld_fault": dec.role == "foldk_withheld"}
    ov_rows = []
    for role in ("foldk_known", "foldk_withheld", "eval"):
        for kind, m in kinds.items():
            for k in (0, 1, 2):
                h = dec[m & (dec.role == role) & (dec.outer_fold == k)]
                if len(h):
                    ov_rows.append({"role": role, "kind": kind, "outer_fold": k, "overrides": int(h.override.sum()),
                                    "n": len(h), "pct": float(100 * h.override.mean())})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(OUT / "ae_override_counts.csv", index=False)
    outcomes = {o: outcome_tables(dec, f"outcome_{o}") for o in ORDERINGS}
    # ---- classical comparison
    lr, mism = lr_recordings(cfg, meta, pool)
    lr["decision"] = np.where(lr.msp >= lr.tau, np.where(lr.pred == "normal", "healthy", lr.pred), "unknown")
    wmask = (lr.role == "foldk_withheld").to_numpy()
    oc = np.empty(len(lr), dtype=object)
    oc[wmask] = outcome_category(lr.decision.to_numpy()[wmask], lr.class10.to_numpy()[wmask], True)
    oc[~wmask] = outcome_category(lr.decision.to_numpy()[~wmask], lr.class10.to_numpy()[~wmask], False)
    lr["outcome"] = oc
    lr.to_parquet(RunPaths("loco", "logreg", 42, protocol="loco").scores("decisions"), index=False)
    out_lr = outcome_tables(lr, "outcome")
    # CNN MSP alone at the classical 95 % target (supplementary like-for-like)
    sup = []
    for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
        tp = -order_statistic_threshold(-g[g.role == "cal"].msp.to_numpy(), cfg.openset.acceptance)["threshold"]
        g = g.copy()
        d = np.where(g.msp >= tp, np.where(g.pred == "normal", "healthy", g.pred), "unknown")
        wm = (g.role == "foldk_withheld").to_numpy()
        o_ = np.empty(len(g), dtype=object)
        o_[wm] = outcome_category(d[wm], g.class10.to_numpy()[wm], True)
        o_[~wm] = outcome_category(d[~wm], g.class10.to_numpy()[~wm], False)
        sup.append(g.assign(outcome=o_))
    out_cnn_msp95 = outcome_tables(pd.concat(sup), "outcome")
    auc_cnn = pd.read_csv(OUT / "loco_per_fold_auroc_oscr.csv")
    auc_cnn = auc_cnn[auc_cnn.score == "msp"].set_index(["withheld", "outer_fold"])
    auc_lr = pd.read_csv(results_dir() / "q1q3" / "q3_per_fold.csv")
    auc_lr = auc_lr[auc_lr.score == "msp"].set_index(["withheld", "outer_fold"])
    dd = (auc_cnn.auroc - auc_lr.auroc.loc[auc_cnn.index]).rename("auroc_diff").reset_index()
    do = (auc_cnn.oscr - auc_lr.oscr.loc[auc_cnn.index]).rename("oscr_diff").reset_index()
    diffs = dd.merge(do, on=["withheld", "outer_fold"])
    diffs.to_csv(OUT / "cnn_minus_lr_msp_per_fold.csv", index=False)

    def practical(o):
        return {"withheld_passing_as_known_fault": o["withheld"]["overall_recordings"]["known_fault"],
                "withheld_declared_healthy": o["withheld"]["overall_recordings"]["healthy"],
                "withheld_unknown": o["withheld"]["overall_recordings"]["unknown"],
                "withheld_unknown_macro_over_classes": o["withheld"]["macro_over_classes"]["unknown"],
                "false_unknown_a_seen_macro": o["known_faults"]["a_seen_folder"]["macro_over_known_fault_classes"]["unknown"],
                "false_unknown_b_unseen_macro": o["known_faults"]["b_unseen_foldk"]["macro_over_known_fault_classes"]["unknown"],
                "folder_gap_b_minus_a": o["known_faults"]["b_unseen_foldk"]["macro_over_known_fault_classes"]["unknown"]
                - o["known_faults"]["a_seen_folder"]["macro_over_known_fault_classes"]["unknown"]}

    res = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "seed": SEED,
                          "thresholds": f"tau_p, tau_d: {ACCEPT:.3f} marginal calibration targets each (order statistic "
                                        "r = ceil(0.975 (n + 1)); ties accepted); tau_e: max calibration-normal AE error",
                          "primary_ordering": "classifier_first", "comparison_ordering": "ae_first",
                          "classical": "LR/MSP of Stage 2D.1, 95 % order-statistic threshold, re-run per recording on the "
                                       "saved assignment", "wording": "observed differences; no winner claims without "
                                                                     "paired intervals"},
           "lr_reproduction": {"exact": not mism, "tolerance": 1e-12, "mismatches": mism[:20],
                               "note": "per-fold AUROC, OSCR and unknown recall vs results/q1q3/q3_per_fold.csv; the "
                                       "tolerance only absorbs the CSV float round-trip (last digit)"},
           "thresholds_summary": {"n_cal_range": [int(thr.n_cal.min()), int(thr.n_cal.max())],
                                  "attainable_level_range": [float(thr.attainable_level.min()), float(thr.attainable_level.max())],
                                  "n_cal_normals_range": [int(thr.n_cal_normals.min()), int(thr.n_cal_normals.max())]},
           "joint_acceptance": {"per_group_mean_over_runs": acc_sum.to_dict(orient="records"),
                                "pooled_all_known": pooled.to_dict()},
           "outcomes": outcomes, "outcomes_classical_lr_msp": out_lr, "outcomes_cnn_msp_only_95_supplementary": out_cnn_msp95,
           "override_counts": {"per_fold": ov.to_dict(orient="records"),
                               "totals": ov.groupby(["role", "kind"])[["overrides", "n"]].sum().reset_index().to_dict(orient="records")},
           "classical_comparison": {"auroc_msp_cnn_minus_lr": {"mean": float(diffs.auroc_diff.mean()), "std": float(diffs.auroc_diff.std()),
                                                               "runs_positive": int((diffs.auroc_diff > 0).sum()), "runs": len(diffs),
                                                               "per_class_mean": diffs.groupby("withheld").auroc_diff.mean().to_dict()},
                                    "oscr_msp_cnn_minus_lr": {"mean": float(diffs.oscr_diff.mean()), "std": float(diffs.oscr_diff.std()),
                                                              "runs_positive": int((diffs.oscr_diff > 0).sum())},
                                    "practical_primary": practical(outcomes["classifier_first"]),
                                    "practical_ae_first": practical(outcomes["ae_first"]),
                                    "practical_classical_lr": practical(out_lr),
                                    "practical_cnn_msp_only_95": practical(out_cnn_msp95)}}
    (OUT / "decision_rule.json").write_text(json.dumps(res, indent=1, default=float))
    print("LR reproduction exact:", not mism, mism[:3])
    print(acc_sum.round(3).to_string())
    for name, p in res["classical_comparison"].items():
        print(name, json.dumps(p, default=float)[:400])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["train", "auroc", "decide"])
    ap.add_argument("--withheld", choices=list(FAULT_CLASSES), default=None)
    ap.add_argument("--fold", type=int, choices=[0, 1, 2], default=None)
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (train task; Stage 3C.1 adds 43 and 44)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.task == "train":
        run_train(cfg, a.withheld, a.fold, a.seed)
    elif a.task == "auroc":
        run_auroc(cfg)
    else:
        run_decide(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
