"""Unknown-fault baselines (Track C1, Q3), CORRECTED protocol (Stage 2D.1; ml/protocol.py). Logistic regression on the
capped R1 features, LOCO over the 9 fault classes on the development pool. No torch.

    python scripts/12_track_c1.py

For each withheld class c and outer dev fold k:
- Training side = recordings of folds ≠ k, known classes (≠ c). Within EACH folder proxy, recordings are sorted by speed
  and dealt 4:1:1 (fixed seed) into FIT, CALIBRATION and SEEN-FOLDER EVALUATION (assignment saved).
- The model is fitted on FIT only (scaler inside the pipeline). Class c never touches fitting, scaling or thresholds.
- Unknown scores, directions FIXED before running (higher = more unknown): −MSP, −max-logit, energy E = −logsumexp(logits).
  Window → recording: probabilities by mean log-probability (renormalised), logits by their mean.
- Each score's threshold comes from its own CALIBRATION distribution (ml.protocol.order_statistic_threshold, 95 %
  acceptance: r = ceil(0.95 (n + 1)); accept iff score ≤ threshold, ties accepted).
- Evaluated with that model and threshold: (a) false-unknown rate on SEEN-FOLDER evaluation recordings; (b) false-unknown
  rate on fold-k known recordings (UNSEEN folders for fault classes; normal is one folder, so its fold-k recordings are
  within-folder and reported separately); (c) unknown recall on fold-k recordings of class c. Every class-c recording is
  evaluated exactly once, in its own outer fold.
- Primary: per-fold AUROC and OSCR, fold-k known vs fold-k class c, then mean and spread over folds and classes.
"""
from __future__ import annotations

import json
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

from ml.cache import load_features
from ml.classical import logreg_pipeline
from ml.config import load_config
from ml.features import feature_columns
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.protocol import order_statistic_threshold, speed_stratified_split
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "q1q3"
SCORES = ("msp", "max_logit", "energy")  # MSP primary


def unknown_scores(model, X: np.ndarray, rec_ids: np.ndarray, floor: float) -> pd.DataFrame:
    """Per recording: predicted class and the three unknown scores (higher = more unknown)."""
    classes = np.asarray(model.classes_)
    lp = pd.DataFrame(np.log(np.clip(model.predict_proba(X), floor, 1.0)), columns=classes).assign(rec_id=rec_ids)
    lp = lp.groupby("rec_id", sort=False).mean()
    prob = np.exp(lp.to_numpy())
    prob /= prob.sum(axis=1, keepdims=True)
    lg = pd.DataFrame(model.decision_function(X), columns=classes).assign(rec_id=rec_ids).groupby("rec_id", sort=False).mean()
    return pd.DataFrame({"pred": classes[prob.argmax(axis=1)], "msp": -prob.max(axis=1),
                         "max_logit": -lg.to_numpy().max(axis=1), "energy": -logsumexp(lg.to_numpy(), axis=1)},
                        index=lp.index)


def oscr(u_known: np.ndarray, known_correct: np.ndarray, u_unknown: np.ndarray) -> float:
    """Area under CCR (knowns accepted AND correctly classified) vs FPR (unknowns accepted) over all thresholds."""
    t = np.sort(np.unique(np.r_[u_known, u_unknown]))
    ccr = [0.0] + [float(np.mean((u_known <= x) & known_correct)) for x in t]
    fpr = [0.0] + [float(np.mean(u_unknown <= x)) for x in t]
    return float(np.trapezoid(ccr, fpr))


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    bc, oc, dc, pre = cfg.baselines, cfg.openset, cfg.development, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    path = max((data_dir() / "features").glob(f"{bc.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, pre).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = feature_columns(df)
    rec = pool.set_index("rec_id")
    folds = sorted(df.dev_fold.unique())
    assign_rows, fold_rows, rate_rows, thr_rows = [], [], [], []
    for c in FAULT_CLASSES:
        for k in folds:
            side = pool[(pool.dev_fold != k) & (pool.class10 != c)]
            parts = pd.concat([speed_stratified_split(g, {"fit": 4, "cal": 1, "eval": 1},
                                                      seed=dc.seed + zlib.crc32(f"{c}|{k}|{f}".encode()) % 10_000)
                               for f, g in side.groupby("folder")])
            assign_rows += [{"withheld": c, "outer_fold": int(k), "rec_id": r, "part": p} for r, p in parts.items()]
            fit = df[df.rec_id.isin(parts[parts == "fit"].index)]
            model = logreg_pipeline(bc).fit(fit[cols].to_numpy(), fit.class10.to_numpy())
            assert c not in model.classes_
            groups = {"cal": df[df.rec_id.isin(parts[parts == "cal"].index)],
                      "seen": df[df.rec_id.isin(parts[parts == "eval"].index)],
                      "unseen": df[(df.dev_fold == k) & (df.class10 != c)],
                      "unknown": df[(df.dev_fold == k) & (df.class10 == c)]}
            sc = {g: unknown_scores(model, d[cols].to_numpy(), d.rec_id.to_numpy(), bc.prob_floor) for g, d in groups.items()}
            correct = (sc["unseen"].pred == rec.loc[sc["unseen"].index, "class10"]).to_numpy()
            for s in SCORES:
                t = order_statistic_threshold(sc["cal"][s].to_numpy(), oc.acceptance)
                thr_rows.append({"withheld": c, "outer_fold": int(k), "score": s, **t})
                u_kn, u_un = sc["unseen"][s].to_numpy(), sc["unknown"][s].to_numpy()
                y = np.r_[np.zeros(len(u_kn)), np.ones(len(u_un))]
                fold_rows.append({"withheld": c, "outer_fold": int(k), "score": s, "auroc": float(roc_auc_score(y, np.r_[u_kn, u_un])),
                                  "oscr": oscr(u_kn, correct, u_un), "unknown_recall": float(np.mean(u_un > t["threshold"])),
                                  "n_unknown": int(len(u_un)), "n_known_unseen": int(len(u_kn))})
                for grp in ("seen", "unseen"):
                    lab = rec.loc[sc[grp].index, "class10"]
                    flagged = sc[grp][s].to_numpy() > t["threshold"]
                    for kc in sorted(set(lab)):
                        m = (lab == kc).to_numpy()
                        rate_rows.append({"withheld": c, "outer_fold": int(k), "score": s, "group": grp, "known_class": kc,
                                          "n": int(m.sum()), "false_unknowns": int(flagged[m].sum())})
            print(f"  {c:24s} fold {k}: n cal {t['n']}, MSP AUROC {fold_rows[-3]['auroc']:.3f}, recall {fold_rows[-3]['unknown_recall']:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    assign = pd.DataFrame(assign_rows)
    assign.to_csv(OUT / "q3_fit_cal_eval_assignment.csv", index=False)
    fr, rr, tr = pd.DataFrame(fold_rows), pd.DataFrame(rate_rows), pd.DataFrame(thr_rows)
    fr.to_csv(OUT / "q3_per_fold.csv", index=False)
    rr.to_csv(OUT / "q3_false_unknown_counts.csv", index=False)
    tr.to_csv(OUT / "q3_thresholds.csv", index=False)

    # Aggregates.
    summary = {}
    for s in SCORES:
        f = fr[fr.score == s]
        per_class = f.groupby("withheld").agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                                              oscr_mean=("oscr", "mean"), oscr_std=("oscr", "std"), n_unknown=("n_unknown", "sum"))
        # (c) pooled over the class's outer folds: every class-c recording counted once
        per_class["unknown_recall"] = (f.unknown_recall * f.n_unknown).groupby(f.withheld).sum() / per_class.n_unknown
        r = rr[rr.score == s]
        faults_r = r[r.known_class != "normal"]
        # per known fault class rates (pooled over folds within a withheld run), then macro over known fault classes
        rate = faults_r.groupby(["withheld", "group", "known_class"])[["false_unknowns", "n"]].sum()
        rate["rate"] = rate.false_unknowns / rate.n
        macro = rate.groupby(["withheld", "group"]).rate.mean().unstack("group")
        normal = r[r.known_class == "normal"].groupby(["withheld", "group"])[["false_unknowns", "n"]].sum()
        normal = (normal.false_unknowns / normal.n).unstack("group")
        per_class["fu_seen_known_faults_macro"] = macro["seen"]
        per_class["fu_unseen_known_faults_macro"] = macro["unseen"]
        per_class["fu_normal_seen"] = normal["seen"]
        per_class["fu_normal_within_folder_foldk"] = normal["unseen"]
        per_class.to_csv(OUT / f"q3_per_class_{s}.csv")
        by_known = rate.groupby(["group", "known_class"])[["false_unknowns", "n"]].sum()
        by_known["rate"] = by_known.false_unknowns / by_known.n
        summary[s] = {
            "auroc_fold_mean_over_classes": float(f.auroc.mean()), "auroc_spread_std_over_all_folds": float(f.auroc.std()),
            "auroc_class_mean_std": [float(per_class.auroc_mean.mean()), float(per_class.auroc_mean.std())],
            "oscr_class_mean_std": [float(per_class.oscr_mean.mean()), float(per_class.oscr_mean.std())],
            "unknown_recall_class_mean_std": [float(per_class.unknown_recall.mean()), float(per_class.unknown_recall.std())],
            "withheld_faults_passing_as_known": float(1 - (f.unknown_recall * f.n_unknown).sum() / f.n_unknown.sum()),
            "fu_seen_known_faults_macro_mean": float(per_class.fu_seen_known_faults_macro.mean()),
            "fu_unseen_known_faults_macro_mean": float(per_class.fu_unseen_known_faults_macro.mean()),
            "fu_normal_seen_mean": float(per_class.fu_normal_seen.mean()),
            "fu_normal_within_folder_foldk_mean": float(per_class.fu_normal_within_folder_foldk.mean()),
            "fu_by_known_class": {f"{g}|{kc}": float(v) for (g, kc), v in by_known.rate.items()},
            "per_class": per_class.round(4).reset_index().to_dict("records"),
        }
    t_s = tr[tr.score == "msp"]
    reuse = assign[assign.part == "eval"].groupby(["withheld", "rec_id"]).size()
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(),
                          "protocol": "ml/protocol.py; CLAUDE.md rule 13", "acceptance_target": oc.acceptance,
                          "threshold_rule": "r = ceil(0.95 (n + 1)); threshold = r-th smallest calibration score; "
                                            "accept iff score <= threshold (ties accepted)",
                          "calibration_n_range": [int(t_s.n.min()), int(t_s.n.max())],
                          "attainable_level_range": [float(t_s.attainable_level.min()), float(t_s.attainable_level.max())],
                          "seen_eval_recordings_reused_across_outer_folds": int((reuse > 1).sum()),
                          "seen_eval_recordings_total_unique_per_withheld_sum": int(len(reuse))},
           "scores": summary}
    (OUT / "q3_corrected.json").write_text(json.dumps(out, indent=1))
    for s in SCORES:
        v = summary[s]
        print(f"{s:9s} AUROC {v['auroc_class_mean_std'][0]:.3f} ± {v['auroc_class_mean_std'][1]:.3f} | OSCR "
              f"{v['oscr_class_mean_std'][0]:.3f} ± {v['oscr_class_mean_std'][1]:.3f} | recall "
              f"{v['unknown_recall_class_mean_std'][0]:.3f} | FU seen faults {v['fu_seen_known_faults_macro_mean']:.3f} vs "
              f"unseen faults {v['fu_unseen_known_faults_macro_mean']:.3f} | normal seen {v['fu_normal_seen_mean']:.3f}, "
              f"fold-k {v['fu_normal_within_folder_foldk_mean']:.3f} | withheld passing as known "
              f"{v['withheld_faults_passing_as_known']:.3f}")
    print(json.dumps(out["provenance"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
