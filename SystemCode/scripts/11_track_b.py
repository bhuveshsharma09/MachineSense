"""Healthy-only detectors (Track B classical baselines, Q1) on the capped R1 features, development pool only.

    python scripts/11_track_b.py

Protocol: detectors are fitted on dev-pool NORMAL recordings only, with results/dev_folds.csv: for fold j, on the normal
recordings outside fold j. Each normal recording is scored out of fold; every dev-pool fault recording is scored by every
fold's model and the scores are averaged. Recording score = mean over its 9 window scores (B1s kNN: recording level).
Threshold at the configured false-alarm rate from the out-of-fold healthy scores. All healthy data is one folder, so
healthy-vs-fault may partly reflect folder proxies. Rungs: B0 rules; B1 Mahalanobis (Ledoit-Wolf), GMM (EM);
B1s speed-conditioned polynomial baseline + kNN-by-speed comparison; B2 PCA reconstruction; B3 one-class SVM, isolation
forest, LOF. No torch.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.cache import load_features
from ml.config import AnomalyConfig, load_config
from ml.features import feature_columns
from ml.labels import CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "q1q3"
RUNGS = ["B0_rules", "B1_mahalanobis", "B1_gmm", "B1s_speed_poly", "B1s_knn_speed", "B2_pca", "B3_ocsvm",
         "B3_iforest", "B3_lof"]


class Prep:
    """Fitted on healthy training windows only: log10 of columns positive there, then standard scaling."""

    def fit(self, X):
        self.log = (X > 0).all(axis=0)
        self.sc = StandardScaler().fit(self._t(X))
        return self

    def _t(self, X):
        Y = X.copy()
        Y[:, self.log] = np.log10(np.maximum(Y[:, self.log], 1e-12))
        return Y

    def __call__(self, X):
        return self.sc.transform(self._t(X))


def window_scores(rung: str, ac: AnomalyConfig, Xtr, str_, Xte, ste, cols):
    """Anomaly score per test window (higher = more anomalous) for window-level rungs."""
    prep = Prep().fit(Xtr)
    A, B = prep(Xtr), prep(Xte)
    if rung == "B0_rules":
        idx = [cols.index(c) for c in ac.b0_features if c in cols]
        med = np.median(Xtr[:, idx], axis=0)
        q = np.quantile(Xtr[:, idx], ac.b0_quantile, axis=0)
        return ((Xte[:, idx] - med) / np.maximum(q - med, 1e-12)).max(axis=1)
    if rung == "B1_mahalanobis":
        lw = LedoitWolf().fit(A)
        return lw.mahalanobis(B)
    if rung == "B1_gmm":
        g = GaussianMixture(ac.gmm_components, covariance_type=ac.gmm_covariance, random_state=ac.seed).fit(A)
        return -g.score_samples(B)
    if rung == "B1s_speed_poly":
        ls_tr, ls_te = np.log(str_), np.log(ste)
        V_tr = np.vander(ls_tr, ac.b1s_degree + 1)
        coef, *_ = np.linalg.lstsq(V_tr, A, rcond=None)
        resid_sd = (A - V_tr @ coef).std(axis=0, ddof=ac.b1s_degree + 1)
        return (np.abs(B - np.vander(ls_te, ac.b1s_degree + 1) @ coef) / np.maximum(resid_sd, 1e-9)).mean(axis=1)
    if rung == "B2_pca":
        p = PCA(n_components=ac.pca_variance, svd_solver="full").fit(A)
        return ((B - p.inverse_transform(p.transform(B))) ** 2).mean(axis=1)
    if rung == "B3_ocsvm":
        return -OneClassSVM(nu=ac.ocsvm_nu, gamma="scale").fit(A).decision_function(B)
    if rung == "B3_iforest":
        return -IsolationForest(n_estimators=ac.iforest_trees, random_state=ac.seed).fit(A).score_samples(B)
    if rung == "B3_lof":
        return -LocalOutlierFactor(n_neighbors=ac.lof_neighbors, novelty=True).fit(A).score_samples(B)
    raise ValueError(rung)


def knn_speed_scores(ac, tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Recording level: distance (standardised recording-mean features) to the k healthy training recordings nearest in
    measured speed; also returns each scored recording's median speed gap to those neighbours."""
    prep = Prep().fit(tr[cols].to_numpy())
    rtr = tr.groupby("rec_id").agg({**{c: "mean" for c in cols}, "fr_measured_hz": "first"})
    rte = te.groupby("rec_id").agg({**{c: "mean" for c in cols}, "fr_measured_hz": "first"})
    A, B = prep(rtr[cols].to_numpy()), prep(rte[cols].to_numpy())
    s_tr, s_te = rtr.fr_measured_hz.to_numpy(), rte.fr_measured_hz.to_numpy()
    scores, gaps = [], []
    for i in range(len(rte)):
        nn = np.argsort(np.abs(s_tr - s_te[i]))[:ac.b1s_knn]
        scores.append(np.linalg.norm(A[nn] - B[i], axis=1).mean())
        gaps.append(np.median(np.abs(s_tr[nn] - s_te[i])))
    return pd.Series(scores, index=rte.index), pd.Series(gaps, index=rte.index)


def run_rung(rung, ac, df, cols, folds):
    """Out-of-fold healthy recording scores and fold-averaged fault recording scores."""
    healthy = df[df.class10 == "normal"]
    faults = df[df.class10 != "normal"]
    h_scores, f_scores, gaps = {}, [], []
    for j in folds:
        tr = healthy[healthy.dev_fold != j]
        te_h = healthy[healthy.dev_fold == j]
        te = pd.concat([te_h, faults])
        if rung == "B1s_knn_speed":
            sc, gap = knn_speed_scores(ac, tr, te, cols)
            gaps.append(gap)
        else:
            w = window_scores(rung, ac, tr[cols].to_numpy(), tr.speed_hz.to_numpy(), te[cols].to_numpy(),
                              te.speed_hz.to_numpy(), cols)
            sc = pd.Series(w, index=te.rec_id.to_numpy()).groupby(level=0).mean()
        h_scores.update(sc.loc[te_h.rec_id.unique()].to_dict())
        f_scores.append(sc.loc[faults.rec_id.unique()])
    out = pd.concat([pd.Series(h_scores), pd.concat(f_scores, axis=1).mean(axis=1)])
    gap = pd.concat(gaps, axis=1).mean(axis=1) if gaps else None
    return out, gap


def metrics(scores: pd.Series, rec: pd.DataFrame, ac) -> dict:
    lab = rec.set_index("rec_id").loc[scores.index]
    h = scores[lab.class10 == "normal"].sort_values(ascending=False)
    n_h = len(h)
    allowed = int(np.floor(ac.false_alarm * n_h))
    thr = float(h.iloc[allowed])  # alarm if score > thr: exactly `allowed` healthy recordings alarm
    k = int((h > thr).sum())
    fa_ci = [float(beta.ppf(0.025, k, n_h - k + 1)) if k > 0 else 0.0, float(beta.ppf(0.975, k + 1, n_h - k))]
    out = {"n_healthy": n_h, "threshold": thr, "false_alarms": k, "false_alarm_rate": k / n_h,
           "false_alarm_ci95": fa_ci, "auroc": {}, "detection": {}, "mildest": {}}
    for c in CLASS10[1:]:
        s = scores[lab.class10 == c]
        y = np.r_[np.zeros(n_h), np.ones(len(s))]
        out["auroc"][c] = float(roc_auc_score(y, np.r_[h.to_numpy(), s.to_numpy()]))
        out["detection"][c] = float((s > thr).mean())
    for fam, sev in ac.mildest.items():
        sel = (lab.fault_family == fam) & np.isclose(lab.severity, sev) if fam == "bearing" else \
              (lab.class10 == fam) & np.isclose(lab.severity, sev)
        for c in sorted(set(lab.class10[sel])):
            s = scores[sel & (lab.class10 == c)]
            out["mildest"][f"{c} @ {sev:g}"] = {"n": int(len(s)), "detection": float((s > thr).mean())}
    out["auroc_mean"] = float(np.mean(list(out["auroc"].values())))
    out["detection_mean"] = float(np.mean(list(out["detection"].values())))
    return out


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    ac, pre, dc = cfg.anomaly, cfg.preprocess, cfg.development
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    folds = pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]]
    pool = pool.merge(folds, on="rec_id")
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, pre).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    rec = pool[["rec_id", "class10", "fault_family", "severity", "folder", "fr_measured_hz"]]
    all_cols = feature_columns(df)
    no_mic = [c for c in all_cols if not c.startswith("f_mic_")]
    folds_ = sorted(df.dev_fold.unique())
    print(f"{path.name}: {df.rec_id.nunique()} pool recordings ({(rec.class10 == 'normal').sum()} healthy, one folder)")
    res = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(),
                          "n_healthy": int((rec.class10 == "normal").sum()), "k_folds": len(folds_),
                          "caveat": "all healthy data is one folder, so healthy-vs-fault may partly reflect folder proxies"},
           "rungs": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    score_tables = {}
    for rung in RUNGS:
        entry = {}
        for tag, cols in (("all", all_cols), ("no_mic", no_mic)):
            sc, gap = run_rung(rung, ac, df, cols, folds_)
            entry[tag] = metrics(sc, rec, ac)
            if gap is not None:
                entry[tag]["median_speed_gap_hz"] = float(gap.median())
            if tag == "all":
                score_tables[rung] = sc
        res["rungs"][rung] = entry
        a = entry["all"]
        print(f"  {rung:16s} AUROC mean {a['auroc_mean']:.3f}, detection mean {a['detection_mean']:.3f} at FA "
              f"{a['false_alarms']}/{a['n_healthy']}; no mic: AUROC {entry['no_mic']['auroc_mean']:.3f}, detection "
              f"{entry['no_mic']['detection_mean']:.3f}" + (f"; median speed gap {a['median_speed_gap_hz']:.2f} Hz"
                                                          if "median_speed_gap_hz" in a else ""))
    pd.DataFrame(score_tables).to_csv(OUT / "track_b_scores.csv")
    (OUT / "track_b.json").write_text(json.dumps(res, indent=1))
    print(f"→ {(OUT / 'track_b.json').relative_to(SYSTEM_CODE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
