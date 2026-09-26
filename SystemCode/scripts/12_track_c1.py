"""Unknown-fault baselines (Track C1, Q3): logistic regression on the capped R1 features, LOCO over the 9 fault classes on
the development pool. No torch.

    python scripts/12_track_c1.py

For each withheld class: fit on the other classes of the pool with results/dev_folds.csv, so every known recording is
scored out of fold; every withheld recording is scored by each fold's model and the scores are averaged. The withheld
class never touches fitting, scaling (inside the pipeline), calibration or the threshold. Window → recording: class
probabilities by mean log-probability (renormalised); logits by their mean. Scores (higher = more "known"): maximum softmax
probability, maximum logit, energy (logsumexp of logits). Threshold: `openset.acceptance` of the known out-of-fold
recordings. Q3 control: known fault recordings are scored from unseen folder proxies (new severity, known class); their
false-unknown rate is reported separately from normal (scored within its single folder).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

from ml.cache import load_features
from ml.classical import logreg_pipeline
from ml.config import load_config
from ml.features import feature_columns
from ml.labels import CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "q1q3"


def recording_scores(model, X: np.ndarray, rec_ids: np.ndarray, floor: float) -> pd.DataFrame:
    p = np.clip(model.predict_proba(X), floor, 1.0)
    logits = model.decision_function(X)
    classes = list(model.classes_)
    df = pd.DataFrame(np.log(p), columns=classes).assign(rec_id=rec_ids)
    lp = df.groupby("rec_id", sort=False).mean()
    prob = np.exp(lp.to_numpy())
    prob /= prob.sum(axis=1, keepdims=True)
    lg = pd.DataFrame(logits, columns=classes).assign(rec_id=rec_ids).groupby("rec_id", sort=False).mean()
    return pd.DataFrame({"msp": prob.max(axis=1), "pred": np.asarray(classes)[prob.argmax(axis=1)],
                         "max_logit": lg.to_numpy().max(axis=1), "energy": logsumexp(lg.to_numpy(), axis=1)},
                        index=lp.index)


def oscr(known_score, known_correct, unknown_score) -> float:
    """Area under the open-set classification rate curve: correct-classification rate of knowns accepted at threshold t
    vs the rate of unknowns accepted, over all thresholds."""
    t = np.sort(np.unique(np.r_[known_score, unknown_score]))[::-1]
    ccr = [0.0] + [float(np.mean((known_score >= x) & known_correct)) for x in t]
    fpr = [0.0] + [float(np.mean(unknown_score >= x)) for x in t]
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
    results, rows = {}, []
    for w in FAULT_CLASSES:
        known = df[df.class10 != w]
        unk = df[df.class10 == w]
        kscores, uscores = [], []
        for j in folds:
            tr, te = known[known.dev_fold != j], known[known.dev_fold == j]
            m = logreg_pipeline(bc).fit(tr[cols].to_numpy(), tr.class10.to_numpy())
            assert w not in m.classes_
            kscores.append(recording_scores(m, te[cols].to_numpy(), te.rec_id.to_numpy(), bc.prob_floor))
            uscores.append(recording_scores(m, unk[cols].to_numpy(), unk.rec_id.to_numpy(), bc.prob_floor))
        ks = pd.concat(kscores)
        us = pd.concat(uscores).drop(columns="pred").groupby(level=0).mean()
        correct = (ks.pred == rec.loc[ks.index, "class10"]).to_numpy()
        is_normal = (rec.loc[ks.index, "class10"] == "normal").to_numpy()
        entry = {"n_known": int(len(ks)), "n_unknown": int(len(us)), "known_accuracy": float(correct.mean())}
        for sname in oc.scores:
            k, u = ks[sname].to_numpy(), us[sname].to_numpy()
            thr = float(np.quantile(k, 1 - oc.acceptance))
            y = np.r_[np.zeros(len(k)), np.ones(len(u))]
            entry[sname] = {
                "auroc_unknown": float(roc_auc_score(y, -np.r_[k, u])), "oscr": oscr(k, correct, u),
                "threshold": thr, "unknown_recall": float(np.mean(u < thr)),
                "false_unknown_rate": float(np.mean(k < thr)),
                "false_unknown_rate_known_faults_unseen_folders": float(np.mean(k[~is_normal] < thr)),
                "false_unknown_rate_normal_within_folder": float(np.mean(k[is_normal] < thr)),
                "n_known_faults": int((~is_normal).sum()), "n_normal": int(is_normal.sum())}
            rows.append({"withheld": w, "score": sname, **{kk: vv for kk, vv in entry[sname].items() if kk != "threshold"}})
        results[w] = entry
        print(f"  {w:24s} known acc {entry['known_accuracy']:.3f} | " + " | ".join(
            f"{s}: AUROC {entry[s]['auroc_unknown']:.3f} OSCR {entry[s]['oscr']:.3f} recall {entry[s]['unknown_recall']:.2f} "
            f"FU faults {entry[s]['false_unknown_rate_known_faults_unseen_folders']:.3f}" for s in ("msp", "energy")))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "track_c1_per_class.csv", index=False)
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(),
                          "model": "logistic regression (baselines.logreg, QuantileTransformer)", "k_folds": len(folds),
                          "acceptance": oc.acceptance},
           "per_class": results,
           "summary": {s: {m: {"mean": float(per[per.score == s][m].mean()), "std": float(per[per.score == s][m].std())}
                           for m in ("auroc_unknown", "oscr", "unknown_recall", "false_unknown_rate",
                                     "false_unknown_rate_known_faults_unseen_folders",
                                     "false_unknown_rate_normal_within_folder")} for s in oc.scores}}
    (OUT / "track_c1.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame({s: {m: f"{v['mean']:.3f} ± {v['std']:.3f}" for m, v in d.items()} for s, d in out["summary"].items()}).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
