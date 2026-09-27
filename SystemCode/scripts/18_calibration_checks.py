"""Sanity report and calibration consistency on the 144 calibration recordings (closed val ∩ development pool) — no
evaluation of the full closed-validation split (recordings outside the pool are reserved under other splits).

    python scripts/18_calibration_checks.py

Sanity (ML-4.1 item 8; label kept): class10 / class6 macro-F1 and confusion matrices of diagnose() (CPU).
Consistency (ML-4.2): since Stage ML-4.2 the thresholds are set from ml.inference.score_recording, the function
diagnose() uses; this script checks that diagnose()'s scores equal the saved calibration scores (bundle
calibration_scores.csv) exactly, and reports the marginal and joint acceptance and the 5 calibration normals under every
rule variant. (The Stage ML-4.1 finding, from the former training-device path, is kept in the git history of
results/bundle/calibration_consistency.json.)
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from ml.arbitrator import VARIANTS, decide
from ml.inference import diagnose, get_bundle
from ml.labels import CLASS6, CLASS10
from ml.paths import SYSTEM_CODE_DIR, results_dir
from ml.splits import development_pool

OUT = results_dir() / "bundle"
LABEL = ("calibration recordings, not an independent test; expected to be optimistic. These recordings were used for "
         "calibration (not for epoch selection: the epochs were fixed in advance); the bundle's training and calibration "
         "sets together use the full development pool; the only independent evaluation left is the reserved tests")


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
    # ---- consistency with the scores that set the thresholds
    from ml.calibration import acceptance
    cal = pd.read_csv(b.directory / "calibration_scores.csv", index_col="rec_id", float_precision="round_trip").loc[ids]
    cols = ["msp", "maha", "ae", "b1s"]
    equal = bool((cal[cols].to_numpy() == inf[cols].to_numpy()).all() and (cal.top_class == inf.pred10).all())
    normals = []
    for r in va.index[va.class10 == "normal"]:
        x = inf.loc[r]
        normals.append({"rec_id": r, "pred10": x.pred10, "msp_ge_tau_p": bool(x.msp >= T["tau_p"]),
                        "maha_le_tau_d": bool(x.maha <= T["tau_d"]), "ae_le_tau_e": bool(x.ae <= T["tau_e"]),
                        "b1s_le_tau_b": bool(x.b1s <= T["tau_b"]),
                        "verdicts": {v: (lambda o: (o["fault_class"] or o["decision"]) + (f" (caution: {o['caution']})" if o["caution"] else ""))(
                            decide(v, x.pred10, x.msp, x.maha, x.ae, x.b1s, T)) for v in VARIANTS}})
    consistency = {"stage": "ML-4.2", "thresholds": T, "scoring_path": "ml.inference.score_recording (CPU)",
                   "diagnose_scores_equal_calibration_scores": equal,
                   "max_abs_difference": {c: float((cal[c] - inf[c]).abs().max()) for c in cols},
                   "acceptance": acceptance(inf, T), "normals": normals}
    (OUT / "calibration_consistency.json").write_text(json.dumps(consistency, indent=1, default=float))
    stale = OUT / "calibration_consistency_scores.csv"
    if stale.exists():
        stale.unlink()  # the ML-4.1 two-path comparison is superseded (kept in git history)
    print(json.dumps({k: sanity[k] for k in ("class10_macro_f1", "class6_macro_f1")}))
    print(json.dumps({k: consistency[k] for k in ("diagnose_scores_equal_calibration_scores", "max_abs_difference", "acceptance")}, indent=1))
    for n in normals:
        print(n["rec_id"], n["verdicts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
