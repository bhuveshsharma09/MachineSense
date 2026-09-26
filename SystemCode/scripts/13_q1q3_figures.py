"""Deck-ready figures for the classical Q1 and Q3 baselines under the corrected protocol (Stage 2D.1), dev pool only.
Reads results/q1q3/q1_corrected.json (scripts/11_track_b.py) and q3_corrected.json / q3_per_class_*.csv
(scripts/12_track_c1.py).

    python scripts/13_q1q3_figures.py

- q1_auroc_best_rung.png: per-class AUROC of the best Q1 rung (highest fold-mean overall AUROC), fold mean ± spread
- q1_thresholds.png: false alarms (out of 25 held-out healthy) and detection at each rung's calibrated threshold, with
  calibration counts and measured vs theoretical acceptance
- q1_mildest_detection.png: detection at the mildest severities: best rung with and without microphone, and B0 rules
- q3_loco_auroc.png: per-withheld-class AUROC (fold mean ± spread) for MSP, max logit and energy
- q3_folder_control.png: MSP false-unknown rate on seen-folder (a) vs unseen-folder (b) known faults (macro over known
  fault classes) and unknown recall (c), per withheld class
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.paths import results_dir

RES = results_dir() / "q1q3"
FIG = results_dir() / "figures"
SCORES = ("msp", "max_logit", "energy")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def q1_figures(q1: dict) -> str:
    rungs = q1["rungs"]
    best = q1["best_two_by_fold_mean_auroc"][0]
    a = rungs[best]["all"]
    cal, n_h = a["calibration_counts"], a["heldout_normals_total"]
    cls = [c for c in a["auroc_fold_mean"] if c != "overall"]

    fig, ax = plt.subplots(figsize=(9, 4))
    m = np.array([a["auroc_fold_mean"][c] for c in cls])
    lo = m - np.array([a["auroc_fold_min"][c] for c in cls])
    hi = np.array([a["auroc_fold_max"][c] for c in cls]) - m
    ax.bar(range(len(cls)), m, color="#4c72b0", yerr=[lo, hi], capsize=3, error_kw={"lw": 0.8})
    for i, v in enumerate(m):
        ax.text(i, 0.505, f"{v:.3f}", ha="center", fontsize=7, color="w")
    ax.set_xticks(range(len(cls)), cls, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0.5, 1.01), ylabel="AUROC, held-out healthy vs class")
    ax.set_title(f"Development finding. Q1 {best}: per-class AUROC, mean of 3 dev folds (bars: fold min–max). Overall "
                 f"{a['auroc_fold_mean']['overall']:.3f} ± {a['auroc_fold_std']['overall']:.3f}. "
                 f"Healthy: 25 recordings, one folder", fontsize=8)
    _save(fig, "q1_auroc_best_rung.png")

    names = list(rungs)
    fa = [rungs[r]["all"]["false_alarms_total"] for r in names]
    det = [rungs[r]["all"]["detection_overall"] for r in names]
    acc = [rungs[r]["all"]["measured_acceptance_total"] for r in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(range(len(names)), fa, color="#c44e52")
    for i, (f, ac) in enumerate(zip(fa, acc)):
        ax1.text(i, f + 0.1, f"{f}/{n_h}\nacc {ac:.2f}", ha="center", fontsize=6.5)
    theo = np.mean(a["theoretical_acceptance_reference"])
    ax1.axhline(n_h * (1 - theo), color="k", ls=":", lw=0.8, label=f"expected ≈ {n_h * (1 - theo):.1f} (acc ≈ {theo:.2f})")
    ax1.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=7)
    ax1.set(ylabel=f"false alarms (out of {n_h} held-out healthy)", ylim=(0, max(fa) + 2))
    ax1.set_title(f"False alarms at threshold = max of calibration healthy (n cal per fold {cal})", fontsize=8)
    ax1.legend(fontsize=7)
    ax2.bar(range(len(names)), det, color="#55a868")
    for i, d in enumerate(det):
        ax2.text(i, d + 0.01, f"{d:.2f}", ha="center", fontsize=7)
    ax2.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=7)
    ax2.set(ylabel="detection rate, all fault recordings", ylim=(0, 1.08))
    ax2.set_title("Detection at the same thresholds (each recording scored once, own fold)", fontsize=8)
    _save(fig, "q1_thresholds.png")

    mild = list(a["detection_mildest"])
    series = [(f"{best}, with microphone", rungs[best]["all"]),
              (f"{best}, without microphone", rungs[best].get("no_mic")),
              ("B0_rules (simple rules)", rungs["B0_rules"]["all"])]
    series = [(lab, s) for lab, s in series if s]
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mild))
    for j, (lab, s) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * w, [s["detection_mildest"][k]["detection"] for k in mild], width=w,
               label=f"{lab}: {s['false_alarms_total']}/{n_h} false alarms")
    ax.set_xticks(x, [f"{k}\n(n={a['detection_mildest'][k]['n']})" for k in mild], rotation=30, ha="right", fontsize=7)
    ax.set(ylim=(0, 1.3), ylabel="detection rate at the calibrated threshold")
    ax.set_title("Development finding. Q1 mildest severity per fault class (dev pool; thresholds from separate calibration healthy)",
                 fontsize=8)
    ax.legend(fontsize=7, loc="upper center", ncol=len(series))
    _save(fig, "q1_mildest_detection.png")
    return best


def q3_figures(q3: dict) -> None:
    per = {s: pd.read_csv(RES / f"q3_per_class_{s}.csv").set_index("withheld") for s in SCORES}
    idx = per["msp"].index
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(idx))
    for i, s in enumerate(SCORES):
        p = per[s].loc[idx]
        v = q3["scores"][s]["auroc_class_mean_std"]
        ax.bar(x + (i - 1) * 0.27, p.auroc_mean, width=0.27, yerr=p.auroc_std, capsize=2, error_kw={"lw": 0.7},
               label=f"{s}{' (primary)' if s == 'msp' else ''}: {v[0]:.3f} ± {v[1]:.3f}")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, idx, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.3), ylabel="AUROC, fold-k known vs withheld class")
    ax.set_title("Development finding. Q3 unknown-fault detection, LOCO over 9 fault classes (logistic regression; mean ± std over 3 dev "
                 "folds)", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", ncol=3)
    _save(fig, "q3_loco_auroc.png")

    p = per["msp"].loc[idx]
    s = q3["scores"]["msp"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (col, lab, colr) in enumerate([
            ("fu_seen_known_faults_macro", f"(a) false unknown, seen-folder known faults (mean {s['fu_seen_known_faults_macro_mean']:.2f})", "#8c8c8c"),
            ("fu_unseen_known_faults_macro", f"(b) false unknown, unseen-folder known faults (mean {s['fu_unseen_known_faults_macro_mean']:.2f})", "#dd8452"),
            ("unknown_recall", f"(c) withheld class flagged unknown (macro over withheld classes {s['unknown_recall_class_mean_std'][0]:.2f}; "
             f"all withheld recordings {1 - s['withheld_faults_passing_as_known']:.2f})", "#4c72b0")]):
        ax.bar(x + (i - 1) * 0.27, p[col], width=0.27, color=colr, label=lab)
    ax.set_xticks(x, idx, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.3), ylabel="rate at the 95 % calibration threshold (MSP)")
    ax.set_title("Development finding. Q3 folder control: known faults from unseen folders are flagged 'unknown' far more often than from "
                 "seen folders", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    _save(fig, "q3_folder_control.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    best = q1_figures(json.loads((RES / "q1_corrected.json").read_text()))
    q3_figures(json.loads((RES / "q3_corrected.json").read_text()))
    print(f"best Q1 rung: {best}; figures written to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
