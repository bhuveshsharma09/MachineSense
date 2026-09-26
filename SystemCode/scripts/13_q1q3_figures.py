"""Deck-ready figures for the classical Q1 (Track B) and Q3 (Track C1) baselines, development data only.

    python scripts/13_q1q3_figures.py

- results/figures/q1_auroc_best_rung.png: per-class AUROC of the best Track B rung (highest mean AUROC; ties broken by
  mean detection at the false-alarm threshold)
- results/figures/q1_mildest_detection.png: detection at the mildest severities, best rung, with vs without microphone
- results/figures/q3_loco_auroc.png: per-class LOCO AUROC (known vs unknown) for MSP, max logit and energy
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


def main() -> int:
    b = json.loads((RES / "track_b.json").read_text())
    rungs = b["rungs"]
    best = max(rungs, key=lambda r: (round(rungs[r]["all"]["auroc_mean"], 3), rungs[r]["all"]["detection_mean"]))
    a = rungs[best]["all"]
    n_h, k = a["n_healthy"], a["false_alarms"]

    fig, ax = plt.subplots(figsize=(9, 4))
    cls = list(a["auroc"])
    ax.bar(range(len(cls)), [a["auroc"][c] for c in cls], color="#4c72b0")
    ax.set_xticks(range(len(cls)), cls, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0.5, 1.01), ylabel="AUROC, healthy vs class",
           title=f"Q1 healthy-only detector {best}: AUROC per fault class (dev pool; {n_h} healthy, one folder)")
    ax.title.set_fontsize(9)
    for i, c in enumerate(cls):
        ax.text(i, a["auroc"][c] + 0.005, f"{a['auroc'][c]:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "q1_auroc_best_rung.png")

    mild = list(a["mildest"])
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mild))
    for off, (tag, lab) in zip((-0.2, 0.2), (("all", "with microphone"), ("no_mic", "without microphone"))):
        ax.bar(x + off, [rungs[best][tag]["mildest"][m]["detection"] for m in mild], width=0.4, label=lab)
    ax.set_xticks(x, [f"{m}\n(n={a['mildest'][m]['n']})" for m in mild], rotation=30, ha="right", fontsize=7)
    ax.set(ylim=(0, 1.05), ylabel=f"detection rate at {k}/{n_h} false alarms",
           title=f"Q1 mildest severities, {best} (false-alarm threshold from out-of-fold healthy scores)")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q1_mildest_detection.png")

    per = pd.read_csv(RES / "track_c1_per_class.csv")
    piv = per.pivot(index="withheld", columns="score", values="auroc_unknown").reindex(per.withheld.unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(piv))
    for i, s in enumerate(["msp", "max_logit", "energy"]):
        ax.bar(x + (i - 1) * 0.27, piv[s], width=0.27, label=f"{s} (mean {piv[s].mean():.3f})")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, piv.index, rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.05), ylabel="AUROC, known vs withheld class",
           title="Q3 unknown-fault detection, LOCO over 9 fault classes (logistic regression, dev pool)")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q3_loco_auroc.png")
    print(f"best Track B rung: {best}; figures written to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
