"""Stage 3D: complementary evidence for Q3 from saved scores (CPU only; no neural-network training; no torch, no LightGBM).
Protocol declared in CLAUDE.md §7 before running; candidates, formulas and criteria are fixed; nothing is selected.

    python scripts/09c_loco_complementary.py e1         # E1 deep ensemble over seeds 42/43/44
    python scripts/09c_loco_complementary.py e2         # E2 LR + CNN combined unfamiliarity (per seed)
    python scripts/09c_loco_complementary.py e3         # E3 B1s healthy veto, D and D′ (per seed)
    python scripts/09c_loco_complementary.py evaluate   # tables vs rules A/B/C, paired threshold-free, criteria

Exploratory development results on one partition; the candidates were chosen after seeing the 3C.1 results.
Outputs: results/loco/complementary/*.json|csv; per-recording decisions results/scores/<name>_decisions.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from ml.loco_eval import ACCEPT, SEEDS, decide, lr_decisions, outcomes, recordings
from ml.openset import empirical_rank, js_divergence, recording_probs
from ml.paths import SYSTEM_CODE_DIR, RunPaths, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, split_hash
from ml.utils import git_sha

OUT = results_dir() / "loco" / "complementary"
ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")
WORDING = ("Exploratory development results on one partition. The candidates were chosen after seeing 3C.1 results. Any "
           "final claim uses the existing reserved-recording protocol (the loco_* and other reserved-test columns in "
           "metadata.csv), opened once at the end. All nine fault classes have informed development decisions. Reserved "
           "recordings test those decisions; they do not establish performance on entirely new fault families.")


def provenance() -> dict:
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    return {"split_hash": split_hash(meta), "git_sha": git_sha(), "wording": WORDING}


def threshold(cal_scores: np.ndarray) -> tuple[float, bool]:
    """Order-statistic threshold (higher = more unfamiliar; accept iff ≤, ties accepted) and whether it is attainable."""
    t = order_statistic_threshold(np.asarray(cal_scores, dtype=np.float64), ACCEPT)
    return float(t["threshold"]), bool(np.isfinite(t["threshold"]))


def rule_decisions(conf: np.ndarray, pred: np.ndarray, healthy_ok: np.ndarray | None = None) -> np.ndarray:
    """Confident → healthy (label normal, and `healthy_ok` if given) or that fault; otherwise unknown. No override."""
    normal = pred == "normal"
    ok = np.ones(len(pred), bool) if healthy_ok is None else healthy_ok
    return np.where(conf & ~normal, pred, np.where(conf & normal & ok, "healthy", "unknown")).astype(object)


def decisions_path(name: str, seed: int | None):
    """results/scores/<name>_s<seed>_loco_decisions.parquet; the ensemble (seed None) names all three seeds."""
    if seed is None:
        return results_dir() / "scores" / f"{name}_s{'-'.join(map(str, SEEDS))}_loco_decisions.parquet"
    return RunPaths("loco3d", name, seed, protocol="loco").scores("decisions")


def save(name: str, seed: int | None, df: pd.DataFrame) -> None:
    p = decisions_path(name, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def load(name: str, seed: int | None) -> pd.DataFrame:
    return pd.read_parquet(decisions_path(name, seed))


def run_e1() -> None:
    """E1: one ensemble predictor per run over seeds 42/43/44."""
    recs = {s: recordings(s) for s in SEEDS}
    parts, thr_rows = [], []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            probs, ranks, base = {}, {}, None
            for s in SEEDS:
                w = pd.read_parquet(RunPaths("loco", "cnn6", s, protocol="loco", withheld=c).scores(f"fold{k}_logits"))
                w = w[w.role.isin(ROLES)]
                classes = [x.removeprefix("logit_") for x in w.columns if x.startswith("logit_")]
                probs[s] = recording_probs(w[[f"logit_{x}" for x in classes]].to_numpy(), w.rec_id, classes)
                r = recs[s][(recs[s].withheld == c) & (recs[s].outer_fold == k)].set_index("rec_id")
                ranks[s] = pd.Series(empirical_rank(r.maha, r.maha[r.role == "cal"]), index=r.index)
                base = r if base is None else base
            classes = sorted(probs[SEEDS[0]].columns)
            idx = base.index
            P = {s: probs[s].loc[idx, classes].to_numpy() for s in SEEDS}
            ens = np.mean([P[s] for s in SEEDS], axis=0)
            g = base[["role", "class10", "folder", "severity", "dev_fold"]].copy()
            g["ens_msp"] = ens.max(axis=1)
            g["pred"] = np.asarray(classes)[ens.argmax(axis=1)]
            g["ens_maha_rank"] = np.mean([ranks[s].loc[idx].to_numpy() for s in SEEDS], axis=0)
            g["disagreement"] = np.mean([js_divergence(P[a], P[b]) for a, b in combinations(SEEDS, 2)], axis=0)
            cal = g[g.role == "cal"]
            tp, ok_p = threshold(-cal.ens_msp.to_numpy())
            td, ok_d = threshold(cal.ens_maha_rank.to_numpy())
            conf = (g.ens_msp.to_numpy() >= -tp) & (g.ens_maha_rank.to_numpy() <= td)
            g["confident"] = conf
            g["decision"] = rule_decisions(conf, g.pred.to_numpy())
            g = g.reset_index().assign(withheld=c, outer_fold=k)
            g["outcome"] = outcomes(g.decision.to_numpy(), g)
            parts.append(g)
            thr_rows.append({"withheld": c, "outer_fold": k, "tau_p_ens": -tp, "tau_d_ens_rank": td, "n_cal": len(cal),
                             "attainable_p": ok_p, "attainable_d": ok_d,
                             "cal_joint_acceptance": float(conf[(g.role == "cal").to_numpy()].mean())})
    dec = pd.concat(parts, ignore_index=True)
    save("e1_ensemble", None, dec)
    OUT.mkdir(parents=True, exist_ok=True)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(OUT / "e1_thresholds.csv", index=False)
    w = dec[dec.role == "foldk_withheld"]
    print("unattainable targets:", int((~thr.attainable_p).sum() + (~thr.attainable_d).sum()),
          "| cal joint acceptance mean", round(thr.cal_joint_acceptance.mean(), 3))
    print("E1 withheld:", w.outcome.value_counts().to_dict())
    print("median disagreement by role:", dec.groupby("role").disagreement.median().round(4).to_dict())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["e1"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"e1": run_e1}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
