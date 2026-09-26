"""Reproduce the speed-aware classical detector (B1s polynomial) under the corrected Q1 protocol from the SAVED Stage
2D.1 healthy fit/calibration assignment, check that its per-fold results match results/q1q3/q1_corrected.json exactly,
and save its recording-level scores so the autoencoder can be compared on identical recordings.

    python scripts/11b_b1s_scores.py

Variants: b1s_all (all capped R1 features) and b1s_nomic (no microphone features), as in Stage 2D.1. B1s itself has no
training seed; the path's seed is the assignment seed (anomaly.seed). Outputs:
results/scores/<variant>_s<seed>_dev_scores.parquet (rec_id, outer_fold, dev_fold, role cal|test, score, threshold,
alarm, variant, seed, run_id) and results/q1q3/<variant>_s<seed>_dev_eval.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml.cache import load_features
from ml.config import load_config
from ml.features import feature_columns
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, data_dir, results_dir
from ml.protocol import max_threshold, speed_stratified_split
from ml.q1 import FoldScores, classical_scores, load_assignment, summarise
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

RUNG = "B1s_speed_poly"
Q1 = results_dir() / "q1q3"


def diff(a, b, path="") -> list[str]:
    """Every leaf where a and b differ (exact equality)."""
    if isinstance(a, dict):
        return [x for k in sorted(set(a) | set(b)) for x in diff(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    ac, pre, dc = cfg.anomaly, cfg.preprocess, cfg.development
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, pre).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    rec = pool.set_index("rec_id")[["class10", "fault_family", "severity", "folder", "fr_measured_hz", "dev_fold"]]
    splits = load_assignment(Q1 / "q1_fit_cal_assignment.csv")
    normals = pool[pool.class10 == "normal"]
    regenerated = all(speed_stratified_split(normals[normals.dev_fold != k], {"fit": 2, "cal": 1}, seed=ac.seed + k)
                      .sort_index().equals(splits[k].sort_index()) for k in splits)
    ref = json.loads((Q1 / "q1_corrected.json").read_text())["rungs"][RUNG]
    all_cols = feature_columns(df)
    variants = {"b1s_all": ("all", all_cols), "b1s_nomic": ("no_mic", [c for c in all_cols if not c.startswith("f_mic_")])}
    report = {}
    for variant, (ref_key, cols) in variants.items():
        rp = RunPaths("q1q3", variant, ac.seed)
        folds, rows = [], []
        for k in sorted(splits):
            normals_tr = df[(df.class10 == "normal") & (df.dev_fold != k)]
            part = splits[k]
            fit = normals_tr[normals_tr.rec_id.isin(part[part == "fit"].index)]
            cal = normals_tr[normals_tr.rec_id.isin(part[part == "cal"].index)]
            fs = FoldScores(k, fit.rec_id.nunique(), classical_scores(RUNG, ac, fit, cal, cols),
                            classical_scores(RUNG, ac, fit, df[df.dev_fold == k], cols))
            folds.append(fs)
            thr = max_threshold(fs.cal.to_numpy())
            for role, s in (("cal", fs.cal), ("test", fs.test)):
                rows.append(pd.DataFrame({"rec_id": s.index, "outer_fold": k, "dev_fold": rec.loc[s.index, "dev_fold"].to_numpy(),
                                          "role": role, "score": s.to_numpy(), "threshold": thr,
                                          "alarm": s.to_numpy() > thr, "variant": variant, "seed": ac.seed,
                                          "run_id": rp.run_name(k)}))
        res, _ = summarise(folds, rec, ac)
        mismatches = diff(res, ref[ref_key])
        out = pd.concat(rows, ignore_index=True)
        out = out.merge(rec[["class10", "folder", "severity", "fault_family"]], left_on="rec_id", right_index=True)
        rp.scores().parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(rp.scores(), index=False)
        test = out[out.role == "test"]
        assert test.rec_id.is_unique and len(test) == len(pool)  # every pool recording scored once, by its own fold
        report[variant] = {"reproduced_exactly": not mismatches, "n_mismatches": len(mismatches),
                           "mismatches": mismatches[:20], "scores_path": str(rp.scores().relative_to(SYSTEM_CODE_DIR)),
                           "per_fold_auroc": [f["auroc"]["overall"] for f in res["per_fold"]],
                           "false_alarms_total": res["false_alarms_total"], "result": res}
        rp.result("eval").write_text(json.dumps({"provenance": {"rung": RUNG, "variant": variant, "seed": ac.seed,
                                                                "feature_table": path.name, "split_hash": split_hash(meta),
                                                                "git_sha": git_sha(),
                                                                "assignment": "results/q1q3/q1_fit_cal_assignment.csv (saved, Stage 2D.1)",
                                                                "assignment_regenerates_identically": bool(regenerated)},
                                                 **report[variant]}, indent=1))
        print(f"{variant}: reproduced exactly = {not mismatches} ({len(mismatches)} differing values); per-fold AUROC "
              f"{np.round(report[variant]['per_fold_auroc'], 4).tolist()}; FA {res['false_alarms_total']}/25 "
              f"→ {rp.scores().relative_to(SYSTEM_CODE_DIR)}")
        for m in mismatches[:10]:
            print("   ", m)
    print(f"saved assignment regenerates identically from the seed: {regenerated}")
    return 0 if all(r["reproduced_exactly"] for r in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
