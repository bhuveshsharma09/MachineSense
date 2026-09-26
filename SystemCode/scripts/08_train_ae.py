"""Healthy-only autoencoders (Q1, Stage 3B) on cache C, under the corrected Q1 protocol of Stage 2D.1. Never imports
LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/08_train_ae.py train --variant ae7   # ae6, wae7, wae6
    python scripts/08_train_ae.py compare

Per outer dev fold k, with the SAVED healthy fit/calibration assignment (results/q1q3/q1_fit_cal_assignment.csv):
- normalisation: norm_stats in development mode, fit_ids = the FIT normals of folds ≠ k (context: calibration normals and
  fold k excluded);
- training on the FIT normals' windows only (frame level: every frame of those windows), a fixed number of epochs;
- threshold = maximum score of the CALIBRATION normals under the same fitted model (higher = more anomalous; alarm iff
  score > threshold; ties do not alarm);
- fold k scored once (25 held-out normals over the folds; every fault recording once).
Variants: ae7 / ae6 = frame-level AE with / without the microphone (PRIMARY); wae7 / wae6 = window-level AE (secondary).
Outputs (ml.paths.RunPaths, name <variant>_s<seed>_dev): results/scores/<name>_channel_errors.parquet (per recording:
rec_id, outer_fold, dev_fold, role cal|test, score, per-channel errors, variant, seed, run_id); artifacts/ae/<name>/
fold<k>.pt; results/ae/<name>/fold<k>_norm.json; results/ae/<name>_eval.json; MLflow runs <name>, <name>_fold<k>.
`compare`: within-fold comparison with B1s on identical recordings and the sensor-anomaly-evidence cross-tab
(results/ae/ae_vs_b1s.json, results/ae/sensor_evidence_<variant>.csv).
"""
from __future__ import annotations

import argparse
import hashlib
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
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, data_dir, results_dir
from ml.q1 import FAULTS, FoldScores, fold_ids, load_assignment, summarise
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

CHANNELS7 = ["under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential", "mic"]
VARIANTS = {"ae7": ("frame", 7), "ae6": ("frame", 6), "wae7": ("window", 7), "wae6": ("window", 6)}
B1S = {"ae7": "b1s_all", "ae6": "b1s_nomic", "wae7": "b1s_all", "wae6": "b1s_nomic"}  # same sensor set
OUT = results_dir() / "ae"
ASSIGNMENT = results_dir() / "q1q3" / "q1_fit_cal_assignment.csv"
WORDING = ("the channel with the largest reconstruction error is the one the autoencoder reconstructs least well; it is "
           "not proof of the fault's physical location, nor an explanation of the classifier's decision")


def load_pool(cfg: Config, meta: pd.DataFrame) -> pd.DataFrame:
    dc = cfg.development
    folds = pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]]
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(folds, on="rec_id")
    assert len(pool) == len(folds)
    return pool


def rec_table(pool: pd.DataFrame) -> pd.DataFrame:
    return pool.set_index("rec_id")[["class10", "fault_family", "severity", "folder", "fr_measured_hz", "dev_fold"]]


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def recording_errors(rows: pd.DataFrame, idx: np.ndarray, err: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-recording per-channel error (mean over its windows = over all its frames) and the total score (equal-weight
    mean over channels)."""
    df = pd.DataFrame(err, columns=[f"err_{c}" for c in names])
    df["rec_id"] = rows.rec_id.to_numpy()[idx]
    out = df.groupby("rec_id", sort=False).mean()
    out.insert(0, "score", out[[f"err_{c}" for c in names]].mean(axis=1))
    return out


def run_train(cfg: Config, variant: str, seed: int) -> None:
    import mlflow
    import torch

    from ml.models.ae import build_ae
    from ml.models.cnn import n_parameters
    from ml.train import gpu_job, release_device_memory
    from ml.train_ae import train_ae, window_channel_errors
    from ml.utils import get_device, mlflow_run, seed_everything

    level, n_ch = VARIANTS[variant]
    ac = cfg.ae.model_copy(update={"seed": seed})
    device = get_device()
    rp = RunPaths("ae", variant, seed)
    channels, names = list(range(n_ch)), CHANNELS7[:n_ch]
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = load_pool(cfg, meta)
    rec = rec_table(pool)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    assignment = load_assignment(ASSIGNMENT)
    cache_key = json.loads(next((data_dir() / "representations").glob(
        f"{cfg.representation.leading_candidate}_*.json")).read_text())["cache_key"]
    n_params = n_parameters(build_ae(level, n_ch, ac))
    if level == "frame" and n_params >= ac.max_params:
        raise ValueError(f"frame AE has {n_params} parameters ≥ {ac.max_params}")
    for p in (rp.weights(0), rp.norm_stats(0), rp.result("eval"), rp.scores("channel_errors")):
        p.parent.mkdir(parents=True, exist_ok=True)
    folds, tables, fold_info = [], [], []
    with gpu_job(device, artifacts_dir(), ac.mps_memory_fraction), \
            mlflow_run(cfg, run_name=rp.run_name()) as parent:
        mlflow.set_tags({"stage": "3B", "question": "Q1", **rp.tags(), "level": level, "split_hash": split_hash(meta),
                         "cache_key": cache_key, "protocol": "corrected Q1 (Stage 2D.1), saved fit/cal assignment"})
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # dev-pool windows only, float16 on the device
        del x
        for k in sorted(assignment):
            fit, cal, ev, ctx = fold_ids(assignment, pool, k)
            st = norm_stats(arr, index, meta, fit_ids=fit, context=ctx)
            mean, std = st["mean"][channels], st["std"][channels]
            rp.norm_stats(k).write_text(json.dumps({"fit_ids": fit, "fit_ids_sha256": ids_hash(fit),
                                                     "n_windows": st["n_windows"], "channels": names,
                                                     "mean": mean.tolist(), "std": std.tolist()}, indent=1))
            fit_rows = np.flatnonzero(rows.rec_id.isin(fit).to_numpy())
            cal_rows = np.flatnonzero(rows.rec_id.isin(cal).to_numpy())
            ev_rows = np.flatnonzero(rows.rec_id.isin(ev).to_numpy())
            run_id = rp.run_name(k)
            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": "3B", "question": "Q1", **rp.tags(k), "level": level,
                                 "fit_ids_sha256": ids_hash(fit), "split_hash": split_hash(meta), "cache_key": cache_key,
                                 "git_sha": git_sha()})
                seed_everything(seed)
                t0 = time.perf_counter()
                model, hist = train_ae(cache, fit_rows, channels, mean, std, level, ac, device,
                                       on_epoch=lambda ep, r: mlflow.log_metric("train_loss", r["train_loss"], step=ep))
                if device.type == "mps":
                    torch.mps.synchronize()
                train_s = time.perf_counter() - t0
                e_cal = recording_errors(rows, cal_rows, window_channel_errors(
                    model, cache, cal_rows, channels, mean, std, level, ac, device), names)
                e_ev = recording_errors(rows, ev_rows, window_channel_errors(
                    model, cache, ev_rows, channels, mean, std, level, ac, device), names)
                torch.save(model.state_dict(), rp.weights(k))
                fs = FoldScores(k, len(fit), e_cal.score, e_ev.score)
                folds.append(fs)
                thr = float(e_cal.score.max())
                for role, e in (("cal", e_cal), ("test", e_ev)):
                    tables.append(e.reset_index().assign(outer_fold=k, role=role, threshold=thr,
                                                         alarm=e.score.to_numpy() > thr, variant=variant, seed=seed,
                                                         run_id=run_id))
                auc = roc_auc_score((rec.loc[e_ev.index].class10 != "normal").to_numpy(), e_ev.score.to_numpy())
                mlflow.log_metrics({"train_seconds": train_s, "auroc_overall": auc, "final_train_loss": hist[-1]["train_loss"],
                                    "threshold": thr, "n_cal": len(cal), "n_fit": len(fit)})
                print(f"  {run_id}: fit {len(fit)} recordings ({len(fit_rows)} windows), cal {len(cal)}, {train_s:.0f} s, "
                      f"final loss {hist[-1]['train_loss']:.4f}, AUROC overall {auc:.3f}")
                fold_info.append({"fold": int(k), "run_id": run_id, "n_fit": len(fit), "n_cal": len(cal),
                                  "fit_ids_sha256": ids_hash(fit), "n_fit_windows": int(len(fit_rows)),
                                  "n_train_units": int(len(fit_rows) * (25 if level == "frame" else 1)),
                                  "train_seconds": train_s, "history": hist})
            release_device_memory(device)
        res, detail = summarise(folds, rec, cfg.anomaly)
        tab = pd.concat(tables, ignore_index=True).merge(rec[["class10", "folder", "severity", "fault_family"]],
                                                          left_on="rec_id", right_index=True)
        tab["dev_fold"] = rec.loc[tab.rec_id, "dev_fold"].to_numpy()
        test = tab[tab.role == "test"]
        assert test.rec_id.is_unique and len(test) == len(pool)
        tab.to_parquet(rp.scores("channel_errors"), index=False)
        out = {"provenance": {"variant": variant, "level": level, "seed": seed, "channels": names,
                              "n_parameters": n_params, "config": ac.model_dump(mode="json"), "device": str(device),
                              "split_hash": split_hash(meta), "cache_key": cache_key, "git_sha": git_sha(),
                              "mlflow_run_id": parent.info.run_id,
                              "assignment": str(ASSIGNMENT.relative_to(SYSTEM_CODE_DIR)),
                              "healthy_caveat": "25 held-out healthy recordings from ONE folder over 3 folds",
                              "error": "squared error in normalised units (healthy FIT stats); per-channel mean over bins "
                                       "and frames; total = equal-weight channel mean; recording = mean over all frames "
                                       "of its 9 windows", "reconstruction_target": ac.reconstruct},
               "result": res, "folds": fold_info,
               "train_seconds_total": float(sum(f["train_seconds"] for f in fold_info))}
        rp.result("eval").write_text(json.dumps(out, indent=1))
        mlflow.log_metrics({"auroc_fold_mean": res["auroc_fold_mean"]["overall"], "false_alarms": res["false_alarms_total"],
                            "detection_overall": res["detection_overall"]})
        mlflow.log_artifact(str(rp.result("eval")))
    print(f"{rp.name}: {n_params} parameters; AUROC {res['auroc_fold_mean']['overall']:.3f} ± "
          f"{res['auroc_fold_std']['overall']:.3f}; FA {res['false_alarms_total']}/{res['heldout_normals_total']} "
          f"(acceptance {res['measured_acceptance_total']:.2f}); detection {res['detection_overall']:.3f}")


def run_compare(cfg: Config, seed: int) -> None:
    """Within-fold comparison with B1s on identical recordings (the basis for conclusions); pooled raw-score AUROC as a
    supplement; sensor anomaly evidence cross-tab."""
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = load_pool(cfg, meta)
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "seed": seed,
                          "basis": "within-fold comparisons; pooled raw-score AUROC is supplementary because different "
                                   "fold models have different score scales",
                          "healthy_caveat": "25 held-out healthy recordings from ONE folder; B1s per-fold AUROC 0.990 "
                                            "leaves little room", "sensor_evidence_wording": WORDING},
           "detectors": {}, "differences": {}, "sensor_evidence": {}}
    scores = {}
    for v in VARIANTS:
        rp = RunPaths("ae", v, seed)
        if not rp.result("eval").exists():
            continue
        scores[v] = pd.read_parquet(rp.scores("channel_errors"))
        out["detectors"][v] = json.loads(rp.result("eval").read_text())["result"]
    for b in sorted(set(B1S.values())):
        rp = RunPaths("q1q3", b, cfg.anomaly.seed)
        scores[b] = pd.read_parquet(rp.scores())
        out["detectors"][b] = json.loads(rp.result("eval").read_text())["result"]
    for name, s in scores.items():
        t = s[s.role == "test"]
        for k, g in t.groupby("outer_fold"):
            ref = scores["b1s_all"]
            ref_ids = set(ref[(ref.role == "test") & (ref.outer_fold == k)].rec_id)
            assert set(g.rec_id) == ref_ids, f"{name} fold {k}: not the same recordings as B1s"
    for v in [v for v in VARIANTS if v in out["detectors"]]:
        b = B1S[v]
        a_f, b_f = out["detectors"][v]["per_fold"], out["detectors"][b]["per_fold"]
        diffs = {c: [fa["auroc"][c] - fb["auroc"][c] for fa, fb in zip(a_f, b_f)] for c in ["overall", *FAULTS]}
        out["differences"][f"{v}_minus_{b}"] = {"per_fold_auroc_diff": diffs,
                                               "mean_overall": float(np.mean(diffs["overall"])),
                                               "folds_ae_better_overall": int(sum(d > 0 for d in diffs["overall"])),
                                               "folds_equal_overall": int(sum(d == 0 for d in diffs["overall"]))}
    for a, b in (("ae7", "ae6"), ("wae7", "wae6"), ("wae7", "ae7")):
        if a in out["detectors"] and b in out["detectors"]:
            out["differences"][f"{a}_minus_{b}"] = {"per_fold_auroc_diff": {
                "overall": [x["auroc"]["overall"] - y["auroc"]["overall"]
                            for x, y in zip(out["detectors"][a]["per_fold"], out["detectors"][b]["per_fold"])]}}
    pooled = {}
    for name, s in scores.items():
        t = s[s.role == "test"]
        pooled[name] = float(roc_auc_score((t.class10 != "normal").to_numpy(), t.score.to_numpy()))
    out["pooled_raw_score_auroc_supplementary"] = pooled
    OUT.mkdir(parents=True, exist_ok=True)
    for v in [v for v in VARIANTS if v in scores]:
        s = scores[v]
        err_cols = [c for c in s.columns if c.startswith("err_")]
        t = s[(s.role == "test") & (s.class10 != "normal")].copy()
        t["worst_channel"] = t[err_cols].idxmax(axis=1).str.removeprefix("err_")
        # supplementary: error relative to the same fold model's calibration-healthy median per channel
        cal_med = s[s.role == "cal"].groupby("outer_fold")[err_cols].median()
        rel = t[err_cols].to_numpy() / cal_med.loc[t.outer_fold].to_numpy()
        t["worst_channel_relative"] = np.array([c.removeprefix("err_") for c in err_cols])[rel.argmax(axis=1)]
        names = [c.removeprefix("err_") for c in err_cols]
        ct = pd.crosstab(t.class10, t.worst_channel).reindex(index=FAULTS, columns=names, fill_value=0)
        ctr = pd.crosstab(t.class10, t.worst_channel_relative).reindex(index=FAULTS, columns=names, fill_value=0)
        ct.to_csv(OUT / f"sensor_evidence_{v}_s{seed}.csv")
        ctr.to_csv(OUT / f"sensor_evidence_relative_{v}_s{seed}.csv")
        out["sensor_evidence"][v] = {"absolute": ct.to_dict(orient="index"), "relative_to_cal_healthy": ctr.to_dict(orient="index")}
    (OUT / f"ae_vs_b1s_s{seed}.json").write_text(json.dumps(out, indent=1))
    rows = []
    for name, d in out["detectors"].items():
        rows.append({"detector": name, **{f"fold{f['fold']}": round(f["auroc"]["overall"], 4) for f in d["per_fold"]},
                     "mean": round(d["auroc_fold_mean"]["overall"], 4), "std": round(d["auroc_fold_std"]["overall"], 4),
                     "FA/25": f"{d['false_alarms_total']}/{d['heldout_normals_total']}",
                     "acceptance": round(d["measured_acceptance_total"], 2), "detection": round(d["detection_overall"], 3),
                     "pooled raw AUROC (suppl.)": round(pooled[name], 4)})
    print(pd.DataFrame(rows).set_index("detector").to_string())
    for k, v in out["differences"].items():
        print(f"  {k:22s} per-fold overall AUROC diff {np.round(v['per_fold_auroc_diff']['overall'], 4).tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["train", "compare"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="ae7")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg.ae.seed if a.seed is None else a.seed
    {"train": lambda: run_train(cfg, a.variant, seed), "compare": lambda: run_compare(cfg, seed)}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
