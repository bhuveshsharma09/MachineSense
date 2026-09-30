"""Stage R1 fit-normal re-score (CLAUDE.md §7, item 2): AE-6 and B1s scores of each Stage 3C run's healthy FIT normals,
which were never saved. No training, no threshold, nothing written to artifacts/ (weights are read only).

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/32_r1_fit_normal_scores.py

- AE-6: inference with the saved 3C weights and the run's saved AE normalisation, the 3C scoring (frame level, mean over
  channels, mean over the 9 windows) on the device 3C used. The run's saved cal/eval AE errors are re-scored only to check
  reproduction (max |diff| ≤ 1e-5 × τ_e, else the run is infeasible for AE).
- B1s: the Stage 3D refit (scripts/09c_loco_complementary.b1s_scores recipe), used only if it reproduces the saved cal/eval
  B1s scores within 1e-12 (else the run is infeasible for B1s; no approximation is substituted).
Output: results/research/r1_fit_normal_scores.parquet (fit normals only) and r1_reproduction.json (per-run evidence).
In-sample: these are the recordings each AE-6 was trained on and each B1s was fitted on (declared bias).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.openset import loco_run_ids, recording_mean
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, data_dir, results_dir
from ml.rejection import TIE_EPS
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEED = 42
VIB = [0, 1, 2, 3, 4, 5]
ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")
AE_REL_TOL = 1e-5
OUT = results_dir() / "research"
DECISIONS = results_dir() / "scores" / "cnn6ae6_s42_loco_decisions.parquet"
B1S_SAVED = results_dir() / "scores" / "b1s_nomic_loco_q3fit_scores.parquet"
THRESHOLDS = results_dir() / "loco" / "decision_thresholds.csv"


def setup():
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    return cfg, meta, pool, pd.read_csv(results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv")


def run_ids(assignment, pool, normals):
    """Per run: the fit normals (checked against the AE's saved fit_ids) and the saved-score recordings."""
    out = {}
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            ids = loco_run_ids(assignment, pool, c, k)
            fit_n = sorted(set(ids["fit"]) & normals)
            saved = json.loads(RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c).norm_stats(k).read_text())["fit_ids"]
            assert fit_n == sorted(saved), f"{c}/{k}: not the AE's fit normals"
            out[(c, k)] = {"fit_normals": fit_n, "scored": sorted(r for role in ROLES for r in ids[role])}
    return out


def b1s(cfg, meta, pool, runs) -> tuple[pd.DataFrame, dict]:
    """Stage 3D refit per run on the fit normals; fit-normal scores plus the reproduction check on the saved scores."""
    from ml.cache import load_features
    from ml.features import feature_columns
    from ml.q1 import classical_scores

    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = [c for c in feature_columns(df) if not c.startswith("f_mic_")]
    saved = pd.read_parquet(B1S_SAVED).set_index(["withheld", "outer_fold", "rec_id"]).b1s
    out, evidence = [], {}
    for (c, k), r in runs.items():
        fit = df[df.rec_id.isin(r["fit_normals"])]
        # the saved scores were computed with the saved recordings as the target set; the fit normals are scored separately
        rep = classical_scores("B1s_speed_poly", cfg.anomaly, fit, df[df.rec_id.isin(r["scored"])], cols)
        diff = float(np.max(np.abs(rep.to_numpy() - saved.loc[(c, k)].loc[rep.index].to_numpy())))
        ok = bool(diff <= TIE_EPS and len(rep) == len(saved.loc[(c, k)]))
        evidence[f"{c}/{k}"] = {"n_saved": int(len(rep)), "max_abs_diff": diff, "reproduced": ok}
        if ok:
            fs = classical_scores("B1s_speed_poly", cfg.anomaly, fit, fit, cols)
            out.append(pd.DataFrame({"rec_id": fs.index, "b1s": fs.to_numpy(), "withheld": c, "outer_fold": k}))
        print(f"  B1s {c}/{k}: max |diff| {diff:.2e} {'ok' if ok else 'INFEASIBLE'}", flush=True)
    return pd.concat(out, ignore_index=True), evidence


def ae(cfg, meta, pool, runs) -> tuple[pd.DataFrame, dict]:
    """AE-6 inference with the saved 3C weights; fit-normal errors plus the reproduction check on the saved errors."""
    import torch

    from ml.config import AeConfig
    from ml.datasets import dev_window_rows, load_representation
    from ml.models.ae import build_ae
    from ml.train import gpu_job, release_device_memory
    from ml.train_ae import window_channel_errors
    from ml.utils import get_device

    device = get_device()
    ac = AeConfig(**cfg.ae.model_dump())
    saved = pd.read_parquet(DECISIONS).set_index(["withheld", "outer_fold", "rec_id"]).ae
    tau_e = pd.read_csv(THRESHOLDS).set_index(["withheld", "outer_fold"]).tau_e
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    out, evidence = [], {}
    with gpu_job(device, artifacts_dir(), cfg.cnn.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # development-pool windows only
        del x
        rec_of = rows.rec_id.to_numpy()
        for (c, k), r in runs.items():
            ra = RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c)
            na = json.loads(ra.norm_stats(k).read_text())
            model = build_ae("frame", len(VIB), cfg.ae)
            model.load_state_dict(torch.load(ra.weights(k), map_location=device, weights_only=True))
            model.to(device)

            def score(ids):
                sel = np.flatnonzero(np.isin(rec_of, ids))
                err = window_channel_errors(model, cache, sel, VIB, np.asarray(na["mean"]), np.asarray(na["std"]), "frame",
                                            ac, device).mean(axis=1)
                return recording_mean(err, rec_of[sel])

            rep = score(r["scored"])
            ref = saved.loc[(c, k)]
            diff = float(np.max(np.abs(rep.loc[ref.index].to_numpy() - ref.to_numpy())))
            tol = AE_REL_TOL * float(tau_e.loc[(c, k)])
            ok = bool(diff <= tol and len(rep) == len(ref))
            evidence[f"{c}/{k}"] = {"n_saved": int(len(ref)), "max_abs_diff": diff, "tolerance": tol, "reproduced": ok}
            if ok:
                fs = score(r["fit_normals"])
                out.append(pd.DataFrame({"rec_id": fs.index, "ae": fs.to_numpy(), "withheld": c, "outer_fold": k}))
            print(f"  AE-6 {c}/{k}: max |diff| {diff:.2e} (tol {tol:.1e}) {'ok' if ok else 'INFEASIBLE'}", flush=True)
            del model
        del cache
        release_device_memory(device)
    return pd.concat(out, ignore_index=True), evidence


def main() -> int:
    cfg, meta, pool, assignment = setup()
    normals = set(pool.loc[pool.class10 == "normal", "rec_id"])
    runs = run_ids(assignment, pool, normals)
    OUT.mkdir(parents=True, exist_ok=True)
    b, ev_b = b1s(cfg, meta, pool, runs)
    a, ev_a = ae(cfg, meta, pool, runs)
    key = ["withheld", "outer_fold", "rec_id"]
    fit = pd.DataFrame([{"withheld": c, "outer_fold": k, "rec_id": rid} for (c, k), r in runs.items() for rid in r["fit_normals"]])
    fit = fit.merge(a, on=key, how="left").merge(b, on=key, how="left")
    assert fit.rec_id.isin(normals).all()
    fit["role"] = "fit"
    fit["in_sample"] = True
    fit.to_parquet(OUT / "r1_fit_normal_scores.parquet", index=False)
    (OUT / "r1_reproduction.json").write_text(json.dumps({
        "stage": "R1", "what": "reproduction of the saved Stage 3C (AE-6) and Stage 3D (B1s) cal/eval scores before the "
        "fit-normal scores are used", "b1s_tolerance": TIE_EPS, "ae_tolerance_rel_tau_e": AE_REL_TOL,
        "device": str(__import__("ml.utils", fromlist=["get_device"]).get_device()), "git_sha": git_sha(),
        "split_hash": split_hash(meta), "b1s": ev_b, "ae": ev_a,
        "infeasible": {"b1s": [r for r, e in ev_b.items() if not e["reproduced"]],
                       "ae": [r for r, e in ev_a.items() if not e["reproduced"]]}}, indent=1))
    print(f"{len(fit)} fit-normal rows; infeasible B1s {sum(not e['reproduced'] for e in ev_b.values())}, "
          f"AE {sum(not e['reproduced'] for e in ev_a.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
