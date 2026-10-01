"""Stage R6: train the autoencoder arms (A0–A3 healthy-only, F fault-only) on the Q1 development folds, seeds 42 / 43 / 44,
and save per-recording scores. Never imports LightGBM. Protocol: CLAUDE.md §7 (Stage R6), declared before training.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/36_r6_train.py            # all arms and seeds; resumable

Only training and scoring live here (≈ 50 min, too long for a notebook cell); every analysis, table, figure and results
file is produced by notebooks/06_research_layer/16_ae_experiments.ipynb from the saved scores.
Outputs: results/scores/r6_<arm>_s<seed>_dev_scores.parquet (one row per scored recording and fold: rec_id, outer_fold,
role cal|test, error, level), results/research/r6_training.json (per model: recordings, views, units, steps, seconds,
noise floor, mismatch SD), weights artifacts/r6/<arm>_s<seed>/fold<k>.pt (gitignored).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ml.ae_experiments import (make_views, mismatch_sd, n_views_for, noise_floor_db, recording_level, train_ae_arm)
from ml.config import AeConfig, load_config
from ml.datasets import dev_window_rows, load_representation, norm_stats
from ml.openset import recording_mean
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, data_dir, results_dir
from ml.q1 import fold_ids, load_assignment
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

SEEDS = (42, 43, 44)
ARMS = ("A0", "A1", "A2", "A3", "F")
VIB = [0, 1, 2, 3, 4, 5]
ASSIGNMENT = results_dir() / "q1q3" / "q1_fit_cal_assignment.csv"
INFO = results_dir() / "research" / "r6_training.json"
GUARD_MIN = 60.0


def scores_path(arm: str, seed: int):
    return results_dir() / "scores" / f"r6_{arm}_s{seed}_dev_scores.parquet"


def main() -> int:
    import torch

    from ml.models.ae import N_FRAMES
    from ml.train import gpu_job, release_device_memory
    from ml.train_ae import window_channel_errors
    from ml.utils import get_device, seed_everything

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    assignment = load_assignment(ASSIGNMENT)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    rec_of = rows.rec_id.to_numpy()
    device = get_device()
    todo = [(a, s) for s in SEEDS for a in ARMS if not scores_path(a, s).exists()]
    print(f"{len(todo)} arm × seed runs to do", flush=True)
    if not todo:
        return 0
    info = json.loads(INFO.read_text()) if INFO.exists() else {"models": [], "split_hash": split_hash(meta)}
    t_start, t_first_f = time.perf_counter(), None
    with gpu_job(device, artifacts_dir(), cfg.ae.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        level = recording_level(x[:, VIB], rec_of)          # mean cached dB level per recording (vibration channels)
        cache = torch.from_numpy(x).to(device)              # development-pool windows only
        for arm, seed in todo:
            ac = AeConfig(**{**cfg.ae.model_dump(), "seed": seed})
            tables = []
            for k in sorted(assignment):
                fit_n, cal, ev, ctx = fold_ids(assignment, pool, k)
                fit = (sorted(pool.loc[(pool.dev_fold != k) & (pool.class10 != "normal"), "rec_id"]) if arm == "F" else fit_n)
                st = norm_stats(arr, index, meta, fit_ids=fit, context=ctx)
                mean, std = st["mean"][VIB], st["std"][VIB]
                fit_rows = np.flatnonzero(np.isin(rec_of, fit))
                codes = {r: i for i, r in enumerate(fit)}
                rec_code = np.array([codes[r] for r in rec_of[fit_rows]])
                fit_vals = x[fit_rows][:, VIB].astype(np.float32)
                floor = noise_floor_db(fit_vals, 5.0)
                ch_levels = np.stack([fit_vals[rec_code == i].mean(axis=(0, 2, 3)) for i in range(len(fit))])
                mm = mismatch_sd(ch_levels)
                views = make_views(len(fit), n_views_for(len(fit)), len(VIB), ac.gain_jitter_db, seed * 100 + k) \
                    if arm == "A1" else None
                seed_everything(seed)
                t0 = time.perf_counter()
                model, hist, inf = train_ae_arm(arm, cache, fit_rows, rec_code, VIB, mean, std, ac, device, views=views,
                                                floor_db=floor, mm_sd=mm)
                if device.type == "mps":
                    torch.mps.synchronize()
                secs = time.perf_counter() - t0
                for role, ids in (("cal", cal), ("test", ev)):
                    sel = np.flatnonzero(np.isin(rec_of, ids))
                    e = window_channel_errors(model, cache, sel, VIB, mean, std, "frame", ac, device).mean(axis=1)
                    r = recording_mean(e, rec_of[sel])
                    tables.append(pd.DataFrame({"rec_id": r.index, "error": r.to_numpy(), "role": role, "outer_fold": k,
                                                "arm": arm, "seed": seed}))
                w = artifacts_dir() / "r6" / f"{arm}_s{seed}" / f"fold{k}.pt"
                w.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), w)
                info["models"].append({**inf, "seed": seed, "outer_fold": int(k), "train_seconds": secs,
                                       "n_fit_recordings": len(fit), "n_calibration_normals": len(cal),
                                       "noise_floor_db": floor.tolist(), "mismatch_sd_db": mm,
                                       "final_train_loss": hist[-1]["train_loss"],
                                       "parameters": int(sum(p.numel() for p in model.parameters()))})
                print(f"  {arm} s{seed} fold{k}: {len(fit)} fit recordings ({inf['n_views']} views, {inf['training_units']} "
                      f"units, {inf['optimizer_steps']} steps), {secs:.0f} s, loss {hist[-1]['train_loss']:.4f}", flush=True)
                del model
                release_device_memory(device)
                if arm == "F" and t_first_f is None:
                    t_first_f = secs
                    n_f = sum(1 for a, _ in todo if a == "F") * len(assignment)
                    proj = (time.perf_counter() - t_start + secs * (n_f - 1)) / 60
                    print(f"  TIME ESTIMATE after the first fault-AE fold: {proj:.0f} min in total (guard {GUARD_MIN:.0f})",
                          flush=True)
                    info["time_estimate_minutes"] = proj
                    if proj > GUARD_MIN:
                        INFO.write_text(json.dumps(info, indent=1))
                        raise SystemExit(f"TIME GUARD: projected {proj:.0f} min > {GUARD_MIN:.0f} min; stopping (§10)")
            out = pd.concat(tables, ignore_index=True)
            out["level"] = out.rec_id.map(level)
            out.to_parquet(scores_path(arm, seed), index=False)
            info.update({"git_sha": git_sha(), "device": str(device), "frames_per_window": N_FRAMES})
            INFO.parent.mkdir(parents=True, exist_ok=True)
            INFO.write_text(json.dumps(info, indent=1))
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
