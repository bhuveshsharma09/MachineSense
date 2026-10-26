"""Stage R10: frozen ImageNet ResNet-18 window embeddings for the saved LOCO runs. Forward passes only, no training.
Protocol: CLAUDE.md §7 (Stage R10), declared before any code. Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps MACHINESENSE_NO_RESERVED_READ=1 python scripts/38_frozen_embeddings.py

For each (withheld class, outer fold), every window of the saved CNN-6 window table (all roles) is normalised exactly as
the Stage 3C run normalised it and embedded by ml.frozen_embed (per-bearing packing, 1,024-d). The 3C normalisation and
window table do not depend on the seed (item 12), so there are 27 passes; the script re-checks both before each pass and
stops if either differs. The first pass is timed and the total projected; above 60 minutes the script stops (§10).
Outputs: results/scores/r18frozen_w-<c>_loco_fold<k>_embeddings.npz (float32, gitignored) and
results/research/r10_embedding_run.json (timing, device, weights sha256, the seed-independence evidence).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.datasets import dev_window_rows, load_representation
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, data_dir, results_dir
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEEDS = (42, 43, 44)
VIB = [0, 1, 2, 3, 4, 5]
BATCH = 128          # windows per batch (two 224 × 224 images each)
GUARD_MIN = 60.0     # §10: stop and ask above one hour
SC = results_dir() / "scores"


def seed_independent(c: str, k: int) -> dict:
    """The fold's 3C normalisation and window table are identical across the three seeds (else the stage stops)."""
    norms = [json.loads((results_dir() / "loco" / f"cnn6_w-{c}_s{s}_loco" / f"fold{k}_norm.json").read_text()) for s in SEEDS]
    shas = {hashlib.sha256(json.dumps(n, sort_keys=True).encode()).hexdigest() for n in norms}
    tabs = [pd.read_parquet(SC / f"cnn6_w-{c}_s{s}_loco_fold{k}_logits.parquet", columns=["rec_id", "window_idx", "role", "class10"])
            .reset_index(drop=True) for s in SEEDS]
    return {"norm_identical": len(shas) == 1, "table_identical": all(tabs[0].equals(t) for t in tabs[1:]),
            "norm": norms[0], "table": tabs[0]}


def main() -> int:
    import torch

    from ml.frozen_embed import frozen_embed, load_frozen_resnet
    from ml.models.resnet import load_resnet_config, verify_weights
    from ml.train import gpu_job, normalise, to_device_f32
    from ml.utils import get_device

    cfg = load_config(DEFAULT_CONFIG)
    rcfg = load_resnet_config()
    sha = verify_weights(rcfg)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    pos_of = {(r, int(w)): i for i, (r, w) in enumerate(zip(rows.rec_id, rows.window_idx))}
    device = get_device()
    model = load_frozen_resnet(rcfg).to(device)
    torch.manual_seed(42)
    rep = {"stage": "R10", "device": str(device), "git_sha": git_sha(), "split_hash": split_hash(meta), "weights_sha256": sha,
           "batch_windows": BATCH, "passes": [], "seed_independence": []}
    out_json = results_dir() / "research" / "r10_embedding_run.json"
    jobs = [(c, k) for c in FAULT_CLASSES for k in (0, 1, 2)]
    t_all = time.perf_counter()
    with gpu_job(device, artifacts_dir(), rcfg.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)
        del x
        ch = torch.tensor(VIB, device=device)
        total_windows = None
        for j, (c, k) in enumerate(jobs):
            si = seed_independent(c, k)
            rep["seed_independence"].append({"withheld": c, "fold": k, "norm_identical": si["norm_identical"],
                                             "table_identical": si["table_identical"]})
            if not (si["norm_identical"] and si["table_identical"]):
                rep["stopped"] = f"{c}/{k}: normalisation or window table differs between seeds (item 3: 81 passes needed)"
                out_json.write_text(json.dumps(rep, indent=1))
                print("STOP:", rep["stopped"])
                return 3
            if total_windows is None:
                total_windows = sum(len(seed_independent(cc, kk)["table"]) for cc, kk in jobs)
            tab, norm = si["table"], si["norm"]
            mu, sd = to_device_f32(np.asarray(norm["mean"]), device), to_device_f32(np.asarray(norm["std"]), device)
            idx = torch.as_tensor(np.array([pos_of[(r, int(w))] for r, w in zip(tab.rec_id, tab.window_idx)]), dtype=torch.long)
            t0 = time.perf_counter()
            embs = []
            for i in range(0, len(idx), BATCH):
                xb = normalise(cache[idx[i:i + BATCH].to(device)][:, ch].to(torch.float32), mu, sd)
                embs.append(frozen_embed(model, xb).float().cpu().numpy())
            if device.type == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            emb = np.concatenate(embs).astype(np.float32)
            np.savez(SC / f"r18frozen_w-{c}_loco_fold{k}_embeddings.npz", embedding=emb, rec_id=tab.rec_id.to_numpy().astype(str),
                     window_idx=tab.window_idx.to_numpy(), role=tab.role.to_numpy().astype(str))
            rep["passes"].append({"withheld": c, "fold": k, "windows": int(len(tab)), "seconds": round(dt, 2)})
            print(f"  {c}/{k}: {len(tab)} windows in {dt:.1f} s", flush=True)
            if j == 0:
                proj = dt / len(tab) * total_windows / 60
                rep["first_pass_seconds"], rep["projected_total_minutes"] = round(dt, 2), round(proj, 2)
                print(f"projected total: {proj:.1f} min for {total_windows} windows", flush=True)
                if proj > GUARD_MIN:
                    rep["stopped"] = f"projected {proj:.1f} min > {GUARD_MIN} min (§10): stop and ask"
                    out_json.write_text(json.dumps(rep, indent=1))
                    print("STOP:", rep["stopped"])
                    return 4
            if device.type == "mps":
                torch.mps.empty_cache()
    rep["total_seconds"] = round(time.perf_counter() - t_all, 1)
    rep["finding"] = ("the 3C normalisation and window tables are identical across seeds 42, 43, 44 for all 27 (class, fold) "
                      "pairs: 27 passes, reused for every seed")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=1))
    print(f"done: {len(rep['passes'])} passes, {rep['total_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
