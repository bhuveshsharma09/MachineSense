"""Train and calibrate the frozen final models of CLAUDE.md §12 (Stage F0), on DEVELOPMENT data only. Never imports
LightGBM. No reserved-test recording is read.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/29_train_final.py train      # GPU: AE-6, 9 × CNN-6, R18-ft
    python scripts/29_train_final.py calibrate                                            # CPU: LOCO thresholds
    python scripts/29_train_final.py manifest                                             # hashes + verify

Data (§12a): fit / train = closed train ∩ development pool (660 recordings; LOCO: minus the withheld class);
calibration = closed val ∩ development pool (144; LOCO: minus the withheld class). AE-6 and B1s on the 20 fit normals,
shared by the 9 LOCO models (normal is never withheld). Thresholds on the CPU deployment scoring path (ml.final): τ_p,
τ_d = 97.5 % order statistic over the calibration recordings; τ_e, τ_b = maximum over the 5 calibration normals.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

from ml.anomaly import SpeedPoly
from ml.bundle import now_iso
from ml.cache import load_features
from ml.calibration import thresholds_from_scores
from ml.config import load_config
from ml.datasets import NormContext, dev_window_rows, load_representation, norm_stats
from ml.features import feature_columns
from ml.final import VIB, final_dir, list_files, recording_inputs, score_cnn, verify_final, write_manifest
from ml.labels import CLASS10
from ml.models.resnet import load_resnet_config, verify_weights
from ml.openset import fit_mahalanobis
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, bundle_dir, data_dir, results_dir
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEED = 42
REPORT = results_dir() / "final_models"


def split_sets(cfg, meta):
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    tr, va = pool[pool.split_closed == "train"], pool[pool.split_closed == "val"]
    reserved = (meta.split_closed == "test") | (meta.split_severity == "test") | (meta.split_speed_block == "test")
    assert not set(pool.rec_id) & set(meta.rec_id[reserved]), "the development pool contains a reserved recording"
    return pool, tr, va


def loco_ids(tr, va, c):
    return sorted(tr.rec_id[tr.class10 != c]), sorted(va.rec_id[va.class10 != c])


def run_train() -> None:
    import torch
    from safetensors.numpy import save_file as save_np
    from safetensors.torch import save_file

    from ml.models.resnet import ResNet18Spectrogram, param_groups
    from ml.train import gpu_job, score_windows, train_model
    from ml.train_ae import train_ae
    from ml.utils import get_device, seed_everything

    cfg = load_config(DEFAULT_CONFIG)
    rc = load_resnet_config().model_copy(update={"seed": SEED})
    verify_weights(rc)
    device = get_device()
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool, tr, va = split_sets(cfg, meta)
    cc, ac = cfg.cnn.model_copy(update={"seed": SEED}), cfg.ae.model_copy(update={"seed": SEED})
    out = final_dir()
    for sub in ("shared", "r18ft", *[f"loco/{c}" for c in FAULT_CLASSES]):
        (out / sub).mkdir(parents=True, exist_ok=True)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool.assign(dev_fold=0))
    tr_norm = sorted(tr.rec_id[tr.class10 == "normal"])
    ctx_all = NormContext(calibration=frozenset(va.rec_id))
    report = {"git_sha": git_sha(), "split_hash": split_hash(meta), "seed": SEED, "device": str(device), "models": {}}
    save = lambda m, p: save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(p))  # noqa: E731
    with gpu_job(device, artifacts_dir(), cc.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # development-pool windows only
        del x
        # ---- shared AE-6 on the 20 fit normals (identical for every withheld class)
        st_ae = norm_stats(arr, index, meta, fit_ids=tr_norm, context=ctx_all)
        mean_a, std_a = st_ae["mean"][VIB], st_ae["std"][VIB]
        seed_everything(SEED)
        t0 = time.perf_counter()
        ae, hist_ae = train_ae(cache, np.flatnonzero(rows.rec_id.isin(tr_norm).to_numpy()), VIB, mean_a, std_a, "frame",
                               ac, device)
        save(ae, out / "shared" / "ae6.safetensors")
        (out / "shared" / "normalisation.json").write_text(json.dumps(
            {"channels": VIB, "mean": mean_a.tolist(), "std": std_a.tolist(), "n_windows": st_ae["n_windows"],
             "fit_normals": tr_norm}, indent=1))
        report["models"]["ae6_shared"] = {"fit_normals": len(tr_norm), "train_seconds": time.perf_counter() - t0,
                                          "final_loss": hist_ae[-1]["train_loss"]}
        # ---- 9 LOCO CNN-6 models
        for c in FAULT_CLASSES:
            fit_ids, cal_ids = loco_ids(tr, va, c)
            known = [k for k in CLASS10 if k != c]
            y = torch.from_numpy(rows.class10.map({k: i for i, k in enumerate(known)}).fillna(0).to_numpy(np.int64)).to(device)
            st = norm_stats(arr, index, meta, fit_ids=fit_ids, context=NormContext(calibration=frozenset(cal_ids)))
            mean, std = st["mean"][VIB], st["std"][VIB]
            fit_rows = np.flatnonzero(rows.rec_id.isin(fit_ids).to_numpy())
            assert c not in set(rows.class10.iloc[fit_rows])
            seed_everything(SEED)
            t0 = time.perf_counter()
            cnn, hist = train_model(cache, y, fit_rows, VIB, mean, std, len(known), cc, device)
            t_cnn = time.perf_counter() - t0
            _, emb = score_windows(cnn, cache, fit_rows, VIB, mean, std, cc, device)
            labels = rows.class10.to_numpy()[fit_rows]
            emb = emb.astype(np.float64)
            mh = fit_mahalanobis(emb, labels)
            from sklearn.covariance import LedoitWolf
            lws = {k: LedoitWolf().fit(emb[labels == k]) for k in mh["classes"]}
            sub = out / "loco" / c
            save(cnn, sub / "cnn6.safetensors")
            save_np({"centroids": mh["centroids"], "precision": mh["precision"]}, str(sub / "mahalanobis.safetensors"))
            save_np({"locations": np.stack([lws[k].location_ for k in mh["classes"]]),
                     "precisions": np.stack([lws[k].precision_ for k in mh["classes"]])}, str(sub / "mahalanobis_pc.safetensors"))
            assert list(mh["classes"]) == sorted(known)
            (sub / "normalisation.json").write_text(json.dumps(
                {"withheld": c, "classes": known, "mahalanobis_classes": mh["classes"], "channels": VIB,
                 "mean": mean.tolist(), "std": std.tolist(), "n_windows": st["n_windows"],
                 "mahalanobis_shared_shrinkage": mh["shrinkage"],
                 "mahalanobis_per_class_shrinkage": {k: float(lws[k].shrinkage_) for k in mh["classes"]},
                 "fit_ids": fit_ids, "calibration_ids": cal_ids}, indent=1))
            report["models"][f"loco_{c}"] = {"fit_recordings": len(fit_ids), "fit_windows": int(len(fit_rows)),
                                             "fit_normals": len(tr_norm), "calibration_recordings": len(cal_ids),
                                             "calibration_normals": int((va.class10 == "normal").sum()),
                                             "train_seconds": t_cnn, "final_loss": hist[-1]["train_loss"]}
            print(f"  loco {c}: fit {len(fit_ids)} rec, cal {len(cal_ids)}, {t_cnn:.0f} s", flush=True)
        # ---- R18-ft on the 660 training recordings (Q2 only)
        tr_ids = sorted(tr.rec_id)
        st = norm_stats(arr, index, meta, fit_ids=tr_ids, context=ctx_all)
        mean, std = st["mean"][VIB], st["std"][VIB]
        yy = torch.from_numpy(rows.class10.map({k: i for i, k in enumerate(CLASS10)}).to_numpy(np.int64)).to(device)
        seed_everything(SEED)
        model = ResNet18Spectrogram(len(VIB), len(CLASS10), rc, "r18ft").to(device)
        t0 = time.perf_counter()
        model, hist = train_model(cache, yy, np.flatnonzero(rows.rec_id.isin(tr_ids).to_numpy()), VIB, mean, std,
                                  len(CLASS10), rc, device, model=model, param_groups=param_groups(model, rc),
                                  on_epoch=lambda ep, r: print(f"    r18ft epoch {ep}: loss {r['train_loss']:.4f}", flush=True))
        save(model, out / "r18ft" / "r18ft.safetensors")
        (out / "r18ft" / "normalisation.json").write_text(json.dumps(
            {"classes": CLASS10, "channels": VIB, "mean": mean.tolist(), "std": std.tolist(), "n_windows": st["n_windows"],
             "train_ids": tr_ids}, indent=1))
        report["models"]["r18ft"] = {"train_recordings": len(tr_ids), "train_seconds": time.perf_counter() - t0,
                                     "final_loss": hist[-1]["train_loss"], "thresholds": None}
    # ---- B1s on the fit normals (CPU, scikit-learn)
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess)
    vib_cols = [col for col in feature_columns(df) if not col.startswith("f_mic_")]
    dn = df[df.rec_id.isin(tr_norm)]
    b1s = SpeedPoly(cfg.anomaly.b1s_degree).fit(dn[vib_cols].to_numpy(), dn.speed_hz.to_numpy())
    joblib.dump({"model": b1s, "feature_columns": vib_cols, "degree": cfg.anomaly.b1s_degree,
                 "doc": "ml.anomaly.SpeedPoly on the 20 fit-normal windows (vibration-only R1); shared by the LOCO models"},
                out / "shared" / "b1s_vib.joblib")
    (out / "model_config.json").write_text(json.dumps({"cnn": cc.model_dump(mode="json"), "ae": ac.model_dump(mode="json"),
                                                       "resnet": rc.model_dump(mode="json"),
                                                       "prob_floor": cfg.baselines.prob_floor}, indent=1))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "training_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["models"], indent=1, default=float))


def run_calibrate() -> None:
    """Score every calibration recording once on the CPU path and set each LOCO model's thresholds (§12a)."""
    import joblib as jl
    import torch
    from safetensors.numpy import load_file as load_np
    from safetensors.torch import load_file

    from ml.config import AeConfig, CnnConfig
    from ml.inference import load_recording
    from ml.models.ae import build_ae
    from ml.models.cnn import ScratchCNN

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool, tr, va = split_sets(cfg, meta)
    out = final_dir()
    mc = json.loads((out / "model_config.json").read_text())
    ae_norm = json.loads((out / "shared" / "normalisation.json").read_text())
    ae = build_ae("frame", len(VIB), AeConfig(**mc["ae"]))
    ae.load_state_dict(load_file(str(out / "shared" / "ae6.safetensors"), device="cpu"))
    ae.eval()
    b1s = jl.load(out / "shared" / "b1s_vib.joblib")
    models = {}
    for c in FAULT_CLASSES:
        sub = out / "loco" / c
        norm = json.loads((sub / "normalisation.json").read_text())
        cnn = ScratchCNN(len(VIB), len(norm["classes"]), CnnConfig(**mc["cnn"]))
        cnn.load_state_dict(load_file(str(sub / "cnn6.safetensors"), device="cpu"))
        mh, pc = load_np(str(sub / "mahalanobis.safetensors")), load_np(str(sub / "mahalanobis_pc.safetensors"))
        models[c] = (cnn.eval(), norm, {"classes": norm["mahalanobis_classes"], "centroids": mh["centroids"],
                                        "precision": mh["precision"]},
                     {"classes": norm["mahalanobis_classes"], "locations": pc["locations"], "precisions": pc["precisions"]})
    rows = {c: [] for c in FAULT_CLASSES}
    t0 = time.perf_counter()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for i, r in enumerate(sorted(va.rec_id)):
        x, fr = load_recording(r)  # development-pool recordings only (reserved rec_ids are refused)
        inp = recording_inputs(x, fr, cfg)
        cls = va.set_index("rec_id").class10[r]
        for c, (cnn, norm, mh, pc) in models.items():
            if cls == c:
                continue
            s = score_cnn(inp, cnn, norm, norm["classes"], mh, ae, ae_norm, mc["ae"], b1s, mc["prob_floor"], maha_pc=pc)
            rows[c].append({"rec_id": r, "class10": cls, "top_class": s["top"], "msp": s["msp"], "maha": s["maha"],
                            "maha_pc": s["maha_pc"], "ae": s["ae"], "b1s": s["b1s"]})
        if i % 20 == 0:
            print(f"  {i + 1}/{len(va)} calibration recordings, {time.perf_counter() - t0:.0f} s", flush=True)
    summary = {}
    for c in FAULT_CLASSES:
        s = pd.DataFrame(rows[c]).set_index("rec_id").sort_index()
        t = thresholds_from_scores(s, s.class10 == "normal")
        t.update({"withheld": c, "scoring_path": "ml.final.score_cnn on the CPU (ML-4.2 deployment path)",
                  "tolerance": None, "mahalanobis": "shared within-class Ledoit–Wolf (primary; τ_d)",
                  "note": "τ_p, τ_d: 97.5 % order statistic over the calibration recordings (ties accepted); τ_e, τ_b: "
                          "maximum over the calibration normals (§12a)"})
        s.to_csv(out / "loco" / c / "calibration_scores.csv", float_format="%.17g")
        (out / "loco" / c / "thresholds.json").write_text(json.dumps(t, indent=1, default=float))
        summary[c] = {k: t[k] for k in ("tau_p", "tau_d", "tau_e", "tau_b", "n_calibration", "order_statistic_r",
                                        "n_calibration_normals_tau_e", "n_calibration_normals_tau_b")}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "calibration.json").write_text(json.dumps({"seconds": time.perf_counter() - t0, "thresholds": summary},
                                                        indent=1, default=float))
    print(pd.DataFrame(summary).T.to_string())


def run_manifest() -> None:
    from ml.bundle import sha256_file
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    out = final_dir()
    rep = json.loads((REPORT / "training_report.json").read_text())
    cal = json.loads((REPORT / "calibration.json").read_text())
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=SYSTEM_CODE_DIR, capture_output=True, text=True).stdout
    protocol = subprocess.run(["git", "log", "-1", "--format=%H", "--", "../CLAUDE.md"], cwd=SYSTEM_CODE_DIR,
                              capture_output=True, text=True).stdout.strip()
    info = {"created_at": now_iso(), "git_sha": git_sha(), "unrelated_worktree_changes": dirty.splitlines(),
            "split_hash": split_hash(meta), "protocol": "CLAUDE.md §12", "protocol_commit": protocol,
            "bundle_manifest_sha256": sha256_file(bundle_dir() / "manifest.json"),
            "loco_classes": list(FAULT_CLASSES), "seed": SEED, "training": rep["models"], "thresholds": cal["thresholds"],
            "resnet_weights_sha256": load_resnet_config().weights_sha256,
            "data": "development pool only: fit/train = closed train ∩ pool (minus the withheld class for LOCO); "
                    "calibration = closed val ∩ pool (minus the withheld class)"}
    write_manifest(out, info)
    m = verify_final(out)
    print(f"manifest: {len(m['files'])} files verified; git {m['git_sha']}; split {m['split_hash'][:12]}")
    print("\n".join(list_files(out)))


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    if task not in ("train", "calibrate", "manifest"):
        print(__doc__)
        return 2
    {"train": run_train, "calibrate": run_calibrate, "manifest": run_manifest}[task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
