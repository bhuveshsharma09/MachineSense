"""Write rule.json, model_config.json and manifest.json (schema 1.0) for the model files produced by
scripts/14_train_bundle.py, then verify the bundle and copy the manifest to results/bundle/manifest.example.json.

    python scripts/15_bundle_manifest.py
"""
from __future__ import annotations

import json
import shutil
import sys

import pandas as pd

from ml.arbitrator import CAUTION_LABEL, CAUTIONS, DEFAULT_VARIANT, UNKNOWN_LABEL, VARIANTS
from ml.bundle import Manifest, file_entries, now_iso, verify_bundle, write_manifest
from ml.config import config_sha256, load_config
from ml.labels import CLASS6, CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, results_dir
from ml.splits import split_hash
from ml.utils import git_sha

B = artifacts_dir() / "bundle"
LIMITATIONS = [
    "Research demonstration; development results; not validated for real machines.",
    "Calibrated on 5 healthy recordings; trained on speeds below 40 Hz and on the development pool only.",
    "Decision rule is a provisional demo default (B_with_caution), not a final rule selection.",
    "The tested scores and rules do not consistently distinguish the imbalance / vertical-misalignment pair when one "
    "class is withheld. The classifier distinguishes them well when both are represented in training. This does not "
    "establish that no representation, training objective or decision method could help.",
    "Predicted fault location is according to the dataset labels; the classifier does not independently verify the "
    "physical sensor mapping. Per-channel reconstruction error is not fault-location evidence.",
]


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    rep = json.loads((results_dir() / "bundle" / "training_report.json").read_text())
    thr = json.loads((B / "thresholds.json").read_text())
    rule = {"healthy_rule": DEFAULT_VARIANT, "variants": list(VARIANTS), "status": "provisional demo default, NOT a final "
            "rule selection; chosen after the Stage 3D results; the final choice is made later under stated assumptions "
            "about fault frequency and costs", "unknown_label": UNKNOWN_LABEL, "caution_label": CAUTION_LABEL,
            "cautions": list(CAUTIONS), "rule_comparison": ["A", "B", "C", "D"]}
    (B / "rule.json").write_text(json.dumps(rule, indent=1, ensure_ascii=False))
    (B / "model_config.json").write_text(json.dumps({"cnn": cfg.cnn.model_dump(mode="json"), "ae": cfg.ae.model_dump(mode="json"),
                                                     "anomaly": cfg.anomaly.model_dump(mode="json"),
                                                     "representation": cfg.representation.model_dump(mode="json"),
                                                     "prob_floor": cfg.baselines.prob_floor, "seed": rep["seed"]}, indent=1))
    spec = {"cnn6": ("cnn6.safetensors", "classifier", "ScratchCNN state_dict, 6 vibration channels, 10 classes (Stage 3A architecture)"),
            "ae6": ("ae6.safetensors", "anomaly", "FrameAE state_dict, 6 vibration channels (Stage 3B frame-level AE)"),
            "mahalanobis": ("mahalanobis.safetensors", "openset", "class centroids and shared Ledoit–Wolf precision of CNN-6 train-window embeddings"),
            "lr_r1": ("lr_r1.joblib", "secondary_classifier", "scikit-learn QuantileTransformer + LogisticRegression on capped R1 features"),
            "b1s_vib": ("b1s_vib.joblib", "healthy_baseline", "ml.anomaly.SpeedPoly (B1s) on vibration-only R1 features of train normals"),
            "thresholds": ("thresholds.json", "thresholds", "τ_p, τ_d, τ_e, τ_b from the calibration recordings only"),
            "normalisation": ("normalisation.json", "normalisation", "per-channel mean/std (norm_stats, development mode)"),
            "rule": ("rule.json", "rule", "healthy-rule configuration (provisional demo default)"),
            "model_config": ("model_config.json", "config", "architecture and preprocessing settings used for training")}
    d = rep["data"]
    manifest = Manifest(
        created_at=now_iso(), git_sha=git_sha(), preprocess_config_sha256=config_sha256(cfg.preprocess),
        split_hash=split_hash(meta), fs_hz=cfg.preprocess.fs_hz, channels=list(cfg.preprocess.channels),
        class10=list(CLASS10), class6=list(CLASS6),
        representation={"candidate": cfg.representation.leading_candidate, "version": cfg.representation.version,
                        "cnn_channels": "6 accelerometer order-spectrograms (vibration only)"},
        files=file_entries(B, spec), thresholds=thr, rule=rule,
        data={"train": "closed train ∩ development pool", "calibration": "closed val ∩ development pool",
              "train_recordings": d["train_recordings"], "train_normals": d["train_normals"],
              "calibration_recordings": d["calibration_recordings"], "calibration_normals": d["calibration_normals"],
              "tau_e_normals": thr["n_calibration_normals_tau_e"], "tau_b_normals": thr["n_calibration_normals_tau_b"],
              "train_speed_range_hz": d["train_speed_range_hz"], "severities": d["severities"],
              "reserved_recordings_used": 0,
              "eligibility": "no reserved-test recording was used for training, calibration or inference, so this bundle "
                             "is eligible for one final evaluation on every reserved test"},
        limitations=LIMITATIONS)
    write_manifest(B, manifest)
    verify_bundle(B)
    shutil.copy(B / "manifest.json", results_dir() / "bundle" / "manifest.example.json")
    print(f"bundle verified: {len(manifest.files)} files; manifest → results/bundle/manifest.example.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
