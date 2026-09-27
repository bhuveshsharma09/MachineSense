"""Deployable inference (Stage ML-4): one entry point, diagnose(rec_id | raw 8-channel array at 50 kHz, 5 s) → a plain
dict (no Pydantic: the backend owns the schema). CPU by default; LightGBM is never imported.

Every number comes from existing ml/ functions: ml.preprocess (speed, window speeds, cleaning, quality statistics),
ml.features.recording_features (R1 features), ml.spectro.stacked_representation (cache-C representation),
ml.train.normalise, ml.openset (probabilities, Mahalanobis), ml.train_ae (AE error), ml.anomaly.SpeedPoly (B1s),
ml.evaluation (recording aggregation), ml.arbitrator (decision rule). A development-pool rec_id is read from
data/raw_npy with its metadata speed (which reproduces the training caches); every reserved-test rec_id is refused.
Research demonstration with documented limitations; development results; not validated for real machines."""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.arbitrator import DEFAULT_VARIANT, decide, rule_comparison
from ml.config import Config, config_sha256, load_config
from ml.evaluation import CLASS10_TO_6, aggregate_recordings, to_class6
from ml.features import comb_lines, envelope_order_spectrum, recording_features
from ml.labels import CLASS10
from ml.openset import mahalanobis_distances, recording_probs
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, artifacts_dir, data_dir
from ml.preprocess import (TACHO, clean_recording, estimate_speed, quality_stats, remove_mic_bursts,
                           tacho_edges_speed, tacho_glitch_count, window_speeds_detail)
from ml.spectro import stacked_representation
from ml.utils import git_sha

GOLDEN_RECORDINGS = ("normal__18.432", "underhang__outer_race__0g__14.7456", "horizontal-misalignment__0.5mm__15.36")
SPEED_WARNING = "speed outside the range the model was trained on; results less reliable"
CALIBRATION_SPEED_WARNING = "healthy thresholds were not calibrated at this speed; a healthy verdict is less reliable"
ENVELOPE_ORDERS = np.round(np.arange(0.0, 12.0 + 1e-9, 0.02), 4)  # downsampled envelope order axis (0.02-order step)
UNSTABLE_SPEED_PCT = 1.0  # window-speed range above this % of the median → "speed not stable within the recording"


class ReservedRecordingError(ValueError):
    """The rec_id belongs to a reserved test (split_closed, split_severity or split_speed_block = test)."""


class InputError(ValueError):
    """The input cannot be diagnosed (unknown rec_id, wrong shape, no measurable speed)."""


@lru_cache(maxsize=1)
def _metadata() -> pd.DataFrame:
    return pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv").set_index("rec_id")


BUNDLE_DIR = artifacts_dir() / "bundle"


@lru_cache(maxsize=1)
def _refusal_config() -> Config:
    """Config for the reserved-rec_id check before any model is loaded: the bundle's packaged configuration (verified)
    when a bundle is present, otherwise the repo config (the check only needs the development-pool rule)."""
    if (BUNDLE_DIR / "manifest.json").exists():
        from ml.bundle import bundle_config
        return bundle_config(BUNDLE_DIR)
    return load_config(DEFAULT_CONFIG)


@lru_cache(maxsize=2)
def get_bundle(directory: str | None = None):
    from ml.bundle import load_bundle
    return load_bundle(Path(directory) if directory else BUNDLE_DIR, device="cpu")


def reserved(rec_id: str, meta: pd.DataFrame, cfg: Config) -> bool:
    """True if the recording is in any reserved test (not in the development pool)."""
    row = meta.loc[rec_id]
    if row["split_closed"] not in cfg.development.allowed_closed:
        return True
    return any(row[c] == "test" for c in cfg.development.reserved_columns)


def check_rec_id(rec_id: str) -> pd.Series:
    """Metadata row of a development-pool recording; refuses unknown and reserved-test rec_ids."""
    meta, cfg = _metadata(), _refusal_config()
    if rec_id not in meta.index:
        raise InputError(f"unknown rec_id {rec_id!r}")
    if reserved(rec_id, meta, cfg):
        raise ReservedRecordingError(f"{rec_id} is a reserved-test recording; inference on reserved recordings is "
                                     "refused (reserved tests stay sealed)")
    return meta.loc[rec_id]


def speed_warnings(speed_hz: float, train_range_hz, calibration_normal_range_hz=None) -> list[str]:
    """Warnings when the measured recording speed lies outside the training speed range, and (separately) outside the
    speed range of the calibration normals that set the healthy thresholds τ_e and τ_b."""
    def outside(rng):
        return not (np.isfinite(speed_hz) and rng[0] <= speed_hz <= rng[1])

    out = [SPEED_WARNING] if outside(train_range_hz) else []
    if calibration_normal_range_hz is not None and outside(calibration_normal_range_hz):
        out.append(CALIBRATION_SPEED_WARNING)
    return out


def _signal_quality(x: np.ndarray, fr: float, cfg: Config, train_range, cal_range) -> tuple[dict, np.ndarray]:
    pre = cfg.preprocess
    tacho = np.asarray(x[:, TACHO], dtype=np.float64)
    speeds, fell_back = window_speeds_detail(tacho, pre.fs_hz, pre.window.win_s, pre.window.hop_s, pre, fr_fallback_hz=fr)
    glitches = tacho_glitch_count(tacho, pre)
    _, bursts = remove_mic_bursts(x[:, pre.channels.index("mic")], pre.mic.burst_z, pre.mic.burst_pad)
    q = quality_stats(x, pre.channels)
    clipped = [ch for ch in pre.channels[1:] if q.get(f"rail_{ch}", 0) >= pre.quality.rail_min]
    spikes = [ch for ch in pre.channels[1:] if q.get(f"peakz_{ch}", 0) > pre.quality.spike_z]
    range_pct = float(100 * (speeds.max() - speeds.min()) / np.median(speeds))
    warnings = speed_warnings(fr, train_range, cal_range)
    if range_pct > UNSTABLE_SPEED_PCT:
        warnings.append(f"shaft speed varies by {range_pct:.1f} % within the recording")
    if fell_back.any():
        warnings.append(f"{int(fell_back.sum())} of {len(speeds)} windows use the recording speed (tacho pulses unusable)")
    if glitches != 0:
        warnings.append("no usable tacho pulse train" if glitches < 0 else f"{glitches} glitched tacho intervals")
    if bursts:
        warnings.append(f"{bursts} microphone bursts removed")
    if q["nan_count"]:
        warnings.append(f"{q['nan_count']} missing (NaN) samples")
    if clipped:
        warnings.append(f"possible clipping on {', '.join(clipped)}")
    return {"shaft_speed_hz": float(fr), "window_speeds_hz": [float(v) for v in speeds],
            "window_speed_fallback": [bool(v) for v in fell_back], "speed_range_pct": range_pct,
            "speed_stable": range_pct <= UNSTABLE_SPEED_PCT, "train_speed_range_hz": list(train_range),
            "calibration_normal_speed_range_hz": list(cal_range),
            "tacho_glitches": int(glitches), "mic_bursts_removed": int(bursts), "nan_count": int(q["nan_count"]),
            "clipping_channels": clipped, "spike_channels": spikes, "warnings": warnings}, speeds


def _physics(x: np.ndarray, fr: float, cfg: Config) -> dict:
    pre = cfg.preprocess
    b = pre.bearing
    xc, _ = clean_recording(x, pre)
    band = tuple(pre.envelope.primary_band_hz)
    env = {ch: envelope_order_spectrum(np.asarray(xc[:, pre.channels.index(ch)], dtype=np.float64), pre.fs_hz, fr, band,
                                      ENVELOPE_ORDERS).tolist() for ch in pre.features.envelope_channels}
    lines = {k: [float(v) for v in vals] for k, vals in comb_lines(pre).items()}
    return {"shaft_speed_hz": float(fr), "shaft_speed_rpm": float(60 * fr),
            "bearing_frequencies_hz": {"bpfo": b.bpfo * fr, "bpfi": b.bpfi * fr, "bsf": b.bsf * fr,
                                       "bsf_defect_line_2x": 2 * b.bsf * fr, "ftf": b.ftf * fr},
            "envelope_spectrum": {"band_hz": list(band), "orders": ENVELOPE_ORDERS.tolist(), "channels": env,
                                  "note": "band-pass 1–5 kHz → Hilbert envelope → spectrum, on the cleaned recording, "
                                          "resampled to orders of the measured shaft speed (0.02-order step)"},
            "cursors_orders": {"shaft": [1.0, 2.0, 3.0], **lines},
            "cursors_hz": {k: [float(o * fr) for o in v] for k, v in {"shaft": [1.0, 2.0, 3.0], **lines}.items()}}


def stable_output(d: dict) -> dict:
    """diagnose() output without the run-dependent keys (timing and the current commit), for golden comparisons."""
    import json

    d = json.loads(json.dumps(d))
    d.pop("timing_ms", None)
    d["provenance"].pop("git_sha", None)
    return d


def load_recording(rec_id: str) -> tuple[np.ndarray, float]:
    """(raw 250000 × 8 array, metadata measured speed) of a development-pool recording (reserved rec_ids are refused)."""
    row = check_rec_id(rec_id)
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy"), float(row["fr_measured_hz"])


def score_recording(x: np.ndarray, fr: float, b) -> dict:
    """THE scoring function (Stage ML-4.2), shared by diagnose() and the calibration script (scripts/19_recalibrate.py):
    from the raw recording and its shaft speed, the packaged configuration's cleaning, windowing, R1 features and cache-C
    representation; float32 on the CPU; CNN-6 recording probabilities (mean log-probability over windows) and MSP;
    Mahalanobis distance (window minimum over classes, mean over windows); AE-6 error (per channel, then the equal-weight
    mean; mean over windows); B1s (mean over windows); LR recording probabilities. No other path computes these scores."""
    import torch

    from ml.config import AeConfig
    from ml.train import normalise
    from ml.train_ae import window_channel_errors

    cfg = b.config
    pre = cfg.preprocess
    feats = recording_features(x, fr, pre)
    rep = stacked_representation(x, fr, pre, cfg.representation.leading_candidate)
    nc, na = b.normalisation["cnn6"], b.normalisation["ae6"]
    xr = torch.from_numpy(rep[:, nc["channels"]].astype(np.float16)).to(torch.float32)
    with torch.no_grad():
        logits, emb = b.cnn(normalise(xr, torch.tensor(nc["mean"], dtype=torch.float32),
                                      torch.tensor(nc["std"], dtype=torch.float32)))
    logits, emb = logits.numpy(), emb.numpy().astype(np.float64)
    probs = recording_probs(logits, ["r"] * len(logits), CLASS10, b.model_config["prob_floor"]).iloc[0]
    ae_ch = window_channel_errors(b.ae, torch.from_numpy(rep), np.arange(len(rep)), na["channels"], np.asarray(na["mean"]),
                                  np.asarray(na["std"]), "frame", AeConfig(**b.model_config["ae"]), torch.device("cpu"))
    bm, lm = b.b1s, b.lr
    pw = lm["model"].predict_proba(feats[lm["feature_columns"]].to_numpy())
    pw = pw[:, [list(lm["classes"]).index(c) for c in CLASS10]]
    return {"feats": feats, "xr": xr, "probs": probs, "top": str(probs.idxmax()), "msp": float(probs.max()),
            "maha": float(mahalanobis_distances(b.mahalanobis, emb).min(axis=1).mean()),
            "ae": float(ae_ch.mean(axis=1).mean()),
            "ae_channels": dict(zip([pre.channels[1 + c] for c in na["channels"]], ae_ch.mean(axis=0).tolist())),
            "b1s": float(bm["model"].score(feats[bm["feature_columns"]].to_numpy(), feats.speed_hz.to_numpy()).mean()),
            "lr_probs": aggregate_recordings(pw, ["r"] * len(pw), lm["prob_floor"]).iloc[0]}


def diagnose(source, healthy_rule: str | None = None, nominal_speed_hz: float | None = None, bundle=None,
             with_xai: bool = True) -> dict:
    """Diagnose one recording: `source` is a development-pool rec_id or a raw (250000, 8) array at 50 kHz (columns:
    tacho, underhang axial/radial/tangential, overhang axial/radial/tangential, microphone). For a raw array the speed is
    measured (nominal_speed_hz, if given, only centres the search). Returns a plain dict. Scores come from
    score_recording(), the same function that set the thresholds."""
    t0 = time.perf_counter()
    if isinstance(source, str):
        check_rec_id(source)  # refuse reserved rec_ids before loading anything
    b = bundle or get_bundle()
    cfg = b.config  # the packaged configuration, never the repo's current configs/
    pre = cfg.preprocess
    rule = healthy_rule or b.rule.get("healthy_rule", DEFAULT_VARIANT)
    if isinstance(source, str):
        x, fr = load_recording(source)
        rec_id, origin = source, "development-pool recording"
    else:
        x = np.asarray(source)
        if x.shape != (pre.n_rows, pre.n_channels):
            raise InputError(f"expected a ({pre.n_rows}, {pre.n_channels}) array at {pre.fs_hz} Hz, got {x.shape}")
        guess = nominal_speed_hz or tacho_edges_speed(np.asarray(x[:, TACHO], dtype=np.float64), pre.fs_hz, pre)
        est = estimate_speed(x, float(guess) / pre.measured_speed_ratio if nominal_speed_hz is None else float(guess), pre)
        if not np.isfinite(est.fr_hz):
            raise InputError("the shaft speed could not be measured from this recording")
        fr, rec_id, origin = float(est.fr_hz), None, "uploaded array"
    train_range = b.manifest.data["train_speed_range_hz"]
    cal_range = b.manifest.data["calibration_normal_speed_range_hz"]
    quality, _ = _signal_quality(x, fr, cfg, train_range, cal_range)
    sc = score_recording(x, fr, b)
    feats, xr, probs, top, msp, maha = sc["feats"], sc["xr"], sc["probs"], sc["top"], sc["msp"], sc["maha"]
    ae_err, b1s, lr_rec = sc["ae"], sc["b1s"], sc["lr_probs"]
    # ---- decision
    thr = {k: float(b.thresholds[k]) for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
    verdict = decide(rule, top, msp, maha, ae_err, b1s, thr)
    diagnosis = {"model_name": "CNN-6 (scratch 2D CNN, vibration order spectrograms) + AE-6 + B1s; LR secondary",
                 "class10_probabilities": {c: float(probs[c]) for c in CLASS10}, "top_class": top,
                 "class6_probabilities": {c: float(v) for c, v in to_class6(probs.to_frame().T).iloc[0].items()},
                 "class6_mapping": dict(CLASS10_TO_6),
                 **{k: verdict[k] for k in ("decision", "fault_class", "display_label", "caution", "rule_variant")},
                 "ae_override": verdict["ae_override"],
                 "rule_comparison": rule_comparison(top, msp, maha, ae_err, b1s, thr),
                 "scores": {"msp": msp, "maha": maha, "ae_error": ae_err, "b1s": b1s, "ae_channel_errors": sc["ae_channels"]},
                 "thresholds": thr,
                 "lr_secondary": {"top_class": str(lr_rec.idxmax()), "probabilities": {c: float(lr_rec[c]) for c in CLASS10}},
                 "rule_status": b.rule.get("status")}
    xai = None
    if with_xai:
        from ml.xai import explain
        xai = explain(b, xr, top, feats, cfg)
    out = {"input": {"rec_id": rec_id, "origin": origin},
           "signal_quality": quality, "diagnosis": diagnosis, "physics": _physics(x, fr, cfg), "xai": xai,
           "limitations": list(b.manifest.limitations),
           "provenance": {"git_sha": git_sha(), "bundle_git_sha": b.manifest.git_sha, "manifest_sha256": b.manifest_sha256,
                          "model_sha256": {e.name: e.sha256 for e in b.manifest.files},
                          "preprocess_config_sha256": config_sha256(pre),
                          "bundle_preprocess_config_sha256": b.manifest.preprocess_config_sha256,
                          "split_hash": b.manifest.split_hash, "schema_version": b.manifest.schema_version}}
    out["timing_ms"] = {"total": 1000 * (time.perf_counter() - t0)}
    return out
