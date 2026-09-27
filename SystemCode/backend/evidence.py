"""Measured evidence for the evidence views (Stage App-2a). Every value comes from existing ml functions:
ml.features.recording_features (R1 features: RMS, kurtosis, order amplitudes, envelope lines), ml.preprocess.clean_recording
(the cleaned waveform) and ml.features.order_spectrum / comb_lines (spectra and physics cursors).

Per-channel change is DESCRIPTIVE, not a calibrated anomaly test: each channel's values are compared with the same values
of the healthy TRAINING recording nearest in measured speed (never a calibration or reserved recording). Per-channel
autoencoder error is not used (CLAUDE.md sensor-evidence finding: not usable as location or anomaly evidence)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

ACCEL = ("under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential")
SENSOR = {**{c: "s_in" for c in ACCEL[:3]}, **{c: "s_out" for c in ACCEL[3:]}, "mic": "s_mic", "tacho": "s_tach"}
ENV_LINES = {"bpfo1": "BPFO", "bpfi1": "BPFI", "bsf2": "2×BSF", "ftf1": "FTF"}
CHANGE_LABEL = "largest change vs healthy reference · measured (descriptive, not a calibrated anomaly test)"
NOT_RATIOED = {"kurtosis"}  # excess kurtosis can be ~0 or negative: reported, never ratioed
SECOND_WITHIN = 0.75  # the second channel is marked only if its change is at least 75 % of the largest


def _metrics(ch: str) -> dict[str, str]:
    if ch == "mic":
        return {"rms": "f_mic_rms"}
    m = {"rms": f"f_{ch}_rms", "kurtosis": f"f_{ch}_kurtosis", "order_1x": f"f_{ch}_o1", "order_2x": f"f_{ch}_o2"}
    if ch.endswith("radial"):
        m.update({f"env_{k}": f"f_env_{ch}_{k}" for k in ENV_LINES})
    return m


def channel_values(feats: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean over the recording's 9 windows of the R1 features used as measured evidence."""
    return {ch: {k: float(feats[col].mean()) for k, col in _metrics(ch).items()} for ch in (*ACCEL, "mic")}


def healthy_reference(meta: pd.DataFrame, cfg, speed_hz: float, exclude: str | None = None) -> pd.Series:
    """The healthy TRAINING recording (closed train ∩ development pool, class normal) nearest in measured speed."""
    from ml.splits import development_pool
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
    tr = pool[(pool.split_closed == "train") & (pool.class10 == "normal") & (pool.rec_id != (exclude or ""))]
    return tr.loc[(tr.fr_measured_hz - speed_hz).abs().idxmin()]


@lru_cache(maxsize=64)
def _features_for_rec(rec_id: str, cfg_json: str) -> pd.DataFrame:
    from ml.config import Config
    from ml.features import recording_features
    from ml.inference import load_recording
    x, fr = load_recording(rec_id)
    return recording_features(x, fr, Config.model_validate_json(cfg_json).preprocess)


def channel_evidence(x: np.ndarray, fr: float, cfg, meta: pd.DataFrame, rec_id: str | None) -> dict:
    from ml.features import recording_features
    feats = recording_features(x, fr, cfg.preprocess)
    ref = healthy_reference(meta, cfg, fr, exclude=rec_id)
    ref_vals = channel_values(_features_for_rec(ref.rec_id, cfg.model_dump_json()))
    vals = channel_values(feats)
    rows = []
    for ch, v in vals.items():
        ratios = {k: (v[k] / ref_vals[ch][k]) if k not in NOT_RATIOED and ref_vals[ch][k] > 0 and v[k] > 0 else None
                  for k in v}
        valid = {k: r for k, r in ratios.items() if r is not None and np.isfinite(r)}
        metric = max(valid, key=lambda k: abs(np.log(valid[k]))) if valid else None
        rows.append({"channel": ch, "sensor": SENSOR[ch], "values": v, "reference_values": ref_vals[ch], "ratios": ratios,
                     "change": valid.get(metric), "change_metric": metric,
                     "change_factor": float(np.exp(abs(np.log(valid[metric])))) if metric else None})
    ranked = sorted([r for r in rows if r["change"] is not None], key=lambda r: -r["change_factor"])
    marked = [ranked[0]["channel"]] if ranked else []
    if len(ranked) > 1 and ranked[1]["change_factor"] >= SECOND_WITHIN * ranked[0]["change_factor"]:
        marked.append(ranked[1]["channel"])
    return {"channels": rows, "largest_change": marked, "label": CHANGE_LABEL,
            "reference": {"rec_id": ref.rec_id, "speed_hz": float(ref.fr_measured_hz), "class10": "normal",
                          "used_for": "model training (healthy)"},
            "method": "mean over the 9 windows of R1 features (ml.features.recording_features); a channel's change = its "
                      "value / the healthy training reference's value (nearest in measured speed), for the metric furthest "
                      "from 1 in either direction (change factor = max(ratio, 1/ratio); kurtosis is not ratioed); the "
                      f"second channel is marked only if its change factor is at least {SECOND_WITHIN:.0%} of the largest",
            "units": {"rms": "V (sensor output)", "order_1x": "V", "order_2x": "V", "kurtosis": "—",
                      **{f"env_{k}": "ratio to envelope median" for k in ENV_LINES}}}


def signals(x: np.ndarray, fr: float, cfg, t0_s: float = 1.0, dur_s: float = 0.5, buckets: int = 1000) -> dict:
    """Per channel: a 0.5 s cleaned waveform (min/max per bucket) and the order spectrum of the cleaned recording, with
    physics cursors at the measured speed."""
    from ml.features import comb_lines, order_spectrum
    from ml.preprocess import clean_recording
    pre = cfg.preprocess
    xc, _ = clean_recording(x, pre)
    fs = pre.fs_hz
    s, n = int(t0_s * fs), int(dur_s * fs)
    orders = np.round(np.arange(0.25, 10.0 + 1e-9, 0.01), 4)
    lines = {k: [float(o) for o in v] for k, v in comb_lines(pre).items()}
    out = {}
    for i, ch in enumerate(pre.channels):
        sig = np.asarray((x if ch == "tacho" else xc)[:, i], dtype=np.float64)
        seg = sig[s:s + n].reshape(buckets, -1)
        out[ch] = {"waveform": {"t0_s": t0_s, "dt_s": dur_s / buckets, "min": seg.min(axis=1).round(6).tolist(),
                                "max": seg.max(axis=1).round(6).tolist()},
                   "spectrum": order_spectrum(sig, fs, fr, orders).round(8).tolist()}
    return {"channels": out, "orders": orders.tolist(), "shaft_speed_hz": fr,
            "cursors_orders": {"shaft": [1.0, 2.0, 3.0], **lines},
            "note": "cleaned recording (ml.preprocess.clean_recording; tachometer untouched); spectrum = ml.features."
                    "order_spectrum of the whole 5 s on orders of the measured speed; waveform 1.0–1.5 s, min/max per bucket"}
