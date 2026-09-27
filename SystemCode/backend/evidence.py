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
CHANGE_LABEL = ("change beyond typical healthy variation · measured (descriptive, not a calibrated anomaly test)")
MAX_MARKED = 2  # at most the two channels furthest beyond their band are marked
MULTIPLICITY_NOTE = ("each recording is compared on ~20 channel × metric values; a healthy recording often exceeds at least "
                     "one 95th-percentile band by chance")
HEALTHY_NOTE = "within what the healthy checks accept"
KURTOSIS_NOTE = "excess kurtosis (Fisher; normal ≈ 0)"


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


def training_values(rec_id: str, cfg) -> dict[str, dict[str, float]]:
    """channel_values of a development-pool recording (cached)."""
    return channel_values(_features_for_rec(rec_id, cfg.model_dump_json()))


def compare(vals: dict, ref_vals: dict, bands: dict) -> list[dict]:
    """Per channel: every metric's ratio, direction, change factor and band; the channel's change is the comparable metric
    furthest beyond its band (factor / band), and the channel is beyond healthy variation if that exceeds 1."""
    from backend.variation import comparable, factor
    rows = []
    for ch, v in vals.items():
        metrics = {}
        for k in v:
            a, b = v[k], ref_vals[ch][k]
            ok = comparable(k, a, b)
            band = (bands.get(ch) or {}).get(k)
            f = factor(a, b) if ok else None
            metrics[k] = {"value": a, "reference": b, "ratio": a / b if ok else None, "factor": f, "band": band,
                          "direction": None if not ok else ("up" if a >= b else "down"),
                          "comparable": ok, "exceeds": bool(ok and band and f > band),
                          "note": KURTOSIS_NOTE if k == "kurtosis" else None if ok else "below the minimum amplitude on both sides"
                          if a > 0 and b > 0 else None}
        scored = {k: m["factor"] / m["band"] for k, m in metrics.items() if m["factor"] and m["band"]}
        top = max(scored, key=scored.get) if scored else None
        m = metrics.get(top) if top else None
        rows.append({"channel": ch, "sensor": SENSOR[ch], "values": v, "reference_values": ref_vals[ch], "metrics": metrics,
                     "change_metric": top, "change": m["ratio"] if m else None, "change_factor": m["factor"] if m else None,
                     "direction": m["direction"] if m else None, "band": m["band"] if m else None,
                     "beyond_band": scored.get(top) if top else None, "exceeds": bool(m and m["exceeds"])})
    return rows


def exceeding_channels(rows: list[dict]) -> list[str]:
    """Channels with a metric beyond its typical healthy variation, furthest beyond first."""
    return [r["channel"] for r in sorted([r for r in rows if r["exceeds"]], key=lambda r: -r["beyond_band"])]


def marked_channels(rows: list[dict], decision: str, caution: str | None) -> list[str]:
    """Lime marker rule (Bhuvesh, App-2a.2): no marker on a plain healthy verdict (a healthy recording often exceeds a
    band by chance); healthy-with-caution, known-fault and unknown verdicts mark up to MAX_MARKED channels beyond band."""
    if decision == "healthy" and not caution:
        return []
    return exceeding_channels(rows)[:MAX_MARKED]


def channel_evidence(x: np.ndarray, fr: float, cfg, meta: pd.DataFrame, rec_id: str | None, variation: dict,
                     decision: str, caution: str | None) -> dict:
    from ml.features import recording_features
    feats = recording_features(x, fr, cfg.preprocess)
    ref = healthy_reference(meta, cfg, fr, exclude=rec_id)
    rows = compare(channel_values(feats), training_values(ref.rec_id, cfg), variation["bands"])
    marked = marked_channels(rows, decision, caution)
    plain_healthy = decision == "healthy" and not caution
    return {"channels": rows, "marked": marked, "largest_change": marked, "exceeding": exceeding_channels(rows),
            "label": CHANGE_LABEL, "multiplicity_note": MULTIPLICITY_NOTE,
            "healthy_note": HEALTHY_NOTE if plain_healthy else None,
            "marker_rule": "no marker on a plain healthy verdict; otherwise up to 2 channels beyond their band",
            "reference": {"rec_id": ref.rec_id, "speed_hz": float(ref.fr_measured_hz), "class10": "normal",
                          "used_for": "model training (healthy)", "speed_difference_hz": float(abs(ref.fr_measured_hz - fr))},
            "variation": {"method": variation["method"], "n_pairs": variation["n_pairs"], "n_recordings": variation["n_recordings"]},
            "method": "mean over the 9 windows of R1 features (ml.features.recording_features), compared with the healthy "
                      "training recording nearest in measured speed; a channel is marked only if a metric's change factor "
                      "max(ratio, 1/ratio) exceeds the typical healthy variation of that channel and metric (95th percentile "
                      "over pairs of training normals within ±3 Hz); volt metrics need at least one side ≥ "
                      f"{variation['method']['min_amplitude_v']} V; excess kurtosis is not ratioed; at most {MAX_MARKED} "
                      "channels, the furthest beyond their band",
            "units": {"rms": "V (sensor output)", "order_1x": "V", "order_2x": "V", "kurtosis": KURTOSIS_NOTE,
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
