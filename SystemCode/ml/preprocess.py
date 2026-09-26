"""Load recordings, measure quality, and estimate shaft speed."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from ml.config import PreprocessConfig

TACHO = 0


def load_csv(path: str | Path) -> np.ndarray:
    """One MAFAULDA CSV (no header) as a float32 array of shape (n_rows, n_cols), parsed with pyarrow."""
    import pyarrow.csv as pacsv

    table = pacsv.read_csv(path, read_options=pacsv.ReadOptions(autogenerate_column_names=True))
    return np.column_stack([c.to_numpy() for c in table.columns]).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------- speed
class SpeedEstimate(NamedTuple):
    fr_hz: float  # shaft rotation frequency (Hz); NaN if every method failed
    method: str  # tacho_edges | tacho_fft | accel_comb | failed
    agreement: float  # max relative difference between the valid methods (fraction); NaN if < 2 valid


def _expected_band(fr_file_hz: float, cfg: PreprocessConfig) -> tuple[float, float, float]:
    centre = fr_file_hz * cfg.measured_speed_ratio
    hw = cfg.speed.search_halfwidth
    return centre, centre * (1 - hw), centre * (1 + hw)


def tacho_edges_speed(tacho: np.ndarray, fs: float, cfg: PreprocessConfig) -> float:
    """Speed (Hz) from rising edges of the once-per-rev pulse, with hysteresis and double-trigger rejection.

    Dropped pulses are handled by counting each interval as round(interval / median) revolutions.
    Returns NaN if the pulse train is unusable.
    """
    lo_level, hi_level = np.percentile(tacho, [10, 99])
    span = hi_level - lo_level
    if not np.isfinite(span) or span <= 0:
        return float("nan")
    lo = lo_level + cfg.speed.hyst_low * span
    hi = lo_level + cfg.speed.hyst_high * span

    # Hysteresis state: 1 above hi, 0 below lo, otherwise hold the last state.
    mark = np.full(tacho.shape[0], -1, dtype=np.int8)
    mark[tacho > hi] = 1
    mark[tacho < lo] = 0
    held = np.where(mark >= 0, np.arange(tacho.shape[0]), 0)
    np.maximum.accumulate(held, out=held)
    state = mark[held]
    edges = np.flatnonzero((state[1:] == 1) & (state[:-1] == 0)) + 1
    if edges.size < 4:
        return float("nan")

    # Reject double triggers: an edge closer than min_interval_ratio × median to the last kept edge.
    for _ in range(2):  # second pass uses the median of the cleaned intervals
        ref = np.median(np.diff(edges))
        keep = [edges[0]]
        for e in edges[1:]:
            if e - keep[-1] >= cfg.speed.min_interval_ratio * ref:
                keep.append(e)
        edges = np.asarray(keep)
        if edges.size < 4:
            return float("nan")

    intervals = np.diff(edges)
    period = np.median(intervals)
    revs = np.maximum(np.rint(intervals / period), 1)
    # Interval lengths per revolution should be consistent; otherwise the pulse train is garbage.
    per_rev = intervals / revs
    if np.std(per_rev) / np.mean(per_rev) > 0.05:
        return float("nan")
    return float(fs * revs.sum() / (edges[-1] - edges[0]))


def _peak_interp(mag: np.ndarray, i: int) -> float:
    """Parabolic interpolation of a spectral peak on log magnitude; returns a fractional bin."""
    if i <= 0 or i >= mag.size - 1:
        return float(i)
    a, b, c = np.log(mag[i - 1: i + 2] + 1e-12)
    denom = a - 2 * b + c
    return i + (0.5 * (a - c) / denom if denom != 0 else 0.0)


def tacho_fft_speed(tacho: np.ndarray, fs: float, fr_file_hz: float, cfg: PreprocessConfig) -> float:
    """Speed (Hz) from the tacho spectrum's fundamental, searched inside the expected band."""
    _, f_lo, f_hi = _expected_band(fr_file_hz, cfg)
    n = tacho.size * cfg.speed.fft_zero_pad
    mag = np.abs(np.fft.rfft(tacho - tacho.mean(), n=n))
    df = fs / n
    i_lo, i_hi = int(np.floor(f_lo / df)), int(np.ceil(f_hi / df))
    band = mag[i_lo: i_hi + 1]
    if band.size == 0 or not np.isfinite(band).all() or band.max() <= 0:
        return float("nan")
    # Noise floor from a wide window (0.5–1.5 × expected speed), not the narrow band the peak's own lobe fills.
    centre = 0.5 * (f_lo + f_hi)
    floor = np.median(mag[int(0.5 * centre / df): int(1.5 * centre / df)])
    if band.max() < cfg.speed.min_peak_ratio * floor:
        return float("nan")  # no real pulse-train line, only noise
    return float(_peak_interp(mag, i_lo + int(np.argmax(band))) * df)


def accel_comb_speed(accel: np.ndarray, fs: float, fr_file_hz: float, cfg: PreprocessConfig) -> float:
    """Speed (Hz) maximising summed log magnitude at the first K shaft harmonics of an accelerometer.

    Never the single largest peak: that can be a harmonic, a resonance or a bearing line.
    """
    centre, f_lo, f_hi = _expected_band(fr_file_hz, cfg)
    k = np.arange(1, cfg.speed.comb_harmonics + 1)
    n = accel.size * 4
    mag = np.abs(np.fft.rfft(accel - accel.mean(), n=n))
    if not np.isfinite(mag).all() or mag.max() <= 0:
        return float("nan")
    df = fs / n
    logmag = np.log(mag + 1e-12 * mag.max())
    candidates = np.arange(f_lo, f_hi, centre * 2e-4)  # 0.02 % steps
    bins = candidates[:, None] * k[None, :] / df
    score = np.interp(bins, np.arange(logmag.size), logmag).sum(axis=1)
    return float(candidates[int(np.argmax(score))])


def speed_candidates(x: np.ndarray, fr_file_hz: float, cfg: PreprocessConfig) -> dict[str, float]:
    """All three speed estimates (Hz, NaN where a method fails) for one recording."""
    fs = cfg.fs_hz
    out: dict[str, float] = {}
    for name, fn in (
        ("tacho_edges", lambda: tacho_edges_speed(x[:, TACHO], fs, cfg)),
        ("tacho_fft", lambda: tacho_fft_speed(x[:, TACHO], fs, fr_file_hz, cfg)),
        ("accel_comb", lambda: accel_comb_speed(x[:, cfg.speed.radial_channel], fs, fr_file_hz, cfg)),
    ):
        try:
            out[name] = fn()
        except Exception:  # noqa: BLE001 — a bad file must never stop indexing
            out[name] = float("nan")
    return out


def choose_speed(cands: dict[str, float], fr_file_hz: float, cfg: PreprocessConfig) -> SpeedEstimate:
    """Pick the winning estimate: tacho edges if another method confirms it, else the two that agree,
    else the accelerometer comb. Estimates outside the search band are treated as failed."""
    _, f_lo, f_hi = _expected_band(fr_file_hz, cfg)
    valid = {k: v for k, v in cands.items() if np.isfinite(v) and f_lo <= v <= f_hi}
    if not valid:
        return SpeedEstimate(float("nan"), "failed", float("nan"))

    vals = list(valid.values())
    agreement = (max(vals) - min(vals)) / np.median(vals) if len(vals) >= 2 else float("nan")
    tol = cfg.speed.disagree_tol

    def agree(a: str, b: str) -> bool:
        return a in valid and b in valid and abs(valid[a] - valid[b]) / valid[b] <= tol

    if agree("tacho_edges", "tacho_fft") or agree("tacho_edges", "accel_comb"):
        return SpeedEstimate(valid["tacho_edges"], "tacho_edges", float(agreement))
    if agree("tacho_fft", "accel_comb"):
        return SpeedEstimate(valid["tacho_fft"], "tacho_fft", float(agreement))
    for method in ("accel_comb", "tacho_edges", "tacho_fft"):
        if method in valid:
            return SpeedEstimate(valid[method], method, float(agreement))
    raise AssertionError("unreachable")


def estimate_speed(x: np.ndarray, fr_file_hz: float, cfg: PreprocessConfig | None = None) -> SpeedEstimate:
    """Measured shaft speed of one recording (n_rows × 8). Never raises: a bad file returns method 'failed'."""
    cfg = cfg or PreprocessConfig()
    try:
        return choose_speed(speed_candidates(x, fr_file_hz, cfg), fr_file_hz, cfg)
    except Exception:  # noqa: BLE001
        return SpeedEstimate(float("nan"), "failed", float("nan"))
