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


class TachoIntervals(NamedTuple):
    edges: np.ndarray  # sample index of each kept rising edge
    intervals: np.ndarray  # samples between kept rising edges
    revs: np.ndarray  # revolutions per interval (1, or more across dropped pulses)
    good: np.ndarray  # bool: interval length per revolution within tolerance of the median period


def tacho_intervals(tacho: np.ndarray, cfg: PreprocessConfig) -> TachoIntervals | None:
    """Rising-edge intervals of the once-per-rev pulse, with hysteresis and double-trigger rejection.

    Dropped pulses count as round(interval / median) revolutions. Glitches (a displaced pulse splits one
    revolution pair into a short and a long interval) are marked not-good. None if there is no pulse train.
    """
    lo_level, hi_level = np.percentile(tacho, [10, 99])
    span = hi_level - lo_level
    if not np.isfinite(span) or span <= 0:
        return None
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
        return None

    # Reject double triggers: an edge closer than min_interval_ratio × median to the last kept edge.
    for _ in range(2):  # second pass uses the median of the cleaned intervals
        ref = np.median(np.diff(edges))
        keep = [edges[0]]
        for e in edges[1:]:
            if e - keep[-1] >= cfg.speed.min_interval_ratio * ref:
                keep.append(e)
        edges = np.asarray(keep)
        if edges.size < 4:
            return None

    intervals = np.diff(edges)
    period = np.median(intervals)
    revs = np.maximum(np.rint(intervals / period), 1)
    good = np.abs(intervals / revs / period - 1) <= cfg.speed.interval_tol
    return TachoIntervals(edges, intervals, revs, good)


def tacho_edges_speed(tacho: np.ndarray, fs: float, cfg: PreprocessConfig) -> float:
    """Speed (Hz) from the good tacho intervals only, so glitched revolutions are corrected, not averaged in.
    NaN if there is no pulse train or fewer than min_good_fraction of intervals are good."""
    ti = tacho_intervals(tacho, cfg)
    if ti is None or ti.good.mean() < cfg.speed.min_good_fraction:
        return float("nan")
    return float(fs * ti.revs[ti.good].sum() / ti.intervals[ti.good].sum())


def window_speeds(tacho: np.ndarray, fs: float, win_s: float = 1.0, hop_s: float = 0.5,
                  cfg: PreprocessConfig | None = None, fr_fallback_hz: float | None = None) -> tuple[np.ndarray, int]:
    """Shaft speed (Hz) per window, from the clean tacho intervals lying entirely inside it. Audit only.

    Windows are win_s long, stepped by hop_s (9 windows for a 5 s recording at the defaults). A window with fewer than
    cfg.speed.window_min_intervals clean intervals falls back to the recording speed (fr_fallback_hz, or the tacho-edge
    speed of the whole recording); so does every window when the pulse train as a whole is unusable (fewer than
    cfg.speed.min_good_fraction clean intervals). Returns (speeds, number of windows that fell back).
    """
    cfg = cfg or PreprocessConfig()
    n, win, hop = tacho.shape[0], int(round(win_s * fs)), int(round(hop_s * fs))
    starts = np.arange(0, n - win + 1, hop)
    fallback = fr_fallback_hz if fr_fallback_hz is not None else tacho_edges_speed(tacho, fs, cfg)
    ti = tacho_intervals(tacho, cfg)
    speeds = np.full(starts.size, float(fallback))
    n_fallback = starts.size
    # Same validity as tacho_edges_speed: on a dead or garbled tacho, some noise intervals look "clean" by chance.
    if ti is not None and ti.good.mean() >= cfg.speed.min_good_fraction:
        n_fallback = 0
        first, last = ti.edges[:-1], ti.edges[1:]
        for i, s in enumerate(starts):
            inside = ti.good & (first >= s) & (last <= s + win)
            if inside.sum() >= cfg.speed.window_min_intervals:
                speeds[i] = fs * ti.revs[inside].sum() / ti.intervals[inside].sum()
            else:
                n_fallback += 1
    return speeds, n_fallback


def tacho_glitch_count(tacho: np.ndarray, cfg: PreprocessConfig) -> int:
    """Number of glitched tacho intervals (−1 if there is no usable pulse train). A quality flag only."""
    ti = tacho_intervals(tacho, cfg)
    return -1 if ti is None else int((~ti.good).sum())


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
    """Pick the winning estimate: tacho edges if another method confirms it (within confirm_tol), else the
    tacho FFT if the comb confirms it, else the median of three, else the most precise valid sensor.
    Estimates outside the search band are treated as failed."""
    _, f_lo, f_hi = _expected_band(fr_file_hz, cfg)
    valid = {k: v for k, v in cands.items() if np.isfinite(v) and f_lo <= v <= f_hi}
    if not valid:
        return SpeedEstimate(float("nan"), "failed", float("nan"))

    vals = list(valid.values())
    agreement = (max(vals) - min(vals)) / np.median(vals) if len(vals) >= 2 else float("nan")
    tol = cfg.speed.confirm_tol

    def agree(a: str, b: str) -> bool:
        return a in valid and b in valid and abs(valid[a] - valid[b]) / valid[b] <= tol

    if agree("tacho_edges", "tacho_fft") or agree("tacho_edges", "accel_comb"):
        return SpeedEstimate(valid["tacho_edges"], "tacho_edges", float(agreement))
    if agree("tacho_fft", "accel_comb"):
        return SpeedEstimate(valid["tacho_fft"], "tacho_fft", float(agreement))
    if len(valid) == 3:  # nobody confirms anybody: take the median estimate
        method = sorted(valid, key=valid.get)[1]
        return SpeedEstimate(valid[method], method, float(agreement))
    for method in ("tacho_edges", "tacho_fft", "accel_comb"):  # tacho is the more precise sensor
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


# ----------------------------------------------------------------------------- quality
def quality_stats(x: np.ndarray, channels: tuple[str, ...]) -> dict[str, float]:
    """Per-recording quality numbers (no decisions): shape, NaNs, and per channel the DC offset, the number of
    samples sitting exactly on the channel's max/min (rail count, a clipping signature) and the largest robust
    z-score |x − median| / (1.4826 · MAD) (a spike signature)."""
    out: dict[str, float] = {"n_rows": x.shape[0], "n_cols": x.shape[1] if x.ndim == 2 else 0,
                             "nan_count": int(np.isnan(x).sum())}
    for i, ch in enumerate(channels[: out["n_cols"]]):
        v = x[:, i]
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[f"dc_{ch}"] = out[f"rail_{ch}"] = out[f"peakz_{ch}"] = float("nan")
            continue
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        out[f"dc_{ch}"] = float(v.mean())
        out[f"rail_{ch}"] = int((v == v.max()).sum() + (v == v.min()).sum())
        out[f"peakz_{ch}"] = float(np.abs(v - med).max() / mad) if mad > 0 else float("inf")
    return out


def quality_flags(row: dict, cfg: PreprocessConfig) -> list[str]:
    """Flags from an index row (quality_stats + speed columns); thresholds in cfg.quality. Audit only."""
    q = cfg.quality
    flags: list[str] = []
    if row.get("n_rows") != cfg.n_rows or row.get("n_cols") != cfg.n_channels:
        flags.append("shape")
    if row.get("nan_count", 0) > 0:
        flags.append("nan")
    if row.get("speed_method") == "failed":
        flags.append("speed_failed")
    if row.get("tacho_glitches", 0) != 0:
        flags.append("tacho_glitch")
    if row.get("speed_agreement", 0) > cfg.speed.disagree_tol:
        flags.append("speed_disagree")
    for ch in cfg.channels[1:]:  # the tacho is a square wave: rails and big z-scores are normal there
        if row.get(f"rail_{ch}", 0) >= q.rail_min:
            flags.append(f"clip_{ch}")
        if row.get(f"peakz_{ch}", 0) > q.spike_z:
            flags.append(f"spike_{ch}")
        if abs(row.get(f"dc_{ch}", 0)) > q.dc_abs:
            flags.append(f"dc_{ch}")
    return flags


# ----------------------------------------------------------------------------- microphone bursts
def remove_mic_bursts(mic: np.ndarray, z_thresh: float, pad: int = 5) -> tuple[np.ndarray, int]:
    """Remove short electrical bursts from the microphone channel (CLAUDE.md §5 rule 10).

    Samples with |x − median| / (1.4826 · MAD) > z_thresh are widened by ±pad samples and replaced by linear
    interpolation between the clean neighbours. Returns (cleaned float32 copy, number of burst runs).
    """
    x = np.asarray(mic, dtype=np.float32)
    if x.size == 0:
        return x.copy(), 0
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return x.copy(), 0
    hit = np.abs(x - med) / mad > z_thresh
    if not hit.any():
        return x.copy(), 0
    mask = np.convolve(hit, np.ones(2 * pad + 1), mode="same") > 0  # widen each run by ±pad
    n_bursts = int(np.count_nonzero(np.diff(mask.astype(np.int8), prepend=0) == 1))
    out = x.copy()
    idx = np.arange(x.size)
    if mask.all():
        out[:] = med
    else:
        out[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return out, n_bursts


# ----------------------------------------------------------------------------- whole-recording cleaning
def propose_stopband(tone, present_db: float = 20.0, margin_hz: float = 25.0, round_hz: float = 5.0) -> dict:
    """Stop band for the ~22 kHz line from a tone table (one row per recording × channel: f0_hz, snr_db and, per
    sideband offset k in ±60/±120 Hz, f{k}_hz and snr{k}_db). The observed extent of present lines (main line
    snr > present_db, sidebands > present_db / 2), widened by margin_hz each side and rounded outward to round_hz."""
    main = tone[tone.snr_db > present_db]
    lo, hi = main.f0_hz.min(), main.f0_hz.max()
    for off in ("-120", "-60", "+60", "+120"):
        s = main[main[f"snr{off}_db"] > present_db / 2]
        lo, hi = min(lo, s[f"f{off}_hz"].min()), max(hi, s[f"f{off}_hz"].max())
    return {"observed_lo_hz": float(lo), "observed_hi_hz": float(hi),
            "stop_lo_hz": float(np.floor((lo - margin_hz) / round_hz) * round_hz),
            "stop_hi_hz": float(np.ceil((hi + margin_hz) / round_hz) * round_hz)}


def bandstop_taps(cfg: PreprocessConfig) -> np.ndarray:
    """FIR taps of the fixed band-stop (odd length, symmetric → linear phase). Cutoffs sit half a transition
    outside the stop band, so the full design attenuation holds from stop_lo_hz to stop_hi_hz."""
    from scipy.signal import firwin

    b = cfg.bandstop
    if b.numtaps % 2 == 0:
        raise ValueError("bandstop.numtaps must be odd for a linear-phase band-stop")
    cut = [b.stop_lo_hz - b.transition_hz / 2, b.stop_hi_hz + b.transition_hz / 2]
    return firwin(b.numtaps, cut, window=tuple(b.window), pass_zero="bandstop", fs=cfg.fs_hz)


def apply_fir_centred(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve each column with a symmetric odd-length FIR and keep the centred part: the (numtaps − 1) / 2 sample
    group delay is removed, so filtered and original stay time-aligned (zero-phase alignment). Edges are
    zero-padded, so the first and last (numtaps − 1) / 2 samples see a partial filter."""
    from scipy.signal import oaconvolve

    x2 = x[:, None] if x.ndim == 1 else x
    y = oaconvolve(x2.astype(np.float64), taps[:, None], mode="same", axes=0)
    return y[:, 0] if x.ndim == 1 else y


def clean_recording(x: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, dict]:
    """Whole-recording cleaning before any windowing (CLAUDE.md §5 rules 10 and 11), at training and serving:
    1. microphone: remove events flagged by the burst threshold (remove_mic_bursts, cfg.mic);
    2. channels 1–7: the fixed band-stop for the ~22 kHz line (cfg.bandstop), if enabled.
    The tachometer (channel 0) is returned untouched. Returns (float32 copy, report)."""
    mic = cfg.channels.index("mic")
    out = np.array(x, dtype=np.float32, copy=True)
    out[:, mic], n_bursts = remove_mic_bursts(out[:, mic], cfg.mic.burst_z, cfg.mic.burst_pad)
    report = {"n_mic_bursts": n_bursts, "bandstop_applied": False, "bandstop_numtaps": 0}
    if cfg.bandstop.enabled:
        taps = bandstop_taps(cfg)
        out[:, 1:] = apply_fir_centred(out[:, 1:], taps).astype(np.float32)
        report.update(bandstop_applied=True, bandstop_numtaps=int(taps.size))
    return out, report
