"""Spectral building blocks: amplitude and order spectra, band-limited envelopes, band kurtosis, comb scores.

Bearing evidence comes only from envelope spectra (CLAUDE.md §3); never low-pass before the envelope.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ml.config import PreprocessConfig


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum (V) of a Hann-windowed signal. Returns (frequencies in Hz, amplitudes)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(x.size)
    amp = 2 * np.abs(np.fft.rfft((x - x.mean()) * w)) / w.sum()
    return np.fft.rfftfreq(x.size, 1 / fs), amp


def band_analytic(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Analytic (complex) signal of x restricted to [lo, hi] Hz, by masking the FFT: band-pass + Hilbert in one step."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.fft(x - x.mean())
    f = np.fft.fftfreq(x.size, 1 / fs)
    keep = (f >= lo) & (f <= hi)  # positive frequencies only → analytic
    spec = np.where(keep, 2 * spec, 0)
    return np.fft.ifft(spec)


def band_kurtosis(x: np.ndarray, fs: float, bands: list[tuple[float, float]]) -> np.ndarray:
    """Spectral kurtosis of each band: E|y|⁴ / (E|y|²)² − 2 for the band's analytic signal y (0 for Gaussian noise).

    Impulsive bearing impacts excite a resonance band, which shows as high kurtosis there.
    """
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.fft(x - x.mean())
    f = np.fft.fftfreq(x.size, 1 / fs)
    out = np.empty(len(bands))
    for i, (lo, hi) in enumerate(bands):
        y = np.fft.ifft(np.where((f >= lo) & (f <= hi), 2 * spec, 0))
        p = np.abs(y) ** 2
        out[i] = np.mean(p**2) / np.mean(p) ** 2 - 2
    return out


def band_grid(fs: float, widths_hz: tuple[float, ...], min_lo_hz: float) -> list[tuple[float, float]]:
    """Fixed band grid for spectral kurtosis: each width stepped by half its width, lower edge ≥ min_lo_hz."""
    nyq = fs / 2
    bands = []
    for bw in widths_hz:
        lo = min_lo_hz
        while lo + bw <= nyq:
            bands.append((lo, lo + bw))
            lo += bw / 2
    return bands


def envelope_order_spectrum(x: np.ndarray, fs: float, fr_hz: float, band: tuple[float, float],
                            orders: np.ndarray) -> np.ndarray:
    """Envelope spectrum (band-pass → Hilbert → |·| → FFT) resampled onto `orders` (multiples of shaft speed)."""
    env = np.abs(band_analytic(x, fs, *band))
    f, amp = amplitude_spectrum(env, fs)
    return np.interp(orders * fr_hz, f, amp)


def order_spectrum(x: np.ndarray, fs: float, fr_hz: float, orders: np.ndarray) -> np.ndarray:
    """Amplitude spectrum resampled onto `orders`, so recordings at different speeds line up."""
    f, amp = amplitude_spectrum(x, fs)
    return np.interp(orders * fr_hz, f, amp)


def line_prominence_db(orders: np.ndarray, spec: np.ndarray, lines: np.ndarray, tol: float,
                       bg_halfwidth: float) -> np.ndarray:
    """For each line (in orders): peak within ±tol over the median background within ±bg_halfwidth (excluding
    the peak zone), in dB. Measures how much a line stands out, independent of overall level."""
    out = np.empty(len(lines))
    for i, o in enumerate(lines):
        d = np.abs(orders - o)
        peak = spec[d <= tol].max(initial=0.0)
        bg = np.median(spec[(d > tol) & (d <= bg_halfwidth)])
        out[i] = 20 * np.log10((peak + 1e-12) / (bg + 1e-12))
    return out


def comb_lines(cfg: PreprocessConfig) -> dict[str, np.ndarray]:
    """Envelope-order lines per bearing fault family: FTF harmonics (cage), 2×BSF ± FTF (ball),
    BPFO harmonics (outer race) and BPFI harmonics with ±1× shaft sidebands (inner race)."""
    b, c = cfg.bearing, cfg.comb
    return {
        "ftf": np.array([h * b.ftf for h in c.ftf_harmonics]),
        "bsf": np.array([c.bsf_multiple * b.bsf + k * b.ftf for k in c.bsf_ftf_sidebands]),
        "bpfo": np.array([h * b.bpfo for h in c.bpfo_harmonics]),
        "bpfi": np.array([h * b.bpfi + k for h in c.bpfi_harmonics for k in c.bpfi_shaft_sidebands]),
    }


def candidate_bands(cfg: PreprocessConfig) -> list[tuple[float, float]]:
    """Demodulation-band candidates: the spectral-kurtosis grid, keeping bands at least min_band_hz wide."""
    env = cfg.envelope
    return [(lo, hi) for lo, hi in band_grid(cfg.fs_hz, env.sk.widths_hz, env.sk.min_lo_hz)
            if hi - lo >= env.min_band_hz]


def spectral_kurtosis_table(rec_ids: list[str], cfg: PreprocessConfig,
                            load: Callable[[str], np.ndarray]) -> pd.DataFrame:
    """Spectral kurtosis of every candidate band, per recording and selection channel.

    Columns: rec_id, channel, lo_hz, hi_hz, sk. `load(rec_id)` returns the (n_rows, n_channels) recording.
    """
    bands = candidate_bands(cfg)
    rows = []
    for rid in rec_ids:
        x = load(rid)
        for ch in cfg.envelope.selection.channels:
            sk = band_kurtosis(x[:, cfg.channels.index(ch)], cfg.fs_hz, bands)
            rows += [{"rec_id": rid, "channel": ch, "lo_hz": lo, "hi_hz": hi, "sk": float(v)}
                     for (lo, hi), v in zip(bands, sk)]
    return pd.DataFrame(rows)


def select_demod_band(recordings: pd.DataFrame, sk: pd.DataFrame, cfg: PreprocessConfig) -> dict[str, dict]:
    """Choose the envelope demodulation band per channel from one evaluation track's TRAINING recordings.

    Rule (parameters in cfg.envelope.selection and cfg.envelope):
      1. Groups. Fault groups: recordings of each class in `fault_classes` with severity in `severities`
         (default: underhang and overhang outer race at 0 g). Reference: recordings of `reference_class`
         (normal). Only recordings present in `recordings` count, so a class withheld from the track is
         excluded by construction. A group with fewer than `min_recordings` recordings is dropped.
      2. Candidates. The spectral-kurtosis grid (widths `sk.widths_hz`, stepped by half a width, lower edge
         ≥ `sk.min_lo_hz`), keeping bands at least `min_band_hz` wide (the envelope must carry 12 orders at
         61 Hz).
      3. Score. For each channel, fault group and band: median spectral kurtosis of the fault group minus
         median spectral kurtosis of the reference. The contrast, not the raw value, because healthy
         machines are impulsive in some bands too.
      4. Choice. Per channel, the (fault group, band) with the largest contrast. Scoring each fault group
         separately lets a channel follow the bearing position it responds to. Ties: lower band edge, then
         narrower band.
    Raises ValueError if the reference or every fault group is too small.

    `sk` must hold spectral kurtosis (ml.features.spectral_kurtosis_table) for at least the grouped recordings.
    Returns {channel: {lo_hz, hi_hz, contrast, fault_class, n_fault, n_reference, rec_ids}}.
    """
    sel = cfg.envelope.selection
    rec = recordings[["rec_id", "class10", "severity"]]
    ref_ids = rec.loc[rec.class10 == sel.reference_class, "rec_id"]
    if len(ref_ids) < sel.min_recordings:
        raise ValueError(f"only {len(ref_ids)} {sel.reference_class} recordings; need {sel.min_recordings}")
    groups = {}
    for cls in sel.fault_classes:
        ids = rec.loc[(rec.class10 == cls) & rec.severity.isin(sel.severities), "rec_id"]
        if len(ids) >= sel.min_recordings:
            groups[cls] = ids
    if not groups:
        raise ValueError("no fault group has enough recordings")

    allowed = set(candidate_bands(cfg))
    sk = sk[[(lo, hi) in allowed for lo, hi in zip(sk.lo_hz, sk.hi_hz)]]
    out = {}
    for ch in sel.channels:
        s = sk[sk.channel == ch]
        ref = s[s.rec_id.isin(ref_ids)].groupby(["lo_hz", "hi_hz"]).sk.median()
        best = None
        for cls, ids in groups.items():
            got = s[s.rec_id.isin(ids)]
            if got.rec_id.nunique() != len(ids) or s[s.rec_id.isin(ref_ids)].rec_id.nunique() != len(ref_ids):
                raise ValueError(f"sk table is missing recordings for {ch}")
            contrast = (got.groupby(["lo_hz", "hi_hz"]).sk.median() - ref).dropna()
            for (lo, hi), c in contrast.items():
                key = (-c, lo, hi - lo)
                if best is None or key < best[0]:
                    best = (key, lo, hi, c, cls)
        _, lo, hi, c, cls = best
        out[ch] = {"lo_hz": float(lo), "hi_hz": float(hi), "contrast": round(float(c), 4), "fault_class": cls,
                   "n_fault": len(groups[cls]), "n_reference": len(ref_ids),
                   "rec_ids": sorted([*groups[cls], *ref_ids])}
    return out



# ============================================================================ R1 physics features (Stage 2A)
ACCEL_CHANNELS = ("under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential")
FEATURE_PREFIX = "f_"


def _order_name(o: float) -> str:
    return f"o{o:g}".replace(".", "p")


def envelope_line_names(cfg: PreprocessConfig) -> list[tuple[str, float]]:
    """(name, order) of every envelope comb line: FTF harmonics, 2×BSF ± FTF, BPFO harmonics, BPFI harmonics ± 1×."""
    b, c = cfg.bearing, cfg.comb
    out = [(f"ftf{h}", h * b.ftf) for h in c.ftf_harmonics]
    out += [(f"bsf2{'' if k == 0 else ('_m' if k < 0 else '_p') + str(abs(k))}", c.bsf_multiple * b.bsf + k * b.ftf)
            for k in c.bsf_ftf_sidebands]
    out += [(f"bpfo{h}", h * b.bpfo) for h in c.bpfo_harmonics]
    out += [(f"bpfi{h}{'' if k == 0 else ('_m' if k < 0 else '_p') + str(abs(k))}", h * b.bpfi + k)
            for h in c.bpfi_harmonics for k in c.bpfi_shaft_sidebands]
    return out


def line_tolerance_hz(f_line_hz: float, cfg: PreprocessConfig) -> float:
    """Half-width (Hz) of the search window around a spectral line: max(k bins, relative fraction of the line)."""
    fc = cfg.features
    return max(fc.order_tol_bins / cfg.window.win_s, fc.order_tol_rel * f_line_hz)


def window_spectrum(x: np.ndarray, fs: float, zero_pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed single-sided amplitude spectrum (sinusoid amplitude, same units as x) along axis 0, zero-padded
    by `zero_pad` for interpolation. Returns (frequencies, amplitudes)."""
    n = x.shape[0]
    w = np.hanning(n)
    xw = (x - x.mean(axis=0)) * (w if x.ndim == 1 else w[:, None])
    amp = 2 * np.abs(np.fft.rfft(xw, n=n * zero_pad, axis=0)) / w.sum()
    return np.fft.rfftfreq(n * zero_pad, 1 / fs), amp


def line_peak(freqs: np.ndarray, amp: np.ndarray, f_line: float, tol_hz: float) -> np.ndarray:
    """Largest amplitude within ±tol_hz of f_line (per column if amp is 2-D)."""
    sel = (freqs >= f_line - tol_hz) & (freqs <= f_line + tol_hz)
    return amp[sel].max(axis=0)


def physics_features(signal_win: np.ndarray, envelope_win: np.ndarray, speed_hz: float,
                     cfg: PreprocessConfig) -> dict[str, float]:
    """R1 features of one window. signal_win: (win_samples, channels) cleaned, DC removed; envelope_win:
    (win_samples, len(features.envelope_channels)) slices of the whole-recording envelopes; speed_hz: this window's
    shaft speed. Every key starts with f_; see feature_dictionary for units and definitions."""
    from scipy.stats import kurtosis

    fs, fc = cfg.fs_hz, cfg.features
    out: dict[str, float] = {}
    acc_idx = [cfg.channels.index(c) for c in ACCEL_CHANNELS]
    acc = signal_win[:, acc_idx].astype(np.float64)

    # Accelerometers: time-domain statistics.
    rms = np.sqrt(np.mean(acc**2, axis=0))
    peak = np.abs(acc).max(axis=0)
    kurt = kurtosis(acc, axis=0, fisher=True)
    # Accelerometers: order amplitudes on the window's own speed.
    f, amp = window_spectrum(acc, fs, fc.zero_pad)
    orders = {o: line_peak(f, amp, o * speed_hz, line_tolerance_hz(o * speed_hz, cfg)) for o in fc.accel_orders}
    for j, ch in enumerate(ACCEL_CHANNELS):
        out[f"f_{ch}_rms"] = rms[j]
        out[f"f_{ch}_peak"] = peak[j]
        out[f"f_{ch}_crest"] = peak[j] / rms[j] if rms[j] > 0 else np.nan
        out[f"f_{ch}_kurtosis"] = kurt[j]
        for o in fc.accel_orders:
            out[f"f_{ch}_{_order_name(o)}"] = orders[o][j]
        out[f"f_{ch}_ratio_2x_1x"] = orders[2][j] / orders[1][j]
    for bearing in ("under", "over"):
        ax, rad = ACCEL_CHANNELS.index(f"{bearing}_axial"), ACCEL_CHANNELS.index(f"{bearing}_radial")
        out[f"f_{bearing}_ratio_axial_radial_1x"] = orders[1][ax] / orders[1][rad]

    # Envelope comb lines, each relative to the median of the envelope spectrum over envelope_ref_orders.
    fe, ae = window_spectrum(envelope_win.astype(np.float64), fs, fc.zero_pad)
    f1, a1 = window_spectrum(envelope_win.astype(np.float64), fs, 1)  # unpadded grid for the median
    lo, hi = fc.envelope_ref_orders
    ref = np.median(a1[(f1 >= lo * speed_hz) & (f1 <= hi * speed_hz)], axis=0)
    for name, o in envelope_line_names(cfg):
        vals = line_peak(fe, ae, o * speed_hz, line_tolerance_hz(o * speed_hz, cfg)) / ref
        for j, ch in enumerate(fc.envelope_channels):
            out[f"f_env_{ch}_{name}"] = vals[j]

    # Microphone (after burst removal): RMS and spectral shape below mic_max_hz. No peak, crest or kurtosis (rule 10).
    m = signal_win[:, cfg.channels.index("mic")].astype(np.float64)
    fm, am = window_spectrum(m, fs, 1)
    pw = am**2 / 2  # mean-square per bin
    keep = (fm >= fc.mic_bands_hz[0][0]) & (fm <= fc.mic_max_hz)
    tot = pw[keep].sum()
    out["f_mic_rms"] = float(np.sqrt(np.mean(m**2)))
    out["f_mic_centroid_hz"] = float((fm[keep] * pw[keep]).sum() / tot)
    out["f_mic_rolloff_hz"] = float(fm[keep][np.searchsorted(np.cumsum(pw[keep]), fc.mic_rolloff * tot)])
    for blo, bhi in fc.mic_bands_hz:
        out[f"f_mic_logband_{blo:g}_{bhi:g}"] = float(np.log10(pw[(fm >= blo) & (fm < bhi)].sum() + 1e-20))
    return {k: float(v) for k, v in out.items()}


def recording_features(x_raw: np.ndarray, fr_hz: float, cfg: PreprocessConfig) -> pd.DataFrame:
    """The one feature entry point, for batch and serving alike. One row per window (9), in this order:
    clean the whole recording → per-window speeds from the UNTOUCHED tachometer (fallback: fr_hz) → whole-recording
    envelopes (primary band, band-pass + Hilbert) on one radial channel per bearing → aligned signal and envelope
    windows → physics_features. Columns: window_idx, speed_hz, speed_fallback (metadata, no prefix) and f_* features."""
    from ml.preprocess import TACHO, clean_recording, window, window_speeds_detail, window_starts

    x_clean, _ = clean_recording(x_raw, cfg)
    speeds, fell_back = window_speeds_detail(np.asarray(x_raw[:, TACHO], dtype=np.float64), cfg.fs_hz,
                                             cfg.window.win_s, cfg.window.hop_s, cfg, fr_fallback_hz=fr_hz)
    lo, hi = cfg.envelope.primary_band_hz
    env = np.stack([np.abs(band_analytic(x_clean[:, cfg.channels.index(ch)], cfg.fs_hz, lo, hi))
                    for ch in cfg.features.envelope_channels], axis=1)
    wins = window(x_clean, cfg)
    starts = window_starts(x_clean.shape[0], cfg)
    n = wins.shape[1]
    rows = []
    for i, s in enumerate(starts):
        feats = physics_features(wins[i], env[s: s + n], float(speeds[i]), cfg)
        rows.append({"window_idx": i, "speed_hz": float(speeds[i]), "speed_fallback": bool(fell_back[i]), **feats})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model inputs: every column with the f_ prefix, and nothing else."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def feature_dictionary(cfg: PreprocessConfig) -> pd.DataFrame:
    """name, unit, definition, channel for every f_ column physics_features produces (same naming as the code)."""
    fc = cfg.features
    tol = (f"peak within ±max({fc.order_tol_bins:g} bins, {fc.order_tol_rel:.1%} of the line frequency), "
           f"Hann window, {fc.zero_pad}× zero-padding for interpolation")
    rows = []
    for ch in ACCEL_CHANNELS:
        rows += [(f"f_{ch}_rms", "V", "RMS of the window (DC removed)", ch),
                 (f"f_{ch}_peak", "V", "largest |sample| in the window", ch),
                 (f"f_{ch}_crest", "ratio", "peak / RMS", ch),
                 (f"f_{ch}_kurtosis", "ratio", "excess kurtosis (Fisher) of the window", ch)]
        rows += [(f"f_{ch}_{_order_name(o)}", "V", f"amplitude at {o:g}× the window's shaft speed; {tol}", ch)
                 for o in fc.accel_orders]
        rows.append((f"f_{ch}_ratio_2x_1x", "ratio", "2× order amplitude / 1× order amplitude", ch))
    for bearing in ("under", "over"):
        rows.append((f"f_{bearing}_ratio_axial_radial_1x", "ratio", "1× amplitude, axial / radial",
                     f"{bearing}_axial, {bearing}_radial"))
    lo, hi = cfg.envelope.primary_band_hz
    rlo, rhi = fc.envelope_ref_orders
    for name, o in envelope_line_names(cfg):
        for ch in fc.envelope_channels:
            rows.append((f"f_env_{ch}_{name}", "ratio",
                         f"envelope spectrum ({lo / 1e3:g}–{hi / 1e3:g} kHz band-pass + Hilbert) at {o:.3f} orders, "
                         f"divided by its median over {rlo:g}–{rhi:g} orders; {tol}", ch))
    rows += [("f_mic_rms", "V", "RMS of the cleaned microphone window", "mic"),
             ("f_mic_centroid_hz", "Hz", f"power-weighted mean frequency, {fc.mic_bands_hz[0][0]:g} Hz–"
                                          f"{fc.mic_max_hz / 1e3:g} kHz", "mic"),
             ("f_mic_rolloff_hz", "Hz", f"frequency below which {fc.mic_rolloff:.0%} of the power lies "
                                         f"(same range)", "mic")]
    rows += [(f"f_mic_logband_{blo:g}_{bhi:g}", "log10 V²", f"log10 of the power in {blo:g}–{bhi:g} Hz", "mic")
             for blo, bhi in fc.mic_bands_hz]
    return pd.DataFrame(rows, columns=["name", "unit", "definition", "channel"])


def line_overlaps(speed_hz: float, cfg: PreprocessConfig) -> pd.DataFrame:
    """Pairs of spectral lines whose search windows overlap at this shaft speed: accelerometer orders, and envelope
    comb lines. Overlapping lines can return the same peak."""
    rows = []
    for family, lines in (("accel orders", [(_order_name(o), o) for o in cfg.features.accel_orders]),
                          ("envelope lines", envelope_line_names(cfg))):
        for i, (na, oa) in enumerate(lines):
            for nb, ob in lines[i + 1:]:
                fa, fb = oa * speed_hz, ob * speed_hz
                gap = abs(fa - fb) - line_tolerance_hz(fa, cfg) - line_tolerance_hz(fb, cfg)
                if gap < 0:
                    rows.append({"family": family, "line_a": na, "line_b": nb, "orders_apart": abs(oa - ob),
                                 "overlap_hz": -gap})
    return pd.DataFrame(rows, columns=["family", "line_a", "line_b", "orders_apart", "overlap_hz"])
