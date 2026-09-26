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
