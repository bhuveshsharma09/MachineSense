"""Spectral building blocks: amplitude and order spectra, band-limited envelopes, band kurtosis, comb scores.

Bearing evidence comes only from envelope spectra (CLAUDE.md §3); never low-pass before the envelope.
"""
from __future__ import annotations

import numpy as np

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
