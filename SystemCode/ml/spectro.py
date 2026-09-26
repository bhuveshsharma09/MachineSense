"""Order-spectrogram and microphone log-Mel PROTOTYPES (Stage 2C-1). CPU only, NumPy; no librosa or torchaudio.

Every candidate starts from clean_recording output and the per-window speeds (window_speeds on the untouched
tachometer). Output per 1 s window: (n_channels, n_orders, T) in dB, on a fixed log-spaced order axis.
"""
from __future__ import annotations

import numpy as np

from ml.config import PreprocessConfig, SpectroCandidate


def order_axis(cfg: PreprocessConfig) -> np.ndarray:
    """The fixed log-spaced order axis (same for every speed)."""
    sc = cfg.spectro
    return np.geomspace(sc.order_min, sc.order_max, sc.n_orders)


def order_bin_edges(cfg: PreprocessConfig) -> np.ndarray:
    """Edges of the log-spaced order bins (geometric midpoints between centres; outer edges half a step out)."""
    o = order_axis(cfg)
    mid = np.sqrt(o[:-1] * o[1:])
    step = o[1] / o[0]
    return np.concatenate([[o[0] / np.sqrt(step)], mid, [o[-1] * np.sqrt(step)]])


def to_order_bins(amp: np.ndarray, axis_values: np.ndarray, centres: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map a fine spectrum (rows × fine points × channels) onto order bins: the MAXIMUM of the fine points inside each
    bin's span, so a line keeps its peak amplitude whichever side of a bin centre it falls; where a bin holds no fine
    point (low orders at low speed), linear interpolation at the bin centre. `axis_values` (fine points, ascending) and
    `centres` / `edges` (rows × bins, rows × bins+1) are in the same unit."""
    rows, _, n_ch = amp.shape
    n_bins = centres.shape[1]
    out = np.empty((rows, n_bins, n_ch))
    for r in range(rows):
        lo = np.searchsorted(axis_values, edges[r, :-1], side="left")
        hi = np.searchsorted(axis_values, edges[r, 1:], side="left")
        filled = np.flatnonzero(hi > lo)
        if filled.size:
            # Bins are contiguous, so reducing between consecutive filled starts covers exactly each bin's points;
            # only the last filled bin must be clipped to its own end.
            red = np.maximum.reduceat(amp[r], lo[filled], axis=0)
            last = filled[-1]
            red[-1] = amp[r, lo[last]:hi[last]].max(axis=0)
            out[r, filled] = red
        empty = np.flatnonzero(hi <= lo)
        for c in range(n_ch):
            if empty.size:
                out[r, empty, c] = np.interp(centres[r, empty], axis_values, amp[r, :, c])
    return out


def frame_centres(n_samples: int, cfg: PreprocessConfig, frames_per_window: int) -> np.ndarray:
    """Sample index of every frame centre, shape (n_windows, T): window_start + (i + 0.5) · win / T."""
    from ml.preprocess import window_starts

    win = int(round(cfg.window.win_s * cfg.fs_hz))
    starts = window_starts(n_samples, cfg)
    step = win / frames_per_window
    return np.round(starts[:, None] + (np.arange(frames_per_window) + 0.5) * step).astype(int)


def _taper(name: str, n: int) -> np.ndarray:
    if name != "hann":
        raise ValueError(f"window {name!r} not supported")
    return np.hanning(n)


def _frames(x: np.ndarray, centres: np.ndarray, length: int) -> np.ndarray:
    """(n_frames, length, channels) slices centred on `centres`, reflect-padding only at the recording ends."""
    half = length // 2
    pad = np.pad(x, ((half + 1, half + 1), (0, 0)), mode="reflect")
    idx = centres.ravel()[:, None] + (half + 1) - half + np.arange(length)[None, :]
    return pad[idx]


def stft_order_spectrogram(xc: np.ndarray, speeds: np.ndarray, cfg: PreprocessConfig,
                           cand: SpectroCandidate) -> np.ndarray:
    """(n_windows, channels, n_orders, T) dB. Each frame: taper, FFT (zero-padded to nfft for interpolation), Hann-
    corrected amplitude, mapped onto the order bins scaled by its window's speed (to_order_bins)."""
    sc, fs = cfg.spectro, cfg.fs_hz
    centres = frame_centres(xc.shape[0], cfg, cand.frames_per_window)
    n_win, T = centres.shape
    fr = _frames(xc, centres, cand.win_samples)  # (n_win*T, L, C)
    w = _taper(cand.window, cand.win_samples)
    fr = fr - fr.mean(axis=1, keepdims=True)
    amp = 2 * np.abs(np.fft.rfft(fr * w[None, :, None], n=cand.nfft, axis=1)) / w.sum()  # (F, bins, C)
    freqs = np.fft.rfftfreq(cand.nfft, 1 / fs)
    sp = np.repeat(speeds, T)[:, None]
    val = to_order_bins(amp, freqs, sp * order_axis(cfg)[None, :], sp * order_bin_edges(cfg)[None, :])
    db = 20 * np.log10(val + sc.eps)
    return db.reshape(n_win, T, sc.n_orders, -1).transpose(0, 3, 2, 1).astype(np.float32)


def _antialias(x: np.ndarray, cutoff_hz: float, stop_hz: float, atten_db: float, fs: float) -> np.ndarray:
    from scipy.signal import firwin, kaiserord

    from ml.preprocess import apply_fir_centred

    n, beta = kaiserord(atten_db, (stop_hz - cutoff_hz) / (fs / 2))
    n += n % 2 == 0
    return apply_fir_centred(x, firwin(n, (cutoff_hz + stop_hz) / 2, window=("kaiser", beta), fs=fs))


def order_resampled_spectrogram(xc: np.ndarray, speeds: np.ndarray, cfg: PreprocessConfig,
                                cand: SpectroCandidate) -> np.ndarray:
    """(n_windows, channels, n_orders, T) dB by computed order tracking. Shaft angle θ(t) integrates a speed profile
    linearly interpolated between the window-centre speeds. For each window, a Kaiser low-pass at
    aa_orders × speed (stop at samples_per_rev/2 × speed) is applied first if that is below the 9.8 kHz band limit;
    then each frame of n_revs revolutions around its centre is resampled uniformly in angle (linear interpolation),
    tapered and transformed. Order resolution = 1 / n_revs regardless of speed; the frame's DURATION is n_revs / speed."""
    from ml.preprocess import window_starts

    sc, fs = cfg.spectro, cfg.fs_hz
    n = xc.shape[0]
    win = int(round(cfg.window.win_s * fs))
    starts = window_starts(n, cfg)
    centres = frame_centres(n, cfg, cand.frames_per_window)
    n_win, T = centres.shape
    t = np.arange(n) / fs
    speed_t = np.interp(t, (starts + win / 2) / fs, speeds)
    theta = np.concatenate([[0.0], np.cumsum(speed_t[:-1]) / fs])  # revolutions
    n_samp = int(round(cand.n_revs * cand.samples_per_rev))
    w = _taper(cand.window, n_samp)
    nfft = n_samp * cand.zero_pad
    ords = np.fft.rfftfreq(nfft, 1 / cand.samples_per_rev)  # orders
    o = order_axis(cfg)
    edges = order_bin_edges(cfg)
    band_limit = cfg.lowpass.pass_hz if cfg.lowpass.enabled else fs / 2
    out = np.empty((n_win, xc.shape[1], sc.n_orders, T), np.float32)
    # Reflect-pad the recording (and extend θ linearly) so frames near the ends have data.
    pad = int(np.ceil(cand.n_revs / 2 / speeds.min() * fs)) + 2
    xp = np.pad(xc, ((pad, pad), (0, 0)), mode="reflect")
    tp = (np.arange(n + 2 * pad) - pad) / fs
    thp = np.concatenate([theta[0] - speeds[0] * (np.arange(pad, 0, -1) / fs), theta,
                          theta[-1] + speeds[-1] * (np.arange(1, pad + 1) / fs)])
    for i in range(n_win):
        cut = cand.aa_orders * speeds[i]
        stop = cand.samples_per_rev / 2 * speeds[i]
        xw = _antialias(xp, cut, stop, cand.aa_atten_db, fs) if cut < band_limit else xp
        for j in range(T):
            th_c = np.interp(centres[i, j] / fs, tp, thp)
            th = th_c + (np.arange(n_samp) - n_samp / 2) / cand.samples_per_rev
            tt = np.interp(th, thp, tp)
            seg = np.stack([np.interp(tt, tp, xw[:, c]) for c in range(xw.shape[1])], axis=1)
            seg = seg - seg.mean(axis=0)
            amp = 2 * np.abs(np.fft.rfft(seg * w[:, None], n=nfft, axis=0)) / w.sum()
            val = to_order_bins(amp[None], ords, o[None, :], edges[None, :])[0]  # (n_orders, C)
            out[i, :, :, j] = 20 * np.log10(val.T + sc.eps)
    return out


def order_spectrogram(xc: np.ndarray, speeds: np.ndarray, cfg: PreprocessConfig, name: str) -> np.ndarray:
    """Candidate `name` on the accelerometer channels of a cleaned recording: (n_windows, 6, n_orders, T) dB."""
    cand = cfg.spectro.candidates[name]
    acc = xc[:, [cfg.channels.index(c) for c in cfg.spectro.channels]].astype(np.float64)
    fn = stft_order_spectrogram if cand.kind == "stft" else order_resampled_spectrogram
    return fn(acc, np.asarray(speeds, dtype=np.float64), cfg, cand)


def mel_filterbank(cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """HTK-style triangular Mel filterbank (mel_n × (mel_nfft/2 + 1)) from mel_fmin_hz to mel_fmax_hz, and the
    filter centre frequencies."""
    sc, fs = cfg.spectro, cfg.fs_hz
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = hz(np.linspace(mel(sc.mel_fmin_hz), mel(sc.mel_fmax_hz), sc.mel_n + 2))
    freqs = np.fft.rfftfreq(sc.mel_nfft, 1 / fs)
    fb = np.zeros((sc.mel_n, len(freqs)))
    for k in range(sc.mel_n):
        lo, c, hi = pts[k], pts[k + 1], pts[k + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[k] = np.clip(np.minimum(up, down), 0, None)
    return fb, pts[1:-1]


def log_mel(mic_clean: np.ndarray, cfg: PreprocessConfig, frames_per_window: int) -> np.ndarray:
    """(n_windows, n_mels, T) dB: power spectra of mel_win_samples Hann frames centred at the SAME times as the
    candidate's frames, through the Mel filterbank, 10·log10(power + eps)."""
    sc = cfg.spectro
    centres = frame_centres(mic_clean.shape[0], cfg, frames_per_window)
    n_win, T = centres.shape
    fr = _frames(mic_clean[:, None].astype(np.float64), centres, sc.mel_win_samples)[:, :, 0]
    w = np.hanning(sc.mel_win_samples)
    fr = fr - fr.mean(axis=1, keepdims=True)
    power = (2 * np.abs(np.fft.rfft(fr * w, n=sc.mel_nfft, axis=1)) / w.sum()) ** 2 / 2
    fb, _ = mel_filterbank(cfg)
    m = power @ fb.T
    return (10 * np.log10(m + sc.mel_eps)).reshape(n_win, T, sc.mel_n).transpose(0, 2, 1).astype(np.float32)


def recording_representation(x_raw: np.ndarray, fr_hz: float, cfg: PreprocessConfig, name: str) -> dict:
    """Prototype entry point: clean the recording, per-window speeds from the untouched tachometer, then the
    candidate's accelerometer order-spectrograms and the time-aligned microphone log-Mel."""
    from ml.preprocess import TACHO, clean_recording, window_speeds_detail

    xc, _ = clean_recording(x_raw, cfg)
    speeds, fell_back = window_speeds_detail(np.asarray(x_raw[:, TACHO], dtype=np.float64), cfg.fs_hz,
                                             cfg.window.win_s, cfg.window.hop_s, cfg, fr_fallback_hz=fr_hz)
    cand = cfg.spectro.candidates[name]
    return {"order": order_spectrogram(xc, speeds, cfg, name),
            "mel": log_mel(xc[:, cfg.channels.index("mic")], cfg, cand.frames_per_window),
            "speeds": speeds, "fallback": fell_back}
