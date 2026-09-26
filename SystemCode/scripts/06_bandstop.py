"""Set, check and verify the fixed band-stop for the ~22,028 Hz line (Stage 2A item 2). Heavy loops for notebook 03.

    python scripts/06_bandstop.py measure    # tone + sidebands on split_closed TRAIN NORMAL recordings → proposed band
    python scripts/06_bandstop.py check      # same measurement on every other train/val recording (no retuning)
    python scripts/06_bandstop.py verify     # real train recordings: tone attenuation, out-of-band change, edge extent
    python scripts/06_bandstop.py shortcut   # per-channel stop-band energy before/after cleaning (train + val)
    python scripts/06_bandstop.py all

Test recordings are never read. Outputs go to data/cache/06_<task>_<sha12>.parquet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import PreprocessConfig, load_config
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, cache_file, data_dir
from ml.preprocess import apply_fir_centred, bandstop_taps, measure_edge_effect, propose_stopband, remove_mic_bursts

SEARCH_HZ = (21_800.0, 22_300.0)  # main line search range (the line drifts by session; see notebook 03)
SIDEBANDS = (-120.0, -60.0, 60.0, 120.0)
SIDEBAND_SEARCH = 5.0  # ± Hz around f0 + offset
PRESENT_DB = 20.0  # line counts as present if this far above the 21.5–22.5 kHz median
MARGIN_HZ = 25.0  # stated margin beyond the observed extent (≈ the cross-recording spread of the line)
ROUND_HZ = 5.0
EDGE_SEG = 50  # samples (1 ms) per segment for the edge-effect measurement
ADJACENT_HZ = ((21_300.0, 21_600.0), (22_450.0, 22_750.0))  # outside the stop band and its transitions
VERIFY_PER_CLASS = 3


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def _spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(x.shape[0])
    amp = 2 * np.abs(np.fft.rfft((x - x.mean(axis=0)) * w[:, None], axis=0)) / w.sum()
    return np.fft.rfftfreq(x.shape[0], 1 / fs), amp


def _peak(f: np.ndarray, a: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Frequency (parabolic interpolation on log amplitude) and amplitude of the largest bin in [lo, hi]."""
    idx = np.flatnonzero((f >= lo) & (f <= hi))
    i = idx[np.argmax(a[idx])]
    la, lb, lc = np.log(a[i - 1: i + 2] + 1e-15)
    d = la - 2 * lb + lc
    return float(f[i] + (0.5 * (la - lc) / d if d else 0.0) * (f[1] - f[0])), float(a[i])


def measure_one(args: tuple[str, PreprocessConfig]) -> list[dict]:
    """Main line and ±60/±120 Hz sidebands per channel 1–7 (mic cleaned first), plus within-recording drift."""
    rec_id, pre = args
    x = load(rec_id).astype(np.float64)
    mic = pre.channels.index("mic")
    x[:, mic] = remove_mic_bursts(x[:, mic], pre.mic.burst_z, pre.mic.burst_pad)[0]
    f, amp = _spectrum(x, pre.fs_hz)
    fs = pre.fs_hz
    rows = []
    for c in range(1, x.shape[1]):
        a = amp[:, c]
        floor = float(np.median(a[(f > 21_500) & (f < 22_500)]))
        f0, a0 = _peak(f, a, *SEARCH_HZ)
        row = {"rec_id": rec_id, "channel": pre.channels[c], "f0_hz": f0, "snr_db": 20 * np.log10(a0 / floor)}
        for off in SIDEBANDS:
            fsb, asb = _peak(f, a, f0 + off - SIDEBAND_SEARCH, f0 + off + SIDEBAND_SEARCH)
            row[f"f{int(off):+d}_hz"] = fsb
            row[f"snr{int(off):+d}_db"] = 20 * np.log10(asb / floor)
        wf = []  # f0 per 1 s window (8× zero-padding for interpolation only)
        for k in range(x.shape[0] // fs):
            seg = x[k * fs:(k + 1) * fs, c]
            s = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(fs), n=8 * fs))
            ff = np.fft.rfftfreq(8 * fs, 1 / fs)
            b = (ff >= SEARCH_HZ[0]) & (ff <= SEARCH_HZ[1])
            wf.append(ff[b][np.argmax(s[b])])
        row["drift_hz"] = float(max(wf) - min(wf))
        rows.append(row)
    return rows


def verify_one(args: tuple[str, PreprocessConfig]) -> dict:
    """Real-recording verification of the enabled band-stop on channels 1–7 (mic cleaned first in both)."""
    rec_id, pre = args
    x = load(rec_id).astype(np.float64)
    mic = pre.channels.index("mic")
    x[:, mic] = remove_mic_bursts(x[:, mic], pre.mic.burst_z, pre.mic.burst_pad)[0]
    y = apply_fir_centred(x[:, 1:], bandstop_taps(pre))
    x = x[:, 1:]
    b, fs = pre.bandstop, pre.fs_hz
    half = (b.numtaps - 1) // 2
    core = slice(half, x.shape[0] - half)
    f, ax = _spectrum(x[core], fs)
    _, ay = _spectrum(y[core], fs)
    out = {"rec_id": rec_id}
    # Line attenuation per channel over the recording without its filter edges, where the line is present.
    att = []
    for c in range(x.shape[1]):
        floor = np.median(ax[(f > 21_500) & (f < 22_500), c])
        f0, a0 = _peak(f, ax[:, c], *SEARCH_HZ)
        if 20 * np.log10(a0 / floor) > PRESENT_DB:
            i = np.argmin(np.abs(f - f0))
            att.append(20 * np.log10(ax[i, c] / (ay[i, c] + 1e-15)))
    out["tone_atten_db_min"] = float(min(att)) if att else np.nan
    out["tone_atten_db_median"] = float(np.median(att)) if att else np.nan
    # Change outside the stop band and its transitions: Welch-free, band-averaged amplitude ratio in 100 Hz bands.
    outside = (f > 20) & ((f < b.stop_lo_hz - b.transition_hz) | (f > b.stop_hi_hz + b.transition_hz)) & (f < fs / 2 - 100)
    edges = np.arange(0, fs / 2, 100.0)
    band = np.digitize(f, edges)
    ratios = []
    for k in np.unique(band[outside]):
        sel = outside & (band == k)
        ratios.append(20 * np.log10(np.sqrt((ay[sel] ** 2).mean(axis=0)) / np.sqrt((ax[sel] ** 2).mean(axis=0))))
    ratios = np.abs(np.array(ratios))
    out["outside_change_db_max"] = float(ratios.max())
    out["outside_change_db_median"] = float(np.median(ratios))
    # Edge extent, measured directly (ml.preprocess.measure_edge_effect: crop vs full-recording filtering).
    out.update(measure_edge_effect(x, bandstop_taps(pre), fs, seg=EDGE_SEG))
    return out


def shortcut_one(args: tuple[str, PreprocessConfig]) -> dict:
    """Energy (V²) in the stop band per channel 1–7, before (mic cleaned) and after the band-stop; plus, as a
    diagnostic, the energy in two bands next to it (outside the transitions), which the filter does not touch."""
    rec_id, pre = args
    x = load(rec_id)
    raw = x.astype(np.float64)
    mic = pre.channels.index("mic")
    raw[:, mic] = remove_mic_bursts(raw[:, mic], pre.mic.burst_z, pre.mic.burst_pad)[0]
    clean = raw.copy()  # mic cleaned, then the band-stop alone (independent of the low-pass setting)
    clean[:, 1:] = apply_fir_centred(clean[:, 1:], bandstop_taps(pre))
    b = pre.bandstop
    row = {"rec_id": rec_id}
    for name, sig in (("before", raw), ("after", clean)):
        f, amp = _spectrum(sig, pre.fs_hz)
        sel = (f >= b.stop_lo_hz) & (f <= b.stop_hi_hz)
        adj = np.zeros_like(f, dtype=bool)
        for lo, hi in ADJACENT_HZ:
            adj |= (f >= lo) & (f <= hi)
        for c in range(1, sig.shape[1]):
            row[f"{name}_{pre.channels[c]}"] = float((amp[sel, c] ** 2).sum())
            row[f"{name}_adjacent_{pre.channels[c]}"] = float((amp[adj, c] ** 2).sum())  # diagnostic only
    return row


def run(task: str, meta: pd.DataFrame, pre: PreprocessConfig, workers: int) -> None:
    train_normal = meta[(meta.split_closed == "train") & (meta.class10 == "normal")]
    trval = meta[meta.split_closed != "test"]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        if task in ("measure", "check"):
            sel = train_normal if task == "measure" else trval[~trval.rec_id.isin(train_normal.rec_id)]
            out = cache_file(f"06_{task}", pre.mic, ext="parquet")
            rows = [r for rs in pool.map(measure_one, [(r, pre) for r in sel.rec_id], chunksize=2) for r in rs]
            df = pd.DataFrame(rows).merge(sel[["rec_id", "class10", "severity", "split_closed"]])
            df.to_parquet(out, index=False)
            if task == "measure":
                print(propose_stopband(df, PRESENT_DB, MARGIN_HZ, ROUND_HZ))
        elif task in ("verify", "shortcut"):
            if not pre.bandstop.enabled:
                raise SystemExit("enable bandstop in configs/base.yaml first")
            if task == "verify":
                sel = (meta[meta.split_closed == "train"].sort_values("rec_id")
                       .groupby("class10").sample(n=VERIFY_PER_CLASS, random_state=0))
                fn, cols = verify_one, ["rec_id", "class10"]
            else:
                sel, fn, cols = trval, shortcut_one, ["rec_id", "class10", "cv_fold"]
            out = cache_file(f"06_{task}", pre.mic, pre.bandstop, ext="parquet")
            df = pd.DataFrame(list(pool.map(fn, [(r, pre) for r in sel.rec_id], chunksize=2))).merge(sel[cols])
            df.to_parquet(out, index=False)
        else:
            raise SystemExit(f"unknown task {task}")
    print(f"{task}: {len(sel)} recordings in {time.perf_counter() - t0:.0f} s → {out.relative_to(SYSTEM_CODE_DIR)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["measure", "check", "verify", "shortcut", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    pre = load_config(DEFAULT_CONFIG).preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    (data_dir() / "cache").mkdir(parents=True, exist_ok=True)
    for task in (["measure", "check", "verify", "shortcut"] if args.task == "all" else [args.task]):
        run(task, meta, pre, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
