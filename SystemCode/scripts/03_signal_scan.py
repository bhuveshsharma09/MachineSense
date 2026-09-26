"""Heavy loops for notebook 02 (signal exploration). Each task writes one cache file under data/cache/.

    python scripts/03_signal_scan.py mic        # mic statistics before/after burst removal (train + val)
    python scripts/03_signal_scan.py meanspec   # mean amplitude spectra per channel (stratified train subset)
    python scripts/03_signal_scan.py sk         # spectral kurtosis band grid (train outer race 0 g + normal)
    python scripts/03_signal_scan.py envelope   # envelope order spectra (train + val, 0 g bearings + normal)
    python scripts/03_signal_scan.py sensor     # > 1 kHz RMS and kurtosis per accelerometer (train, 0 g bearings + normal)
    python scripts/03_signal_scan.py bandcompare  # envelope spectra in the closed-track band and in 1–5 kHz (train only)
    python scripts/03_signal_scan.py all

Test recordings are never read. Cache names carry the sha256 of the config sections each task depends on.
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
from scipy.signal import welch
from scipy.stats import kurtosis

from ml.config import PreprocessConfig, load_config
from ml.features import band_grid, band_kurtosis, envelope_order_spectrum
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, cache_file, data_dir
from ml.preprocess import remove_mic_bursts

MIC = 7
ACCELS = [1, 2, 3, 4, 5, 6]
RADIAL = {"under_radial": 2, "over_radial": 5}
MEANSPEC_PER_CLASS = 15
MEANSPEC_NPERSEG = 8192
RESPONSE_HALF = 100  # samples (±2 ms) around a burst to look for an accelerometer response
SENSOR_HP_HZ = 1000.0  # bearing impacts live above the shaft harmonics


def cache_path(task: str, *sections) -> Path:
    """Cache for one task: data/cache/02_<task>_<sha12>.<ext> (see ml.paths.cache_file)."""
    return cache_file(f"02_{task}", *sections, ext="parquet" if task in ("mic", "sk", "sensor") else "npz")


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    return np.abs(x - med) / (1.4826 * np.median(np.abs(x - med)))


def _mic_stats(m: np.ndarray, prefix: str) -> dict:
    m = m.astype(np.float64)
    rms = np.sqrt(np.mean((m - m.mean()) ** 2))
    return {f"{prefix}max_abs": float(np.abs(m).max()), f"{prefix}peakz": float(_robust_z(m).max()),
            f"{prefix}crest": float(np.abs(m - m.mean()).max() / rms), f"{prefix}kurtosis": float(kurtosis(m))}


# ----------------------------------------------------------------------------- tasks (one recording each)
def mic_one(args: tuple[str, PreprocessConfig]) -> dict:
    rec_id, pre = args
    x = load(rec_id)
    mic = x[:, MIC]
    clean, n_bursts = remove_mic_bursts(mic, pre.mic.burst_z, pre.mic.burst_pad)
    row = {"rec_id": rec_id, "n_bursts": n_bursts, **_mic_stats(mic, "raw_"), **_mic_stats(clean, "clean_")}

    hit = _robust_z(mic) > pre.mic.burst_z
    if hit.any():
        starts = np.flatnonzero(np.diff(hit.astype(np.int8), prepend=0) == 1)
        ends = np.flatnonzero(np.diff(hit.astype(np.int8), append=0) == -1)
        accel_z = np.column_stack([_robust_z(x[:, c]) for c in ACCELS]).max(axis=1)
        mic_rms = np.sqrt(np.mean((clean - clean.mean()) ** 2))
        resp, ring = [], []
        for s, e in zip(starts, ends):
            resp.append(accel_z[max(s - RESPONSE_HALF, 0): e + RESPONSE_HALF + 1].max())
            tail = clean[e + 1 + pre.mic.burst_pad: e + 1 + pre.mic.burst_pad + RESPONSE_HALF]
            ring.append(np.sqrt(np.mean(tail**2)) / mic_rms if tail.size else np.nan)
        rng = np.random.default_rng(0)
        rand = [accel_z[p: p + 2 * RESPONSE_HALF].max() for p in rng.integers(0, x.shape[0] - 2 * RESPONSE_HALF, 50)]
        lengths = ends - starts + 1
        row.update(n_hit_runs=len(starts), run_len_median=float(np.median(lengths)), run_len_max=int(lengths.max()),
                   accel_z_at_burst=float(np.median(resp)), accel_z_random=float(np.median(rand)),
                   ringdown_ratio=float(np.nanmedian(ring)))
    return row


def meanspec_one(args: tuple[str, PreprocessConfig]) -> np.ndarray:
    rec_id, pre = args
    x = load(rec_id).astype(np.float64)
    x[:, MIC] = remove_mic_bursts(x[:, MIC], pre.mic.burst_z, pre.mic.burst_pad)[0]
    _, psd = welch(x, fs=pre.fs_hz, nperseg=MEANSPEC_NPERSEG, axis=0, detrend="constant")
    return np.sqrt(psd).T.astype(np.float32)  # (channels, freqs) amplitude spectral density, V/√Hz


def sk_one(args: tuple[str, PreprocessConfig]) -> list[dict]:
    rec_id, pre = args
    x = load(rec_id)
    bands = band_grid(pre.fs_hz, pre.envelope.sk.widths_hz, pre.envelope.sk.min_lo_hz)
    rows = []
    for name, c in RADIAL.items():
        sk = band_kurtosis(x[:, c], pre.fs_hz, bands)
        rows += [{"rec_id": rec_id, "channel": name, "lo_hz": lo, "hi_hz": hi, "sk": float(v)}
                 for (lo, hi), v in zip(bands, sk)]
    return rows


def envelope_one(args: tuple[str, float, PreprocessConfig]) -> np.ndarray:
    rec_id, fr, pre = args
    x = load(rec_id)
    orders = np.arange(0, pre.envelope.max_order + 1e-9, pre.envelope.order_step)
    return np.stack([envelope_order_spectrum(x[:, c], pre.fs_hz, fr, tuple(pre.envelope.bands_hz[name]), orders)
                     for name, c in RADIAL.items()]).astype(np.float32)


def bandcompare_one(args: tuple[str, float, PreprocessConfig]) -> np.ndarray:
    """Envelope order spectra per radial channel in (selected closed-track band, in-range healthy band)."""
    rec_id, fr, pre = args
    x = load(rec_id)
    orders = np.arange(0, pre.envelope.max_order + 1e-9, pre.envelope.order_step)
    return np.stack([[envelope_order_spectrum(x[:, c], pre.fs_hz, fr, band, orders)
                      for band in (tuple(pre.envelope.bands_hz[name]), tuple(pre.envelope.healthy_band_hz))]
                     for name, c in RADIAL.items()]).astype(np.float32)  # (channel, band, order)


def sensor_one(args: tuple[str, PreprocessConfig]) -> dict:
    """High-passed (> 1 kHz, FFT mask) RMS and kurtosis per accelerometer: which sensor hears the bearing."""
    rec_id, pre = args
    x = load(rec_id)[:, ACCELS].astype(np.float64)
    spec = np.fft.rfft(x - x.mean(axis=0), axis=0)
    spec[np.fft.rfftfreq(x.shape[0], 1 / pre.fs_hz) < SENSOR_HP_HZ] = 0
    hp = np.fft.irfft(spec, n=x.shape[0], axis=0)
    names = [pre.channels[c] for c in ACCELS]
    return {"rec_id": rec_id, **{f"hp_rms_{n}": float(hp[:, i].std()) for i, n in enumerate(names)},
            **{f"hp_kurt_{n}": float(kurtosis(hp[:, i])) for i, n in enumerate(names)}}


# ----------------------------------------------------------------------------- drivers
def run(task: str, meta: pd.DataFrame, pre: PreprocessConfig, workers: int) -> Path:
    trval = meta[meta.split_closed != "test"]
    train = meta[meta.split_closed == "train"]
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        if task == "mic":
            out = cache_path(task, pre.mic)
            rows = list(pool.map(mic_one, [(r, pre) for r in trval.rec_id], chunksize=4))
            df = pd.DataFrame(rows).merge(trval[["rec_id", "class10", "severity", "split_closed", "cv_fold"]])
            df.to_parquet(out, index=False)
            n = len(df)
        elif task == "meanspec":
            out = cache_path(task, pre.mic)
            sub = train.sort_values("rec_id").groupby("class10").sample(n=MEANSPEC_PER_CLASS, random_state=0)
            specs = np.stack(list(pool.map(meanspec_one, [(r, pre) for r in sub.rec_id], chunksize=2)))
            freqs = np.fft.rfftfreq(MEANSPEC_NPERSEG, 1 / pre.fs_hz)
            np.savez_compressed(out, freqs=freqs, asd=specs, rec_id=sub.rec_id.to_numpy(),
                                class10=sub.class10.to_numpy(), channels=np.array(pre.channels))
            n = len(sub)
        elif task == "sk":
            out = cache_path(task, pre.envelope.sk)
            sel = train[((train.fault_type == "outer") & (train.severity == 0)) | (train.class10 == "normal")]
            rows = [r for rs in pool.map(sk_one, [(r, pre) for r in sel.rec_id], chunksize=2) for r in rs]
            df = pd.DataFrame(rows).merge(sel[["rec_id", "class10"]])
            df.to_parquet(out, index=False)
            n = len(sel)
        elif task == "envelope":
            if any(hi <= lo for lo, hi in pre.envelope.bands_hz.values()):
                raise SystemExit("set envelope.bands_hz in configs/base.yaml first (notebook 02, spectral kurtosis)")
            out = cache_path(task, pre.envelope)
            sel = trval[((trval.fault_family == "bearing") & (trval.severity == 0)) | (trval.class10 == "normal")]
            specs = np.stack(list(pool.map(envelope_one, [(r, fr, pre) for r, fr in
                                                         zip(sel.rec_id, sel.fr_measured_hz)], chunksize=2)))
            orders = np.arange(0, pre.envelope.max_order + 1e-9, pre.envelope.order_step)
            np.savez_compressed(out, orders=orders, spec=specs, rec_id=sel.rec_id.to_numpy(),
                                class10=sel.class10.to_numpy(), split=sel.split_closed.to_numpy(),
                                channels=np.array(list(RADIAL)))
            n = len(sel)
        elif task == "bandcompare":
            out = cache_path(task, pre.envelope)
            sel = train[((train.fault_family == "bearing") & (train.severity == 0)) | (train.class10 == "normal")]
            specs = np.stack(list(pool.map(bandcompare_one, [(r, fr, pre) for r, fr in
                                                            zip(sel.rec_id, sel.fr_measured_hz)], chunksize=2)))
            np.savez_compressed(out, orders=np.arange(0, pre.envelope.max_order + 1e-9, pre.envelope.order_step),
                                spec=specs, rec_id=sel.rec_id.to_numpy(), class10=sel.class10.to_numpy(),
                                channels=np.array(list(RADIAL)), bands=np.array(["selected", "in_range_1_5k"]))
            n = len(sel)
        elif task == "sensor":
            out = cache_path(task)  # depends on no config setting
            sel = train[((train.fault_family == "bearing") & (train.severity == 0)) | (train.class10 == "normal")]
            df = pd.DataFrame(list(pool.map(sensor_one, [(r, pre) for r in sel.rec_id], chunksize=2)))
            df.merge(sel[["rec_id", "class10", "fr_measured_hz"]]).to_parquet(out, index=False)
            n = len(sel)
        else:
            raise SystemExit(f"unknown task {task}")
    print(f"{task}: {n} recordings in {time.perf_counter() - t:.0f} s → {out.relative_to(SYSTEM_CODE_DIR)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["mic", "meanspec", "sk", "envelope", "sensor", "bandcompare", "all"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    pre = load_config(args.config).preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    (data_dir() / "cache").mkdir(parents=True, exist_ok=True)
    for task in (["mic", "meanspec", "sk", "envelope", "sensor", "bandcompare"] if args.task == "all"
                 else [args.task]):
        run(task, meta, pre, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
