"""Representation PROTOTYPES (Stage 2C-1): order-spectrogram and microphone log-Mel candidates on a small subset.
No full cache is built and no shape is frozen.

    python scripts/08_representations.py subset      # candidates on the exploratory subset → data/cache/08_subset_*.npz
    python scripts/08_representations.py resolution  # peak separation at low orders: real (12, 30 Hz) + synthetic (12, 30, 60 Hz)
    python scripts/08_representations.py sharpness   # speed-tercile prediction from per-row sharpness (per candidate)
    python scripts/08_representations.py mic         # 10 kHz cap verification and burst-residue diagnostic (log-Mel)
    python scripts/08_representations.py summary     # would-be stacked shapes, float16 sizes, projected build times
    python scripts/08_representations.py all

Exploratory subset: split_closed TRAIN ∩ the 2B.1 development pool, class-stratified, ~10 %. Reserved tests are never
read; recordings at 40 Hz and above are reserved, so 60 Hz examples are synthetic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import PreprocessConfig, config_sha256, load_config
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, cache_file, data_dir, results_dir
from ml.preprocess import clean_recording, lowpass_taps, measure_edge_effect, remove_mic_bursts
from ml.spectro import log_mel, mel_filterbank, order_axis, recording_representation
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "representations"
SEED = 0
FRACTION = 0.10
N_PERM = 20
FS = 50_000


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def subset(meta: pd.DataFrame, cfg) -> pd.DataFrame:
    """split_closed TRAIN ∩ development pool, ceil(10 %) per class, fixed seed."""
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    tr = pool[pool.split_closed == "train"].sort_values("rec_id")
    return pd.concat([g.sample(n=int(np.ceil(FRACTION * len(g))), random_state=SEED) for _, g in tr.groupby("class10")])


def _rep(args):
    rid, fr, pre, name = args
    t = time.perf_counter()
    r = recording_representation(load(rid), fr, pre, name)
    return rid, r["order"].astype(np.float16), r["mel"].astype(np.float16), r["speeds"], time.perf_counter() - t


def run_subset(meta, cfg, workers):
    pre = cfg.preprocess
    sub = subset(meta, cfg)
    for name in pre.spectro.candidates:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool:
            res = list(pool.map(_rep, [(r, f, pre, name) for r, f in zip(sub.rec_id, sub.fr_measured_hz)], chunksize=1))
        out = cache_file(f"08_subset_{name}", pre, ext="npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, rec_id=np.array([r[0] for r in res]), order=np.stack([r[1] for r in res]),
                 mel=np.stack([r[2] for r in res]), speeds=np.stack([r[3] for r in res]),
                 seconds=np.array([r[4] for r in res]), class10=sub.class10.to_numpy(),
                 fr_hz=sub.fr_measured_hz.to_numpy(), folder=sub.folder.to_numpy())
        secs = np.array([r[4] for r in res])
        print(f"{name}: {len(sub)} recordings in {time.perf_counter() - t0:.0f} s wall; {secs.mean():.2f} s per "
              f"recording per core → {out.relative_to(SYSTEM_CODE_DIR)}")


# ----------------------------------------------------------------------------- resolution
def peak_metrics(orders: np.ndarray, prof: np.ndarray, lines: list[float]) -> dict:
    """For each line: peak level (dB, max within ±10 % of the line's order) and −3 dB full width (orders); for each pair
    of adjacent lines: valley depth = min(peak a, peak b) − minimum between them (dB); separated if ≥ 3 dB."""
    out = {}
    peaks = {}
    for L in lines:
        near = np.flatnonzero(np.abs(orders - L) <= 0.1 * L)
        k = near[np.argmax(prof[near])]
        p = prof[k]
        peaks[L] = (k, p)
        lo = k
        while lo > 0 and prof[lo - 1] > p - 3:
            lo -= 1
        hi = k
        while hi < len(prof) - 1 and prof[hi + 1] > p - 3:
            hi += 1
        out[f"{L:g}x_peak_db"] = float(p)
        out[f"{L:g}x_fwhm_orders"] = float(orders[hi] - orders[lo])
    for a, b in zip(lines[:-1], lines[1:]):
        ka, pa = peaks[a]
        kb, pb = peaks[b]
        valley = prof[min(ka, kb): max(ka, kb) + 1].min()
        depth = float(min(pa, pb) - valley)
        out[f"valley_{a:g}x_{b:g}x_db"] = depth
        out[f"separated_{a:g}x_{b:g}x"] = bool(depth >= 3)
    return out


def _synthetic(fr: float, lines: dict[float, float]) -> np.ndarray:
    t = np.arange(250_000) / FS
    x = 1e-3 * np.random.default_rng(0).normal(size=(250_000, 8))
    x[:, 0] = np.where((fr * t) % 1.0 < 0.12, 4.6, -0.7)
    for o, a in lines.items():
        x[:, 2] += a * np.sin(2 * np.pi * o * fr * t + o)
    return x.astype(np.float32)


def run_resolution(meta, cfg, workers):
    pre = cfg.preprocess
    o = order_axis(pre)
    ch = pre.spectro.channels.index("under_radial")
    lines = [0.5, 1.0, 2.0, 3.0]
    rows = []
    for name, cand in pre.spectro.candidates.items():
        for fr in (12.0, 30.0, 60.0):
            rep = recording_representation(_synthetic(fr, {L: 0.1 for L in lines}), fr, pre, name)
            prof = rep["order"][:, ch].mean(axis=(0, 2))
            rows.append({"candidate": name, "source": "synthetic", "speed_hz": fr, **peak_metrics(o, prof, lines)})
    # Real: subset recordings nearest 12 and 30 Hz in classes with strong shaft orders (imbalance, misalignment), on
    # the channel with the largest 1× line; lines 0.5×–3×.
    for name in pre.spectro.candidates:
        z = np.load(cache_file(f"08_subset_{name}", pre, ext="npz"), allow_pickle=True)
        cls, frs = z["class10"], z["fr_hz"]
        for target in (12.0, 30.0):
            cand_idx = [i for i in np.argsort(np.abs(frs - target))
                        if cls[i] in ("imbalance", "horizontal_misalignment", "vertical_misalignment")][:3]
            for i in cand_idx:
                spec = z["order"][i].astype(np.float32).mean(axis=(0, 3))  # (channels, orders)
                k1 = np.argmin(np.abs(o - 1.0))
                c = int(np.argmax(spec[:, k1]))
                rows.append({"candidate": name, "source": "real", "speed_hz": float(frs[i]), "rec_id": str(z["rec_id"][i]),
                             "channel": pre.spectro.channels[c], **peak_metrics(o, spec[c], lines)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "resolution.csv", index=False)
    print(df.groupby(["candidate", "source", "speed_hz"])[["1x_fwhm_orders", "valley_0.5x_1x_db", "valley_1x_2x_db"]]
          .mean().round(3).to_string())


# ----------------------------------------------------------------------------- speed sharpness
def run_sharpness(meta, cfg, workers):
    """Speed-related sharpness: per recording, the local contrast of each order row (row level minus the median of rows
    ±3, averaged over windows, frames and channels) → 128 features; predict the recording's speed tercile (terciles of the
    subset) with logistic regression, 5-fold stratified CV over recordings (a recording is one sample, so its windows stay
    together), against N_PERM label permutations."""
    pre = cfg.preprocess
    out = {}
    for name in pre.spectro.candidates:
        z = np.load(cache_file(f"08_subset_{name}", pre, ext="npz"), allow_pickle=True)
        img = z["order"].astype(np.float32).mean(axis=(1, 2, 4))  # (rec, orders)
        pad = np.pad(img, ((0, 0), (3, 3)), mode="edge")
        med = np.median(np.stack([pad[:, k:k + img.shape[1]] for k in range(7)]), axis=0)
        feats = img - med
        y = pd.qcut(z["fr_hz"], 3, labels=False)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

        def score(labels):
            pred = np.empty_like(labels)
            for tr, te in cv.split(feats, labels):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(feats[tr], labels[tr])
                pred[te] = m.predict(feats[te])
            return f1_score(labels, pred, average="macro")

        real = score(y)
        rng = np.random.default_rng(SEED)
        perm = [score(rng.permutation(y)) for _ in range(N_PERM)]
        out[name] = {"macro_f1": float(real), "perm_mean": float(np.mean(perm)), "perm_p95": float(np.percentile(perm, 95)),
                     "n_recordings": int(len(y))}
        print(f"  {name}: speed-tercile macro-F1 {real:.3f} vs permutation {np.mean(perm):.3f} (95th pct "
              f"{np.percentile(perm, 95):.3f}), {len(y)} recordings")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sharpness.json").write_text(json.dumps(out, indent=1))


def _cap_one(args):
    """Microphone before vs after the 10 kHz low-pass (both with bursts removed and the band-stop applied)."""
    rid, pre = args
    x = load(rid)
    no_lp = pre.model_copy(update={"lowpass": pre.lowpass.model_copy(update={"enabled": False})})
    mic = pre.channels.index("mic")
    before = clean_recording(x, no_lp)[0][:, mic].astype(np.float64)
    after = clean_recording(x, pre)[0][:, mic].astype(np.float64)
    w = np.hanning(before.size)
    spec = lambda v: (2 * np.abs(np.fft.rfft((v - v.mean()) * w)) / w.sum()) ** 2 / 2
    f = np.fft.rfftfreq(before.size, 1 / FS)
    pb, pa = spec(before), spec(after)
    above = f >= pre.lowpass.stop_hz
    below = (f >= 20) & (f <= pre.lowpass.pass_hz)
    edge = measure_edge_effect(before, lowpass_taps(pre), FS)
    return {"rec_id": rid,
            "atten_above_stop_db": float(10 * np.log10(pb[above].sum() / pa[above].sum())),
            "residual_above_vs_passband_db": float(10 * np.log10(pa[above].sum() / pa[below].sum())),
            "passband_change_db": float(10 * np.log10(pa[below].sum() / pb[below].sum())),
            **{f"mic_{k}": v for k, v in edge.items()}}


def run_mic(meta, cfg, workers):
    """(1) 10 kHz cap on the microphone: measured attenuation above the 9.8–10.2 kHz transition, residual level, pass-band
    change, and measured edge effect (subset recordings). (2) Burst-residue diagnostic: in the development pool's
    affected folders, counts and speed ranges of flagged vs unflagged recordings; a flagged-vs-unflagged comparison is run
    only where comparable controls exist (same folder, overlapping speed range, at least 5 of each)."""
    pre = cfg.preprocess
    sub = subset(meta, cfg)
    with ProcessPoolExecutor(workers) as pool:
        cap = pd.DataFrame(list(pool.map(_cap_one, [(r, pre) for r in sub.rec_id], chunksize=2)))
    OUT.mkdir(parents=True, exist_ok=True)
    cap.to_csv(OUT / "mic_cap.csv", index=False)
    print(cap.drop(columns="rec_id").describe().loc[["min", "50%", "max"]].round(3).to_string())

    dc = cfg.development
    pool_df = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    pool_df["flagged"] = pool_df.peakz_mic > pre.mic.burst_z
    aff = pool_df[pool_df.folder.isin(pool_df.loc[pool_df.flagged, "folder"].unique())]
    rows = []
    for f, g in aff.groupby("folder"):
        fl, un = g[g.flagged], g[~g.flagged]
        lo, hi = max(fl.fr_measured_hz.min(), un.fr_measured_hz.min()), min(fl.fr_measured_hz.max(), un.fr_measured_hz.max())
        in_overlap = lambda d: int(((d.fr_measured_hz >= lo) & (d.fr_measured_hz <= hi)).sum()) if lo <= hi else 0
        rows.append({"folder": f, "flagged": len(fl), "unflagged": len(un),
                     "flagged_speed_hz": f"{fl.fr_measured_hz.min():.1f}–{fl.fr_measured_hz.max():.1f}",
                     "unflagged_speed_hz": f"{un.fr_measured_hz.min():.1f}–{un.fr_measured_hz.max():.1f}",
                     "flagged_in_overlap": in_overlap(fl), "unflagged_in_overlap": in_overlap(un)})
    burst = pd.DataFrame(rows)
    burst.to_csv(OUT / "burst_residue_counts.csv", index=False)
    feasible = burst[(burst.flagged_in_overlap >= 5) & (burst.unflagged_in_overlap >= 5)]
    verdict = ("comparable controls exist in: " + ", ".join(feasible.folder)) if len(feasible) else (
        "no conclusion: within every affected folder, flagged and unflagged recordings do not overlap in speed "
        "(or one group has fewer than 5), so there are no comparable controls; a comparison would confound bursts with speed")
    (OUT / "mic_diagnostics.json").write_text(json.dumps({
        "cap": cap.drop(columns="rec_id").describe().loc[["min", "50%", "max"]].to_dict(),
        "burst_residue_counts": rows, "burst_residue_verdict": verdict,
        "split_hash": split_hash(meta), "git_sha": git_sha()}, indent=1))
    print(burst.to_string(index=False))
    print(verdict)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["subset", "resolution", "sharpness", "mic"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    {"subset": run_subset, "resolution": run_resolution, "sharpness": run_sharpness, "mic": run_mic}[args.task](
        meta, cfg, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
