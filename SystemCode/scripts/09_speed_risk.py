"""Speed-risk check of the leading representation candidate C against the fallback B (Stage 2C-2).

    python scripts/09_speed_risk.py

RULE (written and committed before running; parameters in configs/base.yaml `speed_risk`):
- Real data (no tacho-fallback recording is permitted: all 94 are outside the development pool):
  - group (ii): split_closed TRAIN ∩ development pool recordings with speed_range_pct > 1 % (window speeds vary);
  - group (iii): for each group-(ii) recording, the `refs_per_recording` nearest-speed CLEAN recordings of the same class
    from split_closed TRAIN ∩ pool (clean = speed_range_pct ≤ 0.3 %, no tacho glitches, no quality flags).
- Synthetic stress on permitted clean recordings (`synthetic_per_class` per class, fixed seed): every
  `displace_every`-th tacho pulse is displaced by `displace_fraction` of a revolution, so fewer than 80 % of intervals are
  clean and every window falls back to the SUPPLIED speed; the supplied speed is the measured speed × (1 + offset),
  offsets 0, ±0.5, ±1, ±2 %. The clean version (intact tacho, measured speed) is the reference.
- Measures, per window, for C and B, on the accelerometer with the largest clean 1× amplitude: 1× and 2× peak position
  (argmax of the frame-averaged order profile within 0.8–1.25 / 1.6–2.5 orders), position error |pos − k| in orders and
  in order bins (bin = the log-axis step at that order), −3 dB peak width in orders, and (synthetic) peak height change
  vs the clean version in dB.
- Reference width per candidate: its median 1× width on group (iii) (real) or on the clean versions (synthetic).
- PASS for C, required on BOTH group (ii) and the synthetic ±1 % offsets (pooled):
  (1) median 1× position error ≤ `max_error_bins` order bins;
  (2) median 1× width ≤ `max_width_ratio` × C's reference width;
  (3) C's median 1× position error ≤ B's + `max_error_bins` bins;
  (4) C's width ratio (median width / own reference) ≤ `max_width_ratio` × B's width ratio.
  Widths are compared as ratios to each candidate's own reference because B's intrinsic resolution differs from C's.
  ±2 % and ±0.5 % are reported as sensitivity only. If C passes, config `spectro.frozen` may be set; otherwise STOP.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.preprocess import TACHO, tacho_intervals
from ml.spectro import order_axis, recording_representation
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "representations"
CANDS = ("C_order8rev", "B_stft32768")


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def displace_tacho(x: np.ndarray, pre, every: int, frac: float) -> np.ndarray:
    """Copy of x whose tacho has every `every`-th pulse moved `frac` of a revolution later (low elsewhere)."""
    y = x.copy()
    t = y[:, TACHO].astype(np.float64)
    ti = tacho_intervals(t, pre)
    lo, hi = np.percentile(t, [10, 99])
    period = int(np.median(ti.intervals))
    width = max(int(0.12 * period), 1)
    for k, e in enumerate(ti.edges[:-1]):
        if k % every == 0:
            t[e: e + width] = lo
            s = e + int(frac * period)
            t[s: s + width] = hi
    y[:, TACHO] = t.astype(np.float32)
    return y


def peak(orders: np.ndarray, prof: np.ndarray, k: float, lo: float, hi: float) -> dict:
    sel = np.flatnonzero((orders >= lo * k / 1) & (orders <= hi * k / 1)) if k == 1 else np.flatnonzero(
        (orders >= lo) & (orders <= hi))
    i = sel[np.argmax(prof[sel])]
    p = prof[i]
    a = i
    while a > 0 and prof[a - 1] > p - 3:
        a -= 1
    b = i
    while b < len(prof) - 1 and prof[b + 1] > p - 3:
        b += 1
    step = np.log(orders[1] / orders[0])
    return {"pos": float(orders[i]), "err_orders": float(abs(orders[i] - k)),
            "err_bins": float(abs(np.log(orders[i] / k)) / step), "width": float(orders[b] - orders[a]), "height_db": float(p)}


def measure(rep_order: np.ndarray, ch: int, orders: np.ndarray) -> list[dict]:
    """Per window: 1× and 2× peak measures on channel ch (frame-averaged profile)."""
    rows = []
    for w in range(rep_order.shape[0]):
        prof = rep_order[w, ch].astype(np.float64).mean(axis=1)
        p1, p2 = peak(orders, prof, 1, 0.8, 1.25), peak(orders, prof, 2, 1.6, 2.5)
        rows.append({"window": w, **{f"x1_{k}": v for k, v in p1.items()}, **{f"x2_{k}": v for k, v in p2.items()}})
    return rows


def _real(args):
    rid, fr, group, pre = args
    x = load(rid)
    o = order_axis(pre)
    out = []
    reps = {c: recording_representation(x, fr, pre, c)["order"] for c in CANDS}
    ch = int(np.argmax(reps["C_order8rev"].astype(np.float32).mean(axis=(0, 3))[:, np.argmin(np.abs(o - 1))]))
    for c, r in reps.items():
        out += [{"group": group, "rec_id": rid, "candidate": c, "speed_hz": fr, "channel": pre.spectro.channels[ch], **m}
                for m in measure(r, ch, o)]
    return out


def _synthetic(args):
    rid, fr, pre, sr = args
    x = load(rid)
    o = order_axis(pre)
    bad = displace_tacho(x, pre, sr.displace_every, sr.displace_fraction)
    out = []
    for c in CANDS:
        clean = recording_representation(x, fr, pre, c)
        ch = int(np.argmax(clean["order"].astype(np.float32).mean(axis=(0, 3))[:, np.argmin(np.abs(o - 1))]))
        ref = measure(clean["order"], ch, o)
        for off in sr.offsets_pct:
            rep = recording_representation(bad, fr * (1 + off / 100), pre, c)
            assert rep["fallback"].all(), f"{rid}: tacho corruption did not force the fallback"
            for m, r0 in zip(measure(rep["order"], ch, o), ref):
                out.append({"group": "synthetic", "rec_id": rid, "candidate": c, "speed_hz": fr, "offset_pct": off,
                            "channel": pre.spectro.channels[ch], **m,
                            "x1_height_change_db": m["x1_height_db"] - r0["x1_height_db"],
                            "x2_height_change_db": m["x2_height_db"] - r0["x2_height_db"],
                            "ref_x1_width": r0["x1_width"]})
    return out


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    pre, sr, dc = cfg.preprocess, cfg.speed_risk, cfg.development
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    tr = pool[pool.split_closed == "train"]
    g2 = tr[tr.speed_range_pct > sr.group_ii_min_range_pct]
    clean = tr[(tr.speed_range_pct <= sr.clean_max_range_pct) & (tr.tacho_glitches == 0) & tr.quality_flags.isna()]
    refs = []
    for r in g2.itertuples():
        c = clean[(clean.class10 == r.class10) & ~clean.rec_id.isin(refs)]
        refs += list(c.loc[(c.fr_measured_hz - r.fr_measured_hz).abs().sort_values().index[:sr.refs_per_recording], "rec_id"])
    g3 = clean[clean.rec_id.isin(refs)]
    syn = clean.sort_values("rec_id").groupby("class10").sample(n=sr.synthetic_per_class, random_state=sr.seed)
    print(f"group (ii) {len(g2)} recordings, group (iii) {len(g3)} clean references, synthetic {len(syn)} recordings")
    jobs_real = [(r, f, "ii", pre) for r, f in zip(g2.rec_id, g2.fr_measured_hz)] + \
                [(r, f, "iii", pre) for r, f in zip(g3.rec_id, g3.fr_measured_hz)]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 4) - 2)) as pool_ex:
        real = [row for rows in pool_ex.map(_real, jobs_real) for row in rows]
        synth = [row for rows in pool_ex.map(_synthetic, [(r, f, pre, sr) for r, f in zip(syn.rec_id, syn.fr_measured_hz)])
                 for row in rows]
    df = pd.DataFrame(real + synth)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "speed_risk_windows.csv", index=False)

    bin_at_1 = float(np.log(order_axis(pre)[1] / order_axis(pre)[0]))  # log step (≈ relative bin width at 1×)
    res = {"counts": {"group_ii": int(len(g2)), "group_iii": int(len(g3)), "synthetic": int(len(syn))},
           "group_ii_rec_ids": list(g2.rec_id), "group_iii_rec_ids": list(g3.rec_id), "synthetic_rec_ids": list(syn.rec_id),
           "order_bin_at_1x_orders": bin_at_1, "split_hash": split_hash(meta), "git_sha": git_sha(), "candidates": {}}
    for c in CANDS:
        d = df[df.candidate == c]
        ref_real = d[d.group == "iii"].x1_width.median()
        ii = d[d.group == "ii"]
        syn_d = d[d.group == "synthetic"]
        ref_syn = syn_d.ref_x1_width.median()
        entry = {"reference_width_real": float(ref_real), "reference_width_synthetic": float(ref_syn),
                 "group_ii": {"median_x1_err_bins": float(ii.x1_err_bins.median()), "median_x1_width": float(ii.x1_width.median()),
                              "width_ratio": float(ii.x1_width.median() / ref_real),
                              "median_x2_err_bins": float(ii.x2_err_bins.median())},
                 "group_iii": {"median_x1_err_bins": float(d[d.group == "iii"].x1_err_bins.median()),
                               "median_x1_width": float(ref_real)},
                 "synthetic": {}}
        for off in sr.offsets_pct:
            s = syn_d[syn_d.offset_pct == off]
            entry["synthetic"][f"{off:+g}%"] = {"median_x1_err_bins": float(s.x1_err_bins.median()),
                                                "median_x1_err_orders": float(s.x1_err_orders.median()),
                                                "median_x2_err_bins": float(s.x2_err_bins.median()),
                                                "median_x1_width": float(s.x1_width.median()),
                                                "width_ratio": float(s.x1_width.median() / ref_syn),
                                                "median_x1_height_change_db": float(s.x1_height_change_db.median()),
                                                "median_x2_height_change_db": float(s.x2_height_change_db.median())}
        ps = syn_d[syn_d.offset_pct.isin(sr.pass_offsets_pct)]
        entry["synthetic_pass_offsets"] = {"median_x1_err_bins": float(ps.x1_err_bins.median()),
                                           "median_x1_width": float(ps.x1_width.median()),
                                           "width_ratio": float(ps.x1_width.median() / ref_syn)}
        res["candidates"][c] = entry
    C, B = res["candidates"]["C_order8rev"], res["candidates"]["B_stft32768"]
    checks = {}
    for name, cg, bg in (("group_ii", C["group_ii"], B["group_ii"]), ("synthetic_pm1", C["synthetic_pass_offsets"],
                                                                         B["synthetic_pass_offsets"])):
        checks[name] = {
            "1_error_le_1bin": cg["median_x1_err_bins"] <= sr.max_error_bins,
            "2_width_le_1p5_ref": cg["width_ratio"] <= sr.max_width_ratio,
            "3_error_vs_B": cg["median_x1_err_bins"] <= bg["median_x1_err_bins"] + sr.max_error_bins,
            "4_width_ratio_vs_B": cg["width_ratio"] <= sr.max_width_ratio * bg["width_ratio"]}
    res["checks"] = checks
    res["C_passes"] = bool(all(all(v.values()) for v in checks.values()))
    (OUT / "speed_risk.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"checks": checks, "C_passes": res["C_passes"]}, indent=1))
    for c in CANDS:
        e = res["candidates"][c]
        print(c, "group ii", {k: round(v, 3) for k, v in e["group_ii"].items()}, "ref widths", round(e["reference_width_real"], 3),
              round(e["reference_width_synthetic"], 3))
        for off, v in e["synthetic"].items():
            print("   ", off, {k: round(x, 3) for k, x in v.items()})
    plot_examples(res, meta, pre, sr)
    return 0


def plot_examples(res: dict, meta: pd.DataFrame, pre, sr) -> None:
    """One example per group: frame-averaged order profile (window 4) near 0.5–3×, C and B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = meta.set_index("rec_id")
    o = order_axis(pre)
    g2 = max(res["group_ii_rec_ids"], key=lambda r: m.loc[r, "speed_range_pct"])
    g3 = min(res["group_iii_rec_ids"], key=lambda r: abs(m.loc[r, "fr_measured_hz"] - m.loc[g2, "fr_measured_hz"])
             if m.loc[r, "class10"] == m.loc[g2, "class10"] else 1e9)
    sy = res["synthetic_rec_ids"][0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8), sharey=False)
    for ax, (label, rid) in zip(axes, [("group (ii)", g2), ("group (iii)", g3), ("synthetic", sy)]):
        x = load(rid)
        fr = m.loc[rid, "fr_measured_hz"]
        for c, ls in zip(CANDS, ("-", "--")):
            variants = [("", x, fr)] if label != "synthetic" else [
                ("clean", x, fr), ("fallback +1 %", displace_tacho(x, pre, sr.displace_every, sr.displace_fraction), fr * 1.01),
                ("fallback +2 %", displace_tacho(x, pre, sr.displace_every, sr.displace_fraction), fr * 1.02)]
            rep0 = recording_representation(x, fr, pre, c)["order"]
            ch = int(np.argmax(rep0.astype(np.float32).mean(axis=(0, 3))[:, np.argmin(np.abs(o - 1))]))
            for name, xx, ff in variants:
                prof = recording_representation(xx, ff, pre, c)["order"][4, ch].astype(np.float32).mean(axis=1)
                ax.semilogx(o, prof, ls, lw=1, label=f"{c.split('_')[0]} {name}".strip())
        for k in (0.5, 1, 2, 3):
            ax.axvline(k, color="grey", lw=0.5, ls=":")
        ax.set(xlim=(0.45, 3.5), title=f"{label}: {rid} ({fr:.1f} Hz, range {m.loc[rid, 'speed_range_pct']:.2f} %)",
               xlabel="order", ylabel="amplitude (dB re 1 V)")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir() / "figures" / "09_speed_risk_examples.png")


if __name__ == "__main__":
    sys.exit(main())
