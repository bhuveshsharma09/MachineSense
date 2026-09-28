"""F1 — the final evaluation (CLAUDE.md §12). Run ONCE, after committing everything:

    python scripts/30_final_eval.py                     # F1: opens every reserved test once; writes results/final/
    python scripts/30_final_eval.py --dry-run           # F0 check on DEVELOPMENT data only; writes results/final_dry_run/

It refuses to run unless §12 is committed and unmodified, the bundle and artifacts/final/ verify, and (F1 only) this
script and ml/final.py are committed and unmodified and results/final/ does not exist yet. It records its own sha256.
Every score comes from the CPU deployment path (ml.final.score_cnn, equal to ml.inference.score_recording for the
bundle). In --dry-run mode, development-pool recordings replace every reserved set and the loader refuses any
reserved rec_id; the output is labelled "DRY RUN — development data" and its numbers are not results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import confusion_matrix, roc_auc_score

from ml.arbitrator import VARIANTS, decide
from ml.bundle import sha256_file
from ml.config import config_sha256, load_config
from ml.evaluation import CLASS10_TO_6, metrics_mixed, mixed_bootstrap_draws
from ml.final import final_dir, load_final, recording_inputs, score_cnn, score_r18
from ml.labels import CLASS6, CLASS10
from ml.openset import oscr
from ml.paths import DEFAULT_CONFIG, REPO_ROOT, SYSTEM_CODE_DIR, bundle_dir, data_dir, results_dir
from ml.splits import FAULT_CLASSES, development_pool, folder_proxy, split_hash

SCRIPT = Path(__file__).resolve()
PROTOCOL_HEADING = "## 12. Final evaluation protocol"
DRY_LABEL = "DRY RUN — development data"
ORIGINS = {"closed": "split_closed", "severity": "split_severity", "speed_block": "split_speed_block"}
PRIMARY_RULE = "B_with_caution"
CODE = ("SystemCode/scripts/30_final_eval.py", "SystemCode/ml/final.py")


class Refused(RuntimeError):
    """A precondition of §12e is not met."""


class ReservedAccess(RuntimeError):
    """The dry run tried to read a reserved recording."""


# ----------------------------------------------------------------------------------------------------------- checks
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout


def committed_protocol() -> str:
    """§12 must be in the committed CLAUDE.md and the working copy must not change it. Returns the protocol commit."""
    if PROTOCOL_HEADING not in git("show", "HEAD:CLAUDE.md"):
        raise Refused("CLAUDE.md §12 (final evaluation protocol) is not committed")
    if git("diff", "--name-only", "HEAD", "--", "CLAUDE.md").strip():
        raise Refused("CLAUDE.md has uncommitted changes; §12 must be committed and unmodified")
    return git("log", "-1", "--format=%H", "--", "CLAUDE.md").strip()


def committed_code() -> None:
    for p in CODE:
        if not git("ls-files", p).strip():
            raise Refused(f"{p} is not committed")
        if git("diff", "--name-only", "HEAD", "--", p).strip():
            raise Refused(f"{p} has uncommitted changes")


def preconditions(dry_run: bool, out: Path):
    from ml.inference import get_bundle
    protocol = committed_protocol()
    if not dry_run:
        committed_code()
        if out.exists():
            raise Refused(f"{out} already exists: F1 runs once")
    bundle = get_bundle(str(bundle_dir()))       # verifies every bundle hash
    final = load_final(final_dir())               # verifies every final-model hash
    return protocol, bundle, final


# ----------------------------------------------------------------------------------------------------------- sets
def build_sets(meta: pd.DataFrame, pool: pd.DataFrame, dry_run: bool, classes, limit: int | None = None) -> dict:
    """Recording sets of §12b. F1: the reserved columns. Dry run: development-pool substitutes (§12e)."""
    pool_ids = set(pool.rec_id)
    if not dry_run:
        member = {o: set(meta.rec_id[meta[col] == "test"]) for o, col in ORIGINS.items()}
        reserved = set().union(*member.values())
        bundle_sets = dict(member)
        loco = {c: {"known": set(meta.rec_id[meta[f"loco_{c}"] == "test"]),
                    "unknown_primary": set(meta.rec_id[meta.class10 == c]) & reserved,
                    "unknown_all": set(meta.rec_id[meta.class10 == c])} for c in classes}
    else:
        val = pool[pool.split_closed == "val"]
        faults = pool[pool.class10 != "normal"]
        top = faults.groupby("class10").severity.transform("max")
        member = {"closed": set(val.rec_id), "severity": set(faults.rec_id[faults.severity == top]),
                  "speed_block": set(pool.rec_id[pool.fr_measured_hz > 35])}
        bundle_sets = dict(member)
        loco = {c: {"known": set(val.rec_id[val.class10 != c]), "unknown_primary": set(pool.rec_id[pool.class10 == c]),
                    "unknown_all": set(pool.rec_id[pool.class10 == c])} for c in classes}
        allids = set().union(*bundle_sets.values(), *[s for v in loco.values() for s in v.values()])
        assert allids <= pool_ids, "dry-run sets must be development-pool recordings only"
    if limit is not None:
        if not dry_run:
            raise Refused("--limit is only allowed with --dry-run")
        cut = lambda s: set(sorted(s)[:limit])  # noqa: E731
        bundle_sets = {k: cut(v) for k, v in bundle_sets.items()}
        loco = {c: {k: cut(v) for k, v in d.items()} for c, d in loco.items()}
        for c in loco:
            loco[c]["unknown_primary"] = loco[c]["unknown_primary"] & loco[c]["unknown_all"]
    return {"bundle": bundle_sets, "r18": bundle_sets["closed"], "loco": loco, "member": member}


class Loader:
    """Raw recording + metadata speed. In dry-run mode every rec_id outside the development pool is refused."""

    def __init__(self, meta: pd.DataFrame, pool_ids: set, dry_run: bool):
        self.meta, self.pool_ids, self.dry_run, self.touched = meta.set_index("rec_id"), pool_ids, dry_run, []

    def load(self, rec_id: str) -> tuple[np.ndarray, float]:
        if self.dry_run and rec_id not in self.pool_ids:
            raise ReservedAccess(f"dry run refused a non-development recording ({rec_id})")
        self.touched.append(rec_id)
        return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy"), float(self.meta.loc[rec_id, "fr_measured_hz"])


# ----------------------------------------------------------------------------------------------------------- scoring
def decisions(s: dict, thr: dict) -> dict:
    out = {}
    for v in VARIANTS:
        d = decide(v, s["top"], s["msp"], s["maha"], s["ae"], s["b1s"], thr)
        out[f"decision_{v}"] = d["decision"]
        out[f"caution_{v}"] = d["caution"]
        out[f"label_{v}"] = d["fault_class"] or d["decision"]
    return out


def score_all(sets: dict, loader: Loader, bundle, final, meta: pd.DataFrame, cfg) -> tuple[pd.DataFrame, ...]:
    from ml.inference import speed_warnings
    need: dict[str, dict] = {}
    for s in sets["bundle"].values():
        for r in s:
            need.setdefault(r, {"bundle": False, "r18": False, "loco": set()})["bundle"] = True
    for r in sets["r18"]:
        need[r]["r18"] = True
    for c, d in sets["loco"].items():
        for r in d["known"] | d["unknown_all"]:
            need.setdefault(r, {"bundle": False, "r18": False, "loco": set()})["loco"].add(c)
    same_inputs = (config_sha256(bundle.config.preprocess) == config_sha256(cfg.preprocess)
                   and bundle.config.representation == cfg.representation)
    info = meta.set_index("rec_id")
    thr_b = {k: float(bundle.thresholds[k]) for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
    tr_rng, cal_rng = bundle.manifest.data["train_speed_range_hz"], bundle.manifest.data["calibration_normal_speed_range_hz"]
    rb, rr, rl = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(sorted(need)):
        x, fr = loader.load(r)
        inp_b = recording_inputs(x, fr, bundle.config)
        inp = inp_b if same_inputs else recording_inputs(x, fr, cfg)
        base = {"rec_id": r, "class10": info.class10[r], "folder": info.folder[r], "speed_hz": fr}
        if need[r]["bundle"]:
            s = score_cnn(inp_b, bundle.cnn, bundle.normalisation["cnn6"], CLASS10, bundle.mahalanobis, bundle.ae,
                          bundle.normalisation["ae6"], bundle.model_config["ae"], bundle.b1s, bundle.model_config["prob_floor"])
            rb.append({**base, "pred10": s["top"], "msp": s["msp"], "maha": s["maha"], "ae": s["ae"], "b1s": s["b1s"],
                       "p_normal": float(s["probs"]["normal"]), "warnings": "; ".join(speed_warnings(fr, tr_rng, cal_rng)),
                       **decisions(s, thr_b)})
        if need[r]["r18"]:
            s = score_r18(inp, final.r18, final.r18_norm, CLASS10, bundle.model_config["prob_floor"])
            rr.append({**base, "pred10": s["top"], "msp": s["msp"]})
        for c in sorted(need[r]["loco"]):
            m = final.loco[c]
            s = score_cnn(inp, m.cnn, m.cnn_norm, m.classes, m.maha, final.ae, final.ae_norm, final.ae_cfg, final.b1s,
                          bundle.model_config["prob_floor"], maha_pc=m.maha_pc)
            thr = {k: float(m.thresholds[k]) for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
            rl.append({**base, "withheld": c, "pred": s["top"], "msp": s["msp"], "maha": s["maha"], "maha_pc": s["maha_pc"],
                       "ae": s["ae"], "b1s": s["b1s"], **decisions(s, thr)})
        if i % 50 == 0:
            print(f"  scored {i + 1}/{len(need)} recordings, {time.perf_counter() - t0:.0f} s", flush=True)
    secs = time.perf_counter() - t0
    return pd.DataFrame(rb), pd.DataFrame(rr), pd.DataFrame(rl), {"seconds": secs, "recordings": len(need),
                                                                  "seconds_per_recording": secs / max(1, len(need))}


# ----------------------------------------------------------------------------------------------------------- metrics
def cp(k: int, n: int, level: float = 0.95) -> list:
    """Clopper–Pearson interval for k of n."""
    if n == 0:
        return [None, None]
    a = (1 - level) / 2
    return [float(beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0]


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95_clopper_pearson": cp(int(k), int(n))}


def boot_auc(y: np.ndarray, s: np.ndarray, draws) -> dict:
    if len(set(y)) < 2:
        return {"auroc": None, "ci95": [None, None], "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum())}
    vals = [roc_auc_score(y[d], s[d]) for d in draws if len(set(y[d])) == 2]
    return {"auroc": float(roc_auc_score(y, s)), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()), "n_draws": len(vals)}


def q2(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if df.empty:
        return {"n_recordings": 0}
    draws = mixed_bootstrap_draws(df.class10.to_numpy(), df.folder.to_numpy(), n_boot, seed)
    t6, p6 = df.class10.map(CLASS10_TO_6), df.pred10.map(CLASS10_TO_6)
    m10 = metrics_mixed(df.class10, df.pred10, CLASS10, draws, df.folder)
    m6 = metrics_mixed(t6, p6, CLASS6, draws, df.folder)
    keep = ("macro_f1", "macro_f1_ci", "balanced_accuracy", "balanced_accuracy_ci", "per_class_f1")
    return {"n_recordings": len(df), "n_distinct_recordings": int(df.rec_id.nunique()),
            "class10": {k: m10[k] for k in keep}, "class6": {k: m6[k] for k in keep},
            "confusion_class10": pd.DataFrame(confusion_matrix(df.class10, df.pred10, labels=CLASS10), index=CLASS10,
                                              columns=CLASS10).to_dict()}


def q1(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if df.empty:
        return {"n_recordings": 0}
    normal = (df.class10 == "normal").to_numpy()
    out = {"n_recordings": len(df), "n_normals": int(normal.sum()), "n_faults": int((~normal).sum()), "rules": {}}
    for v in VARIANTS:
        not_healthy = (df[f"decision_{v}"] != "healthy").to_numpy()
        caution = df[f"caution_{v}"].notna().to_numpy()
        out["rules"][v] = {"detection": rate((not_healthy & ~normal).sum(), (~normal).sum()),
                           "false_alarm": rate((not_healthy & normal).sum(), normal.sum()),
                           "healthy_with_caution_normals": int((caution & normal & ~not_healthy).sum()),
                           "healthy_with_caution_faults": int((caution & ~normal & ~not_healthy).sum())}
    draws = mixed_bootstrap_draws(df.class10.to_numpy(), df.folder.to_numpy(), n_boot, seed)
    y = (~normal).astype(int)
    out["auroc"] = {"ae_error": boot_auc(y, df.ae.to_numpy(), draws), "b1s": boot_auc(y, df.b1s.to_numpy(), draws)}
    out["speed_warnings"] = df.warnings.fillna("").str.split("; ").explode().loc[lambda s: s != ""].value_counts().to_dict()
    out["recordings_with_any_speed_warning"] = int((df.warnings.fillna("") != "").sum())
    return out


def three_way(g: pd.DataFrame) -> dict:
    return {v: g[f"decision_{v}"].value_counts().reindex(["unknown", "known_fault", "healthy"], fill_value=0).astype(int).to_dict()
            for v in VARIANTS}


def q3_block(kn: pd.DataFrame, un: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Known side vs unknown side: AUROC / OSCR (MSP, Mahalanobis shared and per-class) and three-way outcomes."""
    both = pd.concat([kn.assign(_y=0), un.assign(_y=1)], ignore_index=True)
    out = {"n_known": len(kn), "n_unknown": len(un), "three_way_known": three_way(kn), "three_way_unknown": three_way(un)}
    if len(kn) and len(un):
        draws = mixed_bootstrap_draws(both.class10.to_numpy(), both.folder.to_numpy(), n_boot, seed)
        y = both._y.to_numpy()
        correct = (kn.pred == kn.class10).to_numpy()
        for name, u in (("msp", -both.msp.to_numpy()), ("maha_shared", both.maha.to_numpy()), ("maha_per_class", both.maha_pc.to_numpy())):
            out[name] = {**boot_auc(y, u, draws), "oscr": oscr(u[y == 0], correct, u[y == 1])}
    return out


def origin_groups(ids: pd.Series, member: dict) -> tuple[dict, dict]:
    """Membership groups (a recording in each column it belongs to) and the exclusive combinations."""
    mem = {o: ids.isin(s).to_numpy() for o, s in member.items()}
    combos = {}
    for key in sorted({tuple(o for o in member if mem[o][i]) for i in range(len(ids))}):
        combos[" & ".join(key) or "none"] = np.array([tuple(o for o in member if mem[o][i]) == key for i in range(len(ids))])
    return mem, combos


def q3(loco: pd.DataFrame, sets: dict, member: dict, n_boot: int, seed: int) -> dict:
    res, pooled = {}, {"primary": {"kn": [], "un": []}, "all": {"kn": [], "un": []}}
    for c in sets:
        g = loco[loco.withheld == c]
        kn = g[g.rec_id.isin(sets[c]["known"])]
        block = {}
        for uset in ("unknown_primary", "unknown_all"):
            un = g[g.rec_id.isin(sets[c][uset])]
            b = q3_block(kn, un, n_boot, seed)
            b["breakdown"] = breakdown(kn, un, member)
            block[uset] = b
            key = "primary" if uset == "unknown_primary" else "all"
            pooled[key]["kn"].append(kn)
            pooled[key]["un"].append(un)
        res[c] = block
    head = {}
    for key, d in pooled.items():
        kn, un = pd.concat(d["kn"]), pd.concat(d["un"])
        uset = "unknown_primary" if key == "primary" else "unknown_all"
        aucs = {s: [res[c][uset][s]["auroc"] for c in res if s in res[c][uset] and res[c][uset][s]["auroc"] is not None]
                for s in ("maha_shared", "msp", "maha_per_class")}
        head[key] = {"macro_auroc": {s: (float(np.mean(v)) if v else None) for s, v in aucs.items()},
                     "n_classes": len(res),
                     "unknown_recall_primary_rule": rate((un[f"decision_{PRIMARY_RULE}"] == "unknown").sum(), len(un)),
                     "false_unknown_known_side_primary_rule": rate((kn[f"decision_{PRIMARY_RULE}"] == "unknown").sum(), len(kn)),
                     "instances_known": len(kn), "distinct_known": int(kn.rec_id.nunique()),
                     "instances_unknown": len(un), "distinct_unknown": int(un.rec_id.nunique()),
                     "breakdown_pooled": breakdown(kn, un, member)}
    return {"per_class": res, "headline": head}


def breakdown(kn: pd.DataFrame, un: pd.DataFrame, member: dict) -> dict:
    """By origin (membership and exclusive combinations): n, primary-rule unknown rate and scores on each side; AUROC
    (Mahalanobis shared, MSP) within an origin when both sides are present."""
    def side(g, mask):
        h = g[mask]
        return {"n": len(h), "called_unknown_primary_rule": rate((h[f"decision_{PRIMARY_RULE}"] == "unknown").sum(), len(h)),
                "mean_msp": float(h.msp.mean()) if len(h) else None, "mean_maha": float(h.maha.mean()) if len(h) else None}
    out = {}
    mk, ck = origin_groups(kn.rec_id, member)
    mu, cu = origin_groups(un.rec_id, member)
    for kind, gk, gu in (("membership", mk, mu), ("exclusive", ck, cu)):
        keys = sorted(set(gk) | set(gu))
        out[kind] = {}
        for o in keys:
            a = gk.get(o, np.zeros(len(kn), bool))
            b = gu.get(o, np.zeros(len(un), bool))
            entry = {"known_side": side(kn, a), "unknown_side": side(un, b)}
            if a.any() and b.any():
                y = np.r_[np.zeros(a.sum()), np.ones(b.sum())]
                entry["auroc_maha_shared"] = float(roc_auc_score(y, np.r_[kn.maha[a], un.maha[b]]))
                entry["auroc_msp"] = float(roc_auc_score(y, np.r_[-kn.msp[a], -un.msp[b]]))
            out[kind][o] = entry
    return out


# ----------------------------------------------------------------------------------------------------------- main
def run(dry_run: bool, limit: int | None, classes: list[str], n_boot: int | None, out: Path | None = None) -> dict:
    cfg = load_config(DEFAULT_CONFIG)
    out = out or (results_dir() / ("final_dry_run" if dry_run else "final"))
    protocol, bundle, final = preconditions(dry_run, out)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    meta["folder"] = folder_proxy(meta)
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    sets = build_sets(meta, pool, dry_run, classes, limit)
    loader = Loader(meta, set(pool.rec_id), dry_run)
    nb = n_boot or cfg.baselines.n_bootstrap
    seed = cfg.baselines.seed
    sb, sr, sl, timing = score_all(sets, loader, bundle, final, meta, cfg)
    member = sets["member"]
    bsets = {k: sb[sb.rec_id.isin(v)] for k, v in sets["bundle"].items()}
    results = {"q2": {"bundle": {k: q2(v, nb, seed) for k, v in bsets.items()},
                      "r18ft_secondary": {"closed": q2(sr, nb, seed)}},
               "q1": {"bundle": {k: q1(v, nb, seed) for k, v in bsets.items()}},
               "q3": q3(sl, sets["loco"], member, nb, seed)}
    if out.exists() and dry_run:
        shutil.rmtree(out)
    out.mkdir(parents=True)
    label = DRY_LABEL if dry_run else "F1 final evaluation (CLAUDE.md §12)"
    prov = {"label": label, "dry_run": dry_run, "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git("rev-parse", "HEAD").strip(), "protocol_commit": protocol, "split_hash": split_hash(meta),
            "script_sha256": sha256_file(SCRIPT), "ml_final_sha256": sha256_file(SYSTEM_CODE_DIR / "ml" / "final.py"),
            "bundle_manifest_sha256": bundle.manifest_sha256, "final_manifest_sha256": final.manifest_sha256,
            "primary_rule": PRIMARY_RULE, "n_bootstrap": nb, "bootstrap_seed": seed, "limit": limit, "classes": classes,
            "set_sizes": {"bundle": {k: len(v) for k, v in sets["bundle"].items()}, "r18ft": len(sets["r18"]),
                          "loco": {c: {k: len(v) for k, v in d.items()} for c, d in sets["loco"].items()}},
            "recordings_read": len(loader.touched), "timing": timing}
    for name, obj in (("provenance", prov), ("q1", results["q1"]), ("q2", results["q2"]), ("q3", results["q3"]),
                      ("headline", results["q3"]["headline"])):
        (out / f"{name}.json").write_text(json.dumps({"label": label, **obj} if name != "provenance" else obj,
                                                     indent=1, default=float, ensure_ascii=False))
    for name, df in (("scores_bundle", sb), ("scores_r18ft", sr), ("scores_loco", sl)):
        df.assign(label=label).to_parquet(out / f"{name}.parquet", index=False)
    (out / "README.md").write_text(f"# {label}\n\n" + ("These numbers are NOT results: development recordings stood in for "
                                                        "every reserved set to check shapes, metrics, bootstrap and files.\n"
                                                        if dry_run else "Final evaluation per CLAUDE.md §12, run once.\n"))
    print(json.dumps({"label": label, "timing": timing, "set_sizes": prov["set_sizes"]}, indent=1, default=float))
    return {"provenance": prov, "loader_touched": loader.touched, "out": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="dry run only: recordings per set")
    ap.add_argument("--classes", nargs="*", default=None, help="dry run only: LOCO classes")
    ap.add_argument("--n-bootstrap", type=int, default=None, help="dry run only")
    a = ap.parse_args()
    if not a.dry_run and (a.limit or a.classes or a.n_bootstrap):
        print("--limit, --classes and --n-bootstrap are only allowed with --dry-run")
        return 2
    try:
        run(a.dry_run, a.limit, a.classes or list(FAULT_CLASSES), a.n_bootstrap)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
