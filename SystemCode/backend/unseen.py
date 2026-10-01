"""Stage App-6: "Explore an unseen fault" — a saved leave-one-class-out CNN-6 (artifacts/final/loco/<class>/, §12) run on
a recording of the class it never trained on. A SIMULATED unseen fault inside MAFAULDA, not evidence about genuinely novel
real-world faults. The normal application and its bundle are unchanged.

This module holds the single eligibility predicate (`may_show`), used by the build step and by every request."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.splits import FAULT_CLASSES

ROLES = ("withheld", "familiar")
# The partitions consumed by the single declared final evaluation (F1); a recording in any of them is never shown.
ALLOWED_CLOSED = ("train", "val")
SEALED_COLUMNS = ("split_speed_block", "split_severity")


def split_eligible(row: pd.Series) -> bool:
    """split_closed ∈ {train, val} AND split_speed_block ≠ test AND split_severity ≠ test (metadata.csv)."""
    return row["split_closed"] in ALLOWED_CLOSED and all(row[c] != "test" for c in SEALED_COLUMNS)


def may_show(rec_id: str, withheld: str, role: str, meta: pd.DataFrame, familiar_excluded) -> tuple[bool, str]:
    """May this recording be shown on the page, for this hidden class and role? Returns (allowed, reason).

    - Every role: the recording exists in metadata.csv and passes `split_eligible` (reserved partitions stay sealed).
    - "withheld": its class is the selected model's withheld class (unseen by that model by construction).
    - "familiar": its class is one the model learned, and it is neither one of that run's calibration recordings nor a
      threshold-defining recording (`familiar_excluded`: both sets for that run)."""
    if withheld not in FAULT_CLASSES:
        return False, f"unknown hidden class {withheld!r}"
    if role not in ROLES:
        return False, f"unknown role {role!r}"
    m = meta.set_index("rec_id") if "rec_id" in meta.columns else meta
    if rec_id not in m.index:
        return False, "not a recording of this dataset"
    row = m.loc[rec_id]
    if not split_eligible(row):
        return False, "this is a reserved test recording; reserved partitions stay sealed and are not shown"
    if role == "withheld":
        return (True, "") if row["class10"] == withheld else (False, f"not a recording of the hidden class {withheld}")
    if row["class10"] == withheld:
        return False, "the familiar comparison must be a condition the model learned"
    if rec_id in set(familiar_excluded):
        return False, "this recording calibrated the model's thresholds, so it is not offered as a familiar example"
    return True, ""


# ----------------------------------------------------------------------------------------------------------- copy
SIMULATED = ("This is a simulated unseen fault inside one dataset (MAFAULDA) from one test rig: a model trained without "
             "one fault class is shown recordings of that class. It is not evidence that the system recognises genuinely "
             "novel real-world faults.")
FURTHER_INSPECTION = ("\"Further inspection needed\" means the proposed familiar label was not accepted. It does not "
                      "identify the hidden fault, and it is not a claim that a new fault type has been discovered.")
EASIER_PART = ("The eligible examples exclude the largest severities and the fastest shaft speeds, because those "
               "partitions are reserved and sealed. This page therefore shows the easier part of the unfamiliar-fault "
               "problem, not the hardest part.")
GATES_NOTE = ("The AE-6 and B1s healthy-baseline checks are shown only as a possible caution; under the default rule "
              "(B_with_caution) they do not decide the verdict.")
TOLERANCE_REL = 1e-9  # live vs saved scores (App-6 item 4; repeated CPU scoring is bit-identical, ML-4.2)
SCORE_KEYS = ("msp", "maha", "ae", "b1s")


# ----------------------------------------------------------------------------------------------------------- scoring
def run_info(final_models, withheld: str) -> dict:
    """What one LOCO run learned and what it must never offer as a familiar example (live, from artifacts/final/)."""
    lm = final_models.loco[withheld]
    sub = final_models.directory / "loco" / withheld
    cal = pd.read_csv(sub / "calibration_scores.csv")
    defining = sorted({r for v in lm.thresholds.get("defining_recordings", {}).values() for r in v})
    return {"classes_learned": list(lm.classes), "calibration_ids": sorted(cal.rec_id),
            "calibration_classes": sorted(set(cal.class10)), "defining_recordings": defining,
            "thresholds": {k: float(lm.thresholds[k]) for k in ("tau_p", "tau_d", "tau_e", "tau_b")},
            "n_calibration": int(lm.thresholds["n_calibration"])}


def score(final_models, withheld: str, inputs: dict, floor: float) -> dict:
    """ml.final.score_cnn with the run's CNN-6, normalisation and SHARED Mahalanobis, plus the shared AE-6 and B1s."""
    from ml.final import score_cnn
    lm = final_models.loco[withheld]
    s = score_cnn(inputs, lm.cnn, lm.cnn_norm, lm.classes, lm.maha, final_models.ae, final_models.ae_norm,
                  final_models.ae_cfg, final_models.b1s, floor)
    top3 = s["probs"].sort_values(ascending=False).head(3)
    return {"top": s["top"], "top_probabilities": [[str(k), float(v)] for k, v in top3.items()],
            **{k: float(s[k]) for k in SCORE_KEYS}}


def present(withheld: str, role: str, row: pd.Series, s: dict, info: dict, source: str, warnings: list[str]) -> dict:
    """Every value the page displays for one (hidden class, recording): the checks are ml.rejection.failed_checks (the
    rule's own inequalities, ties accepted) and the verdict is ml.arbitrator.decide; nothing is recomputed here."""
    from ml.arbitrator import DEFAULT_VARIANT, decide, rule_comparison
    from ml.rejection import failed_checks

    t = info["thresholds"]
    f = failed_checks(s["msp"], s["maha"], s["ae"], s["b1s"], **t)
    d = decide(DEFAULT_VARIANT, s["top"], s["msp"], s["maha"], s["ae"], s["b1s"], t)
    return {"withheld": withheld, "role": role, "rec_id": str(row.name), "true_class": row["class10"],
            "severity": float(row["severity"]), "speed_hz": float(row["fr_measured_hz"]),
            "used_for": {"train": "model training", "val": "threshold calibration"}[row["split_closed"]],
            "classes_learned": info["classes_learned"], "proposed_label": s["top"],
            "top_probabilities": s["top_probabilities"], "scores": {k: s[k] for k in SCORE_KEYS}, "thresholds": t,
            "checks": {"confidence": {"value": s["msp"], "threshold": t["tau_p"], "passed": not bool(f["msp"]),
                                      "rule": "top probability ≥ τ_p"},
                       "familiarity": {"value": s["maha"], "threshold": t["tau_d"], "passed": not bool(f["maha"]),
                                       "rule": "Mahalanobis distance ≤ τ_d"},
                       "ae": {"value": s["ae"], "threshold": t["tau_e"], "passed": not bool(f["ae"]), "gates_verdict": False},
                       "b1s": {"value": s["b1s"], "threshold": t["tau_b"], "passed": not bool(f["b1s"]), "gates_verdict": False}},
            "decision": d, "rule_comparison": rule_comparison(s["top"], s["msp"], s["maha"], s["ae"], s["b1s"], t),
            "speed_warnings": warnings, "source": source}


def speed_ranges(meta: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Training speed range (closed train within the eligible pool) and the calibration normals' speed range."""
    el = meta[meta.apply(split_eligible, axis=1)]
    tr = el[el.split_closed == "train"].fr_measured_hz
    cn = el[(el.split_closed == "val") & (el.class10 == "normal")].fr_measured_hz
    return [float(tr.min()), float(tr.max())], [float(cn.min()), float(cn.max())]


# ----------------------------------------------------------------------------------------------------------- build
CATEGORIES = ("successful rejection", "honest failure", "familiar comparison")


def select(rows: list[dict], runs: dict) -> dict:
    """Per hidden class: a successful rejection, an honest failure and a familiar comparison (deterministic; reasons
    recorded). Preference within a category: no speed-range warning, then the median Mahalanobis distance (a typical case,
    not the most extreme), then rec_id. A category with no eligible recording is reported, never substituted."""
    df = pd.DataFrame(rows)
    out = {}
    for c, info in runs.items():
        g = df[df.withheld == c]
        wh, fam = g[g.role == "withheld"], g[g.role == "familiar"]
        rejected = wh[wh.decision == "unknown"]
        accepted = wh[wh.decision != "unknown"]
        wrong_fault = accepted[accepted.decision == "known_fault"]
        proposed = wh.top.value_counts()
        mistaken_for = proposed.index[0] if len(proposed) else None
        ok_fam = fam[((fam.decision == "known_fault") & (fam.fault_class == fam.true_class)) |
                     ((fam.true_class == "normal") & (fam.decision == "healthy"))]
        ok_fam = ok_fam[ok_fam.caution.isna() & ok_fam.variants_agree]
        same = ok_fam[ok_fam.true_class == mistaken_for]
        cands = {"successful rejection": (rejected, "a withheld recording whose checks failed, so the proposed familiar "
                                                    "label was not accepted"),
                 "honest failure": (wrong_fault if len(wrong_fault) else accepted,
                                    "a withheld recording that passed both checks and received a confident familiar label"
                                    + ("" if len(wrong_fault) else " (healthy: no wrong fault label occurred)")),
                 "familiar comparison": (same if len(same) else ok_fam,
                                         "a learned condition the same model accepts with its correct label, no caution, all "
                                         "rule variants agreeing" + (f"; the condition this model most often proposes for "
                                                                     f"the hidden class ({mistaken_for})" if len(same) else ""))}
        healthy_acc = accepted[accepted.decision == "healthy"]
        cand_list = list(cands.items())
        if len(wrong_fault) and len(healthy_acc):  # addendum (Bhuvesh, 1 Oct): the typical healthy acceptance as well
            cand_list.insert(2, ("honest failure", (healthy_acc, "a withheld recording that passed both checks and was "
                                                                 f"called healthy ({len(healthy_acc)} of {len(wh)} were)")))
        picks, missing, why_missing = [], [], {}
        for cat, (frame, why) in cand_list:
            if frame.empty:
                missing.append(cat)
                why_missing[cat] = (f"every one of the {len(wh)} eligible withheld recordings was rejected" if
                                    cat == "honest failure" else "no eligible recording met this category's conditions")
                continue
            f2 = frame.assign(_warn=frame.speed_warnings.map(bool), _dev=(frame.maha - frame.maha.median()).abs())
            r = f2.sort_values(["_warn", "_dev", "rec_id"]).iloc[0]
            pick = {"category": cat, "role": r.role, "rec_id": r.rec_id, "true_class": r.true_class,
                    "reason": f"{why}; median-distance case among {len(frame)}" +
                              ("" if not r._warn else "; carries a speed-range warning (no warning-free case)")}
            if cat == "honest failure":
                pick["failure_kind"] = "called healthy" if r.decision == "healthy" else "a wrong familiar fault"
            picks.append(pick)
        out[c] = {**info, "examples": picks, "categories_without_an_eligible_recording": missing,
                  "why_missing": why_missing,
                  "counts": {"withheld_recordings": int(len(wh)), "rejected": int(len(rejected)),
                             "accepted_as_a_wrong_fault": int(len(wrong_fault)),
                             "accepted_as_healthy": int((accepted.decision == "healthy").sum()),
                             "most_often_proposed_label": mistaken_for,
                             "familiar_candidates": int(len(fam)), "familiar_accepted_correctly": int(len(ok_fam))}}
    return out


def build() -> dict:
    """Sweep every eligible recording through the real LOCO models (live), then curate. Run where artifacts/final/ and the
    raw recordings exist."""
    import json
    import time

    from backend.engine import Engine
    from backend.settings import unseen_file, unseen_sweep_file
    from ml.final import load_final, recording_inputs
    from ml.inference import _metadata, load_recording, speed_warnings

    eng = Engine()
    cfg, floor = eng.bundle.config, eng.bundle.model_config["prob_floor"]
    fm = load_final()
    meta = _metadata().reset_index()
    m = meta.set_index("rec_id")
    runs = {c: run_info(fm, c) for c in fm.manifest["loco_classes"]}
    excluded = {c: set(i["calibration_ids"]) | set(i["defining_recordings"]) for c, i in runs.items()}
    tr_rng, cal_rng = speed_ranges(meta)
    sweep_path = unseen_sweep_file()
    if sweep_path.exists():
        rows = json.loads(sweep_path.read_text())
    else:
        rows, t0 = [], time.perf_counter()
        eligible = sorted(meta.loc[meta.apply(split_eligible, axis=1), "rec_id"])
        for i, rec in enumerate(eligible):
            x, fr = load_recording(rec)
            inputs = recording_inputs(x, fr, cfg)
            warn = speed_warnings(fr, tr_rng, cal_rng)
            for c in runs:
                role = "withheld" if m.at[rec, "class10"] == c else "familiar"
                if not may_show(rec, c, role, meta, excluded[c])[0]:
                    continue
                p = present(c, role, m.loc[rec], score(fm, c, inputs, floor), runs[c], "saved", warn)
                rows.append({"withheld": c, "rec_id": rec, "role": role, "true_class": p["true_class"],
                             "top": p["proposed_label"], "decision": p["decision"]["decision"],
                             "fault_class": p["decision"]["fault_class"], "caution": p["decision"]["caution"],
                             "variants_agree": len(set(p["rule_comparison"].values())) == 1,
                             "speed_warnings": warn, **p["scores"], "result": p})
            if i % 100 == 0:
                print(f"  {i}/{len(eligible)} recordings, {time.perf_counter() - t0:.0f} s", flush=True)
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps(rows))
    sel = select(rows, runs)
    by_key = {(r["withheld"], r["rec_id"]): r["result"] for r in rows}
    for c, run in sel.items():
        for e in run["examples"]:
            e["result"] = by_key[(c, e["rec_id"])]
        run["familiar_excluded"] = sorted(excluded[c])
    out = {"stage": "App-6", "simulated": SIMULATED, "further_inspection": FURTHER_INSPECTION, "easier_part": EASIER_PART,
           "gates_note": GATES_NOTE, "seed": fm.manifest["seed"], "final_manifest_sha256": fm.manifest_sha256,
           "split_hash": fm.manifest["split_hash"], "scoring_path": "ml.final.score_cnn on the CPU; shared Mahalanobis; "
           "verdict ml.arbitrator.decide (B_with_caution)", "tolerance_rel": TOLERANCE_REL, "default_hidden_class": "imbalance",
           "speed_ranges": {"training_hz": tr_rng, "calibration_normals_hz": cal_rng}, "n_swept_rows": len(rows),
           "runs": sel}
    unseen_file().write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# ----------------------------------------------------------------------------------------------------------- service
class UnseenService:
    """Serves the curated file; recomputes live when artifacts/final/ verifies and the raw recordings exist."""

    def __init__(self, bundle, curated_path=None, final_directory=None):
        import json

        from backend.settings import unseen_file
        from ml.inference import _metadata
        self.cfg, self.floor = bundle.config, bundle.model_config["prob_floor"]
        self.meta = _metadata().reset_index()
        p = curated_path or unseen_file()
        self.curated = json.loads(p.read_text()) if p.exists() else None
        self.final_directory = final_directory
        self._live, self._live_reason, self._cache = None, None, {}

    def live(self):
        """The verified final models, or None (with the reason) when they or the raw recordings are absent."""
        if self._live is None and self._live_reason is None:
            from ml.final import final_dir, load_final
            from backend.settings import raw_dir
            d = Path(self.final_directory) if self.final_directory else final_dir()
            if not (d / "manifest.json").exists():
                self._live_reason = "the leave-one-class-out weights (artifacts/final/) are not present; saved values shown"
            elif not raw_dir().exists():
                self._live_reason = "the raw recordings are not present; saved values shown"
            else:
                try:
                    self._live = load_final(d)
                except Exception as e:  # noqa: BLE001 (a failed verification falls back to the saved values)
                    self._live_reason = f"the final models did not verify ({type(e).__name__}); saved values shown"
        return self._live

    def summary(self) -> dict:
        if self.curated is None:
            return {"built": False, "note": "the unseen-fault selection has not been built (python -m backend.unseen build)"}
        live = self.live()
        runs = {c: {k: v for k, v in r.items() if k not in ("familiar_excluded", "calibration_ids")} | {
            "examples": [{k: v for k, v in e.items() if k != "result"} for e in r["examples"]]}
            for c, r in self.curated["runs"].items()}
        return {**{k: v for k, v in self.curated.items() if k != "runs"}, "built": True, "runs": runs,
                "live": {"available": live is not None, "reason": self._live_reason or ""}}

    def _excluded(self, withheld: str) -> set:
        live = self.live()
        if live is not None:
            info = run_info(live, withheld)
            return set(info["calibration_ids"]) | set(info["defining_recordings"])
        return set(self.curated["runs"][withheld]["familiar_excluded"]) if self.curated else set()

    def run(self, withheld: str, rec_id: str, role: str) -> dict:
        """The displayed result for one example. Raises PermissionError (not eligible) or LookupError (not available)."""
        ok, why = may_show(rec_id, withheld, role, self.meta, self._excluded(withheld))
        if not ok:
            raise PermissionError(why)
        live = self.live()
        if live is None:
            for e in (self.curated or {}).get("runs", {}).get(withheld, {}).get("examples", []):
                if e["rec_id"] == rec_id and e["role"] == role:
                    return e["result"]
            raise LookupError("only the curated examples are available without the model weights")
        key = (withheld, rec_id, role)
        if key not in self._cache:
            from ml.final import recording_inputs
            from ml.inference import load_recording, speed_warnings
            x, fr = load_recording(rec_id)
            tr, cal = speed_ranges(self.meta)
            self._cache[key] = present(withheld, role, self.meta.set_index("rec_id").loc[rec_id],
                                       score(live, withheld, recording_inputs(x, fr, self.cfg), self.floor),
                                       run_info(live, withheld), "live", speed_warnings(fr, tr, cal))
        return self._cache[key]


def main() -> int:
    import json
    import sys
    if sys.argv[1:] != ["build"]:
        print("usage: python -m backend.unseen build")
        return 2
    out = build()
    print(json.dumps({c: {"examples": [(e["category"], e["rec_id"], e["result"]["decision"]["display_label"])
                                       for e in r["examples"]],
                          "missing": r["categories_without_an_eligible_recording"], "counts": r["counts"]}
                      for c, r in out["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
