"""A fixed, deterministic plain-language report built only from the evidence record (no language model, no API key).
The same record always gives the same text; nothing is added that is not in the record. Check relations, margins and
the narrow-margin wording come from backend.checks (shared with the UI)."""
from __future__ import annotations

import re
from functools import lru_cache

from backend.checks import RULE_LINE, RULE_TOOLTIP, assessment, fmt_score
from ml.guidance import UNKNOWN, load_entries, select

TAGS = ("Measured", "Model", "Guidance", "Missing")
FIRST_FOR_UNKNOWN = "further-inspection-needed"  # the verdict's own instruction before generic acquisition checks
VERIFICATION_NOTE = {"ai_drafted": "AI-drafted, not yet checked against a published source",
                     "citation_unverified": "Citation not yet checked against the full text"}
LOCATION = {"underhang": "inner bearing housing (underhang bearing)", "overhang": "outer bearing housing (overhang bearing)",
            "imbalance": "rotor discs", "horizontal_misalignment": "coupling and shaft",
            "vertical_misalignment": "coupling and shaft"}
CHANNEL_NAME = {"under_axial": "underhang axial", "under_radial": "underhang radial", "under_tangential": "underhang tangential",
                "over_axial": "overhang axial", "over_radial": "overhang radial", "over_tangential": "overhang tangential",
                "mic": "microphone"}
METRIC_NAME = {"rms": "RMS", "order_1x": "1× amplitude", "order_2x": "2× amplitude", "kurtosis": "excess kurtosis",
               "env_bpfo1": "envelope line at BPFO", "env_bpfi1": "envelope line at BPFI", "env_bsf2": "envelope line at 2×BSF",
               "env_ftf1": "envelope line at FTF"}


def clean_text(x: str) -> str:
    """User-facing text without internal rule codes."""
    return re.sub(r"\s*\((?:rule )?B_with_caution\)", "", x).replace("provisional demo default", "provisional research setting")


def _name(c: str) -> str:
    return c.replace("_", " ")


def location(cls: str | None) -> str | None:
    if not cls or cls == "normal":
        return None
    return LOCATION.get(cls.split("_")[0]) or LOCATION.get(cls)


def feature_label(f: dict) -> str:
    """A feature-dictionary definition in plain words (its first clause), with the channel, instead of the raw f_ name."""
    d = (f.get("definition") or "").split(";")[0].strip() or f["feature"]
    ch = next((v for k, v in CHANNEL_NAME.items() if f"_{k}_" in f["feature"] + "_"), None)
    return f"{d} ({ch})" if ch and ch not in d else d


def check_text(c: dict) -> str:
    return (f"{c['name'][0].upper() + c['name'][1:]} {c['value_text']} {c['relation']} {c['threshold_text']}: {c['status']} "
            f"(margin {c['margin_abs']:+.4g}, {c['margin_rel'] * 100:+.3f} %{' on the 1 − MSP scale' if c['key'] == 'msp' else ''}).")


def change_text(ch: str, c: dict, ref: str) -> str:
    arrow = "↑" if c["direction"] == "up" else "↓"
    return (f"{CHANNEL_NAME.get(ch, ch)}: {METRIC_NAME.get(c['change_metric'], c['change_metric'])} {arrow} ×"
            f"{c['change_factor']:.1f} vs the healthy training recording {ref}, beyond the typical healthy variation "
            f"(×{c['band']:.1f}); descriptive, not a calibrated anomaly test.")


def _classes(d: dict) -> str:
    probs = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:3]
    return ", ".join(f"{_name(c)} {fmt_score(v)}" for c, v in probs)


def build_report(r: dict) -> str:
    d, q, p = r["diagnosis"], r["signal_quality"], r["physics"]
    a = assessment(r)
    src = r["input"]["rec_id"] or "uploaded recording"
    lines = [f"# MachineSense report — {src}", "",
             "Research demonstration; development results; not validated for real machines.", "",
             "## Verdict", f"**{d['display_label']}**. {RULE_LINE}: {RULE_TOOLTIP}", ""]
    if d["decision"] == "unknown":
        lines.append("The recording did not pass the classifier's confidence and familiarity checks, so it is reported as "
                     "needing further inspection rather than as a known condition.")
    elif d["decision"] == "known_fault":
        lines.append(f"The classifier's top class is {_name(d['top_class'])} and both of its checks passed. "
                     "The predicted fault location is according to the dataset labels.")
    else:
        lines.append("The classifier's top class is normal and both of its checks passed.")
    if d["caution"]:
        lines.append(f"Caution ({_name(d['caution'])}): a healthy-baseline check disagrees; consider a recheck.")
    lines += [*a["narrow_notes"], "", "## Checks against the calibrated thresholds", *[f"- {check_text(c)}" for c in a["checks"]],
              "", a["variants_summary"],
              "", "## Most probable classes (CNN-6 scores)", f"- {_classes(d)}",
              f"- Secondary logistic regression's top class: {_name(d['lr_secondary']['top_class'])}",
              "- Model scores are not the probability that the diagnosis is correct.",
              "", "## Signal quality",
              f"- Measured shaft speed {q['shaft_speed_hz']:.2f} Hz ({p['shaft_speed_rpm']:.0f} rpm); "
              f"speed {'stable' if q['speed_stable'] else 'not stable'} within the recording ({q['speed_range_pct']:.2f} %).",
              *([f"- Warning: {w}" for w in q["warnings"]] or ["- No signal-quality warnings."]),
              "", "## Bearing frequencies at the measured speed",
              ", ".join(f"{k.upper().replace('_', ' ')} {v:.1f} Hz" for k, v in p["bearing_frequencies_hz"].items()) + "."]
    if r.get("xai"):
        lr = r["xai"]["lr"]
        lines += ["", "## Secondary logistic-regression evidence",
                  "These are feature contributions of the secondary logistic-regression model; they explain that model, "
                  f"not the main neural network's decision. Top features for {_name(lr['class'])}:",
                  *[f"- {feature_label(f)} [{f['feature']}]: contribution {f['contribution']:+.3f}" for f in lr["top_features"]]]
    lines += ["", "## Limitations", *[f"- {clean_text(x)}" for x in r["limitations"]],
              "", "## Provenance",
              f"Bundle manifest {r['provenance']['manifest_sha256'][:12]}, schema {r['provenance']['schema_version']}, "
              f"split hash {r['provenance']['split_hash'][:12]}, case {r['case_id']}."]
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=1)
def _entries():
    return tuple(load_entries())


def guidance_condition(d: dict) -> str:
    """The arbitrator's verdict as a guidance condition: the fault class, `normal`, or `unknown` (CLAUDE.md §7 App-5)."""
    return {"known_fault": d["fault_class"], "healthy": "normal"}.get(d["decision"], UNKNOWN)


def guidance_items(r: dict, evidence_names=()) -> list[dict]:
    """Section 04: one Guidance item per step of every entry `ml.guidance.select` offers, cited by entry id; for an unknown
    verdict further-inspection-needed comes first. Nothing selected → one Missing item naming the condition."""
    d, warnings = r["diagnosis"], r["signal_quality"]["warnings"]
    cond = guidance_condition(d)
    avail = {*evidence_names, "speed_hz", "decision.decision", *(["quality.warnings"] if warnings else [])}
    sel = select(list(_entries()), condition=cond, available_evidence=avail, warnings=warnings)
    if cond == UNKNOWN:
        sel.sort(key=lambda e: e.id != FIRST_FOR_UNKNOWN)  # stable: the rest keep select's order
    if not sel:
        return [{"tag": "Missing", "text": f"No curated guidance covers this finding ({_name(cond)}); "
                                           "no inspection steps are offered."}]
    out = []
    for e in sel:
        note = VERIFICATION_NOTE.get(e.verification)
        hover = f"{e.title} — {e.source}" + (f" {e.source_url}" if e.source_url else "")
        out += [{"tag": "Guidance", "text": f"{step} (source: {e.id}{'; ' + note if note else ''})", "cite": e.id,
                 "hover": hover} for step in e.steps]
    return out


# ----------------------------------------------------------------------------- structured report (Stage App-2a)
def build_sections(r: dict, channels: dict | None = None, evidence_names=()) -> list[dict]:
    """The design's four report sections from the evidence record (and, if given, the measured channel evidence). Every
    statement is tagged Measured (from the signal), Model (from a model score or rule), Guidance (a cited step from the
    curated library, `cite` = entry id) or Missing (evidence the system does not have). Items may carry `hover` (raw
    names, or the guidance entry's title and source). `evidence_names` = the feature names the scoring path produces (the
    app passes the bundle's feature dictionary). Deterministic; no language model."""
    d, q = r["diagnosis"], r["signal_quality"]
    a = assessment(r)

    def it(tag, text, hover=None):
        return {"tag": tag, "text": text, **({"hover": hover} if hover else {})}

    probs = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])
    s1 = [it("Model", f"Verdict: {d['display_label']}."), it("Model", f"{RULE_LINE}. {RULE_TOOLTIP}")]
    if d["decision"] == "known_fault":
        s1.append(it("Model", f"Location according to the dataset label: {location(d['fault_class'])}."))
    elif d["decision"] == "unknown":
        cands = [c for c, _ in probs[:2] if c != "normal"]
        s1.append(it("Model", "The classifier's checks did not pass; candidate locations from the two most probable classes: "
                     + ("; ".join(f"{location(c)} ({_name(c)})" for c in cands) or "none (normal ranks in the top two)") + "."))
    s1.append(it("Model", f"Most probable classes (CNN-6 scores): {_classes(d)}. "
                          "Model scores are not the probability that the diagnosis is correct."))
    s2 = [it("Model", check_text(c)) for c in a["checks"]]
    s2.append(it("Measured", f"Measured shaft speed {q['shaft_speed_hz']:.2f} Hz ({60 * q['shaft_speed_hz']:.0f} rpm); speed "
                             f"{'stable' if q['speed_stable'] else 'not stable'} within the recording ({q['speed_range_pct']:.2f} %)."))
    if channels:
        rows = {c["channel"]: c for c in channels["channels"]}
        ref = channels["reference"]["rec_id"]
        s2 += [it("Measured", change_text(ch, rows[ch], ref)) for ch in channels["marked"]]
        if channels.get("healthy_note"):
            n = len(channels["exceeding"])
            s2.append(it("Measured", f"{n} channel{'s' if n != 1 else ''} beyond the typical healthy variation vs the healthy "
                                     f"training recording {ref}; {channels['healthy_note']} (no marker on a healthy verdict: "
                                     f"{channels['multiplicity_note']})." if n else
                                     f"No channel changes beyond the typical healthy variation vs the healthy training "
                                     f"recording {ref}."))
        elif not channels["marked"]:
            s2.append(it("Measured", f"No channel changes beyond the typical healthy variation vs the healthy training "
                                     f"recording {ref}."))
    if r.get("xai"):
        lr = r["xai"]["lr"]
        top = lr["top_features"][:3]
        s2.append(it("Model", f"Secondary logistic regression ({_name(lr['class'])}) leans on: "
                     + "; ".join(f"{feature_label(f)} ({f['contribution']:+.2f})" for f in top) + ".",
                     hover="; ".join(f["feature"] for f in top)))
    s3 = [it("Model", n) for n in a["narrow_notes"]]
    if a["variants_disagree"]:
        s3.append(it("Model", a["variants_summary"] + " "
                     + "; ".join(f"{v['name']}: {_name(v['verdict'])}" for v in a["variants"]) + "."))
    if d["caution"]:
        s3.append(it("Model", f"Caution: {_name(d['caution'])} — a healthy-baseline check disagrees; consider a recheck."))
    if d["top_class"] != d["lr_secondary"]["top_class"]:
        s3.append(it("Model", f"CNN-6 ranks {_name(d['top_class'])} first; the secondary logistic regression ranks "
                              f"{_name(d['lr_secondary']['top_class'])} first."))
    s3 += [it("Measured", f"Warning: {w}.") for w in q["warnings"] if d["decision"] == "healthy" or "not calibrated" not in w]
    if r["source"] == "dataset":
        s3.append(it("Missing", "This recording was used to build the model; the result is not evidence of accuracy."))
    s3 += [it("Missing", "No maintenance history, operating context or repeat recording is available."),
           it("Missing", "The healthy baseline is calibrated on 5 healthy recordings from one folder.")]
    s4 = guidance_items(r, evidence_names)
    return [{"n": "01", "title": "What the readings suggest", "items": s1},
            {"n": "02", "title": "Supporting evidence", "items": s2},
            {"n": "03", "title": "Conflicting or missing evidence", "items": s3},
            {"n": "04", "title": "What to inspect next", "items": s4}]
