"""A fixed, deterministic plain-language report built only from the evidence record (no language model, no API key).
The same record always gives the same text; nothing is added that is not in the record."""
from __future__ import annotations


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def build_report(r: dict) -> str:
    d, q, p = r["diagnosis"], r["signal_quality"], r["physics"]
    s, t = d["scores"], d["thresholds"]
    src = r["input"]["rec_id"] or "uploaded recording"
    lines = [f"# MachineSense report — {src}", "",
             "Research demonstration; development results; not validated for real machines.", "",
             "## Verdict", f"**{d['display_label']}** (rule {d['rule_variant']}; {d['rule_status'] or 'rule status not recorded'}).", ""]
    if d["decision"] == "unknown":
        lines.append("The recording did not pass the classifier's confidence and familiarity checks, so it is reported as "
                     "needing further inspection rather than as a known condition.")
    elif d["decision"] == "known_fault":
        lines.append(f"The classifier's top class is {d['top_class'].replace('_', ' ')} and both of its checks passed. "
                     "The predicted fault location is according to the dataset labels.")
    else:
        lines.append("The classifier's top class is normal and both of its checks passed.")
    if d["caution"]:
        lines.append(f"Caution ({d['caution'].replace('_', ' ')}): a healthy-baseline check disagrees; consider a recheck.")
    lines += ["", "## Checks against the calibrated thresholds",
              f"- Classifier confidence (MSP): {s['msp']:.6f} — needs ≥ {t['tau_p']:.6f}: "
              f"{'pass' if s['msp'] >= t['tau_p'] else 'fail'}",
              f"- Familiarity (Mahalanobis distance): {s['maha']:.3f} — needs ≤ {t['tau_d']:.3f}: "
              f"{'pass' if s['maha'] <= t['tau_d'] else 'fail'}",
              f"- Autoencoder reconstruction error: {s['ae_error']:.4f} — healthy baseline ≤ {t['tau_e']:.4f}: "
              f"{'within' if s['ae_error'] <= t['tau_e'] else 'above'}",
              f"- Speed-aware baseline (B1s): {s['b1s']:.4f} — healthy baseline ≤ {t['tau_b']:.4f}: "
              f"{'within' if s['b1s'] <= t['tau_b'] else 'above'}",
              "", "Decisions of the other rule variants: " +
              ", ".join(f"{k}: {v.replace('_', ' ')}" for k, v in d["rule_comparison"].items()) + ".",
              "", "## Most probable classes",
              *[f"- {c.replace('_', ' ')}: {_pct(v)}" for c, v in sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:3]],
              f"- Secondary logistic regression's top class: {d['lr_secondary']['top_class'].replace('_', ' ')}",
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
                  f"not the main neural network's decision. Top features for {lr['class'].replace('_', ' ')}:",
                  *[f"- {f['feature']} ({f['definition'] or 'no definition'}): contribution {f['contribution']:+.3f}"
                    for f in lr["top_features"]]]
    lines += ["", "## Limitations", *[f"- {x}" for x in r["limitations"]],
              "", "## Provenance",
              f"Bundle manifest {r['provenance']['manifest_sha256'][:12]}, schema {r['provenance']['schema_version']}, "
              f"split hash {r['provenance']['split_hash'][:12]}, case {r['case_id']}."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- structured report (Stage App-2a)
TAGS = ("Measured", "Model", "Guidance", "Missing")
GUIDANCE_PLACEHOLDER = "guidance library not yet connected"
LOCATION = {"underhang": "inner bearing housing (underhang bearing)", "overhang": "outer bearing housing (overhang bearing)",
            "imbalance": "rotor discs", "horizontal_misalignment": "coupling and shaft",
            "vertical_misalignment": "coupling and shaft"}


def _name(c: str) -> str:
    return c.replace("_", " ")


def location(cls: str | None) -> str | None:
    if not cls or cls == "normal":
        return None
    return LOCATION.get(cls.split("_")[0]) or LOCATION.get(cls)


def build_sections(r: dict, channels: dict | None = None) -> list[dict]:
    """The design's four report sections from the evidence record (and, if given, the measured channel evidence). Every
    statement is tagged Measured (from the signal), Model (from a model score or rule), Guidance (a placeholder until the
    guidance library exists) or Missing (evidence the system does not have). Deterministic; no language model."""
    d, q = r["diagnosis"], r["signal_quality"]
    s, t = d["scores"], d["thresholds"]
    it = lambda tag, text: {"tag": tag, "text": text}  # noqa: E731
    probs = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])
    s1 = [it("Model", f"Verdict: {d['display_label']} (rule {d['rule_variant']}, {d['rule_status'] or 'status not recorded'}).")]
    if d["decision"] == "known_fault":
        s1.append(it("Model", f"Location according to the dataset label: {location(d['fault_class'])}."))
    elif d["decision"] == "unknown":
        cands = [c for c, _ in probs[:2] if c != "normal"]
        s1.append(it("Model", "The classifier's checks did not pass; candidate locations from the two most probable classes: "
                     + ("; ".join(f"{location(c)} ({_name(c)})" for c in cands) or "none (normal ranks in the top two)") + "."))
    s1.append(it("Model", "Most probable classes (CNN-6): " + ", ".join(f"{_name(c)} {_pct(v)}" for c, v in probs[:3])
                 + ". Model scores are not the probability that the diagnosis is correct."))
    s2 = [it("Model", f"Classifier confidence (MSP) {s['msp']:.6f} vs threshold ≥ {t['tau_p']:.6f}: "
                      f"{'pass' if s['msp'] >= t['tau_p'] else 'fail'}."),
          it("Model", f"Familiarity (Mahalanobis distance) {s['maha']:.3f} vs threshold ≤ {t['tau_d']:.3f}: "
                      f"{'pass' if s['maha'] <= t['tau_d'] else 'fail'}."),
          it("Model", f"Healthy baseline, autoencoder error {s['ae_error']:.4f} vs ≤ {t['tau_e']:.4f}: "
                      f"{'within' if s['ae_error'] <= t['tau_e'] else 'above'}."),
          it("Model", f"Healthy baseline, speed-aware B1s {s['b1s']:.4f} vs ≤ {t['tau_b']:.4f}: "
                      f"{'within' if s['b1s'] <= t['tau_b'] else 'above'}."),
          it("Measured", f"Measured shaft speed {q['shaft_speed_hz']:.2f} Hz ({60 * q['shaft_speed_hz']:.0f} rpm); speed "
                         f"{'stable' if q['speed_stable'] else 'not stable'} within the recording ({q['speed_range_pct']:.2f} %).")]
    if channels:
        rows = {c["channel"]: c for c in channels["channels"]}
        for ch in channels["largest_change"]:
            c = rows[ch]
            s2.append(it("Measured", f"Largest change vs the healthy training reference {channels['reference']['rec_id']}: "
                                     f"{_name(ch)} {c['change_metric'].replace('_', ' ')} × {c['change']:.2f} of the reference "
                                     "(descriptive, not a calibrated anomaly test)."))
    if r.get("xai"):
        lr = r["xai"]["lr"]
        top = lr["top_features"][:3]
        s2.append(it("Model", f"Secondary logistic regression ({_name(lr['class'])}) leans on: "
                     + "; ".join(f"{f['feature']} ({f['contribution']:+.2f})" for f in top) + "."))
    s3 = []
    comp = d["rule_comparison"]
    if len(set(comp.values())) > 1:
        s3.append(it("Model", "Rule variants disagree: " + ", ".join(f"{k} {_name(v)}" for k, v in comp.items()) + "."))
    if d["caution"]:
        s3.append(it("Model", f"Caution: {_name(d['caution'])} — a healthy-baseline check disagrees; consider a recheck."))
    if d["top_class"] != d["lr_secondary"]["top_class"]:
        s3.append(it("Model", f"CNN-6 ranks {_name(d['top_class'])} first; the secondary logistic regression ranks "
                              f"{_name(d['lr_secondary']['top_class'])} first."))
    s3 += [it("Measured", f"Signal-quality warning: {w}.") for w in q["warnings"]
           if d["decision"] == "healthy" or "not calibrated" not in w]
    if r["source"] == "dataset":
        s3.append(it("Missing", "This recording was used to build the model; the result is not evidence of accuracy."))
    s3 += [it("Missing", "No maintenance history, operating context or repeat recording is available."),
           it("Missing", "The healthy baseline is calibrated on 5 healthy recordings from one folder.")]
    s4 = [it("Guidance", f"Inspection steps for this finding — {GUIDANCE_PLACEHOLDER}.")]
    return [{"n": "01", "title": "What the readings suggest", "items": s1},
            {"n": "02", "title": "Supporting evidence", "items": s2},
            {"n": "03", "title": "Conflicting or missing evidence", "items": s3},
            {"n": "04", "title": "What to inspect next", "items": s4}]
