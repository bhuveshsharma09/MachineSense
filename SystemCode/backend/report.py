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
