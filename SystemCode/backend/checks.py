"""How a verdict relates to its checks (Stage App-2a.2), from the evidence record only; used by the report and the UI.

- Each check shows its actual relation to the threshold ("≤" / ">" or "≥" / "<"), and its margin (value − threshold,
  absolute and relative to the threshold).
- A check within NARROW_REL (0.1 %, relative) of its threshold is "narrow". For MSP the relative margin is taken on the
  1 − MSP scale, (MSP − τ_p) / (1 − τ_p), because the whole range above τ_p is only about 0.11 % of τ_p (Bhuvesh,
  App-2a.2); the other checks use (value − threshold) / threshold. When the verdict hinges on a narrow check,
  the UI says "narrowly above/below the … threshold" instead of "checks disagree".
- "Checks disagree" is used only when the decision-rule variants disagree for a reason other than a narrow margin, or
  when CNN-6 and the secondary logistic regression rank different classes first.
- Healthy-baseline checks above their limit on a fault verdict are "above healthy limit — consistent with a fault"
  (neutral), not a failure. On an abstention ("further inspection needed") they are "above healthy limit — not consistent
  with healthy": exceeding a healthy-only detector's limit does not establish a fault (it can equally be an unfamiliar
  condition, a sensor problem or an operating point outside the calibration range) (App-4.4).
"""
from __future__ import annotations

NARROW_REL = 0.001
RULE_LINE = "Decision rule: provisional research setting"
RULE_TOOLTIP = ("The decision rule is a provisional setting, not a final rule selection; the final choice will be made "
                "later under stated assumptions about fault frequency and costs.")
VARIANT_NAMES = {
    "A": "classifier first, with an autoencoder override",
    "B": "classifier first (the rule in use)",
    "C": "strict: healthy also needs the autoencoder check",
    "D": "strict: healthy also needs the speed-aware check",
}
_SPEC = [  # key, score, threshold, name, role, direction (+1: value must be ≥ threshold; -1: ≤)
    ("msp", "msp", "tau_p", "classifier confidence (MSP)", "classifier", +1),
    ("maha", "maha", "tau_d", "familiarity (Mahalanobis distance)", "classifier", -1),
    ("ae", "ae_error", "tau_e", "autoencoder error", "healthy baseline", -1),
    ("b1s", "b1s", "tau_b", "speed-aware baseline (B1s)", "healthy baseline", -1),
]


def fmt_score(v: float, digits: int = 4) -> str:
    """A model score in [0, 1] without implying certainty: never "1.0000" or "0.0000"."""
    lim = 10.0 ** -digits
    if v > 1 - lim:
        return "> 0." + "9" * digits
    if v < lim:
        return "< 0." + "0" * (digits - 1) + "1"
    return f"{v:.{digits}f}"


def _num(v: float) -> str:
    return fmt_score(v, 6) if 0 <= v <= 1 else f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"


def checks(r: dict) -> list[dict]:
    d = r["diagnosis"]
    s, t = d["scores"], d["thresholds"]
    healthy = d["decision"] == "healthy"
    out = []
    for key, sk, tk, name, role, sign in _SPEC:
        v, thr = float(s[sk]), float(t[tk])
        ok = v >= thr if sign > 0 else v <= thr
        rel = (v - thr) / (1 - thr) if key == "msp" else (v - thr) / thr
        relation = ("≥" if ok else "<") if sign > 0 else ("≤" if ok else ">")
        if role == "classifier":
            status, tone = ("pass", "good") if ok else ("fail", "warn")
        elif ok:
            status, tone = "within healthy limit", "good"
        elif healthy:
            status, tone = "above healthy limit — caution", "warn"
        elif d["decision"] == "known_fault":
            status, tone = "above healthy limit — consistent with a fault", "neutral"
        else:  # abstention: a healthy-only detector's limit exceeded says "not healthy", not "a fault" (App-4.4)
            status, tone = "above healthy limit — not consistent with healthy", "neutral"
        out.append({"key": key, "name": name, "role": role, "value": v, "threshold": thr, "value_text": _num(v),
                    "threshold_text": _num(thr), "relation": relation, "required": "≥" if sign > 0 else "≤",
                    "passed": ok, "margin_abs": v - thr, "margin_rel": rel, "narrow": abs(rel) <= NARROW_REL,
                    "margin_scale": "relative to 1 − τ_p (the 1 − MSP scale)" if key == "msp" else "relative to the threshold",
                    "status": status, "tone": tone})
    return out


def assessment(r: dict) -> dict:
    d = r["diagnosis"]
    cs = checks(r)
    by = {c["key"]: c for c in cs}
    classifier = [by["msp"], by["maha"]]
    baseline = [by["ae"], by["b1s"]]
    # checks the verdict hinges on: the classifier checks always; the baseline checks when the verdict is healthy
    hinge = classifier + (baseline if d["decision"] == "healthy" else [])
    narrow = [f"Exactly at the {c['name']} threshold (ties are accepted)." if c["value"] == c["threshold"] else
              f"Narrowly {'above' if c['value'] > c['threshold'] else 'below'} the {c['name']} threshold "
              f"({c['margin_rel'] * 100:+.3f} %{' on the 1 − MSP scale' if c['key'] == 'msp' else ''})."
              for c in hinge if c["narrow"]]
    comp = d["rule_comparison"]
    variants_disagree = len(set(comp.values())) > 1
    failing_baseline = [c for c in baseline if not c["passed"]]
    # the variants differ only in how they use the baseline checks; if every failing baseline check is narrow, the
    # disagreement comes from a narrow margin
    narrow_explains = variants_disagree and bool(failing_baseline) and all(c["narrow"] for c in failing_baseline)
    reasons = []
    if variants_disagree and not narrow_explains:
        reasons.append("the alternative decision rules reach different verdicts")
    if d["top_class"] != d["lr_secondary"]["top_class"]:
        reasons.append("CNN-6 and the secondary logistic regression rank different classes first")
    agree = sum(1 for v in comp.values() if v == comp.get("B"))
    return {"checks": cs, "narrow_notes": narrow, "disagree": bool(reasons), "disagree_reasons": reasons,
            "variants_disagree": variants_disagree, "variants_disagree_narrow": narrow_explains,
            "variants": [{"name": VARIANT_NAMES[k], "verdict": v} for k, v in comp.items() if k in VARIANT_NAMES],
            "variants_summary": (f"All {len(comp)} alternative decision rules agree." if not variants_disagree else
                                 f"{agree} of {len(comp)} alternative decision rules agree with the rule in use"
                                 + (" (the difference comes from a narrow margin)." if narrow_explains else ".")),
            "rule_line": RULE_LINE, "rule_tooltip": RULE_TOOLTIP, "narrow_rel": NARROW_REL}
