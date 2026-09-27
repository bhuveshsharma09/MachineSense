"""The app's decision rule (Stage ML-4): one recording's scores → healthy / known fault / unknown, for the rule variants
A, B, C, D and B_with_caution. Pure functions; the three base orderings are ml.openset.three_way (not duplicated here).

Both classifier checks pass = MSP ≥ τ_p AND Mahalanobis ≤ τ_d (ties accepted); AE healthy = AE error ≤ τ_e; B1s healthy
= B1s ≤ τ_b.
- A: classifier-first with the AE override (3C primary).
- B: A without the AE override.
- C: healthy needs both checks, top class normal AND AE healthy (post hoc, exploratory).
- D: B, and healthy additionally requires B1s healthy (otherwise unknown); a candidate, selectable as a strict mode.
- B_with_caution: B's decisions; when B says healthy and B1s and/or AE disagree, a caution is attached.
**The default (B_with_caution) is a provisional demo default, not a final rule selection** (CLAUDE.md §7).
"""
from __future__ import annotations

from ml.openset import three_way

VARIANTS = ("A", "B", "C", "D", "B_with_caution")
DEFAULT_VARIANT = "B_with_caution"
_ORDERING = {"A": "classifier_first", "B": "classifier_first_no_override", "C": "strict_healthy"}
UNKNOWN_LABEL = "further inspection needed"
CAUTION_LABEL = "healthy — caution: a healthy-baseline check disagrees; consider a recheck"
CAUTIONS = ("b1s_disagrees", "ae_disagrees", "both")


def decide(variant: str, top_class: str, msp: float, maha: float, ae_error: float, b1s: float,
           thresholds: dict) -> dict:
    """Decision for one recording. `thresholds` needs tau_p, tau_d, tau_e, tau_b. Returns decision ("healthy" |
    "known_fault" | "unknown"), fault_class (or None), caution (None or one of CAUTIONS), display_label, ae_override."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown rule variant {variant!r}; expected one of {VARIANTS}")
    t = thresholds
    base = "B" if variant in ("D", "B_with_caution") else variant
    d, ov = three_way([top_class], [msp], [maha], [ae_error], t["tau_p"], t["tau_d"], t["tau_e"], _ORDERING[base])
    label = str(d[0])
    b1s_bad, ae_bad = b1s > t["tau_b"], ae_error > t["tau_e"]
    if variant == "D" and label == "healthy" and b1s_bad:
        label = "unknown"
    caution = None
    if variant == "B_with_caution" and label == "healthy" and (b1s_bad or ae_bad):
        caution = "both" if (b1s_bad and ae_bad) else ("b1s_disagrees" if b1s_bad else "ae_disagrees")
    if label == "healthy":
        decision, fault, display = "healthy", None, (CAUTION_LABEL if caution else "healthy")
    elif label == "unknown":
        decision, fault, display = "unknown", None, UNKNOWN_LABEL
    else:
        decision, fault, display = "known_fault", label, f"known fault: {label.replace('_', ' ')}"
    return {"rule_variant": variant, "decision": decision, "fault_class": fault, "caution": caution,
            "display_label": display, "ae_override": bool(ov[0])}


def rule_comparison(top_class: str, msp: float, maha: float, ae_error: float, b1s: float, thresholds: dict) -> dict:
    """The decision of each of A, B, C, D for the same scores (reported with every diagnosis)."""
    out = {}
    for v in ("A", "B", "C", "D"):
        r = decide(v, top_class, msp, maha, ae_error, b1s, thresholds)
        out[v] = r["fault_class"] if r["decision"] == "known_fault" else r["decision"]
    return out
