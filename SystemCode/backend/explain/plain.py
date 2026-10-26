"""Mechanic-level wording shared by the template explanation (and mirrored in the frontend's Mechanic view): plain words,
no percentages and no raw scores. Derived only from the evidence record, the check assessment and the measured
channel evidence."""
from __future__ import annotations

import re

IN_SAMPLE = "This recording was used to build the model, so the result is not evidence of accuracy."
LIMITATION = "Research demonstration; not validated on real machines."

FAULT_PLAIN = {
    "imbalance": "imbalance", "horizontal_misalignment": "horizontal misalignment",
    "vertical_misalignment": "vertical misalignment",
    "underhang_ball": "bearing ball fault", "underhang_cage": "bearing cage fault", "underhang_outer": "bearing outer-race fault",
    "overhang_ball": "bearing ball fault", "overhang_cage": "bearing cage fault", "overhang_outer": "bearing outer-race fault",
    "normal": "normal operation",
}
# App-8.6 item 7: the page's names everywhere (Underhang / Overhang bearing; axial, radial, tangential)
PART_PLAIN = {"in_brg": "Underhang bearing housing", "out_brg": "Overhang bearing housing", "rotor": "Rotor discs",
              "coupling": "Coupling", "shaft": "Shaft"}
SENSOR_PLAIN = {"under_axial": "Underhang bearing, axial", "under_radial": "Underhang bearing, radial",
                "under_tangential": "Underhang bearing, tangential", "over_axial": "Overhang bearing, axial",
                "over_radial": "Overhang bearing, radial", "over_tangential": "Overhang bearing, tangential",
                "mic": "microphone"}
# App-8.6 item 8: the checks in plain words for the explanation (no classifier confidence, familiarity, autoencoder,
# speed-aware baseline, MSP or Mahalanobis)
CHECK_PLAIN = {"msp": "the model is sure of its match", "maha": "the recording looks like ones the model has seen",
               "ae": "the recording looks like a healthy machine (overall pattern)",
               "b1s": "the recording looks like a healthy machine at this speed"}
# App-8.6 item 17: the limitation sentence given to the model, to be repeated word for word
RESEARCH = "This is a research demonstration. It is not validated on real machines."
MEASURE_PLAIN = {"rms": "overall vibration level (RMS)", "order_1x": "vibration once per rotation (1×)",
                 "order_2x": "vibration twice per rotation (2×)", "env_bpfo1": "repeated impacts at the outer-race rate (envelope spectrum)",
                 "env_bpfi1": "repeated impacts at the inner-race rate (envelope spectrum)",
                 "env_bsf2": "repeated impacts at the ball rate (envelope spectrum)",
                 "env_ftf1": "repeated impacts at the cage rate (envelope spectrum)", "kurtosis": "impulsiveness (excess kurtosis)"}


def next_step(items: list[dict]) -> str:
    """The explanation's next step: section 04's items (backend.report.guidance_items, the one selection) as given, one
    per line, each with its "(source: <id>)" citation and verification note. Set by code, never written by the model."""
    return "\n".join(i["text"] for i in items)


def parts_for(cls: str | None) -> list[str]:
    if not cls or cls == "normal":
        return []
    if cls.startswith("underhang"):
        return ["in_brg"]
    if cls.startswith("overhang"):
        return ["out_brg"]
    if cls == "imbalance":
        return ["rotor"]
    if cls.endswith("misalignment"):
        return ["coupling", "shaft"]
    return []


def finding(record: dict) -> str:
    d = record["diagnosis"]
    if d["decision"] == "known_fault":
        return f"Possible {FAULT_PLAIN.get(d['fault_class'], d['fault_class'].replace('_', ' '))}"
    if d["decision"] == "unknown":
        return "Further inspection needed"
    if d["caution"]:
        return "No fault detected — a check disagrees; repeat the reading or inspect further"
    return "No fault detected in this recording"


def location(record: dict) -> dict:
    d = record["diagnosis"]
    if d["decision"] == "known_fault":
        parts = parts_for(d["fault_class"])
        return {"kind": "predicted", "parts": parts,
                "text": " and ".join(PART_PLAIN[p] for p in parts) + " — according to the dataset label (model prediction)"}
    if d["decision"] == "unknown":
        top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
        parts = list(dict.fromkeys(p for c, _ in top for p in parts_for(c)))
        return {"kind": "candidate", "parts": parts,
                "text": ("Candidate parts: " + ", ".join(PART_PLAIN[p].lower() for p in parts)
                         + " — from the two most likely fault types (model prediction)") if parts else
                        "No candidate part: the most likely types include normal operation"}
    return {"kind": "none", "parts": [], "text": None}


_WARN = [
    (r"speed outside the range the model was trained on", "The rotation speed is outside the range the model was trained on."),
    (r"healthy thresholds were not calibrated at this speed", "The shaft speed is outside the calibrated range; the healthy checks are less reliable at this speed."),
    (r"shaft speed varies", "The rotation speed was not steady during the recording."),
    (r"windows use the recording speed|no usable tacho", "The speed sensor signal was partly unusable."),
    (r"glitched tacho intervals", "The speed sensor signal had glitches."),
    (r"microphone bursts removed", "Short bursts were removed from the microphone signal."),
    (r"possible clipping", "A vibration sensor may have been overloaded (clipping)."),
    (r"missing \(NaN\) samples", "Some samples were missing."),
]


def warnings(record: dict) -> list[str]:
    """Warnings that affect interpretation, in plain words (no numbers); the calibration-speed warning on every verdict
    (App-8.6 item 4)."""
    out = []
    for w in record["signal_quality"]["warnings"]:
        plain = next((p for pat, p in _WARN if re.search(pat, w)), "A recording-quality warning applies.")
        if plain not in out:
            out.append(plain)
    return out


def change_rank(c: dict) -> tuple:
    """App-8.6 item 28: increases beyond typical variation first (largest beyond the band first), then decreases."""
    return (c["direction"] != "up", not c.get("exceeds", True), -float(c["change_factor"]) / float(c.get("band") or 1))


def change_sentence(ch: str, c: dict) -> str:
    """One measured change with its direction and factor against the one healthy reference recording (App-8.6 item 9)."""
    return (f"The {MEASURE_PLAIN.get(c['change_metric'], c['change_metric'])} on the {SENSOR_PLAIN.get(ch, ch)} sensor is "
            f"{float(c['change_factor']):.1f}× {'higher' if c['direction'] == 'up' else 'lower'} than the healthy reference "
            "recording at a similar speed (a difference, not a proven fault).")


def supporting(record: dict, assessment: dict) -> str:
    d = record["diagnosis"]
    by = {c["key"]: c for c in assessment["checks"]}
    base_ok = by["ae"]["passed"] and by["b1s"]["passed"]
    if d["decision"] == "healthy":
        base = ("Both healthy-baseline checks agree it looks like the healthy recordings" if base_ok else
                "One or both healthy-baseline checks see a difference from the healthy recordings")
    else:
        base = ("The healthy-baseline checks also see a difference from the healthy recordings" if not base_ok else
                "The healthy-baseline checks do not see a difference from the healthy recordings")
    lr_same = d["top_class"] == d["lr_secondary"]["top_class"]
    return base + "; the secondary model " + ("agrees." if lr_same else "points to a different condition.")
