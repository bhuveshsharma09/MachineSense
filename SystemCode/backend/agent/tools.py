"""The assistant's four read-only tools over a case stored by /api/diagnose. They read the stored record (and, for the
channel evidence, the same builder the report's channel view uses) and never run a model. Every fact an answer may rest on
is a leaf {"value": …, "ref": "<path>"} so the review can trace it; section 04 items carry their ref beside the item.
No tool exposes the CNN gradient map (CLAUDE.md §11 rule 11)."""
from __future__ import annotations

from backend.checks import _SPEC, assessment
from backend.evidence import ACCEL, CHANGE_LABEL
from backend.report import guidance_items, location

CHANNELS = (*ACCEL, "mic")
SCORE_KEYS = {key: (score, tau) for key, score, tau, *_ in _SPEC}

# what the model sees: no case id (the loop binds every call to the case being asked about)
SCHEMAS = [
    {"type": "function", "function": {
        "name": "diagnose",
        "description": "The verdict for this recording: healthy, known fault or further inspection needed; the fault class "
                       "and its location according to the dataset label if known; each check's score and the threshold "
                       "it was compared with; any caution; the two most probable classes.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "check_signal",
        "description": "Signal quality of this recording: shaft speed (Hz and rpm), whether it was steady, the recorded "
                       "warnings, and whether the speed lies inside the training and calibration speed ranges.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_evidence",
        "description": "Measured per-channel comparison with a healthy training recording at a similar speed (RMS, "
                       "kurtosis, 1x and 2x amplitudes, envelope lines), and the channels marked as changed beyond typical "
                       "healthy variation. Descriptive, not a calibrated anomaly test. Optionally one channel.",
        "parameters": {"type": "object", "properties": {"channel": {"type": "string", "enum": list(CHANNELS)}},
                       "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "lookup_guidance",
        "description": "The inspection steps for this finding from the curated guidance library, each cited by its entry "
                       "id, exactly as in the report's 'What to inspect next' section.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]
NAMES = tuple(s["function"]["name"] for s in SCHEMAS)


def leaf(value, ref: str) -> dict:
    return {"value": value, "ref": ref}


def leaves(result) -> list[tuple[str, object]]:
    """Every (ref, value) of a tool result: a dict carrying "ref" is a leaf (its "value", or a guidance item's "text")."""
    out = []
    if isinstance(result, dict):
        if "ref" in result:
            out.append((result["ref"], result.get("value", result.get("text"))))
        else:
            for v in result.values():
                out += leaves(v)
    elif isinstance(result, list):
        for v in result:
            out += leaves(v)
    return out


class Tools:
    """The tools bound to an Engine (its case store, bundle and healthy variation)."""

    def __init__(self, engine):
        self.engine = engine

    def _record(self, case_id: str):
        return self.engine.case(case_id)

    def diagnose(self, case_id: str) -> dict:
        r = self._record(case_id)
        if r is None:
            return {"error": f"no case {case_id}"}
        d = r["diagnosis"]
        out = {"verdict": leaf(d["decision"], "diagnosis.decision"),
               "display_label": leaf(d["display_label"], "diagnosis.display_label")}
        if d["decision"] == "known_fault":
            out["fault_class"] = leaf(d["fault_class"], "diagnosis.fault_class")
            out["location"] = leaf(f"{location(d['fault_class'])}, according to the dataset label",
                                   "diagnosis.location_according_to_dataset_label")
        if d["caution"]:
            out["caution"] = leaf(d["caution"], "diagnosis.caution")
        out["checks"] = [{"name": c["name"], "role": c["role"], "status": c["status"],
                          "score": leaf(c["value"], f"diagnosis.scores.{SCORE_KEYS[c['key']][0]}"),
                          "threshold": leaf(c["threshold"], f"diagnosis.thresholds.{SCORE_KEYS[c['key']][1]}"),
                          "required": c["required"], "passed": leaf(c["passed"], f"assessment.checks.{c['key']}.passed")}
                         for c in assessment(r)["checks"]]
        top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
        out["top_classes"] = [{"class": c, "probability": leaf(p, f"diagnosis.class10_probabilities.{c}")} for c, p in top]
        return out

    def check_signal(self, case_id: str) -> dict:
        r = self._record(case_id)
        if r is None:
            return {"error": f"no case {case_id}"}
        q, fr = r["signal_quality"], float(r["signal_quality"]["shaft_speed_hz"])

        def rng(key, inside):  # the inside flag's ref is its own key, so a ref read off the result is the right one
            lo, hi = q[key]
            return ([leaf(lo, f"signal_quality.{key}.0"), leaf(hi, f"signal_quality.{key}.1")],
                    leaf(bool(lo <= fr <= hi), f"signal_quality.{inside}"))
        train, in_train = rng("train_speed_range_hz", "inside_training_speed_range")
        cal, in_cal = rng("calibration_normal_speed_range_hz", "inside_calibration_speed_range")
        return {"shaft_speed_hz": leaf(fr, "signal_quality.shaft_speed_hz"),
                "shaft_speed_rpm": leaf(r["physics"]["shaft_speed_rpm"], "physics.shaft_speed_rpm"),
                "speed_stable": leaf(q["speed_stable"], "signal_quality.speed_stable"),
                "speed_range_pct": leaf(q["speed_range_pct"], "signal_quality.speed_range_pct"),
                "warnings": [leaf(w, f"signal_quality.warnings.{i}") for i, w in enumerate(q["warnings"])],
                "train_speed_range_hz": train, "inside_training_speed_range": in_train,
                "calibration_normal_speed_range_hz": cal, "inside_calibration_speed_range": in_cal}

    def get_evidence(self, case_id: str, channel: str | None = None) -> dict:
        if self._record(case_id) is None:
            return {"error": f"no case {case_id}"}
        if channel is not None and channel not in CHANNELS:
            return {"error": f"unknown channel {channel!r}; one of {', '.join(CHANNELS)}"}
        ev = self.engine.channels(case_id)
        if ev is None:
            return {"error": "the signal of this case is no longer stored, so no channel evidence is available"}
        rows = [row for row in ev["channels"] if channel is None or row["channel"] == channel]

        def metric(ch, k, m):
            p = f"evidence.{ch}.{k}"
            out = {"value": leaf(m["value"], f"{p}.value"), "healthy_reference": leaf(m["reference"], f"{p}.reference")}
            if m["comparable"]:
                out |= {"change_factor": leaf(m["factor"], f"{p}.factor"), "direction": m["direction"],
                        "beyond_typical_healthy_variation": leaf(m["exceeds"], f"{p}.exceeds")}
                if m["band"]:
                    out["typical_healthy_band"] = leaf(m["band"], f"{p}.band")
            if m["note"]:
                out["note"] = m["note"]
            return out
        return {"note": CHANGE_LABEL,
                "healthy_reference_recording": leaf(ev["reference"]["rec_id"], "evidence.reference.rec_id"),
                "healthy_reference_speed_hz": leaf(ev["reference"]["speed_hz"], "evidence.reference.speed_hz"),
                "marked_channels": [leaf(c, f"evidence.marked.{i}") for i, c in enumerate(ev["marked"])],
                "channels": [{"channel": row["channel"],
                              "metrics": {k: metric(row["channel"], k, m) for k, m in row["metrics"].items()}}
                             for row in rows]}

    def lookup_guidance(self, case_id: str) -> dict:
        r = self._record(case_id)
        if r is None:
            return {"error": f"no case {case_id}"}
        items, seen = guidance_items(r, self.engine.bundle.feature_dictionary.index), {}
        out = []
        for it in items:
            key = it.get("cite") or it["tag"].lower()
            k = seen[key] = seen.get(key, -1) + 1
            out.append({**it, "ref": f"guidance.{key}.{k}"})
        return {"items": out}

    def call(self, name: str, case_id: str, args: dict | None) -> dict:
        """One model-requested call, bound to `case_id`; unknown tools and bad arguments come back as an error dict."""
        args = dict(args or {})
        if name not in NAMES:
            return {"error": f"unknown tool {name!r}; tools: {', '.join(NAMES)}"}
        allowed = set(next(s for s in SCHEMAS if s["function"]["name"] == name)["function"]["parameters"]["properties"])
        if set(args) - allowed:
            return {"error": f"unexpected arguments {sorted(set(args) - allowed)} for {name}"}
        return getattr(self, name)(case_id, **args)


def summary(name: str, result: dict) -> str:
    """One line describing what a tool returned (code-built; shown as the action list)."""
    if "error" in result:
        return f"error: {result['error']}"
    if name == "diagnose":
        n = sum(c["passed"]["value"] for c in result["checks"])
        return (f"verdict: {result['display_label']['value']}; {n} of {len(result['checks'])} checks passed"
                + ("; caution" if "caution" in result else ""))
    if name == "check_signal":
        w = len(result["warnings"])
        return (f"shaft speed {result['shaft_speed_hz']['value']:.2f} Hz, "
                f"{'steady' if result['speed_stable']['value'] else 'not steady'}; {w} warning{'s' if w != 1 else ''}")
    if name == "get_evidence":
        marked = [m["value"] for m in result["marked_channels"]]
        return (f"{len(result['channels'])} channel(s) compared with healthy recording "
                f"{result['healthy_reference_recording']['value']}; marked: {', '.join(marked) or 'none'}")
    ids = list(dict.fromkeys(i["cite"] for i in result["items"] if i.get("cite")))
    return (f"{len(result['items'])} inspection step(s) from {', '.join(ids)}" if ids else
            f"no inspection steps: {result['items'][0]['text']}")
