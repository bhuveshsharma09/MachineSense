"""The assistant's four read-only tools over a case stored by /api/diagnose. They read the stored record (and, for the
channel evidence, the same builder the report's channel view uses) and never run a model. Every fact an answer may rest on
is a leaf {"value": …, "ref": "<path>"} so the review can trace it; section 04 items are returned exactly as the report
has them and are referred to by their entry id (App-7 addendum 12).
A ref read from a record field is that field's path; a derived value's ref ends in its own key, so a ref inferred from
the result's shape is the real one.
Values are rounded as the report displays them (App-7 addendum 11), so the review's number matching follows what a
reader of the report would see. No tool exposes the CNN gradient map (CLAUDE.md §11 rule 11)."""
from __future__ import annotations

from backend.checks import _SPEC, _num, assessment, fmt_score
from backend.evidence import ACCEL, CHANGE_LABEL
from backend.report import guidance_items, location

CHANNELS = (*ACCEL, "mic")
SCORE_KEYS = {key: (score, tau) for key, score, tau, *_ in _SPEC}
EVIDENCE_RANK_BASIS = "change beyond the healthy band"
CLASS_RANK_BASIS = "class probability"


# ------------------------------------------------------------------ display precision (the report's, App-7 addendum 11)
def _shown(text: str):
    """A displayed number as a number; a bound such as "> 0.999999" stays the string the report shows."""
    try:
        return float(text)
    except ValueError:
        return text


def score(v: float):
    """Check scores and thresholds as the report's check lines show them (backend.checks._num)."""
    return _shown(_num(float(v)))


def probability(v: float):
    """Class probabilities as the report's class list shows them (fmt_score, 4 digits)."""
    return _shown(fmt_score(float(v)))


def fixed(v: float, digits: int) -> float:
    """Speeds (2 decimals in Hz, 0 in rpm), speed variation (2), change factors and bands (1, the report's change_text)."""
    return round(float(v), digits)


def measured(v: float) -> float:
    """Raw channel metric values, which the report does not show: the UI's rule (f3: 0 decimals from 100, 2 from 1, else 4)."""
    v = float(v)
    return round(v, 0 if abs(v) >= 100 else 2 if abs(v) >= 1 else 4)

# what the model sees: no case id (the loop binds every call to the case being asked about)
SCHEMAS = [
    {"type": "function", "function": {
        "name": "diagnose",
        "description": "The verdict for this recording: healthy, known fault or further inspection needed; the fault class "
                       "and its location according to the dataset label if known; each check's score and the threshold "
                       "it was compared with; any caution; the two most probable classes with their rank (rank 1 = most "
                       "probable).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "check_signal",
        "description": "Signal quality of this recording: shaft speed (Hz and rpm), whether it was steady, the recorded "
                       "warnings, and whether the speed lies inside the training and calibration speed ranges.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_evidence",
        "description": "Measured per-channel comparison with a healthy training recording at a similar speed (RMS, "
                       "kurtosis, 1x and 2x amplitudes, envelope lines). Channels are ranked by how far their change goes "
                       "beyond the typical healthy band (rank 1 = furthest outside its healthy range; rank_basis says "
                       "this); largest_change is rank 1. Also the channels marked as changed beyond typical healthy "
                       "variation. Descriptive, not a calibrated anomaly test. Optionally one channel (its rank is still "
                       "among all channels).",
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
                                   "diagnosis.location")
        if d["caution"]:
            out["caution"] = leaf(d["caution"], "diagnosis.caution")
        out["checks"] = [{"name": c["name"], "role": c["role"], "status": c["status"],
                          "score": leaf(score(c["value"]), f"diagnosis.scores.{SCORE_KEYS[c['key']][0]}"),
                          "threshold": leaf(score(c["threshold"]), f"diagnosis.thresholds.{SCORE_KEYS[c['key']][1]}"),
                          "required": c["required"], "passed": leaf(c["passed"], f"assessment.checks.{c['key']}.passed")}
                         for c in assessment(r)["checks"]]
        top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
        out["top_classes_rank_basis"] = leaf(CLASS_RANK_BASIS, "diagnosis.top_classes_rank_basis")
        out["top_classes"] = [{"class": c, "rank": leaf(k + 1, f"diagnosis.top_classes.{k}.rank"),
                               "probability": leaf(probability(p), f"diagnosis.class10_probabilities.{c}")}
                              for k, (c, p) in enumerate(top)]
        return out

    def check_signal(self, case_id: str) -> dict:
        r = self._record(case_id)
        if r is None:
            return {"error": f"no case {case_id}"}
        q, fr = r["signal_quality"], float(r["signal_quality"]["shaft_speed_hz"])

        def rng(key, inside):  # the inside flag's ref is its own key, so a ref read off the result is the right one
            lo, hi = q[key]
            return ([leaf(fixed(lo, 2), f"signal_quality.{key}.0"), leaf(fixed(hi, 2), f"signal_quality.{key}.1")],
                    leaf(bool(lo <= fr <= hi), f"signal_quality.{inside}"))
        train, in_train = rng("train_speed_range_hz", "inside_training_speed_range")
        cal, in_cal = rng("calibration_normal_speed_range_hz", "inside_calibration_speed_range")
        return {"shaft_speed_hz": leaf(fixed(fr, 2), "signal_quality.shaft_speed_hz"),
                "shaft_speed_rpm": leaf(fixed(r["physics"]["shaft_speed_rpm"], 0), "physics.shaft_speed_rpm"),
                "speed_stable": leaf(q["speed_stable"], "signal_quality.speed_stable"),
                "speed_range_pct": leaf(fixed(q["speed_range_pct"], 2), "signal_quality.speed_range_pct"),
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
        # ranked over ALL channels (before any filter) by change beyond the healthy band, the report's marked order
        ranked = sorted([row for row in ev["channels"] if row["beyond_band"] is not None], key=lambda row: -row["beyond_band"])
        rank = {row["channel"]: k + 1 for k, row in enumerate(ranked)}
        order = ranked + [row for row in ev["channels"] if row["channel"] not in rank]
        rows = [row for row in order if channel is None or row["channel"] == channel]

        def metric(ch, k, m):
            p = f"evidence.{ch}.{k}"
            out = {"value": leaf(measured(m["value"]), f"{p}.value"),
                   "healthy_reference": leaf(measured(m["reference"]), f"{p}.healthy_reference")}
            if m["comparable"]:
                out |= {"change_factor": leaf(fixed(m["factor"], 1), f"{p}.change_factor"), "direction": m["direction"],
                        "beyond_typical_healthy_variation": leaf(m["exceeds"], f"{p}.beyond_typical_healthy_variation")}
                if m["band"]:
                    out["typical_healthy_band"] = leaf(fixed(m["band"], 1), f"{p}.typical_healthy_band")
            if m["note"]:
                out["note"] = m["note"]
            return out

        def channel_row(row):
            ch = row["channel"]
            out = {"channel": ch}
            if ch in rank:
                out |= {"rank": leaf(rank[ch], f"evidence.{ch}.rank"), "change_metric": row["change_metric"],
                        "change_factor": leaf(fixed(row["change_factor"], 1), f"evidence.{ch}.change_factor"),
                        "typical_healthy_band": leaf(fixed(row["band"], 1), f"evidence.{ch}.typical_healthy_band")}
            else:
                out["rank"] = "not ranked: no metric comparable with the healthy band"
            out["metrics"] = {k: metric(ch, k, m) for k, m in row["metrics"].items()}
            return out
        out = {"note": CHANGE_LABEL, "rank_basis": leaf(EVIDENCE_RANK_BASIS, "evidence.rank_basis"),
               "healthy_reference_recording": leaf(ev["reference"]["rec_id"], "evidence.reference.rec_id"),
               "healthy_reference_speed_hz": leaf(fixed(ev["reference"]["speed_hz"], 2), "evidence.reference.speed_hz"),
               "marked_channels": [leaf(c, f"evidence.marked.{i}") for i, c in enumerate(ev["marked"])]}
        if ranked:
            top = ranked[0]
            p = "evidence.largest_change"
            out["largest_change"] = {"rank_basis": leaf(EVIDENCE_RANK_BASIS, f"{p}.rank_basis"),
                                     "channel": leaf(top["channel"], f"{p}.channel"),
                                     "metric": leaf(top["change_metric"], f"{p}.metric"),
                                     "change_factor": leaf(fixed(top["change_factor"], 1), f"{p}.change_factor"),
                                     "typical_healthy_band": leaf(fixed(top["band"], 1), f"{p}.typical_healthy_band"),
                                     "rank": leaf(1, f"{p}.rank")}
        out["channels"] = [channel_row(row) for row in rows]
        return out

    def lookup_guidance(self, case_id: str) -> dict:
        r = self._record(case_id)
        if r is None:
            return {"error": f"no case {case_id}"}
        return {"items": guidance_items(r, self.engine.bundle.feature_dictionary.index)}

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
        top = result.get("largest_change")
        return (f"{len(result['channels'])} channel(s) compared with healthy recording "
                f"{result['healthy_reference_recording']['value']}; marked: {', '.join(marked) or 'none'}"
                + (f"; furthest beyond its healthy band: {top['channel']['value']}" if top else ""))
    ids = list(dict.fromkeys(i["cite"] for i in result["items"] if i.get("cite")))
    return (f"{len(result['items'])} inspection step(s) from {', '.join(ids)}" if ids else
            f"no inspection steps: {result['items'][0]['text']}")
