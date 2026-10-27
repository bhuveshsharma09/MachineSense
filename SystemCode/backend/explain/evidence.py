"""Stage App-8.8 item 1: every measured comparison in a written explanation states its direction and magnitude, as the
case's measured channel evidence gives them. Applied to new explanations and, at read time, to stored ones (the stored
text is not rewritten). A sentence that refers to a measured channel (by its sensor name, in the current or the earlier
vocabulary, or by quoting its change factor) is kept only if it names one channel, quotes that channel's factor and states
its direction (higher / lower) correctly in the current vocabulary; otherwise it is replaced by the code-built statement
for each channel it refers to (`plain.change_sentence`). Directions come from the record only, never from the fault label.
"""
from __future__ import annotations

import re

from backend.explain import plain
from backend.explain.guard import DOWN, NUMBER, UP

_SIDE = {"inner": "under", "underhang": "under", "outer": "over", "overhang": "over"}
_AXIS = {"axial": "axial", "along the shaft": "axial", "radial": "radial", "tangential": "tangential", "sideways": "tangential"}
_SIDE_RE = r"(inner|underhang|outer|overhang)"
_AXIS_RE = r"(axial|along the shaft|radial|tangential|sideways)"
# "<side> bearing [sensor|housing][,·] <axis>" and "<axis> [sensor|channel] on|of the <side> bearing"
_SENSOR = [re.compile(_SIDE_RE + r"[- ]bearing(?: sensor| housing| accelerometer)?\s*[,·:]?\s*(?:sensor\s*,?\s*)?" + _AXIS_RE, re.I),
           re.compile(_AXIS_RE + r"(?: sensor| channel| direction)?(?: reading)? (?:on|of|at) the " + _SIDE_RE + r"[- ]bearing", re.I)]
_MIC = re.compile(r"\bmicrophone\b", re.I)
OLD_WORDS = re.compile(r"\b(?:inner|outer) bearing sensor\b|\bsideways\b|\balong the shaft\b|\bhealthy examples\b", re.I)
_UNIT_AFTER = re.compile(r"\s*(?:hz|rpm)\b", re.I)


def _sensors(t: str) -> list[str]:
    out = []
    for i, rx in enumerate(_SENSOR):
        for m in rx.finditer(t):
            side, axis = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
            out.append((m.start(), f"{_SIDE[side.lower()]}_{_AXIS[axis.lower()]}"))
    out += [(m.start(), "mic") for m in _MIC.finditer(t)]
    return [ch for _, ch in sorted(out)]


def _factors(t: str) -> list[tuple[str, float]]:
    """Decimal numbers in the text that are not speeds (followed by Hz or rpm)."""
    out = []
    for m in NUMBER.finditer(t):
        tok = m.group(0)
        if "." in tok and not _UNIT_AFTER.match(t, m.end()):
            out.append((tok, float(tok.replace(",", ""))))
    return out


def referenced(t: str, rows: dict) -> list[str]:
    """The measured channels a sentence refers to, by sensor name or by a quoted change factor, in order, without repeats."""
    chans = [ch for ch in _sensors(t) if ch in rows]
    for tok, x in _factors(t):
        dec = len(tok.split(".")[1])
        hits = [ch for ch, c in rows.items() if abs(round(float(c["change_factor"]), dec) - x) < 10 ** (-dec) / 2 + 1e-9]
        chans += hits[:1]
    return list(dict.fromkeys(chans))


def _states(t: str, c: dict) -> bool:
    """The sentence quotes the channel's factor (1 decimal) and its direction, without the opposite direction."""
    right, wrong = (UP, DOWN) if c["direction"] == "up" else (DOWN, UP)
    has_factor = any(abs(x - round(float(c["change_factor"]), 1)) < 0.05 + 1e-9 for _, x in _factors(t))
    return has_factor and bool(right.search(t)) and not wrong.search(t)


def check_sentence(t: str, rows: dict) -> str | None:
    """None when the sentence is kept; otherwise its replacement built from the record."""
    chans = referenced(t, rows)
    if not chans:
        return None
    if len(chans) == 1 and _states(t, rows[chans[0]]) and not OLD_WORDS.search(t):
        return None
    return " ".join(plain.change_sentence(ch, rows[ch]) for ch in chans)


def comparable_rows(channels: dict | None) -> dict:
    return {c["channel"]: c for c in (channels or {}).get("channels", [])
            if c.get("change_metric") and c.get("change_factor") is not None and c.get("direction") in ("up", "down")}


def review_text(text: dict, channels: dict | None) -> tuple[dict, list[dict]]:
    """(text with every measured comparison stated from the record, the replacements made)."""
    rows = comparable_rows(channels)
    if not rows:
        return text, []
    fixes, out = [], dict(text)
    for key in ("why_flagged", "uncertainty"):
        lines = []
        for s in text.get(key) or []:
            new = check_sentence(s, rows)
            if new is not None:
                fixes.append({"field": key, "original": s, "replacement": new})
            lines.append(new if new is not None else s)
        out[key] = lines
    return out, fixes


def reviewed(explanation: dict, channels: dict | None) -> dict:
    """An explanation as served: its text checked against the measured channel evidence (stored text unchanged)."""
    text, fixes = review_text(explanation["text"], channels)
    return {**explanation, "text": text, "corrections": fixes, "evidence_checked": bool(comparable_rows(channels))}
