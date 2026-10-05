"""The evidence review applied by code to the assistant's final answer before anyone sees it (CLAUDE.md §7 App-7 item 3).
It reuses the explanation guard (parse, number tokens and matching, forbidden phrases) and checks the answer only against
what the tools returned in this turn. Returns (answer or None, failed checks); an empty list means the answer passes."""
from __future__ import annotations

import re

from backend.agent.tools import leaves
from backend.explain.guard import FORBIDDEN, NUMBER, ExplanationRejected, _matches, parse
from backend.explain.prompt import INTEGER_CONSTANTS

CHECKS = ("schema", "numbers", "refs", "citations", "forbidden", "location", "ranking")
INSPECTION = re.compile(r"\b(?:inspect\w*|examine (?:the|it|its|each)|look for|record (?:it |the machine |the recording )?again|"
                        r"re-?record\w*|re-?run|check (?:the|for|that|whether|each|it|its)|measure (?:the|shaft|alignment)|"
                        r"escalat\w*|align (?:the|it)|rebalanc\w*|tighten\w*|lubricat\w*)\b", re.IGNORECASE)
LOCATION = re.compile(r"\b(?:locat(?:ed|ion)|the fault (?:is|lies|sits) (?:in|at|on|near))\b", re.IGNORECASE)
DATASET_LABEL = re.compile(r"according to the dataset labels?", re.IGNORECASE)
TOP_WORDS = re.compile(r"\b(?:most|largest|highest|biggest|strongest|first)\b", re.IGNORECASE)
BOTTOM_WORDS = re.compile(r"\b(?:least|smallest|lowest)\b", re.IGNORECASE)
SENSOR = re.compile(r"\b(?:sensors?|channels?|accelerometers?|microphone|axial|radial|tangential|(?:under|over)_\w+|"
                    r"reconstruction error|autoencoder error)\b", re.IGNORECASE)


def _f(check: str, detail: str) -> dict:
    return {"check": check, "detail": detail}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _strings(o) -> list[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for v in o.values() for s in _strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return []


def _rank_group(ref: str) -> str:
    """The ranking a rank ref belongs to: the channel ranking of get_evidence, or diagnose's class ranking."""
    return "classes" if ref.startswith("diagnosis.top_classes.") else "evidence"


def is_rank_ref(ref: str) -> bool:
    return ref.endswith(".rank") or ref.startswith("evidence.largest_change.")


def ranking_errors(text: str, refs: list[str], values: dict) -> list[str]:
    """App-7 addendum 11: a superlative needs a cited rank ref of the right value — rank 1 (or largest_change) for most /
    largest / highest / biggest / strongest / first; the last rank of that ranking in this turn for least / smallest /
    lowest."""
    rank_refs = [r for r in refs if is_rank_ref(r) and r in values]
    last = {}
    for r, v in values.items():
        if r.endswith(".rank") and isinstance(v, int) and not isinstance(v, bool):
            last[_rank_group(r)] = max(last.get(_rank_group(r), 0), v)
    top = any(r.startswith("evidence.largest_change.") or values[r] == 1 for r in rank_refs)
    bottom = any(r.endswith(".rank") and values[r] == last.get(_rank_group(r)) for r in rank_refs)
    errs = []
    m = TOP_WORDS.search(text)
    if m and not top:
        errs.append(f"{m.group(0)!r} needs a cited rank ref with rank 1 (or largest_change)")
    m = BOTTOM_WORDS.search(text)
    if m and not bottom:
        errs.append(f"{m.group(0)!r} needs a cited rank ref with the last rank")
    return errs


def schema_errors(obj) -> list[str]:
    if not isinstance(obj, dict) or set(obj) != {"answer", "claims", "citations"}:
        return [f"expected exactly the keys answer, claims, citations; got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}"]
    errs = []
    if not isinstance(obj["answer"], str) or not obj["answer"].strip():
        errs.append("answer must be a non-empty string")
    if not isinstance(obj["claims"], list) or not all(
            isinstance(c, dict) and set(c) == {"text", "refs"} and isinstance(c["text"], str) and c["text"].strip()
            and isinstance(c["refs"], list) and c["refs"] and all(isinstance(r, str) for r in c["refs"]) for c in obj["claims"]):
        errs.append("claims must be a list of {text, refs} with a non-empty list of ref strings")
    if not isinstance(obj["citations"], list) or not all(isinstance(c, str) for c in obj["citations"]):
        errs.append("citations must be a list of entry ids")
    return errs


def review(content: str, results: list[dict]) -> tuple[dict | None, list[dict]]:
    """The checks, each failure named: schema, numbers, refs, citations, forbidden, location, ranking."""
    try:
        obj = parse(content)
    except ExplanationRejected as e:
        return None, [_f("schema", e.reason)]
    errs = schema_errors(obj)
    if errs:
        return None, [_f("schema", e) for e in errs]
    found = [kv for r in results for kv in leaves(r)]
    refs = {ref for ref, _ in found}
    values = dict(found)
    allowed = [float(v) for _, v in found if isinstance(v, (int, float)) and not isinstance(v, bool)]
    allowed += [float(t.replace(",", "")) for s in _strings(results) for t in NUMBER.findall(s)]
    cited = {i["cite"] for r in results for i in (r.get("items") or []) if isinstance(i, dict) and i.get("cite")}
    claims = obj["claims"]
    texts = [obj["answer"], *[c["text"] for c in claims]]
    failed = []
    for t in texts:  # (2) numbers only from this turn's tool results
        for tok in NUMBER.findall(t):
            plain_int = "." not in tok and tok.replace(",", "").isdigit() and int(tok.replace(",", "")) in INTEGER_CONSTANTS
            if not (plain_int or _matches(tok, allowed)):
                failed.append(_f("numbers", f"{tok} is not a value returned by a tool in this turn"))
    for k, c in enumerate(claims):  # (3) refs exist: a returned ref, or an entry id lookup_guidance returned (addendum 12)
        for ref in c["refs"]:
            if ref not in refs and ref not in cited:
                failed.append(_f("refs", f"claim {k + 1}: ref {ref!r} was not returned by a tool in this turn"))
    for cid in obj["citations"]:  # (4) citations returned by lookup_guidance; inspection steps cited
        if cid not in cited:
            failed.append(_f("citations", f"{cid!r} is not an entry id returned by lookup_guidance in this turn"))
    if not obj["citations"] and any(INSPECTION.search(s) for s in sentences(obj["answer"])):
        failed.append(_f("citations", "the answer states an inspection step without citing a guidance entry"))
    for k, c in enumerate(claims):
        if INSPECTION.search(c["text"]) and not any(r in cited for r in c["refs"]):
            failed.append(_f("citations", f"claim {k + 1} states an inspection step without the entry id among its refs"))
    for t in texts:  # (5) forbidden phrases
        m = FORBIDDEN.search(t)
        if m:
            failed.append(_f("forbidden", f"forbidden phrase {m.group(0)!r}"))
    for t in texts:  # (6) location wording (§11 rule 5)
        for s in sentences(t):
            if LOCATION.search(s):
                if not DATASET_LABEL.search(s):
                    failed.append(_f("location", f"a fault location is stated without 'according to the dataset label': {s!r}"))
                if SENSOR.search(s):
                    failed.append(_f("location", f"sensor or channel evidence is worded as a fault location: {s!r}"))
    claim_refs = [ref for c in claims for ref in c["refs"]]  # (7) ranking words rest on a rank of the right value
    for k, c in enumerate(claims):
        failed += [_f("ranking", f"claim {k + 1}: {e}") for e in ranking_errors(c["text"], c["refs"], values)]
    for s in sentences(obj["answer"]):
        failed += [_f("ranking", f"answer sentence {s!r}: {e} among the claims") for e in ranking_errors(s, claim_refs, values)]
    return obj, failed
