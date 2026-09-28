"""Checks applied by application code to generated text (never trusted to the prompt): schema, the fixed next step,
numbers (only the record's mechanic numbers; raw scores refused), forbidden phrases and percentages, and the direction
(higher / lower) of any measured change whose factor is quoted."""
from __future__ import annotations

import json
import re

from backend.explain import plain
from backend.explain.prompt import INTEGER_CONSTANTS

FORBIDDEN = re.compile(r"%|\bper ?cent\b|\bconfirm(?:ed|s|ation)?\b|\broot[- ]cause\b|\breplac(?:e|ed|ing|ement)\b|"
                       r"\burgent(?:ly)?\b|\bimmediate(?:ly)?\b|\bimminent\b|\bremaining (?:useful )?life\b|"
                       r"\bguarantee(?:d|s)?\b|\bdefinitely\b|\bcertainly\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
UP = re.compile(r"\b(?:higher|increased?|increases|greater|more|above|larger|raised|elevated)\b", re.IGNORECASE)
DOWN = re.compile(r"\b(?:lower|decreased?|decreases|less|below|smaller|reduced|weaker)\b", re.IGNORECASE)


class ExplanationRejected(ValueError):
    def __init__(self, reason: str, retryable: bool):
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable


def parse(content: str) -> dict:
    t = content.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise ExplanationRejected(f"not valid JSON ({e.msg})", retryable=True) from e
    if not isinstance(obj, dict):
        raise ExplanationRejected("not a JSON object", retryable=True)
    return obj


def check_schema(obj: dict) -> None:
    keys = {"finding", "why_flagged", "uncertainty", "next_step", "numbers_used"}
    if set(obj) != keys:
        raise ExplanationRejected(f"schema: keys {sorted(obj)} != {sorted(keys)}", retryable=True)
    ok = (isinstance(obj["finding"], str) and obj["finding"].strip() and len(obj["finding"].split()) <= 25
          and isinstance(obj["why_flagged"], list) and 2 <= len(obj["why_flagged"]) <= 3
          and all(isinstance(x, str) and x.strip() for x in obj["why_flagged"])
          and isinstance(obj["uncertainty"], list) and 1 <= len(obj["uncertainty"]) <= 3
          and all(isinstance(x, str) and x.strip() for x in obj["uncertainty"])
          and isinstance(obj["next_step"], str) and isinstance(obj["numbers_used"], list))
    if not ok:
        raise ExplanationRejected("schema: wrong types or lengths (finding ≤ 25 words; 2–3 why; 1–3 uncertainty)", retryable=True)
    if obj["next_step"].strip() != plain.GUIDANCE_TEXT:
        raise ExplanationRejected("schema: next_step is not the guidance-not-connected text", retryable=True)


def _matches(token: str, values: list[float]) -> bool:
    x = float(token.replace(",", ""))
    dec = len(token.split(".")[1]) if "." in token else 0
    return any(abs(round(v, dec) - x) < 10 ** (-dec) / 2 + 1e-9 for v in values)


def check_directions(texts: list[str], changes: list[dict]) -> None:
    """A sentence that quotes a change factor must not state the opposite direction (Part D: Gemma once wrote "5.7 times
    higher" for a level that was 5.7 times lower). Only sentences whose quoted factors all share one direction are judged."""
    for t in texts:
        dirs = {c["direction"] for c in changes for tok in NUMBER.findall(t)
                if "." in tok and _matches(tok, [c["factor_vs_healthy_example"]])}
        if len(dirs) != 1:
            continue
        d = dirs.pop()
        right, wrong = (UP, DOWN) if d == "higher" else (DOWN, UP)
        if wrong.search(t) and not right.search(t):
            raise ExplanationRejected(f"direction contradicts the record (should be {d}): {t[:80]!r}", retryable=False)


def check_content(obj: dict, allowed: list[float], raw: list[float], changes: list[dict] | None = None) -> None:
    texts = [obj["finding"], *obj["why_flagged"], *obj["uncertainty"]]
    for t in texts:
        m = FORBIDDEN.search(t)
        if m:
            raise ExplanationRejected(f"forbidden phrase: {m.group(0)!r}", retryable=False)
    tokens = [n for t in texts for n in NUMBER.findall(t)] + [str(n) for n in obj["numbers_used"]]
    for tok in tokens:
        try:
            float(str(tok).replace(",", ""))
        except ValueError:
            raise ExplanationRejected(f"numbers_used contains a non-number {tok!r}", retryable=False)
        tok = str(tok)
        if _matches(tok, allowed) or ("." not in tok and tok.replace(",", "").isdigit() and int(tok.replace(",", "")) in INTEGER_CONSTANTS):
            continue
        if _matches(tok, raw):
            raise ExplanationRejected(f"raw model score quoted: {tok}", retryable=False)
        raise ExplanationRejected(f"number not in the evidence record: {tok}", retryable=False)
    check_directions(texts, changes or [])
