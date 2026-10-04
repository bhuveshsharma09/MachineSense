"""Explain one case: the local model if enabled and reachable and its text passes every check, else the fixed template.
The result carries its source, badge, provider, model, prompt version, latency, attempts and fallback reason."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from backend.explain import plain
from backend.explain.guard import ExplanationRejected, check_content, check_schema, parse
from backend.explain.prompt import PROMPT_VERSION, SCHEMA, allowed_numbers, compact_view, messages, raw_numbers
from backend.explain.provider import ProviderError

BADGE_LLM = "AI-written explanation of the evidence"
BADGE_TEMPLATE = "template explanation"
TEMPERATURE = 0.2


def template_explanation(record: dict, assessment: dict, channels: dict | None, guidance: list[dict]) -> dict:
    """The fixed, deterministic explanation (no language model), in the same schema; plain words, no numbers except in
    `next_step`, which is report section 04's cited steps (`guidance`) as given."""
    d = record["diagnosis"]
    why = []
    if d["decision"] == "known_fault":
        why.append(f"The vibration pattern matches the {plain.FAULT_PLAIN.get(d['fault_class'], d['fault_class'])} examples "
                   "the model learned from.")
    elif d["decision"] == "unknown":
        top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
        why.append("The pattern does not closely match any fault type the model knows well enough to name it.")
        why.append("The most similar known types are " + " and ".join(plain.FAULT_PLAIN.get(c, c) for c, _ in top) + ".")
    else:
        why.append("The vibration pattern matches the healthy examples the model learned from.")
    if channels:
        rows = {c["channel"]: c for c in channels["channels"]}
        for ch in (channels.get("marked") or [])[:1]:
            c = rows[ch]
            why.append(f"Compared with a healthy example, the {plain.MEASURE_PLAIN.get(c['change_metric'], c['change_metric'])} on the "
                       f"{plain.SENSOR_PLAIN.get(ch, ch)} is {'higher' if c['direction'] == 'up' else 'lower'} "
                       "(different from a healthy example, not a proven fault).")
    if len(why) < 2:
        why.append(plain.supporting(record, assessment))
    unc = list(plain.warnings(record))[:2]
    if assessment.get("narrow_notes"):
        unc.append("One of the checks is close to its limit, so a repeat reading could change the result.")
    unc.append("The healthy baseline comes from a small number of healthy recordings on one test rig.")
    return {"finding": plain.finding(record), "why_flagged": why[:3], "uncertainty": unc[:3],
            "next_step": plain.next_step(guidance), "numbers_used": []}


def _log(path, entry: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def explain_case(record: dict, assessment: dict, channels: dict | None, user_fields: dict | None, provider,
                 enabled: bool = True, log_path=None, *, guidance: list[dict]) -> dict:
    """`guidance` = report section 04's items; the next step is filled from them by code for the model's text and the
    template alike (the model is asked for an empty next_step and never writes inspection steps)."""
    t0 = time.perf_counter()
    base = {"case_id": record["case_id"], "prompt_version": PROMPT_VERSION,
            "provider": getattr(provider, "name", None), "model": getattr(provider, "model", None),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    template = template_explanation(record, assessment, channels, guidance)
    view = compact_view(record, assessment, channels, user_fields)
    attempts, reason, text = 0, None, None
    if not enabled or provider is None:
        reason = "language model disabled"
    else:
        while attempts < 2:
            attempts += 1
            try:
                content, _ = provider.complete(messages(view), temperature=TEMPERATURE, schema=SCHEMA)
                obj = parse(content)
                check_schema(obj)
                check_content(obj, allowed_numbers(view), raw_numbers(record, assessment), view["largest_measured_changes"])
                text, reason = {**obj, "next_step": template["next_step"]}, None
                break
            except ProviderError as e:
                reason = f"language model unreachable ({e})"
                break
            except ExplanationRejected as e:
                reason = e.reason
                if not e.retryable:
                    break
    out = {**base, "source": "llm" if text else "template", "badge": BADGE_LLM if text else BADGE_TEMPLATE,
           "text": text or template, "fallback_reason": reason, "attempts": attempts,
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    _log(log_path, {k: out[k] for k in ("case_id", "created_at", "prompt_version", "provider", "model", "source",
                                         "attempts", "latency_ms", "fallback_reason")})
    return out
