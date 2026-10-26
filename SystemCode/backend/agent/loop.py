"""One question about one stored case: the model may call the four read-only tools within a fixed budget, then gives a
JSON answer that the evidence review (code) checks before anyone sees it; one revision; otherwise the fixed fallback with
the report's findings and inspection steps. By default each question stands alone (no chat memory: the D7 behaviour).
App-9 conversation mode passes earlier accepted turns of the same case as context (backend.agent.memory) with its own
prompt version; the review is the same and checks only this turn's tool results. Model reasoning is never kept."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from backend.agent import memory
from backend.agent.prompt import AGENT_PROMPT_VERSION, CONVERSATION_PROMPT_VERSION, CONVERSATION_SYSTEM, SYSTEM, revision_message
from backend.agent.tools import SCHEMAS, Tools, summary
from backend.explain.guard import ExplanationRejected, parse
from backend.explain.provider import ProviderError
from backend.report import _entries, build_sections

TEMPERATURE = 0.2
FALLBACK_TEXT = ("The assistant could not give an answer supported by the case record. The fixed report's findings and "
                 "inspection steps are below.")
REVIEW_PASSED, REVIEW_REVISED, REVIEW_FALLBACK, REVIEW_OFF = "passed", "revised", "fallback", "off"

Review = Callable[[str, list], "tuple[dict | None, list[dict]]"]  # (final content, this turn's tool results) → (answer, failed)


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int = 6
    max_revisions: int = 1
    timeout_s: float = 90.0

    @classmethod
    def from_settings(cls) -> "Budget":
        from backend.settings import agent_max_revisions, agent_max_tool_calls, agent_timeout_s
        return cls(agent_max_tool_calls(), agent_max_revisions(), agent_timeout_s())


def citations_with_titles(ids) -> list[dict]:
    titles = {e.id: e.title for e in _entries()}
    return [{"id": i, "title": titles.get(i)} for i in ids]


def parsed_answer(content: str) -> dict:
    """Review off (D7 arm B, item 4): the first parseable answer as the mechanic would see it. A JSON object with a non-empty
    "answer" string shows that answer (claims and citations only if well-formed); the raw text only if nothing parses."""
    try:
        o = parse(content)
    except ExplanationRejected:
        o = None
    if not (isinstance(o, dict) and isinstance(o.get("answer"), str) and o["answer"].strip()):
        return {"answer": content, "claims": [], "citations": []}
    claims = o.get("claims")
    ok_claims = isinstance(claims, list) and all(
        isinstance(c, dict) and isinstance(c.get("text"), str) and isinstance(c.get("refs"), list)
        and all(isinstance(r, str) for r in c["refs"]) for c in claims)
    cites = o.get("citations")
    ok_cites = isinstance(cites, list) and all(isinstance(c, str) for c in cites)
    return {"answer": o["answer"], "claims": [{"text": c["text"], "refs": c["refs"]} for c in claims] if ok_claims else [],
            "citations": cites if ok_cites else []}


def fallback_sections(engine, record: dict) -> list[dict]:
    """The fixed report's sections 01 (what the readings suggest) and 04 (what to inspect next) for this case."""
    return [s for s in build_sections(record, None, engine.bundle.feature_dictionary.index) if s["n"] in ("01", "04")]


def ask(engine, case_id: str, question: str, provider, review: Review | None = None, budget: Budget = Budget(),
        review_on: bool = True, enabled: bool = True, history: list[dict] | None = None) -> dict:
    """Answer one question about a stored case. `review` defaults to the evidence review of backend.agent.review.
    `history` (App-9 conversation mode): the bounded context from memory.context_turns; None = the frozen single-question
    behaviour (D7), with its prompt and version unchanged."""
    t0 = time.perf_counter()
    if review is None:
        from backend.agent.review import review
    record = engine.case(case_id)
    tools = Tools(engine)
    actions, results = [], []
    conv = history is not None
    base = {"question": question, "prompt_version": CONVERSATION_PROMPT_VERSION if conv else AGENT_PROMPT_VERSION,
            "model": getattr(provider, "model", None)}
    if conv:
        base["context_turn_ids"] = [t["id"] for t in history]

    def done(source, obj=None, status=REVIEW_PASSED, failed=(), reason=None):
        obj = obj or {"answer": FALLBACK_TEXT, "claims": [], "citations": []}
        out = {**base, "answer": obj["answer"], "claims": obj["claims"], "citations": citations_with_titles(obj["citations"]),
               "actions": actions, "review": {"status": status, "failed": list(failed)}, "source": source,
               "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "tool_calls_used": len(actions)}
        if source == "fallback":
            out |= {"reason": reason, "report_sections": fallback_sections(engine, record)}
        return out

    if not enabled or provider is None:
        return done("fallback", status=REVIEW_FALLBACK, reason="language model disabled")
    messages = [{"role": "system", "content": CONVERSATION_SYSTEM if conv else SYSTEM}, *memory.messages(history or []),
                {"role": "user", "content": f"Question about the recording being viewed: {question}"}]
    revisions = 0
    while True:
        if time.perf_counter() - t0 > budget.timeout_s:
            return done("fallback", status=REVIEW_FALLBACK, reason="time budget exhausted")
        try:
            msg = provider.chat(messages, SCHEMAS, temperature=TEMPERATURE)
        except ProviderError as e:
            return done("fallback", status=REVIEW_FALLBACK, reason=f"language model unreachable ({e})")
        calls = msg["tool_calls"]
        if calls:
            if len(actions) + len(calls) > budget.max_tool_calls:
                return done("fallback", status=REVIEW_FALLBACK, reason="tool-call budget exhausted")
            messages.append({"role": "assistant", "content": msg["content"] or "",
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["raw_arguments"]}} for c in calls]})
            for c in calls:
                result = (tools.call(c["name"], case_id, c["arguments"]) if c["arguments"] is not None else
                          {"error": "the arguments are not valid JSON"})
                results.append(result)
                actions.append({"tool": c["name"], "arguments": c["arguments"] if c["arguments"] is not None else c["raw_arguments"],
                                "summary": summary(c["name"], result)})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
            continue
        obj, failed = review(msg["content"], results)
        if not review_on:  # D7 arm "agent without review": shown as written; what the review would reject is recorded
            shown = obj or parsed_answer(msg["content"])
            return done("agent", shown, status=REVIEW_OFF, failed=failed)
        if not failed:
            return done("agent", obj, status=REVIEW_REVISED if revisions else REVIEW_PASSED)
        if revisions >= budget.max_revisions:
            return done("fallback", status=REVIEW_FALLBACK, failed=failed, reason="evidence review failed")
        revisions += 1
        accepted = list(dict.fromkeys(i["cite"] for r in results for i in (r.get("items") or [])
                                      if isinstance(i, dict) and i.get("cite")))
        messages += [{"role": "assistant", "content": msg["content"]},
                     {"role": "user", "content": revision_message(failed, accepted)}]
