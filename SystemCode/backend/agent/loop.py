"""One question about one stored case: the model may call the four read-only tools within a fixed budget, then gives a
JSON answer that the evidence review (code) checks before anyone sees it; one revision; otherwise the fixed fallback with
the report's findings and inspection steps. Each question stands alone (no chat memory). Model reasoning is never kept."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from backend.agent.prompt import AGENT_PROMPT_VERSION, SYSTEM, revision_message
from backend.agent.tools import SCHEMAS, Tools, summary
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


def fallback_sections(engine, record: dict) -> list[dict]:
    """The fixed report's sections 01 (what the readings suggest) and 04 (what to inspect next) for this case."""
    return [s for s in build_sections(record, None, engine.bundle.feature_dictionary.index) if s["n"] in ("01", "04")]


def ask(engine, case_id: str, question: str, provider, review: Review | None = None, budget: Budget = Budget(),
        review_on: bool = True, enabled: bool = True) -> dict:
    """Answer one question about a stored case. `review` defaults to the evidence review of backend.agent.review."""
    t0 = time.perf_counter()
    if review is None:
        from backend.agent.review import review
    record = engine.case(case_id)
    tools = Tools(engine)
    actions, results = [], []
    base = {"question": question, "prompt_version": AGENT_PROMPT_VERSION, "model": getattr(provider, "model", None)}

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
    messages = [{"role": "system", "content": SYSTEM},
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
            shown = obj or {"answer": msg["content"], "claims": [], "citations": []}
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
