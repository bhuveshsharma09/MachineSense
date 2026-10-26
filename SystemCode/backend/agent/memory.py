"""Case-scoped conversational context for the assistant (Stage App-9 slice 1). Earlier turns are CONTEXT, never evidence:
only accepted answers (review passed or revised, source agent) of the same case and the same conversation are used, as
the question and the visible answer text; no claims, tool results, rejected drafts, reasoning or other cases. Bounded by a
number of turns and of characters (the most recent first; older turns are dropped). The review still checks the new
answer against this turn's tool results only, so a fact from an earlier answer has to be re-fetched."""
from __future__ import annotations

MAX_TURNS = 4
MAX_CHARS = 2400
ANSWER_CHARS = 600   # one earlier answer is cut to this many characters (a cut is marked with "…")
ACCEPTED = ("passed", "revised")


def accepted(turn: dict) -> bool:
    return turn.get("source") == "agent" and (turn.get("review") or {}).get("status") in ACCEPTED


def context_turns(turns: list[dict], max_turns: int = MAX_TURNS, max_chars: int = MAX_CHARS) -> list[dict]:
    """The bounded context: [{id, question, answer}] oldest first, from the most recent accepted turns that fit both
    bounds. `turns` must already be one case's one conversation (CaseStore.turns)."""
    out, used = [], 0
    for t in reversed(turns):
        if not accepted(t):
            continue
        ans = str(t.get("answer", ""))
        ans = ans if len(ans) <= ANSWER_CHARS else ans[:ANSWER_CHARS].rstrip() + "…"
        q = str(t.get("question", ""))
        size = len(q) + len(ans)
        if len(out) >= max_turns or used + size > max_chars:
            break
        out.append({"id": t["id"], "question": q, "answer": ans})
        used += size
    return list(reversed(out))


def messages(context: list[dict]) -> list[dict]:
    """The earlier turns as chat messages: the user's question, then the assistant's visible answer as plain text."""
    out = []
    for t in context:
        out += [{"role": "user", "content": f"(earlier question) {t['question']}"},
                {"role": "assistant", "content": f"(earlier answer, context only, not evidence) {t['answer']}"}]
    return out
