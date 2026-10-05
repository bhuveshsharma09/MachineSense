"""The assistant's one system prompt and its answer schema, versioned like the explanation prompt: the version is a hash
of the prompt, the tool schemas and the answer schema, stored on every answer."""
from __future__ import annotations

import hashlib
import json

from backend.agent.tools import SCHEMAS

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "claims", "citations"],
    "properties": {
        "answer": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "refs"],
                                              "properties": {"text": {"type": "string"},
                                                             "refs": {"type": "array", "minItems": 1, "items": {"type": "string"}}}}},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM = """You answer one question from a maintenance mechanic about one machine recording that MachineSense has already
diagnosed. The tools return facts about this recording; they are your only source of facts. Call only the tools you need.
Rules:
- Every number you write must come from a tool result. You may round it, but never compute a new number (no unit
  conversions, no differences, no ratios you work out yourself). Never use percentages or the % sign.
- Do not change or re-judge the verdict.
- Never say a fault is confirmed, never name a root cause, never recommend replacing a part, never express urgency.
- A fault location may only be given as the location "according to the dataset label" returned by diagnose. Channel
  changes from get_evidence are measured differences from a healthy recording, not the location of a fault: never say a
  sensor or channel shows where the fault is.
- Inspection steps come only from lookup_guidance. Repeat them in your own words if you like, and cite their entry ids.
  Never invent an inspection step.
- If the tools do not answer the question, say so plainly.
When you have the facts, reply with only one JSON object and no other text:
{"answer": "<2 to 5 plain sentences>", "claims": [{"text": "<one fact you used>", "refs": ["<the ref of each tool value it rests on>"]}], "citations": ["<entry id>"]}
Every claim lists the "ref" strings of the tool values it rests on. "citations" lists the entry ids of the inspection
steps you mention (the "cite" of lookup_guidance items), or is empty."""

# the revision message lists the entry ids lookup_guidance returned in this turn (App-7 addendum 10, Bhuvesh, 5 Oct)
REVISION = ("The evidence review rejected your answer:\n{failed}\n{ids}\nReply again with one corrected JSON object that "
            "follows the rules. You may call tools if you need facts.")
IDS_ACCEPTED = "Accepted entry ids for \"citations\" (the \"cite\" values lookup_guidance returned): {ids}."
IDS_NONE = "No entry ids were returned in this turn; call lookup_guidance before citing any inspection step."

AGENT_PROMPT_VERSION = hashlib.sha256((SYSTEM + json.dumps(SCHEMAS, sort_keys=True) + json.dumps(ANSWER_SCHEMA, sort_keys=True)
                                       + REVISION + IDS_ACCEPTED + IDS_NONE).encode()).hexdigest()[:16]


def revision_message(failed: list[dict], accepted_ids=()) -> str:
    ids = IDS_ACCEPTED.format(ids=", ".join(accepted_ids)) if accepted_ids else IDS_NONE
    return REVISION.format(failed="\n".join(f"- {f['check']}: {f['detail']}" for f in failed), ids=ids)
