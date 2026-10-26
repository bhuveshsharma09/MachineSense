"""Code-built short answers for the conversation assistant's fallback (App-9 addendum C item 4): when the model is
unavailable or its answer fails the review twice, a question about the technician status or about procedures gets a
short answer built from the stored records, instead of the long generic report fallback. No language model; every value
comes from the case's records. Other questions keep the existing fallback."""
from __future__ import annotations

import re

from backend.agent.tools import when
from backend.casenotes import PARTS, fault_status, procedure_gate

PROCEDURE_Q = re.compile(r"\b(?:procedures?|repair\w*|fix\w*|maintenance steps?)\b", re.IGNORECASE)
STATUS_Q = re.compile(r"\b(?:confirm\w*|technician\w*|status|verified|reject\w*|withdr\w*)\b", re.IGNORECASE)


def kind(question: str) -> str | None:
    if PROCEDURE_Q.search(question):
        return "procedure"
    if STATUS_Q.search(question):
        return "status"
    return None


def _label(d: dict) -> str:
    return d.get("display_label") or d["decision"]


def status_answer(record: dict, events: list[dict]) -> str:
    st = fault_status(record, events)
    t, latest, model = st["technician"], st["technician"]["latest"], _label(record["diagnosis"])
    tail = f" The model prediction is unchanged: {model} (a prediction, not a physical confirmation)."
    if t["status"] == "technician_confirmed":
        return (f"A technician recorded a confirmed fault: {latest['fault_class'].replace('_', ' ')} at the "
                f"{PARTS[latest['part']].lower()}, on {when(latest['created_at'])} by {latest['actor']['name']} (name not "
                f"verified; local demo action). This is a recorded event in this app, not independent proof." + tail)
    if latest:
        return (f"No technician confirmation is currently recorded: the latest technician event is a {latest['status']} "
                f"status, recorded on {when(latest['created_at'])} by {latest['actor']['name']} (name not verified)." + tail)
    return "No technician confirmation is recorded for this case." + tail


def procedure_answer(store, case_id: str) -> str:
    g = procedure_gate(store, case_id)
    if not g["open"]:
        return f"{g['message']} ({g['reason']}.)"
    return "Verified procedure(s) for the recorded confirmation: " + "; ".join(
        f"{p['title']} (reviewed by {p['reviewer']}; source: {p['source']})" for p in g["procedures"]) + "."


def answer(engine, case_id: str, question: str) -> tuple[str, str] | None:
    """(kind, text) for a status or procedure question, else None."""
    k = kind(question)
    if k is None:
        return None
    if k == "procedure":
        return k, procedure_answer(engine.store, case_id)
    return k, status_answer(engine.case(case_id), engine.store.status_events(case_id))
