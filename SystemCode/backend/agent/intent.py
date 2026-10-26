"""Conversation-only routing and answer-fit checks for verdict and physical-cause questions.

These checks complement provenance review: a true statement about the verdict is not necessarily an
answer to why it was given. They never modify the frozen single-question D7 path.
"""
from __future__ import annotations

import re

CAUSE = re.compile(r"\b(?:caus(?:e|ed|es|ing)|reason (?:for|behind) the fault|happen(?:ed)?|origin)\b", re.I)
VERDICT = re.compile(r"\b(?:verdict|diagnos(?:is|ed|e)|flagged?|fault|faulty|healthy|result|prediction|classif(?:y|ied|ication)|imbalance|misalignment|bearing)\b", re.I)
WHY = re.compile(r"\b(?:why|how come|what (?:evidence|led|made)|explain|reason|basis|evidence|support(?:s|ed)?)\b", re.I)
LIMIT = re.compile(
    r"\b(?:cannot|can't|does not|doesn't|not enough|insufficient|unknown|undetermined|not possible)\b"
    r".{0,110}\b(?:caus\w*|determin\w*|establish\w*|identif\w*|prov\w*|confirm\w*)\b"
    r"|\b(?:caus\w*|determin\w*|establish\w*|identif\w*|prov\w*|confirm\w*)\b"
    r".{0,80}\b(?:cannot|can't|not|unknown|undetermined)\b", re.I)
DECISION_WORD = re.compile(r"\b(?:checks?|threshold|top class|classifier|score|confidence|familiarity|decision rule|ranked)\b", re.I)
MEASURED_WORD = re.compile(r"\b(?:sensors?|channels?|measurement|vibration|microphone|rms|amplitude|healthy (?:band|recording|comparison)|envelope)\b", re.I)


def classify(question: str) -> str | None:
    """Recognise only explicit cause or verdict-explanation requests; leave other follow-ups to the model."""
    q = " ".join(question.casefold().split())
    if CAUSE.search(q) and (VERDICT.search(q) or re.search(r"\b(?:this|it|root cause)\b", q)):
        return "physical_cause"
    if WHY.search(q) and VERDICT.search(q):
        return "verdict_reason"
    return None


def required_tools(kind: str | None) -> tuple[str, ...]:
    if kind == "verdict_reason":
        return ("diagnose", "get_evidence")
    if kind == "physical_cause":
        return ("diagnose",)
    return ()


def hint(kind: str) -> str:
    if kind == "physical_cause":
        return ("Answer the physical-cause question directly: the recording and model prediction do not establish what "
                "physically caused the condition. State that limit in the first sentence. You may report the model "
                "prediction as a prediction, but do not repeat it as if it were the cause.")
    return ("Explain why the decision rule gave this verdict using diagnose's checks or ranking. Separately name a "
            "case-specific measured channel comparison from get_evidence when available. This comparison describes "
            "the recording; it did not determine the model's class or locate a physical fault. Do not merely restate "
            "the fault label and dataset-mapped part.")


def fit_failures(kind: str | None, obj: dict | None, results: list[dict]) -> list[dict]:
    """Reject a supported but non-responsive answer in conversation mode after the normal evidence review."""
    if kind is None or obj is None:
        return []
    answer = obj["answer"]
    refs = [r for c in obj["claims"] for r in c["refs"]]
    failed = []
    if kind == "physical_cause":
        first = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
        if not LIMIT.search(first):
            failed.append({"check": "answer_fit", "detail": "Start by saying that this recording cannot establish a physical cause; a model class or mapped part is not a cause."})
    elif kind == "verdict_reason":
        if not any(r.startswith(("diagnosis.", "assessment.")) for r in refs) or not DECISION_WORD.search(answer):
            failed.append({"check": "answer_fit", "detail": "Explain the decision rule or classifier checks, with a diagnose ref; repeating the fault label is insufficient."})
        evidence_available = any("largest_change" in r and "error" not in r for r in results)
        if evidence_available and (not any(r.startswith("evidence.") for r in refs) or not MEASURED_WORD.search(answer)):
            failed.append({"check": "answer_fit", "detail": "Include a case-specific measured channel comparison from get_evidence, with an evidence ref, separately from the model prediction."})
        if not evidence_available and not re.search(r"\b(?:unavailable|cannot|can't|not available|not stored)\b", answer, re.I):
            failed.append({"check": "answer_fit", "detail": "The channel evidence is unavailable; say that plainly instead of implying it was checked."})
    return failed


def safe_answer(kind: str, results: list[dict]) -> dict:
    """A short, code-built answer grounded in the prefetched case tools if the model cannot answer fitfully."""
    diagnosis = next((r for r in results if "verdict" in r), None)
    evidence = next((r for r in results if "healthy_reference_recording" in r), None)
    if diagnosis is None:
        return {"answer": "The case record is unavailable, so I cannot explain this result.", "claims": [], "citations": []}
    decision = diagnosis["verdict"]["value"]
    label = diagnosis["display_label"]["value"].replace("_", " ")
    if kind == "physical_cause":
        text = ("This recording cannot establish what physically caused a fault. " if decision != "healthy" else
                "No fault was detected in this recording, so it cannot establish a fault cause. ")
        text += f"The model's verdict is {label}; that prediction is not a physical cause."
        return {"answer": text, "claims": [{"text": f"The model verdict is {label}.",
                                              "refs": ["diagnosis.display_label"]}], "citations": []}
    checks = [c for c in diagnosis["checks"] if c["role"] == "classifier"]
    if decision == "unknown":
        opening = "The model requested further inspection because a classifier check did not pass."
    elif decision == "healthy":
        opening = "The model reported healthy because its classifier checks passed and the top class was normal."
    else:
        fault = diagnosis["fault_class"]["value"].replace("_", " ")
        opening = f"The model predicted {fault} because its classifier checks passed and that was its top class."
    refs = ["diagnosis.decision", *(c["passed"]["ref"] for c in checks)]
    claims = [{"text": opening, "refs": refs}]
    top = evidence.get("largest_change") if evidence else None
    if top:
        channel = top["channel"]["value"].replace("under_", "underhang ").replace("over_", "overhang ").replace("_", " ")
        measured = (f"Separately, the {channel} channel ranked first for change relative to its healthy band. "
                    "That comparison does not identify a physical fault.")
        claims.append({"text": f"The {channel} channel ranked first for change relative to its healthy band.",
                       "refs": [top["channel"]["ref"], top["rank"]["ref"]]})
    else:
        measured = "A measured channel comparison is unavailable for this case."
    return {"answer": opening + " " + measured, "claims": claims, "citations": []}
