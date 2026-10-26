"""Small, exact conversational replies for App-9's opt-in conversation mode.

These contain no case facts and never enter the evidence-review or model path. A message that
contains an investigation question is intentionally left for the normal tool-using assistant.
"""
from __future__ import annotations

import re


def reply(question: str) -> str | None:
    q = re.sub(r"[\s.!?]+", " ", question.casefold()).strip()
    if q in {"hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening", "yo"}:
        return "Hello! I can help you understand this recording, its evidence, and what to inspect next. What would you like to know?"
    if q in {"thanks", "thank you", "thank you so much", "thx", "ty"}:
        return "You're welcome. Ask me if you'd like to look at any part of this recording more closely."
    if q in {"bye", "goodbye", "see you", "see you later"}:
        return "Goodbye. Your saved questions about this recording will still be here later."
    if q in {"how are you", "how's it going", "how is it going"}:
        return "I'm ready to help with this recording. What would you like to check?"
    if q in {"help", "what can you do", "who are you"}:
        return "I can explain the model's result, point to measured evidence, summarize technician notes, and suggest inspection checks. What would you like to explore?"
    return None
