"""Explanation layer (Stage App-3): a plain-language explanation of an evidence record, written by a local language
model through a provider interface, checked by application code, and replaced by the fixed template whenever the model is
unreachable or its text fails a check. The verdict, caution and warnings are never taken from generated text.
CLAUDE.md §11: numbers come only from tested tools; the app fully works with no model reachable."""
from backend.explain.service import explain_case, template_explanation  # noqa: F401
