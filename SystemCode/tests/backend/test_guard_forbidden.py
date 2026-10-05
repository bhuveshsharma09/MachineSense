"""Stage D7 item 0: guard.FORBIDDEN covers every form of each forbidden word (the rule always meant the whole family;
"confirming" slipped through in App-7 hand run 6). Shared by the explanation guard and the assistant's review."""
import pytest

from backend.explain.guard import FORBIDDEN

ADDED = ["confirming", "confirmatory", "confirmations", "percentage", "percentages", "percents", "root causes",
         "root-caused", "replaces", "replacements", "replaceable", "urgency", "urgencies", "immediateness", "imminently",
         "imminence", "remaining lifetime", "guaranteeing"]
KEPT = ["50 %", "per cent", "percent", "confirm", "confirmed", "confirms", "confirmation", "root cause", "root-cause",
        "replace", "replaced", "replacing", "replacement", "urgent", "urgently", "immediate", "immediately", "imminent",
        "remaining life", "remaining useful life", "guarantee", "guaranteed", "guarantees", "definitely", "certainly"]
ALLOWED = ["the cause is not known", "a certain channel", "the bearing is checked", "an irreplaceable record",
           "remaining recordings", "the root of the shaft", "a definite comb"]


@pytest.mark.parametrize("word", ADDED + KEPT)
def test_every_form_is_forbidden(word):
    assert FORBIDDEN.search(f"The system says: {word}."), word


@pytest.mark.parametrize("text", ALLOWED)
def test_unrelated_words_pass(text):
    assert not FORBIDDEN.search(text), text
