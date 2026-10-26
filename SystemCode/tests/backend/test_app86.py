"""Stage App-8.6: fixes from the 26 Oct user test (backend side). The rule in use (item 1), the calibration-speed warning
on every verdict (4), the report's check margins (22), the trace's feature count (23), the healthy reference spectrum (26),
and the explanation's vocabulary, jargon, change direction and evidence choice (7, 8, 9, 17, 28). D7 stays frozen."""
import copy

import pytest

from backend.checks import RULE_LINE, assessment
from backend.explain import plain
from backend.report import build_sections, clean_text

FURTHER = "imbalance__10g__13.9264"   # further inspection, 13.6 Hz: below the calibrated range
OUT_HEALTHY = "normal__13.1072"       # healthy, below the calibrated range


@pytest.fixture(scope="module")
def rec(client):
    cache = {}

    def get(rid):
        if rid not in cache:
            r = client.post("/api/diagnose", json={"rec_id": rid}).json()
            cache[rid] = {"record": r, "report": client.get(f"/api/cases/{r['case_id']}/report").json()}
        return cache[rid]
    return get


# A ---------------------------------------------------------------------------------------------------------------
def test_rule_in_use_line_replaces_provisional_everywhere(rec):
    r = rec(FURTHER)
    assert assessment(r["record"])["rule_line"] == RULE_LINE == "Rule in use: B_with_caution (frozen 28 Sep 2026 before the final evaluation)"
    text = r["report"]["report"] + str(r["report"]["sections"])
    assert "provisional" not in text and RULE_LINE in text
    old = "Decision rule is a provisional demo default (B_with_caution), not a final rule selection."
    assert clean_text(old) == RULE_LINE + "." and clean_text("rule (B_with_caution) x") == "rule x"


def test_calibration_warning_on_every_verdict(rec):
    for rid in (FURTHER, OUT_HEALTHY):
        r = rec(rid)
        assert any("not calibrated at this speed" in w for w in r["record"]["signal_quality"]["warnings"]), rid
        s3 = " ".join(i["text"] for i in r["report"]["sections"][2]["items"])
        assert "Shaft speed is outside the calibrated range (18–38 Hz). The healthy checks are less reliable at this speed" in s3, rid
        assert any("outside the calibrated range" in w for w in plain.warnings(r["record"])), rid
    assert rec(FURTHER)["record"]["diagnosis"]["decision"] == "unknown"
