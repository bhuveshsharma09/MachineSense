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


# B ---------------------------------------------------------------------------------------------------------------
VMIS = "vertical-misalignment__0.51mm__24.9856"
BEARING = "underhang__cage_fault__6g__35.0208"


def _template(client, rid):
    from backend.explain.service import template_explanation
    r = client.post("/api/diagnose", json={"rec_id": rid}).json()
    ch = client.get(f"/api/cases/{r['case_id']}/channels").json()
    return r, ch, template_explanation(r, assessment(r), ch, [])


def test_one_sensor_vocabulary_and_plain_check_names(client):
    from backend.explain.prompt import compact_view
    for rid in (BEARING, VMIS):
        r, ch, t = _template(client, rid)
        text = " ".join([t["finding"], *t["why_flagged"], *t["uncertainty"]])
        assert not any(w in text.lower() for w in ("inner bearing sensor", "outer bearing sensor", "sideways")), text
        v = compact_view(r, assessment(r), ch)
        names = " ".join(c["name"] for c in v["checks"]) + " ".join(c["sensor"] for c in v["largest_measured_changes"])
        assert not any(w in names.lower() for w in ("classifier confidence", "familiarity", "autoencoder", "speed-aware", "msp", "mahalanobis"))
        assert all(c["sensor"].startswith(("Underhang bearing", "Overhang bearing", "microphone")) for c in v["largest_measured_changes"])
        assert v["limitations"][0] == "This is a research demonstration. It is not validated on real machines."


def test_jargon_and_directionless_factors_are_refused():
    from backend.explain.guard import ExplanationRejected, check_content
    changes = [{"direction": "higher", "factor_vs_healthy_reference": 5.5}]
    ok = {"finding": "Possible fault.", "why_flagged": ["The overall vibration level is 5.5× higher than the healthy reference.",
          "The recording looks like ones the model has seen."], "uncertainty": ["x"], "next_step": "", "numbers_used": [5.5]}
    check_content(ok, [5.5], [], changes)
    for bad, why in ((["The classifier confidence passed.", "x"], "jargon"), (["The Mahalanobis distance is small.", "x"], "jargon"),
                     (["The familiarity check passed.", "x"], "jargon"),
                     (["The change factor of 5.5 on the radial sensor stands out.", "x"], "direction"),
                     (["The level is 5.5× higher than healthy examples.", "x"], "healthy examples")):
        with pytest.raises(ExplanationRejected, match=why):
            check_content({**ok, "why_flagged": bad}, [5.5], [], changes)


def test_template_states_direction_and_prefers_the_largest_increase(client):
    for rid in (VMIS, BEARING, FURTHER):
        r, ch, t = _template(client, rid)
        rows = {c["channel"]: c for c in ch["channels"]}
        pool = ch.get("exceeding") or ch.get("marked") or []
        said = [s for s in t["why_flagged"] if "healthy reference" in s]
        if not pool:
            continue
        assert said and ("× higher than the healthy reference" in said[0] or "× lower than the healthy reference" in said[0])
        assert "healthy examples" not in " ".join(t["why_flagged"])
        ups = [c for c in pool if rows[c]["direction"] == "up"]
        if ups:  # a decrease is cited only when no channel increased (item 28)
            assert "higher than the healthy reference" in said[0], (rid, said[0])
            best = max(ups, key=lambda c: rows[c]["change_factor"] / rows[c]["band"])
            assert f"{rows[best]['change_factor']:.1f}×" in said[0]
