"""Stage App-8.8 item 1: every measured comparison in a written explanation states its direction and magnitude from the
case's channel evidence, for stored explanations (checked when read; the stored text unchanged) and new ones. The sentences
below are verbatim from explanations stored on 27 Oct 2026 under prompt f154953912c969cc for the representative cases."""
import copy
import json

import httpx
import pytest

from backend.explain.evidence import check_sentence, review_text, reviewed


def row(ch, metric, direction, factor, band=1.4):
    return {"channel": ch, "change_metric": metric, "direction": direction, "change_factor": factor, "band": band, "exceeds": True}


# known fault c102b558… (horizontal misalignment): over_axial 2× is 2.85× LOWER; under_tangential RMS 1.81× higher
KNOWN = {"channels": [row("over_axial", "order_2x", "down", 2.85), row("under_tangential", "rms", "up", 1.81),
                      row("under_radial", "order_2x", "down", 1.65)]}
# healthy with caution 58980a3f…: over_radial BPFO line 3.33× lower; under_radial BPFI line 1.70× higher
CAUTION = {"channels": [row("over_radial", "env_bpfo1", "down", 3.33), row("under_radial", "env_bpfi1", "up", 1.70),
                        row("under_tangential", "order_1x", "down", 1.82)]}
# further inspection d0c5dae4…: over_axial RMS 5.69× lower; over_tangential 1× 6.2× higher
FURTHER = {"channels": [row("over_axial", "rms", "down", 5.69), row("over_tangential", "order_1x", "up", 6.2),
                        row("under_tangential", "order_1x", "up", 2.91)]}


def rows(ch):
    return {c["channel"]: c for c in ch["channels"]}


@pytest.mark.parametrize("sentence,channels,want", [
    ("Vibration twice per rotation (2×) was detected on the outer bearing sensor, along the shaft.", KNOWN,
     ["Overhang bearing, axial", "2.9× lower than the healthy reference"]),
    ("The overall vibration level (RMS) from the inner bearing sensor, sideways, was higher than in healthy examples.", KNOWN,
     ["Underhang bearing, tangential", "1.8× higher than the healthy reference"]),
    ("There were impacts detected at the outer and inner bearing race rates with change factors of 3.3 and 1.7.", CAUTION,
     ["Overhang bearing, radial sensor are 3.3× lower", "Underhang bearing, radial sensor are 1.7× higher"]),
    ("The overall vibration level (RMS) on the outer bearing sensor, along the shaft, showed a change factor of 5.7.", FURTHER,
     ["Overhang bearing, axial", "5.7× lower than the healthy reference"]),
    ("The vibration once per rotation (1×) on the outer bearing sensor, sideways, showed a change factor of 6.2.", FURTHER,
     ["Overhang bearing, tangential", "6.2× higher than the healthy reference"]),
])
def test_stored_sentences_without_direction_are_replaced_from_the_record(sentence, channels, want):
    new = check_sentence(sentence, rows(channels))
    assert new is not None and all(w in new for w in want), new


def test_correct_sentences_and_unrelated_ones_are_kept():
    ok = "The vibration twice per rotation (2×) on the Overhang bearing, axial sensor is 2.9× lower than the healthy reference."
    assert check_sentence(ok, rows(KNOWN)) is None
    assert check_sentence("The speed was 19.8 Hz or 1189 RPM.", rows(CAUTION)) is None
    assert check_sentence("The vibration pattern matches the imbalance examples the model learned from.", rows(KNOWN)) is None
    # a wrong direction is never kept, whatever the fault label suggests
    wrong = "The vibration twice per rotation (2×) on the Overhang bearing, axial sensor is 2.9× higher than the healthy reference."
    assert "2.9× lower" in check_sentence(wrong, rows(KNOWN))


def test_review_text_lists_replacements_and_needs_channel_evidence():
    text = {"finding": "x", "why_flagged": ["Vibration twice per rotation (2×) was detected on the outer bearing sensor, along the shaft.",
                                           "The pattern matches."], "uncertainty": ["u"], "next_step": "", "numbers_used": []}
    out, fixes = review_text(text, KNOWN)
    assert len(fixes) == 1 and fixes[0]["field"] == "why_flagged" and "lower" in out["why_flagged"][0]
    assert out["why_flagged"][1] == "The pattern matches." and text["why_flagged"][0].endswith("along the shaft.")  # input unchanged
    assert review_text(text, None) == (text, []) and reviewed({"text": text}, None)["evidence_checked"] is False


class Model:
    def __init__(self):
        self.reply = None

    def __call__(self, request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": self.reply}}]})


@pytest.fixture(scope="module")
def model():
    return Model()


@pytest.fixture(scope="module")
def app(bundle_path, guard_path, tmp_path_factory, model):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    with TestClient(create_app(bundle_path, cases_path=tmp_path_factory.mktemp("app88") / "cases.db", guard_path=guard_path,
                               llm_transport=httpx.MockTransport(model))) as c:
        yield c


def test_a_stored_explanation_is_served_corrected_and_left_unchanged_in_the_store(app):
    r = app.post("/api/diagnose", json={"rec_id": "imbalance__10g__13.9264"}).json()
    ch = app.get(f"/api/cases/{r['case_id']}/channels").json()
    c = next(x for x in ch["channels"] if x["channel"] in ch["exceeding"])
    side, axis = c["channel"].split("_")
    old_sensor = f"{'inner' if side == 'under' else 'outer'} bearing sensor, {'sideways' if axis == 'tangential' else 'along the shaft' if axis == 'axial' else 'radial'}"
    stored = {"case_id": r["case_id"], "source": "llm", "badge": "AI-written explanation of the evidence", "prompt_version": "f154953912c969cc",
              "text": {"finding": "Further inspection is needed.", "why_flagged": [f"A change was detected on the {old_sensor}.", "The pattern is unfamiliar."],
                       "uncertainty": ["One check is close to its limit."], "next_step": "", "numbers_used": []},
              "fallback_reason": None, "attempts": 1, "latency_ms": 1.0, "provider": "x", "model": "y", "created_at": "2026-10-27T00:00:00+00:00"}
    eng = app.app.state.engine
    eng.store.put_explanation(r["case_id"], copy.deepcopy(stored))
    got = app.get(f"/api/cases/{r['case_id']}/explanation").json()
    want = f"{float(c['change_factor']):.1f}× {'higher' if c['direction'] == 'up' else 'lower'} than the healthy reference"
    assert want in got["text"]["why_flagged"][0] and len(got["corrections"]) == 1 and got["evidence_checked"] is True
    assert eng.store.get_explanation(r["case_id"])["text"] == stored["text"]  # the stored text is not rewritten


def test_a_new_ai_explanation_is_corrected_before_it_is_shown(app, model):
    r = app.post("/api/diagnose", json={"rec_id": "imbalance__10g__13.9264"}).json()
    ch = app.get(f"/api/cases/{r['case_id']}/channels").json()
    c = next(x for x in ch["channels"] if x["channel"] in ch["exceeding"])
    side, axis = c["channel"].split("_")
    model.reply = json.dumps({"finding": "Further inspection is needed.",
                              "why_flagged": [f"A change shows on the {'Underhang' if side == 'under' else 'Overhang'} bearing, {axis} sensor.",
                                              "The pattern is unfamiliar."],
                              "uncertainty": ["The healthy baseline comes from a small number of healthy recordings."],
                              "next_step": "", "numbers_used": []})
    e = app.post(f"/api/cases/{r['case_id']}/explanation").json()
    assert e["source"] == "llm" and len(e["corrections"]) == 1
    assert ("higher than the healthy reference" if c["direction"] == "up" else "lower than the healthy reference") in e["text"]["why_flagged"][0]
