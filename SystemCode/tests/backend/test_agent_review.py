"""Stage App-7: the evidence review (code, not a model) rejects each kind of unsupported answer and passes a clean one."""
import json

from backend.agent.review import review
from backend.agent.tools import leaf

RESULTS = [
    {"verdict": leaf("known_fault", "diagnosis.decision"), "fault_class": leaf("underhang_outer", "diagnosis.fault_class"),
     "location": leaf("inner bearing housing (underhang bearing), according to the dataset label",
                      "diagnosis.location"),
     "checks": [{"name": "familiarity (Mahalanobis distance)", "score": leaf(12.4790, "diagnosis.scores.maha"),
                 "threshold": leaf(30.0426, "diagnosis.thresholds.tau_d"), "passed": leaf(True, "assessment.checks.maha.passed")}]},
    {"shaft_speed_hz": leaf(14.3911, "signal_quality.shaft_speed_hz"), "speed_stable": leaf(True, "signal_quality.speed_stable"),
     "warnings": [leaf("6 glitched tacho intervals", "signal_quality.warnings.0")]},
    {"items": [{"tag": "Guidance", "text": "Check that the envelope spectrum shows a repeating comb at the defect line. "
                "(source: bearing-defect-frequencies; Citation not yet checked against the full text)",
                "cite": "bearing-defect-frequencies", "hover": "t"}]},
]


def ans(answer, claims=None, citations=None):
    return json.dumps({"answer": answer, "claims": claims if claims is not None else
                       [{"text": "The verdict is a known fault.", "refs": ["diagnosis.decision"]}],
                       "citations": citations or []})


def checks(content):
    return {f["check"] for f in review(content, RESULTS)[1]}


CLEAN = ans("The recording was called a known fault, underhang outer race, located at the inner bearing housing according "
            "to the dataset label. The familiarity distance was 12.48, below its threshold of 30.04, and the shaft ran at "
            "14.39 Hz. Next, check that the envelope spectrum shows a repeating comb at the defect line.",
            [{"text": "Familiarity distance 12.48 against a threshold of 30.04.",
              "refs": ["diagnosis.scores.maha", "diagnosis.thresholds.tau_d"]},
             {"text": "Check the envelope spectrum for a repeating comb.", "refs": ["bearing-defect-frequencies"]}],
            ["bearing-defect-frequencies"])


def test_a_clean_answer_passes():
    obj, failed = review(CLEAN, RESULTS)
    assert failed == [] and obj["citations"] == ["bearing-defect-frequencies"]


def test_unparseable_or_wrong_shape_fails_schema():
    assert checks("The bearing looks fine.") == {"schema"}
    assert checks(json.dumps({"answer": "x", "claims": [{"text": "y", "refs": []}], "citations": []})) == {"schema"}


def test_an_invented_number_fails():
    assert checks(ans("The shaft ran at 15.2 Hz.")) == {"numbers"}
    assert checks(ans("The shaft ran at 864 rpm.")) == {"numbers"}  # a conversion the tools did not return
    assert checks(ans("Six tacho intervals were glitched: 6 of them.")) == set()  # a number inside a returned warning


def test_a_ref_that_does_not_exist_fails():
    assert checks(ans("The verdict is a known fault.", [{"text": "Known fault.", "refs": ["diagnosis.scores.msp"]}])) == {"refs"}


def test_claims_refer_to_inspection_steps_by_entry_id():
    """Addendum 12: no per-item guidance refs; an entry id returned by lookup_guidance is a valid claim ref, an old-style
    item ref or an entry id that was not returned is not."""
    step = "Check the envelope spectrum for a repeating comb."
    cite = ["bearing-defect-frequencies"]
    assert checks(ans("The verdict is a known fault.", [{"text": step, "refs": cite}], cite)) == set()
    assert checks(ans("The verdict is a known fault.", [{"text": step, "refs": ["guidance.bearing-defect-frequencies.0"]}],
                      cite)) == {"refs", "citations"}
    assert checks(ans("The verdict is a known fault.", [{"text": step, "refs": ["imbalance-1x"]}], cite)) == {"refs", "citations"}


def test_a_citation_not_returned_fails():
    assert checks(ans("Inspect the coupling.", citations=["misalignment-2x-axial"])) == {"citations"}


def test_an_uncited_inspection_step_fails():
    assert checks(ans("Inspect the bearing raceway for damage.")) == {"citations"}
    assert checks(ans("The verdict is a known fault.", [{"text": "Inspect the raceway.", "refs": ["diagnosis.decision"]}],
                      ["bearing-defect-frequencies"])) == {"citations"}


def test_replace_the_bearing_fails():
    assert "forbidden" in checks(ans("You should replace the bearing."))
    assert "forbidden" in checks(ans("The fault is confirmed."))


# ---- App-7 addendum 11: ranking words need a rank ref of the right value
EVIDENCE = {"rank_basis": "change beyond the healthy band",
            "largest_change": {"channel": leaf("over_radial", "evidence.largest_change.channel"),
                               "change_factor": leaf(3.3, "evidence.largest_change.change_factor"),
                               "rank": leaf(1, "evidence.largest_change.rank")},
            "channels": [{"channel": c, "rank": leaf(k + 1, f"evidence.{c}.rank"),
                          "change_factor": leaf(f, f"evidence.{c}.change_factor")}
                         for k, (c, f) in enumerate([("over_radial", 3.3), ("under_radial", 1.7), ("under_tangential", 1.8)])]}


def rchecks(answer, claims):
    return [f["check"] for f in review(json.dumps({"answer": answer, "claims": claims, "citations": []}), [EVIDENCE])[1]]


def test_the_normal_20_2752_answer_fails_the_ranking_check():
    """The recorded finding (App-7 item 10): under_tangential called the largest change; over_radial is rank 1."""
    answer = "The most significant change was observed in the under_tangential channel, with a change factor of 1.8."
    claim = "The under_tangential channel showed the most significant change, a change factor of 1.8."
    assert rchecks(answer, [{"text": claim, "refs": ["evidence.under_tangential.change_factor"]}]) == ["ranking", "ranking"]
    # citing the channel's own rank does not help: it is rank 3, not 1
    assert set(rchecks(answer, [{"text": claim, "refs": ["evidence.under_tangential.rank"]}])) == {"ranking"}


def test_a_rank_1_claim_passes_and_least_needs_the_last_rank():
    answer = "The over_radial channel is furthest outside its healthy range, the largest change of the channels (factor 3.3)."
    ok = [{"text": "over_radial has the largest change beyond its healthy band.", "refs": ["evidence.over_radial.rank"]},
          {"text": "Its change factor is 3.3.", "refs": ["evidence.largest_change.change_factor"]}]
    assert rchecks(answer, ok) == []
    assert rchecks("under_tangential shows the least change of the ranked channels.",
                   [{"text": "under_tangential is ranked lowest.", "refs": ["evidence.under_tangential.rank"]}]) == []
    assert "ranking" in rchecks("under_radial shows the least change.",
                                [{"text": "under_radial is ranked lowest.", "refs": ["evidence.under_radial.rank"]}])


def test_sensor_anomaly_worded_as_a_fault_location_fails():
    assert checks(ans("The overhang radial sensor shows the largest change, so the fault is located there according to "
                      "the dataset label.")) == {"location", "ranking"}  # "largest" with no rank ref, too
    assert checks(ans("The fault is in the inner bearing housing.")) == {"location"}  # no dataset-label wording
