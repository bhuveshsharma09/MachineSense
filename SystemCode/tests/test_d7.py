"""Stage D7: case selection is reproducible and follows the strata; the runner refuses a dirty tree or another prompt
version; the scoring sheet gives no arm away outside the answer text; the key round-trips; the analysis gives the expected
numbers on a small synthetic scores file."""
import importlib.util
import json
import re

import pandas as pd
import pytest

from ml.paths import SYSTEM_CODE_DIR
from ml.pilot import STRATA, InsufficientStratum, mcnemar_exact, paired, per_arm, scored_table, select_cases, wilson

D7 = SYSTEM_CODE_DIR / "results" / "d7"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SYSTEM_CODE_DIR / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pool(n_unknown=6, n_disagree=6, n_clean=8, n_healthy=5, n_warn=5):
    rows = []
    def add(prefix, k, **kw):
        base = {"class10": "x", "decision": "known_fault", "fault_class": "imbalance", "caution": "", "variants_disagree": False,
                "cnn_lr_agree": True, "narrow": False, "acquisition_warning": False, "warnings": ""}
        rows.extend({**base, "rec_id": f"{prefix}{i:02d}", **kw} for i in range(k))
    add("u", n_unknown, decision="unknown", fault_class=None)
    add("d", n_disagree, variants_disagree=True)
    add("c", n_clean)
    add("h", n_healthy, decision="healthy", fault_class=None)
    add("w", n_warn, acquisition_warning=True, warnings="3 glitched tacho intervals")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- selection
def test_selection_is_reproducible_from_its_seed_and_follows_the_strata():
    v = pool()
    a, b = select_cases(v, 42), select_cases(v.sample(frac=1, random_state=1), 42)  # input order does not matter
    assert a.equals(b) and len(a) == 15 and a.rec_id.is_unique and (a.seed == 42).all()
    assert not select_cases(v, 43).rec_id.tolist() == a.rec_id.tolist()
    by = v.set_index("rec_id")
    for name, n, pred in STRATA:
        got = a[a.stratum == name].rec_id
        assert len(got) == n and all(pred(by.loc[r].rename(r).to_frame().T.reset_index(names="rec_id").iloc[0]) for r in got)


def test_csv_booleans_are_read_as_booleans(tmp_path):
    v = pool()
    v.to_csv(tmp_path / "v.csv", index=False)
    back = pd.read_csv(tmp_path / "v.csv", keep_default_na=False)
    assert select_cases(back, 42).equals(select_cases(v, 42))


def test_a_short_stratum_stops_with_the_counts():
    with pytest.raises(InsufficientStratum) as e:
        select_cases(pool(n_unknown=3, n_disagree=0), 42)
    assert e.value.counts["S1_unknown"] == {"needed": 4, "candidates": 3}
    assert e.value.counts["S2_known_fault_caution_or_disagreeing"]["candidates"] == 0


@pytest.mark.skipif(not (D7 / "cases.csv").exists(), reason="the committed case list does not exist yet")
def test_committed_cases_are_eligible_and_reproduce_from_the_committed_pool():
    from backend.unseen import split_eligible
    from ml.inference import _metadata
    meta = _metadata().reset_index().set_index("rec_id")
    verdicts = pd.read_csv(D7 / "pool_verdicts.csv", keep_default_na=False)
    cases = pd.read_csv(D7 / "cases.csv")
    assert cases.equals(select_cases(verdicts, int(cases.seed.iloc[0])))
    assert all(split_eligible(meta.loc[r]) for r in verdicts.rec_id)
    assert set(verdicts.rec_id) == {r for r in meta.index if split_eligible(meta.loc[r])}


# ------------------------------------------------------------------------------------------------------------- runner
def test_runner_refuses_a_dirty_tree_or_another_prompt_version():
    run = script("d7_run")
    run.preconditions(run.FROZEN_PROMPT_VERSION, "")
    with pytest.raises(run.Refused, match="not clean"):
        run.preconditions(run.FROZEN_PROMPT_VERSION, " M backend/agent/review.py\n")
    with pytest.raises(run.Refused, match="not clean"):
        run.preconditions(run.FROZEN_PROMPT_VERSION, "?? notes.txt\n")
    with pytest.raises(run.Refused, match="frozen"):
        run.preconditions("0000000000000000", "")
    assert run.FROZEN_PROMPT_VERSION == "8fb7865754455d65"


# -------------------------------------------------------------------------------------------------------------- sheet
FACTS = {"rec_id": "r1", "case_id": "c1", "stratum": "S1_unknown", "verdict": "further inspection needed", "decision": "unknown",
         "location": None, "checks": [{"name": "classifier confidence (MSP)", "score": 0.95, "required": "≥", "threshold": 0.998895,
                                       "passed": False, "status": "fail"}],
         "top_classes": [{"class": "imbalance", "probability": 0.8}], "speed_hz": 13.62, "warnings": [],
         "ranked_channels": [{"channel": "over_axial", "rank": 1, "change_metric": "rms", "change_factor": 2.1,
                              "typical_healthy_band": 1.4}], "rank_1_channel": "over_axial",
         "section_04": [{"tag": "Guidance", "text": "Do not act on the top class. (source: further-inspection-needed)"}],
         "guidance_ids": ["further-inspection-needed"], "report_sections": {}}


def line(arm, q, response):
    return {"arm": arm, "rec_id": "r1", "case_id": "c1", "stratum": "S1_unknown", "q": q, "question": f"Question {q}?",
            "prompt_version": "8fb7865754455d65", "code_revision": "abc", "run_started": "t", "response": response}


def agent(status, source="agent", **kw):
    return {"answer": "The recording needs further inspection.", "claims": [], "source": source, "reason": kw.get("reason"),
            "citations": [{"id": "further-inspection-needed", "title": "What to do"}], "review": {"status": status, "failed": []},
            "actions": [{"tool": "diagnose", "arguments": {}, "summary": "verdict: further inspection needed"}],
            "prompt_version": "8fb7865754455d65", "model": "m", "latency_ms": 1000, "tool_calls_used": 1,
            "report_sections": kw.get("report_sections")}


RUNS = [line("A", 1, {"fixed_report_section": {"n": "01", "title": "What the readings suggest",
                                               "items": [{"tag": "Model", "text": "Verdict: further inspection needed."}]}}),
        line("B", 1, agent("off")), line("C", 1, agent("passed")), line("C", 2, agent("revised")),
        line("C", 3, agent("fallback", source="fallback", reason="evidence review failed",
                           report_sections=[{"n": "04", "title": "What to inspect next", "items": [{"text": "Do not act."}]}]))]
GIVEAWAY = ("agent", "fallback", "revised", "review", "Checked against the case record", "Revised once",
            "Could not answer with evidence", '"source"', "data-source", '"arm"', "status")
# a bare "source" is allowed: section 04 items in the case facts read "(source: <entry id>)"


def test_sheet_gives_no_arm_away_outside_the_answer_text():
    sh = script("d7_sheet")
    items, _ = sh.items_and_key(RUNS)
    page = sh.page(items, {"r1": FACTS})
    outside = re.sub(re.escape(sh.ANS_OPEN) + ".*?" + re.escape(sh.ANS_CLOSE), " ", page, flags=re.S)
    assert page.count(sh.ANS_OPEN) == len(RUNS) and "data-answer" in outside
    found = [w for w in GIVEAWAY if w.lower() in outside.lower()]
    assert found == [], found
    assert "Arm" not in outside and "arm=" not in outside.lower()


def test_key_round_trips_and_is_reproducible(tmp_path):
    sh = script("d7_sheet")
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RUNS))
    (tmp_path / "cases.json").write_text(json.dumps({"r1": FACTS}))
    sh.build(tmp_path)
    key = json.loads((tmp_path / "key.json").read_text())
    page = (tmp_path / "scoring.html").read_text()
    ids = re.findall(r"data-item='(i[0-9a-f]{10})'", page)
    assert sorted(ids) == sorted(key["items"]) and len(set(ids)) == len(RUNS)
    assert sorted((v["arm"], v["rec_id"], v["q"]) for v in key["items"].values()) == sorted((r["arm"], r["rec_id"], r["q"]) for r in RUNS)
    again, _ = sh.items_and_key(RUNS)
    assert [i["id"] for i in again] == ids  # same seed → same ids and order
    assert "key.json" not in page


# ----------------------------------------------------------------------------------------------------------- analysis
def test_analysis_numbers_on_a_synthetic_scores_file():
    key = {"items": {}}
    rows = []
    # 4 cases × question 5, arms B and C: unsupported_repair B yes on cases 0-2, C yes on case 0 only
    for case in range(4):
        for arm, yes in (("B", case <= 2), ("C", case == 0)):
            i = f"i{arm}{case}"
            key["items"][i] = {"arm": arm, "rec_id": f"r{case}", "q": 5}
            rows.append({"item_id": i, "q": "5", "supported": "yes", "unsupported_repair": "yes" if yes else "no",
                         "useful_next_inspection": "no" if arm == "C" and case == 3 else "yes",
                         "unnecessary_refusal": "", "correct": ""})
    t = scored_table(pd.DataFrame(rows), key)
    arms = per_arm(t).set_index(["arm", "measure"])
    assert (arms.loc[("B", "unsupported_repair"), ["k", "n"]] == [3, 4]).all()
    assert (arms.loc[("C", "unsupported_repair"), ["k", "n"]] == [1, 4]).all()
    assert arms.loc[("C", "unsupported_repair"), "lo"] == pytest.approx(wilson(1, 4)[0])
    assert arms.loc[("B", "unnecessary_refusal"), "n"] == 0  # blanks are not scored
    p = paired(t).set_index("measure")
    assert (p.loc["unsupported_repair", ["pairs", "yes_B_only", "yes_C_only", "both_yes", "both_no"]] == [4, 2, 0, 1, 1]).all()
    assert p.loc["unsupported_repair", "mcnemar_p"] == pytest.approx(0.5)
    assert p.loc["useful_next_inspection", "yes_B_only"] == 1
    assert wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4) and mcnemar_exact(1, 8) == pytest.approx(10 / 256)
