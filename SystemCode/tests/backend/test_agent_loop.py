"""Stage App-7: the assistant loop through the API with a scripted model (httpx.MockTransport; no live model): tool calls
and an answer, the budget, one revision, the fallback that never shows rejected text, the D7 review switch, and the
disabled model with no network call. Reasoning text from the model never reaches a response or the store."""
import json

import httpx
import pytest

BEARING = "underhang__outer_race__0g__14.7456"
SECRET = "PRIVATE-REASONING-TEXT"
CLEAN = {"answer": "The verdict is a known fault, and the cited guidance lists the envelope-spectrum steps for it.",
         "claims": [{"text": "The verdict is a known fault.", "refs": ["diagnosis.decision"]}],
         "citations": ["bearing-defect-frequencies"]}
INVENTED = {**CLEAN, "answer": "The shaft ran at 99.9 Hz, so you should replace the bearing."}


class Model:
    """A scripted chat-completions endpoint: each request pops the next scripted message."""

    def __init__(self):
        self.script, self.calls = [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        msg = self.script.pop(0) if self.script else final(CLEAN)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "reasoning_content": SECRET, **msg}}]})


def tool(*names):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
                                          for i, n in enumerate(names)]}


def final(obj):
    return {"content": json.dumps(obj), "tool_calls": []}


@pytest.fixture(scope="module")
def model():
    return Model()


@pytest.fixture(scope="module")
def app(bundle_path, guard_path, tmp_path_factory, model):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    db = tmp_path_factory.mktemp("app7") / "cases.db"
    with TestClient(create_app(bundle_path, cases_path=db, guard_path=guard_path,
                               llm_transport=httpx.MockTransport(model))) as c:
        yield c


@pytest.fixture(scope="module")
def case(app):
    return app.post("/api/diagnose", json={"rec_id": BEARING}).json()


def ask(app, case, model, script, q="What should I inspect next?"):
    model.script = list(script)
    n = len(model.calls)
    out = app.post(f"/api/cases/{case['case_id']}/ask", json={"question": q})
    assert out.status_code == 200
    return out.json(), model.calls[n:]


def test_two_tools_then_a_checked_answer(app, case, model):
    out, calls = ask(app, case, model, [tool("diagnose"), tool("lookup_guidance"), final(CLEAN)])
    assert out["source"] == "agent" and out["review"] == {"status": "passed", "failed": []}
    assert [a["tool"] for a in out["actions"]] == ["diagnose", "lookup_guidance"] and out["tool_calls_used"] == 2
    assert out["citations"] == [{"id": "bearing-defect-frequencies",
                                 "title": "Confirm a bearing verdict against the envelope defect lines"}]
    assert out["answer"] == CLEAN["answer"] and out["prompt_version"] and out["model"]
    assert len(calls) == 3 and calls[0]["tools"] and calls[1]["messages"][-1]["role"] == "tool"
    assert all("reasoning_content" not in m for c in calls for m in c["messages"])  # never fed back
    assert SECRET not in json.dumps(out)
    stored = app.get(f"/api/cases/{case['case_id']}/questions").json()["questions"]
    assert stored[-1]["id"] == out["id"] and stored[-1]["answer"] == CLEAN["answer"] and SECRET not in json.dumps(stored)


def test_budget_exhausted_falls_back(app, case, model, monkeypatch):
    monkeypatch.setenv("MACHINESENSE_AGENT_MAX_TOOL_CALLS", "2")
    out, calls = ask(app, case, model, [tool("diagnose"), tool("check_signal"), tool("get_evidence")])
    assert out["source"] == "fallback" and out["reason"] == "tool-call budget exhausted" and out["tool_calls_used"] == 2
    assert [s["n"] for s in out["report_sections"]] == ["01", "04"] and out["review"]["status"] == "fallback"


def test_a_bad_answer_is_revised_once_into_a_passing_one(app, case, model):
    out, calls = ask(app, case, model, [tool("diagnose", "lookup_guidance"), final(INVENTED), final(CLEAN)])
    assert out["source"] == "agent" and out["review"]["status"] == "revised" and out["answer"] == CLEAN["answer"]
    feedback = calls[-1]["messages"][-1]
    assert feedback["role"] == "user" and "numbers" in feedback["content"] and "forbidden" in feedback["content"]


def test_two_bad_answers_fall_back_without_the_rejected_text(app, case, model):
    out, _ = ask(app, case, model, [tool("diagnose", "lookup_guidance"), final(INVENTED), final(INVENTED)])
    assert out["source"] == "fallback" and out["reason"] == "evidence review failed"
    assert {f["check"] for f in out["review"]["failed"]} >= {"numbers", "forbidden"}
    stored = app.get(f"/api/cases/{case['case_id']}/questions").json()["questions"][-1]
    for o in (out, stored):
        assert "99.9 Hz" not in json.dumps(o) and "replace the bearing" not in json.dumps(o)
        assert o["answer"].startswith("The assistant could not give an answer supported by the case record")


def test_review_off_shows_the_answer_and_stores_what_it_would_reject(app, case, model, monkeypatch):
    monkeypatch.setenv("MACHINESENSE_AGENT_REVIEW", "false")
    out, calls = ask(app, case, model, [tool("diagnose"), final(INVENTED)])
    assert out["source"] == "agent" and out["answer"] == INVENTED["answer"] and len(calls) == 2  # no revision
    assert out["review"]["status"] == "off" and {f["check"] for f in out["review"]["failed"]} >= {"numbers", "forbidden"}
    stored = app.get(f"/api/cases/{case['case_id']}/questions").json()["questions"][-1]
    assert stored["review"] == out["review"]


def test_llm_disabled_falls_back_with_no_network_call(app, case, model, monkeypatch):
    monkeypatch.setenv("MACHINESENSE_LLM_ENABLED", "false")
    out, calls = ask(app, case, model, [tool("diagnose")])
    assert calls == [] and out["source"] == "fallback" and out["reason"] == "language model disabled"
    assert out["report_sections"][1]["items"] == app.get(f"/api/cases/{case['case_id']}/report").json()["sections"][3]["items"]


def test_unreachable_model_falls_back_and_bad_questions_are_refused(client):
    rec = client.post("/api/diagnose", json={"rec_id": BEARING}).json()
    out = client.post(f"/api/cases/{rec['case_id']}/ask", json={"question": "Why?"}).json()
    assert out["source"] == "fallback" and out["reason"].startswith("language model unreachable")
    assert client.post(f"/api/cases/{rec['case_id']}/ask", json={"question": " "}).status_code == 422
    assert client.post(f"/api/cases/{rec['case_id']}/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/api/cases/no-such-case/ask", json={"question": "Why?"}).status_code == 404
