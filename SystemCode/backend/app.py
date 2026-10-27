"""FastAPI app (Stages App-1, App-2a). Start with:  uvicorn backend.app:create_app --factory
Paths come from backend/settings.py (env vars with defaults). The bundle is loaded and hash-verified at startup; on any
mismatch the app refuses to start."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.engine import (Engine, InputError, RecordingNotAvailable, ReservedRecordingError, ReservedUploadError,
                            StoredUploadMissing, UploadError, parse_upload)
from backend.checks import assessment
from backend.report import build_report, build_sections, guidance_items
from backend.settings import demo_manifest, raw_dir, showcase_file, use_demo_pack

EVIDENCE_MISSING = ("channel evidence is not available: the healthy reference recording it is compared with is not in "
                    "this installation")


def create_app(bundle_directory=None, cases_path=None, guard_path=None, llm_transport=None, unseen_path=None,
               final_directory=None) -> FastAPI:
    """llm_transport: an httpx transport for the explanation provider (tests mock the model with it). unseen_path /
    final_directory: the App-6 curated file and the leave-one-class-out models (tests point them elsewhere)."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        use_demo_pack()  # a clone without data/raw_npy reads the committed demo pack (backend/settings.py)
        # raises BundleIntegrityError → the server does not start
        app.state.engine = Engine(bundle_directory, cases_path=cases_path, guard_path=guard_path)
        app.state.llm_transport = llm_transport
        app.state.unseen = None  # App-6: created on first use (the final models load lazily)
        yield

    app = FastAPI(title="MachineSense API", version="0.2.0", lifespan=lifespan,
                  description="Research demonstration; development results; not validated for real machines.")

    def engine(request: Request) -> Engine:
        return request.app.state.engine

    @lru_cache(maxsize=32)
    def _evidence(eng: Engine, case_id: str, kind: str) -> dict:
        from backend.evidence import signals
        got = eng.case_array(case_id)
        if got is None:
            raise HTTPException(404, f"no stored signal for case {case_id}")
        if kind == "channels":
            ev = eng.channels(case_id)
            if ev is None:
                raise HTTPException(404, EVIDENCE_MISSING)
            return ev
        x, fr, _ = got
        out = signals(x, fr, eng.bundle.config)
        return {**out, "reference": _reference_spectrum(eng, case_id)}

    def _reference_spectrum(eng: Engine, case_id: str) -> dict | None:
        """App-8.6 item 26: the order spectrum of the healthy reference recording the channel comparison uses (a
        development-pool training normal; ml.inference.load_recording refuses reserved ids), at its own measured speed."""
        from backend.evidence import signals
        from ml.inference import load_recording
        try:
            ref = _evidence(eng, case_id, "channels")["reference"]
            xr, _ = load_recording(ref["rec_id"])
            s = signals(xr, float(ref["speed_hz"]), eng.bundle.config)
        except Exception:  # the overlay is optional; the page says so when it is missing
            return None
        return {"rec_id": ref["rec_id"], "speed_hz": float(ref["speed_hz"]),
                "spectrum": {ch: v["spectrum"] for ch, v in s["channels"].items() if ch != "tacho"},
                "note": "order spectrum of the healthy training recording used by the channel comparison (development data)"}

    def _need(eng: Engine, case_id: str) -> dict:
        c = eng.case_entry(case_id)
        if c is None:
            raise HTTPException(404, f"no case {case_id}")
        return c

    @app.get("/api/health")
    def health(request: Request):
        return engine(request).health()

    @app.get("/api/recordings")
    def recordings(request: Request):
        demo = demo_manifest()
        return {"recordings": engine(request).recordings(),
                "note": "development-pool recordings only; reserved-test recordings are never listed",
                "data_mode": "demo-pack" if demo else "full", **({"demo_note": demo["note"]} if demo else {})}

    @app.get("/api/showcase")
    def showcase(request: Request):
        p = showcase_file()
        if not p.exists():
            return {"showcase": [], "states_without_a_development_recording": [], "note": "showcase not built"}
        out = json.loads(p.read_text())
        # only examples that can be run here (in a demo-pack clone: the demo recordings)
        listed = {r["rec_id"] for r in engine(request).recordings()}
        return {**out, "showcase": [s for s in out["showcase"] if s["rec_id"] in listed]}

    def unseen_service(request: Request):
        from backend.unseen import UnseenService
        if request.app.state.unseen is None:
            request.app.state.unseen = UnseenService(engine(request).bundle, unseen_path, final_directory)
        return request.app.state.unseen

    @app.get("/api/unseen")
    def unseen(request: Request):
        """App-6: the curated "Explore an unseen fault" examples (no results until an example is run)."""
        return unseen_service(request).summary()

    @app.get("/api/unseen/run")
    def unseen_run(request: Request, withheld: str, rec_id: str, role: str = "withheld"):
        """App-6: one example through the leave-one-class-out model of `withheld` (live when the weights exist, else the
        saved values). Reserved recordings and ineligible examples are refused."""
        try:
            return unseen_service(request).run(withheld, rec_id, role)
        except (PermissionError, ReservedRecordingError) as e:
            raise HTTPException(403, str(e)) from e
        except LookupError as e:
            raise HTTPException(404, str(e)) from e

    def _echo(record: dict, request_id: str | None, input_id: str | None) -> JSONResponse:
        """The record, with the client's request id and input id (rec_id / upload id) echoed as headers so the UI can
        discard a response that no longer matches its selection (App-2a.3)."""
        headers = {"X-Request-Id": request_id or "", "X-Input-Id": input_id or ""}
        return JSONResponse(record, headers={k: v for k, v in headers.items() if v})

    @app.post("/api/diagnose")
    async def diagnose(request: Request):
        eng = engine(request)
        ctype = request.headers.get("content-type", "")
        rid = request.headers.get("x-request-id")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                f = form.get("file")
                if f is None or not hasattr(f, "read"):
                    raise UploadError("no file in the upload (form field 'file')")
                x = parse_upload(await f.read(), eng.bundle.config)
                nominal = form.get("nominal_speed_hz")
                rec = eng.diagnose(array=x, healthy_rule=form.get("healthy_rule") or None,
                                   nominal_speed_hz=float(nominal) if nominal else None)
                return _echo(rec, rid or form.get("request_id"), form.get("upload_id"))
            body = await request.json()
            if not isinstance(body, dict) or not body.get("rec_id"):
                raise HTTPException(422, "send JSON {\"rec_id\": ...} or a multipart upload with a CSV in field 'file'")
            rec = eng.diagnose(rec_id=str(body["rec_id"]), healthy_rule=body.get("healthy_rule"))
            return _echo(rec, rid or body.get("request_id"), str(body["rec_id"]))
        except (ReservedRecordingError, ReservedUploadError) as e:
            raise HTTPException(403, str(e)) from e
        except RecordingNotAvailable as e:
            raise HTTPException(404, str(e)) from e
        except UploadError as e:
            raise HTTPException(422, str(e)) from e
        except InputError as e:
            raise HTTPException(404 if "unknown rec_id" in str(e) else 422, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/cases")
    def recent(request: Request, limit: int = 20):
        return {"cases": engine(request).store.recent(limit)}

    @app.get("/api/cases/{case_id}")
    def case(case_id: str, request: Request):
        return _need(engine(request), case_id)["record"]

    @app.get("/api/cases/{case_id}/activity")
    def activity(case_id: str, request: Request):
        c = _need(engine(request), case_id)
        return {"case_id": case_id, "steps": c["activity"], "rerun_of": c["rerun_of"],
                "planned": "investigation agents; planned, not connected"}

    @app.get("/api/cases/{case_id}/meta")
    def meta(case_id: str, request: Request):
        m = engine(request).meta(case_id)
        if m is None:
            raise HTTPException(404, f"no case {case_id}")
        return m

    @app.post("/api/cases/{case_id}/rerun")
    def rerun(case_id: str, request: Request):
        """A new case from the stored input of `case_id` (dataset: its rec_id; upload: its stored array, through the
        upload guard). 404 unknown case; 409 when the stored upload is missing; 403 if the guard refuses."""
        eng = engine(request)
        try:
            rec = eng.rerun(case_id)
        except KeyError as e:
            raise HTTPException(404, f"no case {case_id}") from e
        except (StoredUploadMissing, RecordingNotAvailable) as e:
            raise HTTPException(409, str(e)) from e
        except (ReservedRecordingError, ReservedUploadError) as e:
            raise HTTPException(403, str(e)) from e
        return _echo(rec, request.headers.get("x-request-id"), f"case:{case_id}")

    @app.get("/api/cases/{case_id}/channels")
    def channels(case_id: str, request: Request):
        eng = engine(request)
        _need(eng, case_id)
        return {"case_id": case_id, **_evidence(eng, case_id, "channels")}

    @app.get("/api/cases/{case_id}/signals")
    def signals(case_id: str, request: Request):
        eng = engine(request)
        _need(eng, case_id)
        return {"case_id": case_id, **_evidence(eng, case_id, "signals")}

    @app.get("/api/cases/{case_id}/trace")
    def trace(case_id: str, request: Request):
        """Read-only pipeline trace of a saved case (App-4.1): a derived, versioned view; never re-runs inference."""
        from backend.trace import build_trace
        eng = engine(request)
        c = _need(eng, case_id)
        b = eng.bundle
        return build_trace(case_id, c, eng.store.get_explanation(case_id), b.config.model_dump(mode="json"),
                           b.manifest.model_dump(mode="json"), b.manifest_sha256, eng.meta(case_id)["can_rerun"],
                           n_features=len(b.feature_dictionary))  # App-8.6 item 23

    def _explain(request: Request, case_id: str, user_fields: dict | None) -> dict:
        from backend.explain import explain_case
        from backend.explain.provider import OpenAICompatibleProvider
        from backend.settings import (explanations_log, llm_api_key, llm_base_url, llm_enabled, llm_model,
                                      llm_timeout_s)
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        ch = eng.channels(case_id)
        provider = OpenAICompatibleProvider(llm_base_url(), llm_model(), llm_timeout_s(), llm_api_key(),
                                            transport=request.app.state.llm_transport)
        out = explain_case(rec, assessment(rec), ch, user_fields, provider, enabled=llm_enabled(), log_path=explanations_log(),
                           guidance=guidance_items(rec, eng.bundle.feature_dictionary.index))
        eng.store.put_explanation(case_id, out)
        return out

    @app.get("/api/cases/{case_id}/explanation")
    def get_explanation(case_id: str, request: Request):
        eng = engine(request)
        _need(eng, case_id)
        e = eng.store.get_explanation(case_id)
        if e is None:
            raise HTTPException(404, "no explanation yet; POST to generate one")
        # App-8.8 item 1: stored explanations are served with every measured comparison checked against the record
        from backend.explain.evidence import reviewed
        return reviewed(e, eng.channels(case_id))

    @app.post("/api/cases/{case_id}/explanation")
    async def post_explanation(case_id: str, request: Request):
        """Generate (or regenerate) the explanation. Optional JSON {"user_fields": {"upload_filename", "user_name"}}:
        user-provided strings, passed to the model only as data fields."""
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json() or {}
        uf = body.get("user_fields") if isinstance(body, dict) else None
        uf = {k: str(v)[:200] for k, v in (uf or {}).items() if k in ("upload_filename", "user_name") and v}
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_explain, request, case_id, uf)

    QUESTION_MAX = 500

    def _ask(request: Request, case_id: str, question: str, conversation: bool = False) -> dict:
        from datetime import datetime, timezone

        from backend.agent.loop import Budget, ask
        from backend.explain.provider import OpenAICompatibleProvider
        from backend.settings import agent_review, agent_timeout_s, llm_api_key, llm_base_url, llm_enabled, llm_model
        eng = engine(request)
        provider = OpenAICompatibleProvider(llm_base_url(), llm_model(), agent_timeout_s(), llm_api_key(),
                                            transport=request.app.state.llm_transport)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not conversation:  # the App-7 behaviour, exactly as the D7 pilot used it
            out = ask(eng, case_id, question, provider, budget=Budget.from_settings(), review_on=agent_review(),
                      enabled=llm_enabled())
            qid = eng.store.put_question(case_id, created, out)
            return {"id": qid, "case_id": case_id, "created_at": created, **out}
        from backend.agent.memory import context_turns
        conv = eng.store.conversation(case_id)
        history = context_turns(eng.store.turns(case_id, conv))
        # App-9: the evidence review always gates conversation answers (the D7 switch does not apply here)
        out = ask(eng, case_id, question, provider, budget=Budget.from_settings(), review_on=True, enabled=llm_enabled(),
                  history=history)
        qid = eng.store.put_turn(case_id, conv, created, out)
        return {"id": qid, "case_id": case_id, "created_at": created, "conversation_id": conv, "mode": "conversation", **out}

    @app.post("/api/cases/{case_id}/ask")
    async def ask_question(case_id: str, request: Request):
        """The investigation assistant (App-7): JSON {"question": str}, 1–500 characters. One tool-using agent over this
        case, its answer checked by code before it is returned; with no model, the fallback to the fixed report."""
        _need(engine(request), case_id)
        try:
            body = await request.json()
        except ValueError:
            body = None
        q = body.get("question") if isinstance(body, dict) else None
        if not isinstance(q, str) or not q.strip() or len(q) > QUESTION_MAX:
            raise HTTPException(422, f'send JSON {{"question": ...}} with 1–{QUESTION_MAX} characters')
        conv = body.get("conversation", False)
        if not isinstance(conv, bool):
            raise HTTPException(422, '"conversation" must be true or false')
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_ask, request, case_id, q.strip(), conv)

    @app.post("/api/cases/{case_id}/conversations")
    def new_conversation(case_id: str, request: Request):
        """App-9 "New conversation": later questions no longer see this case's earlier turns as context. Deletes nothing
        (the diagnosis, the stored questions and notes stay)."""
        from datetime import datetime, timezone
        eng = engine(request)
        _need(eng, case_id)
        cid = eng.store.new_conversation(case_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return {"case_id": case_id, "conversation_id": cid}

    @app.get("/api/cases/{case_id}/questions")
    def questions(case_id: str, request: Request):
        eng = engine(request)
        _need(eng, case_id)
        return {"case_id": case_id, "questions": eng.store.questions(case_id),
                "current_conversation_id": eng.store.conversation(case_id),
                "conversation": [t["id"] for t in eng.store.turns(case_id, eng.store.conversation(case_id))]}

    # ------------------------------------------------------------- App-9 slice 2: technician observations and status
    # Human records beside the case: they never change the evidence record, the tools or the report (addendum B).
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    async def _body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(422, "send a JSON object")
        return body

    @app.get("/api/cases/{case_id}/observations")
    def observations(case_id: str, request: Request):
        from backend.casenotes import LOCAL_DEMO
        eng = engine(request)
        _need(eng, case_id)
        return {"case_id": case_id, "observations": eng.store.observations(case_id), "local_demo": LOCAL_DEMO}

    @app.post("/api/cases/{case_id}/observations")
    async def add_observation(case_id: str, request: Request):
        from backend.casenotes import Invalid, actor, observation_text
        eng = engine(request)
        _need(eng, case_id)
        body = await _body(request)
        try:
            text = observation_text(body.get("text"))
        except Invalid as e:
            raise HTTPException(422, str(e)) from e
        return eng.store.add_observation(case_id, _now(), actor(body.get("author")), text)

    @app.patch("/api/cases/{case_id}/observations/{oid}")
    async def edit_observation(case_id: str, oid: int, request: Request):
        from backend.casenotes import Invalid, actor, observation_text
        eng = engine(request)
        _need(eng, case_id)
        body = await _body(request)
        try:
            text = observation_text(body.get("text"))
        except Invalid as e:
            raise HTTPException(422, str(e)) from e
        out = eng.store.edit_observation(case_id, oid, _now(), actor(body.get("author")), text)
        if out is None:
            raise HTTPException(404, f"no observation {oid} for case {case_id}")
        return out

    @app.get("/api/cases/{case_id}/fault-status")
    def fault_status(case_id: str, request: Request):
        from backend.casenotes import fault_status as status
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        return status(rec, eng.store.status_events(case_id))

    @app.post("/api/cases/{case_id}/confirmations")
    async def add_confirmation(case_id: str, request: Request):
        from backend.casenotes import Invalid, actor, fault_status as status, status_event
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        body = await _body(request)
        try:
            st, fc, part, note = status_event(body)
        except Invalid as e:
            raise HTTPException(422, str(e)) from e
        event = eng.store.add_status_event(case_id, _now(), actor(body.get("author")), st, fc, part, note)
        return {"event": event, **status(rec, eng.store.status_events(case_id))}

    @app.get("/api/cases/{case_id}/procedures")
    def procedures(case_id: str, request: Request):
        """App-9 addendum C: post-confirmation procedures, offered only through the gate (a recorded technician
        confirmation AND a reviewed entry for that fault and part); otherwise "No verified procedure is available."."""
        from backend.casenotes import procedure_gate
        eng = engine(request)
        _need(eng, case_id)
        return procedure_gate(eng.store, case_id)

    @app.get("/api/cases/{case_id}/report")
    def report(case_id: str, request: Request):
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        ch = eng.channels(case_id)
        return JSONResponse({"case_id": case_id, "format": "markdown", "report": build_report(rec),
                             "sections": build_sections(rec, ch, eng.bundle.feature_dictionary.index),
                             "assessment": assessment(rec),
                             "tags": ["Measured", "Model", "Guidance", "Missing"],
                             "generated_by": "fixed template from the evidence record; no language model"})

    return app
