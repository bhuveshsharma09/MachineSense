"""FastAPI app (Stages App-1, App-2a). Start with:  uvicorn backend.app:create_app --factory
Paths come from backend/settings.py (env vars with defaults). The bundle is loaded and hash-verified at startup; on any
mismatch the app refuses to start."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.engine import (Engine, InputError, ReservedRecordingError, ReservedUploadError, StoredUploadMissing,
                            UploadError, parse_upload)
from backend.checks import assessment
from backend.report import build_report, build_sections
from backend.settings import showcase_file


def create_app(bundle_directory=None, cases_path=None, guard_path=None, llm_transport=None) -> FastAPI:
    """llm_transport: an httpx transport for the explanation provider (tests mock the model with it)."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # raises BundleIntegrityError → the server does not start
        app.state.engine = Engine(bundle_directory, cases_path=cases_path, guard_path=guard_path)
        app.state.llm_transport = llm_transport
        yield

    app = FastAPI(title="MachineSense API", version="0.2.0", lifespan=lifespan,
                  description="Research demonstration; development results; not validated for real machines.")

    def engine(request: Request) -> Engine:
        return request.app.state.engine

    @lru_cache(maxsize=32)
    def _evidence(eng: Engine, case_id: str, kind: str) -> dict:
        from backend.evidence import channel_evidence, signals
        got = eng.case_array(case_id)
        if got is None:
            raise HTTPException(404, f"no stored signal for case {case_id}")
        x, fr, rec_id = got
        cfg = eng.bundle.config
        if kind == "channels":
            from ml.inference import _metadata
            d = eng.case(case_id)["diagnosis"]
            return channel_evidence(x, fr, cfg, _metadata().reset_index(), rec_id, eng.variation, d["decision"], d["caution"])
        return signals(x, fr, cfg)

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
        return {"recordings": engine(request).recordings(),
                "note": "development-pool recordings only; reserved-test recordings are never listed"}

    @app.get("/api/showcase")
    def showcase():
        p = showcase_file()
        if not p.exists():
            return {"showcase": [], "states_without_a_development_recording": [], "note": "showcase not built"}
        return json.loads(p.read_text())

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
        except StoredUploadMissing as e:
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
                           b.manifest.model_dump(mode="json"), b.manifest_sha256, eng.meta(case_id)["can_rerun"])

    def _explain(request: Request, case_id: str, user_fields: dict | None) -> dict:
        from backend.explain import explain_case
        from backend.explain.provider import OpenAICompatibleProvider
        from backend.settings import (explanations_log, llm_api_key, llm_base_url, llm_enabled, llm_model,
                                      llm_timeout_s)
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        ch = _evidence(eng, case_id, "channels") if eng.case_array(case_id) is not None else None
        provider = OpenAICompatibleProvider(llm_base_url(), llm_model(), llm_timeout_s(), llm_api_key(),
                                            transport=request.app.state.llm_transport)
        out = explain_case(rec, assessment(rec), ch, user_fields, provider, enabled=llm_enabled(), log_path=explanations_log())
        eng.store.put_explanation(case_id, out)
        return out

    @app.get("/api/cases/{case_id}/explanation")
    def get_explanation(case_id: str, request: Request):
        eng = engine(request)
        _need(eng, case_id)
        e = eng.store.get_explanation(case_id)
        if e is None:
            raise HTTPException(404, "no explanation yet; POST to generate one")
        return e

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

    @app.get("/api/cases/{case_id}/report")
    def report(case_id: str, request: Request):
        eng = engine(request)
        rec = _need(eng, case_id)["record"]
        ch = _evidence(eng, case_id, "channels") if eng.case_array(case_id) is not None else None
        return JSONResponse({"case_id": case_id, "format": "markdown", "report": build_report(rec),
                             "sections": build_sections(rec, ch), "assessment": assessment(rec),
                             "tags": ["Measured", "Model", "Guidance", "Missing"],
                             "generated_by": "fixed template from the evidence record; no language model"})

    return app
