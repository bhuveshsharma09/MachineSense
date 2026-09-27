"""FastAPI app (Stage App-1). Start with:  MACHINESENSE_BUNDLE_DIR=<approved bundle> uvicorn backend.app:create_app --factory
The bundle is loaded and hash-verified at startup; on any mismatch the app refuses to start."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.engine import Engine, InputError, ReservedRecordingError, UploadError, parse_upload
from backend.report import build_report


def create_app(bundle_directory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = Engine(bundle_directory)  # raises BundleIntegrityError → the server does not start
        yield

    app = FastAPI(title="MachineSense API", version="0.1.0", lifespan=lifespan,
                  description="Research demonstration; development results; not validated for real machines.")

    def engine(request: Request) -> Engine:
        return request.app.state.engine

    @app.get("/api/health")
    def health(request: Request):
        return engine(request).health()

    @app.get("/api/recordings")
    def recordings(request: Request):
        return {"recordings": engine(request).recordings(),
                "note": "development-pool recordings only; reserved-test recordings are never listed"}

    @app.post("/api/diagnose")
    async def diagnose(request: Request):
        eng = engine(request)
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                f = form.get("file")
                if f is None or not hasattr(f, "read"):
                    raise UploadError("no file in the upload (form field 'file')")
                x = parse_upload(await f.read(), eng.bundle.config)
                nominal = form.get("nominal_speed_hz")
                return eng.diagnose(array=x, healthy_rule=form.get("healthy_rule") or None,
                                    nominal_speed_hz=float(nominal) if nominal else None)
            body = await request.json()
            if not isinstance(body, dict) or not body.get("rec_id"):
                raise HTTPException(422, "send JSON {\"rec_id\": ...} or a multipart upload with a CSV in field 'file'")
            return eng.diagnose(rec_id=str(body["rec_id"]), healthy_rule=body.get("healthy_rule"))
        except ReservedRecordingError as e:
            raise HTTPException(403, str(e)) from e
        except UploadError as e:
            raise HTTPException(422, str(e)) from e
        except InputError as e:
            raise HTTPException(404 if "unknown rec_id" in str(e) else 422, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/cases/{case_id}")
    def case(case_id: str, request: Request):
        rec = engine(request).case(case_id)
        if rec is None:
            raise HTTPException(404, f"no case {case_id}")
        return rec

    @app.get("/api/cases/{case_id}/report")
    def report(case_id: str, request: Request):
        rec = engine(request).case(case_id)
        if rec is None:
            raise HTTPException(404, f"no case {case_id}")
        return JSONResponse({"case_id": case_id, "format": "markdown", "report": build_report(rec),
                             "generated_by": "fixed template from the evidence record; no language model"})

    return app
