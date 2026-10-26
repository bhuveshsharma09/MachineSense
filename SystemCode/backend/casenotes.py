"""Technician observations and fault status beside a stored case (Stage App-9 slice 2, CLAUDE.md addendum B). These are
human records: they never change the evidence record (the model verdict), never enter the assistant's tools or the
report, and are labelled by kind everywhere: "model prediction", "technician observation", "technician-confirmed fault".
The app has no authentication: every record carries actor = {name, verified: False, method: "local-demo"}; `actor` is
the extension point for an authenticated user later."""
from __future__ import annotations

from ml.labels import CLASS10

FAULT_CLASSES = tuple(c for c in CLASS10 if c != "normal")
# the rig's components (ids as the frontend's rig parts), in the dataset's naming
PARTS = {"motor": "Motor", "coupling": "Coupling", "shaft": "Shaft", "in_brg": "Underhang bearing housing",
         "rotor": "Rotor discs", "out_brg": "Overhang bearing housing"}
STATUSES = ("confirmed", "rejected", "withdrawn")
TEXT_MAX, NOTE_MAX, NAME_MAX = 1000, 1000, 80
LOCAL_DEMO = ("Local demo action: this app does not sign in technicians; the name is the display name set in this "
              "browser and is not verified.")


class Invalid(ValueError):
    pass


def actor(name) -> dict:
    """Who recorded it, as far as this demo can say: a display name, never verified."""
    name = (name or "").strip()[:NAME_MAX] if isinstance(name, str) else ""
    return {"name": name or "Unnamed (local demo)", "verified": False, "method": "local-demo"}


def observation_text(text) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > TEXT_MAX:
        raise Invalid(f"an observation is text of 1–{TEXT_MAX} characters")
    return text.strip()


def status_event(body: dict) -> tuple[str, str | None, str | None, str]:
    """(status, fault_class, part, note) from a request body. A confirmation names a fault class (not normal) and a rig
    part; a rejection or a withdrawal names neither. The fault class may differ from the model's prediction."""
    status = body.get("status")
    if status not in STATUSES:
        raise Invalid(f"status must be one of {', '.join(STATUSES)}")
    note = body.get("note", "")
    if not isinstance(note, str) or len(note) > NOTE_MAX:
        raise Invalid(f"note is text of at most {NOTE_MAX} characters")
    fc, part = body.get("fault_class"), body.get("part")
    if status == "confirmed":
        if fc not in FAULT_CLASSES:
            raise Invalid(f"a confirmation needs fault_class, one of {', '.join(FAULT_CLASSES)}")
        if part not in PARTS:
            raise Invalid(f"a confirmation needs part, one of {', '.join(PARTS)}")
        return status, fc, part, note.strip()
    if fc is not None or part is not None:
        raise Invalid("only a confirmation names a fault class and a part")
    return status, None, None, note.strip()


def fault_status(record: dict, events: list[dict]) -> dict:
    """The model prediction (read from the record, unchanged) beside the technician's latest status event."""
    d = record["diagnosis"]
    latest = events[-1] if events else None
    current = ("technician_confirmed" if latest and latest["status"] == "confirmed"
               else "technician_rejected" if latest and latest["status"] == "rejected" else "not_confirmed")
    return {"case_id": record["case_id"],
            "model_prediction": {"kind": "model_prediction", "decision": d["decision"], "fault_class": d.get("fault_class"),
                                 "display_label": d.get("display_label"),
                                 "note": "Model prediction from the recording; not a physical confirmation."},
            "technician": {"status": current, "latest": latest,
                           "agrees_with_model": (latest["fault_class"] == d.get("fault_class"))
                           if latest and latest["status"] == "confirmed" else None},
            "events": events, "parts": PARTS, "fault_classes": list(FAULT_CLASSES), "local_demo": LOCAL_DEMO}


def procedure_gate(store, case_id: str) -> dict:
    """The backend-enforced procedure gate for a case (ml.procedures.gate over its technician events)."""
    from backend.settings import procedures_dir
    from ml.procedures import SCHEMA_VERSION, gate, load_procedures
    return {"case_id": case_id, "schema_version": SCHEMA_VERSION, **gate(store.status_events(case_id), load_procedures(procedures_dir()))}
