"""Post-confirmation maintenance procedures (Stage App-9, CLAUDE.md addendum C): a schema separate from the inspection
guidance library (ml/guidance.py), versioned ("procedure-1"), and the one gate that decides whether a procedure may be
offered. A procedure is offered only when the case's LATEST technician event is a confirmation AND an entry reviewed by a
named reviewer lists that fault class and that part. Advisory and read-only: nothing here controls a machine, claims a
repair is complete or supplies values that an entry does not state."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.labels import CLASS10
from ml.paths import SYSTEM_CODE_DIR

PROCEDURES_DIR = SYSTEM_CODE_DIR / "knowledge" / "procedures"
SCHEMA_VERSION = "procedure-1"
PARTS = ("motor", "coupling", "shaft", "in_brg", "rotor", "out_brg")
NONE_AVAILABLE = "No verified procedure is available."
Verification = Literal["reviewed", "awaiting_source", "unavailable", "draft"]


class Procedure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["procedure-1"]
    id: str
    version: str
    title: str
    machine: str
    parts: tuple[str, ...] = Field(min_length=1)
    fault_classes: tuple[str, ...] = Field(min_length=1)
    prerequisites: tuple[str, ...]
    safety_stop: tuple[str, ...] = Field(min_length=1)
    steps: tuple[str, ...] = Field(min_length=1)
    source: str
    source_url: str | None
    licence: str
    verification: Verification
    reviewer: str | None = None
    reviewed_on: date | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check(self):
        bad_parts = set(self.parts) - set(PARTS)
        bad_classes = set(self.fault_classes) - (set(CLASS10) - {"normal"})
        if bad_parts or bad_classes:
            raise ValueError(f"{self.id}: unknown parts {sorted(bad_parts)} or fault classes {sorted(bad_classes)}")
        if self.verification == "reviewed" and not (self.reviewer and self.reviewed_on and self.source.strip()):
            raise ValueError(f"{self.id}: a reviewed procedure needs a reviewer, a review date and a source")
        return self


def load_procedures(path: Path = PROCEDURES_DIR) -> list[Procedure]:
    """Every procedure entry (*.yaml) in `path`; an id different from its file stem, or a duplicate id, is an error."""
    out, seen = [], set()
    for f in sorted(Path(path).glob("*.yaml")) if Path(path).exists() else []:
        p = Procedure(**yaml.safe_load(f.read_text()))
        if p.id != f.stem or p.id in seen:
            raise ValueError(f"{f.name}: id {p.id!r} must equal the file stem and be unique")
        seen.add(p.id)
        out.append(p)
    return out


def gate(events: list[dict], entries: list[Procedure]) -> dict:
    """The procedure gate for one case from its technician events (oldest first) and the procedure entries: open only when
    the latest event is a confirmation and a REVIEWED entry lists its fault class and part. Returns {open, reason,
    latest_status, fault_class, part, procedures}."""
    latest = events[-1] if events else None
    base = {"open": False, "latest_status": latest["status"] if latest else None,
            "fault_class": latest.get("fault_class") if latest else None, "part": latest.get("part") if latest else None,
            "procedures": [], "message": NONE_AVAILABLE}
    if latest is None:
        return base | {"reason": "no technician confirmation has been recorded"}
    if latest["status"] != "confirmed":
        return base | {"reason": f"the latest technician event is a {latest['status']} status, not a confirmation"}
    fit = [p for p in entries if latest["fault_class"] in p.fault_classes and latest["part"] in p.parts]
    reviewed = [p for p in fit if p.verification == "reviewed"]
    if not reviewed:
        why = ("no procedure entry covers this fault and part" if not fit else
               "the entries covering this fault and part are not reviewed (" + ", ".join(f"{p.id}: {p.verification}" for p in fit) + ")")
        return base | {"reason": why}
    return base | {"open": True, "reason": "a technician confirmation is recorded and a reviewed entry applies", "message": None,
                   "procedures": [p.model_dump(mode="json") for p in reviewed]}
