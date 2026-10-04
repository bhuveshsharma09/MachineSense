"""The curated guidance library (knowledge/entries/*.yaml; CLAUDE.md §11 rule 7): a pure loader and the whole selection
rule for "What to inspect next". Nothing is generated: every inspection step offered comes from an entry and is cited by
its id. No formatting here; the report decides how steps are shown."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.labels import CLASS10
from ml.paths import SYSTEM_CODE_DIR

ENTRIES_DIR = SYSTEM_CODE_DIR / "knowledge" / "entries"
Verification = Literal["internal", "ai_drafted", "open_verified", "citation_unverified", "needs_paywalled_source"]
ORDER = ("internal", "open_verified", "citation_unverified", "ai_drafted")  # offered entries, most to least established
NEVER_OFFERED = "needs_paywalled_source"
UNKNOWN = "unknown"  # the further-inspection verdict, a valid applies_to label besides the class10 labels (App-5 addendum)


class Step(BaseModel):
    """A step shown only when one of `when_warning` (substrings) appears in a recorded warning (App-5 addendum)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    when_warning: tuple[str, ...] = Field(min_length=1)


def step_text(s: str | Step) -> str:
    return s if isinstance(s, str) else s.text


def _matches(needles, warn: list[str]) -> bool:
    """Case-insensitive substring of any recorded (lowercased) warning: the matching for `excludes` and `when_warning`."""
    return any(x.lower() in w for x in needles for w in warn)


class Entry(BaseModel):
    """One entry, exactly the README schema."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    title: str
    source: str
    source_url: str | None
    licence: str
    verification: Verification
    drafted_with_ai: bool
    applies_to: Literal["any"] | tuple[str, ...]
    evidence_required: tuple[str, ...]
    excludes: tuple[str, ...]
    steps: tuple[str | Step, ...]
    notes: str

    @field_validator("applies_to")
    @classmethod
    def _labels(cls, v):
        if v != "any":
            bad = [x for x in v if x not in CLASS10 and x != UNKNOWN]
            if bad:
                raise ValueError(f"applies_to holds labels that are neither class10 labels nor 'unknown': {bad}")
        return v


def load_entries(path: Path | str = ENTRIES_DIR) -> list[Entry]:
    """Every knowledge/entries/*.yaml, validated; an id that differs from its file stem, or a duplicate id, is an error."""
    out, seen = [], set()
    for f in sorted(Path(path).glob("*.yaml")):
        e = Entry.model_validate(yaml.safe_load(f.read_text()))
        if e.id != f.stem:
            raise ValueError(f"{f.name}: id {e.id!r} does not equal the file stem {f.stem!r}")
        if e.id in seen:
            raise ValueError(f"duplicate entry id {e.id!r}")
        seen.add(e.id)
        out.append(e)
    return out


def select(entries: list[Entry], *, condition: str, available_evidence, warnings) -> list[Entry]:
    """The selection rule: drop needs_paywalled_source; keep entries that apply to `condition` (or to any); keep only those
    whose every evidence_required name is available; drop any whose `excludes` string appears (case-insensitive substring)
    in a recorded warning. Then each kept entry's steps are filtered: a plain step stays; a `when_warning` step stays only if
    one of its substrings appears in a recorded warning; an entry left without steps is dropped. Returned entries carry only
    their kept steps, in file order. Ordered internal, open_verified, citation_unverified, ai_drafted, then by id."""
    avail = set(available_evidence)
    warn = [w.lower() for w in warnings]
    keep = []
    for e in entries:
        if (e.verification == NEVER_OFFERED or not (e.applies_to == "any" or condition in e.applies_to)
                or not all(n in avail for n in e.evidence_required) or _matches(e.excludes, warn)):
            continue
        steps = tuple(s for s in e.steps if isinstance(s, str) or _matches(s.when_warning, warn))
        if steps:
            keep.append(e if steps == e.steps else e.model_copy(update={"steps": steps}))
    return sorted(keep, key=lambda e: (ORDER.index(e.verification), e.id))
