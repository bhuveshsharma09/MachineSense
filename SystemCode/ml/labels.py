"""Map a MAFAULDA CSV path to its labels. The folder layout is the only source of labels."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CLASS10: tuple[str, ...] = (
    "normal",
    "imbalance",
    "horizontal_misalignment",
    "vertical_misalignment",
    "underhang_ball",
    "underhang_cage",
    "underhang_outer",
    "overhang_ball",
    "overhang_cage",
    "overhang_outer",
)
# 6-class view: the three bearing faults at each position collapse into one class.
CLASS6: tuple[str, ...] = (
    "normal",
    "imbalance",
    "horizontal_misalignment",
    "vertical_misalignment",
    "underhang_bearing",
    "overhang_bearing",
)

_BEARING_FOLDERS = {"ball_fault": "ball", "cage_fault": "cage", "outer_race": "outer"}
_TOP_FOLDERS = {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment", "underhang", "overhang"}
_MASS_RE = re.compile(r"^(\d+(?:\.\d+)?)g$")
_MM_RE = re.compile(r"^(\d+(?:\.\d+)?)mm$")


@dataclass(frozen=True)
class RecordingLabels:
    """Labels for one recording.

    severity: imbalance mass (g), misalignment offset (mm), or for bearing faults the *added imbalance mass*
    (g) — not damage size. 0.0 for normal. fr_file_hz: nominal shaft frequency from the file name (Hz).
    """

    fault_family: str  # normal | imbalance | misalignment | bearing
    fault_type: str  # normal | imbalance | horizontal | vertical | ball | cage | outer
    severity: float
    severity_unit: str  # "g", "mm" or "" (normal)
    class10: str
    class6: str
    fr_file_hz: float


def _number(folder: str, pattern: re.Pattern[str], path: Path) -> float:
    m = pattern.match(folder)
    if not m:
        raise ValueError(f"unexpected severity folder {folder!r} in {path}")
    return float(m.group(1))


def parse_labels(path: str | Path) -> RecordingLabels:
    """Labels from a CSV path (absolute or relative to the data root). Raises ValueError on unknown layouts."""
    path = Path(path)
    if path.suffix != ".csv":
        raise ValueError(f"not a CSV file: {path}")
    fr_file_hz = float(path.stem)

    parts = path.parts[:-1]
    # Nearest enclosing top-level class folder, so any parent directories are ignored.
    idx = next((i for i in range(len(parts) - 1, -1, -1) if parts[i] in _TOP_FOLDERS), None)
    if idx is None:
        raise ValueError(f"no MAFAULDA class folder in {path}")
    top, sub = parts[idx], parts[idx + 1:]

    if top == "normal":
        if sub:
            raise ValueError(f"unexpected subfolders under normal/: {path}")
        return RecordingLabels("normal", "normal", 0.0, "", "normal", "normal", fr_file_hz)

    if top == "imbalance":
        if len(sub) != 1:
            raise ValueError(f"expected imbalance/<mass>g/: {path}")
        g = _number(sub[0], _MASS_RE, path)
        return RecordingLabels("imbalance", "imbalance", g, "g", "imbalance", "imbalance", fr_file_hz)

    if top in ("horizontal-misalignment", "vertical-misalignment"):
        if len(sub) != 1:
            raise ValueError(f"expected {top}/<offset>mm/: {path}")
        mm = _number(sub[0], _MM_RE, path)
        direction = top.split("-")[0]
        cls = f"{direction}_misalignment"
        return RecordingLabels("misalignment", direction, mm, "mm", cls, cls, fr_file_hz)

    # underhang / overhang bearing faults
    if len(sub) != 2 or sub[0] not in _BEARING_FOLDERS:
        raise ValueError(f"expected {top}/{{ball_fault,cage_fault,outer_race}}/<mass>g/: {path}")
    fault = _BEARING_FOLDERS[sub[0]]
    g = _number(sub[1], _MASS_RE, path)
    return RecordingLabels("bearing", fault, g, "g", f"{top}_{fault}", f"{top}_bearing", fr_file_hz)
