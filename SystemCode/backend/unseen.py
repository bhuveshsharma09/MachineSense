"""Stage App-6: "Explore an unseen fault" — a saved leave-one-class-out CNN-6 (artifacts/final/loco/<class>/, §12) run on
a recording of the class it never trained on. A SIMULATED unseen fault inside MAFAULDA, not evidence about genuinely novel
real-world faults. The normal application and its bundle are unchanged.

This module holds the single eligibility predicate (`may_show`), used by the build step and by every request."""
from __future__ import annotations

import pandas as pd

from ml.splits import FAULT_CLASSES

ROLES = ("withheld", "familiar")
# The partitions consumed by the single declared final evaluation (F1); a recording in any of them is never shown.
ALLOWED_CLOSED = ("train", "val")
SEALED_COLUMNS = ("split_speed_block", "split_severity")


def split_eligible(row: pd.Series) -> bool:
    """split_closed ∈ {train, val} AND split_speed_block ≠ test AND split_severity ≠ test (metadata.csv)."""
    return row["split_closed"] in ALLOWED_CLOSED and all(row[c] != "test" for c in SEALED_COLUMNS)


def may_show(rec_id: str, withheld: str, role: str, meta: pd.DataFrame, familiar_excluded) -> tuple[bool, str]:
    """May this recording be shown on the page, for this hidden class and role? Returns (allowed, reason).

    - Every role: the recording exists in metadata.csv and passes `split_eligible` (reserved partitions stay sealed).
    - "withheld": its class is the selected model's withheld class (unseen by that model by construction).
    - "familiar": its class is one the model learned, and it is neither one of that run's calibration recordings nor a
      threshold-defining recording (`familiar_excluded`: both sets for that run)."""
    if withheld not in FAULT_CLASSES:
        return False, f"unknown hidden class {withheld!r}"
    if role not in ROLES:
        return False, f"unknown role {role!r}"
    m = meta.set_index("rec_id") if "rec_id" in meta.columns else meta
    if rec_id not in m.index:
        return False, "not a recording of this dataset"
    row = m.loc[rec_id]
    if not split_eligible(row):
        return False, "this is a reserved test recording; reserved partitions stay sealed and are not shown"
    if role == "withheld":
        return (True, "") if row["class10"] == withheld else (False, f"not a recording of the hidden class {withheld}")
    if row["class10"] == withheld:
        return False, "the familiar comparison must be a condition the model learned"
    if rec_id in set(familiar_excluded):
        return False, "this recording calibrated the model's thresholds, so it is not offered as a familiar example"
    return True, ""
