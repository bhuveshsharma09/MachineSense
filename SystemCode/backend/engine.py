"""The backend's connection to the approved engine: the verified bundle, ml.inference.diagnose(), the permitted
recordings, upload validation and an in-memory case store. Nothing here computes a score; every number comes from
ml.inference.diagnose()."""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.schemas import EvidenceRecord
from ml.inference import InputError, ReservedRecordingError, _metadata, diagnose  # noqa: F401 (errors re-exported)
from ml.paths import SYSTEM_CODE_DIR
from ml.splits import development_pool

BUNDLE_ENV = "MACHINESENSE_BUNDLE_DIR"


class UploadError(ValueError):
    """An uploaded file that cannot be diagnosed; the message is shown to the user."""


def bundle_dir(explicit: str | Path | None = None) -> Path:
    """The approved bundle: an explicit path, else $MACHINESENSE_BUNDLE_DIR, else SystemCode/artifacts/bundle."""
    return Path(explicit or os.environ.get(BUNDLE_ENV) or SYSTEM_CODE_DIR / "artifacts" / "bundle").resolve()


def parse_upload(data: bytes, cfg) -> np.ndarray:
    """A headerless MAFAULDA-style CSV → (250000, 8) float32 array, or UploadError with a clear message. The CSV has no
    time column, so 50 kHz is assumed: 5 s at 50 kHz = 250,000 rows; the row count is what can be checked."""
    from ml.preprocess import load_csv

    pre = cfg.preprocess
    if not data:
        raise UploadError("the uploaded file is empty")
    with tempfile.NamedTemporaryFile(suffix=".csv") as fh:
        fh.write(data)
        fh.flush()
        try:
            x = load_csv(fh.name)
        except Exception as e:  # noqa: BLE001
            raise UploadError(f"the file could not be read as a numeric CSV without a header ({type(e).__name__})") from e
    if x.ndim != 2 or x.shape[1] != pre.n_channels:
        raise UploadError(f"expected {pre.n_channels} columns (tacho, 6 accelerometer axes, microphone), got {x.shape[1]}")
    if x.shape[0] != pre.n_rows:
        raise UploadError(f"expected {pre.n_rows} rows (5 s at {pre.fs_hz // 1000} kHz), got {x.shape[0]}; the sampling "
                          "rate cannot be read from a CSV without a time column, so the row count is checked instead")
    if not np.isfinite(x).all():
        raise UploadError(f"the recording contains {int((~np.isfinite(x)).sum())} missing or non-numeric values")
    return x


class Engine:
    """Loads and verifies the bundle once (refusing on any mismatch) and serves diagnoses as evidence records."""

    def __init__(self, directory: str | Path | None = None):
        from ml.bundle import load_bundle

        self.directory = bundle_dir(directory)
        self.bundle = load_bundle(self.directory, device="cpu")  # verifies every file hash and the config hash
        self.cases: dict[str, dict] = {}

    def health(self) -> dict:
        m = self.bundle.manifest
        return {"status": "ok", "bundle_verified": True, "bundle_dir": str(self.directory),
                "manifest_sha256": self.bundle.manifest_sha256, "schema_version": m.schema_version,
                "bundle_git_sha": m.git_sha, "healthy_rule": self.bundle.rule.get("healthy_rule"),
                "rule_status": self.bundle.rule.get("status"), "train_speed_range_hz": m.data["train_speed_range_hz"],
                "calibration_normal_speed_range_hz": m.data["calibration_normal_speed_range_hz"],
                "test_eligibility": m.test_eligibility, "limitations": m.limitations}

    def recordings(self) -> list[dict]:
        """Development-pool recordings only (reserved-test recordings are never listed), with what they were used for."""
        cfg = self.bundle.config
        meta = _metadata().reset_index()
        pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
        pool = pool.sort_values(["class10", "severity", "fr_measured_hz"])
        used = {"train": "model training", "val": "threshold calibration"}
        return [{"rec_id": r.rec_id, "class10": r.class10, "severity": float(r.severity),
                 "severity_unit": "mm" if "misalignment" in r.class10 else ("g" if r.class10 != "normal" else ""),
                 "speed_hz": float(r.fr_measured_hz), "used_for": used[r.split_closed]} for r in pool.itertuples()]

    def diagnose(self, rec_id: str | None = None, array: np.ndarray | None = None, healthy_rule: str | None = None,
                 nominal_speed_hz: float | None = None) -> dict:
        source = "dataset" if rec_id is not None else "upload"
        out = diagnose(rec_id if rec_id is not None else array, healthy_rule=healthy_rule,
                       nominal_speed_hz=nominal_speed_hz, bundle=self.bundle)
        record = EvidenceRecord.model_validate({"case_id": uuid.uuid4().hex, "source": source,
                                               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                               **out}).to_json()
        self.cases[record["case_id"]] = record
        return record

    def case(self, case_id: str) -> dict | None:
        return self.cases.get(case_id)
