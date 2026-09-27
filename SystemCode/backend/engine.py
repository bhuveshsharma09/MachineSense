"""The backend's connection to the approved engine: the verified bundle, ml.inference.diagnose(), the permitted
recordings, upload validation, the reserved-recording upload guard and the SQLite case history. Nothing here computes a
score; every number comes from ml.inference.diagnose() or, for the evidence views, from ml functions (backend.evidence)."""
from __future__ import annotations

import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend import timing
from backend.guard import ReservedGuard, build_index, reserved_rec_ids
from backend.schemas import EvidenceRecord
from backend.settings import bundle_dir, cases_db, guard_index, raw_dir, uploads_dir
from backend.store import CaseStore
from ml.inference import InputError, ReservedRecordingError, _metadata, diagnose  # noqa: F401 (errors re-exported)
from ml.splits import development_pool

RESERVED_UPLOAD = "this appears to be a reserved test recording"
STEP_NOTE = "deterministic step"


class UploadError(ValueError):
    """An uploaded file that cannot be diagnosed; the message is shown to the user."""


class ReservedUploadError(ValueError):
    """An upload that matches a reserved test recording's fingerprint (best-effort guard)."""


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

    def __init__(self, directory: str | Path | None = None, cases_path: Path | None = None,
                 guard_path: Path | None = None):
        from ml.bundle import load_bundle

        t0 = time.perf_counter()
        self.directory = bundle_dir(directory)
        self.bundle = load_bundle(self.directory, device="cpu")  # verifies every file hash and the config hash
        self.bundle_ms = 1000 * (time.perf_counter() - t0)
        self.store = CaseStore(cases_path or cases_db())
        self.guard = self._guard(Path(guard_path or guard_index()))
        timing.install()

    def _guard(self, path: Path) -> ReservedGuard:
        """The reserved-recording fingerprint index; built read-only from the raw npy files on first start if missing."""
        if not path.exists():
            meta = _metadata().reset_index()
            build_index(reserved_rec_ids(meta, self.bundle.config), raw_dir(), path)
        return ReservedGuard.load(path)

    def health(self) -> dict:
        m = self.bundle.manifest
        return {"status": "ok", "bundle_verified": True, "bundle_dir": str(self.directory),
                "manifest_sha256": self.bundle.manifest_sha256, "schema_version": m.schema_version,
                "bundle_git_sha": m.git_sha, "healthy_rule": self.bundle.rule.get("healthy_rule"),
                "rule_status": self.bundle.rule.get("status"), "train_speed_range_hz": m.data["train_speed_range_hz"],
                "calibration_normal_speed_range_hz": m.data["calibration_normal_speed_range_hz"],
                "test_eligibility": m.test_eligibility, "limitations": m.limitations,
                "upload_guard": {"reserved_recordings_indexed": len(self.guard.index)}}

    def pool(self):
        cfg = self.bundle.config
        return development_pool(_metadata().reset_index(), cfg.development.allowed_closed,
                                cfg.development.reserved_columns)

    def recordings(self) -> list[dict]:
        """Development-pool recordings only (reserved-test recordings are never listed), with what they were used for."""
        pool = self.pool().sort_values(["class10", "severity", "fr_measured_hz"])
        used = {"train": "model training", "val": "threshold calibration"}
        return [{"rec_id": r.rec_id, "class10": r.class10, "severity": float(r.severity),
                 "severity_unit": "mm" if "misalignment" in r.class10 else ("g" if r.class10 != "normal" else ""),
                 "speed_hz": float(r.fr_measured_hz), "used_for": used[r.split_closed]} for r in pool.itertuples()]

    def diagnose(self, rec_id: str | None = None, array: np.ndarray | None = None, healthy_rule: str | None = None,
                 nominal_speed_hz: float | None = None) -> dict:
        source = "dataset" if rec_id is not None else "upload"
        if array is not None:
            hit = self.guard.match(array)
            if hit is not None:
                raise ReservedUploadError(RESERVED_UPLOAD)
        with timing.measure() as steps:
            out = diagnose(rec_id if rec_id is not None else array, healthy_rule=healthy_rule,
                           nominal_speed_hz=nominal_speed_hz, bundle=self.bundle)
        record = EvidenceRecord.model_validate({"case_id": uuid.uuid4().hex, "source": source,
                                               "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                               **out}).to_json()
        from backend.report import build_report
        t = time.perf_counter()
        build_report(record)
        report_ms = 1000 * (time.perf_counter() - t)
        activity = self._activity(record, steps, report_ms)
        array_path = None
        if array is not None:
            d = uploads_dir()
            d.mkdir(parents=True, exist_ok=True)
            array_path = str(d / f"{record['case_id']}.npy")
            np.save(array_path, np.asarray(array, dtype=np.float32))
        self.store.put(record, activity, array_path)
        return record

    def _activity(self, r: dict, steps: dict, report_ms: float) -> list[dict]:
        d, q = r["diagnosis"], r["signal_quality"]
        s = d["scores"]
        rows = [("bundle_verified", "Bundle verified", self.bundle_ms,
                 f"manifest {self.bundle.manifest_sha256[:12]} — every file hash checked at startup"),
                ("signal_quality", "Signal quality", steps.get("signal_quality"),
                 f"speed {q['shaft_speed_hz']:.2f} Hz; {len(q['warnings'])} warning(s)"),
                ("models", "Models scored", steps.get("models"),
                 f"CNN-6 top {d['top_class']} (MSP {s['msp']:.4f}); LR {d['lr_secondary']['top_class']}; "
                 f"Mahalanobis {s['maha']:.2f}; AE {s['ae_error']:.3f}; B1s {s['b1s']:.3f}"),
                ("decision_rule", "Decision rule", steps.get("decision_rule"),
                 f"rule {d['rule_variant']} → {d['display_label']}"),
                ("physics", "Physics cursors and envelope spectrum", steps.get("physics"),
                 f"cursors at {q['shaft_speed_hz']:.2f} Hz"),
                ("explanations", "Explanations", steps.get("explanations"), "LR contributions; CNN map (advanced)"),
                ("report", "Report", report_ms, "fixed template from the evidence record")]
        return [{"step": k, "title": title, "ms": None if ms is None else round(float(ms), 1), "detail": det,
                 "kind": STEP_NOTE, "startup": k == "bundle_verified"} for k, title, ms, det in rows]

    def case(self, case_id: str) -> dict | None:
        c = self.store.get(case_id)
        return None if c is None else c["record"]

    def case_entry(self, case_id: str) -> dict | None:
        return self.store.get(case_id)

    def case_array(self, case_id: str) -> tuple[np.ndarray, float, str | None] | None:
        """(raw array, measured speed, rec_id) for the evidence views: dataset recordings from raw_npy (the same loader as
        diagnose), uploads from their stored array with the speed diagnose() measured."""
        from ml.inference import load_recording
        c = self.store.get(case_id)
        if c is None:
            return None
        rec = c["record"]
        fr = float(rec["signal_quality"]["shaft_speed_hz"])
        if rec["source"] == "dataset":
            x, _ = load_recording(rec["input"]["rec_id"])
            return x, fr, rec["input"]["rec_id"]
        if not c["array_path"] or not Path(c["array_path"]).exists():
            return None
        return np.load(c["array_path"]), fr, None
