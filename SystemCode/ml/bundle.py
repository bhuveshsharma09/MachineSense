"""manifest.json schema for the model bundle the app loads. Stub: fields firm up by checkpoint 2 (18 Oct)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelEntry(BaseModel):
    """One model file in the bundle; weights are state_dict / safetensors, never pickled modules."""

    model_config = ConfigDict(extra="forbid")

    name: str  # e.g. "cnn_scratch", "autoencoder", "lightgbm"
    role: str  # "classifier" | "anomaly" | "openset" | "calibrator"
    file: str  # path relative to the bundle directory
    sha256: str
    representation: str  # e.g. "R1_physics", "R2_order_spectrogram"
    config_sha256: str
    mlflow_run_id: str | None = None


class Manifest(BaseModel):
    """Everything inference needs to reproduce a verdict, pinned by hashes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    created_at: str  # ISO 8601
    git_sha: str
    preprocess_config_sha256: str
    split_hash: str
    fs_hz: int
    channels: list[str]
    class10: list[str]
    class6: list[str]
    models: list[ModelEntry]
    thresholds: dict[str, float] = {}  # e.g. anomaly and open-set rejection thresholds, from validation only
