"""Typed configuration: dataset facts, preprocessing settings, YAML loading and config hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Recordings per class10 label in the MAFAULDA CSV tree (CLAUDE.md §3); indexing fails loudly on mismatch.
EXPECTED_CLASS_COUNTS: dict[str, int] = {
    "normal": 49,
    "imbalance": 333,
    "horizontal_misalignment": 197,
    "vertical_misalignment": 301,
    "underhang_ball": 186,
    "underhang_cage": 188,
    "underhang_outer": 184,
    "overhang_ball": 137,
    "overhang_cage": 188,
    "overhang_outer": 188,
}
EXPECTED_TOTAL_FILES = 1951


class _Strict(BaseModel):
    """Base for config sections: unknown keys are errors, so a YAML typo can't be silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BearingRatios(_Strict):
    """Bearing defect frequencies as multiples of shaft rate (orders)."""

    bpfo: float = 2.998
    bpfi: float = 5.002
    bsf: float = 1.871  # the defect line to look for is 2 × BSF
    ftf: float = 0.375


class PreprocessConfig(_Strict):
    """Signal facts and preprocessing settings; its sha256 names every derived cache."""

    fs_hz: int = 50_000
    n_rows: int = 250_000  # 5 s at 50 kHz
    n_channels: int = 8
    channels: tuple[str, ...] = (
        "tacho",
        "under_axial", "under_radial", "under_tangential",
        "over_axial", "over_radial", "over_tangential",
        "mic",
    )
    # Measured shaft speed ≈ nominal (file name) × this; used to centre speed searches.
    measured_speed_ratio: float = 0.977
    bearing: BearingRatios = Field(default_factory=BearingRatios)
    expected_class_counts: dict[str, int] = Field(default_factory=lambda: dict(EXPECTED_CLASS_COUNTS))
    expected_total_files: int = EXPECTED_TOTAL_FILES


class PathsConfig(_Strict):
    """Locations. Relative paths resolve against SystemCode/; env vars override (see ml.paths)."""

    data_root: str = "../mafaulda"
    mlflow_uri: str = "sqlite:///mlruns.db"


class Config(_Strict):
    """Root config as loaded from configs/*.yaml."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config; missing keys take the defaults above."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


def config_sha256(section: BaseModel) -> str:
    """sha256 of a config section's canonical JSON (sorted keys), so equal settings always hash equally."""
    payload = json.dumps(section.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
