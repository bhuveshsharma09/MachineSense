"""The evidence record (Stage App-1): a typed mirror of ml.inference.diagnose() plus case_id, created_at and source.
Pydantic v2, extra="forbid" at every level, so any key the engine adds or drops fails validation instead of passing
silently. This record is the contract the later agents will read; every number in it comes from the engine."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class InputInfo(_Strict):
    rec_id: str | None
    origin: str


class SignalQuality(_Strict):
    shaft_speed_hz: float
    window_speeds_hz: list[float]
    window_speed_fallback: list[bool]
    speed_range_pct: float
    speed_stable: bool
    train_speed_range_hz: list[float]
    calibration_normal_speed_range_hz: list[float]
    tacho_glitches: int
    mic_bursts_removed: int
    nan_count: int
    clipping_channels: list[str]
    spike_channels: list[str]
    warnings: list[str]


class Scores(_Strict):
    msp: float
    maha: float
    ae_error: float
    b1s: float
    ae_channel_errors: dict[str, float]


class Thresholds(_Strict):
    tau_p: float
    tau_d: float
    tau_e: float
    tau_b: float


class LrSecondary(_Strict):
    top_class: str
    probabilities: dict[str, float]


class Diagnosis(_Strict):
    model_name: str
    class10_probabilities: dict[str, float]
    top_class: str
    class6_probabilities: dict[str, float]
    class6_mapping: dict[str, str]
    decision: Literal["healthy", "known_fault", "unknown"]
    fault_class: str | None
    display_label: str
    caution: Literal["b1s_disagrees", "ae_disagrees", "both"] | None
    rule_variant: Literal["A", "B", "C", "D", "B_with_caution"]
    ae_override: bool
    rule_comparison: dict[Literal["A", "B", "C", "D"], str]
    scores: Scores
    thresholds: Thresholds
    lr_secondary: LrSecondary
    rule_status: str | None


class BearingFrequencies(_Strict):
    bpfo: float
    bpfi: float
    bsf: float
    bsf_defect_line_2x: float
    ftf: float


class EnvelopeSpectrum(_Strict):
    band_hz: list[float]
    orders: list[float]
    channels: dict[str, list[float]]
    note: str


class Physics(_Strict):
    shaft_speed_hz: float
    shaft_speed_rpm: float
    bearing_frequencies_hz: BearingFrequencies
    envelope_spectrum: EnvelopeSpectrum
    cursors_orders: dict[str, list[float]]
    cursors_hz: dict[str, list[float]]


class HeatMap(_Strict):
    channels: list[str]
    order_groups: list[list[float]]
    values: list[list[float]]


class ChannelShare(_Strict):
    channel: str
    share: float


class OrderBand(_Strict):
    orders: list[float]
    share: float


class GradcamProfile(_Strict):
    order_segments: list[list[float]]
    values: list[float]


class CnnExplanation(_Strict):
    class_: str = Field(alias="class")
    label: str
    display_priority: Literal["advanced", "primary"]
    method: str
    heat_map: HeatMap
    top_channels: list[ChannelShare]
    top_order_bands: list[OrderBand]
    gradcam_order_profile: GradcamProfile


class LrFeature(_Strict):
    feature: str
    contribution: float
    transformed_value: float
    definition: str | None
    unit: str | None
    channel: str | None


class LrExplanation(_Strict):
    label: str
    display_priority: Literal["advanced", "primary"]
    method: str
    class_: str = Field(alias="class")
    top_features: list[LrFeature]


class Xai(_Strict):
    note: str
    cnn: CnnExplanation
    lr: LrExplanation


class Provenance(_Strict):
    git_sha: str
    bundle_git_sha: str
    manifest_sha256: str
    model_sha256: dict[str, str]
    preprocess_config_sha256: str
    bundle_preprocess_config_sha256: str
    split_hash: str
    schema_version: str


class Timing(_Strict):
    total: float


class EvidenceRecord(_Strict):
    """What POST /api/diagnose returns and GET /api/cases/{case_id} serves."""
    case_id: str
    created_at: str
    source: Literal["dataset", "upload"]
    input: InputInfo
    signal_quality: SignalQuality
    diagnosis: Diagnosis
    physics: Physics
    xai: Xai | None
    limitations: list[str]
    provenance: Provenance
    timing_ms: Timing

    def to_json(self) -> dict:
        """The record as plain JSON types, with the engine's own key names (e.g. 'class')."""
        return self.model_dump(mode="json", by_alias=True)


# Fields that exist only at run time and are excluded when the API record is compared with diagnose() output.
RUNTIME_ONLY_FIELDS = ("case_id", "created_at", "source", "timing_ms")
