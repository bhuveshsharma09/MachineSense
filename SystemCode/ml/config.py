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


class SpeedConfig(_Strict):
    """Shaft-speed estimation (ml.preprocess.estimate_speed)."""

    hyst_low: float = 0.3  # hysteresis thresholds as fractions between tacho low (p10) and high (p99) levels
    hyst_high: float = 0.7
    min_interval_ratio: float = 0.5  # pulse intervals shorter than this × median are double triggers
    interval_tol: float = 0.05  # an interval (per revolution) this far from the median period is a glitch
    min_good_fraction: float = 0.8  # tacho edges are unusable if fewer intervals than this are clean
    search_halfwidth: float = 0.05  # search ±5 % around fr_file_hz × measured_speed_ratio
    radial_channel: int = 2  # underhang radial accelerometer for the harmonic comb
    comb_harmonics: int = 10
    fft_zero_pad: int = 8  # zero-padding factor: 0.2 Hz bins at 5 s → 0.025 Hz
    min_peak_ratio: float = 10.0  # tacho FFT peak must exceed the median magnitude over 0.5–1.5 × speed by this factor
    # Decision 26 Sep 2026: the order axis uses per-window speed (window_speeds on the untouched tachometer).
    order_axis: str = "window"
    window_min_intervals: int = 3  # window_speeds: fewer clean tacho intervals in a window → recording speed
    confirm_tol: float = 0.002  # two methods confirm each other within 0.2 % (typical agreement is < 0.05 %)
    disagree_tol: float = 0.01  # flag when methods differ by more than 1 %


class QualityConfig(_Strict):
    """Thresholds that turn quality numbers into flags. Flags are for auditing only, never model inputs."""

    rail_min: int = 10  # samples exactly at a channel's max/min; a clean channel has ~2
    spike_z: float = 100.0  # robust z-score of the largest sample; accelerometers stay below ~50
    dc_abs: float = 0.2  # |mean| in channel units


class MicConfig(_Strict):
    """Microphone burst removal (CLAUDE.md §5 rule 10)."""

    # 1.25 × the max whole-recording robust z of split_closed TRAIN normal mics (14.57); notebook 02 checks it.
    burst_z: float = 18.2
    # Samples added on each side of a flagged run before interpolation. Notebook 02: burst-aligned |z| of train+val
    # flagged runs stays > 1.2 × baseline for 15 samples before and after each run (symmetric filter ringing).
    burst_pad: int = 15


class BandStopConfig(_Strict):
    """Fixed band-stop for the ~22,028 Hz line (clean_recording, channels 1–7). Linear-phase FIR (scipy firwin),
    applied by centred convolution, so its group delay is compensated. Edges set in notebook 03 (Stage 2A item 2)."""

    # Set from split_closed TRAIN NORMAL recordings (scripts/06_bandstop.py measure): the line and its ±60/±120 Hz
    # sidebands span 21,899.1–22,166.3 Hz; + 25 Hz margin each side, rounded outward to 5 Hz.
    enabled: bool = True
    stop_lo_hz: float = 21870.0  # attenuation is guaranteed from stop_lo_hz to stop_hi_hz
    stop_hi_hz: float = 22195.0
    transition_hz: float = 200.0  # width of each transition band outside the stop band
    numtaps: int = 1257  # scipy.signal.kaiserord(80 dB, 200 Hz): odd, 12.6 ms half-length
    window: tuple[str, float] = ("kaiser", 7.8573)  # Kaiser beta for 80 dB


class SensorSpec(_Strict):
    """Manufacturer ratings of one accelerometer model (Hz)."""

    range_3db_hz: tuple[float, float]
    resonance_hz: float
    source: str


def _default_sensors() -> dict[str, SensorSpec]:
    # MAFAULDA lists both models but does not say which one sits on which bearing.
    return {
        "601A01": SensorSpec(range_3db_hz=(0.27, 10000.0), resonance_hz=16000.0,
                             source="PCB/IMI 601A01 spec sheet, Revision G, ECN 14824"),
        "604B31": SensorSpec(range_3db_hz=(0.5, 5000.0), resonance_hz=10000.0,
                             source="PCB/IMI 604B31 spec sheet, Revision B, ECN 49214"),
    }


class SpectralKurtosisConfig(_Strict):
    """Fixed band grid for spectral kurtosis (the demodulation-band search)."""

    widths_hz: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0)
    min_lo_hz: float = 1000.0  # never demodulate below 1 kHz


class BandSelectionConfig(_Strict):
    """Parameters of ml.features.select_demod_band (the rule itself is in its docstring)."""

    channels: tuple[str, ...] = ("under_radial", "over_radial")
    fault_classes: tuple[str, ...] = ("underhang_outer", "overhang_outer")  # outer race: the clearest impacts
    severities: tuple[float, ...] = (0.0,)  # no added imbalance mass, so impacts are not masked
    reference_class: str = "normal"  # contrast reference: healthy recordings of the same training data
    min_recordings: int = 5  # per fault class and for the reference, or that group is not used


class EnvelopeConfig(_Strict):
    """Envelope analysis: spectral-kurtosis grid and the chosen demodulation band per radial channel."""

    sk: SpectralKurtosisConfig = Field(default_factory=SpectralKurtosisConfig)
    selection: BandSelectionConfig = Field(default_factory=BandSelectionConfig)
    bands_hz: dict[str, tuple[float, float]] = Field(default_factory=lambda: {
        # Closed-track band (PROVISIONAL): ml.features.select_demod_band on split_closed train. Other tracks
        # have their own band in results/demod_bands.json; feature code must read the band per track there.
        "under_radial": (9000.0, 11000.0),
        "over_radial": (13000.0, 17000.0),
    })
    min_band_hz: float = 2000.0  # envelope must carry 12 orders at 61 Hz (≥ 1.46 kHz)
    bands_provisional: bool = True  # bands above the rated sensor range await a decision (audit report §9)
    sensors: dict[str, SensorSpec] = Field(default_factory=_default_sensors)
    # Healthy-only track (anomaly detectors): fixed a priori, never from recordings. The widest band that is
    # inside the ±3 dB range of BOTH sensor models (the model-to-channel mapping is undocumented), starts at the
    # 1 kHz floor (clear of 12 orders at 61 Hz) and is at least min_band_hz wide: 1–5 kHz, for every channel.
    healthy_band_hz: tuple[float, float] = (1000.0, 5000.0)
    # Decision 26 Sep 2026: the primary envelope band for EVERY track, from the sensor ratings (same rule as above).
    # The per-track bands (bands_hz, results/demod_bands.json) are a separately reported comparison only.
    primary_band_hz: tuple[float, float] = (1000.0, 5000.0)
    max_order: float = 12.0
    order_step: float = 0.005


class CombConfig(_Strict):
    """Envelope-order comb definitions for bearing evidence (orders = multiples of shaft speed)."""

    ftf_harmonics: tuple[int, ...] = (1, 2, 3, 4, 5)
    bsf_multiple: float = 2.0  # the ball defect line is 2 × BSF
    bsf_ftf_sidebands: tuple[int, ...] = (-1, 0, 1)
    bpfo_harmonics: tuple[int, ...] = (1, 2, 3)
    bpfi_harmonics: tuple[int, ...] = (1, 2)
    bpfi_shaft_sidebands: tuple[int, ...] = (-1, 0, 1)
    tol_orders: float = 0.02  # a line's peak is searched within ±tol
    bg_halfwidth_orders: float = 0.3  # background = median within ±bg_halfwidth, excluding the peak zone


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
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    mic: MicConfig = Field(default_factory=MicConfig)
    bandstop: BandStopConfig = Field(default_factory=BandStopConfig)
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    comb: CombConfig = Field(default_factory=CombConfig)
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
