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


class FeatureConfig(_Strict):
    """R1 physics features (ml.features.physics_features). Spectral lines are searched within ±tolerance, where
    tolerance_hz = max(order_tol_bins × bin spacing, order_tol_rel × line frequency); bin spacing = 1 / win_s
    (1 Hz for 1 s windows). Zero-padding interpolates the spectrum only; it adds no resolution."""

    version: str = "r1cap10-1.0"  # bump whenever a definition changes; part of every feature cache's identity
    order_tol_bins: float = 1.5  # Hann main lobe ±2 bins; ±1.5 bins keeps neighbouring integer orders apart ≥ 12 Hz
    order_tol_rel: float = 0.005  # speed uncertainty and within-window drift grow with order
    zero_pad: int = 4
    accel_orders: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    envelope_channels: tuple[str, ...] = ("under_radial", "over_radial")  # one radial channel per bearing
    envelope_ref_orders: tuple[float, float] = (0.2, 12.0)  # median of the envelope spectrum here = the reference
    mic_max_hz: float = 10000.0  # the primary pipeline is capped at 10 kHz (the band-stop-only record used 20 kHz)
    mic_rolloff: float = 0.85
    mic_bands_hz: tuple[tuple[float, float], ...] = (
        (20.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0), (4000.0, 8000.0),
        (8000.0, 10000.0))


class SpectroCandidate(_Strict):
    """One order-spectrogram candidate (Stage 2C-1 prototypes). Frames: T per 1 s window, centred at
    window_start + (i + 0.5) · win_s / T, cut from the whole cleaned recording (reflect-padded at its ends only).
    kind "stft": analysis window of win_samples, FFT length nfft (zero-padding interpolates only), mapped to orders
    with the window's speed. kind "order": resampled to samples_per_rev per revolution along the shaft angle (speed
    profile interpolated between window speeds), frames of n_revs revolutions, FFT length n_revs·samples_per_rev·zero_pad;
    a speed-dependent anti-alias low-pass at aa_orders × speed is applied first whenever that is below the band limit."""

    kind: str  # "stft" | "order"
    frames_per_window: int  # T
    window: str = "hann"
    win_samples: int = 0  # stft
    nfft: int = 0  # stft
    n_revs: float = 0.0  # order
    samples_per_rev: int = 0  # order
    zero_pad: int = 2  # order
    aa_orders: float = 0.0  # order: anti-alias cutoff in orders
    aa_atten_db: float = 60.0


class SpectroConfig(_Strict):
    """Order-spectrogram prototypes (ml.spectro). Fixed order axis for every speed: n_orders log-spaced bins from
    order_min to order_max. Declared operating-speed ceiling speed_ceiling_hz: order_max × ceiling = 9.3 kHz stays below
    the 9.8 kHz pass-band edge of the primary low-pass."""

    n_orders: int = 128
    order_min: float = 0.5
    order_max: float = 150.0
    speed_ceiling_hz: float = 62.0
    eps: float = 1e-9  # dB floor: 20·log10(amplitude + eps)
    channels: tuple[str, ...] = ("under_axial", "under_radial", "under_tangential", "over_axial", "over_radial",
                                 "over_tangential")
    candidates: dict[str, SpectroCandidate] = Field(default_factory=lambda: {
        "A_stft8192": SpectroCandidate(kind="stft", frames_per_window=25, win_samples=8192, nfft=16384),
        "B_stft32768": SpectroCandidate(kind="stft", frames_per_window=10, win_samples=32768, nfft=65536),
        "C_order8rev": SpectroCandidate(kind="order", frames_per_window=25, n_revs=8.0, samples_per_rev=512,
                                        zero_pad=2, aa_orders=205.0),
    })
    # Microphone log-Mel (Hz axis, cleaned mic): frame centres identical to each candidate's.
    mel_n: int = 128
    mel_fmin_hz: float = 20.0
    mel_fmax_hz: float = 10000.0
    mel_win_samples: int = 2048
    mel_nfft: int = 2048
    mel_eps: float = 1e-12  # 10·log10(power + eps)


class WindowConfig(_Strict):
    """Windowing after whole-recording cleaning (ml.preprocess.window)."""

    win_s: float = 1.0
    hop_s: float = 0.5
    n_windows: int = 9  # expected per 5 s recording; anything else is an error


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


class LowpassConfig(_Strict):
    """Fixed low-pass on channels 1–7 in clean_recording. PRIMARY since 26 Sep 2026 (pipeline r1cap10): a pragmatic
    bandwidth limit that reduces exposure to the identified high-frequency interference; it does not guarantee
    calibrated measurements (the 604B31 is rated to 5 kHz) or remove all artefacts. Linear-phase Kaiser FIR (firwin),
    cutoff midway between the pass-band and stop-band edges, applied by centred convolution."""

    enabled: bool = True
    pass_hz: float = 9800.0
    stop_hz: float = 10200.0
    # Requirement: ≥ 80 dB stop band, ≤ 0.01 dB pass-band ripple. kaiserord(80 dB) fell just short (79.73 dB measured),
    # so the design target is 81 dB: kaiserord(81 dB, 400 Hz) → 639 taps, β 7.9675; measured 80.75 dB, 0.0008 dB.
    atten_db: float = 81.0
    numtaps: int = 639  # odd; 6.4 ms half-length
    window: tuple[str, float] = ("kaiser", 7.9675)


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
        # Closed-track COMPARISON band (decided 26 Sep 2026: primary_band_hz is used for every track):
        # ml.features.select_demod_band on split_closed train. Other tracks
        # have their own band in results/demod_bands.json; feature code must read the band per track there.
        "under_radial": (9000.0, 11000.0),
        "over_radial": (13000.0, 17000.0),
    })
    min_band_hz: float = 2000.0  # envelope must carry 12 orders at 61 Hz (≥ 1.46 kHz)
    bands_provisional: bool = True  # the per-track bands are a comparison only, not the pipeline's band
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
    lowpass: LowpassConfig = Field(default_factory=LowpassConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    spectro: SpectroConfig = Field(default_factory=SpectroConfig)
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    comb: CombConfig = Field(default_factory=CombConfig)
    expected_class_counts: dict[str, int] = Field(default_factory=lambda: dict(EXPECTED_CLASS_COUNTS))
    expected_total_files: int = EXPECTED_TOTAL_FILES


class PathsConfig(_Strict):
    """Locations. Relative paths resolve against SystemCode/; env vars override (see ml.paths)."""

    data_root: str = "../mafaulda"
    mlflow_uri: str = "sqlite:///mlruns.db"


class BaselineConfig(_Strict):
    """Closed-set classical baselines (scripts/05_baselines.py). Fixed hyperparameters, stated before training; one
    seed; no tuning. Scalers and any learned preprocessing are fitted on training data only."""

    seed: int = 42
    feature_table: str = "r1cap10"  # data/features/<feature_table>_<key>.parquet (primary since 26 Sep 2026)
    sensitivity_table: str = "r1"  # the band-stop-only table, kept for the record
    sensitivity_config: str = "variants/r1_bandstop_only.yaml"  # relative to configs/
    class_weight: str | None = "balanced"  # LightGBM, random forest, SVM, logistic regression (macro-F1 target)
    # Scaled models (SVM, k-NN, logistic regression): QuantileTransformer to a normal distribution, fitted on train.
    quantile_n: int = 1000
    lgbm: dict = Field(default_factory=lambda: {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0})
    rf: dict = Field(default_factory=lambda: {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2})
    svm: dict = Field(default_factory=lambda: {"C": 10.0, "gamma": "scale"})
    svm_calibration: dict = Field(default_factory=lambda: {"method": "sigmoid", "n_group_folds": 3})
    knn: dict = Field(default_factory=lambda: {"n_neighbors": 15, "weights": "distance"})
    logreg: dict = Field(default_factory=lambda: {"C": 1.0, "max_iter": 3000})
    ensemble: tuple[str, ...] = ("lgbm", "rf", "svm", "knn", "logreg")  # equal weights, soft vote
    prob_floor: float = 1e-6  # window probabilities are clipped to [prob_floor, 1] before log-averaging
    n_bootstrap: int = 2000
    n_permutation_repeats: int = 5
    leaky_n_folds: int = 5


class DevelopmentConfig(_Strict):
    """Folder-grouped development evaluation of the classical baselines (Stage 2B.1). Rule, written before running:
    a recording is in the development pool only if split_closed ∈ allowed_closed AND none of the reserved_columns
    equals "test" (split_closed test, split_severity test and split_speed_block test are reserved and never used).
    A folder is class10 × severity: a PROXY for an acquisition session."""

    allowed_closed: tuple[str, ...] = ("train", "val")
    reserved_columns: tuple[str, ...] = ("split_severity", "split_speed_block")
    seed: int = 42
    # Grouped folds (ml.splits.assign_dev_folds): K = min number of eligible folder proxies over FAULT classes
    # (normal excluded); infeasible below min_k.
    min_k: int = 2
    folds_file: str = "dev_folds.csv"  # results/<folds_file>: rec_id, class10, folder, dev_fold


class Config(_Strict):
    """Root config as loaded from configs/*.yaml."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)  # a variant lists only what it changes
    return _deep_merge(_load_raw((path.parent / parent).resolve()), raw) if parent else raw


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config; missing keys take the defaults above. A file with `extends: <path>` starts
    from that file (relative to itself) and overrides only the keys it lists."""
    return Config.model_validate(_load_raw(Path(path)))


def config_sha256(section: BaseModel) -> str:
    """sha256 of a config section's canonical JSON (sorted keys), so equal settings always hash equally."""
    payload = json.dumps(section.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
