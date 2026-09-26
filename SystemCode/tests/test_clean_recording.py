"""clean_recording: tachometer untouched, mic bursts removed, band-stop removes a tone and keeps the rest aligned."""
import numpy as np

from ml.config import BandStopConfig, PreprocessConfig
import pandas as pd

from ml.preprocess import apply_fir_centred, bandstop_taps, clean_recording, propose_stopband

FS = 50_000
N = 5 * FS
T = np.arange(N) / FS
# A test band-stop (the real edges are set from data in notebook 03).
BS = BandStopConfig(enabled=True, stop_lo_hz=21_850.0, stop_hi_hz=22_200.0, transition_hz=100.0, numtaps=2001,
                    window=("kaiser", 8.0))
_NO_LP = PreprocessConfig().lowpass.model_copy(update={"enabled": False})  # isolate the band-stop in these tests
CFG_ON = PreprocessConfig(bandstop=BS, lowpass=_NO_LP)
CFG_OFF = PreprocessConfig(bandstop=PreprocessConfig().bandstop.model_copy(update={"enabled": False}), lowpass=_NO_LP)


def _recording(seed=0):
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.normal(size=(N, 8))
    x[:, 0] = np.where((T * 30) % 1 < 0.12, 4.6, -0.7)  # tacho
    x[:, 1:] += 0.5 * np.sin(2 * np.pi * 22_028 * T)[:, None]  # the line on every channel
    x[:, 1:] += 0.3 * np.sin(2 * np.pi * 3_000 * T)[:, None]  # an in-band signal to keep
    x[10_000, 7] += 150.0  # one mic burst
    return x.astype(np.float32)


def _amp(sig, f):
    return 2 * np.abs(np.dot(sig, np.exp(-2j * np.pi * f * T[: sig.size]))) / sig.size


def test_tacho_untouched_and_dtype():
    x = _recording()
    y, rep = clean_recording(x, CFG_ON)
    assert np.array_equal(y[:, 0], x[:, 0])
    assert y.dtype == np.float32 and y.shape == x.shape
    assert rep["bandstop_applied"] and rep["bandstop_numtaps"] == 2001


def test_mic_burst_removed():
    y, rep = clean_recording(_recording(), CFG_OFF)
    assert rep["n_mic_bursts"] == 1 and np.abs(y[:, 7]).max() < 5


def test_bandstop_removes_line_keeps_signal_aligned():
    y, _ = clean_recording(_recording(), CFG_ON)
    core = slice(5_000, N - 5_000)  # away from the (numtaps − 1) / 2 = 1000-sample edges
    for c in range(1, 8):
        assert _amp(y[core, c], 22_028) < 0.5 * 10 ** (-60 / 20)  # ≥ 60 dB down
    # Noise-free check of alignment: an out-of-band sine comes through with the same amplitude and phase.
    ref = 0.3 * np.sin(2 * np.pi * 3_000 * T)
    x = np.zeros((N, 8), np.float32)
    x[:, 1:] = (ref + 0.5 * np.sin(2 * np.pi * 22_028 * T))[:, None]
    y, _ = clean_recording(x, CFG_ON)
    assert np.max(np.abs(y[core, 1:] - ref[core, None])) < 1e-3


def test_disabled_bandstop_is_identity_outside_mic():
    x = _recording()
    y, rep = clean_recording(x, CFG_OFF)
    assert not rep["bandstop_applied"]
    assert np.array_equal(y[:, :7], x[:, :7])


def test_centred_fir_has_no_delay():
    taps = bandstop_taps(CFG_ON)
    assert taps.size % 2 == 1 and np.allclose(taps, taps[::-1])
    impulse = np.zeros(20_001)
    impulse[10_000] = 1.0
    y = apply_fir_centred(impulse, taps)
    assert np.argmax(np.abs(y)) == 10_000


def test_propose_stopband_rule():
    tone = pd.DataFrame([
        {"f0_hz": 22_020.0, "snr_db": 60, "f-120_hz": 21_900.0, "snr-120_db": 30, "f-60_hz": 21_960.0,
         "snr-60_db": 30, "f+60_hz": 22_080.0, "snr+60_db": 30, "f+120_hz": 22_140.0, "snr+120_db": 30},
        {"f0_hz": 22_050.0, "snr_db": 60, "f-120_hz": 21_930.0, "snr-120_db": 30, "f-60_hz": 21_990.0,
         "snr-60_db": 30, "f+60_hz": 22_110.0, "snr+60_db": 30, "f+120_hz": 22_171.0, "snr+120_db": 5},  # weak
        {"f0_hz": 23_000.0, "snr_db": 5, "f-120_hz": 0, "snr-120_db": 0, "f-60_hz": 0, "snr-60_db": 0,
         "f+60_hz": 0, "snr+60_db": 0, "f+120_hz": 0, "snr+120_db": 0},  # line absent: ignored
    ])
    band = propose_stopband(tone, present_db=20, margin_hz=25, round_hz=5)
    assert (band["observed_lo_hz"], band["observed_hi_hz"]) == (21_900.0, 22_140.0)
    assert (band["stop_lo_hz"], band["stop_hi_hz"]) == (21_875.0, 22_165.0)


def test_lowpass_variant():
    from ml.config import load_config
    from ml.paths import CONFIGS_DIR

    cap = load_config(CONFIGS_DIR / "variants" / "r1cap10.yaml").preprocess
    old = load_config(CONFIGS_DIR / "variants" / "r1_bandstop_only.yaml").preprocess
    assert cap.lowpass.enabled and PreprocessConfig().lowpass.enabled and not old.lowpass.enabled
    assert cap == PreprocessConfig()  # the capped pipeline is the primary since 26 Sep 2026
    x = np.zeros((N, 8), np.float32)
    x[:, 0] = np.where((T * 30) % 1 < 0.12, 4.6, -0.7)
    keep = 0.3 * np.sin(2 * np.pi * 5_000 * T)
    x[:, 1:] = (keep + 0.3 * np.sin(2 * np.pi * 12_000 * T))[:, None]
    y, rep = clean_recording(x, cap)
    assert rep["lowpass_applied"] and np.array_equal(y[:, 0], x[:, 0])
    core = slice(5_000, N - 5_000)
    assert _amp(y[core, 1], 12_000) < 0.3 * 10 ** (-80 / 20) * 1.5  # ≥ ~76 dB down (probe-limited)
    assert np.max(np.abs(y[core, 1:] - keep[core, None])) < 1e-3  # 5 kHz kept, same amplitude and phase


def test_config_extends(tmp_path):
    from ml.config import load_config

    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\npreprocess:\n  mic:\n    burst_pad: 7\n")
    var = tmp_path / "v" / "var.yaml"
    var.parent.mkdir()
    var.write_text("extends: ../base.yaml\npreprocess:\n  features:\n    version: x\n")
    cfg = load_config(var)
    assert cfg.seed == 1 and cfg.preprocess.mic.burst_pad == 7 and cfg.preprocess.features.version == "x"
