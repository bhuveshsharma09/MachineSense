"""R1 physics features on synthetic recordings at 12 Hz and 60 Hz: known order amplitudes and envelope lines."""
import numpy as np
import pytest

from ml.config import PreprocessConfig
from ml.features import (ACCEL_CHANNELS, band_analytic, envelope_line_names, feature_columns, feature_dictionary,
                         line_peak, line_tolerance_hz, recording_features, window_spectrum)

CFG = PreprocessConfig()
FS = CFG.fs_hz
N = CFG.n_rows
T = np.arange(N) / FS
ORDER_AMPS = {0.5: 0.05, 1: 0.4, 2: 0.2, 3: 0.1, 5: 0.05, 10: 0.03}  # V, on under_radial
CARRIER_HZ, CARRIER_V, MOD = 3000.0, 0.3, 0.5  # over_radial: carrier amplitude-modulated at BPFO


def _recording(fr: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.normal(size=(N, 8))
    x[:, 0] = np.where((fr * T) % 1.0 < 0.12, 4.6, -0.7)
    ur = CFG.channels.index("under_radial")
    for o, a in ORDER_AMPS.items():
        x[:, ur] += a * np.sin(2 * np.pi * o * fr * T + o)
    orad = CFG.channels.index("over_radial")
    x[:, orad] += CARRIER_V * (1 + MOD * np.cos(2 * np.pi * CFG.bearing.bpfo * fr * T)) * np.cos(2 * np.pi * CARRIER_HZ * T)
    return x.astype(np.float32)


@pytest.mark.parametrize("fr", [12.0, 60.0])
def test_order_amplitudes_recovered(fr):
    df = recording_features(_recording(fr), fr, CFG)
    assert len(df) == 9 and not df.speed_fallback.any()
    assert np.allclose(df.speed_hz, fr, rtol=2e-3)
    for o, a in ORDER_AMPS.items():
        name = f"f_under_radial_o{o:g}".replace(".", "p")
        assert np.allclose(df[name], a, rtol=0.03), (name, df[name].tolist())  # within 3 %
    assert np.allclose(df["f_under_radial_ratio_2x_1x"], 0.5, rtol=0.05)
    assert (df["f_under_radial_o4"] < 0.01).all()  # an order with no line stays at the noise level


@pytest.mark.parametrize("fr", [12.0, 60.0])
def test_envelope_line_recovered(fr):
    x = _recording(fr)
    # Raw envelope line amplitude (primary band) on one window: carrier × modulation depth.
    lo, hi = CFG.envelope.primary_band_hz
    env = np.abs(band_analytic(x[:, CFG.channels.index("over_radial")], FS, lo, hi))[FS: 2 * FS]
    f, a = window_spectrum(env, FS, CFG.features.zero_pad)
    f_line = CFG.bearing.bpfo * fr
    assert line_peak(f, a, f_line, line_tolerance_hz(f_line, CFG)) == pytest.approx(CARRIER_V * MOD, rel=0.03)
    # Feature: the BPFO line stands far above the envelope median; unrelated lines do not.
    df = recording_features(x, fr, CFG)
    assert (df["f_env_over_radial_bpfo1"] > 50).all()
    assert (df["f_env_over_radial_ftf1"] < 10).all()


def test_feature_naming_and_dictionary():
    df = recording_features(_recording(30.0), 30.0, CFG)
    cols = feature_columns(df)
    assert all(c.startswith("f_") for c in cols)
    assert {"window_idx", "speed_hz", "speed_fallback"}.isdisjoint(cols)
    d = feature_dictionary(CFG)
    assert list(d.name) == cols  # same names, same order
    assert d.name.is_unique and d[["unit", "definition", "channel"]].notna().all().all()
    assert np.isfinite(df[cols].to_numpy()).all()
    n_env = len(envelope_line_names(CFG)) * len(CFG.features.envelope_channels)
    assert len(cols) == len(ACCEL_CHANNELS) * (4 + len(CFG.features.accel_orders) + 1) + 2 + n_env + 3 + len(
        CFG.features.mic_bands_hz)


def test_tolerance_rule():
    assert line_tolerance_hz(12.0, CFG) == pytest.approx(1.5)  # bins dominate at low frequency
    assert line_tolerance_hz(600.0, CFG) == pytest.approx(3.0)  # relative term dominates at high orders
