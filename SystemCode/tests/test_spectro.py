"""Order-spectrogram and log-Mel prototypes on synthetic recordings."""
import numpy as np
import pytest

from ml.config import PreprocessConfig
from ml.spectro import frame_centres, log_mel, mel_filterbank, order_axis, recording_representation

CFG = PreprocessConfig()
FS, N = CFG.fs_hz, CFG.n_rows
T_ = np.arange(N) / FS


def _rec(fr, lines, tone_hz=None):
    x = 1e-3 * np.random.default_rng(0).normal(size=(N, 8))
    x[:, 0] = np.where((fr * T_) % 1.0 < 0.12, 4.6, -0.7)
    ur = CFG.channels.index("under_radial")
    for o, a in lines.items():
        x[:, ur] += a * np.sin(2 * np.pi * o * fr * T_)
    if tone_hz:
        x[:, 7] += 0.2 * np.sin(2 * np.pi * tone_hz * T_)
    return x.astype(np.float32)


@pytest.mark.parametrize("name", list(CFG.spectro.candidates))
@pytest.mark.parametrize("fr", [30.0, 60.0])
def test_line_lands_on_its_order(name, fr):
    rep = recording_representation(_rec(fr, {3: 0.2}), fr, CFG, name)
    o = order_axis(CFG)
    prof = rep["order"][:, CFG.spectro.channels.index("under_radial")].mean(axis=(0, 2))  # time-averaged dB
    k = int(np.argmax(prof))
    assert abs(o[k] - 3) / 3 < 0.03  # peak within one log-order bin of 3×
    assert abs(prof[k] - 20 * np.log10(0.2)) < 1.5  # amplitude within 1.5 dB (scalloping, interpolation)


def test_mel_frames_share_order_frame_times():
    for cand in CFG.spectro.candidates.values():
        c = frame_centres(N, CFG, cand.frames_per_window)
        assert c.shape == (9, cand.frames_per_window)
        assert np.all(np.diff(c, axis=1) > 0)
    rep = recording_representation(_rec(30.0, {1: 0.1}, tone_hz=3000.0), 30.0, CFG, "A_stft8192")
    assert rep["mel"].shape == (9, CFG.spectro.mel_n, 25) == rep["order"].shape[:1] + (128, 25)


def test_mel_tone_in_right_bin():
    _, centres = mel_filterbank(CFG)
    m = log_mel(np.sin(2 * np.pi * 3000.0 * T_), CFG, 25)
    k = int(np.argmax(m.mean(axis=(0, 2))))
    assert abs(centres[k] - 3000) < 0.1 * 3000
    assert centres.max() <= CFG.spectro.mel_fmax_hz


def test_order_axis_and_ceiling():
    o = order_axis(CFG)
    assert len(o) == 128 and np.isclose(o[0], 0.5) and np.isclose(o[-1], 150)
    assert CFG.spectro.order_max * CFG.spectro.speed_ceiling_hz < CFG.lowpass.pass_hz


def test_to_order_bins_max_and_interp():
    from ml.spectro import to_order_bins

    axis = np.arange(0, 10.0, 0.5)  # fine points
    amp = np.zeros((1, axis.size, 1))
    amp[0, 7, 0] = 5.0  # a line at 3.5
    centres = np.array([[1.0, 3.4, 6.0]])
    edges = np.array([[0.8, 1.2, 4.0, 8.0]])  # bin 0 holds only point 1.0; bin 1 holds 1.5 .. 3.5; bin 2 4.0 .. 7.5
    out = to_order_bins(amp, axis, centres, edges)[0, :, 0]
    assert out[1] == 5.0 and out[2] == 0.0 and out[0] == 0.0
    empty = to_order_bins(amp, axis, np.array([[3.6]]), np.array([[3.55, 3.65]]))[0, 0, 0]  # no fine point inside
    assert np.isclose(empty, 5.0 * (4.0 - 3.6) / 0.5)  # interpolated between 3.5 (5.0) and 4.0 (0.0)
