"""window(): shapes, boundaries, DC removal; window speeds line up with the windows and flag fallbacks."""
import numpy as np
import pytest

from ml.config import PreprocessConfig
from ml.preprocess import window, window_speeds, window_speeds_detail, window_starts

CFG = PreprocessConfig()
FS = CFG.fs_hz
N = CFG.n_rows
WIN, HOP = FS, FS // 2


def _rec(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 8)).astype(np.float32)
    x[:, 1:] += np.linspace(-2, 2, N, dtype=np.float32)[:, None]  # a slow drift, so every window has its own DC
    return x


def test_shape_and_starts():
    w = window(_rec(), CFG)
    assert w.shape == (9, WIN, 8) and w.dtype == np.float32
    assert np.array_equal(window_starts(N, CFG), np.arange(9) * HOP)


def test_boundaries_match_the_recording():
    x = _rec()
    w = window(x, CFG)
    for i in (0, 4, 8):
        seg = x[i * HOP: i * HOP + WIN].astype(np.float64)
        assert np.allclose(w[i, :, 1:], seg[:, 1:] - seg[:, 1:].mean(axis=0), atol=1e-5)
    assert (8 * HOP + WIN) == N  # the last window ends exactly at the end


def test_dc_removed_except_tacho():
    x = _rec()
    w = window(x, CFG)
    assert np.abs(w[:, :, 1:].mean(axis=1, dtype=np.float64)).max() < 1e-5
    assert np.array_equal(w[3, :, 0], x[3 * HOP: 3 * HOP + WIN, 0])


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        window(np.zeros((N - 1000, 8), np.float32), CFG)


def test_speed_fallback_flags_line_up_with_windows():
    t = np.arange(N) / FS
    tacho = np.where((30.0 * t) % 1.0 < 0.12, 4.6, -0.7)
    tacho[t >= 3.0] = -0.7  # dead from 3 s: windows starting at 3.0, 3.5, 4.0 s fall back
    speeds, flags = window_speeds_detail(tacho, FS, cfg=CFG, fr_fallback_hz=29.0)
    assert speeds.size == window_starts(N, CFG).size
    assert flags.tolist() == [False] * 6 + [True] * 3
    assert np.all(speeds[flags] == 29.0) and np.allclose(speeds[~flags], 30.0, rtol=2e-3)
    assert window_speeds(tacho, FS, cfg=CFG, fr_fallback_hz=29.0)[1] == 3
