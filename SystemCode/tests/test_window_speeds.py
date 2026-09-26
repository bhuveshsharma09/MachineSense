"""window_speeds on synthetic tacho pulse trains: constant, chirped, dead and partly dead."""
import numpy as np

from ml.config import PreprocessConfig
from ml.preprocess import window_speeds

CFG = PreprocessConfig()
FS = CFG.fs_hz
N = CFG.n_rows


def _tacho(f0: float, f1: float, dead_from_s: float | None = None) -> np.ndarray:
    """Once-per-rev square pulse (12 % duty, −0.7 → 4.6 V) whose speed ramps linearly from f0 to f1 Hz over 5 s."""
    t = np.arange(N) / FS
    phase = f0 * t + (f1 - f0) * t**2 / (2 * t[-1])  # revolutions
    x = np.where(phase % 1.0 < 0.12, 4.6, -0.7)
    if dead_from_s is not None:
        x[t >= dead_from_s] = -0.7
    return x + 0.05 * np.random.default_rng(0).normal(size=N)


def _true_mean_speed(f0, f1, s0, s1):
    return f0 + (f1 - f0) * ((s0 + s1) / 2) / (N / FS)


def test_constant_speed():
    sp, fb = window_speeds(_tacho(30.0, 30.0), FS, cfg=CFG)
    assert sp.size == 9 and fb == 0
    assert np.allclose(sp, 30.0, rtol=2e-3)


def test_chirp_is_tracked():
    f0, f1 = 29.0, 31.0
    sp, fb = window_speeds(_tacho(f0, f1), FS, cfg=CFG)
    assert fb == 0
    assert np.all(np.diff(sp) > 0)  # speed rises window by window
    true = [_true_mean_speed(f0, f1, 0.5 * i, 0.5 * i + 1.0) for i in range(9)]
    assert np.allclose(sp, true, rtol=3e-3)
    range_pct = 100 * (sp.max() - sp.min()) / sp.mean()
    assert 4.5 < range_pct < 6.5  # windows centred 0.5 s … 4.5 s span 29.2 … 30.8 Hz


def test_dead_tacho_falls_back_everywhere():
    x = np.full(N, -0.7) + 0.01 * np.random.default_rng(1).normal(size=N)
    sp, fb = window_speeds(x, FS, cfg=CFG, fr_fallback_hz=30.0)
    assert fb == 9 and np.all(sp == 30.0)


def test_partly_dead_tacho_counts_fallbacks():
    sp, fb = window_speeds(_tacho(30.0, 30.0, dead_from_s=3.0), FS, cfg=CFG, fr_fallback_hz=29.5)
    # Windows starting at 3.0, 3.5 and 4.0 s lie entirely in the dead part; the one at 2.5 s still has ~15 intervals.
    assert fb == 3
    assert np.allclose(sp[:6], 30.0, rtol=2e-3) and np.all(sp[-3:] == 29.5)
