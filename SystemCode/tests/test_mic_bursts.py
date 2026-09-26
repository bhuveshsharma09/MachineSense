"""remove_mic_bursts: bursts removed, clean signals untouched, edges handled."""
import numpy as np
import pytest

from ml.preprocess import remove_mic_bursts

FS = 50_000
Z = 18.2
N = FS  # 1 s


def _mic(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    return (0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.normal(size=N)).astype(np.float32)


def _robust_peak_z(x):
    med = np.median(x)
    return np.abs(x - med).max() / (1.4826 * np.median(np.abs(x - med)))


def test_bursts_removed():
    clean = _mic()
    x = clean.copy()
    starts = [5_000, 20_000, 33_333]
    for s, amp in zip(starts, [80.0, -150.0, 40.0]):
        x[s: s + 6] += amp * np.array([0.3, 1.0, -0.8, 0.5, -0.2, 0.1], dtype=np.float32)
    out, n = remove_mic_bursts(x, Z, pad=5)
    assert n == len(starts)
    assert _robust_peak_z(out) < Z
    # Outside the widened bursts nothing changes; inside, the fill stays at signal scale.
    untouched = np.ones(N, bool)
    for s in starts:
        untouched[s - 5: s + 6 + 5] = False
    assert np.array_equal(out[untouched], x[untouched])
    assert np.abs(out).max() < 1.0


def test_clean_signal_unchanged():
    x = _mic(1)
    out, n = remove_mic_bursts(x, Z)
    assert n == 0
    assert np.array_equal(out, x)
    assert out is not x  # always a copy


@pytest.mark.parametrize("where", ["start", "end"])
def test_edges(where):
    x = _mic(2)
    if where == "start":
        x[:3] = 100.0
    else:
        x[-3:] = -100.0
    out, n = remove_mic_bursts(x, Z, pad=5)
    assert n == 1
    assert np.isfinite(out).all() and np.abs(out).max() < 1.0
    assert out.shape == x.shape and out.dtype == np.float32


def test_adjacent_bursts_merge():
    x = _mic(3)
    x[1000] = 100.0
    x[1008] = -100.0  # within 2 × pad: one widened run
    _, n = remove_mic_bursts(x, Z, pad=5)
    assert n == 1


def test_degenerate_inputs():
    assert remove_mic_bursts(np.zeros(100, np.float32), Z)[1] == 0
    assert remove_mic_bursts(np.zeros(0, np.float32), Z)[1] == 0
