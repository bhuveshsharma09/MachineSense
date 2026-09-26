"""estimate_speed on synthetic recordings: clean, double-triggered, dropped pulse, noisy baseline, dead tacho."""
import numpy as np
import pytest

from ml.config import PreprocessConfig
from ml.preprocess import choose_speed, estimate_speed, tacho_glitch_count

CFG = PreprocessConfig()
FS = CFG.fs_hz
N = CFG.n_rows
FR_FILE = 30.72
FR_TRUE = FR_FILE * 0.9765  # a little off the nominal ratio, as in real files


def _recording(fr=FR_TRUE, *, double=False, drop=(), displace=(), noise=0.0, seed=0):
    """8-channel recording: square once-per-rev tacho (12 % duty, −0.7 → 4.6 V) and a radial accelerometer
    carrying shaft harmonics. `drop` lists pulse numbers to remove; `displace` lists pulses shifted 40 % of a
    revolution late (the real MAFAULDA glitch); `double` adds a bounce after each edge."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    phase = (t * fr) % 1.0
    high = phase < 0.12
    rev = np.floor(t * fr).astype(int)
    for k in drop:
        high &= rev != k
    for k in displace:
        high &= rev != k
        high |= (rev == k) & (phase >= 0.4) & (phase < 0.52)
    tacho = np.where(high, 4.6, -0.7)
    if double:
        # 0.5 ms after each edge the pulse falls back low for 0.2 ms, then re-triggers.
        bounce = (phase * (1 / fr) >= 0.5e-3) & (phase * (1 / fr) < 0.7e-3)
        tacho = np.where(bounce, -0.7, tacho)
    tacho = tacho + noise * rng.normal(size=N)

    x = np.zeros((N, 8), dtype=np.float32)
    x[:, 0] = tacho
    for h, amp in enumerate([1.0, 0.6, 0.3, 0.2, 0.1], start=1):
        x[:, 2] += amp * np.sin(2 * np.pi * h * fr * t + h)
    x[:, 2] += 0.3 * rng.normal(size=N)
    x[:, 1:] += 0.05 * rng.normal(size=(N, 7))
    return x.astype(np.float32)


def _rel_err(est):
    return abs(est.fr_hz - FR_TRUE) / FR_TRUE


@pytest.mark.parametrize("kwargs", [
    {},
    {"double": True},
    {"drop": (10, 50, 51, 90)},
    {"noise": 0.4},
    {"displace": (15, 40, 41, 100)},
    {"double": True, "drop": (20,), "displace": (60,), "noise": 0.3},
], ids=["clean", "double_triggered", "dropped_pulses", "noisy_baseline", "displaced_pulses", "all_combined"])
def test_tacho_edges_recover_speed(kwargs):
    est = estimate_speed(_recording(**kwargs), FR_FILE, CFG)
    assert est.method == "tacho_edges"
    assert _rel_err(est) < 1e-3
    assert est.agreement < CFG.speed.disagree_tol


def test_dead_tacho_falls_back_to_accelerometer():
    x = _recording()
    x[:, 0] = 0.01 * np.random.default_rng(1).normal(size=N)
    est = estimate_speed(x, FR_FILE, CFG)
    assert est.method in ("accel_comb", "tacho_fft")
    assert _rel_err(est) < 2e-3


@pytest.mark.parametrize("bad", [
    np.full((N, 8), np.nan, dtype=np.float32),
    np.zeros((N, 8), dtype=np.float32),
    np.zeros((10, 8), dtype=np.float32),
    np.zeros((0,), dtype=np.float32),
], ids=["all_nan", "all_zero", "too_short", "wrong_shape"])
def test_never_raises(bad):
    est = estimate_speed(bad, FR_FILE, CFG)
    assert est.method == "failed" or np.isfinite(est.fr_hz)


def test_glitch_count():
    assert tacho_glitch_count(_recording()[:, 0], CFG) == 0
    # Each displaced pulse turns two intervals into one short and one long.
    assert tacho_glitch_count(_recording(displace=(15, 70))[:, 0], CFG) == 4


def test_choice_rule_prefers_confirmed_pair_over_unconfirmed_edges():
    # Edges off by 0.9 % (a missed glitch); FFT and comb agree within 0.05 %.
    est = choose_speed({"tacho_edges": 29.73, "tacho_fft": 30.00, "accel_comb": 30.01}, FR_FILE, CFG)
    assert est.method == "tacho_fft" and est.fr_hz == 30.00


def test_choice_rule_median_when_nothing_confirms():
    est = choose_speed({"tacho_edges": 29.9, "tacho_fft": 30.0, "accel_comb": 29.5}, FR_FILE, CFG)
    assert est.method == "tacho_edges" and est.agreement > CFG.speed.confirm_tol
