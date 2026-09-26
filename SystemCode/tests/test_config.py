"""configs/base.yaml and the ml/config.py defaults describe the same, physically valid settings."""
import numpy as np

from ml.config import PreprocessConfig, load_config
from ml.features import comb_lines
from ml.paths import DEFAULT_CONFIG


def test_yaml_matches_defaults():
    assert load_config(DEFAULT_CONFIG).preprocess == PreprocessConfig()


def test_demodulation_bands_are_valid():
    env = PreprocessConfig().envelope
    nyquist = PreprocessConfig().fs_hz / 2
    for ch, (lo, hi) in env.bands_hz.items():
        assert lo >= env.sk.min_lo_hz, ch  # never demodulate below 1 kHz
        assert hi - lo >= env.min_band_hz, ch  # wide enough for 12 orders at 61 Hz
        assert hi <= nyquist, ch
    assert env.min_band_hz >= 2 * env.max_order * 61.0


def test_comb_lines():
    lines = comb_lines(PreprocessConfig())
    assert np.allclose(lines["ftf"], [0.375, 0.75, 1.125, 1.5, 1.875])
    assert np.allclose(lines["bsf"], [3.367, 3.742, 4.117])
    assert np.allclose(lines["bpfo"], [2.998, 5.996, 8.994])
    assert np.allclose(lines["bpfi"], [4.002, 5.002, 6.002, 9.004, 10.004, 11.004])
