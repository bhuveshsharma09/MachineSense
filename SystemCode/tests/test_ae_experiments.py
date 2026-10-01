"""Stage R6: augmentation recipes, the shared training loop (A0 = ml.train_ae.train_ae), views linked to recordings, and
the common score scale."""
import math

import numpy as np
import pytest
import torch

from ml.ae_experiments import (add_noise_floor, calibrated_z, common_mismatch_offsets, jitter_offsets, make_views,
                               mismatch_sd, n_views_for, noise_floor_db, recording_level, standardised, train_ae_arm)
from ml.config import AeConfig


def tiny():
    cfg = AeConfig(epochs=3, batch_size=64, warmup_epochs=1, seed=7)
    g = np.random.default_rng(0)
    cache = torch.from_numpy(g.normal(-40, 5, size=(6, 6, 128, 25)).astype(np.float16))
    rows, rec = np.arange(6), np.array([0, 0, 0, 1, 1, 1])
    return cfg, cache, rows, rec, np.full(6, -40.0), np.full(6, 5.0)


def test_a0_is_exactly_the_ae6_training():
    from ml.train_ae import train_ae
    cfg, cache, rows, rec, mu, sd = tiny()
    torch.manual_seed(0)
    m0, h0 = train_ae(cache, rows, list(range(6)), mu, sd, "frame", cfg, torch.device("cpu"))
    torch.manual_seed(0)
    m1, h1, info = train_ae_arm("A0", cache, rows, rec, list(range(6)), mu, sd, cfg, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))
    assert [h["train_loss"] for h in h0] == [h["train_loss"] for h in h1] and info["n_views"] == 0


def test_a1_matches_steps_and_links_views_to_recordings():
    cfg, cache, rows, rec, mu, sd = tiny()
    cfg = cfg.model_copy(update={"epochs": 10})
    v = make_views(2, 5, 6, 1.0, seed=1)
    assert v.shape == (2, 5, 6) and np.abs(v).max() <= 1.0
    torch.manual_seed(0)
    _, _, info = train_ae_arm("A1", cache, rows, rec, list(range(6)), mu, sd, cfg, torch.device("cpu"), views=v)
    _, _, i0 = train_ae_arm("A0", cache, rows, rec, list(range(6)), mu, sd, cfg, torch.device("cpu"))
    assert info["n_recordings"] == 2 and info["n_views"] == 10 and info["epochs"] == 3       # A0's 30 steps at 12 per epoch
    assert info["optimizer_steps"] == i0["optimizer_steps"]                                    # same budget as A0
    assert n_views_for(11) == 10 and n_views_for(20) == 5 and n_views_for(11) * 11 >= 100


def test_augmentation_ranges():
    g = torch.Generator().manual_seed(0)
    j = jitter_offsets(g, 5000, 6, 1.0)
    assert j.abs().max() <= 1.0 and j.std(dim=0).min() > 0.5           # independent per channel
    c = common_mismatch_offsets(g, 20000, 6, 1.0, 0.0)
    assert torch.allclose(c, c[:, :1].expand_as(c)) and c.abs().max() <= 1.0   # sd 0: one common offset
    c2 = common_mismatch_offsets(g, 20000, 6, 1.0, 0.3)
    assert abs(float((c2 - c2.mean(dim=1, keepdim=True)).std()) - 0.3 * math.sqrt(5 / 6)) < 0.02


def test_noise_floor_raises_quiet_bins_and_leaves_loud_bins():
    g = torch.Generator().manual_seed(0)
    x = torch.tensor([[[-120.0] * 2000, [0.0] * 2000]])          # one example, 2 channels
    y = add_noise_floor(x, torch.tensor([-60.0, -60.0]), g)
    assert (y >= x - 1e-4).all()                                     # never lowers a value
    assert abs(float(y[0, 0].median()) - (-60 + 10 * math.log10(math.log(2)))) < 0.5   # quiet: at the floor
    assert float((y[0, 1] - x[0, 1]).abs().max()) < 1e-3             # loud: unchanged
    vals = np.random.default_rng(0).normal(0, 1, size=(10, 3, 128, 25))
    vals[:, 2] += 10
    f = noise_floor_db(vals, 5)
    assert f.shape == (3,) and f[2] > f[0] + 9


def test_mismatch_sd_ignores_common_gain():
    lv = np.array([[0, 1, 2], [5, 6, 7], [10, 11, 12.0]])           # only a common level change between recordings
    assert mismatch_sd(lv) == pytest.approx(0.0, abs=1e-12)
    assert mismatch_sd(np.array([[0, 0], [1, -1], [0, 0], [-1, 1.0]])) > 0


def test_score_scale_and_level():
    z = calibrated_z(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0]))
    assert np.allclose(z, np.log([0.5, 1.0, 2.0])) and not (z > 0)[1]          # tie at the max: no alarm
    zf = calibrated_z(np.array([1.0, 2.0, 4.0]), np.array([2.0, 3.0]), higher_is_fault=False)
    assert np.allclose(zf, np.log([2.0, 1.0, 0.5]))                               # low fault-AE error = fault-like
    s = standardised(np.array([1.0, 3.0]), np.array([0.0, 2.0]))
    assert np.allclose(s, [0.0, math.sqrt(2)])
    lv = recording_level(np.stack([np.full((2, 3), -40.0), np.full((2, 3), -20.0), np.full((2, 3), -30.0)]),
                         np.array(["a", "a", "b"]))
    assert lv == {"a": -30.0, "b": -30.0}


def test_clean_target_arms_train_and_differ_from_a0():
    cfg, cache, rows, rec, mu, sd = tiny()
    out = {}
    for arm in ("A0", "A2", "A3"):
        torch.manual_seed(0)
        m, h, _ = train_ae_arm(arm, cache, rows, rec, list(range(6)), mu, sd, cfg, torch.device("cpu"),
                               floor_db=np.full(6, -55.0), mm_sd=0.3)
        out[arm] = m.state_dict()
        assert np.isfinite(h[-1]["train_loss"])
    assert not all(torch.equal(a, b) for a, b in zip(out["A0"].values(), out["A2"].values()))
    with pytest.raises(ValueError):
        train_ae_arm("A3", cache, rows, rec, list(range(6)), mu, sd, cfg, torch.device("cpu"))
