"""Healthy-only autoencoders (Stage 3B), on CPU with synthetic data."""
import numpy as np
import pandas as pd
import pytest
import torch

from ml.config import AeConfig
from ml.models.ae import build_ae
from ml.models.cnn import n_parameters
from ml.q1 import fold_ids
from ml.train_ae import (augmented_batch, normalise_any, reconstruction_loss, train_ae, unit_index,
                         window_channel_errors)

CFG = AeConfig()
CPU = torch.device("cpu")


@pytest.mark.parametrize("level, shape", [("frame", (7, 128)), ("window", (7, 128, 25))])
@pytest.mark.parametrize("channels", [7, 6])
def test_shapes_and_parameter_limit(level, shape, channels):
    m = build_ae(level, channels, CFG).eval()
    x = torch.randn(3, channels, *shape[1:])
    assert m(x).shape == x.shape
    if level == "frame":
        assert n_parameters(m) < CFG.max_params


def test_frame_units_keep_windows_together():
    rows, t = unit_index(np.array([5, 9]), "frame")
    assert rows.tolist() == [5] * 25 + [9] * 25 and t.tolist() == list(range(25)) * 2
    rows, t = unit_index(np.array([5, 9]), "window")
    assert rows.tolist() == [5, 9] and t is None


def test_frame_gather_and_float32():
    cache = torch.randn(4, 7, 128, 25).to(torch.float16)
    x = augmented_batch(cache, torch.tensor([1, 2]), torch.tensor([3, 7]), torch.arange(7), torch.zeros(7),
                        torch.ones(7), None, 0.0)
    assert x.dtype == torch.float32 and x.shape == (2, 7, 128)
    assert torch.equal(x[0], cache[1, :, :, 3].float()) and torch.equal(x[1], cache[2, :, :, 7].float())
    with pytest.raises(TypeError):
        normalise_any(cache[:1], torch.zeros(7), torch.ones(7))


def test_target_is_the_augmented_input():
    """Gain augmentation changes the input, and the loss compares the reconstruction with that same augmented input."""
    cache = torch.zeros(2, 7, 128, 25).to(torch.float16)
    gen = torch.Generator().manual_seed(0)
    x = augmented_batch(cache, torch.tensor([0, 1]), None, torch.arange(7), torch.zeros(7), torch.ones(7), gen, 1.0)
    assert x.abs().max() > 0 and x.abs().max() <= 1.0  # an offset of at most 1 dB per example and channel
    assert torch.allclose(x, x.mean(dim=(2, 3), keepdim=True).expand_as(x))  # constant within each channel
    identity = torch.nn.Identity()
    assert float(reconstruction_loss(identity, x)) == 0.0


class _Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


@pytest.mark.parametrize("level", ["frame", "window"])
def test_error_definition(level):
    """With a model that outputs zeros, the error of channel c is the mean of x² over its bins and frames."""
    cache = torch.randn(3, 7, 128, 25).to(torch.float16)
    mean, std = np.zeros(7), np.ones(7)
    err = window_channel_errors(_Zero(), cache, np.arange(3), list(range(7)), mean, std, level, CFG, CPU)
    expected = (cache.float() ** 2).mean(dim=(2, 3)).numpy()
    assert err.shape == (3, 7) and np.allclose(err, expected, rtol=1e-5)


def test_tiny_training_reduces_the_loss():
    torch.manual_seed(0)
    cache = (torch.randn(1, 7, 128, 1) + 0.1 * torch.randn(4, 7, 128, 25)).to(torch.float16)
    cfg = CFG.model_copy(update={"epochs": 30, "batch_size": 50, "warmup_epochs": 1, "lr": 3e-3})
    _, hist = train_ae(cache, np.arange(4), list(range(7)), np.zeros(7), np.ones(7), "frame", cfg, CPU)
    assert hist[-1]["train_loss"] < 0.5 * hist[0]["train_loss"]


def test_fold_ids_follow_the_saved_assignment():
    pool = pd.DataFrame({"rec_id": [f"n{i}" for i in range(6)] + ["f0", "f1"],
                         "class10": ["normal"] * 6 + ["imbalance"] * 2, "dev_fold": [0, 0, 1, 1, 2, 2, 0, 1]})
    assign = {0: pd.Series({"n2": "fit", "n3": "cal", "n4": "fit", "n5": "fit"})}
    fit, cal, ev, ctx = fold_ids(assign, pool, 0)
    assert fit == ["n2", "n4", "n5"] and cal == ["n3"] and ev == ["f0", "n0", "n1"]
    assert ctx.calibration == {"n3"} and ctx.evaluation == {"f0", "n0", "n1"} and not ctx.withheld
    with pytest.raises(ValueError):
        fold_ids({0: pd.Series({"n2": "fit", "n3": "cal"})}, pool, 0)  # does not cover every outer-training normal
