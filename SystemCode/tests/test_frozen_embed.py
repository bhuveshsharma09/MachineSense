"""Stage R10 helpers (ml/frozen_embed.py): the fixed per-bearing packing, a frozen encoder in eval mode, deterministic
1,024-d output; and the nearest-centroid diagnostic on synthetic data. The encoder is built without the ImageNet weights
(the architecture is the same) so the tests run on CPU without the downloaded file."""
import numpy as np
import pandas as pd
import torch

from ml.frozen_embed import EMBED_DIM, frozen_embed, load_frozen_resnet, nearest_class, pack_bearings
from ml.models.resnet import ResNetConfig


def test_packing_maps_channels_0_2_to_u_and_3_5_to_o_in_order():
    x = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1).expand(2, 6, 4, 3).clone()
    u, o = pack_bearings(x)
    assert u.shape == o.shape == (2, 3, 4, 3)
    assert u[0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert o[0, :, 0, 0].tolist() == [3.0, 4.0, 5.0]


def test_encoder_is_frozen_and_in_eval_mode():
    m = load_frozen_resnet(ResNetConfig(), pretrained=False)
    assert not m.training and all(not mod.training for mod in m.modules())
    assert not any(p.requires_grad for p in m.parameters())
    assert isinstance(m.fc, torch.nn.Identity)


def test_two_calls_give_identical_output_of_shape_b_by_1024():
    torch.manual_seed(0)
    m = load_frozen_resnet(ResNetConfig(), pretrained=False)
    x = torch.randn(3, 6, 128, 25)
    a, b = frozen_embed(m, x, size=64), frozen_embed(m, x, size=64)
    assert a.shape == (3, EMBED_DIM)
    assert torch.equal(a, b)
    assert not m.training


def test_embedding_halves_are_the_two_bearings():
    torch.manual_seed(1)
    m = load_frozen_resnet(ResNetConfig(), pretrained=False)
    x = torch.randn(2, 6, 128, 25)
    y = x.clone()
    y[:, 3:] = torch.randn(2, 3, 128, 25)  # change only the overhang channels
    a, b = frozen_embed(m, x, size=64), frozen_embed(m, y, size=64)
    assert torch.equal(a[:, :512], b[:, :512]) and not torch.equal(a[:, 512:], b[:, 512:])


def test_nearest_class_by_mean_window_distance():
    rng = np.random.default_rng(0)
    fit = np.concatenate([rng.normal(0, 1, (50, 4)), rng.normal(10, 1, (50, 4))]).astype(np.float32)
    labels = ["normal"] * 50 + ["fault"] * 50
    emb = np.concatenate([rng.normal(0, 1, (3, 4)), rng.normal(10, 1, (3, 4))]).astype(np.float32)
    got = nearest_class(fit, labels, emb, ["r1"] * 3 + ["r2"] * 3)
    assert got.to_dict() == {"r1": "normal", "r2": "fault"}
    assert isinstance(got, pd.Series)
