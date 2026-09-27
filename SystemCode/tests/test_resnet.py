"""Stage ML-5: the ResNet-18 wrapper (shapes, adapter → 224 × 224, frozen probe with BatchNorm in eval mode, learning-rate
groups), the per-class Mahalanobis score, and train_model with an injected model. CPU, no ImageNet download."""
import numpy as np
import torch

from ml.models.resnet import ResNet18Spectrogram, ResNetConfig, load_resnet_config, n_parameters, param_groups
from ml.openset import mahalanobis_min, mahalanobis_min_per_class

CFG = ResNetConfig()


def test_config_file_matches_the_declaration():
    c = load_resnet_config()
    assert (c.epochs, c.batch_size, c.lr, c.backbone_lr, c.weight_decay, c.input_size) == (30, 64, 1e-3, 1e-4, 0.01, 224)
    assert c.weights == "IMAGENET1K_V1" and c.weights_sha256.startswith("f37072fd")


def test_shapes_and_resize():
    m = ResNet18Spectrogram(6, 10, CFG, "r18ft", pretrained=False).eval()
    seen = {}
    m.backbone.conv1.register_forward_hook(lambda mod, inp, out: seen.update({"in": inp[0].shape}))
    logits, z = m(torch.randn(2, 6, 128, 25))
    assert logits.shape == (2, 10) and z.shape == (2, 512) and seen["in"] == (2, 3, 224, 224)


def test_probe_freezes_backbone_and_keeps_batchnorm_in_eval():
    m = ResNet18Spectrogram(6, 10, CFG, "r18probe", pretrained=False).train()
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert not m.backbone.training and m.adapter.training and m.head.training
    bn = m.backbone.bn1
    before = bn.running_mean.clone()
    m(torch.randn(4, 6, 128, 25))
    assert torch.equal(before, bn.running_mean)
    n = n_parameters(m)
    assert n["trainable"] == 6 * 3 + 3 + 512 * 10 + 10 and len(param_groups(m, CFG)) == 1


def test_finetune_param_groups():
    m = ResNet18Spectrogram(6, 10, CFG, "r18ft", pretrained=False)
    g = param_groups(m, CFG)
    assert [x["lr"] for x in g] == [1e-3, 1e-4]
    assert sum(p.numel() for x in g for p in x["params"]) == n_parameters(m)["trainable"] == n_parameters(m)["total"]


def test_per_class_mahalanobis_equals_shared_for_identical_class_covariances():
    rng = np.random.default_rng(0)
    fit = np.r_[rng.normal(0, 1, (400, 5)), rng.normal(5, 1, (400, 5))]
    lab = np.r_[["a"] * 400, ["b"] * 400]
    q = rng.normal(0, 1, (50, 5))
    d_pc, info = mahalanobis_min_per_class(fit, lab, q)
    d_sh, _ = mahalanobis_min(fit, lab, q)
    assert set(info["shrinkage"]) == {"a", "b"} and np.allclose(d_pc, d_sh, rtol=0.15)
    far, _ = mahalanobis_min_per_class(fit, lab, q + 50)
    assert far.min() > d_pc.max()


def test_train_model_accepts_an_injected_model():
    from ml.train import train_model
    torch.manual_seed(0)
    cfg = CFG.model_copy(update={"epochs": 1, "batch_size": 8, "eval_batch_size": 8})
    m = ResNet18Spectrogram(6, 3, cfg, "r18probe", pretrained=False)
    cache = torch.randn(24, 7, 128, 25).half()
    y = torch.tensor([0, 1, 2] * 8)
    out, hist = train_model(cache, y, np.arange(24), list(range(6)), np.zeros(6), np.ones(6), 3, cfg, torch.device("cpu"),
                            model=m, param_groups=param_groups(m, cfg))
    assert out is m and len(hist) == 1 and np.isfinite(hist[0]["train_loss"])
