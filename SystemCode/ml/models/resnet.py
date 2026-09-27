"""ImageNet-pretrained ResNet-18 on the order spectrogram (Stage ML-5, development experiment; never in the bundle).

The 6-channel cache-C window (6 × 128 orders × 25 frames, already globally normalised) → a learnable 1×1 conv adapter to
3 channels → bilinear resize to 224 × 224 → ImageNet mean/std → torchvision ResNet-18 (IMAGENET1K_V1) → the 512-d
embedding z → dropout → a new linear head. R18-probe freezes the backbone (no gradients, BatchNorm kept in eval mode with
the ImageNet statistics); R18-ft fine-tunes everything with a lower backbone learning rate."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F

from ml.paths import CONFIGS_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VARIANTS = ("r18probe", "r18ft")
EMBEDDING_DIM = 512


class ResNetConfig(BaseModel):
    """configs/resnet.yaml; field names mirror CnnConfig where ml.train uses them."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 42
    weights: str = "IMAGENET1K_V1"
    weights_file: str = "resnet18-f37072fd.pth"
    weights_sha256: str = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    input_size: int = 224
    dropout: float = 0.2
    vibration_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    gain_jitter_db: float = 1.0
    time_shift_frames: int = 2
    time_mask_frames: int = 4
    eval_batch_size: int = 256
    mps_memory_fraction: float = 0.7
    time_guard_minutes: float = 90
    extra_seeds_if_under_minutes: float = 60


def load_resnet_config(path: Path = CONFIGS_DIR / "resnet.yaml") -> ResNetConfig:
    return ResNetConfig(**yaml.safe_load(Path(path).read_text()))


def weights_path(cfg: ResNetConfig) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / cfg.weights_file


def verify_weights(cfg: ResNetConfig) -> str:
    """sha256 of the downloaded ImageNet weights; raises if it differs from the declared hash."""
    p = weights_path(cfg)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha != cfg.weights_sha256:
        raise RuntimeError(f"{p} has sha256 {sha}, expected {cfg.weights_sha256}")
    return sha


class ResNet18Spectrogram(nn.Module):
    """forward(x: (B, 6, 128, 25) float32, normalised) → (logits (B, n_classes), z (B, 512))."""

    def __init__(self, in_channels: int, n_classes: int, cfg: ResNetConfig, variant: str, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant, self.size = variant, cfg.input_size
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        self.backbone = resnet18(weights=getattr(ResNet18_Weights, cfg.weights) if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.frozen = variant == "r18probe"
        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(EMBEDDING_DIM, n_classes))
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # BatchNorm keeps the ImageNet running statistics
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.adapter(x)
        h = F.interpolate(h, size=(self.size, self.size), mode="bilinear", align_corners=False)
        z = self.backbone((h - self.img_mean) / self.img_std)
        return self.head(z), z


def param_groups(model: ResNet18Spectrogram, cfg: ResNetConfig) -> list[dict]:
    """Adapter and head at cfg.lr; the backbone at cfg.backbone_lr for R18-ft (frozen for R18-probe)."""
    groups = [{"params": [*model.adapter.parameters(), *model.head.parameters()], "lr": cfg.lr}]
    if not model.frozen:
        groups.append({"params": list(model.backbone.parameters()), "lr": cfg.backbone_lr})
    return groups


def n_parameters(model: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
