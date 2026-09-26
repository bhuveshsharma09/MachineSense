"""Scratch 2D CNN for early fusion (Stage 3A): every sensor channel of a cache-C window (C × 128 × 25) enters one
network. Output: class logits and the 128-d embedding z."""
from __future__ import annotations

import torch
from torch import nn

from ml.config import CnnConfig


def _block(c_in: int, c_out: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
              nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ScratchCNN(nn.Module):
    """Conv blocks (2 × conv 3×3–BN–ReLU; the first `pool_blocks` end in 2×2 max-pool) → global average pooling →
    z = ReLU(BN(Linear)) (embedding_dim) → dropout → linear head."""

    def __init__(self, in_channels: int, n_classes: int, cfg: CnnConfig):
        super().__init__()
        blocks, c = [], in_channels
        for i, w in enumerate(cfg.widths):
            blocks.append(_block(c, w, pool=i < cfg.pool_blocks))
            c = w
        self.features = nn.Sequential(*blocks)
        self.embed = nn.Sequential(nn.Linear(c, cfg.embedding_dim, bias=False), nn.BatchNorm1d(cfg.embedding_dim),
                                   nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.embedding_dim, n_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, 128, 25) float32, already normalised → (logits (B, n_classes), z (B, embedding_dim))."""
        z = self.embed(self.features(x).mean(dim=(2, 3)))
        return self.head(z), z


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
