"""Healthy-only autoencoders on cache C (Stage 3B, Q1).

FrameAE (primary): one time frame = C channels × 128 spectral bins; 1D convolutions over the bin axis (vibration bins are
orders, microphone bins are Mel frequencies). It reconstructs individual spectral slices and does not model temporal
relationships across frames. WindowAE (secondary): one window = C × 128 × 25, 2D convolutions, 64-d bottleneck."""
from __future__ import annotations

import torch
from torch import nn

from ml.config import AeConfig

N_BINS, N_FRAMES = 128, 25


class FrameAE(nn.Module):
    """Encoder: stride-2 Conv1d–BN–ReLU blocks (128 → 64 → 32 → 16 bins) → Linear to the bottleneck. Decoder: Linear →
    mirrored ConvTranspose1d–BN–ReLU blocks → a final linear ConvTranspose1d back to C × 128."""

    def __init__(self, in_channels: int, cfg: AeConfig):
        super().__init__()
        chs, k = list(cfg.conv_channels), cfg.kernel
        enc, c = [], in_channels
        for w in chs:
            enc += [nn.Conv1d(c, w, k, stride=2, padding=k // 2), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        self.encoder = nn.Sequential(*enc)
        self.length = N_BINS // 2 ** len(chs)
        self.to_z = nn.Linear(c * self.length, cfg.bottleneck)
        self.from_z = nn.Linear(cfg.bottleneck, c * self.length)
        dec = []
        outs = chs[::-1][1:] + [in_channels]
        for i, w in enumerate(outs):
            dec.append(nn.ConvTranspose1d(c, w, k, stride=2, padding=k // 2, output_padding=1))
            if i < len(outs) - 1:
                dec += [nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        self.decoder = nn.Sequential(*dec)
        self.c_last = chs[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 128) float32, normalised → reconstruction of the same shape."""
        z = self.to_z(self.encoder(x).flatten(1))
        return self.decoder(self.from_z(z).view(-1, self.c_last, self.length))


class WindowAE(nn.Module):
    """Encoder: stride-2 Conv2d–BN–ReLU blocks (128×25 → 64×13 → 32×7 → 16×4) → Linear to a 64-d bottleneck; mirrored
    decoder ending in a linear ConvTranspose2d back to C × 128 × 25."""

    def __init__(self, in_channels: int, cfg: AeConfig):
        super().__init__()
        chs = list(cfg.w_conv_channels)
        enc, c = [], in_channels
        for w in chs:
            enc += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.encoder = nn.Sequential(*enc)
        self.shape = (c, 16, 4)
        self.to_z = nn.Linear(c * 16 * 4, cfg.w_bottleneck)
        self.from_z = nn.Linear(cfg.w_bottleneck, c * 16 * 4)
        dec = []
        outs = chs[::-1][1:] + [in_channels]
        for i, w in enumerate(outs):
            dec.append(nn.ConvTranspose2d(c, w, 3, stride=2, padding=1, output_padding=(1, 0)))
            if i < len(outs) - 1:
                dec += [nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 128, 25) float32, normalised → reconstruction of the same shape."""
        z = self.to_z(self.encoder(x).flatten(1))
        return self.decoder(self.from_z(z).view(-1, *self.shape))


def build_ae(level: str, in_channels: int, cfg: AeConfig) -> nn.Module:
    return {"frame": FrameAE, "window": WindowAE}[level](in_channels, cfg)
