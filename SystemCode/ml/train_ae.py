"""Training and scoring of the healthy-only autoencoders (Stage 3B, Q1). Torch only; never import LightGBM here.

Same Mac/MPS rules as ml.train: a float16 resident cache on the device, every batch cast to float32 BEFORE augmentation,
normalisation and the model; float32 weights and arithmetic; no AMP; no DataLoader workers.

Units: at frame level one example is one time frame of a cached window (C × 128); at window level one window
(C × 128 × 25). Reconstruction target = the same (augmented, normalised) input. Error, defined before training: squared
error in normalised units; per-channel error = mean over that channel's bins (and frames); total = equal-weight mean over
channels; a recording's score = mean over all frames of its 9 windows (= mean of its window errors, equal frame counts).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ml.config import AeConfig
from ml.models.ae import N_FRAMES, build_ae
from ml.train import lr_lambda, to_device_f32


def normalise_any(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Global per-channel normalisation of a (B, C, …) float32 batch."""
    if x.dtype != torch.float32:
        raise TypeError(f"normalise expects a float32 batch, got {x.dtype}")
    shape = (1, -1) + (1,) * (x.dim() - 2)
    return (x - mean.view(shape)) / std.view(shape)


def unit_index(window_rows: np.ndarray, level: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Training units of the given cache rows: every frame of every window (frame level) or every window."""
    if level == "window":
        return np.asarray(window_rows), None
    return np.repeat(window_rows, N_FRAMES), np.tile(np.arange(N_FRAMES), len(window_rows))


def gather(cache: torch.Tensor, rows: torch.Tensor, t: torch.Tensor | None, channels: torch.Tensor) -> torch.Tensor:
    """Index slice of the resident cache → float32: (B, C, 128) frames when t is given, else (B, C, 128, 25) windows."""
    x = cache[rows][:, channels] if t is None else cache[rows, :, :, t][:, channels]
    return x.to(torch.float32)


def augmented_batch(cache, rows, t, channels, mean, std, gen: torch.Generator | None, max_db: float) -> torch.Tensor:
    """float32 → (training only) gain offset per example and channel, uniform in [−max_db, +max_db] dB → normalise.
    The returned tensor is both the model input and the reconstruction target."""
    x = gather(cache, rows, t, channels)
    if gen is not None and max_db > 0:
        off = (torch.rand(x.shape[0], x.shape[1], generator=gen) * 2 - 1) * max_db
        x = x + off.to(x.device).view(x.shape[0], x.shape[1], *([1] * (x.dim() - 2)))
    return normalise_any(x, mean, std)


def reconstruction_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the model's reconstruction of x against x itself (equal weight per channel)."""
    return torch.mean((model(x) - x) ** 2)


def train_ae(cache: torch.Tensor, window_rows: np.ndarray, channels: list[int], mean: np.ndarray, std: np.ndarray,
             level: str, cfg: AeConfig, device: torch.device, epochs: int | None = None,
             on_epoch: Callable[[int, dict], None] | None = None) -> tuple[nn.Module, list]:
    """Train a fresh autoencoder for a fixed number of epochs on every unit of the cache rows `window_rows` (the FIT
    normals' windows). Returns the model in eval mode and the per-epoch training history."""
    frame = level == "frame"
    epochs = (cfg.epochs if frame else cfg.w_epochs) if epochs is None else epochs
    batch = cfg.batch_size if frame else cfg.w_batch_size
    gen = torch.Generator().manual_seed(cfg.seed)
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    rows_u, t_u = unit_index(window_rows, level)
    rows_u = torch.as_tensor(rows_u, dtype=torch.long)
    t_u = None if t_u is None else torch.as_tensor(t_u, dtype=torch.long)
    model = build_ae(level, len(channels), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    spe = math.ceil(len(rows_u) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup_epochs * spe, spe * epochs))
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        perm = torch.randperm(len(rows_u), generator=gen)
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            if len(j) < 2:  # batch-norm needs at least two examples
                continue
            x = augmented_batch(cache, rows_u[j].to(device), None if t_u is None else t_u[j].to(device), ch, mu, sd,
                                gen, cfg.gain_jitter_db)
            loss = reconstruction_loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
            n += len(j)
        rec = {"epoch": ep, "train_loss": tot / n, "seconds": time.perf_counter() - t0}
        history.append(rec)
        if on_epoch:
            on_epoch(ep, rec)
    return model.eval(), history


@torch.no_grad()
def window_channel_errors(model: nn.Module, cache: torch.Tensor, window_rows: np.ndarray, channels: list[int],
                          mean: np.ndarray, std: np.ndarray, level: str, cfg: AeConfig,
                          device: torch.device) -> np.ndarray:
    """Per-window, per-channel reconstruction error (n_windows × C, float64): mean squared error over the channel's
    bins and the window's frames, in normalised units, without augmentation."""
    model.eval()
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    rows = torch.as_tensor(np.asarray(window_rows), dtype=torch.long)
    per = max(1, cfg.eval_batch_size // (N_FRAMES if level == "frame" else 1))
    out = []
    for i in range(0, len(rows), per):
        x = augmented_batch(cache, rows[i:i + per].to(device), None, ch, mu, sd, None, 0.0)  # (B, C, 128, 25)
        if level == "frame":
            b, c, f, t = x.shape
            xf = x.permute(0, 3, 1, 2).reshape(b * t, c, f)
            err = ((model(xf) - xf) ** 2).mean(dim=2).view(b, t, c).mean(dim=1)
        else:
            err = ((model(x) - x) ** 2).mean(dim=(2, 3))
        out.append(err.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)
