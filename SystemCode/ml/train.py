"""Training and scoring loop for the scratch CNN (Stage 3A). Torch only; never import LightGBM in this process.

Mac/MPS rules (CLAUDE.md §2): the resident input cache may stay float16 on the device; every batch is an index slice
cast to float32 BEFORE augmentation, normalisation and the model; weights, optimiser state and all arithmetic are
float32 (no AMP); NumPy outputs are cast to float32 before .to(device); no DataLoader workers.
"""
from __future__ import annotations

import fcntl
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.config import CnnConfig
from ml.models.cnn import ScratchCNN


@contextmanager
def gpu_job(device: torch.device, lock_dir: Path, memory_fraction: float):
    """One GPU job at a time: an exclusive lock file while the job runs; on MPS also cap the process at
    `memory_fraction` of the recommended working set. CPU jobs take no lock."""
    if device.type == "cpu":
        yield
        return
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / "gpu.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("another GPU job holds the lock; run one GPU job at a time") from e
        if device.type == "mps":
            torch.mps.set_per_process_memory_fraction(memory_fraction)
        try:
            yield
        finally:
            release_device_memory(device)
            fcntl.flock(fh, fcntl.LOCK_UN)


def release_device_memory(device: torch.device) -> None:
    """Between runs: return cached MPS memory to the system."""
    if device.type == "mps":
        torch.mps.empty_cache()


def to_device_f32(a: np.ndarray, device: torch.device) -> torch.Tensor:
    """NumPy/sklearn output → float32 tensor on the device (MPS has no float64)."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Global per-channel normalisation (never per window). x must already be float32."""
    if x.dtype != torch.float32:
        raise TypeError(f"normalise expects a float32 batch, got {x.dtype}")
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def augment_gain(x: torch.Tensor, gen: torch.Generator, max_db: float) -> torch.Tensor:
    """Constant offset per window and channel in the dB domain, uniform in [−max_db, +max_db] (before normalisation)."""
    off = (torch.rand(x.shape[0], x.shape[1], generator=gen) * 2 - 1) * max_db
    return x + off.to(x.device)[:, :, None, None]


def augment_time(x: torch.Tensor, gen: torch.Generator, max_shift: int, max_mask: int) -> torch.Tensor:
    """Circular time shift (uniform integer in [−max_shift, +max_shift]) and one time mask of width 0..max_mask set to 0
    (= the channel mean after normalisation), per window."""
    b, _, _, t = x.shape
    if max_shift:
        shift = torch.randint(-max_shift, max_shift + 1, (b,), generator=gen)
        idx = (torch.arange(t)[None, :] - shift[:, None]) % t
        x = torch.gather(x, 3, idx.to(x.device)[:, None, None, :].expand_as(x))
    if max_mask:
        width = torch.randint(0, max_mask + 1, (b,), generator=gen)
        start = (torch.rand(b, generator=gen) * (t - width + 1)).long()
        tt = torch.arange(t)[None, :]
        mask = (tt >= start[:, None]) & (tt < (start + width)[:, None])
        x = x.masked_fill(mask.to(x.device)[:, None, None, :], 0.0)
    return x


def prepare_batch(cache: torch.Tensor, idx: torch.Tensor, channels: torch.Tensor, mean: torch.Tensor,
                  std: torch.Tensor, cfg: CnnConfig, gen: torch.Generator | None) -> torch.Tensor:
    """Index slice of the resident cache → float32 → (training only, gen given) gain jitter → normalise → (training
    only) time shift and time mask."""
    x = cache[idx][:, channels].to(torch.float32)
    if gen is not None:
        x = augment_gain(x, gen, cfg.gain_jitter_db)
    x = normalise(x, mean, std)
    if gen is not None:
        x = augment_time(x, gen, cfg.time_shift_frames, cfg.time_mask_frames)
    return x


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced weights from the TRAINING windows: n / (n_classes × count_c)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("a class has no training windows")
    return len(y) / (n_classes * counts)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warm-up over `warmup` steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train_model(cache: torch.Tensor, y: torch.Tensor, train_idx: np.ndarray, channels: list[int], mean: np.ndarray,
                std: np.ndarray, n_classes: int, cfg: CnnConfig, device: torch.device, epochs: int | None = None,
                augment: bool = True, on_epoch: Callable[[int, dict], None] | None = None) -> tuple[ScratchCNN, list]:
    """Train a fresh ScratchCNN for a fixed number of epochs on the rows `train_idx` of the resident `cache`
    (N × 7 × 128 × T, any float dtype, on `device`) with labels `y` (N, int64, on `device`). mean/std: per-channel
    statistics of the selected channels. Returns the model (eval mode) and the per-epoch history (train loss, train
    accuracy on the augmented batches, seconds, mean step seconds)."""
    epochs = cfg.epochs if epochs is None else epochs
    gen = torch.Generator().manual_seed(cfg.seed)
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    y_np = y[torch.as_tensor(train_idx, device=device)].cpu().numpy()
    model = ScratchCNN(len(channels), n_classes, cfg).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=to_device_f32(class_weights(y_np, n_classes), device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup_epochs * steps_per_epoch, total))
    rows = torch.as_tensor(train_idx, dtype=torch.long)
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot_loss = tot_correct = tot_n = 0.0
        order = rows[torch.randperm(len(rows), generator=gen)]
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size].to(device)
            xb = prepare_batch(cache, idx, ch, mu, sd, cfg, gen if augment else None)
            yb = y[idx]
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_correct += float((logits.detach().argmax(1) == yb).sum())
            tot_n += len(idx)
        secs = time.perf_counter() - t0
        rec = {"epoch": ep, "train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n, "seconds": secs,
               "step_seconds": secs / steps_per_epoch, "lr_end": sched.get_last_lr()[0]}
        history.append(rec)
        if on_epoch:
            on_epoch(ep, rec)
    return model.eval(), history


@torch.no_grad()
def score_windows(model: ScratchCNN, cache: torch.Tensor, idx: np.ndarray, channels: list[int], mean: np.ndarray,
                  std: np.ndarray, cfg: CnnConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Logits (n, n_classes) and embeddings (n, embedding_dim), float32 NumPy, for the rows `idx`, no augmentation."""
    model.eval()
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    rows = torch.as_tensor(idx, dtype=torch.long)
    logits, emb = [], []
    for i in range(0, len(rows), cfg.eval_batch_size):
        xb = prepare_batch(cache, rows[i:i + cfg.eval_batch_size].to(device), ch, mu, sd, cfg, None)
        lg, z = model(xb)
        logits.append(lg.float().cpu().numpy())
        emb.append(z.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(emb)
