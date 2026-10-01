"""Stage R6: autoencoder experiments for healthy vs faulty separation (CLAUDE.md §7). The training arms share one loop with
ml.train_ae (same batching, normalisation, schedule and frame-level model); they differ only in the augmentation and the
reconstruction target. Pure functions and a training function (no file I/O).

Arms: A0 the AE-6 recipe (gain jitter on input = target); A1 fixed recording-level views, jitter off; A2 denoising (jitter
on the input, target = the un-augmented input); A3 common gain + per-channel mismatch + additive noise at the measured
floor, A2 target; F the AE-6 recipe on fault recordings. Cache values are dB of spectral amplitude (20·log10), so power is
10^(x/10) and a gain offset is an addition in dB."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from ml.config import AeConfig
from ml.models.ae import N_FRAMES, build_ae
from ml.train import lr_lambda, to_device_f32
from ml.train_ae import gather, normalise_any, reconstruction_loss, unit_index

ARMS = ("A0", "A1", "A2", "A3", "F")
CLEAN_TARGET = {"A0": False, "A1": False, "A2": True, "A3": True, "F": False}


# ----------------------------------------------------------------------------------------------------------- augmentation
def jitter_offsets(gen: torch.Generator, n: int, n_ch: int, max_db: float) -> torch.Tensor:
    """AE-6's gain jitter: an offset per example and per channel, uniform in [−max_db, +max_db] dB (n × n_ch)."""
    return (torch.rand(n, n_ch, generator=gen) * 2 - 1) * max_db


def common_mismatch_offsets(gen: torch.Generator, n: int, n_ch: int, max_db: float, mismatch_sd: float) -> torch.Tensor:
    """A3: a common gain offset per example, U(±max_db), plus a per-channel mismatch N(0, mismatch_sd) (dB)."""
    common = (torch.rand(n, 1, generator=gen) * 2 - 1) * max_db
    return common + torch.randn(n, n_ch, generator=gen) * mismatch_sd


def add_noise_floor(x_db: torch.Tensor, floor_db: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Additive noise at the measured floor, in power: 10·log10(10^(x/10) + u·10^(F_c/10)), u ~ Exp(1) per bin and frame.
    x_db: (B, C, …) dB; floor_db: (C,) per-channel floor in dB. Never lowers a value."""
    shape = (1, -1) + (1,) * (x_db.dim() - 2)
    u = -torch.log(torch.rand(x_db.shape, generator=gen).clamp_min(1e-12)).to(x_db.device)
    noise = u * torch.pow(10.0, floor_db.to(x_db.device).view(shape) / 10)
    return 10 * torch.log10(torch.pow(10.0, x_db / 10) + noise)


def noise_floor_db(values: np.ndarray, pct: float = 5.0) -> np.ndarray:
    """Per-channel noise floor estimate: the `pct`th percentile of the cached dB values (n, C, …) of the FIT recordings."""
    v = np.moveaxis(np.asarray(values, dtype=np.float64), 1, 0).reshape(values.shape[1], -1)
    return np.percentile(v, pct, axis=1)


def mismatch_sd(channel_levels: np.ndarray) -> float:
    """Pooled across-recording SD of each channel's mean dB level relative to the recording's mean over channels
    (channel_levels: n_recordings × C, FIT recordings only): the observed spread of relative channel gain."""
    rel = np.asarray(channel_levels, dtype=np.float64)
    rel = rel - rel.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(rel.var(axis=0, ddof=1))))


def make_views(n_recordings: int, n_views: int, n_ch: int, max_db: float, seed: int) -> np.ndarray:
    """A1: fixed recording-level views, offsets (n_recordings × n_views × C) dB, each drawn once from U(±max_db) — the
    jitter's distribution, materialised. View j of recording r is linked to r by its index."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-max_db, max_db, size=(n_recordings, n_views, n_ch))


def n_views_for(n_fit: int, minimum: int = 5, total: int = 100) -> int:
    """Views per recording: at least `minimum`, and at least `total` views over the fit recordings."""
    return max(minimum, math.ceil(total / n_fit))


# ----------------------------------------------------------------------------------------------------------- training
def train_ae_arm(arm: str, cache: torch.Tensor, window_rows: np.ndarray, rec_of_row: np.ndarray, channels: list[int],
                 mean: np.ndarray, std: np.ndarray, cfg: AeConfig, device: torch.device, *, views: np.ndarray | None = None,
                 floor_db: np.ndarray | None = None, mm_sd: float | None = None) -> tuple[nn.Module, list, dict]:
    """Train the frame-level AE for one arm on the cache rows `window_rows` (rec_of_row: recording index 0..n−1 of each
    row, linking frames and views to their recording). A0 and F reproduce ml.train_ae.train_ae (same random stream).
    A1 runs exactly A0's number of optimiser steps (the last epoch may be partial). Returns (model, history, info)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    gen = torch.Generator().manual_seed(cfg.seed)
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    rows_u, t_u = unit_index(window_rows, "frame")
    rec_u = np.repeat(np.asarray(rec_of_row), N_FRAMES)
    n_frames = len(rows_u)
    view_u = None
    epochs = cfg.epochs
    if arm == "A1":
        if views is None:
            raise ValueError("A1 needs the materialised views")
        v = views.shape[1]
        rows_u, t_u, rec_u = np.tile(rows_u, v), np.tile(t_u, v), np.tile(rec_u, v)
        view_u = np.repeat(np.arange(v), n_frames)
        offs_table = torch.as_tensor(views, dtype=torch.float32)
    if arm == "A3" and (floor_db is None or mm_sd is None):
        raise ValueError("A3 needs the measured noise floor and the mismatch SD")
    rows_t, t_t = torch.as_tensor(rows_u, dtype=torch.long), torch.as_tensor(t_u, dtype=torch.long)
    model = build_ae("frame", len(channels), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    spe = math.ceil(len(rows_t) / cfg.batch_size)
    # the schedule and the stopping point are counted in optimiser steps; A1 stops at exactly A0's step count
    total = math.ceil(n_frames / cfg.batch_size) * cfg.epochs if arm == "A1" else spe * epochs
    epochs = math.ceil(total / spe)
    warm = cfg.warmup_epochs * math.ceil(n_frames / cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warm, total))
    steps = 0
    floor_t = None if floor_db is None else torch.as_tensor(np.asarray(floor_db), dtype=torch.float32)
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        perm = torch.randperm(len(rows_t), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            j = perm[i:i + cfg.batch_size]
            if len(j) < 2 or steps >= total:
                continue
            x = gather(cache, rows_t[j].to(device), t_t[j].to(device), ch)          # (B, C, 128) float32 dB
            if arm in ("A0", "A2", "F"):
                off = jitter_offsets(gen, x.shape[0], x.shape[1], cfg.gain_jitter_db) if cfg.gain_jitter_db > 0 else None
            elif arm == "A1":
                off = offs_table[torch.as_tensor(rec_u[j.numpy()]), torch.as_tensor(view_u[j.numpy()])]
            else:
                off = common_mismatch_offsets(gen, x.shape[0], x.shape[1], cfg.gain_jitter_db, mm_sd)
            xa = x if off is None else x + off.to(device)[:, :, None]
            if arm == "A3":
                xa = add_noise_floor(xa, floor_t, gen)
            inp = normalise_any(xa, mu, sd)
            target = normalise_any(x, mu, sd) if CLEAN_TARGET[arm] else inp
            loss = torch.mean((model(inp) - target) ** 2) if CLEAN_TARGET[arm] else reconstruction_loss(model, inp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            steps += 1
            tot += float(loss.detach()) * len(j)
            n += len(j)
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), "seconds": time.perf_counter() - t0})
    info = {"arm": arm, "n_recordings": int(len(np.unique(rec_of_row))), "n_windows": int(len(window_rows)),
            "n_frames": int(n_frames), "n_views": 0 if view_u is None else int(views.shape[0] * views.shape[1]),
            "training_units": int(len(rows_t)), "epochs": int(epochs), "optimizer_steps": int(steps)}
    return model.eval(), history, info


# ----------------------------------------------------------------------------------------------------------- scoring
def recording_level(window_values: np.ndarray, rec_ids: np.ndarray) -> dict:
    """Mean cached dB level of each recording (window_values: n_windows × C × …, the channels to average already
    selected), as in Stage R4: mean over channels, bins and frames, then over the recording's windows."""
    w = np.asarray(window_values, dtype=np.float64).reshape(len(window_values), -1).mean(axis=1)
    out: dict = {}
    for r in np.unique(rec_ids):
        out[r] = float(w[rec_ids == r].mean())
    return out


def calibrated_z(err: np.ndarray, cal_err: np.ndarray, higher_is_fault: bool = True) -> np.ndarray:
    """Per-fold common scale (log units, alarm iff z > 0, ties no alarm): healthy-AE z = log e − log max(cal);
    fault-AE z = log min(cal) − log e (low error = fault-like)."""
    e, c = np.log(np.asarray(err, dtype=np.float64)), np.log(np.asarray(cal_err, dtype=np.float64))
    return e - c.max() if higher_is_fault else c.min() - e


def standardised(z: np.ndarray, cal_z: np.ndarray) -> np.ndarray:
    """Two-score combination: a score standardised by its own calibration normals (mean, SD)."""
    c = np.asarray(cal_z, dtype=np.float64)
    return (np.asarray(z, dtype=np.float64) - c.mean()) / c.std(ddof=1)
