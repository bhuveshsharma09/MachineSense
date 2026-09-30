"""Stage R3: supervised contrastive pretraining of the CNN-6 encoder (CLAUDE.md §7). The loss, the pair sampler, the
projection head and the frozen-encoder classifier; the stage-A training function reuses ml.train (batch preparation,
augmentation, normalisation, schedule). No file I/O.

Pair rule (declared): a positive pair is two windows of the same class10 from DIFFERENT recordings; windows of the same
recording are neither positives nor candidates in the denominator; one view per window. Negatives are every window of a
different class10 in the batch (folders are class-specific, so negatives always cross folders)."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict
from torch import nn

from ml.train import lr_lambda, prepare_batch, to_device_f32


class ContrastiveConfig(BaseModel):
    """configs/contrastive.yaml. Augmentation field names mirror CnnConfig so ml.train.prepare_batch applies CNN-6's."""
    model_config = ConfigDict(extra="forbid")
    seed: int
    temperature: float
    proj_hidden: int
    proj_dim: int
    recs_per_class: int
    epochs: int
    head_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    gain_jitter_db: float
    time_shift_frames: int
    time_mask_frames: int
    eval_batch_size: int
    mps_memory_fraction: float
    speed_diff_hz: float
    time_guard_minutes: float


def load_contrastive_config(path: Path) -> ContrastiveConfig:
    return ContrastiveConfig(**yaml.safe_load(Path(path).read_text()))


# ----------------------------------------------------------------------------------------------------------- loss
def positive_mask(labels: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(positives, candidates), both (B, B) bool: candidates = pairs from different recordings (`groups`); positives =
    candidates with the same label. The diagonal and every same-recording pair are excluded from both."""
    diff_group = groups[:, None] != groups[None, :]
    return diff_group & (labels[:, None] == labels[None, :]), diff_group


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020, L_out) on L2-normalised projections z (B, d), with positives and
    the denominator restricted by `positive_mask`. Anchors without a positive are skipped; mean over the other anchors."""
    pos, cand = positive_mask(labels, groups)
    sim = (z @ z.T) / temperature
    lse = torch.logsumexp(sim.masked_fill(~cand, float("-inf")), dim=1)
    n_pos = pos.sum(1)
    keep = n_pos > 0
    if not bool(keep.any()):
        return sim.sum() * 0.0
    log_prob = torch.where(pos, sim - lse[:, None], torch.zeros_like(sim))
    return -(log_prob.sum(1)[keep] / n_pos[keep]).mean()


# ----------------------------------------------------------------------------------------------------------- models
class ProjectionHead(nn.Module):
    """Training-time loss component only (not part of the encoder; discarded after stage A): 128 → 128 → d, L2-normalised."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(z), dim=1)


def encode(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The CNN-6 embedding z of a ScratchCNN (features → global average pool → embed), without the head."""
    return net.embed(net.features(x).mean(dim=(2, 3)))


class FrozenEncoderClassifier(nn.Module):
    """Stage B wrapper around a ScratchCNN: the encoder (features, embed) is frozen, no gradients, BatchNorm kept in eval
    mode even when the wrapper is put in training mode; only the head trains. forward returns (logits, z) as ScratchCNN."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        for p in list(net.features.parameters()) + list(net.embed.parameters()):
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.features.eval()
        self.net.embed.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = encode(self.net, x)
        return self.net.head(z), z


# ----------------------------------------------------------------------------------------------------------- sampler
@dataclass(frozen=True)
class FitRecordings:
    """The fit recordings of one fold: rec_id, class index, folder, measured speed (Hz), and the cache row positions of
    their windows (n_rec × n_windows)."""
    rec_id: np.ndarray
    cls: np.ndarray
    folder: np.ndarray
    speed: np.ndarray
    rows: np.ndarray

    @property
    def weight(self) -> np.ndarray:
        """1 / (fit recordings in the recording's folder): folders equally represented within a class."""
        f = pd.Series(self.folder)
        return (1.0 / f.map(f.value_counts())).to_numpy()


def fit_recordings(rows: pd.DataFrame, fit_ids, class_index: dict[str, int], speed: pd.Series) -> FitRecordings:
    """FitRecordings from the window table `rows` (dev_window_rows order: rec_id, window_idx; positions = row index of the
    resident cache) restricted to `fit_ids`. Every recording must have the same number of windows."""
    sel = rows[rows.rec_id.isin(set(fit_ids))]
    g = sel.groupby("rec_id", sort=True)
    n_win = g.size()
    if n_win.nunique() != 1:
        raise ValueError("fit recordings have different numbers of windows")
    pos = np.stack([idx.to_numpy() for _, idx in g.groups.items()])
    first = g.first()
    return FitRecordings(rec_id=first.index.to_numpy(), cls=first.class10.map(class_index).to_numpy(),
                         folder=first.folder.to_numpy(), speed=speed.loc[first.index].to_numpy(dtype=float), rows=pos)


def sample_batch(fr: FitRecordings, rng: np.random.Generator, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """One class-balanced batch: per class, `per_class` recordings without replacement (all if fewer), probability ∝
    folder weight; one uniformly chosen window each. Returns (recording indices into fr, cache row positions)."""
    w = fr.weight
    picks = []
    for c in np.unique(fr.cls):
        idx = np.flatnonzero(fr.cls == c)
        k = min(per_class, len(idx))
        picks.append(rng.choice(idx, size=k, replace=False, p=w[idx] / w[idx].sum()))
    rec = np.concatenate(picks)
    win = rng.integers(0, fr.rows.shape[1], len(rec))
    return rec, fr.rows[rec, win]


class PairStats:
    """Accumulates the realised positive pairs: total, different recordings, different folders, different speeds
    (|Δ| ≥ speed_diff_hz), and a 0.1 Hz histogram of |Δ speed| for the median."""

    def __init__(self, speed_diff_hz: float):
        self.thr = speed_diff_hz
        self.n = self.diff_rec = self.diff_folder = self.diff_speed = 0
        self.hist = np.zeros(601, np.int64)

    def add(self, fr: FitRecordings, rec: np.ndarray) -> None:
        cls, grp = fr.cls[rec], rec
        pos = (cls[:, None] == cls[None, :]) & (grp[:, None] != grp[None, :])
        i, j = np.nonzero(np.triu(pos, 1))
        d = np.abs(fr.speed[rec[i]] - fr.speed[rec[j]])
        self.n += len(i)
        self.diff_rec += int((fr.rec_id[rec[i]] != fr.rec_id[rec[j]]).sum())
        self.diff_folder += int((fr.folder[rec[i]] != fr.folder[rec[j]]).sum())
        self.diff_speed += int((d >= self.thr).sum())
        self.hist += np.bincount(np.minimum((d * 10).astype(int), 600), minlength=601)

    def summary(self) -> dict:
        cum = np.cumsum(self.hist)
        med = float(np.searchsorted(cum, cum[-1] / 2) / 10) if self.n else float("nan")
        return {"positive_pairs": int(self.n), "different_recordings": self.diff_rec, "different_folders": self.diff_folder,
                "different_speeds": self.diff_speed, "speed_diff_hz": self.thr,
                "frac_different_recordings": self.diff_rec / self.n if self.n else None,
                "frac_different_folders": self.diff_folder / self.n if self.n else None,
                "frac_different_speeds": self.diff_speed / self.n if self.n else None,
                "median_abs_speed_diff_hz": med, "same_recording_positives": int(self.n - self.diff_rec)}


# ----------------------------------------------------------------------------------------------------------- stage A
def train_supcon(net: nn.Module, cache: torch.Tensor, fr: FitRecordings, channels: list[int], mean: np.ndarray,
                 std: np.ndarray, cfg: ContrastiveConfig, device: torch.device, steps_per_epoch: int,
                 on_epoch: Callable[[int, dict], None] | None = None) -> tuple[nn.Module, nn.Module, list, dict]:
    """Stage A: SupCon on the encoder of `net` (a fresh ScratchCNN on `device`; its head is not trained here) plus a
    projection head, for cfg.epochs × steps_per_epoch class-balanced batches from `fr`. Batches go through
    ml.train.prepare_batch (float32, CNN-6's augmentations, global per-channel normalisation). Returns the encoder
    network, the projection head, the per-epoch history and the realised pair statistics."""
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ch = torch.tensor(channels, device=device)
    mu, sd = to_device_f32(mean, device), to_device_f32(std, device)
    proj = ProjectionHead(net.embed[0].out_features, cfg.proj_hidden, cfg.proj_dim).to(device)
    params = list(net.features.parameters()) + list(net.embed.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup_epochs * steps_per_epoch, total))
    stats = PairStats(cfg.speed_diff_hz)
    labels_all = torch.as_tensor(fr.cls, dtype=torch.long)
    history = []
    for ep in range(cfg.epochs):
        net.train()
        proj.train()
        t0 = time.perf_counter()
        tot = 0.0
        for _ in range(steps_per_epoch):
            rec, pos = sample_batch(fr, rng, cfg.recs_per_class)
            stats.add(fr, rec)
            xb = prepare_batch(cache, torch.as_tensor(pos, dtype=torch.long).to(device), ch, mu, sd, cfg, gen)
            loss = supcon_loss(proj(encode(net, xb)), labels_all[rec].to(device),
                               torch.as_tensor(rec, dtype=torch.long).to(device), cfg.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        secs = time.perf_counter() - t0
        rec_ = {"epoch": ep, "supcon_loss": tot / steps_per_epoch, "seconds": secs, "lr_end": sched.get_last_lr()[0]}
        history.append(rec_)
        if on_epoch:
            on_epoch(ep, rec_)
    return net.eval(), proj.eval(), history, stats.summary()


def steps_per_epoch(n_fit_windows: int, batch_size: int) -> int:
    return math.ceil(n_fit_windows / batch_size)
