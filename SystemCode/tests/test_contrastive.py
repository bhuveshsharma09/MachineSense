"""Stage R3: the pair rule, the loss on a hand-checked case, the frozen encoder, and fit-set-only normalisation."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from ml.contrastive import (FitRecordings, FrozenEncoderClassifier, PairStats, fit_recordings, positive_mask,
                            sample_batch, supcon_loss)


def synthetic_fit(n_classes: int = 4, recs: int = 30, folders: int = 3, n_win: int = 9, seed: int = 0) -> FitRecordings:
    rng = np.random.default_rng(seed)
    n = n_classes * recs
    cls = np.repeat(np.arange(n_classes), recs)
    folder = np.array([f"c{c}/f{i % folders}" for c, i in zip(cls, range(n))])
    return FitRecordings(rec_id=np.array([f"r{i}" for i in range(n)]), cls=cls, folder=folder,
                         speed=rng.uniform(12, 40, n), rows=np.arange(n * n_win).reshape(n, n_win))


def test_sampler_never_emits_a_same_recording_positive():
    fr = synthetic_fit()
    rng = np.random.default_rng(1)
    for _ in range(200):
        rec, pos = sample_batch(fr, rng, per_class=24)
        assert len(np.unique(rec)) == len(rec)                      # one window per recording, no repeats
        assert np.bincount(fr.cls[rec]).max() <= 24
        assert all(p in fr.rows[r] for r, p in zip(rec, pos))       # the window belongs to its recording
        pm, cand = positive_mask(torch.as_tensor(fr.cls[rec]), torch.as_tensor(rec))
        i, j = np.nonzero(pm.numpy())
        assert (fr.rec_id[rec[i]] != fr.rec_id[rec[j]]).all() and (fr.cls[rec[i]] == fr.cls[rec[j]]).all()


def test_mask_excludes_same_recording_pairs_even_when_a_recording_repeats():
    labels = torch.tensor([0, 0, 0, 1])
    groups = torch.tensor([7, 7, 8, 9])                               # windows 0 and 1: same recording
    pos, cand = positive_mask(labels, groups)
    assert not pos[0, 1] and not pos[1, 0] and not cand[0, 1] and not cand[0, 0]
    assert pos[0, 2] and pos[1, 2] and cand[0, 3] and not pos[0, 3]


def test_small_classes_are_taken_whole_and_folders_are_balanced():
    fr = synthetic_fit(n_classes=2, recs=10)
    rec, _ = sample_batch(fr, np.random.default_rng(0), per_class=24)
    assert np.bincount(fr.cls[rec]).tolist() == [10, 10]
    folder = np.array(["a"] * 18 + ["b"] * 2)                        # one class, a big and a small folder
    fr2 = FitRecordings(rec_id=np.arange(20).astype(str), cls=np.zeros(20, int), folder=folder,
                        speed=np.linspace(12, 40, 20), rows=np.arange(20 * 9).reshape(20, 9))
    rng = np.random.default_rng(3)
    share_b = np.mean([np.isin(sample_batch(fr2, rng, 1)[0], [18, 19]).mean() for _ in range(4000)])
    assert abs(share_b - 0.5) < 0.03


def test_loss_on_a_hand_checked_case():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels, groups = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2])
    # anchors 0 and 1: one positive (sim 1) among candidates {sim 1, sim 0}; anchor 2 has no positive and is skipped
    expected = math.log(math.e + 1) - 1
    assert supcon_loss(z, labels, groups, 1.0).item() == pytest.approx(expected, abs=1e-6)
    # a second window of recording 0 is neither a positive nor a candidate for anchor 0
    z4 = torch.cat([z, torch.tensor([[1.0, 0.0]])])
    l4 = supcon_loss(z4, torch.tensor([0, 0, 1, 0]), torch.tensor([0, 1, 2, 0]), 1.0).item()
    # anchors 0 and 3 (recording 0): candidates {1, 2}, positive {1} → expected; anchor 1: candidates {0, 2, 3},
    # positives {0, 3} → −(1 − log(2e + 1))
    a1 = -(1 - math.log(2 * math.e + 1))
    assert l4 == pytest.approx((2 * expected + a1) / 3, abs=1e-6)


def test_pair_stats_count_only_cross_recording_positives():
    fr = synthetic_fit()
    st = PairStats(1.0)
    rec, _ = sample_batch(fr, np.random.default_rng(2), 24)
    st.add(fr, rec)
    s = st.summary()
    assert s["positive_pairs"] == 4 * (24 * 23 // 2) and s["same_recording_positives"] == 0
    assert 0 < s["different_folders"] <= s["positive_pairs"]


def test_frozen_encoder_does_not_change():
    from ml.config import load_config
    from ml.models.cnn import ScratchCNN
    from ml.paths import DEFAULT_CONFIG

    cc = load_config(DEFAULT_CONFIG).cnn
    net = ScratchCNN(6, 10, cc)
    before = {k: v.clone() for k, v in net.state_dict().items() if not k.startswith("head")}
    m = FrozenEncoderClassifier(net).train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        logits, _ = m(torch.randn(8, 6, 128, 25))
        loss = logits.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    after = {k: v for k, v in net.state_dict().items() if not k.startswith("head")}
    assert all(torch.equal(before[k], after[k]) for k in before)     # weights and BatchNorm running stats unchanged
    assert not net.features.training and net.head.training


def test_normalisation_uses_the_fit_folds_only():
    """fold_norm_stats (used by the script) reads only the other folds' windows: changing every window of the evaluated
    fold leaves the statistics unchanged; changing one fit window changes them; fit recordings exclude fold k."""
    from ml.config import load_config
    from ml.datasets import fold_fit_context, fold_norm_stats
    from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, results_dir
    from ml.splits import development_pool

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns).merge(
        pd.read_csv(results_dir() / cfg.development.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    index = pd.DataFrame({"rec_id": np.repeat(sorted(pool.rec_id), 2), "window_idx": np.tile([0, 1], len(pool))})
    X = np.random.default_rng(0).normal(size=(len(index), 6, 2, 2)).astype(np.float32)
    k = 1
    base = fold_norm_stats(X, index, meta, pool, k)
    ev = index.rec_id.isin(set(pool.loc[pool.dev_fold == k, "rec_id"])).to_numpy()
    X2 = X.copy(); X2[ev] += 100.0
    assert np.allclose(fold_norm_stats(X2, index, meta, pool, k)["mean"], base["mean"])
    X3 = X.copy(); X3[np.flatnonzero(~ev)[0]] += 100.0
    assert not np.allclose(fold_norm_stats(X3, index, meta, pool, k)["mean"], base["mean"])
    fit_ids, _ = fold_fit_context(pool, k)
    rows = index.merge(pool[["rec_id", "class10", "folder"]], on="rec_id")
    fr = fit_recordings(rows, fit_ids, {c: i for i, c in enumerate(sorted(pool.class10.unique()))},
                        pool.set_index("rec_id").fr_measured_hz)
    assert set(fr.rec_id) == set(fit_ids) and not set(fr.rec_id) & set(pool.loc[pool.dev_fold == k, "rec_id"])
