"""Scratch CNN and its training loop (Stage 3A), on CPU with synthetic data."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import ml.datasets as ds
from ml.config import CnnConfig
from ml.models.cnn import ScratchCNN, n_parameters
from ml.paths import SYSTEM_CODE_DIR
from ml.train import class_weights, normalise, prepare_batch, score_windows, train_model

CFG = CnnConfig()
CPU = torch.device("cpu")


@pytest.mark.parametrize("channels", [7, 6])
def test_forward_shapes_and_parameter_count(channels):
    model = ScratchCNN(channels, 10, CFG).eval()
    logits, z = model(torch.zeros(3, channels, 128, 25))
    assert logits.shape == (3, 10) and z.shape == (3, CFG.embedding_dim)
    assert logits.dtype == torch.float32
    assert n_parameters(model) < CFG.max_params


def test_batches_are_float32_before_normalisation():
    cache = torch.randn(10, 7, 128, 25).to(torch.float16)
    with pytest.raises(TypeError):
        normalise(cache[:2], torch.zeros(7), torch.ones(7))  # a float16 batch is refused
    seen = []
    import ml.train as tr
    orig = tr.normalise

    def spy(x, mean, std):
        seen.append(x.dtype)
        return orig(x, mean, std)

    tr.normalise = spy
    try:
        gen = torch.Generator().manual_seed(0)
        out = prepare_batch(cache, torch.arange(4), torch.arange(7), torch.zeros(7), torch.ones(7), CFG, gen)
    finally:
        tr.normalise = orig
    assert seen == [torch.float32] and out.dtype == torch.float32


def test_eval_batches_are_not_augmented():
    cache = torch.randn(6, 7, 128, 25).to(torch.float16)
    mean, std = torch.zeros(7), torch.ones(7)
    a = prepare_batch(cache, torch.arange(6), torch.arange(7), mean, std, CFG, None)
    assert torch.equal(a, cache.to(torch.float32))


def test_tiny_overfit():
    torch.manual_seed(0)
    n = 20
    cache = torch.randn(n, 7, 128, 25).to(torch.float16)
    y = torch.arange(n) % 10
    cfg = CFG.model_copy(update={"epochs": 60, "batch_size": 20, "lr": 3e-3, "warmup_epochs": 1, "dropout": 0.0})
    model, hist = train_model(cache, y, np.arange(n), list(range(7)), np.zeros(7), np.ones(7), 10, cfg, CPU,
                              augment=False)
    logits, emb = score_windows(model, cache, np.arange(n), list(range(7)), np.zeros(7), np.ones(7), cfg, CPU)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert (logits.argmax(1) == y.numpy()).mean() == 1.0
    assert logits.dtype == np.float32 and emb.shape == (n, CFG.embedding_dim)


def test_class_weights_from_training_windows():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert np.allclose(w, [4 / 6, 2.0])
    with pytest.raises(ValueError):
        class_weights(np.array([0, 0]), 2)


def test_fold_norm_stats_uses_development_mode_with_the_fold_ids(monkeypatch):
    pool = pd.DataFrame({"rec_id": list("abcdef"), "class10": ["normal"] * 6, "folder": ["normal/-"] * 6,
                         "dev_fold": [0, 1, 2, 0, 1, 2]})
    calls = []
    monkeypatch.setattr(ds, "norm_stats", lambda X, index, meta, **kw: calls.append(kw) or {})
    ds.fold_norm_stats(None, None, None, pool, fold=1)
    (kw,) = calls
    assert set(kw) == {"fit_ids", "context"}  # development mode: no track=
    assert sorted(kw["fit_ids"]) == ["a", "c", "d", "f"]
    assert kw["context"].evaluation == frozenset({"b", "e"})
    assert not kw["context"].calibration and not kw["context"].withheld


def test_dev_window_rows_order_and_positions():
    index = pd.DataFrame({"rec_id": ["b", "b", "a", "a", "z"], "window_idx": [1, 0, 0, 1, 0]})
    pool = pd.DataFrame({"rec_id": ["a", "b"], "class10": ["normal", "imbalance"], "folder": ["n", "i"],
                         "dev_fold": [0, 1]})
    rows = ds.dev_window_rows(index, pool)
    assert rows.rec_id.tolist() == ["a", "a", "b", "b"] and rows.row.tolist() == [2, 3, 1, 0]
    with pytest.raises(ValueError):
        ds.dev_window_rows(index, pd.concat([pool, pool.assign(rec_id="q")]))


def test_cnn_training_process_does_not_import_lightgbm():
    script = SYSTEM_CODE_DIR / "scripts" / "06_train_cnn.py"
    code = ("import importlib.util, sys, json\n"
            "import ml.train, ml.models.cnn, ml.datasets, ml.evaluation, ml.classical\n"
            f"spec = importlib.util.spec_from_file_location('cnn_script', {str(script)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "print(json.dumps('lightgbm' in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=SYSTEM_CODE_DIR)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) is False
