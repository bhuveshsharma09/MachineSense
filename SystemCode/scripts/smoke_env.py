"""
smoke_env.py — day-1 environment check for MachineSense.

Run it on the machine that will train the models (the Mac mini):

    python scripts/smoke_env.py                          # platform, memory, imports, MPS, benchmarks
    python scripts/smoke_env.py --data data/mafaulda     # + free disk there, CSV count, one real CSV timed
    python scripts/smoke_env.py --resnet                 # + a ResNet-18 training step (worst case)

Every check prints PASS / WARN / FAIL and the last line says whether the machine is ready.
Nothing here writes outside a temporary directory.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Must be set before torch is imported: unsupported MPS ops then fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

RESULTS: list[tuple[str, str, str]] = []  # (status, check, detail)

EXPECTED_TOP_FOLDERS = {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment",
                        "underhang", "overhang"}
EXPECTED_FILES = 1951
WINDOWS_PER_EPOCH = 12_300      # ~70 % of 17,559 windows
BATCH = 256


def report(status: str, check: str, detail: str = "") -> None:
    RESULTS.append((status, check, detail))
    print(f"[{status:4s}] {check:34s} {detail}")


def gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


# ----------------------------------------------------------------------------- platform
def check_platform() -> None:
    py = platform.python_version()
    major, minor = sys.version_info[:2]
    report("PASS" if (major, minor) >= (3, 12) else "WARN", "python",
           f"{py} ({'ok' if (major, minor) >= (3, 12) else 'pinned numpy/scipy need 3.12+'})")

    machine = platform.machine()
    if sys.platform == "darwin":
        mac = platform.mac_ver()[0]
        if machine == "arm64":
            report("PASS", "macOS / arch", f"macOS {mac}, arm64 (Apple Silicon)")
        else:
            report("FAIL", "macOS / arch",
                   f"macOS {mac}, {machine}: this Python runs under Rosetta; install the arm64 build")
    else:
        report("WARN", "platform", f"{sys.platform} {machine}: not the Mac mini; MPS checks will be skipped")

    try:
        import psutil
        vm = psutil.virtual_memory()
        status = "PASS" if vm.total >= 30e9 else "WARN"
        report(status, "memory", f"{gb(vm.total)} total, {gb(vm.available)} available now")
    except ImportError:
        report("WARN", "memory", "psutil missing (pip install psutil)")


def check_disk(path: Path) -> None:
    probe = path if path.exists() else Path.cwd()
    usage = shutil.disk_usage(probe)
    free = usage.free
    # Still needed on top of the CSV tree: raw_npy ~16 GB + spectrogram cache 1–3 GB + MLflow runs.
    status = "PASS" if free >= 40e9 else ("WARN" if free >= 20e9 else "FAIL")
    report(status, "disk free", f"{gb(free)} at {probe} (need ~20 GB beyond the CSV tree; 40 GB comfortable)")


# ----------------------------------------------------------------------------- data
def check_data(root: Path) -> None:
    if not root.exists():
        report("FAIL", "data folder", f"{root} does not exist")
        return
    top = {p.name for p in root.iterdir() if p.is_dir()}
    missing = EXPECTED_TOP_FOLDERS - top
    report("PASS" if not missing else "FAIL", "class folders",
           f"{sorted(top)}" + (f" — missing {sorted(missing)}" if missing else ""))

    t = time.perf_counter()
    files = sorted(root.rglob("*.csv"))
    report("PASS" if len(files) == EXPECTED_FILES else "WARN", "csv count",
           f"{len(files)} files (expected {EXPECTED_FILES}), walked in {time.perf_counter() - t:.1f} s")
    if not files:
        return

    sample = files[len(files) // 2]
    try:
        import pyarrow.csv as pacsv
        t = time.perf_counter()
        table = pacsv.read_csv(sample, read_options=pacsv.ReadOptions(autogenerate_column_names=True))
        dt = time.perf_counter() - t
        shape_ok = table.num_rows == 250_000 and table.num_columns == 8
        report("PASS" if shape_ok else "WARN", "one real csv",
               f"{sample.relative_to(root)}: {table.num_rows} rows × {table.num_columns} cols in {dt:.2f} s")
        workers = max(1, (os.cpu_count() or 4) - 2)
        report("PASS", "conversion estimate",
               f"~{len(files) * dt / workers / 60:.1f} min for all files with {workers} workers "
               f"(pyarrow parse only; float32 .npy writes add ~16 GB of I/O)")
    except ImportError:
        report("FAIL", "pyarrow", "missing (pip install pyarrow)")


# ----------------------------------------------------------------------------- imports
# One line on purpose: run in a child process, so this process never loads LightGBM's OpenMP runtime.
_LGB_FIT = ("import time, numpy as np, lightgbm as lgb; rng = np.random.default_rng(0); "
            "X = rng.normal(size=(500, 20)).astype(np.float32); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int); "
            "t = time.perf_counter(); lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y); "
            "print(lgb.__version__, f'{time.perf_counter() - t:.2f}')")


def check_lightgbm() -> None:
    """LightGBM fit in a subprocess: proves libomp loads without mixing OpenMP runtimes with torch."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-c", _LGB_FIT], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        report("FAIL", "lightgbm fit (subprocess)", "timed out after 60 s")
        return
    if r.returncode == 0:
        version, secs = r.stdout.split()[-2:]
        report("PASS", "lightgbm fit (subprocess)", f"{version}: 20 trees on 500×20 in {secs} s (libomp ok)")
    else:
        err = (r.stderr.strip().splitlines() or ["no stderr"])[-1]
        hint = " — run: brew install libomp" if "libomp" in r.stderr else ""
        report("FAIL", "lightgbm fit (subprocess)", f"exit {r.returncode}: {err}"[:100] + hint)


def check_sklearn_after_torch() -> None:
    """HistGradientBoosting fit after torch has run ops: proves sklearn's and torch's OpenMP coexist."""
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 20)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        t = time.perf_counter()
        HistGradientBoostingClassifier(max_iter=50).fit(X, y)
        report("PASS", "sklearn HGB fit after torch", f"50 iters on 2000×20 in {time.perf_counter() - t:.2f} s")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "sklearn HGB fit after torch", f"{type(exc).__name__}: {exc}"[:120])


def check_imports() -> None:
    # lightgbm is deliberately absent: it must never share a process with torch (see check_lightgbm).
    wanted = ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "torchvision",
              "mlflow", "fastapi", "pydantic", "matplotlib", "yaml", "safetensors"]
    for name in wanted:
        try:
            mod = importlib.import_module(name)
            report("PASS", f"import {name}", getattr(mod, "__version__", ""))
        except Exception as exc:  # noqa: BLE001 — any import failure is the finding
            report("FAIL", f"import {name}", f"{type(exc).__name__}: {exc}"[:120])

    check_lightgbm()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(os.devnull, format="png")
        plt.close(fig)
        report("PASS", "matplotlib Agg", "headless PNG render ok")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "matplotlib Agg", f"{type(exc).__name__}: {exc}"[:120])


# ----------------------------------------------------------------------------- torch
class TinyCNN:
    """Roughly the planned scratch 2D-CNN on a (7, 128, 25) order-spectrogram tensor."""

    def __new__(cls, n_classes: int = 10):
        import torch.nn as nn
        def block(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers
        return nn.Sequential(*block(7, 32, True), *block(32, 64, True), *block(64, 128, False),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes))


def _train_steps(model, x_all, y_all, device, steps, batch):
    """Time `steps` optimiser steps with data already resident on `device`; returns ms/step."""
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = x_all.shape[0]
    sync = torch.mps.synchronize if device == "mps" else (torch.cuda.synchronize if device == "cuda" else None)

    def one_step(i):
        idx = torch.randint(0, n, (batch,), device=device)
        xb = x_all[idx].float()          # float16 cache → float32 batch, on device
        yb = y_all[idx]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        return loss

    for i in range(3):                    # warm-up: kernel compilation, allocator
        one_step(i)
    if sync:
        sync()
    t = time.perf_counter()
    for i in range(steps):
        one_step(i)
    if sync:
        sync()
    return (time.perf_counter() - t) / steps * 1000


def check_torch(run_resnet: bool) -> None:
    try:
        import torch
    except ImportError:
        report("FAIL", "torch", "missing")
        return

    report("PASS", "torch", f"{torch.__version__}, threads {torch.get_num_threads()}")
    report("PASS" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" else "WARN",
           "PYTORCH_ENABLE_MPS_FALLBACK", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"))

    # CPU STFT: the representation is built once, on CPU, so this is the cache-build cost.
    x = torch.randn(7, 50_000)
    win = torch.hann_window(8192)
    torch.stft(x, n_fft=8192, hop_length=2048, window=win, return_complex=True)  # warm-up
    t = time.perf_counter()
    spec = torch.stft(x, n_fft=8192, hop_length=2048, window=win, return_complex=True)
    dt = time.perf_counter() - t
    report("PASS", "torch.stft on CPU",
           f"(7, 50000) → {tuple(spec.shape)} in {dt * 1000:.0f} ms; "
           f"all 17,559 windows ≈ {17_559 * dt / 60:.1f} min single-threaded")

    mps = torch.backends.mps.is_available()
    if sys.platform == "darwin":
        report("PASS" if mps else "FAIL", "MPS available",
               f"built={torch.backends.mps.is_built()} available={mps}")
    else:
        report("WARN", "MPS available", "not macOS — benchmarking on CPU only")

    device = "mps" if mps else "cpu"
    if mps:
        rec = getattr(torch.mps, "recommended_max_memory", None)
        if rec:
            report("PASS", "MPS memory ceiling", f"{gb(rec())} recommended max working set")
        try:
            torch.zeros(1, dtype=torch.float64, device="mps")
            report("WARN", "float64 on MPS", "did not raise — still cast everything to float32")
        except Exception:  # noqa: BLE001 — the raise is the expected behaviour
            report("PASS", "float64 on MPS", "refused as expected: cast numpy float64 → float32 before .to('mps')")

    # In-memory "cache": the training pattern is index-slicing one resident float16 tensor, no DataLoader.
    n_cache = 4_000
    t = time.perf_counter()
    x_all = torch.randn(n_cache, 7, 128, 25, dtype=torch.float16).to(device)
    y_all = torch.randint(0, 10, (n_cache,), device=device)
    if mps:
        torch.mps.synchronize()
    report("PASS", "cache tensor on device",
           f"{n_cache} windows ({gb(x_all.numel() * 2)}) in {time.perf_counter() - t:.2f} s; "
           f"full cache 17,559 windows ≈ {gb(17_559 * 7 * 128 * 25 * 2)}")

    model = TinyCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ms = _train_steps(model, x_all, y_all, device, steps=20 if mps else 5, batch=BATCH)
    epoch_s = ms / 1000 * (WINDOWS_PER_EPOCH / BATCH)
    status = "PASS" if (not mps or epoch_s < 30) else "WARN"
    report(status, f"scratch CNN step ({device})",
           f"{n_params / 1e6:.2f} M params, batch {BATCH}: {ms:.0f} ms/step → "
           f"{BATCH / ms * 1000:.0f} windows/s → epoch ≈ {epoch_s:.1f} s, 50 epochs ≈ {epoch_s * 50 / 60:.1f} min")
    if mps:
        cpu_ms = _train_steps(TinyCNN(), x_all[:512].cpu(), y_all[:512].cpu(), "cpu", steps=3, batch=BATCH)
        report("PASS", "scratch CNN step (cpu)", f"{cpu_ms:.0f} ms/step → MPS speed-up ×{cpu_ms / ms:.1f}")
        torch.mps.empty_cache()

    if run_resnet:
        import torchvision
        rn = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
        xr = torch.randn(512, 3, 224, 224, dtype=torch.float16).to(device)
        yr = torch.randint(0, 10, (512,), device=device)
        ms = _train_steps(rn, xr, yr, device, steps=10 if mps else 2, batch=64)
        epoch_s = ms / 1000 * (WINDOWS_PER_EPOCH / 64)
        report("PASS", f"ResNet-18 step ({device})",
               f"batch 64 at 224×224 (upper bound): {ms:.0f} ms/step → epoch ≈ {epoch_s / 60:.1f} min; "
               f"10 epochs ≈ {epoch_s * 10 / 60:.0f} min")
        if mps:
            torch.mps.empty_cache()


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="folder containing normal/, imbalance/, ...")
    ap.add_argument("--resnet", action="store_true", help="also benchmark a ResNet-18 training step")
    args = ap.parse_args()

    print(f"MachineSense environment check — {platform.node()} — {time.strftime('%Y-%m-%d %H:%M')}\n")
    check_platform()
    check_disk(args.data or Path.cwd())
    if args.data:
        check_data(args.data)
    check_imports()
    check_torch(args.resnet)
    check_sklearn_after_torch()

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(RESULTS) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("Fix first: " + "; ".join(f"{c} ({d[:60]})" for _, c, d in fails))
        return 1
    print("Ready: the machine can run the MachineSense pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
