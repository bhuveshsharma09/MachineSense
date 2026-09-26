"""LightGBM and torch never share a process: duplicate OpenMP runtimes hang or crash on macOS (CLAUDE.md §2).

Two layers: a static scan for files that import both, and subprocess checks that LightGBM entry points don't
pull torch in indirectly (for example through an ml.* module).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SYSTEM_CODE = Path(__file__).resolve().parents[1]
SCANNED = ("ml", "backend", "scripts")
BASELINES_SCRIPT = SYSTEM_CODE / "scripts" / "05_baselines.py"
BASELINES_NOTEBOOK = SYSTEM_CODE / "notebooks" / "03_ml_layer" / "05_baselines.ipynb"
# ml modules a LightGBM script may use; none may import torch at module level.
TORCH_FREE_ML = ("ml", "ml.config", "ml.labels", "ml.paths", "ml.preprocess", "ml.splits", "ml.utils", "ml.bundle",
                 "ml.evaluation", "ml.q1")

# Any import statement, including indented (lazy) ones; torch also matches torchvision, which loads torch.
LIGHTGBM = re.compile(r"^\s*(import|from)\s+lightgbm\b", re.MULTILINE)
TORCH = re.compile(r"^\s*(import|from)\s+torch", re.MULTILINE)
IMPORT_LINE = re.compile(r"^\s*(import|from)\s+\S.*$", re.MULTILINE)


def _torch_loaded_after(code: str) -> bool:
    """Run `code` in a fresh interpreter and report whether torch ended up in sys.modules."""
    probe = f"{code}\nimport sys, json\nprint(json.dumps('torch' in sys.modules))"
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, cwd=SYSTEM_CODE)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _notebook_code(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_no_file_imports_lightgbm_and_torch():
    offenders = []
    for folder in SCANNED:
        for path in (SYSTEM_CODE / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if LIGHTGBM.search(text) and TORCH.search(text):
                offenders.append(str(path.relative_to(SYSTEM_CODE)))
    if BASELINES_NOTEBOOK.exists():
        text = _notebook_code(BASELINES_NOTEBOOK)
        if TORCH.search(text):
            offenders.append(str(BASELINES_NOTEBOOK.relative_to(SYSTEM_CODE)))
    assert not offenders, f"lightgbm and torch imported together in: {offenders}"


def test_light_ml_modules_do_not_import_torch():
    assert not _torch_loaded_after("\n".join(f"import {m}" for m in TORCH_FREE_ML))


def test_baselines_script_does_not_pull_torch():
    if not BASELINES_SCRIPT.exists():
        pytest.skip("scripts/05_baselines.py does not exist yet")
    # Execute the module body (imports and definitions) without running main().
    code = ("import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('baselines', {str(BASELINES_SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)")
    assert not _torch_loaded_after(code)


def test_baselines_notebook_imports_do_not_pull_torch():
    if not BASELINES_NOTEBOOK.exists():
        pytest.skip("notebooks/03_ml_layer/05_baselines.ipynb does not exist yet")
    imports = [m.group(0).strip() for m in IMPORT_LINE.finditer(_notebook_code(BASELINES_NOTEBOOK))]
    assert not _torch_loaded_after("\n".join(imports))
