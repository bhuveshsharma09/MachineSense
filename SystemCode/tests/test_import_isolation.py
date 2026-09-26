"""No file may import both lightgbm and torch: duplicate OpenMP runtimes hang or crash on macOS (CLAUDE.md §2)."""
import re
from pathlib import Path

SYSTEM_CODE = Path(__file__).resolve().parents[1]
SCANNED = ("ml", "backend", "scripts")

# Any import statement, including indented (lazy) ones; torch also matches torchvision, which loads torch.
LIGHTGBM = re.compile(r"^\s*(import|from)\s+lightgbm\b", re.MULTILINE)
TORCH = re.compile(r"^\s*(import|from)\s+torch", re.MULTILINE)


def test_no_file_imports_lightgbm_and_torch():
    offenders = []
    for folder in SCANNED:
        for path in (SYSTEM_CODE / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if LIGHTGBM.search(text) and TORCH.search(text):
                offenders.append(str(path.relative_to(SYSTEM_CODE)))
    assert not offenders, f"lightgbm and torch imported together in: {offenders}"
