"""MachineSense: reusable logic for data, features, models and inference."""
import os

# Must run before anything imports torch: unsupported MPS ops then fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
