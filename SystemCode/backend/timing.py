"""Real per-step timings of the deterministic pipeline, measured by wrapping (not changing) the functions that
ml.inference.diagnose() calls: signal quality, the model scores, the decision rule, physics and explanations. The
wrappers only record wall-clock time into a per-request context; the functions and their results are untouched."""
from __future__ import annotations

import contextvars
import functools
import time

_steps: contextvars.ContextVar = contextvars.ContextVar("ms_steps", default=None)
_installed = False


def _wrap(module, name: str, step: str):
    fn = getattr(module, name)
    if getattr(fn, "_ms_timed", False):
        return

    @functools.wraps(fn)
    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc = _steps.get()
            if acc is not None:
                acc[step] = acc.get(step, 0.0) + 1000 * (time.perf_counter() - t0)
    timed._ms_timed = True
    setattr(module, name, timed)


def install() -> None:
    global _installed
    if _installed:
        return
    import ml.inference as inf
    import ml.xai as xai
    _wrap(inf, "_signal_quality", "signal_quality")
    _wrap(inf, "score_recording", "models")
    _wrap(inf, "decide", "decision_rule")
    _wrap(inf, "rule_comparison", "decision_rule")
    _wrap(inf, "_physics", "physics")
    _wrap(xai, "explain", "explanations")
    _installed = True


class measure:
    """with measure() as steps: ... → steps is {step: milliseconds}."""
    def __enter__(self):
        self.steps = {}
        self.token = _steps.set(self.steps)
        return self.steps

    def __exit__(self, *exc):
        _steps.reset(self.token)
