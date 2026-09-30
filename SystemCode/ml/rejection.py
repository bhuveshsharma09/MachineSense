"""Stage R0: which checks reject healthy recordings (CLAUDE.md §7). Pure functions over saved scores and saved thresholds;
no training, no threshold derivation, no file I/O (the notebook loads the artifacts).

A check fails by the decision rule's own inequalities (ml.openset.three_way / ml.arbitrator.decide; ties accepted):
msp < τ_p, maha > τ_d, ae > τ_e, b1s > τ_b. A fifth cause, "top class not normal", covers a healthy recording the classifier
confidently calls a fault: that verdict has no failed check. This module does not import `backend` (the dependency runs
backend → ml); tests show it agrees with backend.checks.checks and with ml.arbitrator.decide."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from ml.arbitrator import decide

CHECKS = ("msp", "maha", "ae", "b1s")
CAUSES = CHECKS + ("top_not_normal",)
TAU = {"msp": "tau_p", "maha": "tau_d", "ae": "tau_e", "b1s": "tau_b"}
OUTCOMES = ("healthy", "healthy_with_caution", "abstention", "known_fault")
RULE = "B_with_caution"
# A stored score and the stored threshold it defines (the order-statistic recording) can differ by one floating-point step
# after saving (seen: 1.1e-16 on MSP in the Stage 3C files); the run itself compared them as a tie, and ties are accepted.
TIE_EPS = 1e-12


def snap_ties(df: pd.DataFrame, eps: float = TIE_EPS) -> pd.DataFrame:
    """Scores within `eps` of their threshold are set to the threshold (a tie, accepted by the rule), so stored-value
    rounding does not reverse a saved decision; `tie_snapped` names the checks snapped per row. Thresholds are unchanged."""
    out = df.copy()
    snapped = [[] for _ in range(len(out))]
    for k, t in TAU.items():
        near = (np.abs(out[k].to_numpy(dtype=float) - out[t].to_numpy(dtype=float)) <= eps) & (out[k].to_numpy() != out[t].to_numpy())
        out.loc[near, k] = out.loc[near, t]
        for i in np.flatnonzero(near):
            snapped[i].append(k)
    out["tie_snapped"] = ["+".join(x) for x in snapped]
    return out


def failed_checks(msp, maha, ae, b1s, tau_p, tau_d, tau_e, tau_b) -> dict[str, np.ndarray]:
    """Per check, True where it fails (arrays or scalars; the rule's inequalities, ties accepted)."""
    return {"msp": np.asarray(msp) < np.asarray(tau_p), "maha": np.asarray(maha) > np.asarray(tau_d),
            "ae": np.asarray(ae) > np.asarray(tau_e), "b1s": np.asarray(b1s) > np.asarray(tau_b)}


def outcome(decision: str, caution: str | None) -> str:
    """The four outcomes of the full rule on one recording."""
    if decision == "healthy":
        return "healthy_with_caution" if caution else "healthy"
    return "abstention" if decision == "unknown" else "known_fault"


def attribute(df: pd.DataFrame, rule: str = RULE) -> pd.DataFrame:
    """One row per scored instance (columns: top_class, msp, maha, ae, b1s, tau_p, tau_d, tau_e, tau_b). Adds the full-rule
    decision (ml.arbitrator.decide), the outcome, a flag per failed check and "top_not_normal", the joint pattern of all
    failed checks (e.g. "ae+b1s", "none"), and the causes the rule actually used for a non-accepted verdict:
    abstention → the failing classifier checks (msp, maha); caution → the failing healthy-baseline checks (ae, b1s);
    known fault → top_not_normal."""
    out = snap_ties(df)
    rows = [decide(rule, r.top_class, float(r.msp), float(r.maha), float(r.ae), float(r.b1s),
                   {"tau_p": float(r.tau_p), "tau_d": float(r.tau_d), "tau_e": float(r.tau_e), "tau_b": float(r.tau_b)})
            for r in out.itertuples()]
    out["decision"] = [x["decision"] for x in rows]
    out["caution"] = pd.Series([x["caution"] for x in rows], index=out.index, dtype=object)  # None stays None (no caution)
    out["outcome"] = [outcome(x["decision"], x["caution"]) for x in rows]
    f = failed_checks(out.msp, out.maha, out.ae, out.b1s, out.tau_p, out.tau_d, out.tau_e, out.tau_b)
    for k in CHECKS:
        out[f"fail_{k}"] = f[k]
    out["fail_top_not_normal"] = (out.top_class != "normal").to_numpy()
    out["pattern"] = [pattern(r, CAUSES) for r in out[[f"fail_{k}" for k in CAUSES]].itertuples(index=False)]
    out["rule_causes"] = [rule_causes(o, r) for o, r in zip(out.outcome, out[[f"fail_{k}" for k in CAUSES]].itertuples(index=False))]
    return out


def pattern(flags, names=CAUSES) -> str:
    """Joint pattern of failed checks, in a fixed order ("none" if nothing failed)."""
    got = [n for n, v in zip(names, flags) if bool(v)]
    return "+".join(got) if got else "none"


def rule_causes(outcome_: str, flags) -> str:
    """The failed checks that produced a non-accepted verdict under B_with_caution ("" for plain healthy)."""
    f = dict(zip(CAUSES, (bool(v) for v in flags)))
    if outcome_ == "abstention":
        use = ("msp", "maha")
    elif outcome_ == "healthy_with_caution":
        use = ("ae", "b1s")
    elif outcome_ == "known_fault":
        use = ("top_not_normal",)
    else:
        return ""
    return "+".join(k for k in use if f[k])


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for k of n (for rates from few recordings, CLAUDE.md §5 rule 9)."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": [lo, hi]}


def per_recording(att: pd.DataFrame) -> pd.DataFrame:
    """Distinct-recording view of instance rows: in how many instances each recording was scored, accepted as plain
    healthy, given a caution, abstained on or called a fault, and which checks ever failed."""
    g = att.groupby("rec_id")
    out = pd.DataFrame({"instances": g.size()})
    for o in OUTCOMES:
        out[o] = g.outcome.apply(lambda s, o=o: int((s == o).sum()))
    for k in CAUSES:
        out[f"ever_fail_{k}"] = g[f"fail_{k}"].any()
    out["ever_rejected"] = out.instances > out.healthy
    return out.reset_index()


def joint_patterns(att: pd.DataFrame, rejected_only: bool = True) -> pd.DataFrame:
    """Counts of each joint failed-check pattern, by outcome (instances and distinct recordings)."""
    a = att[att.outcome != "healthy"] if rejected_only else att
    if a.empty:
        return pd.DataFrame(columns=["outcome", "pattern", "instances", "recordings"])
    return (a.groupby(["outcome", "pattern"]).agg(instances=("rec_id", "size"), recordings=("rec_id", "nunique"))
            .reset_index().sort_values(["outcome", "instances"], ascending=[True, False]))


def standalone(att: pd.DataFrame) -> dict:
    """Each check on its own against its own threshold (instances and distinct recordings failing it)."""
    return {k: {"instances": rate(int(att[f"fail_{k}"].sum()), len(att)),
                "recordings": rate(int(att.groupby("rec_id")[f"fail_{k}"].any().sum()), att.rec_id.nunique())}
            for k in CAUSES}


def speed_band(speed: pd.Series, lo: float, hi: float) -> pd.Series:
    """'below' / 'inside' / 'above' a speed range [lo, hi] (Hz)."""
    return pd.Series(np.where(speed < lo, "below", np.where(speed > hi, "above", "inside")), index=speed.index)
