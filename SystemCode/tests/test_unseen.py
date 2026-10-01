"""Stage App-6: the "Explore an unseen fault" page — eligibility, the curated file, the API, and agreement with the scoring
and decision functions."""
import pandas as pd
import pytest

from backend.unseen import may_show, split_eligible
from ml.paths import SYSTEM_CODE_DIR


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")


def only_reserved_in(meta, col):
    """A recording that is a test recording in exactly this one reserved partition (so each rule is exercised alone)."""
    t = meta[["split_closed", "split_speed_block", "split_severity"]] == "test"
    return meta[t[col] & (t.sum(axis=1) == 1)].iloc[0]


@pytest.mark.parametrize("col", ["split_closed", "split_speed_block", "split_severity"])
def test_predicate_rejects_each_reserved_partition(meta, col):
    r = only_reserved_in(meta, col)
    assert not split_eligible(r)
    for role in ("withheld", "familiar"):
        ok, why = may_show(r.rec_id, r.class10 if r.class10 != "normal" else "imbalance", role, meta, set())
        assert not ok and "reserved" in why


def test_eligible_set_is_the_development_pool(meta):
    from ml.config import load_config
    from ml.paths import DEFAULT_CONFIG
    from ml.splits import development_pool
    cfg = load_config(DEFAULT_CONFIG)
    pool = set(development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns).rec_id)
    assert set(meta.loc[meta.apply(split_eligible, axis=1), "rec_id"]) == pool


def test_roles(meta):
    pool = meta[meta.apply(split_eligible, axis=1)]
    imb = pool[pool.class10 == "imbalance"].rec_id.iloc[0]
    hm = pool[pool.class10 == "horizontal_misalignment"].rec_id.iloc[0]
    assert may_show(imb, "imbalance", "withheld", meta, set())[0]
    assert not may_show(hm, "imbalance", "withheld", meta, set())[0]          # withheld role: the hidden class only
    assert may_show(hm, "imbalance", "familiar", meta, set())[0]
    assert not may_show(imb, "imbalance", "familiar", meta, set())[0]         # familiar role: a learned class only
    assert not may_show(hm, "imbalance", "familiar", meta, {hm})[0]           # never a calibration / defining recording
    assert not may_show("no_such_recording", "imbalance", "withheld", meta, set())[0]
    assert not may_show(imb, "normal", "withheld", meta, set())[0]             # normal is never a hidden class
