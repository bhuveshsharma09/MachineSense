"""select_demod_band on synthetic spectral-kurtosis tables."""
import pandas as pd
import pytest

from ml.config import PreprocessConfig
from ml.features import candidate_bands, select_demod_band

CFG = PreprocessConfig()
BANDS = candidate_bands(CFG)


def _data(boost: dict, n=8, classes=("normal", "underhang_outer", "overhang_outer")):
    """Recordings with SK 0 everywhere except `boost[(class, channel, band)] = value`."""
    rec, rows = [], []
    for cls in classes:
        for i in range(n):
            rid = f"{cls}_{i}"
            rec.append({"rec_id": rid, "class10": cls, "severity": 0.0})
            for ch in CFG.envelope.selection.channels:
                for lo, hi in BANDS:
                    rows.append({"rec_id": rid, "channel": ch, "lo_hz": lo, "hi_hz": hi,
                                 "sk": boost.get((cls, ch, (lo, hi)), 0.0)})
    return pd.DataFrame(rec), pd.DataFrame(rows)


def test_picks_largest_contrast_per_channel_and_group():
    b1, b2, b3 = (9000.0, 11000.0), (13000.0, 17000.0), (5000.0, 9000.0)
    rec, sk = _data({("overhang_outer", "under_radial", b1): 2.0, ("underhang_outer", "under_radial", b3): 1.0,
                     ("underhang_outer", "over_radial", b2): 4.0})
    got = select_demod_band(rec, sk, CFG)
    assert (got["under_radial"]["lo_hz"], got["under_radial"]["hi_hz"]) == b1
    assert got["under_radial"]["fault_class"] == "overhang_outer"
    assert (got["over_radial"]["lo_hz"], got["over_radial"]["hi_hz"]) == b2


def test_contrast_uses_reference():
    band = (9000.0, 11000.0)
    # Faults are impulsive in `band`, but so is normal: the contrast there is 0, so another band wins.
    rec, sk = _data({("underhang_outer", "over_radial", band): 5.0, ("normal", "over_radial", band): 5.0,
                     ("underhang_outer", "over_radial", (3000.0, 5000.0)): 0.5})
    assert select_demod_band(rec, sk, CFG)["over_radial"]["lo_hz"] == 3000.0


def test_narrow_bands_never_chosen():
    narrow = next((lo, hi) for lo, hi in [(9500.0, 10500.0)])
    rec, sk = _data({})
    sk.loc[:, "sk"] = 0.0
    extra = pd.DataFrame([{"rec_id": r, "channel": "under_radial", "lo_hz": narrow[0], "hi_hz": narrow[1],
                           "sk": 10.0 if r.startswith("under") else 0.0} for r in sk.rec_id.unique()])
    got = select_demod_band(rec, pd.concat([sk, extra]), CFG)
    assert got["under_radial"]["hi_hz"] - got["under_radial"]["lo_hz"] >= CFG.envelope.min_band_hz


def test_withheld_class_is_excluded_by_construction():
    b = (13000.0, 17000.0)
    rec, sk = _data({("underhang_outer", "over_radial", b): 4.0, ("overhang_outer", "over_radial", (1000.0, 5000.0)): 1.0})
    loco = rec[rec.class10 != "underhang_outer"]
    got = select_demod_band(loco, sk, CFG)
    assert got["over_radial"]["fault_class"] == "overhang_outer"
    assert not any(r.startswith("underhang_outer") for r in got["over_radial"]["rec_ids"])


def test_other_severities_ignored():
    rec, sk = _data({})
    rec.loc[rec.class10 != "normal", "severity"] = 20.0
    with pytest.raises(ValueError):
        select_demod_band(rec, sk, CFG)


def test_too_few_reference_recordings():
    rec, sk = _data({})
    with pytest.raises(ValueError):
        select_demod_band(rec[~rec.rec_id.isin([f"normal_{i}" for i in range(5)])], sk, CFG)


def test_missing_sk_rows_raise():
    rec, sk = _data({})
    with pytest.raises(ValueError):
        select_demod_band(rec, sk[sk.rec_id != "normal_0"], CFG)
