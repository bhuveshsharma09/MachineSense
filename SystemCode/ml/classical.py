"""Classical model pipelines shared by the baseline scripts (no torch, no LightGBM here)."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from ml.config import BaselineConfig


def logreg_pipeline(bc: BaselineConfig) -> Pipeline:
    """The classical reference: QuantileTransformer (normal output) + logistic regression, settings from `baselines`.
    Fitting the pipeline fits the scaler on the same (training) data only."""
    return Pipeline([
        ("scale", QuantileTransformer(n_quantiles=bc.quantile_n, output_distribution="normal", random_state=bc.seed)),
        ("lr", LogisticRegression(**bc.logreg, class_weight=bc.class_weight, random_state=bc.seed)),
    ])
