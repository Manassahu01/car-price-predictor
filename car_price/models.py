"""Candidate models and their hyperparameter search spaces.

Every candidate is wrapped in the same end-to-end pipeline
(``features.build_estimator``), so the comparison is apples-to-apples.
Parameter names use the ``regressor__model__`` prefix: the TransformedTargetRegressor
holds the Pipeline (``regressor``), and the Pipeline's last step is named ``model``.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from car_price import config
from car_price.features import build_estimator

P = "regressor__model__"
SEED = config.RANDOM_STATE

BASELINE = "Baseline (median)"


def candidate_specs() -> dict[str, tuple[object, dict]]:
    """name -> (bare sklearn model, parameter grid)."""
    return {
        BASELINE: (DummyRegressor(strategy="median"), {}),
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {P + "alpha": np.logspace(-2, 3, 11)}),
        "Lasso": (Lasso(max_iter=50_000), {P + "alpha": np.logspace(-5, -1, 9)}),
        "ElasticNet": (
            ElasticNet(max_iter=50_000),
            {P + "alpha": [1e-4, 1e-3, 1e-2], P + "l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
            {P + "min_samples_leaf": [1, 3], P + "max_features": [0.33, 0.6]},
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingRegressor(max_iter=400, early_stopping=False, random_state=SEED),
            {
                P + "learning_rate": [0.05, 0.1],
                P + "max_leaf_nodes": [15, 31],
                P + "l2_regularization": [0.0, 1.0],
            },
        ),
    }


def build_candidates() -> dict[str, tuple[object, dict]]:
    """name -> (full pipeline estimator, parameter grid)."""
    return {
        name: (build_estimator(model), grid)
        for name, (model, grid) in candidate_specs().items()
    }


def build_named_estimator(name: str, params: dict | None = None):
    """Rebuild a single candidate with fixed hyperparameters (used to refit
    the chosen model without re-running the whole search)."""
    specs = candidate_specs()
    if name not in specs:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(specs)}.")
    estimator = build_estimator(specs[name][0])
    if params:
        estimator.set_params(**params)
    return estimator
