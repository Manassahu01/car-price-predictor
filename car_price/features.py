"""Feature engineering and the preprocessing pipeline.

Everything that touches the raw input is inside the sklearn Pipeline, so the
exact same transformations run at training time and at prediction time
(no train/serve skew), and anything learned from data (medians, encodings,
scaling) is fitted on the training folds only (no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from car_price import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw listing columns into model-ready columns.

    * ``EngineV``: implausible values (> MAX_VALID_ENGINE_VOLUME or <= 0) become
      NaN so the imputer handles them, instead of distorting the model.
    * ``car_age``: years between the listing and REFERENCE_YEAR (replaces the
      raw ``Year``, which is perfectly collinear with it).
    * ``mileage_per_year``: usage intensity, a classic used-car price driver.
    """
    out = pd.DataFrame(index=df.index)

    mileage = pd.to_numeric(df["Mileage"], errors="coerce")
    out["Mileage"] = mileage.where(mileage >= 0)

    engine = pd.to_numeric(df["EngineV"], errors="coerce")
    out["EngineV"] = engine.where((engine > 0) & (engine <= config.MAX_VALID_ENGINE_VOLUME))

    year = pd.to_numeric(df["Year"], errors="coerce")
    out["car_age"] = (config.REFERENCE_YEAR - year).clip(lower=0)
    out["mileage_per_year"] = out["Mileage"] / out["car_age"].clip(lower=1)

    for col in config.ENGINEERED_CATEGORICAL:
        out[col] = df[col].astype("object")

    return out[config.ENGINEERED_NUMERIC + config.ENGINEERED_CATEGORICAL]


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, one-hot encode categorical columns.

    One-hot encoding (rather than label encoding) is used on purpose: label
    encoding would tell a linear model that Audi < BMW < Mercedes, an ordering
    that does not exist. Rare categories (< MIN_CATEGORY_FREQUENCY rows) are
    pooled, and categories never seen in training are mapped to that same
    bucket instead of crashing at prediction time.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=config.MIN_CATEGORY_FREQUENCY,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.ENGINEERED_NUMERIC),
            ("cat", categorical, config.ENGINEERED_CATEGORICAL),
        ],
        remainder="drop",
    )


def build_estimator(model) -> TransformedTargetRegressor:
    """Full end-to-end estimator: raw columns in, price (in currency units) out.

    The target is modelled on a log scale (prices are right-skewed and errors
    are relative), and the wrapper converts predictions back automatically, so
    callers never have to remember to apply ``exp()``.
    """
    pipeline = Pipeline(
        steps=[
            ("engineer", FunctionTransformer(engineer_features)),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
