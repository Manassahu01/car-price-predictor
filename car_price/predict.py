"""Prediction API used by the Streamlit app (and usable from any script).

    from car_price.predict import CarSpec, load_model, predict_price
    model, meta = load_model()
    result = predict_price(model, meta, CarSpec(brand="BMW", model="320", ...))
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from car_price import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarSpec:
    """One car listing, using the same fields as the training data."""

    brand: str
    model: str
    body: str
    engine_type: str
    registration: str  # "yes" / "no"
    year: int
    mileage: float
    engine_v: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Brand": self.brand,
                    "Model": self.model,
                    "Body": self.body,
                    "Engine Type": self.engine_type,
                    "Registration": self.registration,
                    "Mileage": float(self.mileage),
                    "EngineV": float(self.engine_v),
                    "Year": int(self.year),
                }
            ]
        )[config.RAW_FEATURES]


def apply_interval(pred, log_lower_offset: float, log_upper_offset: float):
    """Turn point predictions into (lower, upper) using calibrated log offsets.

    The offsets are quantiles of log(actual) - log(predicted) measured on
    out-of-fold training predictions, so the range widens in proportion to price.
    """
    base = np.log1p(np.clip(np.asarray(pred, dtype=float), 0, None))
    return np.expm1(base + log_lower_offset), np.expm1(base + log_upper_offset)


def load_metadata() -> dict:
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            "models/metadata.json not found. Run `python -m car_price.train` first."
        )
    return json.loads(config.METADATA_PATH.read_text())


def load_model():
    """Load (model, metadata). Falls back to refitting if the pickle is unusable.

    A pickled scikit-learn model only loads reliably in the version that wrote it.
    If loading fails for any reason, the chosen model is refitted from the CSV
    with its stored hyperparameters, so the app stays up instead of crashing.
    """
    metadata = load_metadata()
    try:
        model = joblib.load(config.MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure triggers the fallback
        log.warning("Could not load saved model (%s). Refitting from data.", exc)
        from car_price.train import refit_from_metadata  # lazy: avoids heavy import at startup

        model = refit_from_metadata(metadata)
    return model, metadata


def validate_spec(spec: CarSpec, metadata: dict) -> list[str]:
    """Raise ValueError for impossible input; return warnings for risky input."""
    for field_name in ("brand", "model", "body", "engine_type", "registration"):
        if not str(getattr(spec, field_name)).strip():
            raise ValueError(f"'{field_name}' is required.")
    for field_name in ("year", "mileage", "engine_v"):
        value = getattr(spec, field_name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"'{field_name}' must be a finite number.")
    if spec.mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    if spec.engine_v <= 0:
        raise ValueError("Engine volume must be greater than 0.")

    warnings: list[str] = []
    cat = metadata["catalogue"]
    ranges = cat["numeric_ranges"]

    if spec.brand not in cat["brand_to_models"]:
        warnings.append(f"Brand '{spec.brand}' was not in the training data; treat the estimate with caution.")
    elif spec.model not in cat["brand_to_models"][spec.brand]:
        warnings.append(
            f"Model '{spec.model}' was not seen for {spec.brand} in training; "
            "the estimate relies on brand, body, engine and age only."
        )

    if not ranges["Year"]["min"] <= spec.year <= ranges["Year"]["max"]:
        warnings.append(
            f"Year {spec.year} is outside the training range "
            f"({ranges['Year']['min']}-{ranges['Year']['max']}); the model is extrapolating."
        )
    if spec.mileage > ranges["Mileage"]["p99"]:
        warnings.append("Mileage is higher than 99% of the training data; the estimate is less reliable.")
    if not ranges["EngineV"]["min"] <= spec.engine_v <= ranges["EngineV"]["max"]:
        warnings.append("Engine volume is outside the range seen in training.")
    return warnings


def predict_price(model, metadata: dict, spec: CarSpec) -> dict:
    """Estimate a price and an approximate 80% range for one car."""
    warnings = validate_spec(spec, metadata)
    estimate = float(model.predict(spec.to_frame())[0])
    interval = metadata["interval"]
    lower, upper = apply_interval(estimate, interval["log_lower_offset"], interval["log_upper_offset"])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "coverage": float(interval["nominal_coverage"]),
        "warnings": warnings,
    }
