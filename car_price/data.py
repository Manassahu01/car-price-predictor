"""Data loading, schema validation and row-level cleaning.

Only *row filtering that is safe to do before the train/test split* lives
here (duplicates, missing target). Anything that learns from data (imputation
values, encodings, scaling) lives inside the sklearn Pipeline in
``features.py`` so it is fitted on training folds only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from car_price import config

REQUIRED_COLUMNS = config.RAW_FEATURES + [config.TARGET]


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV at data/raw/car_data.csv."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early, with a readable message, if the data is not what we expect."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in config.NUMERIC_INPUTS + [config.TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric, got {df[col].dtype}.")

    if len(df) == 0:
        raise ValueError("Dataset is empty.")


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that cannot be used for training and report what was dropped.

    Steps (all independent of any train/test split):
      1. remove exact duplicate rows - otherwise the same listing can land in
         both train and test and inflate the scores;
      2. remove rows with a missing or non-positive target (cannot train on them).

    Rows with a missing ``EngineV`` are deliberately KEPT; the pipeline imputes
    them. Expensive cars are also kept - they are genuine listings, and
    trimming them would make the reported metrics look better than reality.

    Returns the cleaned frame (index reset) and a data-quality report.
    """
    report = {"rows_raw": int(len(df))}

    deduped = df.drop_duplicates()
    report["duplicates_removed"] = int(len(df) - len(deduped))

    has_target = deduped[config.TARGET].notna() & (deduped[config.TARGET] > 0)
    report["missing_or_invalid_price_removed"] = int((~has_target).sum())
    cleaned = deduped[has_target].reset_index(drop=True)

    report["engine_volume_missing_kept"] = int(cleaned["EngineV"].isna().sum())
    report["engine_volume_invalid_kept"] = int(
        (cleaned["EngineV"] > config.MAX_VALID_ENGINE_VOLUME).sum()
    )
    report["rows_final"] = int(len(cleaned))
    return cleaned, report


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target."""
    return df[config.RAW_FEATURES].copy(), df[config.TARGET].copy()
