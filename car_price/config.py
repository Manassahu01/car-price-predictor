"""Central configuration: paths, column names, and modelling constants.

Keeping these in one place means the training script, the prediction code,
the app and the tests can never disagree about column names or file paths.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "car_price_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
TARGET = "Price"

NUMERIC_INPUTS = ["Mileage", "EngineV", "Year"]
CATEGORICAL_INPUTS = ["Brand", "Model", "Body", "Engine Type", "Registration"]
RAW_FEATURES = CATEGORICAL_INPUTS + NUMERIC_INPUTS  # what a user/app must supply

# Columns produced by feature engineering (see features.engineer_features)
ENGINEERED_NUMERIC = ["Mileage", "EngineV", "car_age", "mileage_per_year"]
ENGINEERED_CATEGORICAL = CATEGORICAL_INPUTS

# --------------------------------------------------------------------------
# Data-quality rules
# --------------------------------------------------------------------------
# Engine volumes above this are placeholders / entry errors in the source data
# (values such as 15, 55, 74, 99.99 for passenger cars). They are treated as
# missing and imputed, not dropped, so no rows are lost.
MAX_VALID_ENGINE_VOLUME = 10.0

# The newest listing in the dataset is from 2016; car age is measured from it.
REFERENCE_YEAR = 2016

# Categories seen fewer than this many times are pooled into one "rare" bucket
# (the Model column has 312 values, more than half with fewer than 5 rows).
MIN_CATEGORY_FREQUENCY = 5

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = "neg_root_mean_squared_log_error"  # price errors are relative

# Prediction interval: empirical quantiles of out-of-fold log-residuals.
INTERVAL_LOWER_Q = 0.10
INTERVAL_UPPER_Q = 0.90
