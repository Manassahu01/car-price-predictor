"""
Training script for the Car Price Prediction project.

Reproduces the exact preprocessing pipeline from your notebook
(Linear_Ridge_&_Lasso_Regression) and saves the model + metadata
artifacts that app.py needs to run.

USAGE
-----
1. Get the dataset. Either:
   a) Place the CSV next to this file, named "car_data.csv"
      (it's the "1.04. Real-life example.csv" file used in the notebook), OR
   b) Run this in an environment where `kagglehub` + Kaggle API
      credentials are already set up (e.g. Google Colab, like your notebook).

2. Run:
       python train_model.py

   This creates a "models/" folder containing:
       linear_regression.pkl
       ridge_regression.pkl
       lasso_regression.pkl
       metadata.json           (feature order, label encodings, R2/MAE scores)
       sample_predictions.csv  (for the "Model Insights" chart in the app)

3. Once "models/" exists, run the app with:
       streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the dataset: local CSV first, kagglehub as a fallback."""
    local_csv = Path("car_data.csv")
    if local_csv.exists():
        print(f"Loading dataset from local file: {local_csv}")
        return pd.read_csv(local_csv)

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print("No local car_data.csv found — trying kagglehub...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "smritisingh1997/car-salescsv",
            "1.04. Real-life example.csv",
        )
        return df
    except Exception as e:
        raise FileNotFoundError(
            "Could not load the dataset.\n"
            "Fix: download the CSV from Kaggle and save it as 'car_data.csv' "
            "in this same folder, then run this script again.\n"
            f"(kagglehub attempt failed with: {e})"
        )


def main():
    df = load_data()

    # ---- cleaning (matches the notebook step-by-step) ----
    df = df.dropna(subset=["Price", "EngineV"])
    df = df[df["EngineV"] <= 10]
    df["Log_price"] = np.log(df["Price"])
    df = df.drop(columns=["Model", "Price"])

    # ---- encode categorical columns, and remember the mapping ----
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = {cls: int(code) for code, cls in enumerate(le.classes_)}

    X = df.drop(columns=["Log_price"])
    y = df["Log_price"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
    }

    results = {}
    predictions = {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }
        predictions[name] = preds
        filename = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"Saved {name:<20} -> {filename.name}  (R2={results[name]['r2']:.4f})")

    best_model_name = max(results, key=lambda n: results[n]["r2"])

    # sample of actual vs predicted price (in dollars) for the insights chart
    sample = pd.DataFrame({
        "actual": np.exp(y_test.values[:200]),
        "predicted": np.exp(predictions[best_model_name][:200]),
    })
    sample.to_csv(MODEL_DIR / "sample_predictions.csv", index=False)

    metadata = {
        "feature_columns": feature_columns,
        "encoders": encoders,
        "results": results,
        "best_model": best_model_name,
        "numeric_ranges": {
            "Mileage": [int(X["Mileage"].min()), int(X["Mileage"].max())],
            "EngineV": [float(X["EngineV"].min()), float(X["EngineV"].max())],
            "Year": [int(X["Year"].min()), int(X["Year"].max())],
        },
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel comparison:")
    for name, r in results.items():
        print(f"  {name:<20} R2={r['r2']:.4f}   MAE={r['mae']:.4f}")
    print(f"\nBest model: {best_model_name}")
    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
