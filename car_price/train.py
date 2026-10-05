"""Train, tune, evaluate and export the car-price model.

Run from the project root:

    python -m car_price.train

Method
------
1. Clean rows (duplicates, missing target) and hold out 20% as a TEST set that
   is never used for any decision.
2. For every candidate model, tune hyperparameters with 5-fold cross-validation
   on the training portion only.
3. Select the best model by cross-validated RMSLE (NOT by test score), so the
   test set stays an unbiased estimate of real-world performance.
4. Report test metrics, permutation importance and error by brand; calibrate an
   80% prediction interval from out-of-fold residuals and check its coverage.
5. Refit the chosen configuration on ALL cleaned data and save it with metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score, train_test_split

from car_price import config, data, evaluate, models
from car_price.predict import apply_interval

log = logging.getLogger("car_price.train")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _jsonable(obj):
    """Recursively convert numpy / pandas types so json.dump never fails."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tune(name, estimator, grid, X_train, y_train, cv):
    """Cross-validate (and tune, if there is a grid) one candidate on training data."""
    start = time.time()
    if grid:
        search = GridSearchCV(
            estimator, grid, scoring=config.CV_SCORING, cv=cv, n_jobs=1, refit=True
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        idx = search.best_index_
        cv_rmsle = float(-search.cv_results_["mean_test_score"][idx])
        cv_std = float(search.cv_results_["std_test_score"][idx])
        params = search.best_params_
    else:
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=config.CV_SCORING)
        best = estimator.fit(X_train, y_train)
        cv_rmsle, cv_std, params = float(-scores.mean()), float(scores.std()), {}
    log.info("%-24s CV RMSLE %.4f (+/- %.4f)  [%.0fs]  %s", name, cv_rmsle, cv_std, time.time() - start, params)
    return best, cv_rmsle, cv_std, params


def _category_catalogue(clean: pd.DataFrame) -> dict:
    """What the app needs to build its dropdowns and warn on out-of-range input."""
    brand_to_models = {
        brand: grp["Model"].value_counts().index.tolist()
        for brand, grp in clean.groupby("Brand")
    }
    valid_engine = clean["EngineV"].where(clean["EngineV"] <= config.MAX_VALID_ENGINE_VOLUME)
    return {
        "brand_to_models": brand_to_models,
        "body": sorted(clean["Body"].unique().tolist()),
        "engine_type": sorted(clean["Engine Type"].unique().tolist()),
        "registration": sorted(clean["Registration"].unique().tolist()),
        "numeric_ranges": {
            "Year": {"min": int(clean["Year"].min()), "max": int(clean["Year"].max())},
            "Mileage": {
                "min": int(clean["Mileage"].min()),
                "max": int(clean["Mileage"].max()),
                "p99": float(clean["Mileage"].quantile(0.99)),
            },
            "EngineV": {"min": float(valid_engine.min()), "max": float(valid_engine.max())},
        },
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    raw = data.load_raw()
    clean, dq_report = data.prepare_training_data(raw)
    log.info("Data quality: %s", dq_report)
    X, y = data.split_xy(clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    log.info("Split: %d train / %d test rows", len(X_train), len(X_test))
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # 2-3. tune every candidate with CV on the training set only -------------------
    fitted, rows = {}, []
    for name, (estimator, grid) in models.build_candidates().items():
        best, cv_rmsle, cv_std, params = _tune(name, estimator, grid, X_train, y_train, cv)
        test_pred = best.predict(X_test)
        test_metrics = evaluate.regression_metrics(y_test, test_pred)
        fitted[name] = (best, test_pred)
        rows.append(
            {"model": name, "cv_rmsle": cv_rmsle, "cv_rmsle_std": cv_std,
             **{f"test_{k}": v for k, v in test_metrics.items()}, "best_params": params}
        )

    table = pd.DataFrame(rows)
    contenders = table[table["model"] != models.BASELINE].sort_values("cv_rmsle")
    best_name = contenders.iloc[0]["model"]  # chosen by CV, never by test score
    table = pd.concat([contenders, table[table["model"] == models.BASELINE]], ignore_index=True)
    best_estimator, best_test_pred = fitted[best_name]
    best_params = table.loc[table["model"] == best_name, "best_params"].iloc[0]
    log.info("Selected model (lowest CV RMSLE): %s", best_name)

    table.drop(columns="best_params").round(4).to_csv(config.COMPARISON_PATH, index=False)

    # 4a. permutation importance + error by brand (test set) ---------------------
    imp = permutation_importance(
        best_estimator, X_test, y_test, scoring=config.CV_SCORING,
        n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}
    ).sort_values("importance_mean", ascending=False)

    test_frame = pd.DataFrame(
        {"actual": y_test.values, "predicted": best_test_pred, "Brand": X_test["Brand"].values}
    )
    by_brand = evaluate.error_by_group(test_frame, "Brand")

    # 4b. prediction interval from out-of-fold residuals on TRAIN data ---------
    oof = cross_val_predict(clone(best_estimator), X_train, y_train, cv=cv)
    log_resid = np.log1p(y_train.values) - np.log1p(np.clip(oof, 0, None))
    q_lo = float(np.quantile(log_resid, config.INTERVAL_LOWER_Q))
    q_hi = float(np.quantile(log_resid, config.INTERVAL_UPPER_Q))
    lower, upper = apply_interval(best_test_pred, q_lo, q_hi)
    coverage = evaluate.interval_coverage(y_test, lower, upper)
    nominal = config.INTERVAL_UPPER_Q - config.INTERVAL_LOWER_Q
    log.info("Prediction interval: nominal %.0f%%, empirical test coverage %.1f%%", nominal * 100, coverage * 100)

    # figures --------------------------------------------------------------------
    evaluate.plot_model_comparison(table, config.FIGURES_DIR / "model_comparison.png")
    evaluate.plot_actual_vs_predicted(
        y_test, best_test_pred, config.FIGURES_DIR / "actual_vs_predicted.png",
        f"{best_name}: actual vs predicted (test set)",
    )
    evaluate.plot_residuals(y_test, best_test_pred, config.FIGURES_DIR / "residuals.png")
    evaluate.plot_feature_importance(importance, config.FIGURES_DIR / "feature_importance.png")
    evaluate.plot_error_by_group(by_brand, config.FIGURES_DIR / "error_by_brand.png", "brand")

    # 5. refit chosen configuration on ALL cleaned data and save ----------------
    final_model = clone(best_estimator).fit(X, y)
    joblib.dump(final_model, config.MODEL_PATH, compress=3)

    best_row = table[table["model"] == best_name].iloc[0]
    base_row = table[table["model"] == models.BASELINE].iloc[0]
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "file": str(config.DATA_PATH.relative_to(config.PROJECT_ROOT)),
            "sha256": _sha256(config.DATA_PATH),
            "quality_report": dq_report,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "selection": {
            "best_model": best_name,
            "best_params": best_params,
            "criterion": f"lowest {config.CV_FOLDS}-fold CV RMSLE on the training split",
            "random_state": config.RANDOM_STATE,
        },
        "test_metrics": {k.removeprefix("test_"): float(best_row[k]) for k in table.columns if k.startswith("test_")},
        "baseline_test_metrics": {k.removeprefix("test_"): float(base_row[k]) for k in table.columns if k.startswith("test_")},
        "cv": {"rmsle": float(best_row["cv_rmsle"]), "rmsle_std": float(best_row["cv_rmsle_std"])},
        "interval": {
            "nominal_coverage": nominal,
            "log_lower_offset": q_lo,
            "log_upper_offset": q_hi,
            "empirical_test_coverage": coverage,
        },
        "comparison": table.drop(columns="best_params").round(5).to_dict(orient="records"),
        "feature_importance": importance.round(5).to_dict(orient="records"),
        "error_by_brand": by_brand.reset_index().to_dict(orient="records"),
        "catalogue": _category_catalogue(clean),
        "features": config.RAW_FEATURES,
        "model_file": config.MODEL_PATH.name,
    }
    config.METADATA_PATH.write_text(json.dumps(_jsonable(metadata), indent=2))
    log.info("Saved %s and %s", config.MODEL_PATH.name, config.METADATA_PATH.name)
    return metadata


def refit_from_metadata(metadata: dict | None = None):
    """Rebuild the selected model from the CSV using the stored hyperparameters.

    Used as a safety net by the app: if the pickled model cannot be loaded (for
    example after a scikit-learn upgrade) the app refits the chosen configuration
    in a few seconds instead of crashing. No search is re-run.
    """
    if metadata is None:
        metadata = json.loads(config.METADATA_PATH.read_text())
    clean, _ = data.prepare_training_data(data.load_raw())
    X, y = data.split_xy(clean)
    sel = metadata["selection"]
    return models.build_named_estimator(sel["best_model"], sel["best_params"]).fit(X, y)


if __name__ == "__main__":
    main()
