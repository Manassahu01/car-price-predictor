"""Drop-column ablation: how much worse is the model without each input?

For each raw input, the column is replaced by a constant (so it carries no
information) and the selected model is re-cross-validated on the training split.
This complements permutation importance: it measures what the model loses when it
must be *re-trained* without a feature, not just when the feature is scrambled.

Run from the project root:  python -m experiments.feature_ablation
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from car_price import config, data, models

# Replacing Year with a constant makes car_age all-NaN; the imputer's warning about it is expected.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def main():
    meta = json.loads(config.METADATA_PATH.read_text())
    sel = meta["selection"]
    clean, _ = data.prepare_training_data(data.load_raw())
    X, y = data.split_xy(clean)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    cv = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    def cv_rmsle(frame):
        est = models.build_named_estimator(sel["best_model"], sel["best_params"])
        return -cross_val_score(est, frame, y_train, cv=cv, scoring=config.CV_SCORING).mean()

    full = cv_rmsle(X_train)
    rows = [{"variant": "all features", "cv_rmsle": full, "change_vs_full": 0.0}]
    for col in config.RAW_FEATURES:
        ablated = X_train.copy()
        ablated[col] = "unknown" if col in config.CATEGORICAL_INPUTS else np.nan
        score = cv_rmsle(ablated)
        rows.append({"variant": f"without {col}", "cv_rmsle": score, "change_vs_full": score - full})

    out = pd.DataFrame(rows).round(4)
    out.to_csv(config.REPORTS_DIR / "feature_ablation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
