# Model Card: Car Price Predictor

## Summary
Regression model that estimates the asking price of a used car from its brand, model, body type, engine type, engine volume, registration status, year and mileage. It returns a point estimate and an approximate 80% range.

- **Selected model:** Histogram Gradient Boosting (scikit-learn), trained on log price
- **Selection rule:** lowest 5-fold cross-validated RMSLE on the training split
- **Training data:** 4,123 listings after cleaning (3,298 train / 825 test for evaluation; the shipped model is refit on all 4,123 with the chosen hyperparameters)
- **Version:** 1.0.0

## Intended use
- Learning and portfolio demonstration of an end-to-end ML workflow.
- Getting a rough sense of where a used car from this dataset's market and period would be priced.

## Not intended for
- Real pricing, trading, insurance or lending decisions.
- Cars newer than 2016 or brands outside the seven in the data (Audi, BMW, Mercedes-Benz, Mitsubishi, Renault, Toyota, Volkswagen). The app warns when inputs fall outside the training range.

## Data
- Source: Kaggle dataset `smritisingh1997/car-salescsv` ("1.04. Real-life example"), 4,345 raw rows.
- Removed: 73 exact duplicate rows and 149 rows with a missing price.
- Kept and imputed: 148 rows with a missing engine volume, and 20 rows with an impossible engine volume (above 10 L, for example 99.99), which are treated as missing.
- Not removed: expensive cars (87 raw listings priced above 100,000). They are genuine listings; removing them would make the metrics look better than real-world performance.
- Currency and mileage unit are not documented by the source. USD and thousands of km are assumed.

## Features
Raw inputs: Brand, Model, Body, Engine Type, Registration, Mileage, EngineV, Year.
Engineered inside the pipeline: `car_age` (2016 minus year) and `mileage_per_year`. Categories are one-hot encoded with categories seen fewer than 5 times pooled together. Numeric columns are median-imputed (with missing indicators) and scaled.

## Evaluation
Hold-out test set of 825 listings, used once after model selection.

| Metric | Selected model | Naive baseline |
|---|---|---|
| Median absolute % error | 11.6% | 54.0% |
| Mean absolute % error | 18.2% | 93.1% |
| Mean absolute error | $3,033 | $11,771 |
| RMSE | $7,822 | $22,753 |
| R² (price) | 0.871 | -0.090 |
| R² (log price) | 0.927 | 0.000 |

Median error by brand: Audi 7.6%, Toyota 9.2%, Renault 10.7%, Volkswagen 11.5%, Mitsubishi 11.5%, BMW 11.9%, Mercedes-Benz 15.5%.

**Prediction range.** The 10th and 90th percentiles of out-of-fold log-residuals are applied around each estimate (about -23% / +28%). Empirical test coverage is 78.4% against an 80% target.

## Known limitations
- **Model choice is within noise.** Random Forest has a CV error of 0.248 versus 0.246 for the selected model (standard deviation about 0.015) and scored slightly better on the test set (MAE $2,742, R² 0.895). The selected model was kept to avoid selecting on test data.
- **Tuned-score optimism.** CV scores for tuned models are the best grid-point scores, not nested cross-validation, so they are slightly optimistic. The test-set figures are unaffected.
- **Luxury cars are less accurate** (higher spread of prices, fewer comparable listings).
- **The range is not uniform.** It is a single multiplicative band, so it is a rough guide; it will be too narrow for some cars and too wide for others.
- **Old, narrow data.** Listings end in 2016 and cover one market; the model does not reflect current prices.
- **Single random split.** Metrics come from one 80/20 split (seed 42). Results would vary somewhat with another split.
- **Correlation, not causation.** Feature importance describes what the model uses, not what causes prices.

## Reproducing
`python -m car_price.train` regenerates the model, `models/metadata.json` (including library versions and a SHA-256 of the training CSV), `reports/` and all figures. `python -m pytest` runs the 48 tests.
