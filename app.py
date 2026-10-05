"""Car Price Predictor - Streamlit app.

All modelling logic lives in the `car_price` package (tested separately);
this file only handles the user interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from car_price import config, data
from car_price.predict import CarSpec, load_model, predict_price

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

OTHER_MODEL = "Other (not listed)"


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_listings() -> pd.DataFrame:
    """Cleaned training listings, used for the 'similar cars' table."""
    clean, _ = data.prepare_training_data(data.load_raw())
    return clean


def money(value: float) -> str:
    return f"${value:,.0f}"


def similar_cars(listings: pd.DataFrame, spec: CarSpec) -> pd.DataFrame:
    """Closest real listings: same brand and model (or body, if model unknown)."""
    mask = listings["Brand"] == spec.brand
    if spec.model == OTHER_MODEL:
        mask &= listings["Body"] == spec.body
    else:
        mask &= listings["Model"] == spec.model
    found = listings[mask].copy()
    if found.empty:
        return found
    found["_distance"] = (found["Year"] - spec.year).abs() * 1000 + (found["Mileage"] - spec.mileage).abs()
    found = found.sort_values("_distance").head(8)
    cols = ["Model", "Year", "Mileage", "EngineV", "Engine Type", "Registration", "Price"]
    out = found[cols].reset_index(drop=True)
    out["Price"] = out["Price"].map(money)
    return out


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
model, meta = get_model()
listings = get_listings()
catalogue = meta["catalogue"]
ranges = catalogue["numeric_ranges"]
test = meta["test_metrics"]

st.title("🚗 Car Price Predictor")
st.caption(
    f"Estimates the asking price of a used car from {meta['data']['quality_report']['rows_final']:,} "
    f"real listings. Selected model: **{meta['selection']['best_model']}**."
)

tab_predict, tab_perf, tab_about = st.tabs(["Price estimate", "Model performance", "About"])

# ---- Tab 1: prediction -----------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Car details")
        brand = st.selectbox("Brand", sorted(catalogue["brand_to_models"]))
        model_name = st.selectbox("Model", catalogue["brand_to_models"][brand] + [OTHER_MODEL])
        body = st.selectbox("Body type", catalogue["body"])
        engine_type = st.selectbox("Engine type", catalogue["engine_type"])
        registration = st.selectbox(
            "Registered?", catalogue["registration"], index=catalogue["registration"].index("yes")
        )
        year = st.slider("Year", ranges["Year"]["min"], ranges["Year"]["max"], 2008)
        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            max_value=float(ranges["Mileage"]["max"]),
            value=150.0,
            step=5.0,
            help="Dataset units (likely thousands of km). Median in the training data: 155.",
        )
        engine_v = st.number_input(
            "Engine volume (litres)",
            min_value=float(ranges["EngineV"]["min"]),
            max_value=float(ranges["EngineV"]["max"]),
            value=2.0,
            step=0.1,
        )

    spec = CarSpec(
        brand=brand, model=model_name, body=body, engine_type=engine_type,
        registration=registration, year=int(year), mileage=float(mileage), engine_v=float(engine_v),
    )

    with right:
        st.subheader("Estimate")
        try:
            result = predict_price(model, meta, spec)
        except ValueError as err:
            st.error(str(err))
        else:
            st.metric("Estimated price", money(result["estimate"]))
            st.write(
                f"**Likely range ({result['coverage']:.0%}):** "
                f"{money(result['lower'])} - {money(result['upper'])}"
            )
            st.caption(
                "The range comes from the model's past errors on unseen cars: about 4 in 5 "
                "real prices fell inside a range built this way."
            )
            for message in result["warnings"]:
                st.warning(message)

            st.markdown("**Closest listings in the dataset**")
            similar = similar_cars(listings, spec)
            if similar.empty:
                st.info("No comparable listings found for this combination.")
            else:
                st.dataframe(similar, hide_index=True)

# ---- Tab 2: performance --------------------------------------------------------
with tab_perf:
    st.subheader("How good is it?")
    st.caption(
        f"Measured on {meta['data']['test_rows']} cars the model never saw during training "
        "or model selection."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typical error (median)", f"{test['median_ape_pct']:.1f}%")
    c2.metric("Average error (MAE)", money(test["mae"]))
    c3.metric("R² (price)", f"{test['r2']:.2f}")
    c4.metric("Range hit rate", f"{meta['interval']['empirical_test_coverage']:.0%}")
    st.caption(
        f"For comparison, always guessing the median price gives a typical error of "
        f"{meta['baseline_test_metrics']['median_ape_pct']:.0f}%."
    )

    st.subheader("Models compared")
    comparison = pd.DataFrame(meta["comparison"])[
        ["model", "cv_rmsle", "test_mae", "test_median_ape_pct", "test_r2"]
    ]
    comparison.columns = ["Model", "CV error (RMSLE)", "Test MAE ($)", "Test median error (%)", "Test R²"]
    st.dataframe(comparison.round(3), hide_index=True)
    st.caption(
        "The model was chosen by cross-validated error on training data only. The top two "
        "tree models are statistically indistinguishable (CV difference is smaller than its noise)."
    )

    figures = [
        ("model_comparison.png", "Cross-validated error by model (lower is better)"),
        ("actual_vs_predicted.png", "Actual vs predicted price on unseen cars"),
        ("feature_importance.png", "What drives the price"),
        ("error_by_brand.png", "Where the model is more and less accurate"),
    ]
    for (file_name, caption), col in zip(figures, st.columns(2) * 2):
        path = config.FIGURES_DIR / file_name
        if path.exists():
            col.image(str(path), caption=caption)

# ---- Tab 3: about ---------------------------------------------------------------
with tab_about:
    quality = meta["data"]["quality_report"]
    st.subheader("About this project")
    st.markdown(
        f"""
**Data.** {quality['rows_raw']:,} used-car listings (7 brands, 300+ models).
{quality['duplicates_removed']} duplicate rows and {quality['missing_or_invalid_price_removed']} rows
without a price were removed. Implausible engine volumes (e.g. 99.99 L) were treated as
missing and imputed rather than dropped. Expensive cars were kept.

**Method.** One-hot encoded categories, engineered car age and mileage per year, a log-price
target, 5-fold cross-validated hyperparameter tuning of seven candidate models, and a held-out
20% test set used once.

**Limitations.**
- Data is from listings up to **{ranges['Year']['max']}**; prices do not reflect today's market.
- Accuracy is lower for luxury brands such as Mercedes-Benz (see the error-by-brand chart).
- The source dataset does not state its currency or mileage unit; USD and thousands of km are assumed.
- This is an estimate from a statistical model, not a professional valuation.
"""
    )
