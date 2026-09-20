"""
Car Price Predictor — Streamlit app
Linear / Ridge / Lasso Regression on used-car sales data.

Run:
    streamlit run app.py

Needs a "models/" folder produced by train_model.py (see that file's
docstring). If it's missing, this app shows instructions instead of
crashing.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_DIR = Path("models")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 16px;
            border: 1px solid #2a2f3d;
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #ff4b4b, #ff9f43);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .main-header p {
            color: #9ca3af;
            font-size: 1rem;
            margin: 0;
        }
        .price-card {
            background: linear-gradient(135deg, #1a2f2b 0%, #161b26 100%);
            border: 1px solid rgba(45, 212, 191, 0.35);
            border-radius: 16px;
            padding: 1.8rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        .price-card .label {
            color: #9ca3af;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .price-card .value {
            font-size: 2.6rem;
            font-weight: 700;
            color: #2dd4bf;
            margin: 0.3rem 0;
        }
        div[data-testid="stMetric"] {
            background-color: #161b26;
            border: 1px solid #2a2f3d;
            padding: 0.8rem 1rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_metadata():
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models(names: tuple):
    models = {}
    for name in names:
        filepath = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        if filepath.exists():
            models[name] = joblib.load(filepath)
    return models


def encode_input(raw: dict, encoders: dict, feature_columns: list) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        row[col] = encoders[col].get(raw[col], 0) if col in encoders else raw[col]
    return pd.DataFrame([row], columns=feature_columns)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🚗 Car Price Predictor</h1>
        <p>Linear, Ridge &amp; Lasso Regression trained on used-car sales data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = load_metadata()

if metadata is None:
    st.error(
        "**Model files not found.**\n\n"
        "This app needs a `models/` folder (with `metadata.json` and the "
        "trained `.pkl` files) sitting next to `app.py`.\n\n"
        "Run this first:\n"
        "```bash\n"
        "python train_model.py\n"
        "```\n"
        "then restart the app."
    )
    st.stop()

encoders = metadata["encoders"]
feature_columns = metadata["feature_columns"]
results = metadata["results"]
ranges = metadata["numeric_ranges"]
model_names = tuple(results.keys())
models = load_models(model_names)

tab_predict, tab_insights, tab_about = st.tabs(
    ["🎯 Predict Price", "📊 Model Insights", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Tab 1 — Predict
# --------------------------------------------------------------------------
with tab_predict:
    col_form, col_result = st.columns([1.3, 1], gap="large")

    with col_form:
        st.subheader("Car details")
        c1, c2 = st.columns(2)

        with c1:
            brand = st.selectbox("Brand", sorted(encoders["Brand"].keys()))
            body = st.selectbox("Body type", sorted(encoders["Body"].keys()))
            engine_type = st.selectbox(
                "Engine type", sorted(encoders["Engine Type"].keys())
            )
            registration = st.selectbox(
                "Registered", sorted(encoders["Registration"].keys())
            )

        with c2:
            mileage = st.number_input(
                "Mileage (thousand km)",
                min_value=0,
                max_value=int(ranges["Mileage"][1] * 1.5),
                value=int((ranges["Mileage"][0] + ranges["Mileage"][1]) / 2),
                step=5,
            )
            engine_v = st.number_input(
                "Engine volume (L)",
                min_value=float(ranges["EngineV"][0]),
                max_value=float(ranges["EngineV"][1]),
                value=round((ranges["EngineV"][0] + ranges["EngineV"][1]) / 2, 1),
                step=0.1,
            )
            year = st.number_input(
                "Year of manufacture",
                min_value=int(ranges["Year"][0]),
                max_value=2026,
                value=int((ranges["Year"][0] + ranges["Year"][1]) / 2),
                step=1,
            )

        model_choice = st.selectbox(
            "Model to use",
            model_names,
            index=model_names.index(metadata["best_model"])
            if metadata["best_model"] in model_names
            else 0,
        )

        predict_clicked = st.button(
            "Predict price", type="primary", use_container_width=True
        )

    with col_result:
        st.subheader("Estimated price")
        if predict_clicked:
            if model_choice not in models:
                st.warning(f"Model file for '{model_choice}' is missing.")
            else:
                raw = {
                    "Brand": brand,
                    "Body": body,
                    "Mileage": mileage,
                    "EngineV": engine_v,
                    "Engine Type": engine_type,
                    "Registration": registration,
                    "Year": year,
                }
                X_input = encode_input(raw, encoders, feature_columns)
                log_price = models[model_choice].predict(X_input)[0]
                price = float(np.exp(log_price))

                st.markdown(
                    f"""
                    <div class="price-card">
                        <div class="label">Predicted price</div>
                        <div class="value">${price:,.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Estimated with **{model_choice}** "
                    f"(test-set R² = {results[model_choice]['r2']:.3f})"
                )
        else:
            st.info("Fill in the car details and click **Predict price**.")

# --------------------------------------------------------------------------
# Tab 2 — Model Insights
# --------------------------------------------------------------------------
with tab_insights:
    st.subheader("Model comparison")

    comp_df = pd.DataFrame(
        [
            {"Model": k, "R2": v["r2"], "MAE (log price)": v["mae"]}
            for k, v in results.items()
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = px.bar(
            comp_df,
            x="Model",
            y="R2",
            color="Model",
            color_discrete_sequence=["#ff4b4b", "#ff9f43", "#2dd4bf"],
            text_auto=".3f",
        )
        fig_r2.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            yaxis_range=[0, 1],
            yaxis_title="R² score",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        st.dataframe(
            comp_df.style.format({"R2": "{:.4f}", "MAE (log price)": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    sample_path = MODEL_DIR / "sample_predictions.csv"
    if sample_path.exists():
        st.subheader(f"Actual vs. predicted price — {metadata['best_model']}")
        sample = pd.read_csv(sample_path)
        fig_scatter = px.scatter(
            sample,
            x="actual",
            y="predicted",
            labels={"actual": "Actual price ($)", "predicted": "Predicted price ($)"},
            color_discrete_sequence=["#2dd4bf"],
            opacity=0.7,
        )
        max_val = max(sample["actual"].max(), sample["predicted"].max())
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="#ff4b4b", dash="dash"),
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3 — About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
        Predicts the price of a used car from its **brand, body type, mileage,
        engine volume, engine type, registration status, and year**.

        **Pipeline**
        1. Drop rows missing `Price` or `EngineV`; remove `EngineV` outliers (> 10L)
        2. Log-transform `Price` to fix the right-skew → `Log_price`
        3. Label-encode the categorical columns (`Brand`, `Body`, `Engine Type`, `Registration`)
        4. Train **Linear**, **Ridge**, and **Lasso** regression models
        5. Predict `Log_price`, then reverse with `exp()` to get the price in dollars

        **Best model on this training run:** {metadata['best_model']}
        (R² = {results[metadata['best_model']]['r2']:.3f})

        Built with `scikit-learn`, `pandas`, and `Streamlit`.
        """
    )
