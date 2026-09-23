"""
Car Price Predictor — Streamlit app
Linear / Ridge / Lasso Regression on used-car sales data.

Run:
    streamlit run app.py

Needs a "models/" folder produced by train_model.py (see that file's
docstring), and the ".streamlit/config.toml" file next to this one for
the color theme. If models/ is missing, this app shows instructions
instead of crashing.
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
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling
#
# Colors, buttons, tabs, sliders, and input accents come from
# .streamlit/config.toml (Streamlit's own theme system) — that's more
# reliable across Streamlit versions than overriding internal CSS classes.
# This block only adds the custom font and a couple of small, STATIC or
# SINGLE-LINE HTML touches. Every dynamic (f-string) HTML snippet below is
# built as one single line on purpose: Streamlit's markdown renderer can
# misparse multi-line HTML that has Python-source indentation baked into
# it, which is what caused literal "<div...>" text to show up before.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Sora', sans-serif;
        }
        [data-testid="stAppViewContainer"] * {
            font-family: 'Sora', sans-serif;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: -220px;
            right: -180px;
            width: 520px;
            height: 520px;
            background: radial-gradient(circle, rgba(124, 108, 245, 0.18) 0%, rgba(124, 108, 245, 0) 70%);
            pointer-events: none;
            z-index: 0;
        }

        .dial-header {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 0.6rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1.6rem;
            position: relative;
            z-index: 1;
        }
        .dial-header h1 {
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.01em;
            background: linear-gradient(90deg, #e9e9f7 0%, #c4b5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .dial-header p {
            color: #8f93ab;
            font-size: 0.92rem;
            margin: 0.15rem 0 0 0;
        }

        .price-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: #a78bfa;
            text-shadow: 0 0 22px rgba(167, 139, 250, 0.35);
            line-height: 1.15;
        }
        .price-label {
            color: #8f93ab;
            font-size: 0.85rem;
        }
        .mono-caption {
            font-family: 'JetBrains Mono', monospace;
            color: #8f93ab;
            font-size: 0.82rem;
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
# Header (static content only — safe to keep as multi-line HTML)
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="dial-header">
        <svg width="42" height="42" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 22 L7 15 Q8 13 10 13 H24 Q27 13 28 16 L30 22" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            <path d="M4 22 H31" stroke="#a78bfa" stroke-width="2" stroke-linecap="round"/>
            <circle cx="10" cy="24" r="2.4" fill="#0a0d1f" stroke="#e9e9f7" stroke-width="1.6"/>
            <circle cx="25" cy="24" r="2.4" fill="#0a0d1f" stroke="#e9e9f7" stroke-width="1.6"/>
        </svg>
        <div>
            <h1>Car Price Predictor</h1>
            <p>Estimate resale value with Linear, Ridge &amp; Lasso regression</p>
        </div>
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
price_range = ranges.get("Price")
model_names = tuple(results.keys())
models = load_models(model_names)

tab_predict, tab_insights, tab_about = st.tabs(
    ["🎯 Predict Price", "📊 Model Insights", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Tab 1 — Predict
# --------------------------------------------------------------------------
with tab_predict:
    col_form, col_result = st.columns([1.2, 1], gap="large")

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
        st.subheader("Readout")

        with st.container(border=True):
            if not predict_clicked:
                st.markdown(
                    '<div class="price-label">Predicted price</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Fill in the car details and press Predict price.")
            elif model_choice not in models:
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
                    '<div class="price-label">Predicted price</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="price-value">${price:,.0f}</div>',
                    unsafe_allow_html=True,
                )

                if price_range:
                    lo, hi = price_range
                    pct = 0.5 if hi == lo else max(0.0, min(1.0, (price - lo) / (hi - lo)))
                    st.progress(pct)
                    lbl_lo, lbl_hi = st.columns(2)
                    lbl_lo.markdown(
                        f'<span class="mono-caption">${lo:,.0f}</span>',
                        unsafe_allow_html=True,
                    )
                    lbl_hi.markdown(
                        f'<span class="mono-caption" style="float:right;">${hi:,.0f}</span>',
                        unsafe_allow_html=True,
                    )

                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Model used", model_choice)
                m2.metric("Test-set R²", f"{results[model_choice]['r2']:.3f}")

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
    # amber marks the strongest model, steel the middle, brake-red the weakest
    bar_colors = ["#7c6cf5", "#8f93ab", "#f87171"][: len(comp_df)]

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = px.bar(
            comp_df,
            x="Model",
            y="R2",
            color="Model",
            color_discrete_sequence=bar_colors,
            text_auto=".3f",
        )
        fig_r2.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e9e9f7",
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
            color_discrete_sequence=["#a78bfa"],
            opacity=0.7,
        )
        max_val = max(sample["actual"].max(), sample["predicted"].max())
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="#f87171", dash="dash"),
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e9e9f7",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3 — About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
Predicts the price of a used car from its brand, body type, mileage,
engine volume, engine type, registration status, and year.

**Pipeline**
1. Drop rows missing `Price` or `EngineV`; remove `EngineV` outliers (over 10L)
2. Log-transform `Price` to fix the right-skew, giving `Log_price`
3. Label-encode the categorical columns (`Brand`, `Body`, `Engine Type`, `Registration`)
4. Train Linear, Ridge, and Lasso regression models
5. Predict `Log_price`, then reverse it with `exp()` to get the price in dollars

Best model on this training run: **{metadata['best_model']}**
(R² = {results[metadata['best_model']]['r2']:.3f})

Built with scikit-learn, pandas, and Streamlit.
"""
    )
