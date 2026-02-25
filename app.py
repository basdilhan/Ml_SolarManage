"""
Solar Energy Wastage Prediction Dashboard
==========================================
A professional Streamlit dashboard that predicts next-day solar energy wastage
based on lag features, rolling statistics, and weather forecast inputs.

Run with:
    streamlit run app.py
"""

import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path("solar_wastage_model.pkl")

# The 39 feature columns the model was trained on (exact order from CSV)
FEATURE_COLUMNS = [
    "daily_pv_kwh", "daily_load_kwh", "household",
    "year", "month", "day", "quarter", "day_of_week", "day_of_year",
    "week_of_year", "is_weekend", "season",
    "pv_lag_1d", "load_lag_1d", "net_export_lag_1d", "wastage_lag_1d",
    "pv_lag_3d", "load_lag_3d", "net_export_lag_3d", "wastage_lag_3d",
    "pv_lag_7d", "load_lag_7d", "net_export_lag_7d", "wastage_lag_7d",
    "pv_rolling_mean_7d", "load_rolling_mean_7d", "pv_rolling_std_7d",
    "wastage_rolling_mean_7d",
    "pv_rolling_mean_14d", "load_rolling_mean_14d", "pv_rolling_std_14d",
    "wastage_rolling_mean_14d",
    "pv_rolling_mean_30d", "load_rolling_mean_30d", "pv_rolling_std_30d",
    "wastage_rolling_mean_30d",
    "district", "irradiance", "temp",
]

HOUSEHOLDS = ["residential1", "residential3", "residential4", "residential6"]
DISTRICTS = ["Colombo", "Galle", "Matara"]
SEASONS = ["Inter_Monsoon_I", "Inter_Monsoon_II", "Southwest_Monsoon"]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Wastage Predictor",
    page_icon="☀️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        st.error(
            "⚠️ Model file not found. Please ensure `solar_wastage_model.pkl` "
            "is placed in the project root directory."
        )
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


model = load_model()

# ---------------------------------------------------------------------------
# Helper – derive calendar features for tomorrow
# ---------------------------------------------------------------------------

def get_calendar_features(target_date: datetime.date) -> dict:
    """Return calendar features for a given date."""
    d = target_date
    return {
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "quarter": (d.month - 1) // 3 + 1,
        "day_of_week": d.weekday(),
        "day_of_year": d.timetuple().tm_yday,
        "week_of_year": int(d.strftime("%W")),
        "is_weekend": 1 if d.weekday() >= 5 else 0,
    }


def get_season(month: int) -> str:
    """Approximate Sri Lankan season from month."""
    if month in (3, 4):
        return "Inter_Monsoon_I"
    elif month in (5, 6, 7, 8, 9):
        return "Southwest_Monsoon"
    else:
        return "Inter_Monsoon_II"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("☀️ Solar Energy Wastage Prediction System")
st.markdown(
    "This system predicts tomorrow's solar energy wastage based on previous "
    "generation, usage, and weather forecast."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar – User Inputs
# ---------------------------------------------------------------------------
st.sidebar.header("📥 Input Features")

# ---- Household & Location ----
st.sidebar.subheader("🏠 Household & Location")
household = st.sidebar.selectbox("Household", HOUSEHOLDS, index=0)
district = st.sidebar.selectbox("District", DISTRICTS, index=0)

st.sidebar.markdown("---")

# ---- Today's Energy ----
st.sidebar.subheader("⚡ Today's Energy")
daily_pv = st.sidebar.number_input(
    "Today's Solar Generation (kWh)",
    min_value=0.0, max_value=600.0, value=57.0, step=1.0,
)
daily_load = st.sidebar.number_input(
    "Today's Load Consumption (kWh)",
    min_value=0.0, max_value=270.0, value=19.0, step=1.0,
)

st.sidebar.markdown("---")

# ---- Lag Features ----
st.sidebar.subheader("📉 Lag Features")

pv_lag_1d = st.sidebar.number_input(
    "PV(t-1) – Yesterday Generation (kWh)",
    min_value=0.0, max_value=600.0, value=57.0, step=1.0,
)
pv_lag_3d = st.sidebar.number_input(
    "PV(t-3) – 3 Days Ago Generation (kWh)",
    min_value=0.0, max_value=600.0, value=58.0, step=1.0,
)
pv_lag_7d = st.sidebar.number_input(
    "PV(t-7) – 7 Days Ago Generation (kWh)",
    min_value=0.0, max_value=600.0, value=60.0, step=1.0,
)
load_lag_1d = st.sidebar.number_input(
    "Load(t-1) – Yesterday Load (kWh)",
    min_value=0.0, max_value=270.0, value=19.0, step=1.0,
)
load_lag_3d = st.sidebar.number_input(
    "Load(t-3) – 3 Days Ago Load (kWh)",
    min_value=0.0, max_value=270.0, value=19.0, step=1.0,
)
load_lag_7d = st.sidebar.number_input(
    "Load(t-7) – 7 Days Ago Load (kWh)",
    min_value=0.0, max_value=270.0, value=19.0, step=1.0,
)
wastage_lag_1d = st.sidebar.number_input(
    "Wastage(t-1) – Yesterday Wastage (kWh)",
    min_value=0.0, max_value=510.0, value=0.0, step=1.0,
)
wastage_lag_3d = st.sidebar.number_input(
    "Wastage(t-3) – 3 Days Ago Wastage (kWh)",
    min_value=0.0, max_value=510.0, value=0.0, step=1.0,
)
wastage_lag_7d = st.sidebar.number_input(
    "Wastage(t-7) – 7 Days Ago Wastage (kWh)",
    min_value=0.0, max_value=510.0, value=0.0, step=1.0,
)

st.sidebar.markdown("---")

# ---- Rolling Features ----
st.sidebar.subheader("📊 Rolling Averages")

pv_rolling_mean_7d = st.sidebar.number_input(
    "PV 7-day Average (kWh)",
    min_value=0.0, max_value=600.0, value=59.0, step=1.0,
)
load_rolling_mean_7d = st.sidebar.number_input(
    "Load 7-day Average (kWh)",
    min_value=0.0, max_value=270.0, value=19.0, step=1.0,
)
wastage_rolling_mean_7d = st.sidebar.number_input(
    "Wastage 7-day Average (kWh)",
    min_value=0.0, max_value=510.0, value=0.0, step=1.0,
)

st.sidebar.markdown("---")

# ---- Weather Forecast ----
st.sidebar.subheader("🌤️ Weather Forecast")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=20.0, max_value=40.0, value=27.0, step=0.5,
)
irradiance = st.sidebar.number_input(
    "Irradiance (kWh/m²/day)",
    min_value=0.0, max_value=10.0, value=5.7, step=0.1,
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.subheader("🔮 Prediction")

predict_button = st.button("Predict Tomorrow Wastage", type="primary")

if predict_button:
    try:
        # --- Calendar features (for tomorrow) ---
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        cal = get_calendar_features(tomorrow)
        season = get_season(cal["month"])

        # --- Derive net_export lags (PV − Load) ---
        net_export_lag_1d = pv_lag_1d - load_lag_1d
        net_export_lag_3d = pv_lag_3d - load_lag_3d
        net_export_lag_7d = pv_lag_7d - load_lag_7d

        # --- Approximate rolling stats from user inputs ---
        pv_rolling_std_7d = abs(pv_lag_1d - pv_rolling_mean_7d) * 0.5
        pv_rolling_mean_14d = pv_rolling_mean_7d
        load_rolling_mean_14d = load_rolling_mean_7d
        pv_rolling_std_14d = pv_rolling_std_7d * 1.2
        wastage_rolling_mean_14d = wastage_rolling_mean_7d
        pv_rolling_mean_30d = pv_rolling_mean_7d
        load_rolling_mean_30d = load_rolling_mean_7d
        pv_rolling_std_30d = pv_rolling_std_7d * 1.5
        wastage_rolling_mean_30d = wastage_rolling_mean_7d

        # --- Build feature row as DataFrame (preserving column names) ---
        row = {
            "daily_pv_kwh": daily_pv,
            "daily_load_kwh": daily_load,
            "household": household,
            "year": cal["year"],
            "month": cal["month"],
            "day": cal["day"],
            "quarter": cal["quarter"],
            "day_of_week": cal["day_of_week"],
            "day_of_year": cal["day_of_year"],
            "week_of_year": cal["week_of_year"],
            "is_weekend": cal["is_weekend"],
            "season": season,
            "pv_lag_1d": pv_lag_1d,
            "load_lag_1d": load_lag_1d,
            "net_export_lag_1d": net_export_lag_1d,
            "wastage_lag_1d": wastage_lag_1d,
            "pv_lag_3d": pv_lag_3d,
            "load_lag_3d": load_lag_3d,
            "net_export_lag_3d": net_export_lag_3d,
            "wastage_lag_3d": wastage_lag_3d,
            "pv_lag_7d": pv_lag_7d,
            "load_lag_7d": load_lag_7d,
            "net_export_lag_7d": net_export_lag_7d,
            "wastage_lag_7d": wastage_lag_7d,
            "pv_rolling_mean_7d": pv_rolling_mean_7d,
            "load_rolling_mean_7d": load_rolling_mean_7d,
            "pv_rolling_std_7d": pv_rolling_std_7d,
            "wastage_rolling_mean_7d": wastage_rolling_mean_7d,
            "pv_rolling_mean_14d": pv_rolling_mean_14d,
            "load_rolling_mean_14d": load_rolling_mean_14d,
            "pv_rolling_std_14d": pv_rolling_std_14d,
            "wastage_rolling_mean_14d": wastage_rolling_mean_14d,
            "pv_rolling_mean_30d": pv_rolling_mean_30d,
            "load_rolling_mean_30d": load_rolling_mean_30d,
            "pv_rolling_std_30d": pv_rolling_std_30d,
            "wastage_rolling_mean_30d": wastage_rolling_mean_30d,
            "district": district,
            "irradiance": irradiance,
            "temp": temperature,
        }

        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        # --- Predict ---
        prediction = model.predict(input_df)[0]
        predicted_wastage = round(max(prediction, 0), 2)

        st.markdown("---")
        st.subheader("📊 Results")

        # Display predicted value
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Tomorrow Wastage (kWh)",
                value=f"{predicted_wastage:.2f}",
            )

        # Warning / success message
        with col2:
            if predicted_wastage > 0:
                st.warning(
                    "⚠ Excess solar energy expected tomorrow. "
                    f"Estimated wastage: **{predicted_wastage:.2f} kWh**"
                )
            else:
                st.success("✅ No solar energy wastage expected.")

        # ---- Visualization ----
        st.markdown("---")
        st.subheader("📈 Wastage Comparison")

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Yesterday's Wastage", "Predicted Tomorrow Wastage"]
        values = [wastage_lag_1d, predicted_wastage]
        ax.bar(labels, values)
        ax.set_ylabel("Energy Wastage (kWh)")
        ax.set_title("Yesterday vs Predicted Tomorrow Wastage")
        for i, v in enumerate(values):
            ax.text(
                i, v + 0.02 * max(max(values), 1),
                f"{v:.2f}", ha="center", fontweight="bold",
            )
        fig.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.85em;'>"
    "Developed for Academic Solar Energy Optimization Project – 2026"
    "</div>",
    unsafe_allow_html=True,
)
