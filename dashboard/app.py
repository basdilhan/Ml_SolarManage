"""
Streamlit dashboard for Sri Lankan solar wastage prediction.
Uses a saved best model pipeline and model metrics.
Auto-fetches weather forecast from Open-Meteo API.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import requests
import streamlit as st


DATA_PATH = Path("data/processed/features_regression_ready.csv")
MODEL_PATH = Path("best_model_pipeline.pkl")

# Coordinates for Sri Lankan districts
DISTRICT_COORDS = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612},
    "Galle": {"lat": 6.0535, "lon": 80.2210},
    "Matara": {"lat": 5.9549, "lon": 80.5550},
}


@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_weather_forecast(district: str) -> Dict[str, float] | None:
    """Fetch tomorrow's weather forecast from Open-Meteo API."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&daily=temperature_2m_mean,shortwave_radiation_sum,precipitation_sum,cloud_cover_mean"
            f"&timezone=Asia/Colombo"
            f"&forecast_days=2"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        # Index 1 = tomorrow
        idx = 1 if len(daily.get("temperature_2m_mean", [])) > 1 else 0

        temp = daily.get("temperature_2m_mean", [27.0])[idx]
        # Convert daily radiation sum (MJ/m²) to average irradiance (W/m²)
        # MJ/m²/day → W/m²: divide by seconds in a day, multiply by 1e6
        radiation_mj = daily.get("shortwave_radiation_sum", [0.0])[idx]
        irradiance = (radiation_mj * 1_000_000) / 86400 if radiation_mj else 0.0
        rainfall = daily.get("precipitation_sum", [0.0])[idx]
        cloud_cover = daily.get("cloud_cover_mean", [50.0])[idx]

        return {
            "temp": round(float(temp), 2) if temp is not None else 27.0,
            "irradiance": round(float(irradiance), 2) if irradiance else 0.0,
            "rainfall_mm": round(float(rainfall), 2) if rainfall is not None else 0.0,
            "cloud_cover": round(float(cloud_cover), 2) if cloud_cover is not None else 50.0,
            "date": daily.get("time", ["N/A"])[idx],
        }
    except Exception as e:
        st.warning(f"⚠️ Could not fetch weather data: {e}")
        return None


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_feature_row(
    feature_names: List[str],
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    row = {name: 0.0 for name in feature_names}
    for key, value in inputs.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row])


def main() -> None:
    st.set_page_config(
        page_title="Solar Energy Wastage Prediction System",
        page_icon="☀️",
        layout="wide",
    )

    st.title("☀️ Solar Energy Wastage Prediction System")
    st.markdown(
        "This system predicts tomorrow's solar energy wastage based on previous "
        "generation, usage, and weather forecast."
    )
    st.markdown("---")

    if not MODEL_PATH.exists():
        st.error("best_model_pipeline.pkl not found.")
        st.stop()

    if not DATA_PATH.exists():
        st.error("features_regression_ready.csv not found.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    df = load_data()

    target_col = "wasted_energy_kwh"
    feature_names = [c for c in df.columns if c != target_col]

    # --- District & Season Selection (needed before weather fetch) ---
    st.subheader("📍 Location")
    loc_c1, loc_c2 = st.columns(2)

    with loc_c1:
        district = st.selectbox("District", list(DISTRICT_COORDS.keys()))

    with loc_c2:
        season = st.selectbox(
            "Season",
            [
                "Southwest_Monsoon",
                "Northeast_Monsoon",
                "Inter_Monsoon_I",
                "Inter_Monsoon_II",
            ],
        )

    # Default household (used internally by the model)
    household = "residential1"

    st.markdown("---")

    # --- Auto-fetch weather ---
    st.subheader("🌤️ Weather Forecast (Auto-Fetched)")
    weather = fetch_weather_forecast(district)

    if weather:
        st.info(
            f"📅 Forecast date: **{weather['date']}** | "
            f"📍 District: **{district}**"
        )
        w_c1, w_c2 = st.columns(2)
        with w_c1:
            temp = st.number_input(
                "Temperature (°C)",
                min_value=-5.0,
                max_value=50.0,
                value=weather["temp"],
                help="Auto-filled from Open-Meteo forecast. Adjust if needed.",
            )
            irradiance = st.number_input(
                "Irradiance (W/m²)",
                min_value=0.0,
                max_value=1500.0,
                value=weather["irradiance"],
                help="Auto-filled from Open-Meteo forecast. Adjust if needed.",
            )
        with w_c2:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=weather["rainfall_mm"],
                help="Auto-filled from Open-Meteo forecast. Adjust if needed.",
            )
            cloud_cover = st.number_input(
                "Cloud Cover (%)",
                min_value=0.0,
                max_value=100.0,
                value=weather["cloud_cover"],
                help="Auto-filled from Open-Meteo forecast. Adjust if needed.",
            )
        st.success("✅ Weather data auto-fetched from Open-Meteo API")
    else:
        st.warning("⚠️ Could not fetch weather. Enter values manually.")
        w_c1, w_c2 = st.columns(2)
        with w_c1:
            temp = st.number_input(
                "Temperature (°C)", min_value=-5.0, max_value=50.0, value=27.0
            )
            irradiance = st.number_input(
                "Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=0.0
            )
        with w_c2:
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0
            )
            cloud_cover = st.number_input(
                "Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0
            )

    st.markdown("---")

    # --- Energy Inputs ---
    st.subheader("⚡ Today's Energy")
    st.caption("Enter today's solar generation and household load consumption.")
    e_c1, e_c2 = st.columns(2)

    with e_c1:
        daily_pv_kwh = st.number_input(
            "Today's Solar Generation (kWh)",
            min_value=0.0,
            max_value=600.0,
            value=57.0,
            step=1.0,
        )

    with e_c2:
        daily_load_kwh = st.number_input(
            "Today's Load Consumption (kWh)",
            min_value=0.0,
            max_value=270.0,
            value=19.0,
            step=1.0,
        )

    st.markdown("---")

    # --- Recent Energy History (Lag Features) ---
    st.subheader("📉 Recent Energy History")
    st.caption(
        "Enter your actual generation, load, and wastage from previous days. "
        "These are the most important inputs for an accurate prediction."
    )

    # --- 1 Day Ago ---
    st.markdown("**1 Day Ago (Yesterday)**")
    lag1_c1, lag1_c2, lag1_c3 = st.columns(3)
    with lag1_c1:
        pv_lag_1d = st.number_input(
            "PV Generation (kWh) – 1 day ago",
            min_value=0.0, max_value=600.0, value=57.0, step=1.0,
        )
    with lag1_c2:
        load_lag_1d = st.number_input(
            "Load (kWh) – 1 day ago",
            min_value=0.0, max_value=270.0, value=19.0, step=1.0,
        )
    with lag1_c3:
        wastage_lag_1d = st.number_input(
            "Wastage (kWh) – 1 day ago",
            min_value=0.0, max_value=510.0, value=0.0, step=1.0,
        )

    # --- 3 Days Ago ---
    st.markdown("**3 Days Ago**")
    lag3_c1, lag3_c2, lag3_c3 = st.columns(3)
    with lag3_c1:
        pv_lag_3d = st.number_input(
            "PV Generation (kWh) – 3 days ago",
            min_value=0.0, max_value=600.0, value=58.0, step=1.0,
        )
    with lag3_c2:
        load_lag_3d = st.number_input(
            "Load (kWh) – 3 days ago",
            min_value=0.0, max_value=270.0, value=19.0, step=1.0,
        )
    with lag3_c3:
        wastage_lag_3d = st.number_input(
            "Wastage (kWh) – 3 days ago",
            min_value=0.0, max_value=510.0, value=0.0, step=1.0,
        )

    # --- 7 Days Ago ---
    st.markdown("**7 Days Ago**")
    lag7_c1, lag7_c2, lag7_c3 = st.columns(3)
    with lag7_c1:
        pv_lag_7d = st.number_input(
            "PV Generation (kWh) – 7 days ago",
            min_value=0.0, max_value=600.0, value=60.0, step=1.0,
        )
    with lag7_c2:
        load_lag_7d = st.number_input(
            "Load (kWh) – 7 days ago",
            min_value=0.0, max_value=270.0, value=19.0, step=1.0,
        )
    with lag7_c3:
        wastage_lag_7d = st.number_input(
            "Wastage (kWh) – 7 days ago",
            min_value=0.0, max_value=510.0, value=0.0, step=1.0,
        )

    st.markdown("---")

    # --- Rolling Averages ---
    st.subheader("📊 Rolling Averages (Last 7 Days)")
    st.caption(
        "Enter approximate averages from the past week. "
        "14-day and 30-day averages will be estimated from these."
    )
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1:
        pv_rolling_mean_7d = st.number_input(
            "Avg PV Generation – 7 days (kWh)",
            min_value=0.0, max_value=600.0, value=59.0, step=1.0,
        )
    with r_c2:
        load_rolling_mean_7d = st.number_input(
            "Avg Load – 7 days (kWh)",
            min_value=0.0, max_value=270.0, value=19.0, step=1.0,
        )
    with r_c3:
        wastage_rolling_mean_7d = st.number_input(
            "Avg Wastage – 7 days (kWh)",
            min_value=0.0, max_value=510.0, value=0.0, step=1.0,
        )

    st.markdown("---")

    # --- Derive computed features ---
    net_export_lag_1d = pv_lag_1d - load_lag_1d
    net_export_lag_3d = pv_lag_3d - load_lag_3d
    net_export_lag_7d = pv_lag_7d - load_lag_7d

    pv_rolling_std_7d = abs(pv_lag_1d - pv_rolling_mean_7d) * 0.5
    pv_rolling_mean_14d = pv_rolling_mean_7d
    load_rolling_mean_14d = load_rolling_mean_7d
    pv_rolling_std_14d = pv_rolling_std_7d * 1.2
    wastage_rolling_mean_14d = wastage_rolling_mean_7d
    pv_rolling_mean_30d = pv_rolling_mean_7d
    load_rolling_mean_30d = load_rolling_mean_7d
    pv_rolling_std_30d = pv_rolling_std_7d * 1.5
    wastage_rolling_mean_30d = wastage_rolling_mean_7d

    # --- Auto-calculate calendar features for tomorrow ---
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    cal_year = tomorrow.year
    cal_month = tomorrow.month
    cal_day = tomorrow.day
    cal_quarter = (cal_month - 1) // 3 + 1
    cal_day_of_week = tomorrow.weekday()
    cal_day_of_year = tomorrow.timetuple().tm_yday
    cal_week_of_year = int(tomorrow.strftime("%W"))
    cal_is_weekend = 1 if cal_day_of_week >= 5 else 0

    # --- Collect all inputs ---
    inputs: Dict[str, Any] = {
        "daily_pv_kwh": daily_pv_kwh,
        "daily_load_kwh": daily_load_kwh,
        "irradiance": irradiance,
        "temp": temp,
        "district": district,
        "season": season,
        "household": household,
        "year": cal_year,
        "month": cal_month,
        "day": cal_day,
        "quarter": cal_quarter,
        "day_of_week": cal_day_of_week,
        "day_of_year": cal_day_of_year,
        "week_of_year": cal_week_of_year,
        "is_weekend": cal_is_weekend,
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
    }

    # Add rainfall/cloud_cover if they are model features
    if "rainfall_mm" in feature_names:
        inputs["rainfall_mm"] = rainfall
    if "cloud_cover" in feature_names:
        inputs["cloud_cover"] = cloud_cover

    if st.button("🔮 Predict Wastage", use_container_width=True):
        try:
            input_df = build_feature_row(feature_names, inputs)
            pred = pipeline.predict(input_df)
            pred_value = max(float(pred[0]), 0.0)

            st.markdown("---")
            st.subheader("📊 Prediction Result")
            st.metric(
                "Predicted wasted_energy_kwh",
                f"{pred_value:.4f} kWh",
            )

            if pred_value > 0:
                st.warning(
                    "⚠️ Excess solar energy expected tomorrow. "
                    "Consider battery storage or load shifting."
                )
            else:
                st.success("✅ No solar energy wastage expected.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 0.85em;'>"
        "Developed for Academic Solar Energy Optimization Project – 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
