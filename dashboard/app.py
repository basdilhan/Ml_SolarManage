"""
Solar Energy Wastage Prediction Dashboard
==========================================
Supports two input modes:
  1. Manual Entry  — user enters last 3 days of PV & Load
  2. CSV Upload    — user uploads a CSV with Date, Solar_Generation, Load

Weather forecast is auto-fetched from the Open-Meteo API.
All lag features, rolling averages, and wastage are computed internally.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/features_regression_ready.csv")
MODEL_PATH = Path("best_model_pipeline.pkl")
HISTORY_CSV = Path("data/sample_csv/manual_entry_history.csv")

# ---------------------------------------------------------------------------
# District coordinates (for weather API)
# ---------------------------------------------------------------------------
DISTRICT_COORDS = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612},
    "Galle":   {"lat": 6.0535, "lon": 80.2210},
    "Matara":  {"lat": 5.9549, "lon": 80.5550},
}

# ---------------------------------------------------------------------------
# Weather helper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_weather_forecast(district: str) -> Dict[str, float] | None:
    """Fetch tomorrow's weather from Open-Meteo (free, no API key)."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            "&daily=temperature_2m_mean,shortwave_radiation_sum,"
            "precipitation_sum,cloud_cover_mean"
            "&timezone=Asia/Colombo&forecast_days=2"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        idx = 1 if len(daily.get("temperature_2m_mean", [])) > 1 else 0

        temp = daily.get("temperature_2m_mean", [27.0])[idx]
        rad  = daily.get("shortwave_radiation_sum", [0.0])[idx]
        irr  = (rad * 1_000_000) / 86400 if rad else 0.0

        return {
            "temp":        round(float(temp or 27.0), 2),
            "irradiance":  round(float(irr), 2),
            "rainfall_mm": round(float(daily.get("precipitation_sum", [0.0])[idx] or 0.0), 2),
            "cloud_cover": round(float(daily.get("cloud_cover_mean", [50.0])[idx] or 50.0), 2),
            "date":        daily.get("time", ["N/A"])[idx],
        }
    except Exception as e:
        st.warning(f"⚠️ Could not fetch weather data: {e}")
        return None

# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_manual_history() -> pd.DataFrame | None:
    """Load previously saved manual entries (if any)."""
    if HISTORY_CSV.exists():
        try:
            hist = pd.read_csv(HISTORY_CSV, parse_dates=["Date"])
            hist = hist.sort_values("Date").tail(3).reset_index(drop=True)
            if len(hist) == 3:
                return hist
        except Exception:
            pass
    return None


def save_manual_history(pv: List[float], load: List[float]) -> None:
    """Persist the 3 manual entries to CSV so they auto-fill next time."""
    today = datetime.date.today()
    dates = [
        today - datetime.timedelta(days=2),
        today - datetime.timedelta(days=1),
        today,
    ]
    df = pd.DataFrame({
        "Date": [d.isoformat() for d in dates],
        "Solar_Generation": pv,
        "Load": load,
    })
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_CSV, index=False)
    return HISTORY_CSV


def build_feature_row(
    feature_names: List[str],
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    """Create a single-row DataFrame with all model features."""
    row = {name: 0.0 for name in feature_names}
    for key, value in inputs.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row])


def compute_features_from_3days(
    pv: List[float], load: List[float],
) -> Dict[str, float]:
    """
    Given 3 days of PV and Load (index 0 = oldest, 2 = most recent),
    compute all lag, rolling, and derived features the model needs.
    """
    # Wastage for each day: max(0, PV - Load)
    wastage = [max(0.0, p - l) for p, l in zip(pv, load)]

    # Net export for each day
    net_export = [p - l for p, l in zip(pv, load)]

    # 3-day rolling averages
    pv_avg_3d      = float(np.mean(pv))
    load_avg_3d    = float(np.mean(load))
    wastage_avg_3d = float(np.mean(wastage))
    pv_std_3d      = float(np.std(pv))

    return {
        # Today's values (most recent day)
        "daily_pv_kwh":   pv[2],
        "daily_load_kwh": load[2],

        # Lag features — day index 2 = t-1, 1 = t-2 (relative to tomorrow)
        "pv_lag_1d":          pv[2],
        "load_lag_1d":        load[2],
        "net_export_lag_1d":  net_export[2],
        "wastage_lag_1d":     wastage[2],

        "pv_lag_3d":          pv[0],
        "load_lag_3d":        load[0],
        "net_export_lag_3d":  net_export[0],
        "wastage_lag_3d":     wastage[0],

        # 7-day lags approximated from 3-day data
        "pv_lag_7d":          pv[0],
        "load_lag_7d":        load[0],
        "net_export_lag_7d":  net_export[0],
        "wastage_lag_7d":     wastage[0],

        # Rolling means (7d approximated from 3-day window)
        "pv_rolling_mean_7d":       pv_avg_3d,
        "load_rolling_mean_7d":     load_avg_3d,
        "pv_rolling_std_7d":        pv_std_3d,
        "wastage_rolling_mean_7d":  wastage_avg_3d,

        # 14-day and 30-day approximated from 7-day
        "pv_rolling_mean_14d":      pv_avg_3d,
        "load_rolling_mean_14d":    load_avg_3d,
        "pv_rolling_std_14d":       pv_std_3d * 1.2,
        "wastage_rolling_mean_14d": wastage_avg_3d,

        "pv_rolling_mean_30d":      pv_avg_3d,
        "load_rolling_mean_30d":    load_avg_3d,
        "pv_rolling_std_30d":       pv_std_3d * 1.5,
        "wastage_rolling_mean_30d": wastage_avg_3d,
    }


def get_calendar_features() -> Dict[str, int]:
    """Auto-calculate calendar features for tomorrow."""
    tmrw = datetime.date.today() + datetime.timedelta(days=1)
    return {
        "year":         tmrw.year,
        "month":        tmrw.month,
        "day":          tmrw.day,
        "quarter":      (tmrw.month - 1) // 3 + 1,
        "day_of_week":  tmrw.weekday(),
        "day_of_year":  tmrw.timetuple().tm_yday,
        "week_of_year": int(tmrw.strftime("%W")),
        "is_weekend":   1 if tmrw.weekday() >= 5 else 0,
    }


# =========================================================================
# MAIN APP
# =========================================================================

def main() -> None:
    # --- Page config ---
    st.set_page_config(
        page_title="Solar Energy Wastage Prediction System",
        page_icon="☀️",
        layout="wide",
    )

    # --- Title ---
    st.title("☀️ Solar Energy Wastage Prediction System")
    st.markdown(
        "Predict tomorrow's solar energy wastage using historical solar "
        "generation, load data, and weather forecast."
    )
    st.markdown("---")

    # --- Load model & data ---
    if not MODEL_PATH.exists():
        st.error("❌ `best_model_pipeline.pkl` not found in project root.")
        st.stop()
    if not DATA_PATH.exists():
        st.error("❌ `features_regression_ready.csv` not found.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    df = load_data()
    target_col = "wasted_energy_kwh"
    feature_names = [c for c in df.columns if c != target_col]

    # =================================================================
    # 1. INPUT MODE SELECTION
    # =================================================================
    input_mode = st.radio(
        "**Select Input Method**",
        ["📝 Manual Entry", "📂 Upload CSV File"],
        horizontal=True,
    )

    st.markdown("---")

    # Placeholders for the 3-day PV and Load arrays
    pv_3days: List[float] | None = None
    load_3days: List[float] | None = None
    data_ready = False

    # =================================================================
    # 2A. MANUAL ENTRY MODE
    # =================================================================
    if input_mode == "📝 Manual Entry":
        st.subheader("📝 Enter Recent Energy Data")
        st.caption(
            "Enter solar generation and load for **Today**, **Yesterday**, "
            "and the **Day before Yesterday**. "
            "Wastage is calculated automatically as max(0, PV − Load)."
        )

        # Auto-load previous entries if available
        prev = load_manual_history()
        if prev is not None:
            st.success("✅ Previous entries loaded from saved history.")
            default_pv = prev["Solar_Generation"].tolist()
            default_load = prev["Load"].tolist()
        else:
            default_pv = [55.0, 57.0, 59.0]
            default_load = [18.0, 19.0, 20.0]

        day_labels = ["Day before Yesterday", "Yesterday", "Today"]
        pv_vals = []
        load_vals = []

        for i, label in enumerate(day_labels):
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            with c1:
                pv_val = st.number_input(
                    f"Solar Generation (kWh) – {label}",
                    min_value=0.0, max_value=600.0,
                    value=float(default_pv[i]),
                    step=0.5, key=f"pv_{i}",
                )
            with c2:
                load_val = st.number_input(
                    f"Load (kWh) – {label}",
                    min_value=0.0, max_value=270.0,
                    value=float(default_load[i]),
                    step=0.5, key=f"load_{i}",
                )
            pv_vals.append(pv_val)
            load_vals.append(load_val)

        pv_3days = pv_vals
        load_3days = load_vals
        data_ready = True

    # =================================================================
    # 2B. CSV UPLOAD MODE
    # =================================================================
    else:
        st.subheader("📂 Upload CSV File")
        st.caption(
            "Upload a CSV with at least 3 rows containing: "
            "**Date**, **Solar_Generation**, **Load**"
        )

        # Show required format
        with st.expander("📄 Required CSV Format"):
            sample = pd.DataFrame({
                "Date": ["2026-02-20", "2026-02-21", "2026-02-22"],
                "Solar_Generation": [18.5, 20.1, 17.8],
                "Load": [14.2, 15.0, 16.5],
            })
            st.dataframe(sample, use_container_width=True, hide_index=True)

        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="csv_upload",
        )

        if uploaded_file is not None:
            try:
                csv_df = pd.read_csv(uploaded_file)

                # --- Validate columns ---
                required_cols = {"Date", "Solar_Generation", "Load"}
                missing = required_cols - set(csv_df.columns)
                if missing:
                    st.error(
                        f"❌ Missing columns: **{', '.join(missing)}**. "
                        "Required: Date, Solar_Generation, Load"
                    )
                else:
                    # --- Validate values ---
                    if (csv_df["Solar_Generation"] < 0).any() or (csv_df["Load"] < 0).any():
                        st.error("❌ Negative values found. All values must be ≥ 0.")
                    else:
                        csv_df["Date"] = pd.to_datetime(csv_df["Date"])
                        csv_df = csv_df.sort_values("Date").reset_index(drop=True)

                        if len(csv_df) < 3:
                            st.error(
                                "❌ At least 3 days of data are required. "
                                f"Only {len(csv_df)} row(s) found."
                            )
                        else:
                            # Take last 3 days
                            last3 = csv_df.tail(3).reset_index(drop=True)
                            st.success(f"✅ Loaded {len(csv_df)} rows. Using last 3 days:")
                            st.dataframe(last3, use_container_width=True, hide_index=True)

                            pv_3days = last3["Solar_Generation"].tolist()
                            load_3days = last3["Load"].tolist()
                            data_ready = True

            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")

    st.markdown("---")

    # =================================================================
    # 3. LOCATION & SEASON
    # =================================================================
    st.subheader("📍 Location & Season")
    loc_c1, loc_c2 = st.columns(2)
    with loc_c1:
        district = st.selectbox("District", list(DISTRICT_COORDS.keys()))
    with loc_c2:
        season = st.selectbox("Season", [
            "Southwest_Monsoon", "Inter_Monsoon_I",
            "Inter_Monsoon_II",
        ])

    household = "residential1"  # internal default (negligible impact)

    st.markdown("---")

    # =================================================================
    # 4. WEATHER FORECAST (auto-fetched, editable)
    # =================================================================
    st.subheader("🌤️ Weather Forecast")
    weather = fetch_weather_forecast(district)

    if weather:
        st.info(
            f"📅 Forecast date: **{weather['date']}** | "
            f"📍 District: **{district}** — "
            "Auto-filled from Open-Meteo. Adjust if needed."
        )
        default_temp = weather["temp"]
        default_irr  = weather["irradiance"]
        default_rain = weather["rainfall_mm"]
        default_cc   = weather["cloud_cover"]
    else:
        st.warning("⚠️ Could not fetch weather. Please enter values manually.")
        default_temp, default_irr, default_rain, default_cc = 27.0, 0.0, 0.0, 50.0

    w_c1, w_c2, w_c3, w_c4 = st.columns(4)
    with w_c1:
        temp = st.number_input("Temperature (°C)", min_value=-5.0, max_value=50.0, value=default_temp)
    with w_c2:
        irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=default_irr)
    with w_c3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=default_rain)
    with w_c4:
        cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=default_cc)

    st.markdown("---")

    # =================================================================
    # 5. PREDICT BUTTON
    # =================================================================
    if st.button("🔮 Predict Tomorrow Wastage", use_container_width=True, type="primary"):
        if not data_ready or pv_3days is None or load_3days is None:
            st.error("❌ Please provide energy data (manual or CSV) before predicting.")
        else:
            try:
                # --- Compute all features from 3-day data ---
                features = compute_features_from_3days(pv_3days, load_3days)

                # --- Calendar features ---
                features.update(get_calendar_features())

                # --- Weather & categorical ---
                features["temp"]       = temp
                features["irradiance"] = irradiance
                features["district"]   = district
                features["season"]     = season
                features["household"]  = household

                # --- Build input DataFrame ---
                input_df = build_feature_row(feature_names, features)

                # --- Predict ---
                prediction = pipeline.predict(input_df)[0]
                predicted_wastage = round(max(float(prediction), 0.0), 2)

                # --- Today's actual wastage (computed from today's inputs) ---
                todays_wastage = round(max(0.0, pv_3days[2] - load_3days[2]), 2)

                # --- Save manual entries for next session ---
                if input_mode == "📝 Manual Entry":
                    saved_path = save_manual_history(pv_3days, load_3days)
                    st.caption(f"💾 Entries saved to `{saved_path}` — they will auto-fill next time.")

                # ============================================
                # 6. DISPLAY RESULTS
                # ============================================
                st.markdown("---")
                st.subheader("📊 Prediction Result")

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.metric(
                        label="Predicted Tomorrow Wastage (kWh)",
                        value=f"{predicted_wastage:.2f}",
                    )
                with res_c2:
                    if predicted_wastage > 0:
                        st.warning(
                            f"⚠ Excess solar energy expected tomorrow. "
                            f"Estimated wastage: **{predicted_wastage:.2f} kWh**"
                        )
                    else:
                        st.success("✅ No solar energy wastage expected.")

                # ============================================
                # 7. VISUALIZATION
                # ============================================
                st.markdown("---")
                st.subheader("📈 Wastage Comparison")

                _, chart_col, _ = st.columns([1, 2, 1])
                with chart_col:
                    colors = ["#2196F3", "#FF9800"]
                    fig, ax = plt.subplots(figsize=(2.8, 1.8))
                    labels = ["Today\n(Actual)", "Tomorrow\n(Predicted)"]
                    values = [todays_wastage, predicted_wastage]
                    ax.bar(labels, values, color=colors, width=0.4)
                    ax.set_ylabel("kWh", fontsize=7)
                    ax.set_title("Today vs Tomorrow", fontsize=8, pad=4)
                    ax.tick_params(labelsize=6)
                    for i, v in enumerate(values):
                        ax.text(
                            i, v + 0.02 * max(max(values), 1),
                            f"{v:.2f}", ha="center", fontweight="bold", fontsize=7,
                        )
                    fig.tight_layout(pad=0.5)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    # =================================================================
    # FOOTER
    # =================================================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 0.85em;'>"
        "Academic Solar Energy Optimization Project – 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
