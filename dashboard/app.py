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
import io
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/features_regression_ready.csv")
MODEL_PATH = Path("best_model_pipeline.pkl")
HISTORY_CSV = Path("data/sample_csv/manual_entry_history.csv")
METRICS_PATH = Path("model_metrics.json")

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
def fetch_weather_forecast(district: str, target_date: datetime.date) -> Dict[str, float] | None:
    """Fetch weather for selected date from Open-Meteo (free, no API key)."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            "&daily=temperature_2m_mean,shortwave_radiation_sum,"
            "precipitation_sum,cloud_cover_mean"
            "&timezone=Asia/Colombo&forecast_days=16"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        times = daily.get("time", [])
        target_date_str = target_date.isoformat()
        if target_date_str in times:
            idx = times.index(target_date_str)
        elif len(daily.get("temperature_2m_mean", [])) > 1:
            idx = 1
        else:
            idx = 0

        temp = daily.get("temperature_2m_mean", [27.0])[idx]
        rad  = daily.get("shortwave_radiation_sum", [0.0])[idx]
        irr  = (rad * 1_000_000) / 86400 if rad else 0.0

        return {
            "temp":        round(float(temp or 27.0), 2),
            "irradiance":  round(float(irr), 2),
            "rainfall_mm": round(float(daily.get("precipitation_sum", [0.0])[idx] or 0.0), 2),
            "cloud_cover": round(float(daily.get("cloud_cover_mean", [50.0])[idx] or 50.0), 2),
            "date":        times[idx] if idx < len(times) else "N/A",
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


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Load model comparison metrics from JSON file."""
    if not METRICS_PATH.exists():
        return pd.DataFrame(columns=["Model", "R2", "MAE", "RMSE"])

    try:
        raw = pd.read_json(METRICS_PATH).T.reset_index().rename(columns={"index": "Model"})
        expected = ["Model", "R2", "MAE", "RMSE"]
        for col in expected:
            if col not in raw.columns:
                raw[col] = np.nan
        return raw[expected].sort_values("RMSE", ascending=True).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Model", "R2", "MAE", "RMSE"])


def apply_professional_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
                color: #e2e8f0;
            }
            .block-container {
                padding-top: 1.2rem;
            }
            .custom-card {
                background: #111b2e;
                border: 1px solid #25324a;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
            }
            .section-title {
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.25rem;
            }
            .section-sub {
                color: #cbd5e1;
                font-size: 0.92rem;
            }
            [data-testid="stSidebar"] {
                background: #0a1324;
                border-right: 1px solid #1e293b;
            }
            [data-testid="stMetricLabel"] {
                color: #cbd5e1;
            }
            [data-testid="stMetricValue"] {
                color: #f8fafc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_manual_history_for_prediction_date(prediction_date: datetime.date) -> pd.DataFrame | None:
    """Load saved entries that exactly match D-3, D-2, D-1 for selected prediction date."""
    if not HISTORY_CSV.exists():
        return None

    try:
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["Date"])
        if hist.empty or "Date" not in hist.columns:
            return None

        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
        required_dates = [
            prediction_date - datetime.timedelta(days=3),
            prediction_date - datetime.timedelta(days=2),
            prediction_date - datetime.timedelta(days=1),
        ]

        rows = []
        for req_date in required_dates:
            matches = hist.loc[hist["Date"] == req_date]
            if matches.empty:
                rows.append({
                    "Date": req_date,
                    "Solar_Generation": np.nan,
                    "Load": np.nan,
                })
            else:
                rows.append(matches.iloc[-1].to_dict())

        return pd.DataFrame(rows)
    except Exception:
        return None


def get_training_feature_ranges(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, tuple[float, float]]:
    """Get min/max ranges for numeric training features."""
    ranges: Dict[str, tuple[float, float]] = {}
    for col in feature_names:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if not values.empty:
                ranges[col] = (float(values.min()), float(values.max()))
    return ranges


def summarize_confidence(input_df: pd.DataFrame, feature_ranges: Dict[str, tuple[float, float]]) -> tuple[str, List[str]]:
    out_of_range: List[str] = []
    for col in input_df.columns:
        if col in feature_ranges:
            lo, hi = feature_ranges[col]
            value = float(input_df.iloc[0][col])
            if value < lo or value > hi:
                out_of_range.append(f"{col}: {value:.2f} not in [{lo:.2f}, {hi:.2f}]")

    if not out_of_range:
        return "high", out_of_range
    if len(out_of_range) <= 3:
        return "medium", out_of_range
    return "low", out_of_range


def run_temporal_error_analysis(pipeline: Any, df: pd.DataFrame, target_col: str, feature_names: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on temporal 80/20 split to visualize error behaviour."""
    X = df[feature_names].copy()
    y = df[target_col].copy()

    if {"year", "month", "day"}.issubset(X.columns):
        order_idx = X.sort_values(["year", "month", "day"]).index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)

    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    y_pred = pipeline.predict(X_test)
    return y_test.to_numpy(), np.array(y_pred)


def build_prediction_report_csv(report: Dict[str, Any]) -> bytes:
    return pd.DataFrame([report]).to_csv(index=False).encode("utf-8")


def build_prediction_report_pdf(report: Dict[str, Any]) -> bytes:
    pdf_bytes = io.BytesIO()
    with PdfPages(pdf_bytes) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        lines = [
            "Solar Wastage Prediction Report",
            "",
            f"Generated At: {report.get('generated_at', '')}",
            f"Input Mode: {report.get('input_mode', '')}",
            f"District: {report.get('district', '')}",
            f"Season: {report.get('season', '')}",
            f"Predicted Tomorrow Wastage (kWh): {report.get('predicted_tomorrow_wastage_kwh', '')}",
            f"Today's Actual Wastage (kWh): {report.get('todays_actual_wastage_kwh', '')}",
            f"Temperature (°C): {report.get('temp_c', '')}",
            f"Irradiance (W/m²): {report.get('irradiance_wm2', '')}",
            f"Rainfall (mm): {report.get('rainfall_mm', '')}",
            f"Cloud Cover (%): {report.get('cloud_cover_pct', '')}",
            f"Confidence: {report.get('confidence_level', '')}",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf_bytes.seek(0)
    return pdf_bytes.getvalue()


def save_manual_history_latest_day(pv_latest: float, load_latest: float, entry_date: datetime.date) -> Path:
    """Upsert only the latest input day (D-1) in manual history CSV."""
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([
        {
            "Date": entry_date.isoformat(),
            "Solar_Generation": pv_latest,
            "Load": load_latest,
        }
    ])

    if HISTORY_CSV.exists():
        existing = pd.read_csv(HISTORY_CSV)
        if "Date" not in existing.columns:
            existing = pd.DataFrame(columns=["Date", "Solar_Generation", "Load"])
    else:
        existing = pd.DataFrame(columns=["Date", "Solar_Generation", "Load"])

    merged = pd.concat([existing, new_row], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last")
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).sort_values("Date")
    merged["Date"] = merged["Date"].dt.date.astype(str)
    merged.to_csv(HISTORY_CSV, index=False)
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


def get_calendar_features(target_date: datetime.date) -> Dict[str, int]:
    """Calculate calendar features for selected prediction date."""
    tmrw = target_date
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

    apply_professional_theme()

    # --- Load model & data ---
    if not MODEL_PATH.exists():
        st.error("❌ `best_model_pipeline.pkl` not found in project root.")
        st.stop()
    if not DATA_PATH.exists():
        st.error("❌ `features_regression_ready.csv` not found.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    df = load_data()
    metrics_df = load_model_metrics()
    target_col = "wasted_energy_kwh"
    feature_names = [c for c in df.columns if c != target_col]
    feature_ranges = get_training_feature_ranges(df, feature_names)

    # --- Sidebar navigation ---
    st.sidebar.title("⚙️ Dashboard")
    section = st.sidebar.radio(
        "Go to",
        [
            "🏠 Predict",
            "🏁 Model Comparison",
            "🎛️ What-if Simulator",
            "🧪 Error Analysis",
            "⬇️ Reports",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Solar Wastage Forecast • Professional Mode")

    # Keep latest prediction for simulator + reports pages
    last_prediction = st.session_state.get("last_prediction")

    if section == "🏁 Model Comparison":
        st.title("🏁 Model Comparison")
        st.markdown(
            "<div class='custom-card'><div class='section-sub'>Compare all trained models using R², MAE, and RMSE.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if not metrics_df.empty:
            best_model = metrics_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Selected Model", str(best_model["Model"]))
            with c2:
                st.metric("Best RMSE", f"{float(best_model['RMSE']):.4f}")
            with c3:
                st.metric("Best R²", f"{float(best_model['R2']):.4f}")

            st.dataframe(metrics_df, width="stretch", hide_index=True)

            fig_cmp, ax_cmp = plt.subplots(figsize=(8, 3.5))
            ax_cmp.bar(metrics_df["Model"], metrics_df["RMSE"], color="#60a5fa")
            ax_cmp.set_ylabel("RMSE")
            ax_cmp.set_title("Model Ranking by RMSE")
            ax_cmp.tick_params(axis="x", labelrotation=30)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
        else:
            st.info("Model metrics file not found yet. Run model training to view comparison.")
        return

    if section == "🧪 Error Analysis":
        st.title("🧪 Error Analysis")
        st.markdown(
            "<div class='custom-card'><div class='section-sub'>Temporal holdout diagnostics: actual vs predicted and residual distribution.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        y_true, y_hat = run_temporal_error_analysis(pipeline, df, target_col, feature_names)
        residuals = y_true - y_hat

        rmse_val = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        mae_val = float(np.mean(np.abs(y_true - y_hat)))
        r2_val = float(1 - np.sum((y_true - y_hat) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if np.sum((y_true - np.mean(y_true)) ** 2) > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Temporal RMSE", f"{rmse_val:.4f}")
        with m2:
            st.metric("Temporal MAE", f"{mae_val:.4f}")
        with m3:
            st.metric("Temporal R²", f"{r2_val:.4f}")

        e1, e2 = st.columns(2)
        with e1:
            fig_scatter, ax_scatter = plt.subplots(figsize=(4.6, 3.2))
            ax_scatter.scatter(y_true, y_hat, alpha=0.45, s=12)
            mn = float(min(np.min(y_true), np.min(y_hat)))
            mx = float(max(np.max(y_true), np.max(y_hat)))
            ax_scatter.plot([mn, mx], [mn, mx], "r--", linewidth=1.2)
            ax_scatter.set_title("Actual vs Predicted")
            ax_scatter.set_xlabel("Actual")
            ax_scatter.set_ylabel("Predicted")
            fig_scatter.tight_layout()
            st.pyplot(fig_scatter)

        with e2:
            fig_res, ax_res = plt.subplots(figsize=(4.6, 3.2))
            ax_res.hist(residuals, bins=28, color="#f59e0b", edgecolor="black", alpha=0.75)
            ax_res.axvline(0, color="red", linestyle="--", linewidth=1)
            ax_res.set_title("Residual Distribution")
            ax_res.set_xlabel("Residual (Actual - Predicted)")
            ax_res.set_ylabel("Count")
            fig_res.tight_layout()
            st.pyplot(fig_res)
        return

    if section == "🎛️ What-if Simulator":
        st.title("🎛️ What-if Simulator")
        st.markdown(
            "<div class='custom-card'><div class='section-sub'>Adjust PV, load, temperature, and irradiance to instantly test tomorrow wastage impact.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        base_pv = [55.0, 57.0, 59.0]
        base_load = [18.0, 19.0, 20.0]
        base_temp = 27.0
        base_irr = 500.0
        base_district = "Colombo"
        base_season = "Southwest_Monsoon"
        base_prediction_date = datetime.date.today() + datetime.timedelta(days=1)

        if last_prediction is not None:
            base_pv = last_prediction.get("pv_3days", base_pv)
            base_load = last_prediction.get("load_3days", base_load)
            base_temp = last_prediction.get("temp", base_temp)
            base_irr = last_prediction.get("irradiance", base_irr)
            base_district = last_prediction.get("district", base_district)
            base_season = last_prediction.get("season", base_season)
            lp_date = last_prediction.get("prediction_date")
            if isinstance(lp_date, datetime.date):
                base_prediction_date = lp_date
            elif isinstance(lp_date, str):
                try:
                    base_prediction_date = datetime.date.fromisoformat(lp_date)
                except ValueError:
                    pass

        wf_c1, wf_c2, wf_c3, wf_c4 = st.columns(4)
        with wf_c1:
            pv_delta = st.slider("PV Δ (kWh)", -20.0, 20.0, 0.0, 0.5)
        with wf_c2:
            load_delta = st.slider("Load Δ (kWh)", -20.0, 20.0, 0.0, 0.5)
        with wf_c3:
            temp_delta = st.slider("Temp Δ (°C)", -10.0, 10.0, 0.0, 0.5)
        with wf_c4:
            irr_delta = st.slider("Irradiance Δ (W/m²)", -300.0, 300.0, 0.0, 10.0)

        wf_prediction_date = st.date_input(
            "Simulation Prediction Date",
            value=base_prediction_date,
            min_value=datetime.date.today(),
            max_value=datetime.date.today() + datetime.timedelta(days=15),
            key="whatif_prediction_date",
        )

        pv_sim = base_pv.copy()
        load_sim = base_load.copy()
        pv_sim[2] = max(0.0, pv_sim[2] + pv_delta)
        load_sim[2] = max(0.0, load_sim[2] + load_delta)

        wf_features = compute_features_from_3days(pv_sim, load_sim)
        wf_features.update(get_calendar_features(wf_prediction_date))
        wf_features["temp"] = base_temp + temp_delta
        wf_features["irradiance"] = max(0.0, base_irr + irr_delta)
        wf_features["district"] = base_district
        wf_features["season"] = base_season
        wf_features["household"] = "residential1"

        wf_input_df = build_feature_row(feature_names, wf_features)
        wf_pred = max(0.0, float(pipeline.predict(wf_input_df)[0]))
        st.metric("What-if Predicted Wastage (kWh)", f"{wf_pred:.2f}")
        return

    if section == "⬇️ Reports":
        st.title("⬇️ Prediction Reports")
        st.markdown(
            "<div class='custom-card'><div class='section-sub'>Download the latest prediction as CSV and PDF snapshot.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if last_prediction is None:
            st.info("No prediction available yet. Go to 'Predict' section and run a prediction first.")
            return

        report_payload = last_prediction.get("report_payload", {})
        if not report_payload:
            st.info("No report payload found. Run a fresh prediction in 'Predict'.")
            return

        st.dataframe(pd.DataFrame([report_payload]), width="stretch", hide_index=True)

        csv_bytes = build_prediction_report_csv(report_payload)
        pdf_bytes = build_prediction_report_pdf(report_payload)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download CSV Report",
                data=csv_bytes,
                file_name="prediction_report.csv",
                mime="text/csv",
                width="stretch",
            )
        with d2:
            st.download_button(
                "Download PDF Snapshot",
                data=pdf_bytes,
                file_name="prediction_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
        return

    # Default section: Predict
    st.title("🏠 Predict Tomorrow Wastage")
    st.markdown(
        "<div class='custom-card'>"
        "<div class='section-title'>Forecast Workspace</div>"
        "<div class='section-sub'>Enter inputs and generate tomorrow's wastage prediction with clear confidence feedback.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # =================================================================
    # 1. DATE + INPUT MODE SELECTION
    # =================================================================
    default_pred_date = datetime.date.today() + datetime.timedelta(days=1)
    prediction_date = st.date_input(
        "📅 Prediction Date",
        value=default_pred_date,
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=15),
        help="Select the date you want to forecast (Open-Meteo supports up to ~16 days).",
    )

    input_mode = st.radio(
        "**Select Input Method**",
        ["📝 Manual Entry", "📂 Upload CSV File"],
        horizontal=True,
    )

    st.markdown("---")

    # Placeholders for the 3-day PV and Load arrays
    pv_3days: List[float] | None = None
    load_3days: List[float] | None = None
    day_dates: List[datetime.date] | None = None
    data_ready = False

    # =================================================================
    # 2A. MANUAL ENTRY MODE
    # =================================================================
    if input_mode == "📝 Manual Entry":
        st.subheader("📝 Enter Recent Energy Data")
        st.caption(
            "Enter solar generation and load for the 3 days before the selected prediction date. "
            "Wastage is calculated automatically as max(0, PV − Load)."
        )

        # Auto-load previous entries for exact D-3, D-2, D-1 dates if available
        default_pv = [55.0, 57.0, 59.0]
        default_load = [18.0, 19.0, 20.0]
        prev = load_manual_history_for_prediction_date(prediction_date)
        if prev is not None and len(prev) == 3:
            matched_days = 0
            for i in range(3):
                pv_val = prev.iloc[i].get("Solar_Generation")
                load_val = prev.iloc[i].get("Load")
                if pd.notna(pv_val):
                    default_pv[i] = float(pv_val)
                if pd.notna(load_val):
                    default_load[i] = float(load_val)
                if pd.notna(pv_val) and pd.notna(load_val):
                    matched_days += 1

            if matched_days == 3:
                st.success("✅ Loaded saved values for D-3, D-2, D-1 of selected date.")
            elif matched_days > 0:
                st.info(f"ℹ Loaded {matched_days}/3 matching day(s) from saved history.")

        day_labels = [
            f"{(prediction_date - datetime.timedelta(days=3)).isoformat()} (D-3)",
            f"{(prediction_date - datetime.timedelta(days=2)).isoformat()} (D-2)",
            f"{(prediction_date - datetime.timedelta(days=1)).isoformat()} (D-1)",
        ]
        day_dates = [
            prediction_date - datetime.timedelta(days=3),
            prediction_date - datetime.timedelta(days=2),
            prediction_date - datetime.timedelta(days=1),
        ]
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
            st.dataframe(sample, width='stretch', hide_index=True)

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
                            st.dataframe(last3, width='stretch', hide_index=True)

                            pv_3days = last3["Solar_Generation"].tolist()
                            load_3days = last3["Load"].tolist()
                            day_dates = [d.date() for d in last3["Date"].tolist()]
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
    weather = fetch_weather_forecast(district, prediction_date)

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

    if data_ready and pv_3days is not None and load_3days is not None and day_dates is not None:
        if st.button("💾 Save Latest Input Day (D-1) to sample CSV", width="stretch"):
            try:
                saved_path = save_manual_history_latest_day(
                    pv_latest=float(pv_3days[2]),
                    load_latest=float(load_3days[2]),
                    entry_date=day_dates[2],
                )
                st.success(f"Saved latest input day to {saved_path}")
            except Exception as ex:
                st.error(f"Failed to save inputs: {ex}")

    st.markdown("---")

    # =================================================================
    # 5. PREDICT BUTTON
    # =================================================================
    if st.button("🔮 Predict Tomorrow Wastage", width='stretch', type="primary"):
        if not data_ready or pv_3days is None or load_3days is None:
            st.error("❌ Please provide energy data (manual or CSV) before predicting.")
        else:
            try:
                # --- Compute all features from 3-day data ---
                features = compute_features_from_3days(pv_3days, load_3days)

                # --- Calendar features ---
                features.update(get_calendar_features(prediction_date))

                # --- Weather & categorical ---
                features["temp"]       = temp
                features["irradiance"] = irradiance
                features["district"]   = district
                features["season"]     = season
                features["household"]  = household

                # --- Build input DataFrame ---
                input_df = build_feature_row(feature_names, features)
                confidence_level, out_of_range_items = summarize_confidence(input_df, feature_ranges)

                # --- Predict ---
                prediction = pipeline.predict(input_df)[0]
                predicted_wastage = round(max(float(prediction), 0.0), 2)

                # --- Today's actual wastage (computed from today's inputs) ---
                todays_wastage = round(max(0.0, pv_3days[2] - load_3days[2]), 2)

                # ============================================
                # INPUT SUMMARY + CONFIDENCE
                # ============================================
                st.markdown("---")
                st.subheader("🧾 Input Summary")

                summary_view = input_df.T.reset_index()
                summary_view.columns = ["Feature", "Value"]
                st.dataframe(summary_view, width='stretch', hide_index=True)

                st.subheader("🛡️ Prediction Confidence")
                if confidence_level == "high":
                    st.success("High confidence: all numeric inputs are within training ranges.")
                elif confidence_level == "medium":
                    st.warning("Medium confidence: a few inputs are outside training ranges.")
                else:
                    st.error("Low confidence: many inputs are outside training ranges.")

                if out_of_range_items:
                    st.caption("Out-of-range fields: " + " | ".join(out_of_range_items[:8]))

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

                # ============================================
                # HISTORICAL TREND CHART
                # ============================================
                st.markdown("---")
                st.subheader("📉 Historical Trend + Tomorrow Forecast")
                hist_chart_df = pd.DataFrame()
                if day_dates is not None:
                    hist_chart_df = pd.DataFrame({
                        "Date": [d.isoformat() for d in day_dates] + [(datetime.date.today() + datetime.timedelta(days=1)).isoformat()],
                        "Wastage_kWh": [max(0.0, pv_3days[i] - load_3days[i]) for i in range(3)] + [predicted_wastage],
                        "Type": ["Actual", "Actual", "Actual", "Predicted"],
                    })
                    st.line_chart(hist_chart_df.set_index("Date")["Wastage_kWh"], width='stretch')

                report_payload = {
                    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "input_mode": input_mode,
                    "district": district,
                    "season": season,
                    "predicted_tomorrow_wastage_kwh": predicted_wastage,
                    "todays_actual_wastage_kwh": todays_wastage,
                    "temp_c": temp,
                    "irradiance_wm2": irradiance,
                    "rainfall_mm": rainfall,
                    "cloud_cover_pct": cloud_cover,
                    "confidence_level": confidence_level,
                    "model_file": str(MODEL_PATH),
                }

                st.session_state["last_prediction"] = {
                    "pv_3days": pv_3days,
                    "load_3days": load_3days,
                    "prediction_date": prediction_date.isoformat(),
                    "temp": temp,
                    "irradiance": irradiance,
                    "district": district,
                    "season": season,
                    "predicted_wastage": predicted_wastage,
                    "confidence": confidence_level,
                    "report_payload": report_payload,
                }

                st.success("Saved this prediction to session. Use sidebar sections for What-if and Report downloads.")

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
