"""
Evaluate all saved model pipelines and save metrics to model_metrics.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_PATH = Path("data/processed/features_regression_ready.csv")
MODELS_DIR = Path("models")
METRICS_PATH = Path("model_metrics.json")
BEST_MODEL_PATH = Path("best_model_pipeline.pkl")

# Load data
df = pd.read_csv(DATA_PATH)
target_col = "wasted_energy_kwh"
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate each model
model_files = list(MODELS_DIR.glob("*.pkl"))
metrics = {}

print(f"Found {len(model_files)} model(s) in {MODELS_DIR}/\n")

best_r2 = -np.inf
best_model_file = None

for model_file in model_files:
    model_name = model_file.stem.replace("_pipeline", "").replace("_", " ").title()
    try:
        pipeline = joblib.load(model_file)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics[model_name] = {
            "R2": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
        }

        print(f"✅ {model_name:30s} | R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_file = model_file

    except Exception as e:
        print(f"❌ {model_name}: {e}")

# Save metrics
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\n📄 Metrics saved to {METRICS_PATH}")

# Copy best model
if best_model_file:
    import shutil
    shutil.copy(best_model_file, BEST_MODEL_PATH)
    best_name = best_model_file.stem.replace("_pipeline", "").replace("_", " ").title()
    print(f"🏆 Best model: {best_name} (R2={best_r2:.4f}) → saved as {BEST_MODEL_PATH}")