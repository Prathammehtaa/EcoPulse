"""
EcoPulse Model Pipeline — Shared Utilities
===========================================
Every modelling script imports from here.
No duplicated code, no hardcoded values.

"""

import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ecopulse-model")

# ============================================================
# PATHS — Relative to Model_Pipeline/
# ============================================================
# ============================================================
# PATHS — Auto-detect based on repo structure
# ============================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PIPELINE_DIR = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(MODEL_PIPELINE_DIR)

TRAIN_PATH = os.path.join(REPO_ROOT, "Data_Pipeline", "data", "processed", "train_split.parquet")
VAL_PATH = os.path.join(REPO_ROOT, "Data_Pipeline", "data", "processed", "val_split.parquet")
TEST_PATH = os.path.join(REPO_ROOT, "Data_Pipeline", "data", "processed", "test_split.parquet")

MODELS_DIR = os.path.join(MODEL_PIPELINE_DIR, "models")
REPORTS_DIR = os.path.join(MODEL_PIPELINE_DIR, "reports")
DATA_DIR = os.path.join(MODEL_PIPELINE_DIR, "data")

# ============================================================
# COLUMN CONFIGURATION (verified from 66/66 validation)
# ============================================================
TARGET_COL = "carbon_intensity_gco2_per_kwh"
DATETIME_COL = "datetime"
ZONE_COL = "zone"

FORECAST_TARGETS = {
    1: "carbon_intensity_target_1h",
    6: "carbon_intensity_target_6h",
    12: "carbon_intensity_target_12h",
    24: "carbon_intensity_target_24h",
}

HORIZONS = [1, 12, 24]

# Columns to EXCLUDE from features
DROP_COLS = [
    TARGET_COL, DATETIME_COL, ZONE_COL,
    "aws_region", "gcp_region", "azure_region", "season",
    # All target columns
    "carbon_intensity_target_1h",
    "carbon_intensity_target_6h",
    "carbon_intensity_target_12h",
    "carbon_intensity_target_24h",
]


# ============================================================
# DATA LOADING
# ============================================================
def load_split(name: str) -> pd.DataFrame:
    """Load a single split by name ('train', 'val', 'test')."""
    paths = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    if name not in paths:
        raise ValueError(f"Unknown split: {name}. Use 'train', 'val', or 'test'.")
    
    df = pd.read_parquet(paths[name])
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    logger.info(f"Loaded {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_all_splits() -> dict:
    """Load train, val, test. Returns dict of DataFrames."""
    return {name: load_split(name) for name in ["train", "val", "test"]}


# ============================================================
# FEATURE PREPARATION
# ============================================================
def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get usable feature column names.
    Excludes target, datetime, zone, cloud regions, season (string),
    and all forecast target columns.
    """
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS
        and df[c].dtype in ["float64", "float32", "int64", "int32", "uint32", "UInt32"]
    ]
    return sorted(feature_cols)


def prepare_Xy(df: pd.DataFrame, horizon: int) -> tuple:
    """
    Prepare feature matrix X and target vector y for a given horizon.
    One-hot encodes the zone column.
    
    Returns: (X, y, feature_column_names)
    """
    target_col = FORECAST_TARGETS[horizon]
    
    # One-hot encode zone
    df_encoded = pd.get_dummies(df, columns=[ZONE_COL], prefix="zone", dtype=int)
    
    # Get feature columns (now includes zone dummies)
    feature_cols = get_feature_columns(df_encoded)
    zone_dummies = [c for c in df_encoded.columns if c.startswith("zone_")]
    feature_cols = sorted(set(feature_cols + zone_dummies))
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    return X, y, feature_cols


def align_columns(X_train, X_val, X_test):
    """Ensure all splits have the same columns in the same order."""
    common = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    return X_train[common], X_val[common], X_test[common], common


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R², MAPE for regression."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "mape": round(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100, 4),
    }


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print a metrics dict."""
    print(f"  {label:<40} MAE={metrics['mae']:<10.2f} "
          f"RMSE={metrics['rmse']:<10.2f} R²={metrics['r2']:<10.4f} "
          f"MAPE={metrics['mape']:.2f}%")


def print_metrics_table(results: list):
    """Print a formatted table from a list of metric dicts."""
    print(f"\n  {'Model':<40} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['model']:<40} {r['mae']:>8.2f} {r['rmse']:>8.2f} "
              f"{r['r2']:>8.4f} {r['mape']:>8.2f}")


# ============================================================
# FILE HELPERS
# ============================================================
def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MODELS_DIR, REPORTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


def save_results(results: list, filename: str):
    """Save results list to CSV in reports/."""
    ensure_dirs()
    path = os.path.join(REPORTS_DIR, filename)
    pd.DataFrame(results).to_csv(path, index=False)
    logger.info(f"Saved results to {path}")


def save_model(model, name: str):
    """Save a trained model to models/."""
    import joblib
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


def get_timestamp():
    """Get current timestamp string for run naming."""
    return datetime.now().strftime("%Y%m%d_%H%M")