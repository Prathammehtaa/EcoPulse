"""
Step 3: Label Engineering & Temporal Split
==========================================
Creates forecast targets + chronological train/val/test split + baseline models
Output: model_ready.parquet, train/val/test.parquet, baseline_results.json
"""

import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from google.cloud import storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# local folders
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "modeling")

# adjust this if your config file lives somewhere else
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pipeline_config", "ingestion_config.yaml")


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_table():
    """Load engineered features from local processed folder."""
    input_path = os.path.join(FEATURES_DIR, "feature_table.parquet")
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def to_native_types(obj):
    """Recursively convert numpy/pandas types into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_native_types(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    if pd.isna(obj):
        return None

    return obj


def add_forecast_targets(df, horizons=[1, 6, 12, 24]):
    """Create shifted target columns for forecasting."""
    print("\n--- ADDING FORECAST TARGETS ---")

    for zone in df["zone"].unique():
        zone_mask = df["zone"] == zone
        zone_data = df.loc[zone_mask].copy()

        for horizon in horizons:
            target_col = f"carbon_intensity_target_{horizon}h"
            df.loc[zone_mask, target_col] = (
                zone_data["carbon_intensity_gco2_per_kwh"].shift(-horizon).values
            )

    print(f'  Added {len(horizons)} forecast targets: {[f"target_{h}h" for h in horizons]}')
    return df


def temporal_train_val_test_split(df):
    """Split chronologically: Train | Val | Test."""
    print("\n--- TEMPORAL TRAIN/VAL/TEST SPLIT ---")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    train_end = pd.Timestamp("2025-06-30 23:00:00", tz="UTC")
    val_end = pd.Timestamp("2025-09-30 23:00:00", tz="UTC")

    train_df = df[df["datetime"] <= train_end].copy()
    val_df = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test_df = df[df["datetime"] > val_end].copy()

    print(f'  Train: {len(train_df)} rows ({train_df["datetime"].min()} to {train_df["datetime"].max()})')
    print(f'  Val:   {len(val_df)} rows ({val_df["datetime"].min()} to {val_df["datetime"].max()})')
    print(f'  Test:  {len(test_df)} rows ({test_df["datetime"].min()} to {test_df["datetime"].max()})')
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} rows")

    return train_df, val_df, test_df


def compute_baselines(df):
    """Compute baseline models for 1h forecast."""
    print("\n--- COMPUTING BASELINES ---")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    target_col = "carbon_intensity_gco2_per_kwh"

    baselines = {
        "persistence_1h": {"mae": 0, "rmse": 0, "mape": 0},
        "persistence_24h": {"mae": 0, "rmse": 0, "mape": 0},
        "historical_mean": {"mae": 0, "rmse": 0, "mape": 0},
    }

    df["pred_persistence_1h"] = df[target_col].shift(1)

    for zone in df["zone"].unique():
        zone_mask = df["zone"] == zone
        zone_data = df.loc[zone_mask].copy()
        df.loc[zone_mask, "pred_persistence_24h"] = zone_data[target_col].shift(24).values

    hourly_means = df.groupby("hour_of_day")[target_col].mean()
    df["pred_historical_mean"] = df["hour_of_day"].map(hourly_means)

    for pred_col, baseline_name in [
        ("pred_persistence_1h", "persistence_1h"),
        ("pred_persistence_24h", "persistence_24h"),
        ("pred_historical_mean", "historical_mean"),
    ]:
        valid_mask = df[[target_col, pred_col]].notna().all(axis=1)
        y_true = df.loc[valid_mask, target_col].values
        y_pred = df.loc[valid_mask, pred_col].values

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        baselines[baseline_name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
        }

        print(f"  {baseline_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    df = df.drop(
        columns=[
            "pred_persistence_1h",
            "pred_persistence_24h",
            "pred_historical_mean",
        ]
    )

    return df, baselines


def remove_rows_with_nan_targets(df):
    """Remove rows where any target is NaN."""
    print("\n--- CLEANING TARGET ROWS ---")

    initial_rows = len(df)
    target_cols = [col for col in df.columns if col.startswith("carbon_intensity_target_")]

    df = df.dropna(subset=target_cols).reset_index(drop=True)

    rows_removed = initial_rows - len(df)
    print(f"  Rows before: {initial_rows}")
    print(f"  Rows removed (NaN targets): {rows_removed}")
    print(f"  Rows after: {len(df)}")

    return df


def upload_parquet_to_gcs(df, bucket_name, object_path, gcs_client):
    """Upload a DataFrame as parquet to GCS."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    buffer.seek(0)

    blob = gcs_client.bucket(bucket_name).blob(object_path)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print(f"  ☁️ Uploaded gs://{bucket_name}/{object_path}")


def upload_json_to_gcs(data, bucket_name, object_path, gcs_client):
    """Upload a JSON object to GCS."""
    safe_data = to_native_types(data)
    blob = gcs_client.bucket(bucket_name).blob(object_path)
    blob.upload_from_string(
        json.dumps(safe_data, indent=2),
        content_type="application/json",
    )
    print(f"  ☁️ Uploaded gs://{bucket_name}/{object_path}")


def save_outputs(df, train_df, val_df, test_df, baselines, config):
    """Save outputs locally and upload to GCS using existing YAML config."""
    print("\n--- SAVING OUTPUTS ---")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    bucket_name = config["gcs"]["bucket"]
    processed_prefix = "processed"
    reports_prefix = "reports/modeling"

    gcs_client = storage.Client()

    # full model-ready dataset
    model_ready_path = os.path.join(PROCESSED_DIR, "model_ready.parquet")
    df.to_parquet(model_ready_path, index=False)
    print(f"  ✅ {model_ready_path} ({len(df)} rows)")

    upload_parquet_to_gcs(
        df,
        bucket_name,
        f"{processed_prefix}/model_ready.parquet",
        gcs_client,
    )

    # train / val / test
    train_path = os.path.join(PROCESSED_DIR, "train_split.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val_split.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_split.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"  ✅ {train_path} ({len(train_df)} rows)")
    print(f"  ✅ {val_path} ({len(val_df)} rows)")
    print(f"  ✅ {test_path} ({len(test_df)} rows)")

    upload_parquet_to_gcs(
        train_df,
        bucket_name,
        f"{processed_prefix}/train_split.parquet",
        gcs_client,
    )
    upload_parquet_to_gcs(
        val_df,
        bucket_name,
        f"{processed_prefix}/val_split.parquet",
        gcs_client,
    )
    upload_parquet_to_gcs(
        test_df,
        bucket_name,
        f"{processed_prefix}/test_split.parquet",
        gcs_client,
    )

    # baseline results
    baseline_data = {
        "generated_at": datetime.now().isoformat(),
        "baselines": to_native_types(baselines),
        "data_split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "total_rows": int(len(df)),
        },
    }

    baseline_data = to_native_types(baseline_data)

    baseline_path = os.path.join(REPORTS_DIR, "baseline_results.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"  ✅ {baseline_path}")

    upload_json_to_gcs(
        baseline_data,
        bucket_name,
        f"{reports_prefix}/baseline_results.json",
        gcs_client,
    )


def main():
    print("=" * 60)
    print("STEP 3: LABEL ENGINEERING & TEMPORAL SPLIT")
    print("=" * 60)

    config = load_config()

    df = load_feature_table()
    df = add_forecast_targets(df)

    train_df, val_df, test_df = temporal_train_val_test_split(df)

    df, baselines = compute_baselines(df)

    df = remove_rows_with_nan_targets(df)
    train_df = remove_rows_with_nan_targets(train_df)
    val_df = remove_rows_with_nan_targets(val_df)
    test_df = remove_rows_with_nan_targets(test_df)

    save_outputs(df, train_df, val_df, test_df, baselines, config)

    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE")
    print("=" * 60)
    print("Ready for: Model training (XGBoost, LightGBM, etc.)")


if __name__ == "__main__":
    main()