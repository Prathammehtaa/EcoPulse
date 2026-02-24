import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "preprocessing_config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(grid_path, weather_path):
    logger.info(f"Loading grid: {grid_path}")
    grid = pd.read_parquet(grid_path)
    grid["datetime"] = pd.to_datetime(grid["datetime"], utc=True)

    logger.info(f"Loading weather: {weather_path}")
    weather = pd.read_parquet(weather_path)
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)

    logger.info(f"Grid: {grid.shape}, Weather: {weather.shape}")
    return grid, weather


def merge_datasets(grid, weather):
    merged = pd.merge(grid, weather, on=["datetime", "zone"], how="inner")
    logger.info(f"Merged: {merged.shape} (grid={len(grid)}, weather={len(weather)}, merged={len(merged)})")

    row_loss = len(grid) - len(merged)
    if row_loss > 0:
        logger.warning(f"Lost {row_loss} rows ({row_loss/len(grid)*100:.1f}%) in join")

    zones = merged["zone"].unique().tolist()
    logger.info(f"Zones: {zones}")
    logger.info(f"Zones: {zones}")
    return merged


def add_temporal_features(df):
    dt = df["datetime"]
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_daytime"] = ((dt.dt.hour >= 6) & (dt.dt.hour <= 20)).astype(int)

    season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
                  6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    df["season"] = dt.dt.month.map(season_map)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Added temporal features")
    return df


def add_lag_features(df, target, lags):
    for lag in lags:
        col = f"lag_{lag}h"
        df[col] = df.groupby("zone")[target].shift(lag)
    logger.info(f"Added lag features: {lags}")
    return df


def add_rolling_features(df, target, windows):
    for w in windows:
        df[f"rolling_mean_{w}h"] = (
            df.groupby("zone")[target]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
    df["rolling_std_24h"] = (
        df.groupby("zone")[target]
        .transform(lambda x: x.rolling(24, min_periods=1).std())
    )
    logger.info(f"Added rolling features: {windows}")
    return df


def add_interaction_features(df):
    if "shortwave_radiation_wm2" in df.columns and "cloud_cover_pct" in df.columns:
        df["solar_potential"] = df["shortwave_radiation_wm2"] * (1 - df["cloud_cover_pct"] / 100)

    if "temperature_2m_c" in df.columns:
        df["temp_demand_proxy"] = np.abs(df["temperature_2m_c"] - 20)

    target = "carbon_intensity_gco2_per_kwh"
    if target in df.columns:
        df["carbon_change_1h"] = df.groupby("zone")[target].diff(1)

    logger.info("Added interaction features")
    return df


def handle_feature_nulls(df):
    lag_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "carbon_change"))]
    for col in lag_cols:
        df[col] = df.groupby("zone")[col].transform(lambda x: x.fillna(x.median()))
    df = df.fillna(0)
    logger.info(f"Handled feature nulls. Total nulls: {df.isnull().sum().sum()}")
    return df


def run_merge_and_feature_engineering(config, use_gcs=False):
    if use_gcs:
        proc_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['processed']}"
        feat_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['features']}"
    else:
        proc_dir = PROJECT_ROOT / config["local"]["processed"]
        feat_dir = PROJECT_ROOT / config["local"]["features"]
        os.makedirs(feat_dir, exist_ok=True)

    grid_path = os.path.join(proc_dir, config["output"]["files"]["grid_processed"])
    weather_path = os.path.join(proc_dir, config["output"]["files"]["weather_processed"])
    merged_path = os.path.join(proc_dir, config["output"]["files"]["merged"])
    feature_path = os.path.join(feat_dir, config["output"]["files"]["feature_table"])

    target = config["features"]["target_column"]
    lags = config["features"]["lag_hours"]
    windows = config["features"]["rolling_windows"]

    # Phase 1: Merge
    logger.info("=" * 60)
    logger.info("PHASE 1: Merging datasets")
    logger.info("=" * 60)

    grid, weather = load_processed_data(grid_path, weather_path)
    df = merge_datasets(grid, weather)

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    df.to_parquet(merged_path, index=False, compression="snappy")
    logger.info(f"Saved merged: {merged_path}")

    # Phase 2: Features
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Feature engineering")
    logger.info("=" * 60)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_lag_features(df, target, lags)
    df = add_rolling_features(df, target, windows)
    df = add_interaction_features(df)

    # Phase 3: Save
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Final cleanup and save")
    logger.info("=" * 60)

    df = handle_feature_nulls(df)
    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df.to_parquet(feature_path, index=False, compression="snappy")
    logger.info(f"Saved feature table: {feature_path}")
    logger.info(f"Shape: {df.shape} | Columns: {len(df.columns)}")

    return df

def main():
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )
    df = run_merge_and_feature_engineering(config, use_gcs=False)
    print(f"\nFeature table done! Shape: {df.shape}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    main()