import logging
import os
from pathlib import Path
from typing import Dict, List

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


def read_weather_csvs(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Weather folder not found: {folder_path}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {folder_path}")

    dfs = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {fp.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Read {len(combined)} rows from {len(csv_files)} files in {folder.name}")
    return combined


def read_all_regions(base_path, regions_config):
    all_dfs = []
    for region in regions_config:
        folder_name = region["weather_folder"]
        zone = region["grid_zone"]
        folder_path = os.path.join(base_path, folder_name)

        logger.info(f"Reading weather: {folder_name} -> {zone}")
        df = read_weather_csvs(folder_path)
        df["zone"] = zone
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined all regions: {combined.shape}")
    return combined


def select_and_rename(df, datetime_col, selected_features, column_mapping):
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found. Available: {df.columns.tolist()}")

    keep = [datetime_col, "zone"]
    rename_map = {}

    for feat in selected_features:
        if feat in df.columns:
            keep.append(feat)
            if feat in column_mapping:
                rename_map[feat] = column_mapping[feat]
        else:
            logger.warning(f"Feature '{feat}' not found in CSV. Skipping.")

    df = df[keep].copy()
    rename_map[datetime_col] = "datetime"
    df = df.rename(columns=rename_map)

    logger.info(f"Selected {len(keep) - 2} features. Columns: {df.columns.tolist()}")
    return df


def parse_datetime(df):
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    logger.info(f"Parsed datetimes: {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "zone"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    return df


def fill_timeline_gaps(df, zone):
    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()
    full_timeline = pd.date_range(start=min_dt, end=max_dt, freq="h", tz="UTC")
    timeline_df = pd.DataFrame({"datetime": full_timeline, "zone": zone})

    before = len(df)
    df = pd.merge(timeline_df, df, on=["datetime", "zone"], how="left")
    gaps = len(df) - before
    if gaps > 0:
        logger.info(f"Filled {gaps} timeline gaps for {zone}")
    return df


def handle_missing_values(df, feature_cols):
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        df.loc[mask, feature_cols] = df.loc[mask, feature_cols].ffill().bfill()
    remaining = df[feature_cols].isnull().sum().sum()
    logger.info(f"Remaining nulls after ffill/bfill: {remaining}")
    return df


def validate_and_clip(df, value_ranges):
    for col, bounds in value_ranges.items():
        if col in df.columns:
            outliers = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).sum()
            if outliers > 0:
                logger.warning(f"Clipping {outliers} outliers in {col}")
            df[col] = df[col].clip(lower=bounds["min"], upper=bounds["max"])
    return df


def filter_training_window(df, start, end):
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    before = len(df)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    logger.info(f"Training window [{start} -> {end}]: {before} -> {len(df)} rows")
    return df


def process_weather_data(config, use_gcs=False):
    if use_gcs:
        base_path = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['raw_weather']}"
        output_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['processed']}"
    else:
        base_path = PROJECT_ROOT / config["local"]["raw_weather"]
        output_dir = PROJECT_ROOT / config["local"]["processed"]
        os.makedirs(output_dir, exist_ok=True)

    weather_cfg = config["weather"]
    datetime_col = weather_cfg["datetime_column"]
    selected = weather_cfg["selected_features"]
    col_mapping = weather_cfg["column_mapping"]
    value_ranges = weather_cfg["value_ranges"]
    regions_cfg = config["regions"]
    train_start = config["training_window"]["start"]
    train_end = config["training_window"]["end"]
    output_file = config["output"]["files"]["weather_processed"]

    # Step 1: Read all regions
    logger.info("=" * 60)
    logger.info("STEP 1: Reading weather data")
    logger.info("=" * 60)

    df = read_all_regions(base_path, regions_cfg)
    logger.info(f"Raw columns: {df.columns.tolist()}")

    # Step 2: Select and rename
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Selecting and renaming features")
    logger.info("=" * 60)

    df = select_and_rename(df, datetime_col, selected, col_mapping)
    feature_cols = [c for c in df.columns if c not in ("datetime", "zone")]

    # Step 3: Parse and clean
    df = parse_datetime(df)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Cleaning")
    logger.info("=" * 60)

    df = remove_duplicates(df)

    filled = []
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone].copy()
        zone_df = fill_timeline_gaps(zone_df, zone)
        filled.append(zone_df)
    df = pd.concat(filled, ignore_index=True)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df = handle_missing_values(df, feature_cols)
    df = validate_and_clip(df, value_ranges)

    # Step 4: Filter and save
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training window + saving")
    logger.info("=" * 60)

    df = filter_training_window(df, train_start, train_end)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved: {output_path}")

    return df

def main():
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )
    df = process_weather_data(config, use_gcs=True)
    print(f"\nWeather done! Shape: {df.shape} | Nulls: {df.isnull().sum().sum()}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()