import json
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


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )


def read_jsonl_files(folder_path: str) -> list:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_records = []
    jsonl_files = sorted(folder.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in: {folder_path}")

    for fp in jsonl_files:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Bad line in {fp.name}: {e}")

    logger.info(f"Read {len(all_records)} records from {len(jsonl_files)} files in {folder.name}")
    return all_records


def read_single_signal(base_path, zone, signal, value_field_overrides):
    folder_path = f"{base_path}/zone={zone}/{signal}"
    records = read_jsonl_files(folder_path)

    if not records:
        logger.warning(f"No data for zone={zone}, signal={signal}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    value_field = value_field_overrides.get(signal, "value")

    if value_field not in df.columns:
        logger.error(f"Field '{value_field}' not found in {signal}. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df[["datetime", value_field]].copy()
    df = df.rename(columns={value_field: signal})
    df["zone"] = zone
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    logger.info(f"  {signal} [{zone}]: {len(df)} rows, range [{df[signal].min():.1f} - {df[signal].max():.1f}]")
    return df


def merge_grid_signals(base_path, zone, signals, value_field_overrides):
    merged = None
    for signal in signals:
        df = read_single_signal(base_path, zone, signal, value_field_overrides)
        if df.empty:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["datetime", "zone"], how="outer")

    if merged is None:
        raise ValueError(f"No signal data found for zone {zone}")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    logger.info(f"Merged {len(signals)} signals for {zone}: {merged.shape}")
    return merged


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


def handle_missing_values(df, signal_cols):
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        df.loc[mask, signal_cols] = df.loc[mask, signal_cols].ffill().bfill()
    remaining = df[signal_cols].isnull().sum().sum()
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


def add_cloud_mapping(df, regions_config):
    zone_to_cloud = {r["grid_zone"]: r["cloud_mapping"] for r in regions_config}
    df["aws_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("aws", "unknown"))
    df["gcp_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("gcp", "unknown"))
    df["azure_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("azure", "unknown"))
    logger.info("Added cloud region mapping columns")
    return df


def rename_columns(df, column_mapping):
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    logger.info(f"Renamed: {list(rename_map.keys())} -> {list(rename_map.values())}")
    return df


def filter_training_window(df, start, end):
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    before = len(df)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    logger.info(f"Training window [{start} -> {end}]: {before} -> {len(df)} rows")
    return df


def process_grid_data(config, use_gcs=False):
    if use_gcs:
        base_path = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['raw_grid']}"
        output_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['processed']}"
    else:
        base_path = PROJECT_ROOT / config["local"]["raw_grid"]
        output_dir = PROJECT_ROOT / config["local"]["processed"]
        os.makedirs(output_dir, exist_ok=True)

    grid_cfg = config["grid"]
    signals = grid_cfg["priority_signals"]
    value_overrides = grid_cfg.get("value_field_overrides", {})
    col_mapping = grid_cfg["column_mapping"]
    value_ranges = grid_cfg["value_ranges"]
    regions_cfg = config["regions"]
    train_start = config["training_window"]["start"]
    train_end = config["training_window"]["end"]
    output_file = config["output"]["files"]["grid_processed"]

    # Step 1: Read and merge per zone
    logger.info("=" * 60)
    logger.info("STEP 1: Reading and merging grid signals")
    logger.info("=" * 60)

    zone_dfs = []
    for region in regions_cfg:
        zone = region["grid_zone"]
        logger.info(f"\nProcessing zone: {zone}")
        zone_df = merge_grid_signals(base_path, zone, signals, value_overrides)
        zone_dfs.append(zone_df)

    df = pd.concat(zone_dfs, ignore_index=True)
    logger.info(f"\nCombined all zones: {df.shape}")

    # Step 2: Rename columns
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Renaming and cleaning")
    logger.info("=" * 60)

    df = rename_columns(df, col_mapping)
    signal_cols = list(col_mapping.values())

    # Step 3: Clean
    df = remove_duplicates(df)

    filled = []
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone].copy()
        zone_df = fill_timeline_gaps(zone_df, zone)
        filled.append(zone_df)
    df = pd.concat(filled, ignore_index=True)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df = handle_missing_values(df, signal_cols)

    # Step 4: Validate
    df = validate_and_clip(df, value_ranges)

    # Step 5: Cloud mapping
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Cloud mapping + training window")
    logger.info("=" * 60)

    df = add_cloud_mapping(df, regions_cfg)
    df = filter_training_window(df, train_start, train_end)

    # Step 6: Save
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Saving")
    logger.info("=" * 60)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved: {output_path}")

    return df

def main():
    config = load_config()
    setup_logging(config)
    df = process_grid_data(config, use_gcs=True)
    print(f"\nGrid done! Shape: {df.shape} | Nulls: {df.isnull().sum().sum()}")
    print(f"Columns: {df.columns.tolist()}")




if __name__ == "__main__":
    main()
