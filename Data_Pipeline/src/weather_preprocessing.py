# src/weather_preprocessing.py

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from google.cloud import storage

logger = logging.getLogger(__name__)

# Project layout:
# /opt/airflow/src/weather_preprocessing.py
# /opt/airflow/pipeline_config/preprocessing_config.yaml
BASE_DIR = Path(__file__).resolve().parent.parent  # /opt/airflow
CONFIG_PATH = BASE_DIR / "pipeline_config" / "preprocessing_config.yaml"


# ----------------------------
# Config
# ----------------------------
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# GCS helpers
# ----------------------------
def is_gs_uri(path: str) -> bool:
    return str(path).startswith("gs://")


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    """
    Returns (bucket, object_prefix_or_name)
    Example: gs://my-bucket/raw/weather -> ("my-bucket", "raw/weather")
    """
    p = urlparse(gs_uri)
    if p.scheme != "gs":
        raise ValueError(f"Not a gs:// URI: {gs_uri}")
    return p.netloc, p.path.lstrip("/")


def ensure_trailing_slash(prefix: str) -> str:
    if prefix and not prefix.endswith("/"):
        return prefix + "/"
    return prefix


# ----------------------------
# IO: read weather CSVs (local or GCS)
# ----------------------------
def read_weather_csvs(folder_path: str, gcs_client: Optional[storage.Client] = None) -> pd.DataFrame:
    """
    Reads all CSVs in a folder and concatenates them.
    folder_path can be:
      - local folder path
      - gs://bucket/prefix/folder
    """
    if is_gs_uri(folder_path):
        if gcs_client is None:
            gcs_client = storage.Client()
        return read_weather_csvs_gcs(folder_path, gcs_client)

    return read_weather_csvs_local(folder_path)


def read_weather_csvs_local(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Weather folder not found: {folder_path}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {folder_path}")

    dfs: List[pd.DataFrame] = []
    for fp in csv_files:
        try:
            dfs.append(pd.read_csv(fp))
        except Exception as e:
            logger.warning(f"Failed to read {fp.name}: {e}")

    if not dfs:
        raise FileNotFoundError(f"All CSV reads failed in: {folder_path}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Read {len(combined)} rows from {len(csv_files)} files in {folder.name}")
    return combined


def read_weather_csvs_gcs(folder_gs_uri: str, client: storage.Client) -> pd.DataFrame:
    bucket_name, prefix = parse_gs_uri(folder_gs_uri)
    prefix = ensure_trailing_slash(prefix)

    # List objects under prefix and keep CSVs
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    csv_blobs = [b for b in blobs if b.name.lower().endswith(".csv")]

    if not csv_blobs:
        raise FileNotFoundError(f"No CSV files in: gs://{bucket_name}/{prefix}")

    dfs: List[pd.DataFrame] = []
    for b in sorted(csv_blobs, key=lambda x: x.name):
        try:
            data = b.download_as_bytes()
            dfs.append(pd.read_csv(BytesIO(data)))
        except Exception as e:
            logger.warning(f"Failed to read gs://{bucket_name}/{b.name}: {e}")

    if not dfs:
        raise FileNotFoundError(f"All CSV reads failed in: gs://{bucket_name}/{prefix}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Read {len(combined)} rows from {len(csv_blobs)} files in gs://{bucket_name}/{prefix}")
    return combined


def read_all_regions(base_path: str, regions_config: List[dict], gcs_client: Optional[storage.Client] = None) -> pd.DataFrame:
    """
    base_path:
      - local base folder (e.g., /opt/airflow/data/raw/weather)
      - gs://bucket/raw/weather
    Each region has a weather_folder, which is appended to base_path.
    """
    all_dfs: List[pd.DataFrame] = []

    for region in regions_config:
        folder_name = region["weather_folder"]
        zone = region["grid_zone"]

        folder_path = os.path.join(base_path, folder_name)
        logger.info(f"Reading weather: {folder_name} -> {zone}")

        df = read_weather_csvs(folder_path, gcs_client=gcs_client)
        df["zone"] = zone
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No region weather data could be read.")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined all regions: {combined.shape}")
    return combined


# ----------------------------
# Transformations
# ----------------------------
def select_and_rename(
    df: pd.DataFrame,
    datetime_col: str,
    selected_features: List[str],
    column_mapping: Dict[str, str],
) -> pd.DataFrame:
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found. Available: {df.columns.tolist()}")

    keep = [datetime_col, "zone"]
    rename_map: Dict[str, str] = {datetime_col: "datetime"}

    for feat in selected_features:
        if feat in df.columns:
            keep.append(feat)
            rename_map[feat] = column_mapping.get(feat, feat)
        else:
            logger.warning(f"Feature '{feat}' not found in CSV. Skipping.")

    df = df[keep].copy()
    df = df.rename(columns=rename_map)

    logger.info(f"Selected {len(keep) - 2} features. Columns: {df.columns.tolist()}")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    logger.info(f"Parsed datetimes: {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "zone"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    return df


def fill_timeline_gaps(df: pd.DataFrame, zone: str) -> pd.DataFrame:
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


def handle_missing_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        df.loc[mask, feature_cols] = df.loc[mask, feature_cols].ffill().bfill()

    remaining = int(df[feature_cols].isnull().sum().sum())
    logger.info(f"Remaining nulls after ffill/bfill: {remaining}")
    return df


def validate_and_clip(df: pd.DataFrame, value_ranges: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    for col, bounds in value_ranges.items():
        if col not in df.columns:
            continue
        outliers = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).sum()
        if outliers > 0:
            logger.warning(f"Clipping {int(outliers)} outliers in {col}")
        df[col] = df[col].clip(lower=bounds["min"], upper=bounds["max"])
    return df


def filter_training_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start, tz="UTC")
    # inclusive end-of-day
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    before = len(df)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    logger.info(f"Training window [{start} -> {end}]: {before} -> {len(df)} rows")
    return df


# ----------------------------
# IO: write parquet (local or GCS)
# ----------------------------
def write_parquet_local(df: pd.DataFrame, out_path: str) -> None:
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"Saved: {out_path}")


def write_parquet_gcs(df: pd.DataFrame, out_gs_uri: str, client: storage.Client) -> None:
    bucket_name, obj_name = parse_gs_uri(out_gs_uri)

    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    bucket = client.bucket(bucket_name)
    bucket.blob(obj_name).upload_from_file(buf, content_type="application/octet-stream")
    logger.info(f"Saved: gs://{bucket_name}/{obj_name}")


# ----------------------------
# Main pipeline
# ----------------------------
def process_weather_data(config: dict, use_gcs: bool = True) -> pd.DataFrame:
    weather_cfg = config["weather"]
    datetime_col = weather_cfg["datetime_column"]
    selected = weather_cfg["selected_features"]
    col_mapping = weather_cfg["column_mapping"]
    value_ranges = weather_cfg["value_ranges"]

    regions_cfg = config["regions"]
    train_start = config["training_window"]["start"]
    train_end = config["training_window"]["end"]
    output_file = config["output"]["files"]["weather_processed"]

    if use_gcs:
        bucket = config["gcs"]["bucket"]
        raw_weather_prefix = config["gcs"]["paths"]["raw_weather"]  # e.g., raw/weather
        processed_prefix = config["gcs"]["paths"]["processed"]      # e.g., processed
        project_id = config["gcs"].get("project_id")

        gcs_client = storage.Client(project=project_id) if project_id else storage.Client()
        base_path = f"gs://{bucket}/{raw_weather_prefix}"
        out_gs_uri = f"gs://{bucket}/{processed_prefix.rstrip('/')}/{output_file}"
    else:
        gcs_client = None
        project_root = BASE_DIR
        base_path = str(project_root / config["local"]["raw_weather"])
        out_local_path = str((project_root / config["local"]["processed"]) / output_file)

    logger.info("=" * 60)
    logger.info("STEP 1: Reading weather data")
    logger.info("=" * 60)

    df = read_all_regions(base_path, regions_cfg, gcs_client=gcs_client)
    logger.info(f"Raw columns: {df.columns.tolist()}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Selecting and renaming features")
    logger.info("=" * 60)

    df = select_and_rename(df, datetime_col, selected, col_mapping)
    feature_cols = [c for c in df.columns if c not in ("datetime", "zone")]

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Cleaning")
    logger.info("=" * 60)

    df = parse_datetime(df)
    df = remove_duplicates(df)

    filled: List[pd.DataFrame] = []
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone].copy()
        zone_df = fill_timeline_gaps(zone_df, zone)
        filled.append(zone_df)
    df = pd.concat(filled, ignore_index=True)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df = handle_missing_values(df, feature_cols)
    df = validate_and_clip(df, value_ranges)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training window + saving")
    logger.info("=" * 60)

    df = filter_training_window(df, train_start, train_end)
    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)

    if use_gcs:
        write_parquet_gcs(df, out_gs_uri, gcs_client)
    else:
        write_parquet_local(df, out_local_path)

    return df


def main() -> None:
    config = load_config()

    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )

    df = process_weather_data(config, use_gcs=True)
    logger.info(f"Weather done! Shape: {df.shape} | Nulls: {int(df.isnull().sum().sum())}")
    logger.info(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()