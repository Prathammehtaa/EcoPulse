# src/grid_preprocessing.py
from __future__ import annotations

import json
import logging
import os
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

BASE_DIR = Path(__file__).resolve().parent.parent  # /opt/airflow
CONFIG_PATH = BASE_DIR / "pipeline_config" / "preprocessing_config.yaml"


# ----------------------------
# Config + logging
# ----------------------------
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )


# ----------------------------
# GCS helpers
# ----------------------------
def is_gs_uri(path: str) -> bool:
    return str(path).startswith("gs://")


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    p = urlparse(gs_uri)
    if p.scheme != "gs":
        raise ValueError(f"Not a gs:// URI: {gs_uri}")
    return p.netloc, p.path.lstrip("/")


def ensure_trailing_slash(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


# ----------------------------
# IO: read JSONL records (local or GCS)
# ----------------------------
def read_jsonl_records(path: str, gcs_client: Optional[storage.Client] = None) -> List[dict]:
    """
    Reads all *.jsonl under a folder/prefix and returns a list of dict records.

    path can be:
      - local folder path: /opt/airflow/data/raw/grid_signals/zone=.../signal
      - GCS prefix:        gs://bucket/raw/grid_signals/zone=.../signal
    """
    if is_gs_uri(path):
        if gcs_client is None:
            gcs_client = storage.Client()
        return read_jsonl_records_gcs(path, gcs_client)
    return read_jsonl_records_local(path)


def read_jsonl_records_local(folder_path: str) -> List[dict]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    jsonl_files = sorted(folder.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in: {folder_path}")

    all_records: List[dict] = []
    for fp in jsonl_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Bad JSON line in %s: %s", fp.name, e)
        except Exception as e:
            logger.warning("Failed reading %s: %s", fp.name, e)

    logger.info("Read %s records from %s files in %s", len(all_records), len(jsonl_files), folder.name)
    return all_records


def read_jsonl_records_gcs(prefix_gs_uri: str, client: storage.Client) -> List[dict]:
    bucket_name, prefix = parse_gs_uri(prefix_gs_uri)
    prefix = ensure_trailing_slash(prefix)

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    jsonl_blobs = [b for b in blobs if b.name.lower().endswith(".jsonl")]

    if not jsonl_blobs:
        raise FileNotFoundError(f"No JSONL files in: gs://{bucket_name}/{prefix}")

    all_records: List[dict] = []
    for b in sorted(jsonl_blobs, key=lambda x: x.name):
        try:
            data = b.download_as_bytes()
            for raw in data.splitlines():
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Bad JSON line in gs://%s/%s: %s", bucket_name, b.name, e)
        except Exception as e:
            logger.warning("Failed reading gs://%s/%s: %s", bucket_name, b.name, e)

    logger.info(
        "Read %s records from %s files in gs://%s/%s",
        len(all_records),
        len(jsonl_blobs),
        bucket_name,
        prefix,
    )
    return all_records


# ----------------------------
# Grid signal read/merge
# ----------------------------
def read_single_signal(
    base_path: str,
    zone: str,
    signal: str,
    value_field_overrides: Dict[str, str],
    gcs_client: Optional[storage.Client] = None,
) -> pd.DataFrame:
    folder_path = f"{base_path}/zone={zone}/{signal}"
    records = read_jsonl_records(folder_path, gcs_client=gcs_client)

    if not records:
        logger.warning("No data for zone=%s, signal=%s", zone, signal)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # default ElectricityMap style field is usually "value", but yours overrides some signals
    value_field = value_field_overrides.get(signal, "value")
    if value_field not in df.columns:
        logger.error(
            "Field '%s' not found for signal=%s. Available: %s",
            value_field,
            signal,
            df.columns.tolist(),
        )
        return pd.DataFrame()

    if "datetime" not in df.columns:
        logger.error("Field 'datetime' not found for signal=%s. Available: %s", signal, df.columns.tolist())
        return pd.DataFrame()

    df = df[["datetime", value_field]].copy()
    df = df.rename(columns={value_field: signal})
    df["zone"] = zone
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # avoid formatting errors if non-numeric
    try:
        mn = float(pd.to_numeric(df[signal], errors="coerce").min())
        mx = float(pd.to_numeric(df[signal], errors="coerce").max())
        logger.info("  %s [%s]: %s rows, range [%.3f - %.3f]", signal, zone, len(df), mn, mx)
    except Exception:
        logger.info("  %s [%s]: %s rows", signal, zone, len(df))

    return df


def merge_grid_signals(
    base_path: str,
    zone: str,
    signals: List[str],
    value_field_overrides: Dict[str, str],
    gcs_client: Optional[storage.Client] = None,
) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None

    for signal in signals:
        df = read_single_signal(base_path, zone, signal, value_field_overrides, gcs_client=gcs_client)
        if df.empty:
            continue
        merged = df if merged is None else pd.merge(merged, df, on=["datetime", "zone"], how="outer")

    if merged is None:
        raise ValueError(f"No signal data found for zone {zone}")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    logger.info("Merged %s signals for %s: %s", len(signals), zone, merged.shape)
    return merged


# ----------------------------
# Cleaning / enrichment
# ----------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "zone"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %s duplicate rows", removed)
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
        logger.info("Filled %s timeline gaps for %s", gaps, zone)
    return df


def handle_missing_values(df: pd.DataFrame, signal_cols: List[str]) -> pd.DataFrame:
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        df.loc[mask, signal_cols] = df.loc[mask, signal_cols].ffill().bfill()
    remaining = int(df[signal_cols].isnull().sum().sum())
    logger.info("Remaining nulls after ffill/bfill: %s", remaining)
    return df


def validate_and_clip(df: pd.DataFrame, value_ranges: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    for col, bounds in value_ranges.items():
        if col not in df.columns:
            continue
        outliers = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).sum()
        if outliers > 0:
            logger.warning("Clipping %s outliers in %s", int(outliers), col)
        df[col] = df[col].clip(lower=bounds["min"], upper=bounds["max"])
    return df


def add_cloud_mapping(df: pd.DataFrame, regions_config: List[dict]) -> pd.DataFrame:
    zone_to_cloud = {r["grid_zone"]: r.get("cloud_mapping", {}) for r in regions_config}
    df["aws_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("aws", "unknown"))
    df["gcp_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("gcp", "unknown"))
    df["azure_region"] = df["zone"].map(lambda z: zone_to_cloud.get(z, {}).get("azure", "unknown"))
    logger.info("Added cloud region mapping columns")
    return df


def rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    logger.info("Renamed: %s -> %s", list(rename_map.keys()), list(rename_map.values()))
    return df


def filter_training_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    before = len(df)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    logger.info("Training window [%s -> %s]: %s -> %s rows", start, end, before, len(df))
    return df


# ----------------------------
# IO: write parquet (local or GCS)
# ----------------------------
def write_parquet_local(df: pd.DataFrame, out_path: str) -> None:
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info("Saved: %s", out_path)


def write_parquet_gcs(df: pd.DataFrame, out_gs_uri: str, client: storage.Client) -> None:
    bucket_name, obj_name = parse_gs_uri(out_gs_uri)

    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    bucket = client.bucket(bucket_name)
    bucket.blob(obj_name).upload_from_file(buf, content_type="application/octet-stream")
    logger.info("Saved: gs://%s/%s", bucket_name, obj_name)


# ----------------------------
# Pipeline
# ----------------------------
def process_grid_data(config: dict, use_gcs: bool = True) -> pd.DataFrame:
    grid_cfg = config["grid"]
    signals = grid_cfg["priority_signals"]
    value_overrides = grid_cfg.get("value_field_overrides", {})
    col_mapping = grid_cfg["column_mapping"]
    value_ranges = grid_cfg["value_ranges"]

    regions_cfg = config["regions"]
    train_start = config["training_window"]["start"]
    train_end = config["training_window"]["end"]
    output_file = config["output"]["files"]["grid_processed"]

    if use_gcs:
        bucket = config["gcs"]["bucket"]
        raw_grid_prefix = config["gcs"]["paths"]["raw_grid"]      # raw/grid_signals
        processed_prefix = config["gcs"]["paths"]["processed"]    # processed
        project_id = config["gcs"].get("project_id")

        gcs_client = storage.Client(project=project_id) if project_id else storage.Client()
        base_path = f"gs://{bucket}/{raw_grid_prefix}"
        out_gs_uri = f"gs://{bucket}/{processed_prefix.rstrip('/')}/{output_file}"
    else:
        gcs_client = None
        base_path = str(BASE_DIR / config["local"]["raw_grid"])
        out_local_path = str((BASE_DIR / config["local"]["processed"]) / output_file)

    # Step 1: Read and merge per zone
    logger.info("=" * 60)
    logger.info("STEP 1: Reading and merging grid signals")
    logger.info("=" * 60)

    zone_dfs: List[pd.DataFrame] = []
    for region in regions_cfg:
        zone = region["grid_zone"]
        logger.info("\nProcessing zone: %s", zone)
        zone_df = merge_grid_signals(base_path, zone, signals, value_overrides, gcs_client=gcs_client)
        zone_dfs.append(zone_df)

    df = pd.concat(zone_dfs, ignore_index=True)
    logger.info("\nCombined all zones: %s", df.shape)

    # Step 2: Rename columns
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Renaming and cleaning")
    logger.info("=" * 60)

    df = rename_columns(df, col_mapping)
    signal_cols = [v for v in col_mapping.values() if v in df.columns]

    # Step 3: Clean
    df = remove_duplicates(df)

    filled: List[pd.DataFrame] = []
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone].copy()
        zone_df = fill_timeline_gaps(zone_df, zone)
        filled.append(zone_df)
    df = pd.concat(filled, ignore_index=True)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)
    df = handle_missing_values(df, signal_cols)
    df = validate_and_clip(df, value_ranges)

    # Step 4: Cloud mapping + training window
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Cloud mapping + training window")
    logger.info("=" * 60)

    df = add_cloud_mapping(df, regions_cfg)
    df = filter_training_window(df, train_start, train_end)

    # Step 5: Save
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Saving")
    logger.info("=" * 60)

    df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)

    if use_gcs:
        write_parquet_gcs(df, out_gs_uri, gcs_client)
    else:
        write_parquet_local(df, out_local_path)

    return df


def main() -> None:
    config = load_config()
    setup_logging(config)
    df = process_grid_data(config, use_gcs=True)
    logger.info("Grid done! Shape: %s | Nulls: %s", df.shape, int(df.isnull().sum().sum()))
    logger.info("Columns: %s", df.columns.tolist())


if __name__ == "__main__":
    main()