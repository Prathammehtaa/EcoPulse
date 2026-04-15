from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import pandas as pd
import yaml
from google.cloud import storage


# -------------------------------------------------------
# Config
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INGESTION_CONFIG_PATH = BASE_DIR / "pipeline_config" / "ingestion_config.yaml"
PREPROCESSING_CONFIG_PATH = BASE_DIR / "pipeline_config" / "preprocessing_config.yaml"


def load_ingestion_config() -> dict:
    with open(INGESTION_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_preprocessing_config() -> dict:
    with open(PREPROCESSING_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# GCS helpers
# -------------------------------------------------------
def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    p = urlparse(gs_uri)
    if p.scheme != "gs":
        raise ValueError(f"Not a gs:// URI: {gs_uri}")
    return p.netloc, p.path.lstrip("/")


def list_blobs_with_suffix(client: storage.Client, bucket_name: str, prefix: str, suffix: str):
    return [b for b in client.list_blobs(bucket_name, prefix=prefix) if b.name.lower().endswith(suffix.lower())]


def latest_blob_for_prefix(client: storage.Client, bucket_name: str, prefix: str, suffix: str):
    blobs = list_blobs_with_suffix(client, bucket_name, prefix, suffix)
    if not blobs:
        return None
    return sorted(blobs, key=lambda b: b.name)[-1]


# -------------------------------------------------------
# File readers
# -------------------------------------------------------
def read_jsonl_blob(blob) -> List[dict]:
    raw = blob.download_as_bytes()
    records: List[dict] = []
    for line in raw.splitlines():
        line = line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def read_csv_blob(blob) -> pd.DataFrame:
    return pd.read_csv(BytesIO(blob.download_as_bytes()))


# -------------------------------------------------------
# Result container
# -------------------------------------------------------
@dataclass
class CheckResult:
    scope: str
    issues: List[str]

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


# -------------------------------------------------------
# Grid checks
# -------------------------------------------------------
def check_latest_grid_hourly_files() -> CheckResult:
    ingest_cfg = load_ingestion_config()
    prep_cfg = load_preprocessing_config()

    bucket_name = ingest_cfg["gcs"]["bucket"]
    raw_prefix = ingest_cfg["gcs"].get("raw_prefix", "raw").strip("/")
    zones = ingest_cfg["ingestion"]["zones"]
    signals = ingest_cfg["ingestion"]["signals"]
    value_field_overrides = prep_cfg.get("grid", {}).get("value_field_overrides", {})

    null_rate_threshold = float(ingest_cfg.get("monitoring", {}).get("hourly_null_rate_threshold", 0.5))
    min_records = int(ingest_cfg.get("monitoring", {}).get("hourly_min_records", 1))

    client = storage.Client()
    issues: List[str] = []

    base_prefix = f"{raw_prefix}/grid_signals/hourly"

    for zone in zones:
        for signal in signals:
            if signal == "electricity-source":
                continue

            prefix = f"{base_prefix}/zone={zone}/{signal}/"
            blob = latest_blob_for_prefix(client, bucket_name, prefix, ".jsonl")

            if blob is None:
                issues.append(f"[GRID] Missing hourly file for zone={zone}, signal={signal}")
                continue

            try:
                records = read_jsonl_blob(blob)
            except Exception as e:
                issues.append(f"[GRID] Failed reading {blob.name}: {e}")
                continue

            if len(records) < min_records:
                issues.append(f"[GRID] Too few records in {blob.name}: {len(records)}")
                continue

            df = pd.DataFrame(records)

            # datetime is always required
            required_cols = {"datetime"}
            missing = required_cols - set(df.columns)
            if missing:
                issues.append(
                    f"[GRID] Missing required columns in {blob.name}: {sorted(missing)} | available={df.columns.tolist()}"
                )
                continue

            # signal-specific required payload fields
            if signal == "electricity-flows":
                if not ({"import", "export"} & set(df.columns)):
                    issues.append(
                        f"[GRID] electricity-flows missing import/export fields in {blob.name} | available={df.columns.tolist()}"
                    )
                    continue

            elif signal == "electricity-mix":
                if "mix" not in df.columns:
                    issues.append(
                        f"[GRID] electricity-mix missing 'mix' field in {blob.name} | available={df.columns.tolist()}"
                    )
                    continue

            else:
                value_field = value_field_overrides.get(signal, "value")
                if value_field not in df.columns:
                    issues.append(
                        f"[GRID] Missing expected value field '{value_field}' for signal={signal} in {blob.name} | available={df.columns.tolist()}"
                    )
                    continue

            dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            bad_dt = int(dt.isna().sum())
            if bad_dt > 0:
                issues.append(f"[GRID] {bad_dt} bad datetimes in {blob.name}")

            dupes = int(pd.Series(dt).duplicated().sum())
            if dupes > 0:
                issues.append(f"[GRID] {dupes} duplicate datetimes in {blob.name}")

            # Numeric null-rate checks only for scalar signals
            if signal not in {"electricity-mix", "electricity-flows"}:
                value_field = value_field_overrides.get(signal, "value")
                values = pd.to_numeric(df[value_field], errors="coerce")
                null_rate = float(values.isna().mean()) if len(values) else 1.0
                if null_rate > null_rate_threshold:
                    issues.append(
                        f"[GRID] High null/non-numeric rate for zone={zone}, signal={signal}: {null_rate:.1%} in {blob.name}"
                    )

    return CheckResult(scope="grid_hourly", issues=issues)


# -------------------------------------------------------
# Weather checks
# -------------------------------------------------------
def check_latest_weather_hourly_files() -> CheckResult:
    ingest_cfg = load_ingestion_config()

    bucket_name = ingest_cfg["gcs"]["bucket"]
    raw_prefix = ingest_cfg["gcs"].get("raw_prefix", "raw").strip("/")
    locations = ingest_cfg["locations"]
    weather_cfg = ingest_cfg["ingestion_weather"]
    expected_features = weather_cfg["signals"]

    null_rate_threshold = float(ingest_cfg.get("monitoring", {}).get("hourly_null_rate_threshold", 0.5))
    min_rows = int(ingest_cfg.get("monitoring", {}).get("hourly_min_records", 1))

    client = storage.Client()
    issues: List[str] = []

    base_prefix = f"{raw_prefix}/weather/hourly"

    for loc in locations:
        loc_name = loc["name"]
        prefix = f"{base_prefix}/{loc_name}/"
        blob = latest_blob_for_prefix(client, bucket_name, prefix, ".csv")

        if blob is None:
            issues.append(f"[WEATHER] Missing hourly file for location={loc_name}")
            continue

        try:
            df = read_csv_blob(blob)
        except Exception as e:
            issues.append(f"[WEATHER] Failed reading {blob.name}: {e}")
            continue

        if len(df) < min_rows:
            issues.append(f"[WEATHER] Too few rows in {blob.name}: {len(df)}")
            continue

        required = {"time"} | set(expected_features)
        missing = required - set(df.columns)
        if missing:
            issues.append(
                f"[WEATHER] Missing required columns in {blob.name}: {sorted(missing)} | available={df.columns.tolist()}"
            )
            continue

        dt = pd.to_datetime(df["time"], utc=True, errors="coerce")
        bad_dt = int(dt.isna().sum())
        if bad_dt > 0:
            issues.append(f"[WEATHER] {bad_dt} bad timestamps in {blob.name}")

        dupes = int(pd.Series(dt).duplicated().sum())
        if dupes > 0:
            issues.append(f"[WEATHER] {dupes} duplicate timestamps in {blob.name}")

        feature_df = df[expected_features].apply(pd.to_numeric, errors="coerce")
        null_rate = float(feature_df.isna().mean().max()) if not feature_df.empty else 1.0
        if null_rate > null_rate_threshold:
            issues.append(
                f"[WEATHER] High null/non-numeric rate for location={loc_name}: max feature null rate {null_rate:.1%} in {blob.name}"
            )

    return CheckResult(scope="weather_hourly", issues=issues)


# -------------------------------------------------------
# Airflow callables
# -------------------------------------------------------
def run_grid_hourly_anomaly_checks() -> None:
    result = check_latest_grid_hourly_files()
    if not result.ok:
        raise RuntimeError("Hourly grid anomaly checks failed:\n" + "\n".join(result.issues))


def run_weather_hourly_anomaly_checks() -> None:
    result = check_latest_weather_hourly_files()
    if not result.ok:
        raise RuntimeError("Hourly weather anomaly checks failed:\n" + "\n".join(result.issues))