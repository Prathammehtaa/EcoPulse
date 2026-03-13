import os
from pathlib import Path
from io import StringIO
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml
import requests
from google.cloud import storage


# ---------------------------------------------------
# Resolve project structure correctly
# src/ -> project root; data/ sits at project root
# ---------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_config():
    """Reads configuration from ingestion_config.yaml (from project root)."""
    config_path = PROJECT_ROOT / "pipeline_config" / "ingestion_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_iso_z(s: str) -> datetime:
    """Parses ISO date strings, handling the 'Z' suffix for UTC."""
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def upload_csv(bucket, blob_path: str, df: pd.DataFrame) -> None:
    """Uploads a Pandas DataFrame directly to GCS as a CSV file."""
    blob = bucket.blob(blob_path)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")


def save_csv_local(base_dir: Path, relative_path: str, df: pd.DataFrame) -> str:
    """
    Saves a Pandas DataFrame locally as CSV.
    Creates directories automatically if they do not exist.

    base_dir: Path to local root (e.g., PROJECT_ROOT / "data")
    relative_path: mirrors your GCS blob path
    """
    relative_path = relative_path.lstrip("/\\")
    local_path = base_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_path, index=False)
    return str(local_path)


def blob_exists_and_nonempty(bucket, blob_path: str, min_bytes: int = 10) -> bool:
    """Returns True if blob exists and is larger than min_bytes."""
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return False

    blob.reload()
    size = blob.size or 0
    return size >= min_bytes


def get_weather_df(api_response: dict) -> pd.DataFrame | None:
    """
    Transforms API response into a DataFrame using UTC.
    This avoids DST issues.
    """
    hourly = api_response.get("hourly", {})
    if not hourly:
        return None

    df = pd.DataFrame(hourly)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_and_store_chunk(
    *,
    bucket,
    bucket_name: str,
    base_url: str,
    signals_list: str,
    loc: dict,
    start_dt: datetime,
    end_dt: datetime,
    gcs_subdir: str,
    skip_if_exists: bool,
    min_bytes: int = 10,
) -> None:
    """
    Fetch one weather chunk and store locally + GCS.
    Supports idempotency by skipping existing non-empty blobs.
    """
    filename = f"start={start_dt:%Y%m%dT%H%M%SZ}_end={end_dt:%Y%m%dT%H%M%SZ}.csv"
    relative_path = f"{gcs_subdir}/{loc['name']}/{filename}"

    if skip_if_exists and blob_exists_and_nonempty(bucket, relative_path, min_bytes=min_bytes):
        print(f"Skipping existing blob → gs://{bucket_name}/{relative_path}")
        return

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": signals_list,
        "timezone": loc.get("timezone", "UTC"),
    }

    print(f"Fetching {loc['name']}: {params['start_date']} to {params['end_date']}")

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()

    df = get_weather_df(response.json())
    if df is None or df.empty:
        print(f"No data returned for {loc['name']} from {params['start_date']} to {params['end_date']}")
        return

    local_saved_path = save_csv_local(DATA_DIR, relative_path, df)
    print(f"Saved locally → {local_saved_path}")

    upload_csv(bucket, relative_path, df)
    print(f"Uploaded → gs://{bucket_name}/{relative_path}")


def run_backfill(config: dict, bucket, bucket_name: str) -> None:
    """
    Historical / backfill mode.
    Uses configured start, end, and chunk_days.
    """
    om_config = config["openmeteo"]
    ing = config["ingestion_weather"]

    base_url = om_config["base_url"]
    signals_list = ",".join(ing["signals"])
    chunk_days = int(ing.get("chunk_days", 30))
    skip_if_exists = bool(ing.get("skip_if_exists", True))
    min_bytes = int(ing.get("min_bytes", 10))

    for loc in config["locations"]:
        print(f"\n--- Backfill Weather: {loc['name']} ---")

        current = parse_iso_z(ing["start"])
        end_dt = parse_iso_z(ing["end"])

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)

            try:
                fetch_and_store_chunk(
                    bucket=bucket,
                    bucket_name=bucket_name,
                    base_url=base_url,
                    signals_list=signals_list,
                    loc=loc,
                    start_dt=current,
                    end_dt=chunk_end,
                    gcs_subdir="raw/weather/backfill",
                    skip_if_exists=skip_if_exists,
                    min_bytes=min_bytes,
                )
            except Exception as e:
                print(f"Failed backfill chunk for {loc['name']} at {current}: {e}")
                break

            current = chunk_end + timedelta(days=1)


def run_hourly(config: dict, bucket, bucket_name: str) -> None:
    """
    Hourly mode.
    Pulls a strict hourly window for idempotent hourly ingestion.
    """
    om_config = config["openmeteo"]
    ing = config["ingestion_weather"]

    base_url = om_config["base_url"]
    signals_list = ",".join(ing["signals"])
    skip_if_exists = bool(ing.get("skip_if_exists", True))
    min_bytes = int(ing.get("min_bytes", 10))
    lookback_hours = int(ing.get("hourly_lookback_hours", 1))

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = now_utc - timedelta(hours=lookback_hours)
    end_dt = start_dt + timedelta(hours=1)

    for loc in config["locations"]:
        print(f"\n--- Hourly Weather: {loc['name']} ---")
        try:
            fetch_and_store_chunk(
                bucket=bucket,
                bucket_name=bucket_name,
                base_url=base_url,
                signals_list=signals_list,
                loc=loc,
                start_dt=start_dt,
                end_dt=end_dt,
                gcs_subdir="raw/weather/hourly",
                skip_if_exists=skip_if_exists,
                min_bytes=min_bytes,
            )
        except Exception as e:
            print(f"Failed hourly weather ingestion for {loc['name']}: {e}")
            raise


def main(mode=None):
    """
    mode = 'backfill' or 'hourly'
    Falls back to config pipeline.mode if not provided.
    """
    config = load_config()
    bucket_name = config["gcs"]["bucket"]
    mode = (mode or config["pipeline"]["mode"]).strip().lower()

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if mode == "backfill":
        run_backfill(config, bucket, bucket_name)
    elif mode == "hourly":
        run_hourly(config, bucket, bucket_name)
    else:
        raise ValueError("mode must be either 'backfill' or 'hourly'")


if __name__ == "__main__":
    main()