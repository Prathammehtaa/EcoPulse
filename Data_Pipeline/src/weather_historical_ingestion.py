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
    config_path = PROJECT_ROOT / "config" / "ingestion_config.yaml"
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
    relative_path: mirrors your GCS blob path (e.g., weather_raw/Boston/...)
    """
    relative_path = relative_path.lstrip("/\\")
    local_path = base_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_path, index=False)
    return str(local_path)


def get_weather_df(api_response: dict) -> pd.DataFrame | None:
    """
    Transforms API response into a DataFrame using UTC.
    This avoids all Daylight Saving Time (DST) errors permanently.
    """
    hourly = api_response.get("hourly", {})
    if not hourly:
        return None

    df = pd.DataFrame(hourly)

    # Treat the API string as UTC immediately.
    # This creates a linear 24-hour timeline that never breaks in March or Nov.
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")

    return df


def main():
    config = load_config()
    om_config = config["openmeteo"]
    ing = config["ingestion_weather"]
    bucket_name = config["gcs"]["bucket"]

    # Initialize Google Cloud Storage Client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    base_url = om_config["base_url"]
    signals_list = ",".join(ing["signals"])

    for loc in config["locations"]:
        print(f"\n--- Processing Location: {loc['name']} (UTC-Safe Mode) ---")
        current = parse_iso_z(ing["start"])
        end_dt = parse_iso_z(ing["end"])

        while current < end_dt:
            chunk_end = min(current + timedelta(days=ing["chunk_days"]), end_dt)

            # Note: We send the location's timezone to the API so it handles
            # the day-boundaries correctly, but we parse the result as UTC.
            params = {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "start_date": current.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": signals_list,
                "timezone": loc["timezone"],
            }

            try:
                print(f"Fetching {loc['name']}: {params['start_date']} to {params['end_date']}")
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()

                # Transform using the UTC-safe logic
                df = get_weather_df(response.json())

                if df is not None:
                    blob_path = (
                        f"raw/weather/{loc['name']}/"
                        f"start={params['start_date']}_end={params['end_date']}.csv"
                    )

                    # ✅ Save locally to <project_root>/data/<blob_path>
                    local_saved_path = save_csv_local(DATA_DIR, blob_path, df)
                    print(f"Saved locally → {local_saved_path}")

                    # ✅ Upload to GCS
                    upload_csv(bucket, blob_path, df)
                    print(f"Uploaded → gs://{bucket_name}/{blob_path}")

            except Exception as e:
                print(f"Failed {loc['name']} chunk at {current}: {e}")
                break

            current = chunk_end + timedelta(days=1)


if __name__ == "__main__":
    main()