"""
EcoPulse Weather Data Ingestion (Clyde's Locations)
====================================================
Fetches weather data for Northern Virginia and Portland Oregon.

Usage:
    python ingest_weather_clyde.py
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage
from io import StringIO

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Your bucket
BUCKET_NAME = "ecopulse-kapish"

# Open-Meteo API
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Locations (matching grid zones)
LOCATIONS = [
    {
        "name": "northern_virginia",
        "latitude": 39.0465,
        "longitude": -77.2997,
        "timezone": "America/New_York",
        "grid_zone": "US-MIDA-PJM"
    },
    {
        "name": "portland_oregon",
        "latitude": 45.5234,
        "longitude": -122.6762,
        "timezone": "America/Los_Angeles",
        "grid_zone": "US-NW-PACW"
    },
    {
        "name": "boston_area",
        "latitude": 42.3601,
        "longitude": -71.0589,
        "timezone": "America/New_York",
        "grid_zone": "US-NE-ISNE"
    }
]

# Weather signals to fetch
WEATHER_SIGNALS = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_speed_100m",
    "cloud_cover",
    "weather_code",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "shortwave_radiation",
    "direct_radiation"
]

# Date range (matching your grid data)
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
CHUNK_DAYS = 30


# ----------------------------
# Helper Functions
# ----------------------------
def parse_iso_z(s):
    """Parse ISO datetime string."""
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def upload_csv(bucket, blob_path, df):
    """Upload DataFrame as CSV to GCS."""
    blob = bucket.blob(blob_path)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")


def get_weather_df(api_response):
    """Transform API response into DataFrame."""
    hourly = api_response.get("hourly", {})
    if not hourly:
        return None

    df = pd.DataFrame(hourly)

    # Parse timestamps as UTC
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
    df.rename(columns={'time': 'timestamp_utc'}, inplace=True)

    return df


# ----------------------------
# Main Ingestion
# ----------------------------
def main():
    print("=" * 60)
    print("EcoPulse Weather Data Ingestion (Clyde's Locations)")
    print("=" * 60)

    print(f"\nBucket: {BUCKET_NAME}")
    print(f"Locations: {[loc['name'] for loc in LOCATIONS]}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Weather signals: {len(WEATHER_SIGNALS)}")

    # Connect to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    signals_str = ",".join(WEATHER_SIGNALS)
    total_uploaded = 0

    for loc in LOCATIONS:
        print(f"\n{'='*60}")
        print(f"LOCATION: {loc['name']} ({loc['grid_zone']})")
        print(f"{'='*60}")

        current = datetime.strptime(START_DATE, "%Y-%m-%d")
        end = datetime.strptime(END_DATE, "%Y-%m-%d")

        while current < end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)

            params = {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "start_date": current.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": signals_str,
                "timezone": loc["timezone"]
            }

            try:
                print(
                    f"  Fetching {params['start_date']} → {params['end_date']}...", end=" ")

                response = requests.get(BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                df = get_weather_df(response.json())

                if df is not None and not df.empty:
                    # Add location metadata
                    df['location'] = loc['name']
                    df['grid_zone'] = loc['grid_zone']

                    blob_path = (
                        f"weather_raw/{loc['name']}/"
                        f"start={params['start_date']}_end={params['end_date']}.csv"
                    )
                    upload_csv(bucket, blob_path, df)
                    print(f"✅ {len(df)} records")
                    total_uploaded += 1
                else:
                    print("⚠️ No data")

            except requests.HTTPError as e:
                print(f"❌ API error: {e.response.status_code}")
            except Exception as e:
                print(f"❌ Error: {e}")

            current = chunk_end + timedelta(days=1)

    print("\n" + "=" * 60)
    print(f"✅ Weather ingestion complete! {total_uploaded} files uploaded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
