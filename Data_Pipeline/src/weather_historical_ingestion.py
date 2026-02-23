import pandas as pd
import os
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage
from io import StringIO

def load_config():
    """Reads configuration from ingestion_config.yaml."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "pipeline config", "ingestion_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_iso_z(s):
    """Parses ISO date strings, handling the 'Z' suffix for UTC."""
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def upload_csv(bucket, blob_path, df):
    """Uploads a Pandas DataFrame directly to GCS as a CSV file."""
    blob = bucket.blob(blob_path)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

def get_weather_df(api_response):
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
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
    
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
                "timezone": loc["timezone"]
            }

            try:
                print(f"Fetching {loc['name']}: {params['start_date']} to {params['end_date']}")
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()
                
                # Transform using the new UTC-safe logic
                df = get_weather_df(response.json())
                
                if df is not None:
                    blob_path = f"weather_raw/{loc['name']}/start={params['start_date']}_end={params['end_date']}.csv"
                    upload_csv(bucket, blob_path, df)
                    print(f"Uploaded → gs://{bucket_name}/{blob_path}")

            except Exception as e:
                print(f"Failed {loc['name']} chunk at {current}: {e}")
                break

            current = chunk_end + timedelta(days=1)

if __name__ == "__main__":
    main()