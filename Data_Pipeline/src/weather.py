import json
import os
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage

def load_config():
    # This matches your existing logic to find the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "config", "ingestion_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_iso_z(s):
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def upload_jsonl(bucket, blob_path, records):
    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records) + "\n"
    blob.upload_from_string(jsonl_content, content_type="application/x-ndjson")

def transform_to_jsonl(api_response):
    """Converts Open-Meteo's columnar lists into row-based JSONL records."""
    hourly = api_response.get("hourly", {})
    times = hourly.get("time", [])
    variables = [k for k in hourly.keys() if k != "time"]
    
    records = []
    for i, timestamp in enumerate(times):
        record = {"time": timestamp}
        for var in variables:
            record[var] = hourly[var][i]
        records.append(record)
    return records

def main():
    config = load_config()
    
    # Extract weather-specific settings
    om_config = config["openmeteo"]
    ing = config["ingestion_weather"]
    
    # Setup GCS (reusing your existing GCS bucket setting)
    bucket_name = config["gcs"]["bucket"]
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # API Parameters
    base_url = om_config["base_url"]
    signals_list = ",".join(ing["signals"])
    current = parse_iso_z(ing["start"])
    end_dt = parse_iso_z(ing["end"])

    while current < end_dt:
        chunk_end = min(current + timedelta(days=ing["chunk_days"]), end_dt)
        
        params = dict(om_config["static_params"])
        params["start_date"] = current.strftime("%Y-%m-%d")
        params["end_date"] = chunk_end.strftime("%Y-%m-%d")
        params["hourly"] = signals_list

        try:
            print(f"Fetching weather: {params['start_date']} to {params['end_date']}")
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            records = transform_to_jsonl(response.json())
            
            # Save to a dedicated weather folder in your bucket
            blob_path = f"weather_raw/start={params['start_date']}_end={params['end_date']}.jsonl"
            upload_jsonl(bucket, blob_path, records)
            print(f"Uploaded → gs://{bucket_name}/{blob_path}")

        except Exception as e:
            print(f"Failed to fetch weather: {e}")
            break

        current = chunk_end + timedelta(days=1)

if __name__ == "__main__":
    main()