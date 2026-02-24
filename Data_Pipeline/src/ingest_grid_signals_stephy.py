"""
EcoPulse Grid Signals Ingestion (Stephy's Zones)
=================================================
Fetches grid signals for US-NW-PACW and US-MIDA-PJM zones.

Usage:
    python ingest_grid_signals_stephy.py
"""

import json
import os
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config",
                           "ingestion_config_stephy.yaml")


# ----------------------------
# Helper Functions
# ----------------------------
def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_iso_z(s):
    """Parse ISO datetime string with Z suffix."""
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def iso_z(dt):
    """Format datetime as ISO string with Z suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(base_url, endpoint, token, params):
    """Fetch JSON from API."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"auth-token": token.strip()}

    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def upload_jsonl(bucket, blob_path, records):
    """Upload records as JSONL to GCS."""
    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records) + "\n"
    blob.upload_from_string(jsonl_content, content_type="application/x-ndjson")


# ----------------------------
# Main Ingestion
# ----------------------------
def main():
    print("=" * 60)
    print("EcoPulse Grid Signals Ingestion (Stephy's Zones)")
    print("=" * 60)

    config = load_config()

    base_url = config["electricitymaps"]["base_url"]
    token = config["electricitymaps"]["token"]

    ing = config["ingestion"]
    signals = ing["signals"]
    zones = ing["zones"]
    endpoint_template = ing["endpoint_template"]
    static_params = ing.get("static_params", {})
    chunk_days = int(ing.get("chunk_days", 10))

    electricity_sources = ing.get("electricity_sources", [])
    electricity_source_template = ing.get(
        "electricity_source_template",
        "/v3/electricity-source/{source}/past-range"
    )

    start_dt = parse_iso_z(ing["start"])
    end_dt = parse_iso_z(ing["end"])

    bucket_name = config["gcs"]["bucket"]
    raw_prefix = config["gcs"].get("raw_prefix", "raw").strip("/")

    print(f"\nBucket: {bucket_name}")
    print(f"Zones: {zones}")
    print(f"Signals: {signals}")
    print(f"Date range: {start_dt} to {end_dt}")
    print(f"Electricity sources: {electricity_sources}")

    # Connect to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    total_uploaded = 0

    # Process each zone
    for zone in zones:
        print(f"\n{'='*60}")
        print(f"ZONE: {zone}")
        print(f"{'='*60}")

        # Process regular signals
        for signal in signals:
            print(f"\n--- Signal: {signal} ---")
            endpoint = endpoint_template.format(signal=signal)
            current = start_dt

            while current < end_dt:
                chunk_end = min(current + timedelta(days=chunk_days), end_dt)

                params = dict(static_params)
                params["zone"] = zone
                params["start"] = iso_z(current)
                params["end"] = iso_z(chunk_end)

                try:
                    print(
                        f"  Fetching {current:%Y-%m-%d} → {chunk_end:%Y-%m-%d}...", end=" ")
                    response = fetch_json(base_url, endpoint, token, params)
                    records = response.get("data", response)

                    if records:
                        blob_path = (
                            f"{raw_prefix}/{signal}/"
                            f"zone={zone}/"
                            f"start={current:%Y%m%dT%H%M%SZ}_end={chunk_end:%Y%m%dT%H%M%SZ}.jsonl"
                        )
                        upload_jsonl(bucket, blob_path, records)
                        print(f"✅ {len(records)} records")
                        total_uploaded += 1
                    else:
                        print("⚠️ No data")

                except requests.HTTPError as e:
                    print(f"❌ API error: {e.response.status_code}")
                    if e.response.status_code == 429:
                        print("    Rate limited! Waiting 60 seconds...")
                        import time
                        time.sleep(60)
                        continue
                except Exception as e:
                    print(f"❌ Error: {e}")

                current = chunk_end

        # Process electricity sources
        for source in electricity_sources:
            print(f"\n--- Electricity Source: {source} ---")
            endpoint = electricity_source_template.format(source=source)
            current = start_dt

            while current < end_dt:
                chunk_end = min(current + timedelta(days=chunk_days), end_dt)

                params = dict(static_params)
                params["zone"] = zone
                params["start"] = iso_z(current)
                params["end"] = iso_z(chunk_end)

                try:
                    print(
                        f"  Fetching {current:%Y-%m-%d} → {chunk_end:%Y-%m-%d}...", end=" ")
                    response = fetch_json(base_url, endpoint, token, params)
                    records = response.get("data", response)

                    if records:
                        blob_path = (
                            f"{raw_prefix}/electricity-source/"
                            f"zone={zone}/"
                            f"source={source}/"
                            f"start={current:%Y%m%dT%H%M%SZ}_end={chunk_end:%Y%m%dT%H%M%SZ}.jsonl"
                        )
                        upload_jsonl(bucket, blob_path, records)
                        print(f"✅ {len(records)} records")
                        total_uploaded += 1
                    else:
                        print("⚠️ No data")

                except requests.HTTPError as e:
                    print(f"❌ API error: {e.response.status_code}")
                except Exception as e:
                    print(f"❌ Error: {e}")

                current = chunk_end

    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete! {total_uploaded} files uploaded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
