import json
import os
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage


# ----------------------------
# Helpers
# ----------------------------

def load_config():
    # Get directory where this file lives (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go one level up (to Data_Pipeline/)
    project_root = os.path.dirname(current_dir)

    # Build safe path
    config_path = os.path.join(project_root, "config", "ingestion_config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_iso_z(s):
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(base_url, endpoint, token, params):
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"auth-token": token}
    print("Token present:", bool(token), "Token length:", len(token) if token else 0)

    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def upload_jsonl(bucket, blob_path, records):
    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records) + "\n"
    blob.upload_from_string(
        jsonl_content,
        content_type="application/x-ndjson"
    )


# ----------------------------
# Main Ingestion
# ----------------------------

def main():

    config = load_config()

    base_url = config["electricitymaps"]["base_url"]
    token = config["electricitymaps"]["token"].strip()

    ing = config["ingestion"]
    signals = ing["signals"]  # ← array of signals
    endpoint_template = ing["endpoint_template"]
    static_params = ing["static_params"]
    chunk_days = ing["chunk_days"]

    start_dt = parse_iso_z(ing["start"])
    end_dt = parse_iso_z(ing["end"])

    bucket_name = config["gcs"]["bucket"]
    raw_prefix = config["gcs"]["raw_prefix"]
    zone = static_params["zone"]

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for signal in signals:

        print(f"\n============================")
        print(f"Processing signal: {signal}")
        print(f"============================")

        endpoint = endpoint_template.format(signal=signal)

        current = start_dt

        while current < end_dt:

            chunk_end = min(current + timedelta(days=chunk_days), end_dt)

            print(f"Fetching {current:%Y-%m-%d} → {chunk_end:%Y-%m-%d}")

            params = dict(static_params)
            params["start"] = iso_z(current)
            params["end"] = iso_z(chunk_end)

            try:
                response = fetch_json(base_url, endpoint, token, params)
            except requests.HTTPError as e:
                print(f"API error for {signal}: {e.response.status_code}")
                print(e.response.text[:500])
                break

            records = response.get("data", response)

            blob_path = (
                f"{raw_prefix}/{signal}/"
                f"zone={zone}/"
                f"start={current:%Y%m%dT%H%M%SZ}_end={chunk_end:%Y%m%dT%H%M%SZ}.jsonl"
            )

            upload_jsonl(bucket, blob_path, records)

            print(f"Uploaded → gs://{bucket_name}/{blob_path}")

            current = chunk_end

    print("\n All signals ingestion complete.")



if __name__ == "__main__":
    main()
