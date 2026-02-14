import json
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage
import os


def load_config():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))          # .../src
    project_root = os.path.dirname(base_dir)                       # .../
    config_path = os.path.join(project_root, "config", "ingestion_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_iso_z(s: str) -> datetime:
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(base_url: str, endpoint: str, token: str, params: dict) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"auth-token": token.strip()}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def upload_jsonl(bucket, blob_path: str, records: list) -> None:
    blob = bucket.blob(blob_path)
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    blob.upload_from_string(content, content_type="application/x-ndjson")


def main():
    config = load_config()

    base_url = config["electricitymaps"]["base_url"]
    token = config["electricitymaps"]["token"]

    ing = config["ingestion"]
    signals = ing["signals"]                    # list
    zones = ing["zones"]                        # list
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

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    
    for zone in zones:

        for signal in signals:
            if signal == "electricity-source":
                print("Skipping electricity-source here (handled in electricity_sources loop).")
                continue
            print(f"\n=== Zone: {zone} | Signal: {signal} ===")

            endpoint = endpoint_template.format(signal=signal)
            current = start_dt

            while current < end_dt:
                chunk_end = min(current + timedelta(days=chunk_days), end_dt)
                print(f"Fetching {current:%Y-%m-%d} → {chunk_end:%Y-%m-%d}")

                params = dict(static_params)
                params["zone"] = zone
                params["start"] = iso_z(current)
                params["end"] = iso_z(chunk_end)

                resp = fetch_json(base_url, endpoint, token, params)
                records = resp.get("data", resp)

                blob_path = (
                    f"{raw_prefix}/grid_signals/"
                    f"zone={zone}/"
                    f"{signal}/"
                    f"start={current:%Y%m%dT%H%M%SZ}_end={chunk_end:%Y%m%dT%H%M%SZ}.jsonl"
                )

                upload_jsonl(bucket, blob_path, records)
                print(f"Uploaded → gs://{bucket_name}/{blob_path}")

                current = chunk_end


        for source in electricity_sources:
            print(f"\n=== Zone: {zone} | Signal: electricity-source | Source: {source} ===")

            endpoint = electricity_source_template.format(source=source)
            current = start_dt

            while current < end_dt:
                chunk_end = min(current + timedelta(days=chunk_days), end_dt)
                print(f"Fetching {current:%Y-%m-%d} → {chunk_end:%Y-%m-%d}")

                params = dict(static_params)
                params["zone"] = zone
                params["start"] = iso_z(current)
                params["end"] = iso_z(chunk_end)

                resp = fetch_json(base_url, endpoint, token, params)
                records = resp.get("data", resp)

                blob_path = (
                    f"{raw_prefix}/grid_signals/"
                    f"zone={zone}/"
                    f"electricity-source/"
                    f"source={source}/"
                    f"start={current:%Y%m%dT%H%M%SZ}_end={chunk_end:%Y%m%dT%H%M%SZ}.jsonl"
                )

                upload_jsonl(bucket, blob_path, records)
                print(f"Uploaded → gs://{bucket_name}/{blob_path}")

                current = chunk_end

    print("\n Raw ingestion complete for all zones and signals.")


if __name__ == "__main__":
    main()
