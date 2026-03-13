import json
import yaml
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import storage
import os


# ----------------------------
# Config
# ----------------------------
def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    config_path = os.path.join(project_root, "pipeline_config", "ingestion_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------
# Time helpers
# ----------------------------
def parse_iso_z(s: str) -> datetime:
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


# ----------------------------
# HTTP
# ----------------------------
def fetch_json(base_url: str, endpoint: str, token: str, params: dict) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"auth-token": token.strip()}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


# ----------------------------
# JSONL
# ----------------------------
def _records_to_jsonl(records) -> str:
    if isinstance(records, dict):
        records = [records]
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def upload_jsonl(bucket, blob_path: str, records) -> None:
    blob = bucket.blob(blob_path)
    blob.upload_from_string(_records_to_jsonl(records), content_type="application/x-ndjson")


def write_jsonl_local(project_root: str, blob_path: str, records) -> str:
    data_root = os.path.join(project_root, "data")
    local_path = os.path.join(data_root, *blob_path.split("/"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_records_to_jsonl(records))

    return local_path


# ----------------------------
# Idempotency helpers
# ----------------------------
def gcs_blob_exists_nonempty(bucket, blob_path: str, min_bytes: int = 10) -> bool:
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return False
    blob.reload()
    return (blob.size or 0) >= min_bytes


def local_file_exists_nonempty(project_root: str, blob_path: str, min_bytes: int = 10) -> bool:
    data_root = os.path.join(project_root, "data")
    local_path = os.path.join(data_root, *blob_path.split("/"))
    return os.path.exists(local_path) and os.path.getsize(local_path) >= min_bytes


# ----------------------------
# Chunk planning
# ----------------------------
def iter_chunks_backfill(start_dt: datetime, end_dt: datetime, chunk_days: int):
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        yield current, chunk_end
        current = chunk_end


def iter_chunks_hourly(now_utc: datetime, lookback_hours: int = 2, window_hours: int = 1):
    """
    Example with lookback_hours=2:
      returns [t-2h, t-1h) and [t-1h, t)
    """
    end_hour = floor_to_hour(now_utc)
    start_hour = end_hour - timedelta(hours=lookback_hours)

    current = start_hour
    while current < end_hour:
        chunk_end = current + timedelta(hours=window_hours)
        yield current, chunk_end
        current = chunk_end


# ----------------------------
# Core ingestion
# ----------------------------
def ingest_range(
    *,
    base_url: str,
    token: str,
    bucket,
    bucket_name: str,
    project_root: str,
    endpoint: str,
    params_base: dict,
    blob_path_builder,
    chunks,
    skip_if_exists: bool,
    min_bytes: int = 10,
    prefer_gcs_truth: bool = True,
):
    """
    Generic ingestion runner with idempotency.
    """
    for current, chunk_end in chunks:
        blob_path = blob_path_builder(current, chunk_end)

        if skip_if_exists:
            exists_gcs = gcs_blob_exists_nonempty(bucket, blob_path, min_bytes=min_bytes)
            exists_local = local_file_exists_nonempty(project_root, blob_path, min_bytes=min_bytes)

            if prefer_gcs_truth and exists_gcs:
                print(f"SKIP (exists in GCS) → gs://{bucket_name}/{blob_path}")
                continue

            if (not prefer_gcs_truth) and (exists_gcs or exists_local):
                print(f"SKIP (exists) → gs://{bucket_name}/{blob_path} or local")
                continue

            if exists_local and not exists_gcs:
                print(f"SKIP (exists locally) → data/{blob_path}")
                continue

        print(f"Fetching {current:%Y-%m-%d %H:%M} → {chunk_end:%Y-%m-%d %H:%M}")

        params = dict(params_base)
        params["start"] = iso_z(current)
        params["end"] = iso_z(chunk_end)

        resp = fetch_json(base_url, endpoint, token, params)
        records = resp.get("data", resp)

        local_path = write_jsonl_local(project_root, blob_path, records)
        print(f"Saved local → {local_path}")

        upload_jsonl(bucket, blob_path, records)
        print(f"Uploaded → gs://{bucket_name}/{blob_path}")


# ----------------------------
# Main
# ----------------------------
def main(mode: str | None = None) -> None:
    """
    mode:
      - "backfill": uses ingestion.start/end and chunk_days
      - "hourly": uses now UTC and writes strict 1-hour windows

    Precedence:
      1. explicit function argument (used by Airflow DAG)
      2. config["pipeline"]["mode"]
      3. fallback = "backfill"
    """
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
        "/v3/electricity-source/{source}/past-range",
    )

    skip_if_exists = bool(ing.get("skip_if_exists", True))
    min_bytes = int(ing.get("min_bytes", 10))

    pipeline_mode = config.get("pipeline", {}).get("mode")
    mode = (mode or pipeline_mode or "backfill").strip().lower()

    if mode not in {"backfill", "hourly"}:
        raise ValueError("mode must be 'backfill' or 'hourly'")

    bucket_name = config["gcs"]["bucket"]
    raw_prefix = config["gcs"].get("raw_prefix", "raw").strip("/")

    # Separate raw subdirectories by ingestion mode
    # backfill -> raw/grid_signals/backfill/...
    # hourly   -> raw/grid_signals/hourly/...
    grid_subdir = f"{raw_prefix}/grid_signals/{mode}"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if mode == "backfill":
        start_dt = parse_iso_z(ing["start"])
        end_dt = parse_iso_z(ing["end"])
        chunks = list(iter_chunks_backfill(start_dt, end_dt, chunk_days=chunk_days))
        print(f"\nMODE=BACKFILL | {start_dt.isoformat()} → {end_dt.isoformat()} | chunk_days={chunk_days}\n")
    else:
        now_utc = datetime.now(timezone.utc)
        lookback_hours = int(ing.get("hourly_lookback_hours", 2))
        chunks = list(iter_chunks_hourly(now_utc, lookback_hours=lookback_hours, window_hours=1))
        print(f"\nMODE=HOURLY | now={now_utc.isoformat()} | lookback_hours={lookback_hours} | window=1h\n")

    for zone in zones:
        for signal in signals:
            if signal == "electricity-source":
                print("Skipping electricity-source here (handled in electricity_sources loop).")
                continue

            print(f"\n=== Zone: {zone} | Signal: {signal} | Mode: {mode} ===")
            endpoint = endpoint_template.format(signal=signal)

            params_base = dict(static_params)
            params_base["zone"] = zone

            def blob_builder(cur: datetime, end: datetime, zone=zone, signal=signal) -> str:
                return (
                    f"{grid_subdir}/"
                    f"zone={zone}/"
                    f"{signal}/"
                    f"start={cur:%Y%m%dT%H%M%SZ}_end={end:%Y%m%dT%H%M%SZ}.jsonl"
                )

            ingest_range(
                base_url=base_url,
                token=token,
                bucket=bucket,
                bucket_name=bucket_name,
                project_root=project_root,
                endpoint=endpoint,
                params_base=params_base,
                blob_path_builder=blob_builder,
                chunks=chunks,
                skip_if_exists=skip_if_exists,
                min_bytes=min_bytes,
                prefer_gcs_truth=True,
            )

        for source in electricity_sources:
            print(f"\n=== Zone: {zone} | Signal: electricity-source | Source: {source} | Mode: {mode} ===")
            endpoint = electricity_source_template.format(source=source)

            params_base = dict(static_params)
            params_base["zone"] = zone

            def blob_builder(cur: datetime, end: datetime, zone=zone, source=source) -> str:
                return (
                    f"{grid_subdir}/"
                    f"zone={zone}/"
                    f"electricity-source/"
                    f"source={source}/"
                    f"start={cur:%Y%m%dT%H%M%SZ}_end={end:%Y%m%dT%H%M%SZ}.jsonl"
                )

            ingest_range(
                base_url=base_url,
                token=token,
                bucket=bucket,
                bucket_name=bucket_name,
                project_root=project_root,
                endpoint=endpoint,
                params_base=params_base,
                blob_path_builder=blob_builder,
                chunks=chunks,
                skip_if_exists=skip_if_exists,
                min_bytes=min_bytes,
                prefer_gcs_truth=True,
            )

    print("\nRaw ingestion complete for all zones and signals.")


if __name__ == "__main__":
    main()