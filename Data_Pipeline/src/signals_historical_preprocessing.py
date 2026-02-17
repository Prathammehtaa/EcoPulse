import os
import re
import json
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage


# ----------------------------
# SETTINGS
# ----------------------------
BUCKET_NAME = "ecopulse"
RAW_PREFIX = "raw"  # raw in GCS
GCS_PREFIX = f"{RAW_PREFIX}/grid_signals/"  # raw/grid_signals/

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
GCS_STAGE_PREFIX = "stage"   # writes to gs://<bucket>/stage/...
UPLOAD_STAGE_TO_GCS = True   # flip to False if you want local-only

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "stage")


START_UTC = "2023-01-01T00:00:00Z"
END_UTC   = "2026-01-01T00:00:00Z"


# ----------------------------
# Helpers: time + naming
# ----------------------------


def to_utc_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)

def sanitize_key(k: str) -> str:
    k = str(k).strip().lower()
    k = re.sub(r"[^\w]+", "_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k


# ----------------------------
# GCS path parsing
# Your raw structure:
# raw/grid_signals/zone=<zone>/<signal>/start=..._end=....jsonl
# raw/grid_signals/zone=<zone>/electricity-source/source=<source>/start=..._end=....jsonl
# ----------------------------
ZONE_RE = re.compile(r"/zone=([^/]+)/")
RANGE_RE = re.compile(r"start=(\d{8}T\d{6}Z)_end=(\d{8}T\d{6}Z)\.jsonl$")

def parse_zone_from_blob(blob_name: str) -> str:
    m = ZONE_RE.search(blob_name)
    if not m:
        raise ValueError(f"Cannot parse zone from blob: {blob_name}")
    return m.group(1)

def parse_signal_and_source(blob_name: str) -> tuple[str, str | None]:
    parts = blob_name.split("/")
    zi = next(i for i, p in enumerate(parts) if p.startswith("zone="))
    signal = parts[zi + 1]
    source = None
    if signal == "electricity-source":
        source_part = parts[zi + 2]
        if source_part.startswith("source="):
            source = source_part.split("=", 1)[1]
    return signal, source

def parse_range_from_blob(blob_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    m = RANGE_RE.search(blob_name)
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    start = pd.to_datetime(start_s, format="%Y%m%dT%H%M%SZ", utc=True)
    end = pd.to_datetime(end_s, format="%Y%m%dT%H%M%SZ", utc=True)
    return start, end

def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return (a_start < b_end) and (b_start < a_end)


# ----------------------------
# Read JSONL from GCS
# ----------------------------
def download_jsonl_records(bucket: storage.Bucket, blob_name: str) -> list[dict]:
    text = bucket.blob(blob_name).download_as_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ----------------------------
# Normalization (raw -> joinable)
# ----------------------------
SCALAR_SIGNAL_TO_COL = {
    "carbon-free-energy": "carbon_free_energy_pct",
    "renewable-energy": "renewable_energy_pct",
    "net-load": "net_load_mw",
    "total-load": "total_load_mw",
    "total-reported-load": "total_reported_load_mw",
}

CARBON_INTENSITY_COL = "carbon_intensity_gco2_per_kwh"
CARBON_INTENSITY_FOSSIL_COL = "carbon_intensity_fossil_gco2_per_kwh"

FLOW_TOTAL_IMPORT_COL = "total_import_mw"
FLOW_TOTAL_EXPORT_COL = "total_export_mw"
FLOW_NET_FLOW_COL = "net_flow_mw"

def normalize_records(zone: str, signal: str, source: str | None, records: list[dict]) -> pd.DataFrame:
    """
    Output always: zone, timestamp_utc, <signal cols>
    """
    if not records:
        return pd.DataFrame(columns=["zone", "timestamp_utc"])

    if signal == "carbon-intensity":
        df = pd.DataFrame(records)
        df["zone"] = zone
        df["timestamp_utc"] = to_utc_ts(df["datetime"])
        df[CARBON_INTENSITY_COL] = df["carbonIntensity"]
        return df[["zone", "timestamp_utc", CARBON_INTENSITY_COL]]

    if signal == "carbon-intensity-fossil-only":
        df = pd.DataFrame(records)
        df["zone"] = zone
        df["timestamp_utc"] = to_utc_ts(df["datetime"])
        df[CARBON_INTENSITY_FOSSIL_COL] = df["value"]
        return df[["zone", "timestamp_utc", CARBON_INTENSITY_FOSSIL_COL]]

    if signal == "electricity-mix":
        rows = []
        for r in records:
            ts = to_utc_ts(r["datetime"])
            mix = r.get("mix", {}) or {}
            row = {"zone": zone, "timestamp_utc": ts}
            for k, v in mix.items():
                row[f"mix_{sanitize_key(k)}_mw"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    if signal == "electricity-flows":
        rows = []
        for r in records:
            ts = to_utc_ts(r["datetime"])
            imp = r.get("import", {}) or {}
            exp = r.get("export", {}) or {}
            total_imp = float(sum(imp.values())) if imp else 0.0
            total_exp = float(sum(exp.values())) if exp else 0.0
            rows.append({
                "zone": zone,
                "timestamp_utc": ts,
                FLOW_TOTAL_IMPORT_COL: total_imp,
                FLOW_TOTAL_EXPORT_COL: total_exp,
                FLOW_NET_FLOW_COL: total_imp - total_exp,
            })
        return pd.DataFrame(rows)

    if signal == "electricity-source":
        if not source:
            raise ValueError("electricity-source requires source=<...> in path")
        df = pd.DataFrame(records)
        df["zone"] = zone
        df["timestamp_utc"] = to_utc_ts(df["datetime"])
        col = f"source_{sanitize_key(source)}_mw"
        df[col] = df["value"]
        return df[["zone", "timestamp_utc", col]]

    if signal in SCALAR_SIGNAL_TO_COL:
        df = pd.DataFrame(records)
        df["zone"] = zone
        df["timestamp_utc"] = to_utc_ts(df["datetime"])
        col = SCALAR_SIGNAL_TO_COL[signal]
        df[col] = df["value"]
        return df[["zone", "timestamp_utc", col]]

    # skip unmapped signals safely
    return pd.DataFrame(columns=["zone", "timestamp_utc"])


# ----------------------------
# Join + output
# ----------------------------
def hourly_spine(zone: str, start_utc: str, end_utc: str) -> pd.DataFrame:
    start = pd.to_datetime(start_utc, utc=True)
    end = pd.to_datetime(end_utc, utc=True)
    ts = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    return pd.DataFrame({"zone": zone, "timestamp_utc": ts})

def join_all(zone: str, dfs: list[pd.DataFrame], start_utc: str, end_utc: str) -> pd.DataFrame:
    out = hourly_spine(zone, start_utc, end_utc)

    # Group dfs by the set of value-columns they contain (so chunks of the same signal get combined)
    grouped: dict[tuple[str, ...], list[pd.DataFrame]] = {}

    for df in dfs:
        if df is None or df.empty:
            continue

        df = df.drop_duplicates(subset=["zone", "timestamp_utc"])

        value_cols = tuple(sorted([c for c in df.columns if c not in ("zone", "timestamp_utc")]))
        if not value_cols:
            continue

        grouped.setdefault(value_cols, []).append(df[["zone", "timestamp_utc", *value_cols]])

    # For each group, concat chunks then merge once
    for value_cols, chunk_dfs in grouped.items():
        combined = pd.concat(chunk_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["zone", "timestamp_utc"], keep="last")

        # Merge with safe coalesce if columns already exist
        overlap = [c for c in value_cols if c in out.columns]
        if not overlap:
            out = out.merge(combined, on=["zone", "timestamp_utc"], how="left")
        else:
            out = out.merge(combined, on=["zone", "timestamp_utc"], how="left", suffixes=("", "_new"))
            for c in overlap:
                # keep existing if not null, otherwise take new
                out[c] = out[c].combine_first(out[f"{c}_new"])
                out.drop(columns=[f"{c}_new"], inplace=True)

            # for any non-overlap cols that got added, nothing to do

    return out.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)

def upload_file_to_gcs(bucket: storage.Bucket, local_path: str, gcs_path: str) -> None:
    """
    Upload a local file to GCS at gcs_path.
    """
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded stage → gs://{bucket.name}/{gcs_path}")


def local_to_gcs_path(local_path: str, out_dir: str, gcs_stage_prefix: str) -> str:
    """
    Convert a local path under OUT_DIR into a GCS path under GCS_STAGE_PREFIX.

    Example:
      local: <OUT_DIR>/grid_signals_hourly_base/zone=.../year=.../month=.../data.parquet
      gcs:   stage/grid_signals_hourly_base/zone=.../year=.../month=.../data.parquet
    """
    rel = os.path.relpath(local_path, start=out_dir)          # relative path from OUT_DIR
    rel_posix = rel.replace(os.sep, "/")                      # make it GCS-friendly
    return f"{gcs_stage_prefix.rstrip('/')}/{rel_posix}"

def write_monthly_parquet(
    df: pd.DataFrame,
    out_dir: str,
    zone: str,
    bucket: storage.Bucket | None = None,
    upload_stage: bool = False,
    gcs_stage_prefix: str = "stage",
) -> None:
    df = df.copy()
    df["year"] = df["timestamp_utc"].dt.year
    df["month"] = df["timestamp_utc"].dt.month

    for (y, m), part in df.groupby(["year", "month"]):
        local_path = os.path.join(
            out_dir,
            "grid_signals",
            f"zone={zone}",
            f"year={y}",
            f"month={m:02d}",
            "data.parquet",
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        part.drop(columns=["year", "month"]).to_parquet(
            local_path, engine="pyarrow", index=False, compression="snappy"
        )
        print(f"Wrote {len(part)} rows -> {local_path}")

        if upload_stage:
            if bucket is None:
                raise ValueError("upload_stage=True requires a GCS bucket.")
            gcs_path = local_to_gcs_path(local_path, out_dir, gcs_stage_prefix)
            upload_file_to_gcs(bucket, local_path, gcs_path)

def main():
    start = pd.to_datetime(START_UTC, utc=True)
    end = pd.to_datetime(END_UTC, utc=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # List all raw blobs
    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=GCS_PREFIX))
    blob_names = [b.name for b in blobs if b.name.endswith(".jsonl")]

    # Group by zone
    zone_to_blobs: dict[str, list[str]] = {}
    for name in blob_names:
        zone = parse_zone_from_blob(name)
        zone_to_blobs.setdefault(zone, []).append(name)

    for zone, names in zone_to_blobs.items():
        print(f"\n============================")
        print(f"ZONE: {zone}")
        print(f"============================")

        normalized = []

        for name in names:
            sig, src = parse_signal_and_source(name)

            # Only process blobs that overlap the requested time range (based on filename)
            rng = parse_range_from_blob(name)
            if rng:
                b_start, b_end = rng
                if not overlaps(b_start, b_end, start, end):
                    continue

            records = download_jsonl_records(bucket, name)
            df = normalize_records(zone=zone, signal=sig, source=src, records=records)

            if not df.empty:
                df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)]
                normalized.append(df)

        joined = join_all(zone, normalized, START_UTC, END_UTC)
        write_monthly_parquet(joined,OUT_DIR,zone,bucket=bucket,upload_stage=UPLOAD_STAGE_TO_GCS,gcs_stage_prefix=GCS_STAGE_PREFIX,)


    print("\nDone. Raw read from GCS, final parquet saved locally.")


if __name__ == "__main__":
    main()
