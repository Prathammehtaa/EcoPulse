"""
EcoPulse Complete Data Staging Script
======================================
Stages ALL data from GCS bucket:
- Grid signals for all zones (US-NE-ISNE, US-NW-PACW, US-MIDA-PJM)
- Weather data for all locations

Usage:
    python stage_all_data.py
"""

import os
import json
import re
import pandas as pd
from datetime import datetime, timezone
from google.cloud import storage

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "stage")

BUCKET_NAME = "ecopulse-kapish"
RAW_PREFIX = "raw"
WEATHER_PREFIX = "weather_raw"

# Date range
START_DATE = "2024-01-01T00:00:00Z"
END_DATE = "2025-01-01T00:00:00Z"


# ----------------------------
# Helper Functions
# ----------------------------
def download_jsonl(bucket, blob_name):
    """Download and parse JSONL file from GCS."""
    blob = bucket.blob(blob_name)
    content = blob.download_as_text(encoding="utf-8")
    records = []
    for line in content.strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return records


def download_csv(bucket, blob_name):
    """Download CSV file from GCS."""
    blob = bucket.blob(blob_name)
    content = blob.download_as_text(encoding="utf-8")
    from io import StringIO
    return pd.read_csv(StringIO(content))


def parse_zone_from_path(blob_name):
    """Extract zone from blob path."""
    match = re.search(r'zone=([^/]+)', blob_name)
    return match.group(1) if match else None


def parse_signal_from_path(blob_name):
    """Extract signal from blob path."""
    parts = blob_name.split("/")
    if len(parts) >= 2:
        return parts[1]
    return None


def parse_source_from_path(blob_name):
    """Extract electricity source from blob path."""
    match = re.search(r'source=([^/]+)', blob_name)
    return match.group(1) if match else None


def parse_location_from_path(blob_name):
    """Extract location from weather blob path."""
    parts = blob_name.split("/")
    if len(parts) >= 2:
        return parts[1]
    return None


# ----------------------------
# Normalizers for Grid Signals
# ----------------------------
def normalize_carbon_intensity(records, zone):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["zone"] = zone
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["carbon_intensity_gco2_per_kwh"] = df["carbonIntensity"]
    return df[["zone", "timestamp_utc", "carbon_intensity_gco2_per_kwh"]]


def normalize_scalar_signal(records, zone, col_name):
    """Generic normalizer for signals with 'value' field."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["zone"] = zone
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df[col_name] = df["value"]
    return df[["zone", "timestamp_utc", col_name]]


def normalize_electricity_flows(records, zone):
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        ts = pd.to_datetime(r["datetime"], utc=True)
        imports = r.get("import", {}) or {}
        exports = r.get("export", {}) or {}
        total_import = sum(imports.values()) if imports else 0
        total_export = sum(exports.values()) if exports else 0
        rows.append({
            "zone": zone,
            "timestamp_utc": ts,
            "total_import_mw": total_import,
            "total_export_mw": total_export,
            "net_flow_mw": total_import - total_export
        })
    return pd.DataFrame(rows)


def normalize_electricity_mix(records, zone):
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        ts = pd.to_datetime(r["datetime"], utc=True)
        mix = r.get("mix", {}) or {}
        row = {"zone": zone, "timestamp_utc": ts}
        for k, v in mix.items():
            col = f"mix_{k.lower().replace(' ', '_').replace('-', '_')}_mw"
            row[col] = v
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_electricity_source(records, zone, source):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["zone"] = zone
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True)
    col = f"source_{source.lower().replace('-', '_')}_mw"
    df[col] = df["value"]
    return df[["zone", "timestamp_utc", col]]


SIGNAL_NORMALIZERS = {
    "carbon-intensity": lambda r, z: normalize_carbon_intensity(r, z),
    "carbon-free-energy": lambda r, z: normalize_scalar_signal(r, z, "carbon_free_energy_pct"),
    "carbon-intensity-fossil-only": lambda r, z: normalize_scalar_signal(r, z, "carbon_intensity_fossil_gco2_per_kwh"),
    "net-load": lambda r, z: normalize_scalar_signal(r, z, "net_load_mw"),
    "renewable-energy": lambda r, z: normalize_scalar_signal(r, z, "renewable_energy_pct"),
    "total-load": lambda r, z: normalize_scalar_signal(r, z, "total_load_mw"),
    "total-reported-load": lambda r, z: normalize_scalar_signal(r, z, "total_reported_load_mw"),
    "electricity-flows": lambda r, z: normalize_electricity_flows(r, z),
    "electricity-mix": lambda r, z: normalize_electricity_mix(r, z),
}


# ----------------------------
# Create Hourly Spine
# ----------------------------
def create_hourly_spine(zone, start_date, end_date):
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)
    timestamps = pd.date_range(
        start=start, end=end, freq="h", inclusive="left")
    return pd.DataFrame({"zone": zone, "timestamp_utc": timestamps})


# ----------------------------
# Stage Grid Signals
# ----------------------------
def stage_grid_signals(bucket, output_dir):
    """Stage all grid signals from GCS."""
    print("\n" + "=" * 60)
    print("STAGING GRID SIGNALS")
    print("=" * 60)

    # List all raw blobs
    blobs = list(bucket.list_blobs(prefix=RAW_PREFIX))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
    print(f"Found {len(jsonl_blobs)} JSONL files in raw/")

    # Group by zone
    zone_data = {}

    for blob in jsonl_blobs:
        zone = parse_zone_from_path(blob.name)
        if not zone:
            continue

        signal = parse_signal_from_path(blob.name)
        source = parse_source_from_path(blob.name)

        if zone not in zone_data:
            zone_data[zone] = []

        zone_data[zone].append({
            "blob_name": blob.name,
            "signal": signal,
            "source": source
        })

    print(f"Zones found: {list(zone_data.keys())}")

    # Process each zone
    all_zone_dfs = []

    for zone, blobs_info in zone_data.items():
        print(f"\n--- Processing Zone: {zone} ({len(blobs_info)} files) ---")

        zone_dfs = []

        for info in blobs_info:
            blob_name = info["blob_name"]
            signal = info["signal"]
            source = info["source"]

            # Download records
            records = download_jsonl(bucket, blob_name)

            # Normalize based on signal type
            if signal == "electricity-source" and source:
                df = normalize_electricity_source(records, zone, source)
            elif signal in SIGNAL_NORMALIZERS:
                df = SIGNAL_NORMALIZERS[signal](records, zone)
            else:
                continue

            if not df.empty:
                zone_dfs.append(df)

        if not zone_dfs:
            print(f"  No data normalized for {zone}")
            continue

        # Create spine and join all signals
        spine = create_hourly_spine(zone, START_DATE, END_DATE)
        result = spine.copy()

        # Group dataframes by their columns and merge
        for df in zone_dfs:
            df = df.drop_duplicates(subset=["zone", "timestamp_utc"])
            value_cols = [c for c in df.columns if c not in [
                "zone", "timestamp_utc"]]

            # Check for column conflicts
            for col in value_cols:
                if col in result.columns:
                    # Merge and coalesce
                    result = result.merge(
                        df[["zone", "timestamp_utc", col]],
                        on=["zone", "timestamp_utc"],
                        how="left",
                        suffixes=("", "_new")
                    )
                    result[col] = result[col].combine_first(
                        result.get(f"{col}_new", result[col]))
                    if f"{col}_new" in result.columns:
                        result.drop(columns=[f"{col}_new"], inplace=True)
                else:
                    result = result.merge(
                        df[["zone", "timestamp_utc", col]],
                        on=["zone", "timestamp_utc"],
                        how="left"
                    )

        print(f"  ✅ {zone}: {len(result)} rows, {len(result.columns)} columns")
        all_zone_dfs.append(result)

        # Save zone-specific parquet
        zone_path = os.path.join(output_dir, f"grid_signals_{zone}.parquet")
        result.to_parquet(zone_path, engine="pyarrow", index=False)
        print(f"  Saved: {zone_path}")

    # Combine all zones
    if all_zone_dfs:
        combined = pd.concat(all_zone_dfs, ignore_index=True)
        combined_path = os.path.join(
            output_dir, "grid_signals_all_zones.parquet")
        combined.to_parquet(combined_path, engine="pyarrow", index=False)
        print(f"\n✅ Combined grid data: {len(combined)} rows")
        print(f"   Saved: {combined_path}")
        return combined

    return pd.DataFrame()


# ----------------------------
# Stage Weather Data
# ----------------------------
def stage_weather_data(bucket, output_dir):
    """Stage all weather data from GCS."""
    print("\n" + "=" * 60)
    print("STAGING WEATHER DATA")
    print("=" * 60)

    # List all weather blobs
    blobs = list(bucket.list_blobs(prefix=WEATHER_PREFIX))
    csv_blobs = [b for b in blobs if b.name.endswith(".csv")]
    print(f"Found {len(csv_blobs)} CSV files in weather_raw/")

    if not csv_blobs:
        print("No weather data found.")
        return pd.DataFrame()

    # Group by location
    location_data = {}

    for blob in csv_blobs:
        location = parse_location_from_path(blob.name)
        if not location:
            continue

        if location not in location_data:
            location_data[location] = []
        location_data[location].append(blob.name)

    print(f"Locations found: {list(location_data.keys())}")

    # Process each location
    all_weather_dfs = []

    for location, blob_names in location_data.items():
        print(
            f"\n--- Processing Location: {location} ({len(blob_names)} files) ---")

        location_dfs = []
        for blob_name in blob_names:
            df = download_csv(bucket, blob_name)
            location_dfs.append(df)

        if location_dfs:
            combined = pd.concat(location_dfs, ignore_index=True)

            # Ensure timestamp column
            if 'timestamp_utc' in combined.columns:
                combined['timestamp_utc'] = pd.to_datetime(
                    combined['timestamp_utc'], utc=True)
            elif 'time' in combined.columns:
                combined['timestamp_utc'] = pd.to_datetime(
                    combined['time'], utc=True)
                combined.drop(columns=['time'], inplace=True, errors='ignore')

            # Add location if not present
            if 'location' not in combined.columns:
                combined['location'] = location

            # Remove duplicates
            combined = combined.drop_duplicates(
                subset=['timestamp_utc'], keep='last')
            combined = combined.sort_values(
                'timestamp_utc').reset_index(drop=True)

            print(
                f"  ✅ {location}: {len(combined)} rows, {len(combined.columns)} columns")
            all_weather_dfs.append(combined)

            # Save location-specific parquet
            loc_path = os.path.join(output_dir, f"weather_{location}.parquet")
            combined.to_parquet(loc_path, engine="pyarrow", index=False)
            print(f"  Saved: {loc_path}")

    # Combine all locations
    if all_weather_dfs:
        combined = pd.concat(all_weather_dfs, ignore_index=True)
        combined_path = os.path.join(
            output_dir, "weather_all_locations.parquet")
        combined.to_parquet(combined_path, engine="pyarrow", index=False)
        print(f"\n✅ Combined weather data: {len(combined)} rows")
        print(f"   Saved: {combined_path}")
        return combined

    return pd.DataFrame()


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 60)
    print("EcoPulse Complete Data Staging Pipeline")
    print("=" * 60)
    print(f"\nBucket: {BUCKET_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Stage grid signals
    grid_df = stage_grid_signals(bucket, OUTPUT_DIR)

    # Stage weather data
    weather_df = stage_weather_data(bucket, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("STAGING COMPLETE - SUMMARY")
    print("=" * 60)

    print("\n📊 Grid Signals:")
    if not grid_df.empty:
        print(f"   Total rows: {len(grid_df):,}")
        print(f"   Zones: {grid_df['zone'].unique().tolist()}")
        print(f"   Columns: {len(grid_df.columns)}")
        print(
            f"   Date range: {grid_df['timestamp_utc'].min()} to {grid_df['timestamp_utc'].max()}")

    print("\n🌤️ Weather Data:")
    if not weather_df.empty:
        print(f"   Total rows: {len(weather_df):,}")
        if 'location' in weather_df.columns:
            print(f"   Locations: {weather_df['location'].unique().tolist()}")
        print(f"   Columns: {len(weather_df.columns)}")

    print("\n📁 Files created:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.parquet'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"   {f} ({size:.1f} KB)")

    print("\n✅ All data staged successfully!")


if __name__ == "__main__":
    main()
