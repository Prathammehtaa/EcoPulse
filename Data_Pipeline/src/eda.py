import os
import re
import glob
import pandas as pd

# ----------------------------
# CONFIG (edit if needed)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ...Data_Pipeline\src
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # ...Data_Pipeline

STAGE_ROOT = os.path.join(PROJECT_ROOT, "data", "stage")

GLOB_PATTERN = os.path.join(STAGE_ROOT, "grid_signals_hourly_base", "**", "data.parquet")
print(os.listdir(STAGE_ROOT))

START_UTC = "2023-01-01T00:00:00Z"
END_UTC   = "2026-01-01T00:00:00Z"  # exclusive

# Parse from path like: ...\zone=PJM\year=2023\month=01\data.parquet
PATH_RE = re.compile(r"zone=([^\\/]+).*year=(\d{4}).*month=(\d{2})", re.IGNORECASE)


def extract_partitions(path: str):
    m = PATH_RE.search(path)
    if not m:
        return None, None, None
    zone = m.group(1)
    year = int(m.group(2))
    month = int(m.group(3))
    return zone, year, month


def load_parquets_for_range(paths, start_utc, end_utc):
    start = pd.to_datetime(start_utc, utc=True)
    end = pd.to_datetime(end_utc, utc=True)

    dfs = []
    meta_rows = []

    for p in paths:
        zone, year, month = extract_partitions(p)

        # If your path parsing fails for some reason, still try reading
        df_part = pd.read_parquet(p)

        # Ensure timestamp is UTC datetime
        if "timestamp_utc" in df_part.columns:
            df_part["timestamp_utc"] = pd.to_datetime(df_part["timestamp_utc"], utc=True)

        # Filter to 3-year range
        if "timestamp_utc" in df_part.columns:
            df_part = df_part[(df_part["timestamp_utc"] >= start) & (df_part["timestamp_utc"] < end)]

        # Add partition columns (helpful for validation)
        if zone is not None:
            df_part["zone_from_path"] = zone
        if year is not None:
            df_part["year_from_path"] = year
        if month is not None:
            df_part["month_from_path"] = month

        dfs.append(df_part)

        meta_rows.append({
            "path": p,
            "zone_from_path": zone,
            "year_from_path": year,
            "month_from_path": month,
            "rows_loaded_after_filter": len(df_part),
        })

    meta = pd.DataFrame(meta_rows)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df, meta


def main():
    # 1) Find all parquet files (across all years/months)
    paths = glob.glob(GLOB_PATTERN, recursive=True)
    print(f"Found {len(paths)} parquet files under: {STAGE_ROOT}")

    # 2) Load & filter to 2023-01-01 .. 2026-01-01 (exclusive)
    df, meta = load_parquets_for_range(paths, START_UTC, END_UTC)

    # Optional: save file-level metadata so you can verify all months/years exist
    meta.to_csv("eda_parquet_files_loaded.csv", index=False)
    print("Saved file metadata -> eda_parquet_files_loaded.csv")

    # 3) Quick EDA
    print("\nRows, Cols:", df.shape)
    print("\nHead (first 5):")
    print(df.head())

    df.head(20).to_csv("eda_sample_head20.csv", index=False)
    print("\nSaved sample -> eda_sample_head20.csv")

    if "zone" in df.columns:
        print("\nZones:", df["zone"].nunique(), df["zone"].unique())

    if "timestamp_utc" in df.columns:
        print("\nTime range:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
        print("Duplicates (zone,timestamp):", df.duplicated(["zone", "timestamp_utc"]).sum())

        # sanity: counts by year
        df["year"] = df["timestamp_utc"].dt.year
        print("\nRows by year:\n", df["year"].value_counts().sort_index())

    null_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
    print("\nTop missing columns (%):\n", null_pct.head(10))

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        print("\nNumeric summary (first 8 numeric cols):\n",
              df[num_cols[:8]].describe().T[["mean", "std", "min", "max"]].round(2))

    # 4) Simple correlation check for key signals (if present)
    cols = [c for c in ["carbon_intensity_gco2_per_kwh", "total_load_mw", "renewable_energy_mw", "net_flow_mw"]
            if c in df.columns]
    if len(cols) >= 2:
        print("\nCorrelations:\n", df[cols].corr().round(3))


if __name__ == "__main__":
    main()
