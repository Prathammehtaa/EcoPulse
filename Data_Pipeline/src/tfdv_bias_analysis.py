"""
EcoPulse TFDV Bias Analysis Module
===================================
Uses TensorFlow Data Validation (TFDV) to detect data bias across slices.

This version analyzes train / val / test splits separately.
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import tensorflow_data_validation as tfdv
    TFDV_AVAILABLE = True
except ImportError:
    TFDV_AVAILABLE = False
    logger.warning(
        "TFDV not installed. Install with: pip install tensorflow-data-validation"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "tfdv_bias")

DATA_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_split.parquet"),
    "val": os.path.join(PROCESSED_DIR, "val_split.parquet"),
    "test": os.path.join(PROCESSED_DIR, "test_split.parquet"),
}

SLICE_CONFIGS = {
    'hour_of_day': {
        'description': 'Hour of day (0-23)',
        'expected_count': 24
    },
    'day_of_week': {
        'description': 'Day of the week',
        'expected_values': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    },
    'season': {
        'description': 'Season of the year',
        'expected_values': ['Winter', 'Spring', 'Summer', 'Fall']
    },
    'month': {
        'description': 'Month of the year',
        'expected_count': 12
    },
    'is_weekend': {
        'description': 'Weekend vs Weekday',
        'expected_values': ['True', 'False']
    },
    'carbon_intensity_bucket': {
        'description': 'Carbon intensity level',
        'expected_values': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    }
}

IMBALANCE_THRESHOLD = 0.5


def load_staged_data(data_file: str) -> pd.DataFrame:
    logger.info(f"Loading: {data_file}")

    if not os.path.exists(data_file):
        logger.error(f"File not found: {data_file}")
        return pd.DataFrame()

    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} rows")
    return df


def add_slice_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'datetime' not in df.columns:
        logger.warning("No datetime column found")
        return df

    df['timestamp_utc'] = pd.to_datetime(df['datetime'], utc=True)

    df['hour_of_day'] = df['timestamp_utc'].dt.hour.astype(str)
    df['day_of_week'] = df['timestamp_utc'].dt.day_name()
    df['month'] = df['timestamp_utc'].dt.month.astype(str)
    df['year'] = df['timestamp_utc'].dt.year

    def get_season(month):
        month = int(month) if isinstance(month, str) else month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['timestamp_utc'].dt.month.apply(get_season)
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(str)

    ci_col = 'carbon_intensity_gco2_per_kwh'
    if ci_col in df.columns:
        bins = [0, 100, 200, 300, 400, float('inf')]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['carbon_intensity_bucket'] = pd.cut(
            df[ci_col].fillna(0),
            bins=bins,
            labels=labels,
            include_lowest=True
        ).astype(str)

    logger.info(f"Added slice features. New columns: {list(df.columns)}")
    return df


def generate_tfdv_statistics(df: pd.DataFrame):
    if not TFDV_AVAILABLE:
        logger.error("TFDV not available")
        return None

    logger.info("Generating TFDV statistics...")
    stats = tfdv.generate_statistics_from_dataframe(df)
    logger.info("Statistics generated successfully")
    return stats


def infer_and_save_schema(stats, output_dir: str):
    if not TFDV_AVAILABLE or stats is None:
        return None

    logger.info("Inferring schema...")
    schema = tfdv.infer_schema(stats)

    os.makedirs(output_dir, exist_ok=True)
    schema_path = os.path.join(output_dir, 'schema.pbtxt')
    tfdv.write_schema_text(schema, schema_path)
    logger.info(f"Schema saved to: {schema_path}")

    return schema


def validate_and_detect_anomalies(stats, schema, output_dir: str):
    if not TFDV_AVAILABLE or stats is None or schema is None:
        return None

    logger.info("Validating data against schema...")
    anomalies = tfdv.validate_statistics(stats, schema)

    anomaly_count = len(anomalies.anomaly_info)
    if anomaly_count > 0:
        logger.warning(f"Found {anomaly_count} anomalies")
    else:
        logger.info("No anomalies detected")

    anomalies_path = os.path.join(output_dir, 'anomalies.pbtxt')
    tfdv.write_anomalies_text(anomalies, anomalies_path)
    logger.info(f"Anomalies saved to: {anomalies_path}")

    return anomalies


def analyze_slice_distribution(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    if slice_col not in df.columns:
        logger.warning(f"Column '{slice_col}' not found")
        return pd.DataFrame()

    dist = df[slice_col].value_counts().reset_index()
    dist.columns = ['slice_value', 'count']
    dist['percentage'] = (dist['count'] / dist['count'].sum() * 100).round(2)

    n_slices = len(dist)
    expected_pct = 100 / n_slices
    dist['expected_pct'] = round(expected_pct, 2)
    dist['deviation_pct'] = (
        (dist['percentage'] - expected_pct) / expected_pct * 100
    ).round(2)

    dist['is_underrepresented'] = dist['percentage'] < (expected_pct * IMBALANCE_THRESHOLD)
    dist['is_overrepresented'] = dist['percentage'] > (expected_pct * (2 - IMBALANCE_THRESHOLD))

    return dist.sort_values('slice_value')


def compute_target_stats_by_slice(
    df: pd.DataFrame,
    slice_col: str,
    target_col: str = 'carbon_intensity_gco2_per_kwh'
) -> pd.DataFrame:
    if slice_col not in df.columns:
        return pd.DataFrame()

    if target_col not in df.columns:
        logger.warning(f"No target column found for statistics")
        return pd.DataFrame()

    stats = df.groupby(slice_col)[target_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max'),
        ('missing_pct', lambda x: x.isna().mean() * 100)
    ]).round(2)

    return stats.reset_index()


def run_bias_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for slice_col, config in SLICE_CONFIGS.items():
        if slice_col not in df.columns:
            logger.warning(f"Slice '{slice_col}' not in data, skipping")
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Analyzing slice: {slice_col}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'=' * 50}")

        dist = analyze_slice_distribution(df, slice_col)
        target_stats = compute_target_stats_by_slice(df, slice_col)

        underrep = dist[dist['is_underrepresented']]['slice_value'].tolist()
        overrep = dist[dist['is_overrepresented']]['slice_value'].tolist()

        results[slice_col] = {
            'distribution': dist,
            'target_stats': target_stats,
            'underrepresented': underrep,
            'overrepresented': overrep
        }

        dist.to_csv(os.path.join(output_dir, f'distribution_{slice_col}.csv'), index=False)
        if not target_stats.empty:
            target_stats.to_csv(os.path.join(output_dir, f'stats_{slice_col}.csv'), index=False)

        logger.info(f"Unique values: {len(dist)}")
        if underrep:
            logger.warning(f"⚠️ Underrepresented: {underrep}")
        if overrep:
            logger.warning(f"⚠️ Overrepresented: {overrep}")
        if not underrep and not overrep:
            logger.info("✅ Distribution is balanced")

    return results


def generate_report(split_name: str, results: Dict, df: pd.DataFrame, output_dir: str) -> str:
    lines = [
        f"# EcoPulse TFDV Bias Analysis Report - {split_name.upper()}",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Dataset Split:** {split_name}",
        f"\n**Tool:** TensorFlow Data Validation (TFDV)",
        f"\n**Total Records:** {len(df):,}",
        f"\n**Date Range:** {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}",
        "\n---\n",
        "## Executive Summary\n"
    ]

    total_issues = 0
    issue_details = []

    for slice_name, data in results.items():
        under = data.get('underrepresented', [])
        over = data.get('overrepresented', [])

        if under:
            total_issues += len(under)
            issue_details.append(
                f"- **{slice_name}**: {len(under)} underrepresented: `{under}`"
            )
        if over:
            total_issues += len(over)
            issue_details.append(
                f"- **{slice_name}**: {len(over)} overrepresented: `{over}`"
            )

    if total_issues > 0:
        lines.append(f"### ⚠️ {total_issues} Bias Issues Detected\n")
        lines.extend(issue_details)
    else:
        lines.append("### ✅ No Significant Bias Issues Detected\n")
        lines.append("All slices are within acceptable distribution thresholds.\n")

    lines.append("\n---\n")
    lines.append("## Detailed Slice Analysis\n")

    for slice_name, data in results.items():
        dist = data.get('distribution')
        if dist is None or dist.empty:
            continue

        config = SLICE_CONFIGS.get(slice_name, {})

        lines.append(f"\n### {slice_name}\n")
        lines.append(f"**Description:** {config.get('description', 'N/A')}\n")
        lines.append("\n| Slice Value | Count | % | Expected % | Deviation | Status |")
        lines.append("|-------------|-------|---|------------|-----------|--------|")

        for _, row in dist.iterrows():
            if row['is_underrepresented']:
                status = "⚠️ Under"
            elif row['is_overrepresented']:
                status = "⚠️ Over"
            else:
                status = "✅ OK"

            lines.append(
                f"| {row['slice_value']} | {row['count']:,} | {row['percentage']:.1f}% | "
                f"{row['expected_pct']:.1f}% | {row['deviation_pct']:+.1f}% | {status} |"
            )

    lines.append("\n---\n")
    lines.append("## Recommendations\n")

    if total_issues > 0:
        lines.append("""
### Mitigation Strategies

1. **For Underrepresented Slices:**
   - Collect more data for these time periods
   - Apply oversampling only on the training split during model training

2. **For Overrepresented Slices:**
   - Apply undersampling or sample weighting on the training split only

3. **General:**
   - Keep temporal train/val/test splits
   - Monitor model performance separately for each slice
   - Set up alerts for distribution drift in production
""")
    else:
        lines.append("Data distribution is acceptable. Continue monitoring for drift.\n")

    report_content = "\n".join(lines)
    report_path = os.path.join(output_dir, f'tfdv_bias_report_{split_name}.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Report saved to: {report_path}")
    return report_content


def process_split(split_name: str, data_file: str):
    print("\n" + "=" * 60)
    print(f"Processing split: {split_name.upper()}")
    print("=" * 60)

    split_output_dir = os.path.join(REPORTS_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    df = load_staged_data(data_file)
    if df.empty:
        print(f"\n❌ No data found for split: {split_name}")
        return 0

    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    df = add_slice_features(df)

    if TFDV_AVAILABLE:
        print("\n--- TFDV Statistics ---")
        stats = generate_tfdv_statistics(df)

        print("\n--- Inferring Schema ---")
        schema = infer_and_save_schema(stats, split_output_dir)

        print("\n--- Detecting Anomalies ---")
        validate_and_detect_anomalies(stats, schema, split_output_dir)
    else:
        print("TFDV not installed - skipping statistics generation")

    print("\n--- Bias Analysis Across Slices ---")
    results = run_bias_analysis(df, split_output_dir)

    print("\n--- Generating Report ---")
    generate_report(split_name, results, df, split_output_dir)

    total_issues = sum(
        len(r.get('underrepresented', [])) + len(r.get('overrepresented', []))
        for r in results.values()
    )

    print(f"\n📊 Summary for {split_name}: {total_issues} bias issues found")
    return total_issues


def main():
    print("=" * 60)
    print("EcoPulse TFDV Bias Analysis Pipeline")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    total_issues_all = 0
    processed_splits = 0

    for split_name, data_file in DATA_FILES.items():
        if os.path.exists(data_file):
            total_issues_all += process_split(split_name, data_file)
            processed_splits += 1
        else:
            print(f"\nSkipping {split_name}: file not found -> {data_file}")

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\nReports saved to: {REPORTS_DIR}")
    print(f"Processed splits: {processed_splits}")
    print(f"Total bias issues across all splits: {total_issues_all}")


if __name__ == "__main__":
    main()