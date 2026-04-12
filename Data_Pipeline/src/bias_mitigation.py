import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'bias_mitigation')

CARBON_BUCKETS = {
    'Very Low': (0, 100),
    'Low': (100, 200),
    'Medium': (200, 350),
    'High': (350, 500),
    'Very High': (500, float('inf'))
}


def create_carbon_bucket(value):
    if pd.isna(value):
        return 'Unknown'
    for bucket, (low, high) in CARBON_BUCKETS.items():
        if low <= value < high:
            return bucket
    return 'Unknown'


def get_severity(ratio):
    if ratio <= 2:
        return 'LOW'
    elif ratio <= 5:
        return 'MODERATE'
    elif ratio <= 10:
        return 'HIGH'
    else:
        return 'SEVERE'


def random_oversample(df, target_col, random_state=42):
    print('Applying Random Oversampling...')
    np.random.seed(random_state)

    counts = df[target_col].value_counts()
    max_count = counts.max()
    print(f'  Before: {counts.to_dict()}')

    resampled = []
    for label in counts.index:
        class_df = df[df[target_col] == label]
        if len(class_df) < max_count:
            extra = class_df.sample(
                n=max_count - len(class_df),
                replace=True,
                random_state=random_state
            )
            resampled.append(pd.concat([class_df, extra], ignore_index=True))
        else:
            resampled.append(class_df)

    result = pd.concat(resampled, ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    print(f'  After:  {result[target_col].value_counts().to_dict()}')
    return result


def main():
    print('=' * 60)
    print('ECOPULSE BIAS MITIGATION PIPELINE')
    print('=' * 60)

    # Load TRAIN split only
    input_path = os.path.join(PROCESSED_DIR, 'train_split.parquet')
    print(f'Loading: {input_path}')
    df = pd.read_parquet(input_path)
    print(f'Loaded {len(df)} rows')

    carbon_col = 'carbon_intensity_gco2_per_kwh'
    df['carbon_bucket'] = df[carbon_col].apply(create_carbon_bucket)
    df = df[df['carbon_bucket'] != 'Unknown'].copy()

    counts_before = df['carbon_bucket'].value_counts()
    ratio_before = float(counts_before.max() / counts_before.min())
    print(f'\nBefore Mitigation:')
    print(f'  Samples: {len(df)}')
    print(f'  Imbalance Ratio: {ratio_before:.2f}')
    print(f'  Severity: {get_severity(ratio_before)}')

    # Apply mitigation only on training data
    df_mitigated = random_oversample(df, 'carbon_bucket')

    counts_after = df_mitigated['carbon_bucket'].value_counts()
    ratio_after = float(counts_after.max() / counts_after.min())
    print(f'\nAfter Mitigation:')
    print(f'  Samples: {len(df_mitigated)}')
    print(f'  Imbalance Ratio: {ratio_after:.2f}')
    print(f'  Severity: {get_severity(ratio_after)}')

    # Save mitigated training split
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, 'train_split_mitigated.parquet')
    df_mitigated.to_parquet(output_path, index=False)
    print(f'\nSaved mitigated train split to: {output_path}')

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        'generated_at': datetime.now().isoformat(),
        'strategy': 'random_oversampling',
        'input_file': 'train_split.parquet',
        'output_file': 'train_split_mitigated.parquet',
        'before': {
            'samples': len(df),
            'imbalance_ratio': round(ratio_before, 2),
            'severity': get_severity(ratio_before),
            'distribution': {str(k): int(v) for k, v in counts_before.items()}
        },
        'after': {
            'samples': len(df_mitigated),
            'imbalance_ratio': round(ratio_after, 2),
            'severity': get_severity(ratio_after),
            'distribution': {str(k): int(v) for k, v in counts_after.items()}
        }
    }

    report_path = os.path.join(REPORTS_DIR, 'mitigation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Report saved to: {report_path}')
    print('\nDone!')


if __name__ == '__main__':
    main()