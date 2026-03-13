import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage')
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
    if ratio <= 2: return 'LOW'
    elif ratio <= 5: return 'MODERATE'
    elif ratio <= 10: return 'HIGH'
    else: return 'SEVERE'

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
            extra = class_df.sample(n=max_count - len(class_df), replace=True, random_state=random_state)
            resampled.append(pd.concat([class_df, extra], ignore_index=True))
        else:
            resampled.append(class_df)
    result = pd.concat(resampled, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'  After:  {result[target_col].value_counts().to_dict()}')
    return result

def stratified_split(df, target_col, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    print('Creating Stratified Splits...')
    np.random.seed(random_state)
    train, val, test = [], [], []
    for label in df[target_col].unique():
        class_df = df[df[target_col] == label].sample(frac=1, random_state=random_state)
        n = len(class_df)
        t1, t2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        train.append(class_df.iloc[:t1])
        val.append(class_df.iloc[t1:t2])
        test.append(class_df.iloc[t2:])
    train_df = pd.concat(train, ignore_index=True).sample(frac=1, random_state=random_state)
    val_df = pd.concat(val, ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test, ignore_index=True).sample(frac=1, random_state=random_state)
    print(f'  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    return train_df, val_df, test_df

def main():
    print('='*60)
    print('ECOPULSE BIAS MITIGATION PIPELINE')
    print('='*60)
    
    # Load data
    input_path = os.path.join(STAGE_DIR, 'grid_signals_all_zones.parquet')
    print(f'Loading: {input_path}')
    df = pd.read_parquet(input_path)
    print(f'Loaded {len(df)} rows')
    
    # Add carbon buckets
    carbon_col = 'carbon_intensity_gco2_per_kwh'
    df['carbon_bucket'] = df[carbon_col].apply(create_carbon_bucket)
    df = df[df['carbon_bucket'] != 'Unknown']
    
    # Analyze before
    counts_before = df['carbon_bucket'].value_counts()
    ratio_before = float(counts_before.max() / counts_before.min())
    print(f'\nBefore Mitigation:')
    print(f'  Samples: {len(df)}')
    print(f'  Imbalance Ratio: {ratio_before:.2f}')
    print(f'  Severity: {get_severity(ratio_before)}')
    
    # Apply oversampling
    df_mitigated = random_oversample(df, 'carbon_bucket')
    
    # Analyze after
    counts_after = df_mitigated['carbon_bucket'].value_counts()
    ratio_after = float(counts_after.max() / counts_after.min())
    print(f'\nAfter Mitigation:')
    print(f'  Samples: {len(df_mitigated)}')
    print(f'  Imbalance Ratio: {ratio_after:.2f}')
    print(f'  Severity: {get_severity(ratio_after)}')
    
    # Stratified split
    train_df, val_df, test_df = stratified_split(df_mitigated, 'carbon_bucket')
    
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train_balanced.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, 'val_balanced.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test_balanced.csv'), index=False)
    print(f'\nSaved to: {PROCESSED_DIR}')
    
    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        'generated_at': datetime.now().isoformat(),
        'strategy': 'random_oversampling',
        'before': {'samples': len(df), 'imbalance_ratio': round(ratio_before, 2), 'severity': get_severity(ratio_before), 'distribution': {str(k): int(v) for k, v in counts_before.items()}},
        'after': {'samples': len(df_mitigated), 'imbalance_ratio': round(ratio_after, 2), 'severity': get_severity(ratio_after), 'distribution': {str(k): int(v) for k, v in counts_after.items()}},
        'splits': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
    }
    with open(os.path.join(REPORTS_DIR, 'mitigation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to: {REPORTS_DIR}')
    print('\nDone!')

if __name__ == '__main__':
    main()
