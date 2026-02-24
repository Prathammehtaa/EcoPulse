"""
EcoPulse Schema Validation Module
Validates data against baseline schemas using TFDV
"""

import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def validate_dataset(df, dataset_name, schemas_dir='data_validation/schemas'):
    """
    Validate a DataFrame against its baseline schema
    
    Args:
        df: pandas DataFrame to validate
        dataset_name: name of dataset ('weather', 'grid', 'merged', 'features')
        schemas_dir: directory containing baseline schemas
        
    Returns:
        True if valid, False if anomalies detected
    """
    try:
        schema_path = os.path.join(schemas_dir, f'{dataset_name}_schema.pbtxt')
        
        if not os.path.exists(schema_path):
            logger.warning(f"Baseline schema not found: {schema_path}. Skipping validation.")
            return True
        
        logger.info(f"Validating {dataset_name} against baseline schema...")
        
        new_stats = tfdv.generate_statistics_from_dataframe(df)
        baseline_schema = tfdv.load_schema_text(schema_path)
        
        anomalies = tfdv.validate_statistics(
            statistics=new_stats,
            schema=baseline_schema
        )
        
        if anomalies.anomaly_info:
            logger.error(f"Schema validation FAILED for {dataset_name}")
            logger.error(f"Anomalies detected: {len(anomalies.anomaly_info)}")
            
            for feature, info in anomalies.anomaly_info.items():
                logger.error(f"  - {feature}: {info.short_description}")
            
            return False
        else:
            logger.info(f"Schema validation PASSED for {dataset_name}")
            return True
            
    except Exception as e:
        logger.error(f"Validation error for {dataset_name}: {e}")
        return False


def detect_drift(new_df, baseline_stats_path, dataset_name):
    """
    Detect data drift by comparing new data statistics to baseline
    
    Args:
        new_df: DataFrame with new data
        baseline_stats_path: Path to baseline statistics file
        dataset_name: Name of dataset for logging
        
    Returns:
        None (visualization only in current implementation)
    """
    logger.info(f"Checking drift for {dataset_name}...")
    
    new_stats = tfdv.generate_statistics_from_dataframe(new_df)
    baseline_stats = tfdv.load_stats_text(baseline_stats_path)
    
    tfdv.visualize_statistics(
        lhs_statistics=baseline_stats,
        rhs_statistics=new_stats,
        lhs_name='BASELINE',
        rhs_name='NEW DATA'
    )
    
    return None