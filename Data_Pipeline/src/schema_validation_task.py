# src/schema_validation_task.py
import logging
from pathlib import Path

import pandas as pd
import yaml

from schema_validation import validate_dataset

logger = logging.getLogger(__name__)


def _load_config(project_root: Path) -> dict:
    cfg_path = project_root / "pipeline_config" / "preprocessing_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def run_tfdv_schema_validation():
    from schema_validation import validate_dataset
    """
    Loads datasets from local processed paths and validates them against baseline schemas.
    Returns True if all pass; raises if any fails (so Airflow marks task failed).
    """
    project_root = Path(__file__).resolve().parent.parent
    config = _load_config(project_root)

    processed_dir = (project_root / config["local"]["processed"]).resolve()
    schemas_dir = (project_root / "data_validation" / "schemas").resolve()

    # Update these filenames to match your config keys / outputs
    files = config["output"]["files"]

    datasets = {
        "grid": processed_dir / files["grid_processed"],
        "weather": processed_dir / files["weather_processed"],
        "merged": processed_dir / files.get("merged_processed", "merged_dataset.parquet"),
        "features": processed_dir / files.get("feature_table", "feature_table.parquet"),
    }

    all_ok = True

    for name, path in datasets.items():
        if not path.exists():
            logger.warning("Dataset parquet missing for %s: %s (skipping)", name, path)
            continue

        logger.info("Loading %s: %s", name, path)
        df = pd.read_parquet(path)

        ok = validate_dataset(df, dataset_name=name, schemas_dir=str(schemas_dir))
        all_ok = all_ok and ok

    if not all_ok:
        raise ValueError("TFDV schema validation failed for one or more datasets.")

    logger.info("TFDV schema validation passed for all available datasets.")
    return True