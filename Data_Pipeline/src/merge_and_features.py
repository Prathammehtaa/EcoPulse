import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "pipeline_config" / "preprocessing_config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(grid_path, weather_path):
    logger.info(f"Loading grid: {grid_path}")
    grid = pd.read_parquet(grid_path)
    grid["datetime"] = pd.to_datetime(grid["datetime"], utc=True)

    logger.info(f"Loading weather: {weather_path}")
    weather = pd.read_parquet(weather_path)
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)

    logger.info(f"Grid: {grid.shape}, Weather: {weather.shape}")
    return grid, weather


def merge_datasets(grid, weather):
    merged = pd.merge(grid, weather, on=["datetime", "zone"], how="inner")
    logger.info(f"Merged: {merged.shape} (grid={len(grid)}, weather={len(weather)}, merged={len(merged)})")

    row_loss = len(grid) - len(merged)
    if row_loss > 0:
        logger.warning(f"Lost {row_loss} rows ({row_loss/len(grid)*100:.1f}%) in join")

    zones = merged["zone"].unique().tolist()
    logger.info(f"Zones: {zones}")
    logger.info(f"Zones: {zones}")
    return merged


def run_merge(config, use_gcs=False):
    if use_gcs:
        proc_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['processed'].strip('/')}"
        grid_path = f"{proc_dir}/{config['output']['files']['grid_processed']}"
        weather_path = f"{proc_dir}/{config['output']['files']['weather_processed']}"
        merged_path = f"{proc_dir}/{config['output']['files']['merged']}"
    else:
        proc_dir = PROJECT_ROOT / config["local"]["processed"]
        grid_path = proc_dir / config["output"]["files"]["grid_processed"]
        weather_path = proc_dir / config["output"]["files"]["weather_processed"]
        merged_path = proc_dir / config["output"]["files"]["merged"]

    logger.info("=" * 60)
    logger.info("PHASE 1: Merging datasets")
    logger.info("=" * 60)

    grid, weather = load_processed_data(str(grid_path), str(weather_path))
    df = merge_datasets(grid, weather)

    Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(merged_path), index=False, compression="snappy")
    logger.info(f"Saved merged: {merged_path}")

   
def main():
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )
    run_merge(config, use_gcs=False)


if __name__ == "__main__":
    main()