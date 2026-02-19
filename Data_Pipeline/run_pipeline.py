import argparse
import logging
import sys
import os
import time
from datetime import datetime

import yaml

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from grid_preprocessing import process_grid_data
from weather_preprocessing import process_weather_data
from merge_and_features import run_merge_and_feature_engineering


def load_config(path=None):
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "config", "preprocessing_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_step(name, func, config, use_gcs, logger):
    logger.info(f"{'#' * 60}")
    logger.info(f"# {name}")
    logger.info(f"{'#' * 60}")

    start = time.time()
    try:
        result = func(config, use_gcs=use_gcs)
        elapsed = time.time() - start
        logger.info(f">>> {name} DONE in {elapsed:.1f}s | Shape: {result.shape}")
        return {"status": "SUCCESS", "shape": result.shape, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f">>> {name} FAILED after {elapsed:.1f}s: {e}", exc_info=True)
        return {"status": "FAILED", "error": str(e), "time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="EcoPulse Preprocessing Pipeline")
    parser.add_argument("--gcs", action="store_true", help="Use GCS bucket")
    parser.add_argument("--step", choices=["grid", "weather", "merge", "all"],
                        default="all", help="Run specific step (default: all)")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    # Setup logging FIRST before anything else
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("pipeline")

    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  EcoPulse - Data Preprocessing Pipeline")
    logger.info(f"  Started: {datetime.now().isoformat()}")
    logger.info(f"  Mode: {'GCS' if args.gcs else 'Local'}")
    logger.info(f"  Step: {args.step}")
    logger.info("=" * 60)

    results = {}
    t0 = time.time()

    if args.step in ("grid", "all"):
        results["grid"] = run_step("Grid Preprocessing", process_grid_data, config, args.gcs, logger)
        if results["grid"]["status"] == "FAILED" and args.step == "all":
            logger.error("Grid failed - stopping")
            sys.exit(1)

    if args.step in ("weather", "all"):
        results["weather"] = run_step("Weather Preprocessing", process_weather_data, config, args.gcs, logger)
        if results["weather"]["status"] == "FAILED" and args.step == "all":
            logger.error("Weather failed - stopping")
            sys.exit(1)

    if args.step in ("merge", "all"):
        results["merge"] = run_step("Merge & Features", run_merge_and_feature_engineering, config, args.gcs, logger)

    total = time.time() - t0

    # Print summary to stdout directly (not through logging)
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for step, r in results.items():
        icon = "OK" if r["status"] == "SUCCESS" else "FAIL"
        shape = r.get("shape", "N/A")
        print(f"  [{icon}] {step:12s} | {r['time']:6.1f}s | shape={shape}")
    print(f"\n  Total: {total:.1f}s")
    print(f"{'=' * 60}")

    if any(r["status"] == "FAILED" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()