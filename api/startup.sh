#!/bin/bash
set -e

echo "Downloading models and data from GCS..."
python api/download_assets.py

echo "Starting FastAPI..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000