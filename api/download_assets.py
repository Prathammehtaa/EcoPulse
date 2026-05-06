"""
Downloads models from GCP Artifact Registry (Generic) and test data from GCS at container startup.
Uses gcloud CLI for model downloads — the requests-based REST download produces corrupt files.
"""
import os
import subprocess
from google.cloud import storage

# ── Model registry config ──────────────────────────────────────────────────────
REGISTRY_PROJECT  = os.environ.get("REGISTRY_PROJECT",  "ecopulse-mlops-pratham")
REGISTRY_LOCATION = os.environ.get("REGISTRY_LOCATION", "us-central1")
MODELS_REPO       = os.environ.get("MODELS_REPO",       "ecopulse-models-generic")

# ── Data config ────────────────────────────────────────────────────────────────
DATA_BUCKET = os.environ.get("DATA_BUCKET", "ecopulse-pratham-data")
DATA_BLOB   = os.environ.get("DATA_BLOB",   "processed/test_split.parquet")

# ── Local paths ────────────────────────────────────────────────────────────────
MODELS_DIR = "/app/Model_Pipeline/models"
DATA_DIR   = "/app/Data_Pipeline/data/processed"

MODELS = [
    ("ecopulse-xgboost-1h",   "xgboost_1h.ubj"),
    ("ecopulse-xgboost-12h",  "xgboost_12h.ubj"),
    ("ecopulse-xgboost-24h",  "xgboost_24h.ubj"),
    ("ecopulse-lightgbm-1h",  "lightgbm_1h.joblib"),
    ("ecopulse-lightgbm-12h", "lightgbm_12h.joblib"),
    ("ecopulse-lightgbm-24h", "lightgbm_24h.joblib"),
]


def _gcloud(*args, timeout=300):
    cmd = ["gcloud", "--quiet"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"gcloud failed (exit {result.returncode}):\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _latest_version(package: str) -> str:
    output = _gcloud(
        "artifacts", "versions", "list",
        f"--project={REGISTRY_PROJECT}",
        f"--location={REGISTRY_LOCATION}",
        f"--repository={MODELS_REPO}",
        f"--package={package}",
        "--sort-by=~createTime",
        "--limit=1",
        "--format=value(name)",
        timeout=60,
    )
    if not output:
        raise RuntimeError(f"No versions found for package {package}")
    return output.split("/")[-1]


def download_model(package: str, filename: str):
    version = _latest_version(package)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"  {package} v{version} → {MODELS_DIR}/{filename}")
    _gcloud(
        "artifacts", "generic", "download",
        f"--project={REGISTRY_PROJECT}",
        f"--location={REGISTRY_LOCATION}",
        f"--repository={MODELS_REPO}",
        f"--package={package}",
        f"--version={version}",
        f"--destination={MODELS_DIR}",
        f"--name={filename}",
        timeout=300,
    )


def download_test_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "test_split.parquet")
    client = storage.Client()
    blob = client.bucket(DATA_BUCKET).blob(DATA_BLOB)
    print(f"  gs://{DATA_BUCKET}/{DATA_BLOB} → {dest}")
    blob.download_to_filename(dest)


if __name__ == "__main__":
    print(f"Downloading models from {MODELS_REPO}...")
    for package, filename in MODELS:
        try:
            download_model(package, filename)
        except Exception as e:
            print(f"  WARNING: {package} failed — {e}")

    print(f"Downloading test data from gs://{DATA_BUCKET}/{DATA_BLOB}...")
    try:
        download_test_data()
    except Exception as e:
        print(f"WARNING: Test data unavailable — {e}. API will use fallback forecasts.")

    print("Asset download complete.")
