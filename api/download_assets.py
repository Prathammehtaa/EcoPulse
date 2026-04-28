"""
Downloads models from GCP Artifact Registry (Generic) and test data from GCS at container startup.
Uses google-auth + requests (already in requirements.txt) — no gcloud CLI needed.
"""
import os
import requests
import google.auth
import google.auth.transport.requests
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
    ("ecopulse-xgboost-1h",   "xgboost_1h.joblib"),
    ("ecopulse-xgboost-12h",  "xgboost_12h.joblib"),
    ("ecopulse-xgboost-24h",  "xgboost_24h.joblib"),
    ("ecopulse-lightgbm-1h",  "lightgbm_1h.joblib"),
    ("ecopulse-lightgbm-12h", "lightgbm_12h.joblib"),
    ("ecopulse-lightgbm-24h", "lightgbm_24h.joblib"),
]


def _credentials():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _latest_version(creds, package: str) -> str:
    url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{REGISTRY_PROJECT}/locations/{REGISTRY_LOCATION}/"
        f"repositories/{MODELS_REPO}/packages/{package}/versions"
        f"?orderBy=createTime+desc&pageSize=1"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {creds.token}"})
    resp.raise_for_status()
    versions = resp.json().get("versions", [])
    if not versions:
        raise RuntimeError(f"No versions found for package {package}")
    return versions[0]["name"].split("/")[-1]


def download_model(creds, package: str, filename: str):
    version = _latest_version(creds, package)
    url = (
        f"https://{REGISTRY_LOCATION}-generic.pkg.dev/"
        f"{REGISTRY_PROJECT}/{MODELS_REPO}/{package}/{version}/{filename}"
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, filename)
    print(f"  {package} v{version} → {dest}")
    resp = requests.get(url, headers={"Authorization": f"Bearer {creds.token}"}, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def download_test_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "test_split.parquet")
    client = storage.Client()
    blob = client.bucket(DATA_BUCKET).blob(DATA_BLOB)
    print(f"  gs://{DATA_BUCKET}/{DATA_BLOB} → {dest}")
    blob.download_to_filename(dest)


if __name__ == "__main__":
    print(f"Downloading models from {MODELS_REPO}...")
    try:
        creds = _credentials()
        for package, filename in MODELS:
            try:
                download_model(creds, package, filename)
            except Exception as e:
                print(f"  WARNING: {package} failed — {e}")
    except Exception as e:
        print(f"WARNING: Could not authenticate for model download — {e}")

    print(f"Downloading test data from gs://{DATA_BUCKET}/{DATA_BLOB}...")
    try:
        download_test_data()
    except Exception as e:
        print(f"WARNING: Test data unavailable — {e}. API will use fallback forecasts.")

    print("Asset download complete.")
