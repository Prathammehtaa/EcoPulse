"""
Downloads models and test data from GCS at container startup.
Uses google-cloud-storage (already in requirements.txt) — no gsutil needed.
"""
import os
from google.cloud import storage

BUCKET      = os.environ.get("GCS_BUCKET", "ecopulse-lakehouse")
MODELS_PFX  = os.environ.get("MODELS_GCS_PREFIX", "models")
DATA_PFX    = os.environ.get("DATA_GCS_PREFIX", "processed")

MODELS_DIR  = "/app/Model_Pipeline/models"
DATA_DIR    = "/app/Data_Pipeline/data/processed"

def download_prefix(client, bucket_name, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        print(f"  WARNING: No files found at gs://{bucket_name}/{prefix}")
        return
    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip folder placeholders
        filename = os.path.basename(blob.name)
        dest = os.path.join(local_dir, filename)
        print(f"  Downloading {blob.name} → {dest}")
        blob.download_to_filename(dest)

def download_file(client, bucket_name, blob_name, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    print(f"  Downloading {blob_name} → {dest_path}")
    blob.download_to_filename(dest_path)

if __name__ == "__main__":
    client = storage.Client()

    print(f"Downloading models from gs://{BUCKET}/{MODELS_PFX}/")
    download_prefix(client, BUCKET, MODELS_PFX, MODELS_DIR)

    print(f"Downloading test split from gs://{BUCKET}/{DATA_PFX}/test_split.parquet")
    download_file(
        client, BUCKET,
        f"{DATA_PFX}/test_split.parquet",
        f"{DATA_DIR}/test_split.parquet"
    )

    print("Assets downloaded successfully.")