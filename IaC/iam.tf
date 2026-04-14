# -------------------------------------------------
# GitHub Actions Service Account
# Used for CI/CD: build/push images and deploy to GKE
# -------------------------------------------------
resource "google_service_account" "github_actions_sa" {
  account_id   = "ecopulse-github-actions"
  display_name = "EcoPulse GitHub Actions Deployer"
}

resource "google_project_iam_member" "github_actions_roles" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/container.developer",
    "roles/iam.serviceAccountUser",
    "roles/storage.admin"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_actions_sa.email}"
}

# -------------------------------------------------
# Backend Service Account
# Used by backend runtime in GKE to read/write objects in GCS
# -------------------------------------------------
resource "google_service_account" "backend_sa" {
  account_id   = "ecopulse-backend"
  display_name = "EcoPulse Backend Service Account"
}

resource "google_storage_bucket_iam_member" "backend_bucket_access" {
  bucket = google_storage_bucket.ecopulse_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.backend_sa.email}"
}

# -------------------------------------------------
# Airflow Service Account
# Used by local Airflow pipeline to read/write GCS
# Removed artifactregistry.admin - Airflow has no reason to manage images
# Removed storage.objectAdmin duplicate - storage.admin is a superset
# -------------------------------------------------
resource "google_service_account" "airflow_sa" {
  account_id   = "ecopulse-airflow"
  display_name = "EcoPulse Airflow Service Account"
}

resource "google_project_iam_member" "airflow_roles" {
  for_each = toset([
    "roles/storage.admin"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.airflow_sa.email}"
}
