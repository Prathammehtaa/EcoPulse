output "bucket_name" {
  value = google_storage_bucket.ecopulse_bucket.name
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.docker_repo.repository_id
}

output "artifact_registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "gke_cluster_name" {
  value = google_container_cluster.primary.name
}

output "gke_cluster_location" {
  value = google_container_cluster.primary.location
}

output "github_actions_service_account_email" {
  value = google_service_account.github_actions_sa.email
}

output "backend_service_account_email" {
  value = google_service_account.backend_sa.email
}