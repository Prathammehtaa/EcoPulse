variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Primary GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "bucket_name" {
  description = "GCS bucket name for EcoPulse lakehouse/artifacts"
  type        = string
}

variable "artifact_registry_repo" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "ecopulse-images"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "ecopulse-gke"
}

variable "node_pool_name" {
  description = "GKE node pool name"
  type        = string
  default     = "ecopulse-node-pool"
}

variable "node_count" {
  description = "Initial node count"
  type        = number
  default     = 1
}


variable "machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-2"
}