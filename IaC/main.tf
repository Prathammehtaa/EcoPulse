locals {
  required_apis = [
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ]
}

# -------------------------------------------------
# Enable required GCP APIs
# -------------------------------------------------
resource "google_project_service" "required" {
  for_each = toset(local.required_apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# -------------------------------------------------
# GCS bucket for EcoPulse lakehouse / artifacts
# -------------------------------------------------
resource "google_storage_bucket" "ecopulse_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  depends_on = [google_project_service.required]
}

# -------------------------------------------------
# Artifact Registry for Docker images
# -------------------------------------------------
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.artifact_registry_repo
  description   = "Docker images for EcoPulse frontend and backend"
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

# -------------------------------------------------
# GKE Cluster
# -------------------------------------------------
resource "google_container_cluster" "primary" {
  name           = var.cluster_name
  location       = var.region
  node_locations = ["us-central1-a"]

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  # Disable basic auth
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Release channel (replaces pinned cluster_version)
  release_channel {
    channel = "REGULAR"
  }

  # Logging & Monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "STORAGE",
      "POD",
      "DEPLOYMENT",
      "STATEFULSET",
      "DAEMONSET",
      "HPA",
      "CADVISOR",
      "KUBELET"
    ]
    managed_prometheus {
      enabled = true
    }
  }

  # Networking
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}

  # Addons
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Security
  enable_shielded_nodes = true

  security_posture_config {
    mode               = "BASIC"
    vulnerability_mode = "VULNERABILITY_DISABLED"
  }

  binary_authorization {
    evaluation_mode = "DISABLED"
  }

  # No intra-node visibility
  enable_intranode_visibility = false

  depends_on = [google_project_service.required]
}

# -------------------------------------------------
# GKE Node Pool
# -------------------------------------------------
resource "google_container_node_pool" "primary_nodes" {
  name           = var.node_pool_name
  location       = var.region
  cluster        = google_container_cluster.primary.name
  node_count     = var.node_count
  node_locations = ["us-central1-a"]

  # Auto upgrade and repair
  management {
    auto_upgrade = true
    auto_repair  = true
  }

  # Surge upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  node_config {
    machine_type = var.machine_type
    image_type   = "COS_CONTAINERD"
    disk_type    = "pd-balanced"
    disk_size_gb = 50

    service_account = google_service_account.backend_sa.email

    # Scopes matching the gcloud command
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Disable legacy endpoints
    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    labels = {
      env     = "single"
      project = "ecopulse"
    }
  }
}

# -------------------------------------------------
# Static external IP for frontend LoadBalancer
# -------------------------------------------------
resource "google_compute_address" "frontend_ip" {
  name   = "ecopulse-frontend-ip"
  region = var.region

  depends_on = [google_project_service.required]
}
