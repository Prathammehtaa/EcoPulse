terraform {
  backend "gcs" {
    bucket = "ecopulse-mlops-pratham-tf-state"
    prefix = "terraform/state"
  }
}
