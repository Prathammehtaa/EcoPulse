terraform {
  backend "gcs" {
    bucket = "ecopulse-tf-state"
    prefix = "terraform/state"
  }
}
