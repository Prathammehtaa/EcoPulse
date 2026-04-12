# MLflow Log Persistence — GitHub Actions


## Problem
Previously, MLflow was using a local SQLite database on the GitHub Actions runner.
This meant all experiment history was lost after every run.

## Fix
Added artifact upload/download steps to tests.yml:
- At the start of each run: MLflow DB is restored from previous run
- At the end of each run: MLflow DB is saved for the next run
- History persists for 90 days

## How It Works
Run 1 → No DB yet (skipped) → trains → saves DB
Run 2 → Downloads DB from Run 1 → trains → saves DB
Run 3 → Downloads DB from Run 2 → trains → saves DB

## Notes
- continue-on-error: true ensures first run does not fail
- retention-days: 90 is the maximum on GitHub free plan
- MLFLOW_TRACKING_URI is set to sqlite:///mlflow/mlflow.db


