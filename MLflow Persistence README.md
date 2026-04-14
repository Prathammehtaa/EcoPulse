# MLflow Log Persistence — GitHub Actions

## Problem
Previously, MLflow was using a local SQLite database on the GitHub Actions runner.
This meant **all experiment history was lost after every run** because the runner is a fresh machine each time.

## Fix
Added artifact upload/download steps to the GitHub Actions workflow (`.github/workflows/tests.yml`):
- **At the start of each run** → the MLflow DB is downloaded from the previous run
- **At the end of each run** → the MLflow DB is uploaded so the next run can use it

This persists experiment history across runs for up to **90 days**.

## How It Works
```
Run 1 → No DB yet (skipped) → trains → saves DB ✅
Run 2 → Downloads DB from Run 1 → trains → saves DB ✅
Run 3 → Downloads DB from Run 2 → trains → saves DB ✅
```

## Changes Made

### `.github/workflows/tests.yml`

**1. Restore MLflow DB (at the start)**
```yaml
- name: Restore MLflow DB
  uses: actions/download-artifact@v4
  with:
    name: mlflow-db
    path: ./mlflow
  continue-on-error: true
```

**2. Save MLflow DB (at the end)**
```yaml
- name: Save MLflow DB
  uses: actions/upload-artifact@v4
  with:
    name: mlflow-db
    path: ./mlflow
    retention-days: 90
```

## Notes
- `continue-on-error: true` ensures the first run doesn't fail when no DB exists yet
- `retention-days: 90` is the maximum allowed on GitHub free plan
- The MLflow tracking URI is set to `sqlite:///mlflow/mlflow.db` in the workflow
