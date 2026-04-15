# EcoPulse

**Carbon-Responsible Data Center Scheduling Using Grid Emissions Forecasting**

EcoPulse is an MLOps project for carbon-aware scheduling of time-flexible data center workloads. The system combines hourly grid-emissions signals with weather data, builds forecasting features, and supports low-carbon execution-window recommendations for operational decision-making.

---

## Project Overview

Data centers consume a meaningful share of global electricity, but many scheduling systems still ignore how grid carbon intensity changes over time. EcoPulse addresses this by forecasting short-term carbon intensity and identifying lower-carbon execution windows so flexible workloads can be delayed or shifted with minimal operational disruption.

This project currently focuses on:
- ingesting hourly grid/carbon data
- ingesting hourly weather data
- preprocessing and aligning both sources in UTC
- merging datasets and creating model-ready features
- validating processed data with schema checks
- orchestrating the pipeline with Apache Airflow in Docker
- storing raw and processed data in Google Cloud Storage
- supporting alerting through Slack and email

---

## Current Tech Stack

- **Orchestration:** Apache Airflow 3 (Dockerized)
- **Execution model:** CeleryExecutor with Redis + Postgres
- **Language:** Python 3.11+
- **Storage:** Google Cloud Storage (GCS)
- **Data format:** JSON, CSV, Parquet
- **Validation:** TensorFlow Data Validation (TFDV) / schema validation utilities
- **Monitoring & alerts:** Slack webhook, Gmail SMTP email notifications
- **Containerization:** Docker + Docker Compose

---

## Data Sources

- **Electricity Maps** for grid/carbon-intensity and energy signals
- **Open-Meteo** for weather history and related meteorological signals
- **NOAA** as an additional public weather source referenced in project scoping

---

## Current Repository Structure

```text
EcoPulse/
├── Data_Pipeline/
│   ├── dags/
│   │   ├── hourly_ingestion.py
│   │   ├── backfill_ingestion.py
│   │   ├── daily_pipeline.py              # add/update as daily DAG is finalized
│   │   └── ...
│   ├── src/
│   │   ├── signals_historical_ingestion.py
│   │   ├── daily_grid_ingestion.py        # optional, if used later
│   │   ├── daily_weather_ingestion.py     # optional, if used later
│   │   ├── weather_ingestion.py
│   │   ├── preprocessing_*.py
│   │   ├── feature_engineering.py
│   │   ├── merge_and_features.py
│   │   ├── schema_validation_task.py
│   │   ├── schema_validation_module.py
│   │   └── ...
│   ├── config/
│   │   ├── airflow.cfg
│   │   ├── *.yaml
│   │   ├── *.yml
│   │   └── *.json
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── features/
│   ├── logs/
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

> Update the filenames above if your local repo uses slightly different names. The structure reflects the project as currently discussed and implemented.

---

## Pipeline Flow

### 1. Hourly ingestion
The hourly pipeline fetches:
- grid/carbon signals
- weather signals

### 2. Preprocessing
The raw feeds are cleaned and standardized:
- timestamps normalized to UTC
- schema / field cleanup
- region and zone validation
- missing-value handling

### 3. Merge + feature engineering
Processed grid and weather datasets are aligned on hourly timestamps and region, then transformed into model-ready features.

### 4. Schema validation
The merged feature dataset is validated before downstream model training or inference use.

### 5. Notifications
Pipeline success/failure notifications are sent through Slack and email.

---

## Expected Data Layout

### In GCS bucket
```text
gs://ecopulse/
├── raw/
│   ├── grid_signals/
│   └── weather/
├── processed/
└── features/
```

### Local project data folders
```text
Data_Pipeline/data/
├── raw/
├── processed/
└── features/
```

---

## Prerequisites

Before deployment, make sure the following are installed and configured:

- Docker Desktop
- Docker Compose
- Python 3.11+
- A GCP service account with access to the `ecopulse` bucket
- Electricity Maps API key
- Internet access for external APIs

You should also have:
- a Slack webhook URL for alerts
- Gmail SMTP credentials or an app password for email notifications

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
AIRFLOW_PROJ_DIR=.
AIRFLOW_UID=50000

GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET=ecopulse
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/gcp-service-account.json

ELECTRICITY_MAPS_API_KEY=your_electricity_maps_key
OPEN_METEO_BASE_URL=https://archive-api.open-meteo.com/v1/archive

SLACK_WEBHOOK_URL=your_slack_webhook

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_MAIL_FROM=your_email@gmail.com
```

---

## Docker Volume Mounts

A working Docker Compose setup should mount the Airflow folders like this:

```yaml
volumes:
  - ${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/dags:/opt/airflow/dags
  - ${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/logs:/opt/airflow/logs
  - ${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/config:/opt/airflow/config
  - ${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/src:/opt/airflow/src
  - ${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/data:/opt/airflow/data
```

This ensures Airflow can access DAGs, logs, config, source code, and data paths correctly inside the containers.

---

## Deployment Steps

### Step 1: Clone the repository
```bash
git clone <your-repo-url>
cd EcoPulse
```

### Step 2: Create required local folders
```bash
mkdir -p Data_Pipeline/dags
mkdir -p Data_Pipeline/logs
mkdir -p Data_Pipeline/config
mkdir -p Data_Pipeline/src
mkdir -p Data_Pipeline/data/raw
mkdir -p Data_Pipeline/data/processed
```

On Windows PowerShell, create them manually if needed.

### Step 3: Add credentials
Place your GCP service account JSON inside the config folder, for example:

```text
Data_Pipeline/config/gcp-service-account.json
```

Make sure the path matches `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.

### Step 4: Build Airflow image
```bash
docker compose build
```

If you install Python dependencies through the Dockerfile, rebuild after every dependency change.

### Step 5: Initialize Airflow
```bash
docker compose up airflow-init
```

This initializes the Airflow metadata database and creates required directories.

### Step 6: Start all services
```bash
docker compose up -d
```

Typical services:
- airflow-webserver
- airflow-scheduler
- airflow-worker
- airflow-triggerer
- postgres
- redis

### Step 7: Open Airflow UI
Open:

```text
http://localhost:8080
```

Log in using the credentials configured in your Compose setup.

### Step 8: Verify DAGs are loaded
Confirm your DAGs appear in the Airflow UI, such as:
- `hourly_ingestion`
- `backfill_ingestion`
- `daily_pipeline` (if added)

### Step 9: Trigger the pipeline
You can run DAGs from the Airflow UI or CLI.

Example CLI:
```bash
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-scheduler airflow dags trigger hourly_ingestion
```

### Step 10: Validate outputs
Check:
- raw data written to GCS
- processed/merged outputs created
- feature tables generated
- schema validation completed
- Slack/email notifications delivered

---

## Running the Main Pipelines

### Hourly DAG
Use this for recurring ingestion and feature generation from live/hourly feeds.

**Typical flow:**
```text
start
  ├── grid_pipeline
  ├── weather_pipeline
  └── join_before_merge
        ↓
   merge_and_features
        ↓
 schema_validation_tfdv
        ↓
   slack_success
        ↓
 notify_success_email
        ↓
       end
```

### Backfill DAG
Use this to collect historical data over a defined time range for training and evaluation.

### Daily DAG
Use this for daily aggregation or daily model-ready outputs once the daily workflow is finalized. Since your current setup uses data populated by the hourly ingestion, the daily DAG should consume already-ingested hourly data rather than re-pulling from scratch.

---

## Recommended Deployment Order

1. Set up Docker, `.env`, and credentials
2. Confirm GCS access works
3. Run Airflow init
4. Start the full stack
5. Test one ingestion task manually
6. Test hourly DAG end to end
7. Test Slack/email alerts
8. Run historical backfill
9. Enable scheduled DAG execution

---

## Common Issues

### 1. Airflow cannot see dags/logs/config
Cause: incorrect Docker volume mount path.

Fix: ensure the left-hand side includes the missing `/`:

```yaml
${AIRFLOW_PROJ_DIR:-.}/Data_Pipeline/dags:/opt/airflow/dags
```

not

```yaml
${AIRFLOW_PROJ_DIR:-.}Data_Pipeline/dags:/opt/airflow/dags
```

### 2. SMTP SSL wrong version number
Cause: mismatch between SSL and TLS settings.

Fix:
- use `smtp.gmail.com`
- use port `587` with STARTTLS, or port `465` with SSL
- do not mix both modes

### 3. Airflow imports fail
Cause: files in `src/` are not mounted or not on the Python path.

Fix:
- mount `Data_Pipeline/src:/opt/airflow/src`
- add the source directory to the container path if needed

### 4. GCS authentication issues
Cause: credentials file missing inside container or invalid permissions.

Fix:
- confirm JSON file exists inside `/opt/airflow/config/`
- confirm bucket permissions for the service account

### 5. Docker creates unexpected folders
Cause: bind-mount target exists in Compose but local source path is missing or misspelled.

Fix:
- create the local directories first
- recheck path names carefully

---

## Future Extensions

Planned or possible next additions:
- model training pipeline
- MLflow experiment tracking and registry
- drift detection with Evidently
- FastAPI inference service
- daily recommendation generation
- conservative fallback policy for missing data or drift
- workload simulator / synthetic workload generation for scheduling evaluation

---

## Project Goal

EcoPulse is designed as an augmentation system, not a fully autonomous controller. It recommends low-carbon scheduling windows while allowing operators to retain control over final execution decisions.

---


