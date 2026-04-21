# EcoPulse

**Carbon-Aware Workload Scheduling Platform for Data Centers**

EcoPulse is a full-stack MLOps project that forecasts electricity grid carbon intensity and recommends optimal low-carbon execution windows for flexible data center workloads. The system combines real-time grid emissions data with weather signals, trains XGBoost forecasting models, and serves recommendations through a FastAPI backend and a React frontend dashboard — keeping human operators in the loop for all scheduling decisions.

---

## Team Contributions

| Role | Responsibility |
|------|---------------|
| Data Pipeline | Grid and weather ingestion, preprocessing, schema validation, Airflow orchestration |
| Model Pipeline | XGBoost training, hyperparameter tuning, validation, SHAP explainability |
| Bias Detection & Mitigation | Fairness analysis, bias reports, mitigated model variants |
| Inference & API | FastAPI backend, WorkloadScheduler, CarbonPredictor inference pipeline |
| Frontend | React dashboard, workload scheduler UI, landing page |
| Deployment | Docker, Kubernetes, IaC, CI/CD, GCP deployment |

---

## What EcoPulse Does

Every hour, electricity grids get cleaner or dirtier depending on how much wind, solar, and hydropower is available versus coal and gas. EcoPulse watches those shifts and tells data center operators exactly when to run their flexible compute jobs — same deadline, less carbon.

**Core flow:**
1. Ingest hourly grid carbon signals and weather data
2. Preprocess, validate, and merge into model-ready features
3. Forecast carbon intensity 1h, 12h, and 24h ahead using XGBoost
4. Detect the lowest-carbon window within an operator's priority window
5. Present the recommendation to the operator via the dashboard
6. Operator approves or denies — human always decides

---

## Repository Structure

```text
EcoPulse/
├── api/                          # FastAPI backend
│   ├── main.py                   # All endpoints — /predict, /forecast, /regions, /metrics etc.
│   └── __init__.py
├── Data_Pipeline/                # Data ingestion and preprocessing
│   ├── dags/                     # Airflow DAGs
│   │   ├── hourly_ingestion.py
│   │   └── backfill_ingestion.py
│   ├── src/                      # Pipeline source code
│   │   ├── signals_historical_ingestion.py
│   │   ├── weather_historical_ingestion.py
│   │   ├── grid_preprocessing.py
│   │   ├── weather_preprocessing.py
│   │   ├── merge_and_features.py
│   │   ├── schema_validation.py
│   │   ├── bias_mitigation.py
│   │   ├── alerts.py
│   │   └── ...
│   ├── data/processed/           # Processed parquet files (DVC tracked)
│   ├── pipeline_config/          # YAML configs for ingestion and preprocessing
│   └── docs/
│       └── BIAS_MITIGATION_REPORT.md
├── Model_Pipeline/               # ML training and validation
│   ├── src/
│   │   ├── train_xgboost.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── model_validation.py
│   │   ├── bias_detection.py
│   │   ├── drift_detection.py
│   │   ├── inference/
│   │   │   ├── predict.py        # CarbonPredictor
│   │   │   ├── feature_builder.py
│   │   │   └── green_window.py   # WorkloadScheduler, GreenWindowDetector
│   │   └── ...
│   ├── models/                   # Trained joblib model files
│   │   ├── xgboost_tuned_1h.joblib
│   │   ├── xgboost_tuned_12h.joblib
│   │   └── xgboost_tuned_24h.joblib
│   └── reports/validation/       # Confusion matrices, SHAP plots, sensitivity reports
├── frontend/                     # React + Vite frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── SchedulerPage.jsx
│   │   │   ├── AdminPages.jsx
│   │   │   └── AlertsPage.jsx
│   │   ├── components/
│   │   │   ├── LoginPage.jsx
│   │   │   ├── LogoMark.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── SimpleChart.jsx
│   │   ├── api.js                # API client calling FastAPI
│   │   ├── App.jsx               # App router
│   │   └── styles.css
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── IaC/                          # Infrastructure as Code
├── k8s/                          # Kubernetes manifests
├── monitoring/                   # Monitoring configuration
├── .github/workflows/            # CI/CD GitHub Actions
│   ├── tests.yml
│   └── model_training.yml
├── docker-compose.yaml           # Full stack Docker Compose
├── Dockerfile                    # Root Dockerfile
├── requirements.txt
└── streamlit_app.py              # Streamlit evaluation viewer
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Ingestion | Python, Electricity Maps API, Open-Meteo API |
| Orchestration | Apache Airflow 3 (Dockerized, CeleryExecutor) |
| Data Storage | Google Cloud Storage (GCS), Parquet, DVC |
| Data Validation | TensorFlow Data Validation (TFDV) |
| ML Training | XGBoost, LightGBM, scikit-learn |
| Experiment Tracking | MLflow |
| Explainability | SHAP, LIME |
| Inference API | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Cloud | Google Cloud Platform (GCP) |
| Alerts | Slack webhook, Gmail SMTP |

---

## Grid Zones Covered

| Zone ID | Region | Characteristics |
|---------|--------|----------------|
| US-MIDA-PJM | Northern Virginia Region | Mid-Atlantic grid, heavy coal/gas mix, higher carbon intensity |
| US-NW-PACW | Portland Oregon Region | Pacific Northwest, dominated by hydropower, lower carbon intensity |

---

## Machine Learning Models

EcoPulse uses tuned XGBoost models trained on 100 engineered features combining grid signals and weather data:

| Model | Forecast Horizon | File |
|-------|-----------------|------|
| XGBoost Tuned | 1 hour ahead | `xgboost_tuned_1h.joblib` |
| XGBoost Tuned | 12 hours ahead | `xgboost_tuned_12h.joblib` |
| XGBoost Tuned | 24 hours ahead | `xgboost_tuned_24h.joblib` |

Mitigated variants are also available for bias-corrected inference.

---

## FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model load status |
| GET | `/regions` | Live carbon intensity for all zones |
| GET | `/forecast/{zone}` | 24-hour carbon intensity forecast array |
| POST | `/predict` | WorkloadScheduler recommendation with CO2 savings |
| GET | `/metrics` | Model performance metrics |
| GET | `/drift` | Drift detection report |
| GET | `/shap` | SHAP feature importance |
| GET | `/alerts` | System alerts |
| GET | `/logs` | API logs |
| GET | `/users` | User list |
| POST | `/retrain` | Trigger model retraining |

**Example /predict request:**
```json
{
  "zone": "US-MIDA-PJM",
  "energy_kwh": 120,
  "runtime_hours": 4,
  "horizon": 12,
  "priority_hours": 6
}
```

**Example /predict response:**
```json
{
  "recommended_start": "2026-04-14 06:00:00",
  "hours_to_wait": 2,
  "expected_intensity_gco2_kwh": 359.2,
  "immediate_intensity_gco2_kwh": 377.15,
  "co2_saved_kg": 2.154,
  "co2_savings_pct": 4.8,
  "recommendation": "Wait 2 hours — start at 06:00. Save 2.2 kg CO2 (4.8% reduction)."
}
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker Desktop

### 1. Clone the repository
```bash
git clone https://github.com/Prathammehtaa/EcoPulse.git
cd EcoPulse
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
```

FastAPI will load the XGBoost models and test data automatically. Visit `http://localhost:8000/docs` for the Swagger UI.

### 4. Start the React frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 5. Login credentials
- **Username:** any email address
- **Password:** `ecopulse`
- **Admin access:** select Admin role with the same password

---

## Running the Full Stack with Docker

```bash
docker compose up -d
```

Services started:
- FastAPI backend
- React frontend (served via Nginx)
- Airflow webserver, scheduler, worker
- Postgres, Redis

---

## Data Pipeline Setup

### Environment Variables
Create a `.env` file in the project root:

```env
AIRFLOW_UID=50000
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET=ecopulse
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/gcp-service-account.json
ELECTRICITY_MAPS_API_KEY=your_key
OPEN_METEO_BASE_URL=https://archive-api.open-meteo.com/v1/archive
SLACK_WEBHOOK_URL=your_slack_webhook
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Initialize Airflow
```bash
docker compose up airflow-init
docker compose up -d
```

Open Airflow UI at `http://localhost:8080` and trigger the `hourly_ingestion` DAG.

---

## CI/CD

GitHub Actions workflows run automatically on push to main:

- **tests.yml** — runs unit tests for data pipeline, model pipeline, and bias mitigation
- **model_training.yml** — triggers model retraining pipeline

---

## Frontend App Flow

---
- **Landing Page** — project overview, how it works, impact stats, grid zones
- **Login** — role selection (Operator or Admin), email + password
- **Dashboard** — live carbon intensity for both regions, 24h forecast chart with auto-refresh every 60 seconds, recommended green window, forecast cards
- **Workload Scheduler** — what-if simulator with live API calls, schedule new workloads, approve or deny recommendations, view scheduled workload history
- **Admin Pages** — metrics, SHAP explainability, drift detection, API status, users, logs

---

## Bias Mitigation

EcoPulse includes a full bias detection and mitigation pipeline. The system checks for disparate model performance across grid zones and time periods, generates bias reports, and provides mitigated model variants using reweighting techniques. See `Data_Pipeline/docs/BIAS_MITIGATION_REPORT.md` for details.

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `vite: Permission denied` in Docker | Use `node:20` not `node:20-slim`, run `npm install` inside container |
| Models not loading | Check `Model_Pipeline/models/` has `xgboost_tuned_*.joblib` files |
| API returns 404 on `/` | Normal — FastAPI has no root route, use `/docs` or `/health` |
| node_modules blocking git checkout | Close all terminals, delete node_modules, then switch branches |
| Frontend shows `—` for intensity | FastAPI is not running — start uvicorn first |

---

## Project Goal

EcoPulse is designed as an augmentation system, not a fully autonomous controller. It recommends low-carbon scheduling windows while keeping human operators in control of all final execution decisions — making data centers greener without compromising operational reliability.


