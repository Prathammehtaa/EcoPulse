# EcoPulse Data Pipeline

> **Comprehensive data ingestion, preprocessing, bias detection, and versioning pipeline for carbon intensity forecasting.**

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Data Sources](#data-sources)
- [Running the Pipeline](#running-the-pipeline)
- [Data Versioning (DVC)](#data-versioning-dvc)
- [Bias Detection & Mitigation](#bias-detection--mitigation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The EcoPulse Data Pipeline collects, processes, and versions electricity grid and weather data for carbon intensity forecasting. The pipeline is designed for reproducibility, automated testing, and bias-free model training.

### Pipeline Summary

| Metric | Value |
|--------|-------|
| **Grid Zones** | 3 (US-NE-ISNE, US-NW-PACW, US-MIDA-PJM) |
| **Weather Locations** | 3 (Boston, N. Virginia, Portland OR) |
| **Total Records** | 26,352 hourly observations |
| **Balanced Dataset** | 65,105 samples (after mitigation) |
| **Test Coverage** | 30 unit tests |

---

## Pipeline Stages

```
[1] DATA INGESTION
    • Fetch grid signals from Electricity Maps API
    • Fetch weather data from Open-Meteo API
    • Store raw JSON responses in GCS
    • Scripts: signals_historical_ingestion.py, weather_historical_ingestion.py
    
              ↓

[2] DATA STAGING
    • Download data from GCS bucket
    • Convert to Parquet format
    • Combine multi-zone grid data
    • Combine multi-location weather data
    • Script: stage_data.py
    
              ↓

[3] PREPROCESSING
    • Clean and normalize data
    • Handle DST transitions (UTC conversion)
    • Create temporal features (hour, day, month, season)
    • Merge grid and weather data
    • Scripts: grid_preprocessing.py, weather_preprocessing.py, merge_and_features.py
    
              ↓

[4] SCHEMA VALIDATION
    • Validate data types and ranges
    • Check for missing values
    • Detect schema anomalies
    • Script: schema_validation.py
    
              ↓

[5] BIAS DETECTION
    • Analyze distribution across 6 slices
    • Calculate imbalance ratios
    • Generate visualizations
    • Script: tfdv_bias_analysis.py, bias_visualization.py
    
              ↓

[6] BIAS MITIGATION
    • Apply random oversampling
    • Create stratified train/val/test splits
    • Script: bias_mitigation.py
    
              ↓

[7] DATA VERSIONING
    • Track datasets with DVC
    • Push to GCS remote storage
    • Commands: dvc add, dvc push
```

---

## Data Sources

### Electricity Maps API

| Field | Description |
|-------|-------------|
| `carbonIntensity` | gCO2eq/kWh |
| `fossilFuelPercentage` | % from fossil fuels |
| `renewablePercentage` | % from renewables |
| `powerConsumptionTotal` | Total consumption (MW) |
| `powerProductionTotal` | Total production (MW) |
| Power breakdown | By source (wind, solar, hydro, nuclear, etc.) |

**Zones Covered:**
- `US-NE-ISNE` - New England ISO
- `US-NW-PACW` - Pacific Northwest
- `US-MIDA-PJM` - Mid-Atlantic PJM

### Open-Meteo API

| Field | Description |
|-------|-------------|
| `temperature_2m` | Temperature at 2m (°C) |
| `relative_humidity_2m` | Relative humidity (%) |
| `wind_speed_10m` | Wind speed at 10m (km/h) |
| `wind_direction_10m` | Wind direction (°) |
| `shortwave_radiation` | Solar radiation (W/m²) |
| `precipitation` | Precipitation (mm) |
| `cloud_cover` | Cloud cover (%) |

**Locations Covered:**
- Boston, Massachusetts
- Northern Virginia
- Portland, Oregon

---

## Running the Pipeline

### Option 1: Run Individual Scripts

```bash
cd Data_Pipeline

# Step 1: Ingest data (if not using cached data)
python src/signals_historical_ingestion.py
python src/weather_historical_ingestion.py

# Step 2: Stage data from GCS
python src/stage_data.py

# Step 3: Preprocess
python src/grid_preprocessing.py
python src/weather_preprocessing.py
python src/merge_and_features.py

# Step 4: Validate schema
python src/schema_validation.py

# Step 5: Bias analysis
python src/tfdv_bias_analysis.py
python src/bias_visualization.py

# Step 6: Bias mitigation
python src/bias_mitigation.py
```

### Option 2: Use DVC Pipeline

```bash
cd Data_Pipeline

# Run entire pipeline
dvc repro

# Run specific stage
dvc repro bias_analysis
```

### Option 3: Airflow DAG

```bash
# Start Airflow
airflow webserver --port 8080
airflow scheduler

# Trigger DAG from UI at http://localhost:8080
```

---

## Data Versioning (DVC)

### Configuration

```yaml
# .dvc/config
[core]
    remote = gcs_remote
['remote "gcs_remote"']
    url = gs://ecopulse-kapish/dvc-store
```

### Tracked Files

| File | Description | Size |
|------|-------------|------|
| `data/stage/grid_signals_all_zones.parquet` | Combined grid data | ~1.7 MB |
| `data/stage/weather_all_locations.parquet` | Combined weather | ~400 KB |
| `data/processed/train_balanced.csv` | Training set | ~4.5 MB |
| `data/processed/val_balanced.csv` | Validation set | ~1 MB |
| `data/processed/test_balanced.csv` | Test set | ~1 MB |

### Commands

```bash
# Check status
dvc status

# Pull data from remote
dvc pull

# Push data to remote
dvc push

# View pipeline DAG
dvc dag
```

### DVC Pipeline (`dvc.yaml`)

```yaml
stages:
  stage_data:
    cmd: python src/stage_data.py
    deps:
      - src/stage_data.py
      - config/ingestion_config.yaml
    outs:
      - data/stage/grid_signals_all_zones.parquet
      - data/stage/weather_all_locations.parquet

  bias_analysis:
    cmd: python src/tfdv_bias_analysis.py
    deps:
      - src/tfdv_bias_analysis.py
      - data/stage/grid_signals_all_zones.parquet
    outs:
      - reports/tfdv_bias/

  bias_mitigation:
    cmd: python src/bias_mitigation.py
    deps:
      - src/bias_mitigation.py
      - data/stage/grid_signals_all_zones.parquet
    outs:
      - data/processed/train_balanced.csv
      - data/processed/val_balanced.csv
      - data/processed/test_balanced.csv
```

---

## Bias Detection & Mitigation

### Slicing Dimensions

| Slice | Description | Categories |
|-------|-------------|------------|
| `carbon_intensity_bucket` | Carbon intensity level | Very Low, Low, Medium, High, Very High |
| `hour_of_day` | Hour (0-23) | 24 categories |
| `day_of_week` | Day of week | Monday - Sunday |
| `month` | Month of year | January - December |
| `season` | Season | Winter, Spring, Summer, Fall |
| `is_weekend` | Weekend indicator | True, False |

### Detection Results

| Slice | Imbalance Ratio | Severity | Action |
|-------|-----------------|----------|--------|
| `carbon_intensity_bucket` | **16.12x** | 🔴 SEVERE | Mitigation required |
| `hour_of_day` | 1.00x | 🟢 LOW | None |
| `day_of_week` | 1.00x | 🟢 LOW | None |
| `month` | 1.00x | 🟢 LOW | None |
| `season` | 1.00x | 🟢 LOW | None |
| `is_weekend` | 2.50x | 🟡 MODERATE | Acceptable |

### Mitigation Results

| Metric | Before | After |
|--------|--------|-------|
| Total Samples | 26,352 | 65,105 |
| Imbalance Ratio | 16.12x | 1.00x |
| Severity | 🔴 SEVERE | 🟢 LOW |

### Output Files

| File | Location |
|------|----------|
| Distribution CSVs | `reports/tfdv_bias/distribution_*.csv` |
| Statistics CSVs | `reports/tfdv_bias/stats_*.csv` |
| Visualization PNGs | `reports/tfdv_bias/plots/*.png` |
| Bias Summary | `reports/tfdv_bias/bias_summary.json` |
| Mitigation Report | `reports/bias_mitigation/mitigation_report.json` |
| Full Documentation | `docs/BIAS_MITIGATION_REPORT.md` |

---

## Testing

### Running Tests

```bash
cd Data_Pipeline

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bias_mitigation.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `test_bias_mitigation.py` | 18 | ✅ Passing |
| `test_tfdv_bias_analysis.py` | 12 | ✅ Passing |

### Test Categories

- **Carbon Bucket Creation**: Verify correct bucket assignment
- **Severity Classification**: Test imbalance thresholds
- **Oversampling**: Validate class balancing
- **Stratified Splitting**: Ensure distribution preservation
- **Edge Cases**: Empty data, single class, NaN handling

---

## Configuration

### `config/ingestion_config.yaml`

```yaml
electricity_maps:
  api_key: ${ELECTRICITY_MAPS_API_KEY}
  zones:
    - US-NE-ISNE
    - US-NW-PACW
    - US-MIDA-PJM

open_meteo:
  locations:
    - name: boston_area
      latitude: 42.3601
      longitude: -71.0589
    - name: northern_virginia
      latitude: 38.9072
      longitude: -77.0369
    - name: portland_oregon
      latitude: 45.5152
      longitude: -122.6784

gcs:
  bucket: ecopulse-kapish
  raw_prefix: raw/
  stage_prefix: stage/
```

### Environment Variables

```bash
# Required
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
ELECTRICITY_MAPS_API_KEY=your_api_key

# Optional
GCS_BUCKET_NAME=ecopulse-kapish
```

---

## Code Style & Modularity

### PEP 8 Compliance

All code follows Python's PEP 8 style guidelines:

```bash
# Check code style
pip install flake8
flake8 src/ --max-line-length=100

# Auto-format code
pip install black
black src/
```

### Modular Design Principles

Each component is designed to be **independent and reusable**:

```
src/
├── ingestion/                    # Data fetching (independent)
│   ├── signals_historical_ingestion.py
│   └── weather_historical_ingestion.py
├── preprocessing/                # Data cleaning (independent)
│   ├── grid_preprocessing.py
│   └── weather_preprocessing.py
├── validation/                   # Quality checks (independent)
│   └── schema_validation.py
├── bias/                         # Bias analysis (independent)
│   ├── tfdv_bias_analysis.py
│   ├── bias_visualization.py
│   └── bias_mitigation.py
└── utils/                        # Shared utilities
    ├── alerts.py
    └── stage_data.py
```

### Modularity Benefits

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Each script does one thing |
| **Reusability** | Functions can be imported elsewhere |
| **Testability** | Each module has its own test file |
| **Maintainability** | Easy to update without breaking others |

### Example: Modular Function

```python
# src/bias_mitigation.py

def create_carbon_bucket(value):
    """Assign carbon intensity to bucket. Can be reused anywhere."""
    if pd.isna(value):
        return 'Unknown'
    elif value < 100:
        return 'Very Low'
    elif value < 200:
        return 'Low'
    # ... etc

def random_oversample(df, target_column):
    """Generic oversampling function. Reusable for any dataset."""
    # ... implementation

def stratified_split(df, target_column, train_ratio=0.7):
    """Generic stratified split. Reusable for any dataset."""
    # ... implementation
```

---

## Error Handling

### Strategy

All scripts implement robust error handling:

```python
import logging

logger = logging.getLogger(__name__)

def fetch_data_from_api(url):
    """Fetch data with error handling and retries."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                logger.warning("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
            else:
                logger.error(f"HTTP Error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    raise Exception(f"Failed after {max_retries} attempts")
```

### Error Types Handled

| Error Type | Handling | Recovery |
|------------|----------|----------|
| **API Timeout** | Retry with backoff | 3 retries, then fail |
| **Rate Limiting (429)** | Wait and retry | 60 second delay |
| **File Not Found** | Log and alert | Use cached data if available |
| **Schema Violation** | Halt pipeline | Send alert, require fix |
| **Memory Error** | Chunk processing | Process in smaller batches |
| **GCS Connection** | Retry | 3 retries with exponential backoff |

### Try-Except Pattern

```python
def run_pipeline_stage(stage_name, function, *args):
    """Wrapper for error handling in pipeline stages."""
    try:
        logger.info(f"Starting {stage_name}...")
        result = function(*args)
        logger.info(f"Completed {stage_name} successfully")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"{stage_name} failed: File not found - {e}")
        send_alert(f"Pipeline failed at {stage_name}: {e}", severity='critical')
        raise
        
    except ValueError as e:
        logger.error(f"{stage_name} failed: Invalid data - {e}")
        send_alert(f"Data validation error in {stage_name}: {e}", severity='warning')
        raise
        
    except Exception as e:
        logger.error(f"{stage_name} failed: Unexpected error - {e}")
        send_alert(f"Unexpected error in {stage_name}: {e}", severity='critical')
        raise
```

---

## Environment Setup

### Option 1: requirements.txt

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
# Core
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0

# Cloud & Storage
google-cloud-storage>=2.0.0
dvc>=3.0.0
dvc-gs>=2.0.0

# Pipeline
apache-airflow>=2.7.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Code Quality
flake8>=6.0.0
black>=23.0.0
```

### Option 2: environment.yml (Conda)

```yaml
# environment.yml
name: ecopulse
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pyarrow>=14.0.0
  - matplotlib>=3.7.0
  - pytest>=7.0.0
  - pip
  - pip:
    - google-cloud-storage>=2.0.0
    - dvc>=3.0.0
    - dvc-gs>=2.0.0
    - apache-airflow>=2.7.0
```

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate ecopulse
```

---

## Logging

### Implementation

All pipeline scripts use Python's `logging` module for tracking progress and debugging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Log Levels

| Level | Usage |
|-------|-------|
| `INFO` | Normal pipeline progress |
| `WARNING` | Non-critical issues (missing optional fields) |
| `ERROR` | Failed operations (API errors, file not found) |
| `DEBUG` | Detailed debugging information |

### Log Files

```
Data_Pipeline/
└── logs/
    ├── pipeline.log           # Main pipeline log
    ├── ingestion.log          # Data ingestion logs
    ├── preprocessing.log      # Preprocessing logs
    └── airflow/               # Airflow task logs
```

---

## Anomaly Detection & Alerts

### Anomaly Types Detected

| Anomaly | Detection Method | Action |
|---------|------------------|--------|
| Missing values > 5% | Schema validation | Alert + Log |
| Outliers (>3 std dev) | Statistical analysis | Flag in report |
| Schema violations | TFDV validation | Pipeline halt |
| API failures | HTTP status codes | Retry + Alert |
| Data freshness | Timestamp check | Warning |

### Alert Configuration

```python
# src/alerts.py
def send_alert(message, severity='warning'):
    """Send alert via email or Slack."""
    if severity == 'critical':
        send_email(ALERT_EMAIL, message)
        send_slack(SLACK_WEBHOOK, message)
    else:
        logger.warning(message)
```

### Alert Triggers

- ❌ Pipeline failure
- ⚠️ Data quality below threshold
- ⚠️ Missing data for >2 hours
- ❌ Schema validation failure
- ⚠️ Unusual carbon intensity values

---

## Airflow DAG Details

### DAG Structure

```python
# dags/backfill_ingestion.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ecopulse',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ecopulse_data_pipeline',
    default_args=default_args,
    description='EcoPulse carbon intensity data pipeline',
    schedule_interval='@hourly',
    catchup=False,
) as dag:

    ingest_grid = PythonOperator(
        task_id='ingest_grid_signals',
        python_callable=ingest_grid_data,
    )

    ingest_weather = PythonOperator(
        task_id='ingest_weather',
        python_callable=ingest_weather_data,
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_preprocessing,
    )

    validate_schema = PythonOperator(
        task_id='validate_schema',
        python_callable=validate_data_schema,
    )

    bias_check = PythonOperator(
        task_id='bias_detection',
        python_callable=run_bias_analysis,
    )

    version_data = PythonOperator(
        task_id='version_with_dvc',
        python_callable=dvc_push,
    )

    # Task Dependencies
    [ingest_grid, ingest_weather] >> preprocess >> validate_schema >> bias_check >> version_data
```

### Task Flow Diagram

```
    ┌─────────────────┐     ┌─────────────────┐
    │  ingest_grid    │     │ ingest_weather  │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ↓
              ┌─────────────────────┐
              │    preprocess       │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │  validate_schema    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   bias_detection    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │  version_with_dvc   │
              └─────────────────────┘
```

### Running Airflow

```bash
# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &

# Access UI
open http://localhost:8080
```

### Monitoring with Gantt Chart

Access Airflow UI → DAGs → ecopulse_data_pipeline → Graph/Gantt

The Gantt chart helps identify:
- ⏱️ Slow tasks (bottlenecks)
- 🔄 Parallelization opportunities
- ❌ Failed task retries

---

## Pipeline Optimization

### Identified Bottlenecks

| Task | Avg Duration | Optimization |
|------|--------------|--------------|
| `ingest_grid` | 2-3 min | Parallel API calls |
| `ingest_weather` | 1-2 min | Batch requests |
| `preprocess` | 30 sec | Chunked processing |

### Optimization Techniques

1. **Parallel Ingestion**: Grid and weather data fetched simultaneously
2. **Batch Processing**: Process data in chunks to reduce memory
3. **Caching**: Cache API responses to avoid redundant calls
4. **Incremental Updates**: Only process new data since last run

---

## Troubleshooting

### DVC Issues

**Problem**: `dvc pull` fails with authentication error

```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Check remote configuration
dvc remote list -v

# Test GCS access
gsutil ls gs://ecopulse-kapish/
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Memory Issues

**Problem**: Out of memory when processing large files

```bash
# Process in chunks (modify stage_data.py)
chunk_size = 10000

# Or increase available memory
# Docker: Increase memory limit in settings
```

### API Rate Limits

**Problem**: `429 Too Many Requests`

```bash
# Add delay between requests
import time
time.sleep(1)  # 1 second delay

# Or use API key for higher limits
```

---

## File Structure

```
Data_Pipeline/
├── config/
│   ├── ingestion_config.yaml
│   └── preprocessing_config.yaml
├── dags/
│   └── backfill_ingestion.py
├── data/
│   ├── raw/
│   ├── stage/
│   │   ├── grid_signals_all_zones.parquet
│   │   ├── grid_signals_US-NE-ISNE.parquet
│   │   ├── grid_signals_US-MIDA-PJM.parquet
│   │   ├── grid_signals_US-NW-PACW.parquet
│   │   ├── weather_all_locations.parquet
│   │   └── *.dvc
│   └── processed/
│       ├── train_balanced.csv
│       ├── val_balanced.csv
│       └── test_balanced.csv
├── docs/
│   └── BIAS_MITIGATION_REPORT.md
├── reports/
│   ├── tfdv_bias/
│   │   ├── distribution_*.csv
│   │   ├── stats_*.csv
│   │   ├── bias_summary.json
│   │   └── plots/
│   └── bias_mitigation/
│       └── mitigation_report.json
├── src/
│   ├── signals_historical_ingestion.py
│   ├── weather_historical_ingestion.py
│   ├── stage_data.py
│   ├── grid_preprocessing.py
│   ├── weather_preprocessing.py
│   ├── merge_and_features.py
│   ├── schema_validation.py
│   ├── tfdv_bias_analysis.py
│   ├── bias_visualization.py
│   ├── bias_mitigation.py
│   └── alerts.py
├── tests/
│   ├── test_bias_mitigation.py
│   └── test_tfdv_bias_analysis.py
└── dvc.yaml
```

---

**Last Updated**: February 2026
