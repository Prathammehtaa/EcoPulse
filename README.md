# EcoPulse: Carbon-Responsible Data Center Scheduling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.7+-orange.svg)](https://airflow.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DVC](https://img.shields.io/badge/DVC-3.0+-purple.svg)](https://dvc.org/)

> **An MLOps pipeline for carbon-aware workload scheduling using grid emissions forecasting. Reduces data center carbon footprint by shifting flexible workloads to low-carbon electricity periods.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)

---

## Documentation

Comprehensive documentation is available in the project directories:

- **[Data Pipeline README](Data_Pipeline/README.md)** - Detailed pipeline stages, scripts, and execution guide
- **[Bias Detection & Mitigation Report](Data_Pipeline/docs/BIAS_MITIGATION_REPORT.md)** - Bias analysis methodology, results, and mitigation strategies
- **[DVC Configuration](Data_Pipeline/dvc.yaml)** - Data versioning pipeline definition

---

## Overview

**EcoPulse** is a production-ready MLOps pipeline that enables carbon-aware workload scheduling for data centers. Data centers consume approximately 1–1.5% of global electricity, yet current scheduling practices ignore the temporal variability in grid carbon intensity.

### Key Objectives

- **Carbon Intensity Forecasting**: Predict grid carbon intensity 1-24 hours ahead
- **Multi-Zone Coverage**: Ingest data from 3 US electricity grid zones
- **Weather Integration**: Correlate weather patterns with renewable energy availability
- **Bias-Free Training**: Detect and mitigate data biases for fair model training
- **Reproducible Pipeline**: DVC versioning with cloud storage integration

---

## Features

### Data Ingestion
- **Electricity Maps API**: Carbon intensity, grid signals, power breakdown
- **Open-Meteo API**: Temperature, wind, solar radiation, precipitation
- **Multi-Zone Support**: US-NE-ISNE, US-NW-PACW, US-MIDA-PJM

### Data Processing
- **Schema Validation**: Automated quality checks with TFDV
- **Preprocessing**: Cleaning, normalization, DST handling
- **Feature Engineering**: Temporal features, rolling aggregates

### Bias Detection & Mitigation
- **6 Dimensional Slices**: Hour, day, month, season, weekend, carbon bucket
- **Automated Detection**: Imbalance ratio and severity classification
- **Mitigation Strategy**: Random oversampling with stratified splits

### MLOps Infrastructure
- **Apache Airflow**: Pipeline orchestration with DAGs
- **DVC**: Data versioning with Google Cloud Storage
- **Docker**: Containerized deployment
- **Testing**: 30+ unit tests with pytest
- **Logging**: Python logging throughout pipeline
- **Alerting**: Anomaly detection with email/Slack notifications

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Apache Airflow Orchestration                  │
│                                                             │
│  Ingest → Preprocess → Validate → Bias Check → Version      │
└─────────────────────────────────────────────────────────────┘
         ↓          ↓           ↓          ↓          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│                                                             │
│     GCS (raw/staged)  |  Parquet  |  DVC (versions)         │
└─────────────────────────────────────────────────────────────┘
         ↑
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                             │
│                                                             │
│         Electricity Maps API  |  Open-Meteo API             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **RAM**: 8GB minimum
- **Python**: 3.11+
- **Docker**: 20.10+ (optional)

### Dependencies (requirements.txt)

```
apache-airflow>=2.7.0
dvc>=3.0.0
dvc-gs>=2.0.0
pandas>=2.0.0
pyarrow>=14.0.0
google-cloud-storage>=2.0.0
pytest>=7.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

### API Access
- **Electricity Maps**: API key for grid data
- **Google Cloud**: Service account for GCS storage

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Prathammehtaa/EcoPulse.git
cd EcoPulse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# 5. Pull data with DVC
cd Data_Pipeline
dvc pull

# 6. Run the pipeline
python src/stage_data.py
python src/tfdv_bias_analysis.py
python src/bias_mitigation.py

# 7. Run tests
pytest tests/ -v
```

For detailed instructions, see the [Data Pipeline README](Data_Pipeline/README.md).

---

## Project Structure

```
EcoPulse/
├── Data_Pipeline/
│   ├── config/              # Configuration files
│   ├── dags/                # Airflow DAG definitions
│   ├── data/
│   │   ├── raw/             # Raw API responses
│   │   ├── stage/           # Staged parquet files
│   │   └── processed/       # Bias-mitigated splits
│   ├── docs/                # Documentation
│   ├── reports/             # Analysis reports & plots
│   ├── src/                 # Source code
│   ├── tests/               # Unit tests
│   └── dvc.yaml             # DVC pipeline
├── .dvc/                    # DVC configuration
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Electricity Maps](https://electricitymaps.com) for grid carbon intensity data
- [Open-Meteo](https://open-meteo.com) for weather data
- Course instructors for MLOps guidance

---

**Last Updated**: February 2026  
**Status**: ✅ Data Pipeline Complete
