# EcoPulse Model Development Pipeline

**Date:** March 2026  
**Status:** ✅ Complete

---

## Overview

This document describes the machine learning modelling pipeline for EcoPulse — a carbon-aware data center scheduling system. The pipeline forecasts grid carbon intensity 1–24 hours ahead for two US electricity zones, enabling data centers to shift flexible workloads to low-carbon periods.

### Key Results

| Horizon | Best Model | MAE (gCO₂/kWh) | R² Score | Improvement over Baseline |
|---------|-----------|-----------------|----------|--------------------------|
| 1h | XGBoost | 25.14 | 0.9032 | 56.3% |
| 6h | XGBoost | 34.34 | 0.8172 | 51.9% |
| 12h | XGBoost | 39.97 | 0.7579 | 47.7% |
| 24h | XGBoost | 43.01 | 0.7179 | 37.5% |

---

## Data Summary

**Source:** Preprocessed parquet files from the Data Pipeline team 

| Property | Value |
|----------|-------|
| Training set | 43,776 rows (Jan 2023 – Jun 2025) |
| Validation set | 4,416 rows (Jul 2025 – Sep 2025) |
| Test set | 4,368 rows (Oct 2025 – Dec 2025) |
| Zones | US-MIDA-PJM (Virginia), US-NW-PACW (Oregon) |
| Features | 91 numeric features |
| Target | carbon_intensity_gco2_per_kwh |
| Forecast horizons | 1h, 6h, 12h, 24h |
| Split method | Temporal (chronological, no data leakage) |

### Feature Groups (91 total)

- **Grid signals (6):** carbon_intensity, carbon_free_energy_pct, renewable_energy_pct, total_load_mw, net_load_mw, carbon_intensity_fossil
- **Weather (6):** temperature, wind_speed, cloud_cover, solar_radiation, rain, snowfall
- **Temporal (9):** hour_of_day, day_of_week, month, quarter, year, day_of_year, week_of_year, is_weekend, is_daytime
- **Cyclical (6):** sin/cos encodings for hour, month, day_of_week
- **Lag features (20):** 1h, 3h, 6h, 24h, 168h lags for carbon_intensity, total_load, temperature, wind_speed
- **Rolling features (36):** mean, std, min, max at 4h, 12h, 24h windows for carbon_intensity, total_load, temperature
- **Interaction features (5):** solar_potential, temp_demand_proxy, carbon_change_1h, clean_energy_score, load_variability
- **Zone encoding (2):** one-hot encoded zone dummies

### Data Validation

66 out of 66 validation checks passed before modelling, covering:
- Schema consistency across splits
- Temporal integrity (no leakage between train/val/test)
- 100% hourly frequency with no gaps
- Zero nulls, zero duplicates, zero cross-split overlap
- Forecast target label alignment verified at 100%

---

## Modelling Approach

### Baseline Models

Three naive baselines were established as benchmarks:

1. **Naive Persistence:** Predicts future carbon intensity equals current value. Best for short horizons (1h MAE=57.48) due to temporal autocorrelation in grid data.

2. **24h-Ago Persistence:** Predicts future equals same hour yesterday. Captures daily seasonality but ignores day-to-day weather and grid changes.

3. **Historical Hourly Mean:** Predicts future equals the training set average for this (zone, hour) combination. Best for longer horizons (24h MAE=68.79) where the typical daily pattern is more reliable than the current value.

### XGBoost (Selected Model)

Gradient-boosted decision tree ensemble trained separately for each forecast horizon.

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 500 | Maximum trees; early stopping monitors validation |
| max_depth | 6 | Balanced complexity; prevents memorization |
| learning_rate | 0.05 | Small steps for better generalization |
| subsample | 0.8 | Row sampling to reduce overfitting |
| colsample_bytree | 0.8 | Feature sampling for ensemble diversity |
| min_child_weight | 5 | Minimum samples per leaf node |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |
| early_stopping_rounds | 30 | Stop if validation doesn't improve |

**Results (Test Set):**
| Horizon | MAE | RMSE | R² | MAPE |
|---------|-----|------|-----|------|
| 1h | 25.14 | 33.26 | 0.9032 | 8.96% |
| 6h | 34.34 | 45.76 | 0.8172 | 12.56% |
| 12h | 39.97 | 52.73 | 0.7579 | 14.84% |
| 24h | 43.01 | 57.02 | 0.7179 | 16.35% |

**Key Feature Importances:**
- **1h horizon:** Dominated by recent rolling averages (carbon_intensity_mean_4h = 0.40), indicating short-term momentum drives near-future predictions
- **6h horizon:** 24h rolling statistics take over (carbon_intensity_mean_24h = 0.56), capturing daily trend patterns
- **12h horizon:** Mix of 12h and 24h rolling features, with is_daytime becoming relevant
- **24h horizon:** Structural features dominate (clean_energy_score = 0.43, renewable_energy_pct = 0.17), showing that grid energy mix composition matters most for day-ahead forecasts

This shift from "recent momentum" features at short horizons to "structural grid characteristics" at long horizons is consistent with domain knowledge of electricity grid behavior.

### LightGBM (Alternative Model)

Same architecture as XGBoost with histogram-based splitting for comparison.

**Results (Test Set):**
| Horizon | MAE | RMSE | R² | MAPE |
|---------|-----|------|-----|------|
| 1h | 25.96 | 34.10 | 0.8983 | 9.12% |
| 6h | 35.58 | 46.94 | 0.8076 | 12.98% |
| 12h | 40.04 | 52.45 | 0.7605 | 14.82% |
| 24h | 43.76 | 57.91 | 0.7089 | 16.65% |

LightGBM uses different feature importance patterns — valuing carbon_change_1h and temporal encodings (hour_cos, hour_sin) more heavily than rolling statistics. Despite finding different patterns, it achieves comparable but slightly lower accuracy than XGBoost across all horizons.

### Model Selection Recommendation

**XGBoost is recommended for deployment across all 4 horizons** based on:
- Consistently lower MAE across all horizons (margins of 0.07 to 1.24 gCO₂/kWh)
- Higher R² scores at critical short horizons (0.9032 vs 0.8983 at 1h)
- More interpretable feature importance alignment with domain knowledge

---

## Pipeline Execution

### Prerequisites
```bash
pip install pandas numpy xgboost lightgbm scikit-learn matplotlib joblib pyarrow
```

### Run Order
```bash
cd Model_Pipeline/src

# Step 1: Validate data readiness
python ../validate_model_data.py

# Step 2: Establish baselines
python baselines.py

# Step 3: Train XGBoost
python train_xgboost.py

# Step 4: Train LightGBM
python train_lightgbm.py

# Step 5: Compare all models
python model_comparison.py
```

### Output Files

```
Model_Pipeline/
├── models/
│   ├── xgboost_1h.joblib          # Trained XGBoost models
│   ├── xgboost_6h.joblib
│   ├── xgboost_12h.joblib
│   ├── xgboost_24h.joblib
│   ├── lightgbm_1h.joblib         # Trained LightGBM models
│   ├── lightgbm_6h.joblib
│   ├── lightgbm_12h.joblib
│   └── lightgbm_24h.joblib
├── reports/
│   ├── baseline_results.csv        # Baseline metrics
│   ├── xgboost_results.csv         # XGBoost metrics
│   ├── lightgbm_results.csv        # LightGBM metrics
│   ├── full_comparison.csv         # All models compared
│   ├── xgb_importance_*.csv        # Feature importances
│   ├── lgb_importance_*.csv
│   ├── model_comparison_mae.png    # MAE comparison chart
│   ├── model_comparison_r2.png     # R² comparison chart
│   └── model_improvement.png       # Improvement over baselines chart
└── src/
    ├── utils.py                    # Shared utilities
    ├── baselines.py                # Baseline models
    ├── train_xgboost.py            # XGBoost training
    ├── train_lightgbm.py           # LightGBM training
    └── model_comparison.py         # Comparison & charts
```

---

## Handoff Notes for Downstream Tasks

### For Model Serving (FastAPI)
- Load models: `joblib.load("models/xgboost_1h.joblib")`
- Feature columns expected: see `utils.py > get_feature_columns()`
- Zone must be one-hot encoded before prediction
- Drop columns: datetime, zone (string), aws/gcp/azure_region, season (string), all target columns

### For Monitoring (Evidently)
- Training data statistics available in `reports/` CSVs
- Target distribution: Train mean=296.1, Test mean=333.9 (12.8% shift — within acceptable range)
- PJM zone has higher carbon intensity (mean ~400) than PACW (mean ~200)

### For Future Improvements
- Both models used all 500 trees without early stopping triggering — increasing n_estimators to 1000+ or tuning with Optuna could improve results
- Adding a 3rd zone (US-NE-ISNE) would improve generalization
- Seasonal retraining trigger recommended as carbon intensity patterns shift with seasons