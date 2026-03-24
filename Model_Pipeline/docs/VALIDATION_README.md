# EcoPulse — Model Validation & Sensitivity Analysis (Person 4)

## Overview
This module validates the trained XGBoost models for carbon intensity forecasting across 4 forecast horizons (1h, 6h, 12h, 24h). It generates validation metrics, SHAP/LIME explanations, hyperparameter sensitivity analysis, and full reports.

---

## Author
- **Name**: Hitarth Upadhyay
- **Email**: hitupadhyay110@gmail.com
- **Branch**: `hitarth_dev`

---

## What This Module Does

### 1. Validation Metrics
Evaluates each trained model on validation and test sets using:
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average prediction error in gCO2/kWh |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **R²** | R-squared — how well the model explains variance (1.0 = perfect) |
| **MAPE** | Mean Absolute Percentage Error — error as a percentage |

### 2. SHAP Feature Importance
- Uses `TreeExplainer` (fast, exact for XGBoost)
- Generates bar plot (top 20 features by mean |SHAP|)
- Generates beeswarm plot (shows direction of impact)
- Runs on 500-row sample per horizon for speed

### 3. LIME Local Explanations
- Explains 3 individual predictions per horizon
- Shows which features pushed a specific prediction up or down
- Saved as interactive HTML files

### 4. Hyperparameter Sensitivity Analysis
- Tests `n_estimators`, `max_depth`, `learning_rate`
- Ranges centered around Person 2's best hyperparameters
- Uses 3-fold cross-validation to measure MAE at each value
- Shows how robust the model is to hyperparameter changes

---

## Results Summary

### Test Set Performance
| Model | MAE | RMSE | R² | MAPE% | Improvement vs Baseline |
|-------|-----|------|----|-------|------------------------|
| XGBoost Tuned (1h)  | 21.62 | 29.96 | 0.9215 | 7.91%  | +62.4% |
| XGBoost Tuned (6h)  | 31.42 | 43.12 | 0.8377 | 11.66% | +56.0% |
| XGBoost Tuned (12h) | 36.83 | 49.80 | 0.7841 | 13.81% | +51.8% |
| XGBoost Tuned (24h) | 40.41 | 53.65 | 0.7502 | 15.44% | +41.3% |

### Key Findings
- All 4 models significantly outperform the baseline
- R² above 0.75 for all horizons including 24h ahead forecasting
- 1h model achieves R²=0.92 — excellent for carbon intensity forecasting
- Performance degrades gracefully with longer horizons (expected behaviour)

---

## How to Run

### Setup
```bash
cd ~/EcoPulse/Model_Pipeline
python3 -m venv venv
source venv/bin/activate
pip install shap lime matplotlib seaborn xgboost scikit-learn pandas numpy joblib pyarrow
```

### Run Validation
```bash
cd ~/EcoPulse/Model_Pipeline/src
python3 model_validation.py
```

### Expected Runtime
~10-15 minutes (sensitivity analysis is the slowest step)

---

## Output Files

All files saved to `Model_Pipeline/reports/validation/`

| File | Description |
|------|-------------|
| `validation_report.json` | Master report with all metrics for all horizons |
| `confusion_matrix_1h/6h/12h/24h.png` | Bucket-level prediction heatmaps |
| `shap_summary_1h/6h/12h/24h.png` | SHAP feature importance bar plots |
| `shap_beeswarm_1h/6h/12h/24h.png` | SHAP beeswarm plots (impact direction) |
| `lime_sample_1h/6h/12h/24h_0/1/2.html` | LIME local explanation HTML files |
| `sensitivity_1h/6h/12h/24h.png` | Hyperparameter sensitivity charts |
| `sensitivity_1h/6h/12h/24h.json` | Sensitivity analysis raw data |

**Total: 33 files generated**

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| xgboost | 3.2.0 | Load trained models |
| shap | 0.51.0 | SHAP feature explanations |
| lime | 0.2.0.1 | LIME local explanations |
| scikit-learn | 1.8.0 | Cross-validation for sensitivity |
| matplotlib | 3.10.8 | Plots and visualizations |
| seaborn | 0.13.2 | Confusion matrix heatmap |
| pandas | 3.0.1 | Data loading |
| joblib | 1.5.3 | Load model files |

---

## Project Structure

```
Model_Pipeline/
├── src/
│   ├── model_validation.py   ← Person 4 (this file)
│   ├── train_xgboost.py      ← Person 1
│   ├── train_lightgbm.py     ← Person 1
│   ├── hyperparameter_tuning.py ← Person 2
│   └── utils.py              ← Shared utilities
├── models/
│   ├── xgboost_tuned_1h.joblib
│   ├── xgboost_tuned_6h.joblib
│   ├── xgboost_tuned_12h.joblib
│   └── xgboost_tuned_24h.joblib
└── reports/
    └── validation/           ← All Person 4 outputs
```

---

## Data

| Split | Rows | Source |
|-------|------|--------|
| Train | 43,776 | `Data_Pipeline/data/processed/train_split.parquet` |
| Val   | 4,416  | `Data_Pipeline/data/processed/val_split.parquet` |
| Test  | 4,368  | `Data_Pipeline/data/processed/test_split.parquet` |

- **Grid zones**: US-NW-PACW, US-MIDA-PJM
- **Date range**: 2024-01-01 → 2024-12-31
- **Features**: 91 features (grid signals + weather + time features + lag features)
- **Target**: Carbon intensity (gCO2/kWh) at 1h, 6h, 12h, 24h ahead

---

*EcoPulse · Person 4: Model Validation · Hitarth Upadhyay · March 2026*