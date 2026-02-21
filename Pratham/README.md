## **README: EcoPulse Schema Validation**

### **Overview**
This module provides automated schema validation and data quality checks for the EcoPulse preprocessing pipeline using TensorFlow Data Validation (TFDV). It ensures data integrity at each stage of the pipeline by validating incoming data against baseline schemas.

---

### **Requirements**

**Python Version:** Python 3.11 (required for TFDV compatibility)

**Dependencies:**
```bash
pip install tensorflow-data-validation pandas pyyaml
```

**Note:** TFDV does not support Python 3.12+. Ensure you're using Python 3.11 when running this code.

---

### **Project Structure**

```
EcoPulse/
├── schema_validation.py          # Validation module
├── run_pipeline.py                # Main pipeline with validation integration
├── schema_generation.py           # Baseline schema/stats generation (run once)
├── data_validation/
│   ├── schemas/                   # Baseline schemas (.pbtxt files)
│   │   ├── grid_schema.pbtxt
│   │   ├── weather_schema.pbtxt
│   │   └── features_schema.pbtxt
│   └── stats/                     # Baseline statistics (.pbtxt files)
└── data/
    ├── processed/                 # Preprocessed datasets
    └── features/                  # Feature-engineered datasets
```

---

### **Pipeline Flow**

**1. One-Time Setup: Generate Baseline Schemas**
```bash
python schema_generation.py
```
- Reads existing preprocessed data (grid, weather, merged, features)
- Generates baseline schemas and statistics
- Saves to `data_validation/schemas/` and `data_validation/stats/`

**2. Integrated Preprocessing Pipeline with Validation**
```bash
python run_pipeline.py
```

**Flow:**
1. **Grid Preprocessing** → Validates grid schema → Stop if failed
2. **Weather Preprocessing** → Validates weather schema → Stop if failed
3. **Merge & Feature Engineering** → Validates features schema → Stop if failed

**Each validation checks:**
- Missing or extra columns
- Incorrect data types
- Values outside expected ranges
- Unexpected categorical values

---

### **Usage**

**Run full pipeline with validation (default):**
```bash
python run_pipeline.py
```

**Run specific step:**
```bash
python run_pipeline.py --step grid
python run_pipeline.py --step weather
python run_pipeline.py --step merge
```

**Skip validation (not recommended):**
```bash
python run_pipeline.py --skip-validation
```

**Use GCS storage:**
```bash
python run_pipeline.py --gcs
```

---

### **Validation Logic**

**What gets validated:**
- **Grid data:** 11 columns including carbon intensity, renewable %, total load
- **Weather data:** 9 columns including temperature, wind speed, cloud cover
- **Features data:** 40 columns including engineered features (lags, rolling windows, interactions)

**When validation fails:**
- Pipeline stops immediately
- Logs specific anomalies (which column, what issue)
- Prevents bad data from reaching model training

**When validation passes:**
- Pipeline continues to next step
- Data quality confirmed

---

### **Data Drift Detection**

**Detect drift in production:**
```python
from schema_validation import detect_drift

# Compare new data to baseline
detect_drift(
    new_df=new_data,
    baseline_stats_path='data_validation/stats/weather_stats.pbtxt',
    dataset_name='weather'
)
```

**Recommended frequency:**
- **Hourly:** Schema validation only (fast checks)
- **Daily/Weekly:** Drift detection (pattern changes)
- **Monthly:** Model retraining if drift confirmed

---

### **Functions**

**`validate_dataset(df, dataset_name, schemas_dir)`**
- Validates DataFrame against baseline schema
- Returns: `True` if valid, `False` if anomalies detected

**`detect_drift(new_df, baseline_stats_path, dataset_name)`**
- Compares new data statistics to baseline
- Visualizes distribution changes
- Returns: `None` (manual review required)

---

### **Key Features**

✅ **Automated validation** - Integrated into preprocessing pipeline  
✅ **Early failure detection** - Catches issues before model training  
✅ **Non-invasive** - Doesn't modify teammate's preprocessing code  
✅ **Configurable** - Can be disabled with `--skip-validation` flag  
✅ **Graceful fallback** - Continues if baseline schemas don't exist  

---

### **Troubleshooting**

**Issue:** `ModuleNotFoundError: No module named 'tensorflow_data_validation'`  
**Fix:** Ensure Python 3.11 is active and TFDV is installed

**Issue:** `Baseline schema not found`  
**Fix:** Run `python schema_generation.py` first to create baseline schemas

**Issue:** Validation fails unexpectedly  
**Fix:** Check logs for specific anomalies, verify data quality issues

---

### **Team Integration**

**For teammates running the pipeline:**
1. Ensure Python 3.11 environment
2. Install TFDV: `pip install tensorflow-data-validation`
3. Run normally: `python run_pipeline.py`
4. Validation happens automatically at each step

**To update baseline schemas:**
```bash
# After significant data changes
python schema_generation.py
```
