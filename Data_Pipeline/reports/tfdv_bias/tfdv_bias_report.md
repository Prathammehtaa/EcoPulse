# EcoPulse TFDV Bias Analysis Report

**Generated:** 2026-02-22 01:39:26

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 105,408

**Date Range:** 2024-01-01 00:00:00+00:00 to 2024-12-31 23:00:00+00:00

---

## Executive Summary

### ⚠️ 3 Bias Issues Detected

- **carbon_intensity_bucket**: 2 underrepresented: `['Low', 'Very High']`
- **carbon_intensity_bucket**: 1 overrepresented: `['Very Low']`

---

## Detailed Slice Analysis


### hour_of_day

**Description:** Hour of day (0-23)


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 0 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 4,392 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 14,976 | 14.2% | 14.3% | -0.5% | ✅ OK |
| Monday | 15,264 | 14.5% | 14.3% | +1.4% | ✅ OK |
| Saturday | 14,976 | 14.2% | 14.3% | -0.5% | ✅ OK |
| Sunday | 14,976 | 14.2% | 14.3% | -0.5% | ✅ OK |
| Thursday | 14,976 | 14.2% | 14.3% | -0.5% | ✅ OK |
| Tuesday | 15,264 | 14.5% | 14.3% | +1.4% | ✅ OK |
| Wednesday | 14,976 | 14.2% | 14.3% | -0.5% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 26,208 | 24.9% | 25.0% | -0.6% | ✅ OK |
| Spring | 26,496 | 25.1% | 25.0% | +0.6% | ✅ OK |
| Summer | 26,496 | 25.1% | 25.0% | +0.6% | ✅ OK |
| Winter | 26,208 | 24.9% | 25.0% | -0.6% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 1 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 10 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 11 | 8,640 | 8.2% | 8.3% | -1.6% | ✅ OK |
| 12 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 2 | 8,352 | 7.9% | 8.3% | -5.0% | ✅ OK |
| 3 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 4 | 8,640 | 8.2% | 8.3% | -1.6% | ✅ OK |
| 5 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 6 | 8,640 | 8.2% | 8.3% | -1.6% | ✅ OK |
| 7 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 8 | 8,928 | 8.5% | 8.3% | +1.6% | ✅ OK |
| 9 | 8,640 | 8.2% | 8.3% | -1.6% | ✅ OK |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 75,456 | 71.6% | 50.0% | +43.2% | ✅ OK |
| True | 29,952 | 28.4% | 50.0% | -43.2% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 15,812 | 15.0% | 20.0% | -25.0% | ✅ OK |
| Low | 8,672 | 8.2% | 20.0% | -58.9% | ⚠️ Under |
| Medium | 17,090 | 16.2% | 20.0% | -18.9% | ✅ OK |
| Very High | 9,254 | 8.8% | 20.0% | -56.1% | ⚠️ Under |
| Very Low | 54,580 | 51.8% | 20.0% | +158.9% | ⚠️ Over |

---

## Recommendations


### Mitigation Strategies

1. **For Underrepresented Slices:**
   - Collect more data for these time periods
   - Apply oversampling during model training
   - Use stratified train/val/test splits

2. **For Overrepresented Slices:**
   - Apply undersampling to balance dataset
   - Use class weights during training

3. **General:**
   - Monitor model performance separately for each slice
   - Set up alerts for distribution drift in production
