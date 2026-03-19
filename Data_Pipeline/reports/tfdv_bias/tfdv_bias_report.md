# EcoPulse TFDV Bias Analysis Report

**Generated:** 2026-03-17 17:02:01

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 52,608

**Date Range:** 2023-01-01 00:00:00+00:00 to 2025-12-31 23:00:00+00:00

---

## Executive Summary

### ⚠️ 1 Bias Issues Detected

- **carbon_intensity_bucket**: 1 underrepresented: `['Very Low']`

---

## Detailed Slice Analysis


### hour_of_day

**Description:** Hour of day (0-23)


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 0 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 2,192 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 7,488 | 14.2% | 14.3% | -0.4% | ✅ OK |
| Monday | 7,536 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Saturday | 7,488 | 14.2% | 14.3% | -0.4% | ✅ OK |
| Sunday | 7,536 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Thursday | 7,488 | 14.2% | 14.3% | -0.4% | ✅ OK |
| Tuesday | 7,536 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Wednesday | 7,536 | 14.3% | 14.3% | +0.2% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 13,104 | 24.9% | 25.0% | -0.4% | ✅ OK |
| Spring | 13,248 | 25.2% | 25.0% | +0.7% | ✅ OK |
| Summer | 13,248 | 25.2% | 25.0% | +0.7% | ✅ OK |
| Winter | 13,008 | 24.7% | 25.0% | -1.1% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 1 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 10 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 11 | 4,320 | 8.2% | 8.3% | -1.5% | ✅ OK |
| 12 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 2 | 4,080 | 7.8% | 8.3% | -6.9% | ✅ OK |
| 3 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 4 | 4,320 | 8.2% | 8.3% | -1.5% | ✅ OK |
| 5 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 6 | 4,320 | 8.2% | 8.3% | -1.5% | ✅ OK |
| 7 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 8 | 4,464 | 8.5% | 8.3% | +1.9% | ✅ OK |
| 9 | 4,320 | 8.2% | 8.3% | -1.5% | ✅ OK |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 37,584 | 71.4% | 50.0% | +42.9% | ✅ OK |
| True | 15,024 | 28.6% | 50.0% | -42.9% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 14,855 | 28.2% | 20.0% | +41.2% | ✅ OK |
| Low | 8,963 | 17.0% | 20.0% | -14.8% | ✅ OK |
| Medium | 11,463 | 21.8% | 20.0% | +8.9% | ✅ OK |
| Very High | 13,295 | 25.3% | 20.0% | +26.4% | ✅ OK |
| Very Low | 4,032 | 7.7% | 20.0% | -61.7% | ⚠️ Under |

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
