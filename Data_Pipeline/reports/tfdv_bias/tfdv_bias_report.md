# EcoPulse TFDV Bias Analysis Report

**Generated:** 2026-03-31 01:00:17

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 56,304

**Date Range:** 2023-01-01 00:00:00+00:00 to 2026-03-18 23:00:00+00:00

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
| 0 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 2,346 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 8,016 | 14.2% | 14.3% | -0.3% | ✅ OK |
| Monday | 8,064 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Saturday | 8,016 | 14.2% | 14.3% | -0.3% | ✅ OK |
| Sunday | 8,064 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Thursday | 8,016 | 14.2% | 14.3% | -0.3% | ✅ OK |
| Tuesday | 8,064 | 14.3% | 14.3% | +0.2% | ✅ OK |
| Wednesday | 8,064 | 14.3% | 14.3% | +0.2% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 13,104 | 23.3% | 25.0% | -6.9% | ✅ OK |
| Spring | 14,112 | 25.1% | 25.0% | +0.2% | ✅ OK |
| Summer | 13,248 | 23.5% | 25.0% | -5.9% | ✅ OK |
| Winter | 15,840 | 28.1% | 25.0% | +12.5% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 1 | 5,952 | 10.6% | 8.3% | +26.8% | ✅ OK |
| 10 | 4,464 | 7.9% | 8.3% | -4.8% | ✅ OK |
| 11 | 4,320 | 7.7% | 8.3% | -8.0% | ✅ OK |
| 12 | 4,464 | 7.9% | 8.3% | -4.8% | ✅ OK |
| 2 | 5,424 | 9.6% | 8.3% | +15.6% | ✅ OK |
| 3 | 5,328 | 9.5% | 8.3% | +13.5% | ✅ OK |
| 4 | 4,320 | 7.7% | 8.3% | -8.0% | ✅ OK |
| 5 | 4,464 | 7.9% | 8.3% | -4.8% | ✅ OK |
| 6 | 4,320 | 7.7% | 8.3% | -8.0% | ✅ OK |
| 7 | 4,464 | 7.9% | 8.3% | -4.8% | ✅ OK |
| 8 | 4,464 | 7.9% | 8.3% | -4.8% | ✅ OK |
| 9 | 4,320 | 7.7% | 8.3% | -8.0% | ✅ OK |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 40,224 | 71.4% | 50.0% | +42.9% | ✅ OK |
| True | 16,080 | 28.6% | 50.0% | -42.9% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 15,636 | 27.8% | 20.0% | +38.9% | ✅ OK |
| Low | 9,918 | 17.6% | 20.0% | -11.9% | ✅ OK |
| Medium | 12,294 | 21.8% | 20.0% | +9.2% | ✅ OK |
| Very High | 14,376 | 25.5% | 20.0% | +27.6% | ✅ OK |
| Very Low | 4,080 | 7.2% | 20.0% | -63.8% | ⚠️ Under |

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
