# EcoPulse TFDV Bias Analysis Report - TRAIN

**Generated:** 2026-04-08 16:27:37

**Dataset Split:** train

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 113,904

**Date Range:** 2019-01-01 00:00:00+00:00 to 2025-06-30 23:00:00+00:00

---

## Executive Summary

### ⚠️ 3 Bias Issues Detected

- **carbon_intensity_bucket**: 1 underrepresented: `['Very Low']`
- **carbon_intensity_bucket**: 2 overrepresented: `['High', 'Very High']`

---

## Detailed Slice Analysis


### hour_of_day

**Description:** Hour of day (0-23)


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 0 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 4,746 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Monday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Saturday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Sunday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Thursday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Tuesday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |
| Wednesday | 16,272 | 14.3% | 14.3% | +0.0% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 26,208 | 23.0% | 25.0% | -8.0% | ✅ OK |
| Spring | 30,912 | 27.1% | 25.0% | +8.6% | ✅ OK |
| Summer | 27,936 | 24.5% | 25.0% | -1.9% | ✅ OK |
| Winter | 28,848 | 25.3% | 25.0% | +1.3% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 1 | 10,416 | 9.1% | 8.3% | +9.7% | ✅ OK |
| 10 | 8,928 | 7.8% | 8.3% | -5.9% | ✅ OK |
| 11 | 8,640 | 7.6% | 8.3% | -8.9% | ✅ OK |
| 12 | 8,928 | 7.8% | 8.3% | -5.9% | ✅ OK |
| 2 | 9,504 | 8.3% | 8.3% | +0.1% | ✅ OK |
| 3 | 10,416 | 9.1% | 8.3% | +9.7% | ✅ OK |
| 4 | 10,080 | 8.8% | 8.3% | +6.2% | ✅ OK |
| 5 | 10,416 | 9.1% | 8.3% | +9.7% | ✅ OK |
| 6 | 10,080 | 8.8% | 8.3% | +6.2% | ✅ OK |
| 7 | 8,928 | 7.8% | 8.3% | -5.9% | ✅ OK |
| 8 | 8,928 | 7.8% | 8.3% | -5.9% | ✅ OK |
| 9 | 8,640 | 7.6% | 8.3% | -8.9% | ✅ OK |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 81,360 | 71.4% | 50.0% | +42.9% | ✅ OK |
| True | 32,544 | 28.6% | 50.0% | -42.9% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 34,233 | 30.1% | 20.0% | +50.2% | ⚠️ Over |
| Low | 11,764 | 10.3% | 20.0% | -48.4% | ✅ OK |
| Medium | 17,477 | 15.3% | 20.0% | -23.3% | ✅ OK |
| Very High | 45,493 | 39.9% | 20.0% | +99.7% | ⚠️ Over |
| Very Low | 4,937 | 4.3% | 20.0% | -78.3% | ⚠️ Under |

---

## Recommendations


### Mitigation Strategies

1. **For Underrepresented Slices:**
   - Collect more data for these time periods
   - Apply oversampling only on the training split during model training

2. **For Overrepresented Slices:**
   - Apply undersampling or sample weighting on the training split only

3. **General:**
   - Keep temporal train/val/test splits
   - Monitor model performance separately for each slice
   - Set up alerts for distribution drift in production
