# EcoPulse TFDV Bias Analysis Report - VAL

**Generated:** 2026-04-08 16:27:38

**Dataset Split:** val

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 4,416

**Date Range:** 2025-07-01 00:00:00+00:00 to 2025-09-30 23:00:00+00:00

---

## Executive Summary

### ⚠️ 2 Bias Issues Detected

- **carbon_intensity_bucket**: 1 underrepresented: `['Very Low']`
- **carbon_intensity_bucket**: 1 overrepresented: `['Very High']`

---

## Detailed Slice Analysis


### hour_of_day

**Description:** Hour of day (0-23)


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 0 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 184 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |
| Monday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |
| Saturday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |
| Sunday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |
| Thursday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |
| Tuesday | 672 | 15.2% | 14.3% | +6.5% | ✅ OK |
| Wednesday | 624 | 14.1% | 14.3% | -1.1% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 1,440 | 32.6% | 50.0% | -34.8% | ✅ OK |
| Summer | 2,976 | 67.4% | 50.0% | +34.8% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 7 | 1,488 | 33.7% | 33.3% | +1.1% | ✅ OK |
| 8 | 1,488 | 33.7% | 33.3% | +1.1% | ✅ OK |
| 9 | 1,440 | 32.6% | 33.3% | -2.2% | ✅ OK |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 3,168 | 71.7% | 50.0% | +43.5% | ✅ OK |
| True | 1,248 | 28.3% | 50.0% | -43.5% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 1,215 | 27.5% | 20.0% | +37.5% | ✅ OK |
| Low | 605 | 13.7% | 20.0% | -31.5% | ✅ OK |
| Medium | 1,156 | 26.2% | 20.0% | +30.9% | ✅ OK |
| Very High | 1,412 | 32.0% | 20.0% | +59.9% | ⚠️ Over |
| Very Low | 28 | 0.6% | 20.0% | -96.8% | ⚠️ Under |

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
