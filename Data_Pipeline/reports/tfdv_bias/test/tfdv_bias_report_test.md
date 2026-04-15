# EcoPulse TFDV Bias Analysis Report - TEST

**Generated:** 2026-04-08 16:27:38

**Dataset Split:** test

**Tool:** TensorFlow Data Validation (TFDV)

**Total Records:** 8,976

**Date Range:** 2025-10-01 00:00:00+00:00 to 2026-04-05 23:00:00+00:00

---

## Executive Summary

### ⚠️ 2 Bias Issues Detected

- **month**: 1 underrepresented: `['4']`
- **carbon_intensity_bucket**: 1 underrepresented: `['Very Low']`

---

## Detailed Slice Analysis


### hour_of_day

**Description:** Hour of day (0-23)


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 0 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 1 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 10 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 11 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 12 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 13 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 14 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 15 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 16 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 17 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 18 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 19 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 2 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 20 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 21 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 22 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 23 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 3 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 4 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 5 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 6 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 7 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 8 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |
| 9 | 374 | 4.2% | 4.2% | +0.1% | ✅ OK |

### day_of_week

**Description:** Day of the week


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Friday | 1,296 | 14.4% | 14.3% | +1.1% | ✅ OK |
| Monday | 1,248 | 13.9% | 14.3% | -2.7% | ✅ OK |
| Saturday | 1,296 | 14.4% | 14.3% | +1.1% | ✅ OK |
| Sunday | 1,296 | 14.4% | 14.3% | +1.1% | ✅ OK |
| Thursday | 1,296 | 14.4% | 14.3% | +1.1% | ✅ OK |
| Tuesday | 1,248 | 13.9% | 14.3% | -2.7% | ✅ OK |
| Wednesday | 1,296 | 14.4% | 14.3% | +1.1% | ✅ OK |

### season

**Description:** Season of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| Fall | 2,928 | 32.6% | 33.3% | -2.1% | ✅ OK |
| Spring | 1,728 | 19.2% | 33.3% | -42.2% | ✅ OK |
| Winter | 4,320 | 48.1% | 33.3% | +44.4% | ✅ OK |

### month

**Description:** Month of the year


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| 1 | 1,488 | 16.6% | 14.3% | +16.1% | ✅ OK |
| 10 | 1,488 | 16.6% | 14.3% | +16.1% | ✅ OK |
| 11 | 1,440 | 16.0% | 14.3% | +12.3% | ✅ OK |
| 12 | 1,488 | 16.6% | 14.3% | +16.1% | ✅ OK |
| 2 | 1,344 | 15.0% | 14.3% | +4.8% | ✅ OK |
| 3 | 1,488 | 16.6% | 14.3% | +16.1% | ✅ OK |
| 4 | 240 | 2.7% | 14.3% | -81.3% | ⚠️ Under |

### is_weekend

**Description:** Weekend vs Weekday


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| False | 6,384 | 71.1% | 50.0% | +42.2% | ✅ OK |
| True | 2,592 | 28.9% | 50.0% | -42.2% | ✅ OK |

### carbon_intensity_bucket

**Description:** Carbon intensity level


| Slice Value | Count | % | Expected % | Deviation | Status |
|-------------|-------|---|------------|-----------|--------|
| High | 2,688 | 29.9% | 20.0% | +49.8% | ✅ OK |
| Low | 1,393 | 15.5% | 20.0% | -22.4% | ✅ OK |
| Medium | 2,263 | 25.2% | 20.0% | +26.1% | ✅ OK |
| Very High | 2,503 | 27.9% | 20.0% | +39.5% | ✅ OK |
| Very Low | 129 | 1.4% | 20.0% | -92.8% | ⚠️ Under |

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
