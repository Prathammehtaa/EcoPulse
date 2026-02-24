\# EcoPulse Bias Detection and Mitigation Report



\## 1. Overview



This document describes the bias detection and mitigation process implemented for the EcoPulse carbon intensity forecasting project. The goal is to ensure equitable model performance across different temporal and categorical subgroups.



\## 2. Data Slicing Dimensions



We analyzed bias across 6 dimensions:



| Dimension | Description | Categories |

|-----------|-------------|------------|

| carbon\_intensity\_bucket | Carbon intensity level | Very Low, Low, Medium, High, Very High |

| hour\_of\_day | Hour of day (0-23) | 24 categories |

| day\_of\_week | Day of the week | Monday-Sunday |

| month | Month of the year | January-December |

| season | Season | Winter, Spring, Summer, Fall |

| is\_weekend | Weekend indicator | True, False |



\## 3. Bias Detection Methodology



\### 3.1 Tools Used

\- \*\*Pandas\*\*: Custom statistical analysis for distribution and imbalance calculation

\- \*\*TFDV (TensorFlow Data Validation)\*\*: Schema and statistics generation (fallback to Pandas on Windows)

\- \*\*Matplotlib\*\*: Visualization of distributions



\### 3.2 Metrics Calculated

\- \*\*Distribution percentage\*\*: Percentage of samples in each category

\- \*\*Imbalance ratio\*\*: Max count / Min count across categories

\- \*\*Severity classification\*\*: LOW (≤2x), MODERATE (2-5x), HIGH (5-10x), SEVERE (>10x)



\## 4. Bias Detection Results



\### 4.1 Summary of Findings



| Dimension | Imbalance Ratio | Severity | Action Needed |

|-----------|-----------------|----------|---------------|

| hour\_of\_day | 1.00 | LOW | No |

| day\_of\_week | 1.00 | LOW | No |

| month | 1.00 | LOW | No |

| season | 1.00 | LOW | No |

| is\_weekend | 2.50 | MODERATE | No |

| carbon\_intensity\_bucket | \*\*16.12\*\* | \*\*SEVERE\*\* | \*\*Yes\*\* |



\### 4.2 Detailed Analysis - Carbon Intensity Bucket



The most significant bias was found in the carbon\_intensity\_bucket dimension:



| Bucket | Count | Percentage |

|--------|-------|------------|

| Very Low | 885 | 3.4% |

| Low | 4,320 | 16.4% |

| Medium | 13,021 | 49.4% |

| High | 7,318 | 27.8% |

| Very High | 808 | 3.1% |



\*\*Root Cause\*\*: Grid carbon intensity naturally clusters around medium values. Extreme values (Very Low during high renewable periods, Very High during peak fossil fuel usage) are less common.



\## 5. Bias Mitigation Strategy



\### 5.1 Technique Selected: Random Oversampling



We chose random oversampling to balance the carbon\_intensity\_bucket distribution because:

\- Preserves all original data points

\- Effective for severe class imbalance

\- Simple to implement and interpret



\### 5.2 Implementation

```python

\# Oversample minority classes to match majority class count

for each class in \[Very Low, Low, High, Very High]:

&nbsp;   sample with replacement until count == Medium count (13,021)

```



\### 5.3 Results After Mitigation



| Metric | Before | After |

|--------|--------|-------|

| Total Samples | 26,352 | 65,105 |

| Imbalance Ratio | 16.12x | 1.00x |

| Severity | SEVERE | LOW |



\*\*Distribution After Mitigation:\*\*

\- All 5 buckets: 13,021 samples each (20%)



\## 6. Stratified Train/Val/Test Splits



To ensure balanced representation in model training:



| Split | Samples | Ratio |

|-------|---------|-------|

| Train | 45,570 | 70% |

| Validation | 9,765 | 15% |

| Test | 9,770 | 15% |



Each split maintains the balanced 20% distribution across all carbon intensity buckets.



\## 7. Trade-offs and Limitations



\### 7.1 Trade-offs Made

1\. \*\*Data duplication\*\*: Oversampling creates duplicate records, which may lead to overfitting

2\. \*\*Synthetic patterns\*\*: Duplicated minority samples don't add new information



\### 7.2 Alternatives Considered

\- \*\*Undersampling\*\*: Rejected due to significant data loss (would reduce to ~800 samples per class)

\- \*\*SMOTE\*\*: Not applicable for tabular time-series with mixed feature types

\- \*\*Class weights\*\*: Could be used during model training as complementary approach



\### 7.3 Recommendations for Model Training

1\. Use cross-validation to detect overfitting from oversampling

2\. Consider class weights as additional mitigation

3\. Monitor performance on original (unbalanced) test set



\## 8. Files Generated



| File | Location | Description |

|------|----------|-------------|

| distribution\_\*.csv | reports/tfdv\_bias/ | Distribution stats per slice |

| stats\_\*.csv | reports/tfdv\_bias/ | Statistical summaries |

| bias\_summary.json | reports/tfdv\_bias/ | Overall bias summary |

| \*\_distribution.png | reports/tfdv\_bias/plots/ | Visualization plots |

| mitigation\_report.json | reports/bias\_mitigation/ | Mitigation results |

| train\_balanced.csv | data/processed/ | Balanced training set |

| val\_balanced.csv | data/processed/ | Balanced validation set |

| test\_balanced.csv | data/processed/ | Balanced test set |



\## 9. Conclusion



The EcoPulse dataset exhibited severe class imbalance in the carbon\_intensity\_bucket dimension (16.12x imbalance ratio). Through random oversampling, we successfully balanced the dataset to a 1.00x ratio while preserving all original data. Stratified splitting ensures consistent representation across train/validation/test sets for fair model evaluation.

