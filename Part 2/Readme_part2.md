# Part 2 - Supervised ML: Regression and Classification

## Overview
This part builds two models on the cleaned diamonds data - one for regression (predicting price as a number) and one for classification (predicting if price is above or below median).

## How to run
```bash
python part2_ml.py
```
Needs `cleaned_data.csv` from part 1.

## Labels
- **Regression target**: `price` (continuous, range 326 to 18823)
- **Classification target**: 1 if price > 2401 (the median), else 0. The split is almost perfectly balanced at about 50/50.

## Encoding
The three string columns have a clear quality order so I used ordinal encoding:
- cut: Fair < Good < Very Good < Premium < Ideal
- color: J < I < H < G < F < E < D (D is best)
- clarity: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF

## Train/Test Split and Scaling
80/20 split with random_state=42. The scaler was fitted only on training data and then used to transform both train and test sets. Fitting the scaler on the whole dataset would be data leakage - the scaler would learn the mean and std of the test set and pass that information into training, which makes metrics look better than they really are.

## Linear Regression Results
| Metric | Value |
|--------|-------|
| MSE | 1,402,687.80 |
| R² | 0.9080 |

Top 3 features by coefficient size: carat, x, clarity

The `carat` coefficient is +5116 which means one unit increase in standardised carat is associated with about $5116 increase in predicted price. The `x` coefficient is negative (-1005) even though logically a longer diamond should cost more. This happens because carat and x are very correlated - they both measure size so the model splits the effect between them and gives one a negative sign.

## Ridge vs OLS
| Model | MSE | R² |
|-------|-----|----|
| Linear Regression | 1,402,687.80 | 0.908 |
| Ridge (alpha=1.0) | 1,402,697.74 | 0.908 |

Almost identical results. Ridge adds an L2 penalty to shrink coefficients, which helps when features are correlated. With this many rows the OLS is already stable so Ridge doesn't change much.

## Logistic Regression
The classes were 49.9% / 50.1% so no real imbalance, but I used `class_weight='balanced'` anyway.

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|--|------------|------------|
| Actual 0 | 5227 | 118 |
| Actual 1 | 153  | 5261 |

Accuracy: 97%   AUC: 0.9974

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

Recall matters more here because if we predict a diamond is cheap when it's actually expensive, we'd underprice it and lose revenue. So we want to catch as many true positives as possible.

## Threshold sensitivity
| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.30 | 0.9567 | 0.9884 | 0.9723 |
| 0.40 | 0.9699 | 0.9808 | **0.9753** |
| 0.50 | 0.9781 | 0.9717 | 0.9749 |
| 0.60 | 0.9849 | 0.9612 | 0.9729 |
| 0.70 | 0.9890 | 0.9474 | 0.9677 |

Best F1 is at 0.40. Lower threshold = more recall but more false positives. Since false negatives (missing expensive diamonds) cost more in this scenario, I'd go with 0.40.

## Regularisation experiment
| Model | Precision | Recall | AUC |
|-------|-----------|--------|-----|
| C=1.0 | 0.9781 | 0.9717 | 0.9974 |
| C=0.01 | 0.9777 | 0.9714 | 0.9972 |

C in LogisticRegression is the inverse of regularisation strength. Smaller C = stronger penalty = smaller coefficients. Going from 1.0 to 0.01 barely changed anything here because the dataset is large enough that the model doesn't overfit.

## Bootstrap CI for AUC difference
Ran 500 bootstrap samples, computed AUC difference between C=1.0 and C=0.01 on each sample.

- Mean difference: 0.0002
- 95% CI: [0.0001, 0.0005]
- CI excludes zero: Yes

The interval is entirely above zero so C=1.0 is consistently better, but only by a tiny amount. In practice they're equivalent.


