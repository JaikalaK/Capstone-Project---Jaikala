# Part 3 - Ensembles, Hyperparameter Tuning, Pipeline

## Overview
Part 3 builds on the same cleaned data and tries ensemble methods, does cross-validation, tunes a Random Forest with GridSearchCV, and saves the best model to disk.

## How to run
```bash
python part3_ml.py
```
Needs `../part1/cleaned_data.csv` from part 1.

## Decision Tree - unconstrained vs controlled

### No depth limit
| Train acc | Test acc | Gap |
|-----------|----------|-----|
| 0.9999 | 0.9737 | 0.0262 |

The tree basically memorised the training data. Train accuracy is nearly 100% but test drops by 2.6 points. Decision trees without constraints will keep splitting until each leaf has 1 sample, which is obvious overfitting. They're called high variance models because small changes in the data can completely change the tree structure.

### With max_depth=5 and min_samples_split=20
| Train acc | Test acc | Gap |
|-----------|----------|-----|
| 0.9720 | 0.9710 | 0.0010 |

The gap dropped from 0.026 to 0.001. `max_depth` limits how deep the tree can go, so it can't memorise individual training examples. `min_samples_split` stops the tree from making splits when there are fewer than 20 rows in a node, which avoids noise-based splits.

## Gini vs Entropy
| Criterion | Test acc |
|-----------|----------|
| Gini | 0.9710 |
| Entropy | 0.9602 |

Gini impurity formula: 1 - sum(pi^2)

Entropy formula: -sum(pi * log2(pi))

When Gini = 0, all samples in that node belong to one class, meaning it's a pure node. Both criteria measure how mixed a node is - Gini is a bit faster to compute since it doesn't need a log.

## Random Forest
| Metric | Value |
|--------|-------|
| Train acc | 0.9895 |
| Test acc | 0.9796 |
| AUC | 0.9985 |

Much better than a single tree. The forest builds 100 trees, each on a random bootstrap sample (about 63% of training rows, sampled with replacement). At each split it only considers a random subset of features (~3 features for our 9 total). This decorrelates the trees so their errors don't all go the same direction. Averaging predictions from 100 trees reduces variance significantly.

**Top 5 feature importances:**
| Feature | Importance |
|---------|-----------|
| y | 0.3846 |
| x | 0.2115 |
| carat | 0.1990 |
| z | 0.1209 |
| clarity | 0.0380 |

Feature importance here is the average reduction in Gini impurity from splitting on that feature, across all trees. It's different from a linear coefficient because it only tells you importance, not direction.

## Gradient Boosting
| Metric | Value |
|--------|-------|
| Train acc | 0.9807 |
| Test acc | 0.9785 |
| AUC | 0.9984 |

Similar to Random Forest but builds trees sequentially instead of in parallel. Each new tree tries to correct the errors from the previous ones. learning_rate=0.1 shrinks each tree's contribution so it doesn't overfit.

## Feature Ablation
Removed the 5 lowest importance features: clarity, color, depth, cut, table.

| Model | AUC |
|-------|-----|
| Full (9 features) | 0.9985 |
| Reduced (4 features) | 0.9918 |

AUC dropped by 0.0067. Not a lot but not nothing either. These features aren't pure noise — they do contribute a bit. In production you'd only drop them if the savings in inference time / data collection costs outweigh the small accuracy loss.

## Cross-validation (5-fold)
| Model | CV mean AUC | CV std |
|-------|------------|--------|
| Logistic Regression | 0.9975 | 0.0001 |
| Decision Tree (d=5) | 0.9956 | 0.0007 |
| Random Forest | 0.9985 | 0.0002 |
| Gradient Boosting | 0.9985 | 0.0002 |

Cross-validation is more reliable than a single split because it tests the model on every part of the data. One random split could be lucky or unlucky. With 5 folds you get 5 independent estimates, and the std tells you how stable the model is.

## GridSearchCV Pipeline Tuning
I built a pipeline: SimpleImputer → StandardScaler → RandomForestClassifier.

Grid searched over: n_estimators=[50,100,200], max_depth=[5,10,None], min_samples_leaf=[1,5]

Total fits: 3 * 3 * 2 * 5 folds = 90

Best params: n_estimators=200, max_depth=None, min_samples_leaf=5  
Best CV AUC: 0.9986  
Test AUC: 0.9985

GridSearch checks every combination. For large grids you'd use RandomizedSearch instead which samples a random subset of combinations - faster but might miss the best one.

## Learning curve
| Fraction | Train AUC | Test AUC |
|----------|-----------|---------|
| 20% | 0.9995 | 0.9980 |
| 40% | 0.9996 | 0.9983 |
| 60% | 0.9996 | 0.9984 |
| 80% | 0.9996 | 0.9985 |
| 100% | 0.9996 | 0.9985 |

Test AUC basically flattens out between 80% and 100%. This suggests the model has hit its capacity ceiling with this data — adding more rows beyond 43k won't help much. You'd need better features or a different model architecture to push higher.

## Serialisation
Best pipeline saved to `best_model.pkl` (14 MB) in this `part3` folder. Can be loaded with `joblib.load('best_model.pkl')`. The pipeline includes the imputer and scaler so there's no need to preprocess separately when using it in production.

## Final model comparison
| Model | CV mean AUC | CV std | Test AUC |
|-------|------------|--------|---------|
| LogReg C=1.0 | 0.9975 | 0.0001 | 0.9974 |
| Decision Tree (d=5) | 0.9956 | 0.0007 | 0.9961 |
| Random Forest | 0.9985 | 0.0002 | 0.9985 |
| Gradient Boosting | 0.9985 | 0.0002 | 0.9984 |
| Best RF Pipeline | 0.9986 | - | 0.9985 |

I'd recommend the Best RF Pipeline. It has the highest CV AUC, the full preprocessing steps are baked into the pipeline (no leakage risk at serving time), and it's easy to redeploy with just `joblib.load`.
