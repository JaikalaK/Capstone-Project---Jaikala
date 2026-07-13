# Part 2 - Supervised ML: Regression and Classification
# Loads cleaned_data.csv from part 1

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

os.chdir(os.path.dirname(__file__) or ".")

warnings.filterwarnings("ignore")

SHOW_PLOTS = False
OUT = "output"
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)

PART1_CLEANED = "../part1/cleaned_data.csv"


def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, name), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# Task 1 - load and define labels
print("TASK 1 - LOAD DATA AND DEFINE LABELS")
if not os.path.exists(PART1_CLEANED):
    raise FileNotFoundError("Missing part1/cleaned_data.csv. Run part1 script first.")

df = pd.read_csv(PART1_CLEANED)
print("Shape:", df.shape)

y_reg = df["price"].copy()
median_price = y_reg.median()
y_clf = (y_reg > median_price).astype(int)

print("Regression target: price, range", y_reg.min(), "-", y_reg.max())
print("Classification target: 1 if price >" , median_price, "else 0")
print(y_clf.value_counts())


# Task 2 - encode categoricals
print("\nTASK 2 - ENCODE CATEGORICAL COLUMNS")
X = df.drop(columns=["price"]).copy()

cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order   = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

enc = OrdinalEncoder(categories=[cut_order, color_order, clarity_order],
                     handle_unknown="use_encoded_value", unknown_value=-1)
X[["cut", "color", "clarity"]] = enc.fit_transform(X[["cut", "color", "clarity"]])
print("Encoding done, X shape:", X.shape)
print(X.dtypes)


# Task 3 - split and scale
print("\nTASK 3 - TRAIN/TEST SPLIT AND SCALING")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
print("Train:", X_train.shape[0], "rows  Test:", X_test.shape[0], "rows")

# scaler fitted only on train data - important to avoid data leakage
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("Scaler fitted on X_train only")
print("Class balance in train:", y_clf_train.value_counts().to_dict())


# Task 4a - Linear Regression
print("\nTASK 4a - LINEAR REGRESSION")
lr = LinearRegression()
lr.fit(X_train_sc, y_reg_train)
y_pred = lr.predict(X_test_sc)

mse_lr = mean_squared_error(y_reg_test, y_pred)
r2_lr  = r2_score(y_reg_test, y_pred)
print(f"MSE: {mse_lr:,.2f}   R2: {r2_lr:.4f}")

coef_df = pd.DataFrame({"feature": X.columns, "coef": lr.coef_})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)
print("\nCoefficients:")
print(coef_df[["feature", "coef"]].to_string(index=False))
print("Top 3 by absolute value:", coef_df.head(3)["feature"].tolist())


# Task 4b - Ridge Regression
print("\nTASK 4b - RIDGE REGRESSION")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_reg_train)
y_pred_ridge = ridge.predict(X_test_sc)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge  = r2_score(y_reg_test, y_pred_ridge)
print(f"MSE: {mse_ridge:,.2f}   R2: {r2_ridge:.4f}")

compare_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge (alpha=1.0)"],
    "MSE":   [round(mse_lr, 2),    round(mse_ridge, 2)],
    "R2":    [round(r2_lr, 4),     round(r2_ridge, 4)]
})
print("\nOLS vs Ridge:")
print(compare_df.to_string(index=False))
compare_df.to_csv(OUT + "/regression_comparison.csv", index=False)


# Task 5 - Logistic Regression
print("\nTASK 5 - LOGISTIC REGRESSION (C=1.0)")
print("Class balance before training:")
print(y_clf_train.value_counts())

# dataset is roughly balanced but using class_weight anyway for good practice
log_reg = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_sc, y_clf_train)

y_pred_clf  = log_reg.predict(X_test_sc)
y_proba_clf = log_reg.predict_proba(X_test_sc)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_clf))

auc_base = roc_auc_score(y_clf_test, y_proba_clf)
print(f"AUC: {auc_base:.4f}")

# save confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix - Logistic Regression")
save_plot("confusion_matrix.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_base:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.annotate(f"AUC = {auc_base:.4f}", xy=(0.6, 0.3), fontsize=11)
save_plot("roc_curve_logreg.png")


# Task 5b - threshold sensitivity
print("\nTASK 5b - THRESHOLD SENSITIVITY (0.3 to 0.7)")
thresh_rows = []
for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
    preds = (y_proba_clf >= thresh).astype(int)
    thresh_rows.append({
        "Threshold": thresh,
        "Precision": round(precision_score(y_clf_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_clf_test, preds, zero_division=0), 4),
        "F1":        round(f1_score(y_clf_test, preds, zero_division=0), 4)
    })

thresh_df = pd.DataFrame(thresh_rows)
print(thresh_df.to_string(index=False))
best_t = thresh_df.loc[thresh_df["F1"].idxmax(), "Threshold"]
print("Best F1 threshold:", best_t)
thresh_df.to_csv(OUT + "/threshold_sensitivity.csv", index=False)


# Task 6 - stronger regularisation
print("\nTASK 6 - STRONGER REGULARISATION (C=0.01)")
log_reg2 = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
log_reg2.fit(X_train_sc, y_clf_train)

y_pred2  = log_reg2.predict(X_test_sc)
y_proba2 = log_reg2.predict_proba(X_test_sc)[:, 1]
auc2 = roc_auc_score(y_clf_test, y_proba2)

reg_df = pd.DataFrame({
    "Model":     ["LogReg C=1.0", "LogReg C=0.01"],
    "Precision": [round(precision_score(y_clf_test, y_pred_clf, zero_division=0), 4),
                  round(precision_score(y_clf_test, y_pred2,    zero_division=0), 4)],
    "Recall":    [round(recall_score(y_clf_test, y_pred_clf, zero_division=0), 4),
                  round(recall_score(y_clf_test, y_pred2,    zero_division=0), 4)],
    "AUC":       [round(auc_base, 4), round(auc2, 4)]
})
print(reg_df.to_string(index=False))
reg_df.to_csv(OUT + "/logreg_comparison.csv", index=False)


# Task 7 - bootstrap CI for AUC difference
print("\nTASK 7 - BOOTSTRAP CI (n=500)")
np.random.seed(42)
diffs = []
y_arr = np.array(y_clf_test)

for _ in range(500):
    idx = np.random.choice(len(y_arr), size=len(y_arr), replace=True)
    yt = y_arr[idx]
    if len(np.unique(yt)) < 2:
        continue
    diffs.append(roc_auc_score(yt, y_proba_clf[idx]) - roc_auc_score(yt, y_proba2[idx]))

diffs = np.array(diffs)
print(f"Mean AUC diff: {diffs.mean():.4f}")
print(f"95% CI: [{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]")
print("CI excludes zero:", not (np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)))

# bootstrap distribution plot
plt.figure(figsize=(7, 4))
plt.hist(diffs, bins=30, edgecolor="white")
plt.axvline(np.percentile(diffs, 2.5),  color="red",   linestyle="--", label="2.5th pct")
plt.axvline(np.percentile(diffs, 97.5), color="blue",  linestyle="--", label="97.5th pct")
plt.axvline(0, color="black", linewidth=1.5, label="Zero")
plt.xlabel("AUC difference (C=1.0 - C=0.01)")
plt.ylabel("Count")
plt.title("Bootstrap Distribution of AUC Difference")
plt.legend(fontsize=9)
save_plot("bootstrap_auc_diff.png")

pd.DataFrame({"metric": ["mean", "ci_low", "ci_high"],
              "value":  [diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)]
              }).to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)
coef_df[["feature", "coef"]].to_csv(os.path.join(OUT, "linear_regression_coefficients.csv"), index=False)

print("\nDone. Outputs in", OUT)
