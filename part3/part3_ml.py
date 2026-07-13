# Part 3 - Ensembles, Tuning, and Sklearn Pipeline
# Needs cleaned_data.csv from part 1

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

os.chdir(os.path.dirname(__file__) or ".")

warnings.filterwarnings("ignore")

SHOW_PLOTS = False
OUT = "output"
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)

PART1_CLEANED = "../part1/cleaned_data.csv"
MODEL_PATH = "best_model.pkl"


def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, name), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# Data prep - same as part 2
print("DATA PREPARATION")
if not os.path.exists(PART1_CLEANED):
    raise FileNotFoundError("Missing part1/cleaned_data.csv. Run part1 script first.")

df = pd.read_csv(PART1_CLEANED)

y_clf = (df["price"] > df["price"].median()).astype(int)
X = df.drop(columns=["price"]).copy()

cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order   = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

enc = OrdinalEncoder(categories=[cut_order, color_order, clarity_order],
                     handle_unknown="use_encoded_value", unknown_value=-1)
X[["cut", "color", "clarity"]] = enc.fit_transform(X[["cut", "color", "clarity"]])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Train:", X_train_sc.shape[0], "  Test:", X_test_sc.shape[0])


# Task 1 - unconstrained decision tree
print("\nTASK 1 - DECISION TREE (no depth limit)")
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_sc, y_train)
print("Train acc:", round(dt_full.score(X_train_sc, y_train), 4))
print("Test acc: ", round(dt_full.score(X_test_sc,  y_test),  4))
print("Gap:      ", round(dt_full.score(X_train_sc, y_train) - dt_full.score(X_test_sc, y_test), 4))


# Task 2 - controlled tree
print("\nTASK 2 - CONTROLLED DECISION TREE (max_depth=5)")
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_ctrl.fit(X_train_sc, y_train)

train_acc = dt_ctrl.score(X_train_sc, y_train)
test_acc  = dt_ctrl.score(X_test_sc,  y_test)
print("Train acc:", round(train_acc, 4))
print("Test acc: ", round(test_acc,  4))
print("Gap:      ", round(train_acc - test_acc, 4))

dt_cmp = pd.DataFrame({
    "Model":     ["DT unconstrained", "DT depth=5"],
    "Train acc": [round(dt_full.score(X_train_sc, y_train), 4), round(train_acc, 4)],
    "Test acc":  [round(dt_full.score(X_test_sc,  y_test),  4), round(test_acc,  4)],
    "Gap":       [round(dt_full.score(X_train_sc, y_train) - dt_full.score(X_test_sc, y_test), 4),
                  round(train_acc - test_acc, 4)]
})
print(dt_cmp.to_string(index=False))
dt_cmp.to_csv(OUT + "/dt_comparison.csv", index=False)


# Task 3 - gini vs entropy
print("\nTASK 3 - GINI VS ENTROPY")
dt_gini    = DecisionTreeClassifier(criterion="gini",    max_depth=5, random_state=42)
dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_gini.fit(X_train_sc, y_train)
dt_entropy.fit(X_train_sc, y_train)
print("Gini test acc:   ", round(dt_gini.score(X_test_sc, y_test), 4))
print("Entropy test acc:", round(dt_entropy.score(X_test_sc, y_test), 4))


# Task 4 - random forest
print("\nTASK 4 - RANDOM FOREST")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_sc, y_train)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
print("Train acc:", round(rf.score(X_train_sc, y_train), 4))
print("Test acc: ", round(rf.score(X_test_sc,  y_test),  4))
print("AUC:      ", round(rf_auc, 4))

feat_imp = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
feat_imp = feat_imp.sort_values("importance", ascending=False)
print("\nTop 5 features:")
print(feat_imp.head(5).to_string(index=False))
feat_imp.to_csv(OUT + "/rf_feature_importances.csv", index=False)

plt.figure(figsize=(8, 4))
plt.bar(feat_imp["feature"], feat_imp["importance"])
plt.title("Random Forest Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=20)
save_plot("rf_feature_importances.png")


# Task 4a - gradient boosting
print("\nTASK 4a - GRADIENT BOOSTING")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_sc, y_train)
gb_proba = gb.predict_proba(X_test_sc)[:, 1]
gb_auc   = roc_auc_score(y_test, gb_proba)
print("Train acc:", round(gb.score(X_train_sc, y_train), 4))
print("Test acc: ", round(gb.score(X_test_sc,  y_test),  4))
print("AUC:      ", round(gb_auc, 4))


# Task 4b - feature ablation (drop bottom 5 features)
print("\nTASK 4b - FEATURE ABLATION")
bottom5 = feat_imp.tail(5)["feature"].tolist()
print("Dropping:", bottom5)

keep = [i for i, f in enumerate(feature_names) if f not in bottom5]
X_train_red = X_train_sc[:, keep]
X_test_red  = X_test_sc[:, keep]

rf_red = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_red.fit(X_train_red, y_train)
rf_red_auc = roc_auc_score(y_test, rf_red.predict_proba(X_test_red)[:, 1])

print(f"Full model AUC:    {rf_auc:.4f}")
print(f"Reduced model AUC: {rf_red_auc:.4f}")
print(f"Drop:              {rf_auc - rf_red_auc:.4f}")
pd.DataFrame({"model": ["full", "reduced"], "auc": [rf_auc, rf_red_auc]}).to_csv(
    OUT + "/ablation_results.csv", index=False)


# Task 5 - cross validation
print("\nTASK 5 - CROSS-VALIDATION (5-fold)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

cv_rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_rows.append({"Model": name, "Mean AUC": round(scores.mean(), 4), "Std": round(scores.std(), 4)})
    print(f"  {name:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}")

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT + "/cv_comparison.csv", index=False)


# Task 6 - GridSearchCV with pipeline
print("\nTASK 6 - GRIDSEARCHCV PIPELINE")
pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth":    [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

# 3 x 3 x 2 x 5 = 90 fits total
print("Total fits: 90  (3 * 3 * 2 combos, 5 CV folds each)")
gs = GridSearchCV(pipe, param_grid,
                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                  scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)  # unscaled, pipeline handles it
print("Best params:", gs.best_params_)
print("Best CV AUC:", round(gs.best_score_, 4))

best_pipe = gs.best_estimator_
best_auc  = roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1])
print("Test AUC:", round(best_auc, 4))

pd.DataFrame([gs.best_params_]).to_csv(OUT + "/gridsearch_best_params.csv", index=False)


# Task 6b - manual learning curve
print("\nTASK 6b - LEARNING CURVE")
X_train_arr = np.array(X_train)
y_train_arr = np.array(y_train)

lc_rows = []
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    n = int(frac * len(X_train_arr))
    X_sub = X_train_arr[:n]
    y_sub = y_train_arr[:n]

    # rebuild best params into a fresh pipeline
    p = gs.best_params_
    tmp_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=p["randomforestclassifier__n_estimators"],
            max_depth=p["randomforestclassifier__max_depth"],
            min_samples_leaf=p["randomforestclassifier__min_samples_leaf"],
            random_state=42
        )
    )
    tmp_pipe.fit(X_sub, y_sub)
    tr_auc = roc_auc_score(y_sub,   tmp_pipe.predict_proba(X_sub)[:, 1])
    te_auc = roc_auc_score(y_test,  tmp_pipe.predict_proba(X_test)[:, 1])
    lc_rows.append({"fraction": frac, "train_auc": round(tr_auc, 4), "test_auc": round(te_auc, 4)})
    print(f"  frac={frac}  train={tr_auc:.4f}  test={te_auc:.4f}")

lc_df = pd.DataFrame(lc_rows)
lc_df.to_csv(OUT + "/learning_curve.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(lc_df["fraction"], lc_df["train_auc"], marker="o", label="Train AUC")
plt.plot(lc_df["fraction"], lc_df["test_auc"],  marker="s", label="Test AUC")
plt.title("Learning Curve - Best RF Pipeline")
plt.xlabel("Training fraction")
plt.ylabel("AUC")
plt.ylim(0.90, 1.01)
plt.legend()
save_plot("learning_curve.png")


# Task 7 - save model
print("\nTASK 7 - SAVE BEST MODEL")
joblib.dump(best_pipe, MODEL_PATH)
print("Saved best_model.pkl")


# Task 8 - reload and test predict
print("\nTASK 8 - RELOAD AND PREDICT")
loaded = joblib.load(MODEL_PATH)
test_rows = pd.DataFrame({
    "carat":   [0.30, 1.50],
    "cut":     [4.0,  1.0],
    "color":   [6.0,  1.0],
    "clarity": [7.0,  1.0],
    "depth":   [61.5, 63.0],
    "table":   [55.0, 61.0],
    "x":       [4.29, 7.22],
    "y":       [4.31, 7.18],
    "z":       [2.64, 4.55]
})
preds  = loaded.predict(test_rows)
probas = loaded.predict_proba(test_rows)[:, 1]
for i in range(len(preds)):
    print(f"  Row {i+1}: class={preds[i]}  P(above median)={probas[i]:.4f}")


# summary table
print("\nSUMMARY - all models")
lr_scores = cross_val_score(
    LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    X_train_sc, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
)
lr_test_auc = roc_auc_score(y_test,
    LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    ).fit(X_train_sc, y_train).predict_proba(X_test_sc)[:, 1])

summary_rows = [{"Model": "LogReg C=1.0", "CV mean": round(lr_scores.mean(), 4),
                 "CV std": round(lr_scores.std(), 4), "Test AUC": round(lr_test_auc, 4)}]
test_auc_map = {
    "Decision Tree (d=5)": roc_auc_score(y_test, dt_ctrl.predict_proba(X_test_sc)[:, 1]),
    "Random Forest":       rf_auc,
    "Gradient Boosting":   gb_auc,
}
for row in cv_rows[1:]:
    summary_rows.append({"Model": row["Model"], "CV mean": row["Mean AUC"],
                          "CV std": row["Std"], "Test AUC": round(test_auc_map.get(row["Model"], 0), 4)})
summary_rows.append({"Model": "Best RF Pipeline", "CV mean": round(gs.best_score_, 4),
                     "CV std": "-", "Test AUC": round(best_auc, 4)})

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT + "/model_summary.csv", index=False)

print("\nDone. Outputs in", OUT)
