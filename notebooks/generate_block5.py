#!/usr/bin/env python3
"""Generate prompt_block_5.ipynb — RandomForest, GradientBoosting, XGBoost, LightGBM + RandomizedSearchCV."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 5 — Ensemble Models: RF, GBM, XGBoost & LightGBM

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

**Models (with RandomizedSearchCV, cv=5, n_iter=40, scoring='neg_root_mean_squared_error'):**
1. RandomForestRegressor
2. GradientBoostingRegressor
3. XGBRegressor
4. LGBMRegressor

**Plus Block 4 baselines for comparison (LinearRegression + DecisionTree).**""")

# CELL 1: IMPORTS
md("""---
## 1. Imports & Data Loading""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import category_encoders as ce

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

# CELL 2: FEATURE ENGINEERING
md("""---
## 2. Feature Engineering (from Block 3)

Pipeline: target split, interactions, bool->int, 80/20 split, impute, encode, scale. All fits on TRAIN only.""")
code("""target = "Yield_tons_per_hectare"
y = df[target].copy()
X = df.drop(columns=[target]).copy()

numeric_features = [
    "Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
    "soil_pH", "humidity_pct", "sunlight_hours",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]
low_card_cat = ["Region", "Soil_Type", "Weather_Condition", "irrigation_type"]
high_card_cat = ["Crop"]
binary_features = ["Fertilizer_Used", "Irrigation_Used"]
all_cat = low_card_cat + high_card_cat + binary_features

# Interactions
X["Rainfall_x_Temp"] = X["Rainfall_mm"] * X["Temperature_Celsius"]
X["NPK_product"] = X["N"] * X["P"] * X["K"]
X["pH_x_Humidity"] = X["soil_pH"] * X["humidity_pct"]
numeric_features = numeric_features + ["Rainfall_x_Temp", "NPK_product", "pH_x_Humidity"]

# Bool->int
for col in binary_features:
    X[col] = X[col].astype(int)

# Duplicates
dup_mask = X.duplicated(keep="first")
if dup_mask.sum() > 0:
    X = X[~dup_mask]; y = y[~dup_mask]

# 80/20 split BEFORE fits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: X {X_train.shape}, y {y_train.shape}  Test: X {X_test.shape}, y {y_test.shape}")

# Impute (fit on train)
num_imp = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(num_imp.fit_transform(X_train[numeric_features]), columns=numeric_features, index=X_train.index)
X_test_num = pd.DataFrame(num_imp.transform(X_test[numeric_features]), columns=numeric_features, index=X_test.index)
cat_imp = SimpleImputer(strategy="most_frequent")
X_train_cat = pd.DataFrame(cat_imp.fit_transform(X_train[all_cat]), columns=all_cat, index=X_train.index)
X_test_cat = pd.DataFrame(cat_imp.transform(X_test[all_cat]), columns=all_cat, index=X_test.index)

# Encode (fit on train)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", drop="first")
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train_cat[low_card_cat]), columns=ohe.get_feature_names_out(low_card_cat), index=X_train.index)
X_test_ohe = pd.DataFrame(ohe.transform(X_test_cat[low_card_cat]), columns=ohe.get_feature_names_out(low_card_cat), index=X_test.index)
te = ce.TargetEncoder(cols=["Crop"], handle_missing="value", handle_unknown="value")
X_train_te = pd.DataFrame(te.fit_transform(X_train_cat[high_card_cat], y_train), columns=high_card_cat, index=X_train.index)
X_test_te = pd.DataFrame(te.transform(X_test_cat[high_card_cat]), columns=high_card_cat, index=X_test.index)

# Scale (fit on train)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numeric_features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=numeric_features, index=X_test.index)

# Combine
X_train_final = pd.concat([X_train_scaled, X_train_ohe, X_train_te, X_train_cat[binary_features].astype(int)], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_ohe, X_test_te, X_test_cat[binary_features].astype(int)], axis=1)
print(f"Final: Train {X_train_final.shape}  Test {X_test_final.shape}")
assert X_train_final.isna().sum().sum() == 0 and X_test_final.isna().sum().sum() == 0
print("No NaN values. Ready for modelling.")""")

# CELL 3: BLOCK 4 BASELINES
md("""---
## 3. Block 4 Baselines: LinearRegression + DecisionTree

Fitted quickly for the complete results_dict comparison.""")
code("""# ====================================================================
# LinearRegression (no tuning)
# ====================================================================
lr = LinearRegression()
lr.fit(X_train_final, y_train)
y_pred_lr_test = lr.predict(X_test_final)

results_dict = {
    "LinearRegression": {
        "model": "LinearRegression",
        "test_mae": mean_absolute_error(y_test, y_pred_lr_test),
        "test_mse": mean_squared_error(y_test, y_pred_lr_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
        "test_r2": r2_score(y_test, y_pred_lr_test),
    }
}

# DecisionTree (best params from Block 4 GridSearchCV)
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
dt.fit(X_train_final, y_train)
y_pred_dt_test = dt.predict(X_test_final)

results_dict["DecisionTree"] = {
    "model": "DecisionTree",
    "test_mae": mean_absolute_error(y_test, y_pred_dt_test),
    "test_mse": mean_squared_error(y_test, y_pred_dt_test),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_dt_test)),
    "test_r2": r2_score(y_test, y_pred_dt_test),
}

print("Block 4 baselines complete:")
for name in ["LinearRegression", "DecisionTree"]:
    r = results_dict[name]
    print(f"  {name:20s}  RMSE={r['test_rmse']:.4f}  R2={r['test_r2']:.4f}")""")

# HELPER
md("""---
## 4. Helper: Train + Evaluate a Model with RandomizedSearchCV

All models share: cv=5, n_iter=40, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1.""")
code("""def train_with_random_search(model, param_dist, name, X_tr, y_tr, X_te, y_te, n_iter=40):
    \"\"\"Run RandomizedSearchCV, evaluate best on test, return results dict.\"\"\"
    rs = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    print(f"  Running RandomizedSearchCV for {name}...")
    rs.fit(X_tr, y_tr)
    best = rs.best_estimator_
    y_pred = best.predict(X_te)

    results = {
        "model": name,
        "best_params": rs.best_params_,
        "best_cv_score": -rs.best_score_,  # convert neg RMSE to RMSE
        "test_mae": mean_absolute_error(y_te, y_pred),
        "test_mse": mean_squared_error(y_te, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
        "test_r2": r2_score(y_te, y_pred),
    }
    print(f"  {name}: best RMSE={results['test_rmse']:.4f}  R2={results['test_r2']:.4f}")
    print(f"    Best params: {rs.best_params_}")
    return results, best""")

# CELL 5: RANDOM FOREST
md("""---
## 5. RandomForestRegressor

Hyperparameter ranges: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.""")
code("""print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)
rf_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
rf_results, best_rf = train_with_random_search(
    RandomForestRegressor(random_state=42, n_jobs=1),
    rf_param_dist, "RandomForest", X_train_final, y_train, X_test_final, y_test
)
results_dict["RandomForest"] = rf_results""")

# CELL 6: GRADIENT BOOSTING
md("""---
## 6. GradientBoostingRegressor

Hyperparameter ranges: n_estimators, max_depth, learning_rate, subsample, min_samples_leaf.""")
code("""print("=" * 60)
print("GRADIENT BOOSTING")
print("=" * 60)
gbm_param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "min_samples_leaf": [1, 2, 5],
}
gbm_results, best_gbm = train_with_random_search(
    GradientBoostingRegressor(random_state=42),
    gbm_param_dist, "GradientBoosting", X_train_final, y_train, X_test_final, y_test
)
results_dict["GradientBoosting"] = gbm_results""")

# CELL 7: XGBOOST
md("""---
## 7. XGBoost Regressor

Hyperparameter ranges: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda.""")
code("""print("=" * 60)
print("XGBOOST")
print("=" * 60)
xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [0, 0.1, 1],
}
xgb_results, best_xgb = train_with_random_search(
    xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
    xgb_param_dist, "XGBoost", X_train_final, y_train, X_test_final, y_test
)
results_dict["XGBoost"] = xgb_results""")

# CELL 8: LIGHTGBM
md("""---
## 8. LightGBM Regressor

Hyperparameter ranges: n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda.""")
code("""print("=" * 60)
print("LIGHTGBM")
print("=" * 60)
lgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [0, 0.1, 1],
}
lgb_results, best_lgb = train_with_random_search(
    lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
    lgb_param_dist, "LightGBM", X_train_final, y_train, X_test_final, y_test
)
results_dict["LightGBM"] = lgb_results""")

# CELL 9: RESULTS COMPARISON
md("""---
## 9. Model Comparison Table

All 6 models compared on test set metrics.""")
code("""# ====================================================================
# 9.  COMPARISON TABLE
# ====================================================================

print("=" * 72)
print("  FINAL MODEL COMPARISON (Test Set)")
print("=" * 72)
print(f"{'Model':<22s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s} {'Best CV RMSE':>14s}")
print("-" * 68)

best_r2_model = None
best_r2_val = -np.inf
best_rmse_model = None
best_rmse_val = np.inf

for model_name in ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]:
    r = results_dict.get(model_name)
    if r is None:
        continue
    cv_str = f"{r.get('best_cv_score', 0):.4f}" if 'best_cv_score' in r else "  N/A  "
    print(f"{model_name:<22s} {r['test_rmse']:>10.4f} {r['test_mae']:>10.4f} {r['test_r2']:>10.4f} {cv_str:>14s}")
    if r['test_r2'] > best_r2_val:
        best_r2_val = r['test_r2']
        best_r2_model = model_name
    if r['test_rmse'] < best_rmse_val:
        best_rmse_val = r['test_rmse']
        best_rmse_model = model_name

print()
print(f"Best R2:   {best_r2_model} ({best_r2_val:.4f})")
print(f"Best RMSE: {best_rmse_model} ({best_rmse_val:.4f})")

# Save best model
best_model_name = best_rmse_model
best_model_map = {
    "LinearRegression": lr, "DecisionTree": dt,
    "RandomForest": best_rf, "GradientBoosting": best_gbm,
    "XGBoost": best_xgb, "LightGBM": best_lgb,
}
best_model = best_model_map.get(best_model_name)
print(f"\\nBest model ({best_model_name}) saved for export.")""")

# ========== NEW: Cross-validation mean +/- std reporting ==========
md("""---
## 9b. Cross-Validation Stability (5-fold)

5-fold cross-validation on the training set to report mean ± std for RMSE and R².
This provides a more robust estimate than a single train-test split and
complements the RandomizedSearchCV best-score values above.""")
code("""print("=" * 72)
print("  5-FOLD CROSS-VALIDATION RESULTS (mean +/- std)")
print("=" * 72)

cv_rows = []
model_items = [
    ("LinearRegression", lr),
    ("DecisionTree", dt),
    ("RandomForest", best_rf),
    ("GradientBoosting", best_gbm),
    ("XGBoost", best_xgb),
    ("LightGBM", best_lgb),
]

for name, model in model_items:
    print(f"  Running 5-fold CV for {name}...")
    rmse_scores = -cross_val_score(model, X_train_final, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    r2_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring="r2", n_jobs=-1)
    cv_rows.append({
        "Model": name,
        "CV_RMSE_mean": rmse_scores.mean(),
        "CV_RMSE_std": rmse_scores.std(),
        "CV_R2_mean": r2_scores.mean(),
        "CV_R2_std": r2_scores.std(),
    })
    print(f"    {name:20s}  RMSE = {rmse_scores.mean():.4f} +/- {rmse_scores.std():.4f}  |  R2 = {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}")

cv_df = pd.DataFrame(cv_rows).sort_values("CV_RMSE_mean", ascending=True).reset_index(drop=True)
print()
print("Cross-validation summary (sorted by CV RMSE):")
print(f"{'Model':<20s} {'RMSE mean':>10s} {'+/-':>5s} {'std':>10s} {'R2 mean':>10s} {'+/-':>5s} {'std':>10s}")
print("-" * 72)
for _, row in cv_df.iterrows():
    print(f"{row['Model']:<20s} {row['CV_RMSE_mean']:>10.4f} {'+/-':>5s} {row['CV_RMSE_std']:>10.4f} {row['CV_R2_mean']:>10.4f} {'+/-':>5s} {row['CV_R2_std']:>10.4f}")

best_cv = cv_df.iloc[0]
best_split = min(results_dict, key=lambda k: results_dict[k]["test_rmse"])
print(f"\\nCV best:        {best_cv['Model']} (RMSE={best_cv['CV_RMSE_mean']:.4f} +/- {best_cv['CV_RMSE_std']:.4f})")
print(f"Single-split best: {best_split} (RMSE={results_dict[best_split]['test_rmse']:.4f})")
if best_cv['Model'] != best_split:
    print(f"\\nNOTE: CV and single-split disagree on the best model. This is normal when different")
    print(f"data subsets produce slightly different rankings. CV is more reliable.")""")

# CELL 10: FEATURE IMPORTANCE
md("""---
## 10. Feature Importance (Top 15) — Best Tree-Based Model

If the best model supports feature importances, plot the top 15.""")
code("""# ====================================================================
# 10. FEATURE IMPORTANCE PLOT
# ====================================================================

import matplotlib.pyplot as plt

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_names = X_train_final.columns
    idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(15), importances[idx][::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(15))
    ax.set_yticklabels([feat_names[i] for i in idx[::-1]])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 15 Feature Importances — {best_model_name}", fontweight="bold")
    ax.invert_yaxis()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"../data/feature_importance_{best_model_name.lower()}.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Feature importance plot saved for {best_model_name}.")
else:
    print(f"{best_model_name} does not support feature importances.")""")

# CELL 11: SAVE
md("""---
## 11. Save Fitted Objects

All transformers, all 6 models, and results_dict saved for export and Block 6.""")
code("""fitted_objects = {
    "num_imputer": num_imp, "cat_imputer": cat_imp,
    "ohe": ohe, "target_encoder": te, "scaler": scaler,
    "linear_regression": lr, "decision_tree": dt,
    "random_forest": best_rf, "gradient_boosting": best_gbm,
    "xgboost": best_xgb, "lightgbm": best_lgb,
    "results_dict": results_dict,
    "best_model": best_model,
    "best_model_name": best_model_name,
}
print("All fitted objects ready for joblib export:")
for k in fitted_objects:
    print(f"  {k}")
print(f"\\nUse: joblib.dump(fitted_objects, 'models_and_transformers.pkl')")""")

md("""---
**Prompt Block 5 Complete.** 4 ensemble models trained with RandomizedSearchCV, evaluated, and compared against Block 4 baselines. Best model identified.""")
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_5.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_5.ipynb with {len(cells)} cells")
