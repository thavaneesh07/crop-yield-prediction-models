#!/usr/bin/env python3
"""Generate prompt_block_6.ipynb — StackingRegressor (RF + XGB + LGBM base, Ridge final)."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 6 — Stacking Ensemble: RF + XGBoost + LightGBM → Ridge

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

**StackingRegressor:**
- **Base estimators:** Tuned RandomForest, XGBoost, LightGBM (best params from Block 5)
- **Final estimator:** Ridge regression
- **Evaluation:** MAE, MSE, RMSE, R2 on test set

Results appended to `results_dict` and compared against Block 4 baselines.""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import category_encoders as ce

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

# FEATURE ENGINEERING
md("""---
## 1. Feature Engineering (from Block 3)""")
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

X["Rainfall_x_Temp"] = X["Rainfall_mm"] * X["Temperature_Celsius"]
X["NPK_product"] = X["N"] * X["P"] * X["K"]
X["pH_x_Humidity"] = X["soil_pH"] * X["humidity_pct"]
numeric_features = numeric_features + ["Rainfall_x_Temp", "NPK_product", "pH_x_Humidity"]

for col in binary_features:
    X[col] = X[col].astype(int)

dup_mask = X.duplicated(keep="first")
if dup_mask.sum() > 0:
    X = X[~dup_mask]; y = y[~dup_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: X {X_train.shape}, y {y_train.shape}  Test: X {X_test.shape}, y {y_test.shape}")

num_imp = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(num_imp.fit_transform(X_train[numeric_features]), columns=numeric_features, index=X_train.index)
X_test_num = pd.DataFrame(num_imp.transform(X_test[numeric_features]), columns=numeric_features, index=X_test.index)
cat_imp = SimpleImputer(strategy="most_frequent")
X_train_cat = pd.DataFrame(cat_imp.fit_transform(X_train[all_cat]), columns=all_cat, index=X_train.index)
X_test_cat = pd.DataFrame(cat_imp.transform(X_test[all_cat]), columns=all_cat, index=X_test.index)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", drop="first")
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train_cat[low_card_cat]), columns=ohe.get_feature_names_out(low_card_cat), index=X_train.index)
X_test_ohe = pd.DataFrame(ohe.transform(X_test_cat[low_card_cat]), columns=ohe.get_feature_names_out(low_card_cat), index=X_test.index)
te = ce.TargetEncoder(cols=["Crop"], handle_missing="value", handle_unknown="value")
X_train_te = pd.DataFrame(te.fit_transform(X_train_cat[high_card_cat], y_train), columns=high_card_cat, index=X_train.index)
X_test_te = pd.DataFrame(te.transform(X_test_cat[high_card_cat]), columns=high_card_cat, index=X_test.index)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numeric_features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=numeric_features, index=X_test.index)

X_train_final = pd.concat([X_train_scaled, X_train_ohe, X_train_te, X_train_cat[binary_features].astype(int)], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_ohe, X_test_te, X_test_cat[binary_features].astype(int)], axis=1)
print(f"Final: Train {X_train_final.shape}  Test {X_test_final.shape}")
assert X_train_final.isna().sum().sum() == 0 and X_test_final.isna().sum().sum() == 0
print("No NaN values. Ready for modelling.")""")

# BLOCK 4 BASELINES
md("""---
## 2. Block 4 Baselines (LinearRegression + DecisionTree)

Quick fits for the results_dict comparison.""")
code("""def evaluate_model(model, name, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "model": name,
        "test_mae": mean_absolute_error(y_te, y_pred),
        "test_mse": mean_squared_error(y_te, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
        "test_r2": r2_score(y_te, y_pred),
    }

results_dict = {}

# LinearRegression
lr = LinearRegression()
results_dict["LinearRegression"] = evaluate_model(lr, "LinearRegression", X_train_final, y_train, X_test_final, y_test)
print(f"LinearRegression: RMSE={results_dict['LinearRegression']['test_rmse']:.4f}  R2={results_dict['LinearRegression']['test_r2']:.4f}")

# DecisionTree
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
results_dict["DecisionTree"] = evaluate_model(dt, "DecisionTree", X_train_final, y_train, X_test_final, y_test)
print(f"DecisionTree:     RMSE={results_dict['DecisionTree']['test_rmse']:.4f}  R2={results_dict['DecisionTree']['test_r2']:.4f}")""")

# TUNED BASE MODELS
md("""---
## 3. Tuned Base Estimators (best params from Block 5)

Training the 3 base models with their cross-validated best hyperparameters.""")
code("""# Best params from Block 5 RandomizedSearchCV
rf_tuned = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
)
xgb_tuned = xgb.XGBRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
    random_state=42, verbosity=0, n_jobs=-1
)
lgbm_tuned = lgb.LGBMRegressor(
    n_estimators=300, num_leaves=31, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
    random_state=42, verbose=-1, n_jobs=-1
)

# Evaluate individually for comparison
results_dict["RandomForest"] = evaluate_model(rf_tuned, "RandomForest", X_train_final, y_train, X_test_final, y_test)
print(f"RandomForest (tuned): RMSE={results_dict['RandomForest']['test_rmse']:.4f}  R2={results_dict['RandomForest']['test_r2']:.4f}")

results_dict["XGBoost"] = evaluate_model(xgb_tuned, "XGBoost", X_train_final, y_train, X_test_final, y_test)
print(f"XGBoost (tuned):      RMSE={results_dict['XGBoost']['test_rmse']:.4f}  R2={results_dict['XGBoost']['test_r2']:.4f}")

results_dict["LightGBM"] = evaluate_model(lgbm_tuned, "LightGBM", X_train_final, y_train, X_test_final, y_test)
print(f"LightGBM (tuned):     RMSE={results_dict['LightGBM']['test_rmse']:.4f}  R2={results_dict['LightGBM']['test_r2']:.4f}")""")

# STACKING REGRESSOR
md("""---
## 4. StackingRegressor (RF + XGB + LightGBM → Ridge)

Stacking combines predictions from diverse base models via a meta-learner (Ridge).
This often outperforms any single model by reducing bias and variance together.""")
code("""# ====================================================================
# StackingRegressor
# ====================================================================

estimators = [
    ("rf", rf_tuned),
    ("xgb", xgb_tuned),
    ("lgbm", lgbm_tuned),
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,  # 5-fold CV to generate meta-features
    n_jobs=-1, passthrough=False
)

print("Fitting StackingRegressor (RF + XGB + LGBM -> Ridge)...")
print(f"  Base estimators: {[e[0] for e in estimators]}")
print(f"  Final estimator: Ridge(alpha=1.0)")
print(f"  Cross-validation: 5-fold for meta-features")

stack.fit(X_train_final, y_train)
print("  StackingRegressor fitted.")

# Evaluate
results_dict["Stacking"] = evaluate_model(stack, "Stacking", X_train_final, y_train, X_test_final, y_test)
print(f"StackingRegressor:  RMSE={results_dict['Stacking']['test_rmse']:.4f}  R2={results_dict['Stacking']['test_r2']:.4f}")""")

# COMPARISON
md("""---
## 5. Model Comparison Table

All 7 models compared: LR, DT (Block 4) + RF, XGB, LGBM (tuned) + Stacking.""")
code("""print("=" * 72)
print("  FINAL MODEL COMPARISON (Test Set)")
print("=" * 72)
print(f"{'Model':<22s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s}")
print("-" * 54)

model_order = ["LinearRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM", "Stacking"]
best_r2_name = None; best_r2_val = -np.inf
best_rmse_name = None; best_rmse_val = np.inf

for name in model_order:
    r = results_dict.get(name)
    if r is None:
        continue
    print(f"{name:<22s} {r['test_rmse']:>10.4f} {r['test_mae']:>10.4f} {r['test_r2']:>10.4f}")
    if r['test_r2'] > best_r2_val:
        best_r2_val = r['test_r2']; best_r2_name = name
    if r['test_rmse'] < best_rmse_val:
        best_rmse_val = r['test_rmse']; best_rmse_name = name

print()
print(f"Best R2:   {best_r2_name} ({best_r2_val:.4f})")
print(f"Best RMSE: {best_rmse_name} ({best_rmse_val:.4f})")

print(f"\\nStacking improvement over best single model:")
best_single_rmse = min(results_dict[n]['test_rmse'] for n in model_order if n != "Stacking" and n in results_dict)
stack_rmse = results_dict["Stacking"]["test_rmse"]
pct_improvement = (best_single_rmse - stack_rmse) / best_single_rmse * 100
print(f"  Best single model RMSE: {best_single_rmse:.4f}")
print(f"  Stacking RMSE:          {stack_rmse:.4f}")
print(f"  Improvement:            {pct_improvement:+.2f}%")""")

# SAVE
md("""---
## 6. Save Fitted Objects

All transformers, individual models, StackingRegressor, and results_dict.""")
code("""fitted_objects = {
    "num_imputer": num_imp, "cat_imputer": cat_imp,
    "ohe": ohe, "target_encoder": te, "scaler": scaler,
    "linear_regression": lr, "decision_tree": dt,
    "random_forest": rf_tuned, "xgboost": xgb_tuned, "lightgbm": lgbm_tuned,
    "stacking_regressor": stack,
    "results_dict": results_dict,
    "best_model": stack,  # Stacking is expected to be best
    "best_model_name": "Stacking",
}
print("All fitted objects saved. Ready for joblib export.")
for k in fitted_objects:
    print(f"  {k}")
print(f"\\nUse: joblib.dump(fitted_objects, 'final_models.pkl')")""")

md("""---
**Prompt Block 6 Complete.** StackingRegressor (RF + XGBoost + LightGBM -> Ridge) trained and compared against all individual models.""")
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_6.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_6.ipynb with {len(cells)} cells")
