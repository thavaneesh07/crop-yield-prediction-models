#!/usr/bin/env python3
"""Generate prompt_block_4.ipynb — LinearRegression + DecisionTree + GridSearchCV."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 4 — Baseline Models: Linear Regression & Decision Tree

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

**Models:**
1. **LinearRegression** (baseline, no tuning)
2. **DecisionTreeRegressor** (tuned via GridSearchCV on `max_depth` and `min_samples_leaf`)

**Evaluation:** MAE, MSE, RMSE, R2 on the test set. Results stored in a dictionary.

**Pipeline:** Load data -> feature engineering (from Block 3) -> train models -> evaluate -> store results""")

# CELL 1: IMPORTS & LOAD
md("""---
## 1. Imports & Data Loading""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

# CELL 2: FEATURE ENGINEERING (from Block 3)
md("""---
## 2. Feature Engineering (reproduced from Block 3)

Following the same pipeline: split target, create interactions, convert bools, split 80/20, impute, encode, scale.""")
code("""target = "Yield_tons_per_hectare"
y = df[target].copy()
X = df.drop(columns=[target]).copy()

# Feature groups
numeric_features = [
    "Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
    "soil_pH", "humidity_pct", "sunlight_hours",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]
low_card_cat = ["Region", "Soil_Type", "Weather_Condition", "irrigation_type"]
high_card_cat = ["Crop"]
binary_features = ["Fertilizer_Used", "Irrigation_Used"]
all_cat = low_card_cat + high_card_cat + binary_features

# Interaction features
X["Rainfall_x_Temp"] = X["Rainfall_mm"] * X["Temperature_Celsius"]
X["NPK_product"] = X["N"] * X["P"] * X["K"]
X["pH_x_Humidity"] = X["soil_pH"] * X["humidity_pct"]
numeric_features = numeric_features + ["Rainfall_x_Temp", "NPK_product", "pH_x_Humidity"]

# Bool to int
for col in binary_features:
    X[col] = X[col].astype(int)

# Drop duplicates
dup_mask = X.duplicated(keep="first")
if dup_mask.sum() > 0:
    X = X[~dup_mask]; y = y[~dup_mask]

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: X {X_train.shape}, y {y_train.shape}")
print(f"Test:  X {X_test.shape}, y {y_test.shape}")

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
print(f"Total features: {X_train_final.shape[1]}")
assert X_train_final.isna().sum().sum() == 0 and X_test_final.isna().sum().sum() == 0, "NaN found!"
print("No missing values. Ready for modelling.")""")

# CELL 3: LINEAR REGRESSION
md("""---
## 3. Linear Regression (Baseline)

No hyperparameter tuning — LinearRegression is a closed-form solution.""")
code("""# ====================================================================
# 3.  LINEAR REGRESSION
# ====================================================================

lr = LinearRegression()
lr.fit(X_train_final, y_train)

y_pred_lr_train = lr.predict(X_train_final)
y_pred_lr_test = lr.predict(X_test_final)

lr_results = {
    "model": "LinearRegression",
    "train_mae": mean_absolute_error(y_train, y_pred_lr_train),
    "train_mse": mean_squared_error(y_train, y_pred_lr_train),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_lr_train)),
    "train_r2": r2_score(y_train, y_pred_lr_train),
    "test_mae": mean_absolute_error(y_test, y_pred_lr_test),
    "test_mse": mean_squared_error(y_test, y_pred_lr_test),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
    "test_r2": r2_score(y_test, y_pred_lr_test),
}

print("Linear Regression Results:")
print(f"  Train: MAE={lr_results['train_mae']:.4f}  RMSE={lr_results['train_rmse']:.4f}  R2={lr_results['train_r2']:.4f}")
print(f"  Test:  MAE={lr_results['test_mae']:.4f}  RMSE={lr_results['test_rmse']:.4f}  R2={lr_results['test_r2']:.4f}")""")

# CELL 4: DECISION TREE + GRIDSEARCHCV
md("""---
## 4. Decision Tree Regressor (GridSearchCV)

Hyperparameter grid:
- `max_depth`: [3, 5, 7, 10, 15, 20, None]
- `min_samples_leaf`: [1, 5, 10, 20, 50]

Using 5-fold cross-validation on the training set.""")
code("""# ====================================================================
# 4.  DECISION TREE WITH GRIDSEARCHCV
# ====================================================================

param_grid = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}

dt_base = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(
    dt_base, param_grid, cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1
)

print("Running GridSearchCV for DecisionTreeRegressor...")
grid_search.fit(X_train_final, y_train)

best_dt = grid_search.best_estimator_
y_pred_dt_train = best_dt.predict(X_train_final)
y_pred_dt_test = best_dt.predict(X_test_final)

dt_results = {
    "model": "DecisionTree (GridSearchCV)",
    "best_params": grid_search.best_params_,
    "train_mae": mean_absolute_error(y_train, y_pred_dt_train),
    "train_mse": mean_squared_error(y_train, y_pred_dt_train),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_dt_train)),
    "train_r2": r2_score(y_train, y_pred_dt_train),
    "test_mae": mean_absolute_error(y_test, y_pred_dt_test),
    "test_mse": mean_squared_error(y_test, y_pred_dt_test),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_dt_test)),
    "test_r2": r2_score(y_test, y_pred_dt_test),
}

print(f"\\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
print(f"\\nDecision Tree Results:")
print(f"  Train: MAE={dt_results['train_mae']:.4f}  RMSE={dt_results['train_rmse']:.4f}  R2={dt_results['train_r2']:.4f}")
print(f"  Test:  MAE={dt_results['test_mae']:.4f}  RMSE={dt_results['test_rmse']:.4f}  R2={dt_results['test_r2']:.4f}")""")

# CELL 5: RESULTS SUMMARY
md("""---
## 5. Results Dictionary & Comparison

All evaluation metrics stored in a structured dictionary for later reference.""")
code("""# ====================================================================
# 5.  RESULTS SUMMARY DICTIONARY
# ====================================================================

results_dict = {
    "LinearRegression": lr_results,
    "DecisionTree_CV": dt_results,
}

print("=" * 72)
print("  MODEL COMPARISON SUMMARY")
print("=" * 72)
print(f"{'Model':<30s} {'R2 Test':>10s} {'RMSE Test':>12s} {'MAE Test':>12s}")
print("-" * 66)

for model_name, res in results_dict.items():
    print(f"{model_name:<30s} {res['test_r2']:>10.4f} {res['test_rmse']:>12.4f} {res['test_mae']:>12.4f}")

print()
print("Best test R2: ", end="")
best_model = max(results_dict, key=lambda k: results_dict[k]["test_r2"])
print(f"{best_model} ({results_dict[best_model]['test_r2']:.4f})")

print("Lowest test RMSE: ", end="")
best_rmse = min(results_dict, key=lambda k: results_dict[k]["test_rmse"])
print(f"{best_rmse} ({results_dict[best_rmse]['test_rmse']:.4f})")

print("\\nResults dictionary keys:")
for model_name in results_dict:
    print(f"  {model_name}: {list(results_dict[model_name].keys())}")""")

# CELL 6: SAVE TRANSFORMERS + MODELS
md("""---
## 6. Save Fitted Transformers & Models (for later export)""")
code("""fitted_objects = {
    "num_imputer": num_imp, "cat_imputer": cat_imp,
    "ohe": ohe, "target_encoder": te, "scaler": scaler,
    "linear_regression": lr, "decision_tree_cv": best_dt,
    "results_dict": results_dict,
}
print("Fitted objects ready for joblib export.")
print(f"  Models: LinearRegression, DecisionTree (best from GridSearchCV)")
print(f"  Results: {len(results_dict)} models in results_dict")
print(f"\\nUse: joblib.dump(fitted_objects, 'models_and_transformers.pkl')")""")

md("""---
**Prompt Block 4 Complete.** Linear Regression and Decision Tree (tuned via GridSearchCV) trained and evaluated on the test set. Results stored in `results_dict`.""")
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_4.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_4.ipynb with {len(cells)} cells")
