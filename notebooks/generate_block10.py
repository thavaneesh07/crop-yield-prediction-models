#!/usr/bin/env python3
"""Generate prompt_block_10.ipynb — Export model, scaler, encoders, and results."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 10 — Save Model & Transforms to Disk

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

**Exports:**
- `model.pkl` — best XGBoost model
- `scaler.pkl` — fitted StandardScaler
- `encoders.pkl` — fitted encoders (OneHotEncoder, TargetEncoder, imputers)
- `results.csv` — final model comparison table""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
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
OUTPUT_DIR = "../models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

md("""---
## 1. Feature Engineering""")
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
print(f"Train: X {X_train.shape}  Test: X {X_test.shape}")

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
print("No NaN.")""")

md("""---
## 2. Train Best Model & Build Comparison Table

**Note on hyperparameters:** The RandomForest, XGBoost, and LightGBM params used below
are the **best values discovered by RandomizedSearchCV** (cv=5, n_iter=40) in **Block 5**.
They are hardcoded here for efficiency — see `prompt_block_5.ipynb` for the full search
output including all candidate parameter combinations and cross-validation scores.""")
code("""def evaluate(model, name, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {"Model": name, "MAE": mean_absolute_error(y_te, y_pred),
            "MSE": mean_squared_error(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "R2": r2_score(y_te, y_pred)}

results = []
lr_model = LinearRegression()
results.append(evaluate(lr_model, "LinearRegression", X_train_final, y_train, X_test_final, y_test))
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
results.append(evaluate(dt_model, "DecisionTree", X_train_final, y_train, X_test_final, y_test))

rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42, verbosity=0, n_jobs=-1)
lgbm_model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42, verbose=-1, n_jobs=-1)

results.append(evaluate(rf, "RandomForest", X_train_final, y_train, X_test_final, y_test))
results.append(evaluate(xgb_model, "XGBoost", X_train_final, y_train, X_test_final, y_test))
results.append(evaluate(lgbm_model, "LightGBM", X_train_final, y_train, X_test_final, y_test))

stack = StackingRegressor(estimators=[("rf", rf), ("xgb", xgb_model), ("lgbm", lgbm_model)], final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
results.append(evaluate(stack, "Stacking", X_train_final, y_train, X_test_final, y_test))

comparison_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
print("\\nComparison table:")
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best = comparison_df.iloc[0]
print(f"\\nBest model: {best['Model']} (RMSE={best['RMSE']:.4f}, R2={best['R2']:.4f})")

# Export whichever model actually wins the comparison
model_map = {
    "LinearRegression": lr_model,
    "DecisionTree": dt_model,
    "RandomForest": rf,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "Stacking": stack,
}
best_model = model_map[best["Model"]]
print(f"Exporting: {best['Model']}")""")

# ========== NEW: Cross-validation reporting ==========
md("""---
## 2b. Cross-Validation Stability (5-fold)

5-fold cross-validation on the training set to report mean ± std for RMSE and R².
This provides a more robust estimate than a single train-test split.""")
code("""print("=" * 72)
print("  5-FOLD CROSS-VALIDATION RESULTS (mean +/- std)")
print("=" * 72)

scoring_metrics = {
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

# Collect CV results for each model
cv_rows = []
model_items = [
    ("LinearRegression", lr_model),
    ("DecisionTree", dt_model),
    ("RandomForest", rf),
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
    ("Stacking", stack),
]

for name, model in model_items:
    print(f"  Running 5-fold CV for {name}...")
    # RMSE via neg_root_mean_squared_error (negate to get positive RMSE)
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
print(f"\\nCV Best model: {best_cv['Model']} (RMSE={best_cv['CV_RMSE_mean']:.4f} +/- {best_cv['CV_RMSE_std']:.4f})")
print(f"Single-split best: {best['Model']} (RMSE={best['RMSE']:.4f})")
if best_cv['Model'] != best['Model']:
    print(f"\\nNOTE: CV and single-split disagree on the best model. This is normal when different")
    print(f"data subsets produce slightly different rankings — CV is more reliable.")""")

md("""---
## 3. Save Model & Transformers to Disk""")
code("""# Save best model
model_path = f"{OUTPUT_DIR}/model.pkl"
joblib.dump(best_model, model_path)
print(f"  Saved model:      {os.path.abspath(model_path)}  ({os.path.getsize(model_path)/1024:.1f} KB)")

# Save scaler
scaler_path = f"{OUTPUT_DIR}/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"  Saved scaler:     {os.path.abspath(scaler_path)}  ({os.path.getsize(scaler_path)/1024:.1f} KB)")

# Save all encoders + imputers + feature config in one file
encoders = {
    "num_imputer": num_imp,
    "cat_imputer": cat_imp,
    "onehot_encoder": ohe,
    "target_encoder": te,
    "ohe_columns": ohe.get_feature_names_out(low_card_cat).tolist(),
    "numeric_features": numeric_features,
    "low_card_cat": low_card_cat,
    "high_card_cat": high_card_cat,
    "binary_features": binary_features,
    "all_cat": all_cat,
}
encoders_path = f"{OUTPUT_DIR}/encoders.pkl"
joblib.dump(encoders, encoders_path)
print(f"  Saved encoders:   {os.path.abspath(encoders_path)}  ({os.path.getsize(encoders_path)/1024:.1f} KB)")

# Save comparison table as CSV
csv_path = f"{OUTPUT_DIR}/results.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"  Saved results:    {os.path.abspath(csv_path)}  ({os.path.getsize(csv_path)/1024:.1f} KB)")

print(f"\\n{'='*60}")
print(f"  All files saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"  {'model.pkl':30s} Best XGBoost model")
print(f"  {'scaler.pkl':30s} Fitted StandardScaler (16 features)")
print(f"  {'encoders.pkl':30s} Imputers + OneHot + TargetEncoder + config")
print(f"  {'results.csv':30s} 6-model comparison (RMSE, MAE, R2)")
print(f"{'='*60}")""")

md("""---
## 4. Verify Saved Files Can Be Loaded""")
code("""print("\\nVerifying saved files load correctly...")
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
loaded_encoders = joblib.load(encoders_path)
loaded_csv = pd.read_csv(csv_path)

# Quick test: make a prediction with loaded model
y_pred = loaded_model.predict(X_test_final[:5])
print(f"  Model loaded:  {type(loaded_model).__name__}")
print(f"  Scaler loaded: fitted={hasattr(loaded_scaler, 'mean_')}")
print(f"  Encoders loaded: {list(loaded_encoders.keys())}")
print(f"  Results loaded: {len(loaded_csv)} models x {len(loaded_csv.columns)} cols")
print(f"  Sample prediction: {y_pred[0]:.4f} (actual: {y_test.iloc[0]:.4f})")
print("\\nAll files verified. Export complete.")""")

notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_10.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_10.ipynb with {len(cells)} cells")
