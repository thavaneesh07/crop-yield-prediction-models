#!/usr/bin/env python3
"""Generate prompt_block_7.ipynb — Model Comparison DataFrame + Grouped Bar Chart."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 7 — Final Model Comparison

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

Builds a DataFrame comparing all 6 models (LR, DT, RF, XGB, LGBM, Stacking) on MAE, MSE, RMSE, R2, sorted by RMSE ascending. Plots a grouped bar chart and identifies the best model.""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.titlesize": 14})

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

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
print("No NaN. Ready.")""")

md("""---
## 2. Train All Models

**Note on hyperparameters:** The RandomForest, XGBoost, and LightGBM params used below
are the **best values discovered by RandomizedSearchCV** (cv=5, n_iter=40) in **Block 5**.
They are hardcoded here for efficiency — re-running the full search in each notebook
would be redundant since the same training data and pipeline are used.

See `prompt_block_5.ipynb` for the complete search output including all candidate
parameter combinations and cross-validation scores.""")
code("""def evaluate(model, name, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {"Model": name, "MAE": mean_absolute_error(y_te, y_pred),
            "MSE": mean_squared_error(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "R2": r2_score(y_te, y_pred)}

results = []

# --- LinearRegression ---
results.append(evaluate(LinearRegression(), "LinearRegression", X_train_final, y_train, X_test_final, y_test))
print(f"  LR: RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

# --- DecisionTree ---
results.append(evaluate(DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42), "DecisionTree", X_train_final, y_train, X_test_final, y_test))
print(f"  DT:  RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

# --- RandomForest ---
rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
results.append(evaluate(rf, "RandomForest", X_train_final, y_train, X_test_final, y_test))
print(f"  RF:  RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42, verbosity=0, n_jobs=-1)
results.append(evaluate(xgb_model, "XGBoost", X_train_final, y_train, X_test_final, y_test))
print(f"  XGB: RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

# --- LightGBM ---
lgbm_model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42, verbose=-1, n_jobs=-1)
results.append(evaluate(lgbm_model, "LightGBM", X_train_final, y_train, X_test_final, y_test))
print(f"  LGB: RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

# --- StackingRegressor ---
estimators = [("rf", rf), ("xgb", xgb_model), ("lgbm", lgbm_model)]
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
print("  Fitting StackingRegressor (may take a moment)...")
results.append(evaluate(stack, "Stacking", X_train_final, y_train, X_test_final, y_test))
print(f"  STK: RMSE={results[-1]['RMSE']:.4f}  R2={results[-1]['R2']:.4f}")

print(f"\\nAll {len(results)} models trained.")""")

md("""---
## 3. Comparison DataFrame (sorted by RMSE ascending)""")
code("""# Build sorted DataFrame
comparison_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
comparison_df.index = comparison_df.index + 1  # 1-based index

print("=" * 68)
print("  MODEL COMPARISON (sorted by RMSE)")
print("=" * 68)
print(comparison_df.to_string(index=True, float_format=lambda x: f"{x:.4f}"))
print()

# Identify best by each metric
best_rmse = comparison_df.loc[comparison_df["RMSE"].idxmin()]
best_r2 = comparison_df.loc[comparison_df["R2"].idxmax()]

print(f"Best by RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
print(f"Best by R2:   {best_r2['Model']} ({best_r2['R2']:.4f})")

if best_rmse["Model"] != best_r2["Model"]:
    print(f"\\n⚠ DISAGREEMENT: {best_rmse['Model']} wins by RMSE but {best_r2['Model']} wins by R2.")
    print(f"   RMSE and R2 can disagree when models have different error distributions.")
    print(f"   RMSE penalizes large errors more heavily; R2 measures explained variance.")
else:
    print(f"\\n✓ Agreement: {best_rmse['Model']} wins on both RMSE and R2.")

# Best model references (for later export)
best_model_name = best_rmse["Model"]""")

md("""---
## 4. Grouped Bar Chart: RMSE and R2 Comparison""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = sns.color_palette("viridis", len(comparison_df))
model_names = comparison_df["Model"].tolist()

# RMSE bar chart
ax1 = axes[0]
bars1 = ax1.bar(range(len(model_names)), comparison_df["RMSE"].values, color=colors, edgecolor="white")
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=30, ha="right")
ax1.set_ylabel("RMSE (tons/hectare)")
ax1.set_title("Test Set RMSE by Model (lower is better)", fontweight="bold")
for bar, val in zip(bars1, comparison_df["RMSE"].values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# R2 bar chart
ax2 = axes[1]
bars2 = ax2.bar(range(len(model_names)), comparison_df["R2"].values, color=colors, edgecolor="white")
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=30, ha="right")
ax2.set_ylabel("R² Score")
ax2.set_title("Test Set R² by Model (higher is better)", fontweight="bold")
for bar, val in zip(bars2, comparison_df["R2"].values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Highlight best
best_idx = comparison_df["RMSE"].idxmin() - 1
bars1[best_idx].set_color("crimson")
bars1[best_idx].set_edgecolor("black")
bars1[best_idx].set_linewidth(2)

best_r2_idx = comparison_df["R2"].idxmax() - 1
bars2[best_r2_idx].set_color("crimson")
bars2[best_r2_idx].set_edgecolor("black")
bars2[best_r2_idx].set_linewidth(2)

sns.despine()
plt.tight_layout()
plt.savefig("../data/model_comparison_barchart.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Grouped bar chart saved to data/model_comparison_barchart.png")""")

md("""---
## 5. Summary""")
code("""print("=" * 68)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 68)
print(f"{'Rank':<6s} {'Model':<20s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s}")
print("-" * 58)
for i, row in comparison_df.iterrows():
    rank = i  # already sorted
    print(f"{rank:<6d} {row['Model']:<20s} {row['RMSE']:>10.4f} {row['MAE']:>10.4f} {row['R2']:>10.4f}")

print(f"\\nBest  RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
print(f"Best  R²:   {best_r2['Model']} ({best_r2['R2']:.4f})")

# Compute improvement over baseline
baseline_rmse = comparison_df.loc[comparison_df["Model"] == "LinearRegression", "RMSE"].values[0]
best_rmse_val = comparison_df["RMSE"].min()
improvement = (baseline_rmse - best_rmse_val) / baseline_rmse * 100
print(f"\\nImprovement from LinearRegression to best model: {improvement:+.2f}%")""")

md("""---
**Prompt Block 7 Complete.** Full comparison DataFrame, grouped bar chart with best model highlighted, and clear identification of the best model by RMSE and R2.""")
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_7.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_7.ipynb with {len(cells)} cells")
