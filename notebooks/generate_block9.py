#!/usr/bin/env python3
"""Generate prompt_block_9.ipynb — 5 random test samples with predictions."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

md("""# Prompt Block 9 — Test Set Sample Predictions

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

Takes 5 random test samples and shows:
- Original (untransformed) feature values
- Actual Yield
- Predicted Yield (from best model)
- % Error

Presented as a clean DataFrame.""")
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import category_encoders as ce

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")""")

md("""---
## 1. Feature Engineering (preserving raw test set for display)""")
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
display_features = numeric_features + low_card_cat + high_card_cat + binary_features

# Interactions (keep raw copies for display)
X["Rainfall_x_Temp"] = X["Rainfall_mm"] * X["Temperature_Celsius"]
X["NPK_product"] = X["N"] * X["P"] * X["K"]
X["pH_x_Humidity"] = X["soil_pH"] * X["humidity_pct"]
numeric_features = numeric_features + ["Rainfall_x_Temp", "NPK_product", "pH_x_Humidity"]

for col in binary_features:
    X[col] = X[col].astype(int)

dup_mask = X.duplicated(keep="first")
if dup_mask.sum() > 0:
    X = X[~dup_mask]; y = y[~dup_mask]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# KEEP RAW TEST COPY for display
X_test_raw = X_test[display_features].copy()
X_test_raw[target] = y_test  # pandas index alignment

print(f"Train: X {X_train.shape}  Test: X {X_test.shape}")
print(f"Raw test copy preserved with {len(display_features)} original features + target")""")

md("""---
## 2. Transform Pipeline (fit on train only)""")
code("""num_imp = SimpleImputer(strategy="median")
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
print(f"Train final: {X_train_final.shape}  Test final: {X_test_final.shape}")
assert X_train_final.isna().sum().sum() == 0 and X_test_final.isna().sum().sum() == 0
print("No NaN.")""")

md("""---
## 3. Train Best Model (XGBoost)

**Note on hyperparameters:** The XGBoost params below are the **best values discovered
by RandomizedSearchCV** (cv=5, n_iter=40) in **Block 5**. They are hardcoded here
for efficiency — see `prompt_block_5.ipynb` for the complete search output with all
candidate parameter combinations and cross-validation scores.""")
code("""best_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
    random_state=42, verbosity=0, n_jobs=-1
)
best_model.fit(X_train_final, y_train)

y_pred = best_model.predict(X_test_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost test RMSE: {rmse:.4f}")""")

md("""---
## 4. Select 5 Random Test Samples & Build Prediction DataFrame

We pick 5 random rows from the test set and display their original (untransformed)
values alongside actual and predicted yield.""")
code("""# Pick 5 random test indices
np.random.seed(42)
sample_indices = np.random.choice(X_test.index, size=5, replace=False)

# Build the display DataFrame
rows = []
for idx in sample_indices:
    actual = y_test.loc[idx]
    pred = best_model.predict(X_test_final.loc[[idx]])[0]
    pct_error = abs(actual - pred) / actual * 100 if actual != 0 else np.nan

    row_data = {
        # Original feature values (untransformed)
        "Region": X_test_raw.loc[idx, "Region"],
        "Soil_Type": X_test_raw.loc[idx, "Soil_Type"],
        "Crop": X_test_raw.loc[idx, "Crop"],
        "Rainfall_mm": round(X_test_raw.loc[idx, "Rainfall_mm"], 1),
        "Temp_C": round(X_test_raw.loc[idx, "Temperature_Celsius"], 1),
        "Fertilizer": "Yes" if X_test_raw.loc[idx, "Fertilizer_Used"] else "No",
        "Irrigation": "Yes" if X_test_raw.loc[idx, "Irrigation_Used"] else "No",
        "Days_to_Harvest": int(X_test_raw.loc[idx, "Days_to_Harvest"]),
        "soil_pH": round(X_test_raw.loc[idx, "soil_pH"], 2),
        "humidity_pct": round(X_test_raw.loc[idx, "humidity_pct"], 1),
        "sunlight_hours": round(X_test_raw.loc[idx, "sunlight_hours"], 1),
        # Predictions
        "Actual_Yield": round(actual, 4),
        "Predicted_Yield": round(pred, 4),
        "%_Error": round(pct_error, 2),
    }
    rows.append(row_data)

pred_df = pd.DataFrame(rows)
pred_df.index = [f"Sample {i+1}" for i in range(len(sample_indices))]

print("=" * 100)
print("  5 RANDOM TEST SAMPLES — ORIGINAL FEATURES, ACTUAL vs PREDICTED YIELD")
print("=" * 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
display(pred_df)

print(f"\\nModel: XGBoost (tuned)  |  Overall Test RMSE: {rmse:.4f}")
print(f"Average % Error on these 5 samples: {pred_df['%_Error'].mean():.2f}%")""")

md("""---
**Observation:** The table shows 5 randomly selected test samples with their original input values (rainfall, temperature, soil type, crop, etc.), the actual measured yield, the model's predicted yield, and the percentage error for each. This gives a transparent, row-level view of how the model performs on unseen data — revealing whether errors are concentrated in specific crops or input ranges.

Notice which samples have the highest and lowest % error. Discrepancies can often be traced to unusual combinations of inputs (e.g., very low rainfall with a high-yield crop) where the model has less training data. Conversely, samples near the centre of the training distribution (average rainfall, common crop) tend to have lower errors.""")

notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_9.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_9.ipynb with {len(cells)} cells")
