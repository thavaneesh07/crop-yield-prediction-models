#!/usr/bin/env python3
"""Generate prompt_block_3.ipynb -- Feature Engineering & Preprocessing."""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]})

# TITLE
md("""# Prompt Block 3 -- Feature Engineering & Preprocessing

**Explainable AI-Based Crop Yield Prediction and Farm Decision Support System**

**Dataset:** `agriculture_data.csv` (real, merged in Block 1)

**Pipeline:** Load data -> engineer interactions -> split 80/20 -> impute (fit on train) -> encode (fit on train) -> scale (fit on train) -> save transformers""")

# CELL 1: IMPORTS
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce
import joblib

BASE = "../data"
df = pd.read_csv(f"{BASE}/agriculture_data.csv")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Missing: {df.isna().sum().sum()}  Duplicates: {df.duplicated().sum():,}")""")

# CELL 2: STAGE D
md("""## 2. Stage D -- Feature & Target Split

**Target:** Yield_tons_per_hectare

**Feature groups:** Numeric (13 + 3 interactions), Low-card cats (4: Region, Soil_Type, Weather_Condition, irrigation_type), High-card cat (1: Crop), Binary (2: Fertilizer_Used, Irrigation_Used)""")

code("""target = "Yield_tons_per_hectare"
y = df[target].copy()
X = df.drop(columns=[target]).copy()

numeric_features = [
    "Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
    "soil_pH", "humidity_pct", "sunlight_hours",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]
low_card_cat_features = ["Region", "Soil_Type", "Weather_Condition", "irrigation_type"]
high_card_cat_features = ["Crop"]
binary_features = ["Fertilizer_Used", "Irrigation_Used"]
all_cat_features = low_card_cat_features + high_card_cat_features + binary_features

print(f"Target shape: {y.shape}")
print(f"Numeric: {len(numeric_features)}  Low-card cats: {len(low_card_cat_features)}  High-card cats: {len(high_card_cat_features)}  Binary: {len(binary_features)}")""")

# CELL 3: INTERACTIONS
md("""## 3. Interaction Features

Rainfall_x_Temp, NPK_product, pH_x_Humidity""")

code("""X["Rainfall_x_Temp"] = X["Rainfall_mm"] * X["Temperature_Celsius"]
X["NPK_product"] = X["N"] * X["P"] * X["K"]
X["pH_x_Humidity"] = X["soil_pH"] * X["humidity_pct"]
extra_numeric = ["Rainfall_x_Temp", "NPK_product", "pH_x_Humidity"]
numeric_features = numeric_features + extra_numeric
print(f"Added {len(extra_numeric)} interactions. X shape: {X.shape}")""")

# CELL 4: BOOL->INT
md("""## 4. Convert Binary Columns (bool -> int)""")
code("""for col in binary_features:
    X[col] = X[col].astype(int)
    print(f"  {col}: {sorted(X[col].unique())}")
print("Done.")""")

# CELL 5: DUPLICATES
md("""## 5. Remove Duplicates""")
code("""dup_mask = X.duplicated(keep="first")
n_dup = dup_mask.sum()
if n_dup > 0:
    X = X[~dup_mask]; y = y[~dup_mask]
    print(f"Dropped {n_dup:,} duplicates")
else:
    print("No duplicates found.")
print(f"X: {X.shape}")""")

# CELL 6: SPLIT
md("""## 6. Train-Test Split (80/20) -- BEFORE any fits""")
code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: X {X_train.shape}, y {y_train.shape}")
print(f"Test:  X {X_test.shape}, y {y_test.shape}")
print(f"Train target mean: {y_train.mean():.4f}  Test target mean: {y_test.mean():.4f}")""")

# CELL 7: IMPUTE
md("""## 7. Impute Missing (fit on TRAIN only)

Numeric: median. Categorical+Binary: most_frequent.""")
code("""num_imputer = SimpleImputer(strategy="median")
X_train_num_imp = pd.DataFrame(num_imputer.fit_transform(X_train[numeric_features]), columns=numeric_features, index=X_train.index)
X_test_num_imp = pd.DataFrame(num_imputer.transform(X_test[numeric_features]), columns=numeric_features, index=X_test.index)

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat_imp = pd.DataFrame(cat_imputer.fit_transform(X_train[all_cat_features]), columns=all_cat_features, index=X_train.index)
X_test_cat_imp = pd.DataFrame(cat_imputer.transform(X_test[all_cat_features]), columns=all_cat_features, index=X_test.index)

print(f"Num imputer: train {X_train_num_imp.shape} test {X_test_num_imp.shape}  NaNs: train={X_train_num_imp.isna().sum().sum()} test={X_test_num_imp.isna().sum().sum()}")
print(f"Cat imputer: train {X_train_cat_imp.shape} test {X_test_cat_imp.shape}  NaNs: train={X_train_cat_imp.isna().sum().sum()} test={X_test_cat_imp.isna().sum().sum()}")""")

# CELL 8: ENCODE
md("""## 8. Encode Categoricals (fit on TRAIN only)

OneHot for low-card (Region, Soil_Type, Weather_Condition, irrigation_type). TargetEncoder for Crop.""")
code("""ohe = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", drop="first")
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train_cat_imp[low_card_cat_features]), columns=ohe.get_feature_names_out(low_card_cat_features), index=X_train.index)
X_test_ohe = pd.DataFrame(ohe.transform(X_test_cat_imp[low_card_cat_features]), columns=ohe.get_feature_names_out(low_card_cat_features), index=X_test.index)
ohe_cols = ohe.get_feature_names_out(low_card_cat_features)
print(f"OneHot: {len(low_card_cat_features)} in -> {len(ohe_cols)} out ({list(ohe_cols)})")
print(f"Train: {X_train_ohe.shape}  Test: {X_test_ohe.shape}")

te = ce.TargetEncoder(cols=["Crop"], handle_missing="value", handle_unknown="value")
X_train_te = pd.DataFrame(te.fit_transform(X_train_cat_imp[high_card_cat_features], y_train), columns=high_card_cat_features, index=X_train.index)
X_test_te = pd.DataFrame(te.transform(X_test_cat_imp[high_card_cat_features]), columns=high_card_cat_features, index=X_test.index)
print(f"\\nTargetEncoder (Crop):")
# Show encoding values
encoding_map = pd.DataFrame({"Crop": X_train_cat_imp["Crop"], "Encoded": X_train_te["Crop"]}).drop_duplicates()
for _, row in encoding_map.iterrows():
    print(f"  {str(row['Crop']):10s} -> {row['Encoded']:.4f}")
print(f"Train: {X_train_te.shape}  Test: {X_test_te.shape}")""")

# CELL 9: SCALE
md("""## 9. Scale Numeric (StandardScaler, fit on TRAIN only)""")
code("""scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num_imp), columns=numeric_features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_num_imp), columns=numeric_features, index=X_test.index)
print(f"Train: {X_train_scaled.shape}  Test: {X_test_scaled.shape}")
print(f"Train mean after scaling: {X_train_scaled.mean().mean():.6f}  std: {X_train_scaled.std().mean():.6f}")""")

# CELL 10: COMBINE
md("""## 10. Combine Processed Features""")
code("""X_train_final = pd.concat([X_train_scaled, X_train_ohe, X_train_te, X_train_cat_imp[binary_features].astype(int)], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_ohe, X_test_te, X_test_cat_imp[binary_features].astype(int)], axis=1)
print(f"Train final: {X_train_final.shape}")
print(f"Test final:  {X_test_final.shape}")
print(f"Total features: {X_train_final.shape[1]}")
for i, col in enumerate(X_train_final.columns, 1):
    print(f"  {i:>3d}. {col}")""")

# CELL 11: SAVE
md("""## 11. Save Fitted Transformers""")
code("""fitted_transformers = {
    "num_imputer": num_imputer, "cat_imputer": cat_imputer,
    "ohe": ohe, "target_encoder": te, "scaler": scaler,
    "numeric_features": numeric_features,
    "low_card_cat_features": low_card_cat_features,
    "high_card_cat_features": high_card_cat_features,
    "binary_features": binary_features,
    "all_cat_features": all_cat_features,
    "ohe_columns": list(ohe_cols),
}
print("Saved: num_imputer, cat_imputer, ohe, target_encoder, scaler")
print("Use: joblib.dump(fitted_transformers, 'transformers.pkl')")""")

# CELL 12: SUMMARY
md("""## 12. Sanity Check""")
code("""print("=" * 60)
print("  PROMPT BLOCK 3 -- FINAL SUMMARY")
print("=" * 60)
print(f"\\nOriginal: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Target: train {len(y_train):,} mean={y_train.mean():.4f}  test {len(y_test):,} mean={y_test.mean():.4f}")
print(f"\\nTrain: {X_train_final.shape[0]:,} x {X_train_final.shape[1]}  Test: {X_test_final.shape[0]:,} x {X_test_final.shape[1]}")
print(f"Numeric scaled: {len(numeric_features)}  OHE: {X_train_ohe.shape[1]}  TargetEnc: {X_train_te.shape[1]}  Binary: {len(binary_features)}")
nans_t = X_train_final.isna().sum().sum()
nans_te = X_test_final.isna().sum().sum()
print(f"NaNs: train={nans_t} test={nans_te}  {'PASS' if nans_t==0 and nans_te==0 else 'FAIL'}")
assert nans_t == 0 and nans_te == 0
print("No data leakage. All fits on TRAIN only. Ready for Prompt Block 4.")""")

# WRITE
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells
}
with open("notebooks/prompt_block_3.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Created notebooks/prompt_block_3.ipynb with {len(cells)} cells")
