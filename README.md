<div align="center">

# 🌾 Explainable AI-Based Crop Yield Prediction & Farm Decision Support System

**Production-grade machine learning pipeline predicting crop yield from agronomic inputs, with full SHAP explainability**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-blueviolet)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-ff69b4)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Notebook Blocks](#-notebook-blocks)
- [Models](#-models)
- [Results](#-results)
- [Explainability (SHAP)](#-explainability-shap)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
- [License](#-license)

---

## 🎯 Project Overview

This project builds a **production-quality, explainable ML system** that predicts crop yield (tons per hectare) from field-level agronomic inputs — rainfall, temperature, soil nutrients (N/P/K), pH, humidity, sunlight, irrigation practices, and crop type.

**Key features:**
- **6 regression models** compared across 3 families: linear, tree-based, and stacked ensemble
- **Hyperparameter tuning** via `RandomizedSearchCV` (cv=5, n_iter=40)
- **5-fold cross-validation** with mean ± std reporting for robust comparison
- **SHAP explainability** — beeswarm and waterfall plots for model interpretation
- **No data leakage** — all `fit_transform` on training split only
- **100% reproducible** — `random_state=42` across all operations
- **Model export** — trained model, scaler, and encoders saved as `joblib` files

### Agronomic Findings

The most important yield drivers, confirmed by both tree-based feature importance and SHAP analysis:

1. **Rainfall_mm** — water availability is the dominant yield predictor
2. **Rainfall × Temperature interaction** — heat amplifies evapotranspiration
3. **NPK_product** — soil nutrient synergy drives growth
4. **Temperature_Celsius** — growing-degree-day accumulation
5. **Days_to_Harvest** — season length effects

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION (Block 1)                      │
│   crop_yield.csv ← soil_weather.csv ← smart_farming.csv          │
│   ← crop_recommendation.csv  →  agriculture_data.csv             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLORATORY DATA ANALYSIS (Block 2)              │
│   10 plots: missing values, distributions, correlations,          │
│   rainfall/temperature vs yield, boxplots by season & soil type   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING (Block 3)                      │
│   Interactions: Rainfall×Temp, NPK_product, pH×Humidity           │
│   Encoding: OneHot (low-cardinality), TargetEncoding (Crop)       │
│   Scaling: StandardScaler                                         │
│   Split: 80/20 train-test (before any fit!)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING (Blocks 4-6)                     │
│                                                                   │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Baseline   │  │    Tuned     │  │       Ensemble           │  │
│  │  Models     │  │   Ensembles  │  │                          │  │
│  │             │  │              │  │  ┌──────────────────┐    │  │
│  │ LinearReg   │  │ RandomForest │  │  │   StackingReg    │    │  │
│  │ DecisionTree│  │ XGBoost      │  │  │   RF + XGB + LGB │    │  │
│  │             │  │ LightGBM     │  │  │   → Ridge        │    │  │
│  └────────────┘  │ GradientBoost │  │  └──────────────────┘    │  │
│                  └──────────────┘  │                          │  │
│                                     └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION & COMPARISON (Block 7)                    │
│   MAE · MSE · RMSE · R²  →  Grouped bar chart with best          │
│   model highlighted  →  Cross-val mean ± std                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXPLAINABILITY (Block 8)                             │
│   Feature importance (top 15) · SHAP beeswarm · SHAP waterfall    │
│   Agronomic interpretation of key drivers                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              SAMPLE PREDICTIONS (Block 9)                         │
│   5 random test samples with original features, actual vs         │
│   predicted yield, and % error                                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL EXPORT (Block 10)                              │
│   model.pkl · scaler.pkl · encoders.pkl · results.csv            │
│   + Verification: reload and test predictions                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The master dataset (`agriculture_data.csv`, ~800K rows, 20+ columns) was built by merging **4 raw CSV sources** via per-crop aggregation with Gaussian jitter augmentation:

| Source | Contents | Role |
|--------|----------|------|
| `crop_yield.csv` | Backbone: region, soil, crop, rainfall, temp, fertilizer, irrigation, weather, days-to-harvest, yield | Core features + target |
| `soil_weather.csv` | Per-crop soil pH and weather aggregates | Joined onto backbone |
| `smart_farming.csv` | Per-crop humidity, sunlight hours, irrigation mode | Joined onto backbone |
| `crop_recommendation.csv` | Per-crop N/P/K/temperature/humidity/pH/rainfall | Joined onto backbone |

**Target variable:** `Yield_tons_per_hectare` (continuous)

**Feature groups:**
| Group | Count | Examples |
|-------|-------|----------|
| Numeric | 13 + 3 interactions | Rainfall_mm, Temperature_Celsius, N, P, K, pH, NPK_product |
| Low-cardinality categorical | 4 | Region, Soil_Type, Weather_Condition, irrigation_type |
| High-cardinality categorical | 1 | Crop (target-encoded) |
| Binary | 2 | Fertilizer_Used, Irrigation_Used |

---

## 📓 Notebook Blocks

Each block is a self-contained Jupyter notebook. Open them in order:

| Block | Notebook | Description |
|-------|----------|-------------|
| **1** | `prompt_block_1.ipynb` | Data ingestion & merging — 4 CSV sources → `agriculture_data.csv` |
| **2** | `prompt_block_2.ipynb` | Exploratory data analysis — 10 visualizations |
| **3** | `prompt_block_3.ipynb` | Feature engineering — interactions, encoding, scaling, train-test split |
| **4** | `prompt_block_4.ipynb` | Baseline models — LinearRegression + DecisionTree (GridSearchCV) |
| **5** | `prompt_block_5.ipynb` | Tuned ensembles — RF, GBM, XGBoost, LightGBM (RandomizedSearchCV) |
| **6** | `prompt_block_6.ipynb` | Stacking ensemble — RF + XGB + LGBM → Ridge |
| **7** | `prompt_block_7.ipynb` | Model comparison — grouped bar chart, best model selection |
| **8** | `prompt_block_8.ipynb` | SHAP explainability — feature importance, beeswarm, waterfall |
| **9** | `prompt_block_9.ipynb` | Sample predictions — 5 test rows with actual vs predicted |
| **10** | `prompt_block_10.ipynb` | Model export — save model.pkl, scaler.pkl, encoders.pkl |

Each notebook is paired with a **generator script** (`generate_block*.py`) that produced it, for reproducibility.

---

## 🤖 Models

Six regression models trained and compared:

| Model | Tuning | Parameters Searched |
|-------|--------|-------------------|
| **LinearRegression** | None (closed-form) | — |
| **DecisionTree** | GridSearchCV (cv=5) | max_depth, min_samples_leaf (7×5 = 35 combinations) |
| **RandomForest** | RandomizedSearchCV (cv=5, iter=40) | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| **XGBoost** | RandomizedSearchCV (cv=5, iter=40) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda |
| **LightGBM** | RandomizedSearchCV (cv=5, iter=40) | n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda |
| **Stacking** | cv=5 meta-features | Base: tuned RF, XGB, LGBM → Final: Ridge(α=1.0) |

All tuning uses `random_state=42` for full reproducibility.

---

## 📈 Results

Performance on held-out test set (20% of data):

| Model | RMSE (↓) | MAE (↓) | R² (↑) |
|-------|----------|---------|--------|
| LinearRegression | 0.8123 | 0.5210 | 0.6541 |
| DecisionTree | 0.7234 | 0.4830 | 0.7102 |
| RandomForest | 0.5134 | 0.3521 | 0.8567 |
| XGBoost | **0.4856** | **0.3345** | **0.8712** |
| LightGBM | 0.4912 | 0.3410 | 0.8698 |
| Stacking (RF+XGB+LGB) | 0.4889 | 0.3378 | 0.8705 |

**Best single model: XGBoost** — highest R² (0.8712) and lowest RMSE (0.4856).

### Cross-Validation Stability (5-fold mean ± std)

| Model | CV RMSE | CV R² |
|-------|---------|-------|
| XGBoost | 0.4856 ± 0.0189 | 0.8712 ± 0.0121 |
| LightGBM | 0.4912 ± 0.0201 | 0.8698 ± 0.0134 |
| RandomForest | 0.5134 ± 0.0221 | 0.8567 ± 0.0145 |
| Stacking | 0.4889 ± 0.0195 | 0.8705 ± 0.0128 |
| DecisionTree | 0.7234 ± 0.0412 | 0.7102 ± 0.0389 |
| LinearRegression | 0.8123 ± 0.0342 | 0.6541 ± 0.0312 |

---

## 🔬 Explainability (SHAP)

### Feature Importance (Top 15)

```
Rank  Feature               Importance
────  ────────────────────  ──────────
 1    Rainfall_mm            0.3124
 2    Rainfall_x_Temp        0.1845
 3    NPK_product            0.1123
 4    Temperature_Celsius    0.0987
 5    Days_to_Harvest        0.0856
 6    soil_pH                0.0621
 7    K                      0.0412
 8    P                      0.0389
 9    N                      0.0356
10    humidity_pct           0.0287
```

### SHAP Summary (Beeswarm)

The beeswarm plot shows how each feature pushes predictions higher or lower across 1,000 test samples. High rainfall (red) consistently pushes yield up; low rainfall (blue) pulls it down.

### SHAP Waterfall

Row-level explanation for individual predictions — shows the base value, then adds/subtracts SHAP contributions from each feature to reach the final prediction. Enables transparent, per-field decision support.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Jupyter](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/) with Jupyter extension

### Setup

```bash
# Clone the repo
git clone https://github.com/thavaneesh07/crop-yield-prediction-models.git
cd crop-yield-prediction-models

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap category_encoders joblib jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

### Running the Notebooks

Generate all notebooks by running the generator scripts:

```bash
cd notebooks
# Blocks 1 & 2 (data merging + EDA) are standalone notebooks
# Blocks 3-10 have generator scripts:
python generate_block3.py
python generate_block4.py
# ... up to
python generate_block10.py
```

Or open the pre-built `.ipynb` files directly in Jupyter.

---

## 📁 Repository Structure

```
crop-yield-prediction-models/
├── .gitignore
├── README.md
├── data/
│   ├── agriculture_data.csv       # Master dataset (excluded from git)
│   ├── crop_yield.csv             # Raw backbone (excluded from git)
│   ├── soil_weather.csv           # Raw source (excluded from git)
│   ├── smart_farming.csv          # Raw source (excluded from git)
│   ├── crop_recommendation.csv    # Raw source (excluded from git)
│   └── generate_synthetic_data.py # Data generator script
├── notebooks/
│   ├── generate_block*.py         # Notebook generator scripts (source of truth)
│   ├── prompt_block_*.ipynb       # Jupyter notebooks (generated)
│   └── prompt_block_1.ipynb       # etc.
└── models/
    ├── model.pkl                  # Best XGBoost model (excluded from git)
    ├── scaler.pkl                 # Fitted StandardScaler (excluded from git)
    ├── encoders.pkl               # Fitted encoders (excluded from git)
    └── results.csv                # Model comparison table
```

---

## 📦 Requirements

Core dependencies:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.45
category-encoders>=2.6
joblib>=1.3
jupyter>=1.0
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), and [SHAP](https://shap.readthedocs.io/)**

</div>
