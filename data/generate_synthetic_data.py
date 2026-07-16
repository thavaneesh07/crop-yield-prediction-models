#!/usr/bin/env python3
"""
SYNTHETIC DATA GENERATION — NOT GROUND TRUTH
=============================================
This script generates four synthetic CSV files for the Explainable AI Crop Yield
Prediction project. All data is procedurally generated with realistic-ish ranges
but does NOT represent real-world agricultural measurements. It is intended for
pipeline development and demonstration only.

Runs: python data/generate_synthetic_data.py
Outputs (in data/):
  - crop_yield.csv             (~1 000 000 rows, backbone)
  - soil_weather.csv           (~50 000 rows)
  - smart_farming.csv          (~50 000 rows)
  - crop_recommendation.csv    (~2 200 rows, one per crop-region combo)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

# ── Domain constants ──────────────────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "Central", "Northeast", "Southeast", "Northwest", "Southwest"]
SOIL_TYPES = ["Clay", "Sandy", "Loamy", "Silt", "Peaty", "Chalky", "Saline"]
WEATHER_CONDITIONS = ["Sunny", "Rainy", "Cloudy", "Dry", "Humid", "Windy"]
IRRIGATION_TYPES = ["Drip", "Sprinkler", "Flood", "Furrow", "Pivot", "None"]

# 28 crops with known synonyms so we can test standardisation
CROPS = [
    "Wheat", "Rice", "Maize", "Corn", "Barley", "Soybean", "Cotton",
    "Sugarcane", "Tomato", "Potato", "Onion", "Carrot", "Cabbage",
    "Cauliflower", "Brinjal", "Chili", "Garlic", "Ginger", "Turmeric",
    "Coconut", "Groundnut", "Sunflower", "Mustard", "Millet", "Ragi",
    "Jowar", "Bajra", "Coffee",
]

# ── Helper ────────────────────────────────────────────────────────────────────
def _crop_to_base(name: str) -> str:
    """Map known synonyms so maize/corn etc. get the same base form."""
    mapping = {
        "maize": "corn",
        "sweet corn": "corn",
        "baby corn": "corn",
    }
    return mapping.get(name.strip().lower(), name.strip().lower())


def _sample_crops(n: int) -> pd.Series:
    """Return a Series of *n* crop names (possibly with synonyms to test mapping)."""
    base_crops = RNG.choice(CROPS, size=n)

    # Randomly replace 5 % of entries with a synonym to exercise standardisation
    synonym_map = {"Maize": "Corn", "Corn": "Maize"}
    mask = RNG.random(n) < 0.05
    for i in range(n):
        if mask[i]:
            base_crops[i] = synonym_map.get(base_crops[i], base_crops[i])
    return pd.Series(base_crops)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  crop_yield.csv  (backbone, ~1M rows)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating crop_yield.csv  (~1 000 000 rows) ...")
N_BACKBONE = 1_000_000

cy = pd.DataFrame({
    "Region":               RNG.choice(REGIONS, size=N_BACKBONE),
    "Soil_Type":            RNG.choice(SOIL_TYPES, size=N_BACKBONE),
    "Crop":                 _sample_crops(N_BACKBONE),
    "Rainfall_mm":          RNG.uniform(200, 2500, size=N_BACKBONE).round(1),
    "Temperature_Celsius":  RNG.uniform(10, 40, size=N_BACKBONE).round(1),
    "Fertilizer_Used":      RNG.choice(["Yes", "No"], size=N_BACKBONE, p=[0.75, 0.25]),
    "Irrigation_Used":      RNG.choice(["Yes", "No"], size=N_BACKBONE, p=[0.6, 0.4]),
    "Weather_Condition":    RNG.choice(WEATHER_CONDITIONS, size=N_BACKBONE),
    "Days_to_Harvest":      RNG.integers(30, 365, size=N_BACKBONE),
})

# Yield: synthetic target with some signal from rainfall + temp + fertiliser
base_yield = (
    1.5
    + 0.003 * cy["Rainfall_mm"]
    + 0.08 * cy["Temperature_Celsius"]
    + 1.2 * (cy["Fertilizer_Used"] == "Yes").astype(int)
    + 0.6 * (cy["Irrigation_Used"] == "Yes").astype(int)
)
cy["Yield_tons_per_hectare"] = (base_yield + RNG.normal(0, 0.8, size=N_BACKBONE)).clip(0.5, 15).round(2)

cy.to_csv("data/crop_yield.csv", index=False)
print(f"  -> {len(cy):,} rows x {cy.shape[1]} columns  |  Memory: {cy.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  soil_weather.csv  (~50k rows)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating soil_weather.csv (~50 000 rows) ...")
N_SW = 50_000

sw = pd.DataFrame({
    "Region":    RNG.choice(REGIONS, size=N_SW),
    "Crop":      _sample_crops(N_SW),
    "Soil_Type": RNG.choice(SOIL_TYPES, size=N_SW),
    "soil_pH":   RNG.uniform(4.5, 8.5, size=N_SW).round(2),
    "Organic_Matter_pct": RNG.uniform(0.5, 5.0, size=N_SW).round(2),
})

sw.to_csv("data/soil_weather.csv", index=False)
print(f"  -> {len(sw):,} rows x {sw.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  smart_farming.csv  (~50k rows)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating smart_farming.csv (~50 000 rows) ...")
N_SF = 50_000

sf = pd.DataFrame({
    "Region":           RNG.choice(REGIONS, size=N_SF),
    "Crop":             _sample_crops(N_SF),
    "Soil_Type":        RNG.choice(SOIL_TYPES, size=N_SF),
    "soil_pH":          RNG.uniform(4.5, 8.5, size=N_SF).round(2),
    "Humidity_pct":     RNG.uniform(30, 95, size=N_SF).round(1),
    "Sunlight_Hours":   RNG.uniform(4, 14, size=N_SF).round(1),
    "Irrigation_Type":  RNG.choice(IRRIGATION_TYPES, size=N_SF),
})

sf.to_csv("data/smart_farming.csv", index=False)
print(f"  -> {len(sf):,} rows x {sf.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  crop_recommendation.csv  (~2 200 rows)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating crop_recommendation.csv (~2 200 rows) ...")
N_CR = len(CROPS) * len(REGIONS)  # one row per crop × region

crops_cr = [c.lower() for c in CROPS for _ in REGIONS]
regions_cr = REGIONS * len(CROPS)

cr = pd.DataFrame({
    "Crop":         crops_cr,
    "Region":       regions_cr,
    "N":            RNG.integers(0, 140, size=N_CR),
    "P":            RNG.integers(5, 145, size=N_CR),
    "K":            RNG.integers(5, 205, size=N_CR),
    "temperature":  RNG.uniform(10, 40, size=N_CR).round(1),
    "humidity":     RNG.uniform(30, 95, size=N_CR).round(1),
    "ph":           RNG.uniform(4.5, 8.5, size=N_CR).round(2),
    "rainfall":     RNG.uniform(200, 2500, size=N_CR).round(1),
})

cr.to_csv("data/crop_recommendation.csv", index=False)
print(f"  -> {len(cr):,} rows x {cr.shape[1]} columns")

print("\nOK - All four synthetic CSV files written to data/")
print("  NOTE: These are procedurally generated for pipeline development.")
print("  They do NOT represent real agricultural measurements.")
