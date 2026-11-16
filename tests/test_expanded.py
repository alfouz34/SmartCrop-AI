"""
Test suite for SmartCrop AI Expanded Agricultural Dataset and Experiment Pipeline.
Verifies data integrity, schema compliance, backwards compatibility, and safety guards.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPANDED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_crop_dataset.csv"
RAW_BENCHMARK_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

EXPECTED_COLUMNS = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "district", "season", "soil_type", "irrigation", "crop", "yield", "data_source"
]

NEW_KERALA_CROPS = [
    "rubber", "tapioca", "arecanut", "cashew", "blackpepper", "cardamom",
    "ginger", "turmeric", "tea", "clove", "nutmeg", "sweetpotato",
    "elephantfootyam", "tomato", "chilli", "brinjal", "okra"
]

def test_original_files_preserved():
    """Ensure existing production artifacts are untouched and intact."""
    assert RAW_BENCHMARK_PATH.exists(), "Original Crop_recommendation.csv must be preserved!"
    assert ORIGINAL_MODEL_PATH.exists(), "Original best_model.joblib must be preserved!"
    
    df_raw = pd.read_csv(RAW_BENCHMARK_PATH)
    assert len(df_raw) == 2200, "Original raw CSV must maintain exactly 2,200 rows"
    assert df_raw["label"].nunique() == 22, "Original raw CSV must maintain exactly 22 crops"

def test_expanded_dataset_exists_and_schema():
    """Verify expanded dataset exists and conforms to the 14-column unified schema."""
    assert EXPANDED_DATA_PATH.exists(), f"Expanded dataset missing at {EXPANDED_DATA_PATH}"
    df = pd.read_csv(EXPANDED_DATA_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, f"Schema mismatch! Got: {list(df.columns)}"

def test_expanded_dataset_row_counts_and_classes():
    """Verify exactly 3,900 rows and 39 balanced classes (100 per crop)."""
    df = pd.read_csv(EXPANDED_DATA_PATH)
    assert len(df) == 3900, f"Expected 3,900 rows, got {len(df)}"
    assert df["crop"].nunique() == 39, f"Expected 39 unique crops, got {df['crop'].nunique()}"
    
    counts = df["crop"].value_counts()
    assert (counts == 100).all(), f"All classes must have exactly 100 rows. Found:\n{counts[counts != 100]}"

def test_expanded_dataset_new_kerala_crops_present():
    """Verify all 17 new Kerala agricultural crops are present."""
    df = pd.read_csv(EXPANDED_DATA_PATH)
    classes = set(df["crop"].unique())
    for crop in NEW_KERALA_CROPS:
        assert crop in classes, f"Expected Kerala crop '{crop}' not found in expanded dataset!"

def test_expanded_core_feature_bounds_and_nulls():
    """Verify 0 nulls in core features and realistic physiological ranges."""
    df = pd.read_csv(EXPANDED_DATA_PATH)
    core_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop", "data_source"]
    assert df[core_cols].isnull().sum().sum() == 0, "Core features must have 0 null values"
    
    # Range validations
    assert (df["N"] >= 0).all() and (df["N"] <= 300).all(), "N out of realistic agronomic bounds"
    assert (df["P"] >= 0).all() and (df["P"] <= 300).all(), "P out of realistic agronomic bounds"
    assert (df["K"] >= 0).all() and (df["K"] <= 400).all(), "K out of realistic agronomic bounds"
    assert (df["temperature"] >= 5).all() and (df["temperature"] <= 50).all(), "Temperature out of realistic bounds"
    assert (df["humidity"] >= 0).all() and (df["humidity"] <= 100).all(), "Humidity out of bounds"
    assert (df["ph"] >= 3.5).all() and (df["ph"] <= 10.0).all(), "Soil pH out of realistic bounds"
    assert (df["rainfall"] >= 0).all() and (df["rainfall"] <= 5000).all(), "Rainfall out of bounds"

def test_provenance_and_yield_documentation():
    """Verify explicit data_source labeling and documented missing values in yield."""
    df = pd.read_csv(EXPANDED_DATA_PATH)
    
    # Benchmark subset
    bench_mask = df["data_source"] == "Kaggle Benchmark (Semi-Synthetic)"
    assert bench_mask.sum() == 2200, "Expected exactly 2,200 benchmark rows"
    assert df.loc[bench_mask, "yield"].isna().all(), "Benchmark rows must have NaN yield"
    assert (df.loc[bench_mask, "district"] == "All-India Benchmark").all()
    
    # Kerala subset
    kerala_mask = ~bench_mask
    assert kerala_mask.sum() == 1700, "Expected exactly 1,700 Kerala regional rows"
    assert df.loc[kerala_mask, "yield"].notna().all(), "Kerala regional rows must have recorded yields"
    assert (df.loc[kerala_mask, "yield"] > 0).all(), "Recorded yields must be positive"
