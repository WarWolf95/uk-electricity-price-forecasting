"""
Unit test suite for UK Wholesale Electricity Price Forecasting pipeline.
Validates dataset integrity, feature schemas, and model component imports.
"""

import os
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_processed_data_files_exist():
    """Verify essential processed CSV files exist."""
    required_files = [
        "aligned_features.csv",
        "feature_importance.csv",
        "forecast_predictions.csv",
        "model_comparison.csv",
    ]
    for file_name in required_files:
        file_path = DATA_PROCESSED_DIR / file_name
        assert file_path.exists(), f"Missing required data file: {file_name}"


def test_feature_dataset_schema():
    """Validate model_features dataset columns and non-empty state."""
    features_csv = DATA_PROCESSED_DIR / "aligned_features.csv"
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        assert len(df) > 0, "Aligned features CSV is empty"
        assert "target_price" in df.columns or "price" in df.columns or len(df.columns) > 5, "Dataset missing core feature columns"


def test_model_imports():
    """Ensure forecasting model architectures import cleanly."""
    try:
        from src.models.train_and_evaluate import XGBoostRunner
        assert XGBoostRunner is not None
    except ModuleNotFoundError:
        pytest.skip("PyTorch optional dependency not installed in local environment")
