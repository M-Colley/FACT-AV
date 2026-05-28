#!/usr/bin/env python3
"""
Extracted ML Analysis Script for testing TabPFN.
"""

import logging
import warnings
import os
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Hardware Detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("PyTorch not found; GPU detection disabled. TabPFN will default to CPU.")

# =====================================================================
# FIX FOR WINDOWS CRASH: Safely prompt for API key before TabPFN loads
# =====================================================================
if "TABPFN_TOKEN" not in os.environ:
    print("\n--- TabPFN Authentication ---")
    print("TabPFN requires an API key to download weights, but its built-in prompt crashes on Windows.")
    print("1. Open https://ux.priorlabs.ai/account (log in or register)")
    print("2. Accept the license at https://ux.priorlabs.ai/account/licenses")
    print("3. Copy your API Key")
    token = input("Paste your API key here and press Enter: ").strip()
    if token:
        os.environ["TABPFN_TOKEN"] = token
        print("API key loaded!\n")
    else:
        print("No token provided. TabPFN may fail if it hasn't cached the weights yet.")
# =====================================================================

# TabPFN Imports
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import save_fitted_tabpfn_model
except ImportError:
    raise ImportError("TabPFN is not installed. Please install it to test this script.")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the ML analysis."""
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
    results_path: Path = Path("results") / "ML-Approaches"
    sheet_name: str = "Sheet1"
    test_size: float = 0.2
    random_state: int = 42

    numerical_features: List[str] = None
    categorical_features: List[str] = None
    target_column: str = "trust"
    group_column: str = "ProlificID"

    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = ["mIoU", "License", "Age"]
        if self.categorical_features is None:
            self.categorical_features = [
                "SCENARIO",
                "INTRODUCTION",
                "Gender",
                "Education",
                "Job",
                "DrivingFrequency",
                "Distance",
            ]
        self.results_path.mkdir(parents=True, exist_ok=True)


class DataProcessor:
    """Handles data loading, validation, and pipeline preprocessing construction."""

    def __init__(self, config: Config):
        self.config = config

    def get_label_mappings(self) -> Dict[str, Dict[str, str]]:
        return {
            "Gender": {"A1": "F", "A2": "M", "A3": "non-binary", "A4": "Prefer not to tell"},
            "Education": {
                "A1": "Secondary School",
                "A2": "Middle School",
                "A3": "High School",
                "A4": "College",
                "A5": "Vocational training",
            },
            "Job": {
                "A1": "Student (school)",
                "A2": "Student (college)",
                "A3": "Employee",
                "A4": "Self-employed",
                "A5": "Jobseeker",
                "A6": "Other",
            },
            "DrivingFrequency": {
                "A1": "Daily",
                "A2": "On working days",
                "A3": "3-4 times a week",
                "A4": "1 time a week",
                "A5": "1-3 times a month",
                "A6": "less than 1 time a month",
            },
            "Distance": {
                "A1": "less than 7.000km",
                "A2": "7.000 - 14.999km",
                "A3": "15.000 - 24.999km",
                "A4": "25.000 - 32.999km",
                "A5": "33.000 or more km",
            },
        }

    def load_and_preprocess_data(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        df = pd.read_excel(self.config.data_path, sheet_name=self.config.sheet_name)

        required_columns = (
            self.config.numerical_features
            + self.config.categorical_features
            + [self.config.target_column]
        )
        missing_columns = sorted(set(required_columns) - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.config.data_path}: {missing_columns}"
            )

        before_rows = len(df)
        df = df.dropna(subset=required_columns).copy()
        dropped_rows = before_rows - len(df)
        if dropped_rows > 0:
            logger.info("Dropped %d rows containing missing required fields.", dropped_rows)

        if self.config.group_column in df.columns:
            df = df.dropna(subset=[self.config.group_column])
        else:
            logger.warning(
                "Group column %r not found; falling back to a random (non-grouped) split.",
                self.config.group_column,
            )

        for column, mapping in self.get_label_mappings().items():
            if column in df.columns:
                df[column] = df[column].replace(mapping)

        df[self.config.target_column] = pd.to_numeric(df[self.config.target_column], errors="raise")

        return df

    def get_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            sparse_output=False,
            drop="first",
            handle_unknown="ignore",
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.numerical_features),
                ("cat", categorical_transformer, self.config.categorical_features),
            ]
        )

def get_tabpfn_quantile_columns(q_preds: Any, quantiles: List[float]) -> List[tuple[float, np.ndarray]]:
    """Return (quantile, predictions_vector) pairs independent of output orientation."""
    values = np.asarray(q_preds)

    if values.ndim == 1:
        if len(quantiles) != 1:
            raise ValueError("TabPFN returned 1D quantile predictions for multiple quantiles.")
        return [(quantiles[0], values)]

    if values.ndim != 2:
        raise ValueError(f"Unexpected TabPFN quantile output shape: {values.shape}")

    if values.shape[1] == len(quantiles):
        return [(q, values[:, i]) for i, q in enumerate(quantiles)]

    if values.shape[0] == len(quantiles):
        return [(q, values[i, :]) for i, q in enumerate(quantiles)]

    raise ValueError(
        f"Cannot align TabPFN quantiles. output_shape={values.shape}, requested_quantiles={len(quantiles)}"
    )

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def main() -> None:
    if GPU_AVAILABLE:
        logger.info("Nvidia GPU detected. Supported models will try hardware acceleration.")
    else:
        logger.info("No compatible GPU detected. Running on CPU.")

    config = Config()
    np.random.seed(config.random_state)

    # 1. Load Data
    data_processor = DataProcessor(config)
    df = data_processor.load_and_preprocess_data()

    X = df[config.numerical_features + config.categorical_features]
    y = df[config.target_column]

    # 2. Split Data (Grouped by Participant)
    if config.group_column in df.columns:
        groups = df[config.group_column]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        (train_idx, test_idx), = splitter.split(X, y, groups=groups)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
        )

    # 3. Preprocess Data
    preprocessor = data_processor.get_preprocessor()
    X_train_tab = preprocessor.fit_transform(X_train)
    X_test_tab = preprocessor.transform(X_test)

    # 4. Train TabPFN
    logger.info("Training TabPFN on %d rows...", len(X_train_tab))
    tabpfn_params = {
        "ignore_pretraining_limits": True,
        "device": "cuda" if GPU_AVAILABLE else "cpu",
    }
    
    reg = TabPFNRegressor(**tabpfn_params)
    try:
        reg.fit(X_train_tab, y_train.values)
    except Exception as exc:
        if tabpfn_params["device"] == "cuda":
            logger.warning("TabPFN GPU fit failed (%s). Falling back to CPU...", exc)
            reg = TabPFNRegressor(ignore_pretraining_limits=True, device="cpu")
            reg.fit(X_train_tab, y_train.values)
        else:
            raise

    # 5. Save model
    save_path = config.results_path / "fact-av.tabpfn_fit"
    save_fitted_tabpfn_model(reg, save_path)
    logger.info(f"TabPFN model saved to {save_path}")

    # 6. Evaluate Predictions
    preds = reg.predict(X_test_tab)
    metrics_tab = calculate_metrics(y_test, preds)
    
    results_txt = [
        "--- TabPFN Results ---",
        f"Mean Squared Error (MSE): {metrics_tab['mse']:.4f}",
        f"Mean Absolute Error (MAE): {metrics_tab['mae']:.4f}",
        f"Root Mean Squared Error (RMSE): {metrics_tab['rmse']:.4f}",
        f"R-squared (R^2): {metrics_tab['r2']:.4f}",
        ""
    ]

    # 7. Evaluate Quantiles
    quantiles = [0.25, 0.5, 0.75]
    logger.info("Computing quantile predictions...")
    q_preds = reg.predict(X_test_tab, output_type="quantiles", quantiles=quantiles)
    for quantile, quantile_preds in get_tabpfn_quantile_columns(q_preds, quantiles):
        q_mae = mean_absolute_error(y_test, quantile_preds)
        results_txt.append(f"Quantile {quantile} MAE: {q_mae:.4f}")

    # 8. Evaluate Mode
    logger.info("Computing mode predictions...")
    mode_preds = reg.predict(X_test_tab, output_type="mode")
    mode_mae = mean_absolute_error(y_test, mode_preds)
    results_txt.append(f"Mode MAE: {mode_mae:.4f}")

    # 9. Save Results
    results_file = config.results_path / "results_tabpfnregressor.txt"
    with results_file.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(results_txt))
        
    logger.info(f"Results successfully saved to {results_file}")
    print("\n" + "\n".join(results_txt))

if __name__ == "__main__":
    main()