"""Utility helpers for the sales forecasting project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/daily_sales.csv")
MODEL_PATH = Path("models/model.joblib")
PREDICTIONS_PATH = Path("outputs/predictions.csv")
PLOT_PATH = Path("outputs/actual_vs_pred.png")
METRICS_PATH = Path("outputs/metrics.json")


def ensure_directories(paths: list[Path | str]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_sales_data(path: Path | str) -> pd.DataFrame:
    """Load sales data with schema checks and sorted dates."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"date", "total_sales"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found. Each date must appear only once.")

    if (df["total_sales"] < 0).any():
        raise ValueError("Negative total_sales values found. Sales must be non-negative.")

    return df


def train_test_time_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets by time order (no shuffling)."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition.")

    return train_df, test_df


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE for predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}


def save_json(path: Path | str, obj: dict[str, Any]) -> None:
    """Persist dictionary to JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
