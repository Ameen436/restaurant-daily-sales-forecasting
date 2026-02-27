"""Feature engineering for daily restaurant sales forecasting."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time and lag-based features without leakage."""
    feature_df = df.copy()

    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek
    feature_df["is_weekend"] = feature_df["day_of_week"].isin([5, 6]).astype(int)
    feature_df["month"] = feature_df["date"].dt.month

    feature_df["lag_1"] = feature_df["total_sales"].shift(1)
    feature_df["lag_7"] = feature_df["total_sales"].shift(7)

    # Shift by 1 day so rolling stats only use past information.
    shifted_sales = feature_df["total_sales"].shift(1)
    feature_df["rolling_mean_7"] = shifted_sales.rolling(window=7).mean()
    feature_df["rolling_mean_14"] = shifted_sales.rolling(window=14).mean()

    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df
