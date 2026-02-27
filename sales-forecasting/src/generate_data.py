"""Generate synthetic daily restaurant sales data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_PATH, ensure_directories


def generate_synthetic_data(days: int, start_date: str, seed: int) -> pd.DataFrame:
    """Create synthetic sales with trend, weekday effects, and noise."""
    if days < 365:
        raise ValueError("days must be at least 365.")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    t = np.arange(days)

    base_sales = 450.0
    trend = 0.18 * t

    weekday_boost = np.array([0.88, 0.90, 0.95, 1.00, 1.08, 1.22, 1.15])
    weekday_factor = weekday_boost[dates.dayofweek]

    yearly_seasonality = 25.0 * np.sin(2 * np.pi * t / 365.0)
    noise = rng.normal(loc=0.0, scale=20.0, size=days)

    sales = (base_sales + trend + yearly_seasonality) * weekday_factor + noise
    sales = np.clip(sales, 50, None)

    df = pd.DataFrame(
        {
            "date": dates,
            "total_sales": np.round(sales, 2),
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic restaurant daily sales data.")
    parser.add_argument("--output", type=str, default=str(DATA_PATH), help="Output CSV path.")
    parser.add_argument("--days", type=int, default=730, help="Number of daily records to generate.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    ensure_directories([output_path.parent])
    df = generate_synthetic_data(days=args.days, start_date=args.start_date, seed=args.seed)
    df.to_csv(output_path, index=False)

    print(f"Saved synthetic data to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Average daily sales: {df['total_sales'].mean():.2f}")


if __name__ == "__main__":
    main()
