"""Evaluate saved model and export predictions + visualization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS, create_features
from src.utils import (
    DATA_PATH,
    MODEL_PATH,
    PLOT_PATH,
    PREDICTIONS_PATH,
    compute_metrics,
    ensure_directories,
    load_sales_data,
    train_test_time_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained sales forecasting model.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Input CSV path.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Saved model path.")
    parser.add_argument(
        "--pred-out",
        type=str,
        default=str(PREDICTIONS_PATH),
        help="Path to save predictions CSV.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(PLOT_PATH),
        help="Path to save actual-vs-predicted plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_out = Path(args.pred_out)
    plot_out = Path(args.plot_out)

    ensure_directories([pred_out.parent, plot_out.parent])

    raw_df = load_sales_data(args.data_path)
    df = create_features(raw_df)
    _, test_df = train_test_time_split(df, train_ratio=0.8)

    model = joblib.load(args.model_path)

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["total_sales"]

    predicted = model.predict(x_test)
    baseline = test_df["lag_1"].to_numpy()

    model_metrics = compute_metrics(y_test, predicted)
    baseline_metrics = compute_metrics(y_test, baseline)

    output_df = pd.DataFrame(
        {
            "date": test_df["date"].dt.strftime("%Y-%m-%d"),
            "actual": y_test.values,
            "predicted": predicted,
            "baseline": baseline,
        }
    )
    output_df.to_csv(pred_out, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(test_df["date"], y_test, label="Actual", linewidth=2)
    plt.plot(test_df["date"], predicted, label="Predicted (Best Model)", linewidth=2)
    plt.plot(test_df["date"], baseline, label="Baseline (Lag 1)", linestyle="--", alpha=0.8)
    plt.title("Daily Restaurant Sales: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()

    print("Evaluation complete:")
    print(f"Model      -> MAE: {model_metrics['mae']:.3f}, RMSE: {model_metrics['rmse']:.3f}")
    print(f"Baseline   -> MAE: {baseline_metrics['mae']:.3f}, RMSE: {baseline_metrics['rmse']:.3f}")
    print(f"Predictions saved to: {pred_out}")
    print(f"Plot saved to: {plot_out}")


if __name__ == "__main__":
    main()
