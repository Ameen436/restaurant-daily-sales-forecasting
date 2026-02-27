"""Train baseline and ML models for daily sales forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.features import FEATURE_COLUMNS, create_features
from src.utils import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    compute_metrics,
    ensure_directories,
    load_sales_data,
    save_json,
    train_test_time_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sales forecasting models.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Input CSV path.")
    parser.add_argument("--model-out", type=str, default=str(MODEL_PATH), help="Path to save best model.")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=str(METRICS_PATH),
        help="Path to save metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)

    ensure_directories([model_out.parent, metrics_out.parent])

    raw_df = load_sales_data(args.data_path)
    df = create_features(raw_df)

    train_df, test_df = train_test_time_split(df, train_ratio=0.8)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["total_sales"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["total_sales"]

    baseline_pred = test_df["lag_1"]

    models = {
        "linear_regression": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    metrics: dict[str, dict[str, float]] = {}

    metrics["baseline"] = compute_metrics(y_test, baseline_pred)

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics[name] = compute_metrics(y_test, pred)

    ranked_models = sorted(
        [(name, values) for name, values in metrics.items() if name != "baseline"],
        key=lambda item: (item[1]["rmse"], item[1]["mae"]),
    )
    best_model_name = ranked_models[0][0]
    best_model = models[best_model_name]

    joblib.dump(best_model, model_out)

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_columns": FEATURE_COLUMNS,
        "best_model": best_model_name,
        "metrics": metrics,
    }
    save_json(metrics_out, summary)

    print("Model comparison (lower is better):")
    print(f"{'model':<20} {'MAE':>12} {'RMSE':>12}")
    for name in ["baseline", "linear_regression", "gradient_boosting"]:
        print(f"{name:<20} {metrics[name]['mae']:>12.3f} {metrics[name]['rmse']:>12.3f}")

    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to: {model_out}")
    print(f"Saved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()
