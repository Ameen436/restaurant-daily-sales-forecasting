# Restaurant Daily Sales Forecasting

Local machine learning project for predicting daily restaurant sales from historical data.

## Quick start (2 minutes)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_data
python -m src.train
python -m src.evaluate
```

Then confirm these files exist:
- `data/daily_sales.csv`
- `models/model.joblib`
- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/actual_vs_pred.png`

## What this project does

Given a CSV with:
- `date`
- `total_sales`

it builds features from date patterns and historical sales, trains multiple models, compares them to a baseline, selects the best model, and saves predictions + a plot.

If no real data is available, it can generate realistic synthetic daily sales data.

## Project structure

```text
sales-forecasting/
  src/
    __init__.py
    generate_data.py
    features.py
    train.py
    evaluate.py
    utils.py
  data/
  models/
  outputs/
  README.md
  requirements.txt
```

## Input data format

The training data must contain exactly:
- `date` in `YYYY-MM-DD` format
- `total_sales` as numeric daily totals

Example:

```csv
date,total_sales
2025-01-01,512.40
2025-01-02,486.30
```

## Setup

From the `sales-forecasting` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version: **3.10+**

## Run the pipeline

### 1) Generate synthetic data

```bash
python -m src.generate_data
```

Default output:
- `data/daily_sales.csv`

You can customize generation:

```bash
python -m src.generate_data --days 730 --start-date 2024-01-01 --seed 42
```

### 2) Train baseline + ML models

```bash
python -m src.train
```

This script:
- Creates required features:
  - `day_of_week`
  - `is_weekend`
  - `month`
  - `lag_1`
  - `lag_7`
  - `rolling_mean_7`
  - `rolling_mean_14`
- Uses a time-based split (first 80% train, last 20% test)
- Evaluates:
  - Baseline (predict yesterday's sales)
  - `LinearRegression`
  - `GradientBoostingRegressor`
- Saves best model to `models/model.joblib`
- Saves metrics summary to `outputs/metrics.json`

### 3) Evaluate and create outputs

```bash
python -m src.evaluate
```

This script saves:
- `outputs/predictions.csv` with columns: `date,actual,predicted,baseline`
- `outputs/actual_vs_pred.png`

## Technical design decisions

- Time-aware split: first 80% of dates for training, last 20% for testing
- Baseline first: compares ML models against a lag-1 baseline (yesterday's sales)
- Leakage-safe features: lag and rolling features use only historical values
- Model selection rule: choose best model by lowest RMSE (MAE as secondary context)
- Reproducibility: synthetic data generation uses a fixed random seed by default

## How to interpret the metrics

- **MAE (Mean Absolute Error):** average absolute prediction error in sales units.
- **RMSE (Root Mean Squared Error):** like MAE but penalizes larger misses more.

Lower is better for both.

Use baseline vs model metrics to check whether ML is providing real business value beyond a simple yesterday-based prediction.

## Expected outputs

After running all scripts, you should have:
- `data/daily_sales.csv`
- `models/model.joblib`
- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/actual_vs_pred.png`

## Future improvements

- Add holiday/promotion/weather/event features
- Add hyperparameter tuning (grid/random search)
- Use walk-forward (rolling) validation
- Add model explainability and feature importance plots
- Deploy as an API or dashboard for daily forecasting operations

## Practical use notes

This project is built to be practical for small restaurant operations:
- Compare model performance against a simple baseline before trusting forecasts
- Re-run training as new sales data is added
- Use `predictions.csv` and the plot to monitor model behavior over time
