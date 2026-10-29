"""Fixed-seed regression, causal recursive forecasts, and rolling backtests."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.forecasting.data import TARGET
from src.forecasting.features import (
    CATEGORICAL, LOOKBACK, feature_row, training_examples,
)

METHODS = ("seasonal_naive", "moving_average", "random_forest")
MODEL_PARAMETERS = {
    "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 8,
    "max_features": 1.0, "random_state": 42, "n_jobs": 1,
}


def fit_regressor(frame, cutoff, parameters=None):
    features, labels = training_examples(frame, cutoff)
    encoder = ColumnTransformer(
        [("series", OneHotEncoder(
            handle_unknown="error", sparse_output=False
        ), CATEGORICAL)], remainder="passthrough",
    )
    model = Pipeline([
        ("features", encoder),
        ("regressor", RandomForestRegressor(
            **(MODEL_PARAMETERS if parameters is None else parameters)
        )),
    ])
    model.fit(features, labels)
    return model


def forecast(frame, start, method, model=None, horizon=7):
    """Forecast using only observations strictly before start.

    Each method has its own evolving history. Predictions, not hidden actuals,
    are appended during the forecast window. New actuals become available only
    when the next independent backtest origin is reached.
    """
    if method not in METHODS or not 1 <= horizon <= 7:
        raise ValueError("Unknown method or horizon outside 1..7.")
    if method == "random_forest" and model is None:
        raise ValueError("A fitted regressor is required.")
    start = pd.Timestamp(start)
    past = frame[frame.date.lt(start)]
    histories = {}
    for key, group in past.groupby(CATEGORICAL):
        group = group.sort_values("date")
        expected = pd.date_range(
            end=start - pd.offsets.Day(1), periods=28
        )
        if not np.array_equal(
            group.date.tail(28).to_numpy(), expected.to_numpy()
        ):
            raise ValueError("History must end the day before origin.")
        histories[key] = group[TARGET].tail(LOOKBACK).astype(float).to_list()
    if not histories:
        raise ValueError("No history available before the forecast origin.")
    records = []
    keys = sorted(histories)
    for step in range(1, horizon + 1):
        date = start + pd.offsets.Day(step - 1)
        if method == "random_forest":
            features = pd.DataFrame([
                feature_row(date, *key, histories[key]) for key in keys
            ])
            predictions = np.maximum(0, model.predict(features))
        elif method == "seasonal_naive":
            predictions = [histories[key][-7] for key in keys]
        else:
            predictions = [np.mean(histories[key][-28:]) for key in keys]
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite forecast produced.")
        for key, prediction in zip(keys, predictions):
            histories[key].append(float(prediction))
            records.append({
                "origin": start - pd.offsets.Day(1), "date": date,
                "store_id": key[0], "product_id": key[1],
                "horizon_day": step, "method": method,
                "prediction": float(prediction),
            })
    return pd.DataFrame(records)


def backtest(frame, start, end, method, model=None):
    """Weekly 7-day origins; skip incomplete trailing windows."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_start = end - pd.offsets.Day(6)
    origins = pd.date_range(start, last_start, freq="7D")
    if len(origins) == 0:
        raise ValueError("Backtest needs at least one complete 7-day window.")
    predictions = pd.concat([
        forecast(frame, origin, method, model) for origin in origins
    ], ignore_index=True)
    actual = frame[["date", "store_id", "product_id", TARGET]]
    predictions = predictions.merge(
        actual, on=["date", "store_id", "product_id"],
        how="left", validate="many_to_one",
    )
    if predictions[TARGET].isna().any():
        raise ValueError("Missing backtest actuals; cannot evaluate.")
    return predictions.rename(columns={TARGET: "actual"})


def forecast_all_methods(frame, start, model):
    """Use the same observed history and cutoff for all three candidates."""
    return pd.concat([
        forecast(frame, start, method, model) for method in METHODS
    ], ignore_index=True)
