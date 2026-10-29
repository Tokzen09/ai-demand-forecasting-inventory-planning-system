"""One feature definition shared by training and recursive prediction."""

import numpy as np
import pandas as pd

from src.forecasting.data import TARGET

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
LOOKBACK = 28
CATEGORICAL = ["store_id", "product_id"]


def feature_row(date, store, product, history):
    """History ends at t-1; same-day outcomes are never accepted."""
    if len(history) < LOOKBACK:
        raise ValueError("At least 28 prior daily observations are required.")
    history = np.asarray(history[-LOOKBACK:], dtype=float)
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Demand history must be finite and nonnegative.")
    date = pd.Timestamp(date)
    angle = 2 * np.pi * (date.dayofyear - 1) / 365.25
    row = {
        "store_id": store, "product_id": product,
        "weekday": date.dayofweek, "month": date.month,
        "weekend": int(date.dayofweek >= 5),
        "year_sin": float(np.sin(angle)), "year_cos": float(np.cos(angle)),
        "elapsed_days": (date - pd.Timestamp("2024-01-01")).days,
    }
    row.update({f"lag_{lag}": history[-lag] for lag in LAGS})
    for window in WINDOWS:
        row[f"mean_{window}"] = float(np.mean(history[-window:]))
        row[f"std_{window}"] = float(np.std(history[-window:], ddof=0))
    return row


def training_examples(frame, cutoff):
    """Filter labels by cutoff BEFORE constructing any training features."""
    available = frame[frame.date.le(pd.Timestamp(cutoff))]
    rows, labels = [], []
    for (store, product), group in available.groupby(CATEGORICAL):
        group = group.sort_values("date")
        values = group[TARGET].to_numpy(dtype=float)
        dates = group.date.to_list()
        for position in range(LOOKBACK, len(group)):
            rows.append(feature_row(
                dates[position], store, product,
                values[position - LOOKBACK:position],
            ))
            labels.append(values[position])
    if not rows:
        raise ValueError("No training examples after the 28-day warm-up.")
    return pd.DataFrame(rows), np.asarray(labels)
