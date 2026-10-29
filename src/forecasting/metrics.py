"""Demand forecast metrics; WAPE is percentage, undefined at zero demand."""

import numpy as np
import pandas as pd


def metrics(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (
        actual.size == 0 or actual.shape != prediction.shape
        or not np.isfinite(actual).all() or not np.isfinite(prediction).all()
    ):
        raise ValueError("Metrics require aligned finite nonempty values.")
    errors = actual - prediction
    denominator = np.abs(actual).sum()
    return {
        "n": len(actual), "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "wape_pct": (
            float(100 * np.abs(errors).sum() / denominator)
            if denominator else np.nan
        ),
    }


def evaluation_table(predictions):
    records = []
    scopes = {
        "overall": [], "store": ["store_id"], "product": ["product_id"],
        "store_product": ["store_id", "product_id"],
        "horizon": ["horizon_day"],
    }
    for (split, method), group in predictions.groupby(["split", "method"]):
        for scope, columns in scopes.items():
            groups = group.groupby(columns) if columns else [((), group)]
            for key, subset in groups:
                key = key if isinstance(key, tuple) else (key,)
                records.append({
                    "split": split, "method": method, "scope": scope,
                    **dict(zip(columns, key)),
                    **metrics(subset.actual, subset.prediction),
                })
    return pd.DataFrame(records)


def select_method(table):
    """Select on validation rows only; never inspect test scores."""
    validation = table[
        table.split.eq("validation") & table.scope.eq("overall")
    ].copy()
    if validation.empty:
        raise ValueError("Validation metrics are required for selection.")
    return validation.sort_values(
        ["wape_pct", "mae", "method"], na_position="last"
    ).iloc[0]["method"]
