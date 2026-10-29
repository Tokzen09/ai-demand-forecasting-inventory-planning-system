"""Validate saved forecast files and prepare filtered dashboard views."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.forecasting.data import fingerprint
from src.forecasting.engine import METHODS
from src.forecasting.metrics import metrics

ARTIFACT_DIR = ROOT / "data/processed/phase3a"
ARTIFACT_FILES = (
    "next_7_days_all_methods.csv", "backtest_predictions.csv",
    "evaluation.csv", "run_metadata.json",
)
SERIES_KEYS = ["store_id", "product_id"]
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive", "moving_average": "Moving average",
    "random_forest": "Random forest",
}


class ForecastArtifactError(ValueError):
    """Missing, incompatible, or stale saved forecasting results."""


@dataclass
class ForecastArtifacts:
    forecasts: pd.DataFrame
    backtests: pd.DataFrame
    evaluation: pd.DataFrame
    metadata: dict


def artifact_revision(directory=ARTIFACT_DIR):
    """Use every file's revision so replacements invalidate the UI cache."""
    directory = Path(directory)
    try:
        return tuple(
            (name, (directory / name).stat().st_mtime_ns,
             (directory / name).stat().st_size)
            for name in ARTIFACT_FILES
        )
    except OSError as error:
        raise ForecastArtifactError(
            "Saved forecasts are missing. Run python -m src.forecast."
        ) from error


def load_forecast_artifacts(directory=ARTIFACT_DIR):
    """Read CSV/JSON without deserializing models or training."""
    directory = Path(directory)
    try:
        metadata = json.loads(
            (directory / "run_metadata.json").read_text(encoding="utf-8")
        )
        forecasts = pd.read_csv(directory / "next_7_days_all_methods.csv")
        backtests = pd.read_csv(directory / "backtest_predictions.csv")
        evaluation = pd.read_csv(directory / "evaluation.csv")
        base_columns = {
            "date", "origin", "store_id", "product_id", "horizon_day",
            "method", "prediction",
        }
        if not base_columns.issubset(forecasts.columns):
            raise ValueError("Missing forecast columns")
        if not (base_columns | {"actual", "split"}).issubset(
            backtests.columns
        ):
            raise ValueError("Missing backtest columns")
        if not {"split", "method", "scope", "n", "mae", "rmse",
                "wape_pct"}.issubset(evaluation.columns):
            raise ValueError("Missing evaluation columns")
        if (
            metadata["target"] != "customer_demand"
            or metadata["horizon"] != 7
            or metadata["selected_method"] not in METHODS
        ):
            raise ValueError("Incompatible forecast metadata")
        for frame in (forecasts, backtests):
            for column in ("date", "origin"):
                frame[column] = pd.to_datetime(frame[column], errors="raise")
            if frame[list(base_columns)].isna().any().any():
                raise ValueError("Missing forecast values")
            values = pd.to_numeric(frame.prediction, errors="raise")
            if not np.isfinite(values).all() or values.lt(0).any():
                raise ValueError("Invalid forecast values")
            frame["prediction"] = values
            if set(frame.method) != set(METHODS):
                raise ValueError("All three forecast methods must be saved")
            if not frame.horizon_day.isin(range(1, 8)).all():
                raise ValueError("Invalid horizon")
            expected = (frame.date - frame.origin).dt.days
            if not expected.eq(frame.horizon_day).all():
                raise ValueError("Dates disagree with saved horizon")
        identity = ["origin", *SERIES_KEYS, "method", "horizon_day"]
        if forecasts.duplicated(identity).any():
            raise ValueError("Duplicate forecasts")
        if forecasts.origin.nunique() != 1:
            raise ValueError("Expected one forecast origin")
        if not forecasts.groupby(
            [*SERIES_KEYS, "method"]
        ).size().eq(7).all():
            raise ValueError("Incomplete seven-day forecasts")
        if not forecasts.groupby(SERIES_KEYS).method.nunique().eq(3).all():
            raise ValueError("Incomplete method coverage")
        series_count = forecasts[SERIES_KEYS].drop_duplicates().shape[0]
        if series_count != metadata["series"]:
            raise ValueError("Series count differs from metadata")
        if (
            forecasts.date.min() != pd.Timestamp(
                metadata["future_forecast"][0]
            )
            or forecasts.date.max() != pd.Timestamp(
                metadata["future_forecast"][1]
            )
        ):
            raise ValueError("Forecast range differs from metadata")
        if backtests.duplicated(["split", *identity]).any():
            raise ValueError("Duplicate backtest rows")
        if set(backtests.split) != {"validation", "test"}:
            raise ValueError("Both evaluation periods are required")
        if (
            not np.isfinite(backtests.actual).all()
            or backtests.actual.lt(0).any()
        ):
            raise ValueError("Invalid backtest actuals")
        # Confirm the saved overall evaluation matches its underlying rows.
        for (split, method), group in backtests.groupby(["split", "method"]):
            row = evaluation[
                evaluation.scope.eq("overall") & evaluation.split.eq(split)
                & evaluation.method.eq(method)
            ]
            if len(row) != 1:
                raise ValueError("Missing or duplicate overall metrics")
            calculated = metrics(group.actual, group.prediction)
            for column, value in calculated.items():
                if not np.isclose(
                    row.iloc[0][column], value, equal_nan=True,
                    rtol=1e-8, atol=1e-8,
                ):
                    raise ValueError("Evaluation differs from predictions")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ForecastArtifactError(
            "Saved forecasting results are missing or inconsistent. "
            "Run python -m src.forecast to regenerate them."
        ) from error
    return ForecastArtifacts(forecasts, backtests, evaluation, metadata)


def verify_history(artifacts, history):
    """Reject stale source data instead of drawing mismatched forecasts."""
    try:
        matches = fingerprint(history) == artifacts.metadata["data_sha256"]
    except ValueError as error:
        raise ForecastArtifactError("History is invalid.") from error
    if not matches:
        raise ForecastArtifactError(
            "Saved forecasts do not match the current demand history. "
            "Run python -m src.forecast for this data source."
        )


def filter_series(frame, keys):
    """Match exact store-product pairs rather than a Cartesian set."""
    return frame.merge(
        keys[SERIES_KEYS].drop_duplicates(), on=SERIES_KEYS,
        how="inner", validate="many_to_one",
    )


def comparison_metrics(artifacts, keys, split):
    """Use saved overall results or recompute subset errors from rows."""
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test.")
    all_keys = artifacts.forecasts[SERIES_KEYS].drop_duplicates()
    selected_keys = keys[SERIES_KEYS].drop_duplicates()
    selected_set = set(map(tuple, selected_keys.values))
    if selected_set == set(map(tuple, all_keys.values)):
        return artifacts.evaluation[
            artifacts.evaluation.scope.eq("overall")
            & artifacts.evaluation.split.eq(split)
        ][["method", "n", "mae", "rmse", "wape_pct"]].reset_index(drop=True)
    rows = filter_series(artifacts.backtests, selected_keys)
    rows = rows[rows.split.eq(split)]
    return pd.DataFrame([
        {"method": method, **metrics(group.actual, group.prediction)}
        for method, group in rows.groupby("method")
    ])


def chart_data(history, forecasts, keys, method, days=56):
    """Aggregate demand, separating observed and future points."""
    future = filter_series(forecasts, keys)
    future = future[future.method.eq(method)]
    if future.empty:
        raise ForecastArtifactError("No forecasts for this selection.")
    origin = future.origin.iloc[0]
    past = filter_series(history, keys)
    first = origin - pd.offsets.Day(days - 1)
    past = past[past.date.between(first, origin)]
    actuals = past.groupby("date", as_index=False).customer_demand.sum()
    actuals = actuals.rename(columns={"customer_demand": "demand"})
    actuals["series"] = "Historical customer demand"
    predicted = future.groupby("date", as_index=False).prediction.sum()
    predicted = predicted.rename(columns={"prediction": "demand"})
    predicted["series"] = "Seven-day forecast"
    # An observed endpoint joins the lines; future actuals stay unavailable.
    bridge = actuals.tail(1).copy()
    bridge["series"] = "Seven-day forecast"
    return pd.concat([actuals, bridge, predicted], ignore_index=True)
