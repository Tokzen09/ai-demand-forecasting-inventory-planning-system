"""Causality, recursive forecasting, metrics, sources, and reproducibility."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.database import DatabaseError
from src.forecasting import data
from src.forecasting.data import fingerprint, validate_demand
from src.forecasting.engine import backtest, fit_regressor, forecast
from src.forecasting.features import feature_row, training_examples
from src.forecasting.metrics import evaluation_table, metrics, select_method


@pytest.fixture
def history():
    dates = pd.date_range("2024-01-01", periods=90)
    records = []
    for store, scale in (("S1", 1), ("S2", 10)):
        for index, date in enumerate(dates):
            records.append({
                "date": date, "store_id": store, "product_id": "P1",
                "customer_demand": scale * (10 + index % 7),
            })
    return validate_demand(pd.DataFrame(records))


def test_lags_and_rolling_use_prior_values_only():
    row = feature_row("2024-02-01", "S1", "P1", list(range(1, 29)))
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["lag_28"] == 1
    assert row["mean_7"] == 25
    assert row["mean_28"] == 14.5


def test_training_cutoff_isolates_future_and_other_outcomes(history):
    cutoff = pd.Timestamp("2024-02-29")
    first_x, first_y = training_examples(history, cutoff)
    changed = history.copy()
    changed.loc[changed.date.gt(cutoff), "customer_demand"] = 999999
    changed["promotion_flag"] = 1
    changed["estimated_customer_demand"] = 999999
    changed["units_sold"] = 999999
    second_x, second_y = training_examples(changed, cutoff)
    pd.testing.assert_frame_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    excluded = {"promotion_flag", "estimated_customer_demand", "units_sold"}
    assert not excluded & set(first_x.columns)


class LastDemandRegressor:
    """A spy makes recursive lag behavior directly observable."""

    def predict(self, features):
        return features.lag_1.to_numpy() + 1


def test_recursive_horizons_never_read_hidden_actuals(history):
    origin = pd.Timestamp("2024-03-01")
    changed = history.copy()
    changed.loc[changed.date.ge(origin), "customer_demand"] = 123456
    first = forecast(history, origin, "random_forest", LastDemandRegressor())
    second = forecast(changed, origin, "random_forest", LastDemandRegressor())
    pd.testing.assert_frame_equal(first, second)
    values = first[first.store_id.eq("S1")].prediction.to_numpy()
    np.testing.assert_array_equal(np.diff(values), np.ones(6))
    assert len(first) == 14
    assert first.date.max() == origin + pd.offsets.Day(6)


@pytest.mark.parametrize("method", ["seasonal_naive", "moving_average"])
def test_baselines_ignore_future_actuals(history, method):
    origin = pd.Timestamp("2024-03-01")
    changed = history.copy()
    changed.loc[changed.date.ge(origin), "customer_demand"] = 99999
    pd.testing.assert_frame_equal(
        forecast(history, origin, method), forecast(changed, origin, method)
    )


def test_seasonal_naive_matches_same_weekday(history):
    result = forecast(history, "2024-03-01", "seasonal_naive")
    for row in result.itertuples():
        actual = history[
            history.store_id.eq(row.store_id)
            & history.date.eq(row.date - pd.offsets.Day(7))
        ].customer_demand.iloc[0]
        assert row.prediction == actual


def test_moving_average_recurses_on_predictions(history):
    origin = pd.Timestamp("2024-03-01")
    result = forecast(history, origin, "moving_average")
    values = history[
        history.store_id.eq("S1") & history.date.lt(origin)
    ].customer_demand.to_list()
    for prediction in result[result.store_id.eq("S1")].prediction:
        assert prediction == pytest.approx(np.mean(values[-28:]))
        values.append(prediction)


def test_backtest_windows_and_series_boundaries(history):
    result = backtest(
        history, "2024-03-01", "2024-03-15", "seasonal_naive"
    )
    assert len(result) == 2 * 7 * 2
    assert result.origin.nunique() == 2
    assert result.date.max() == pd.Timestamp("2024-03-14")
    assert not result.duplicated(["date", "store_id", "product_id"]).any()
    assert result[result.store_id.eq("S2")].prediction.min() >= 100


@pytest.mark.parametrize("problem", ["null", "gap", "negative", "duplicate"])
def test_rejects_invalid_forecast_input(history, problem):
    changed = history.copy()
    if problem == "null":
        changed.loc[0, "customer_demand"] = np.nan
    elif problem == "negative":
        changed.loc[0, "customer_demand"] = -1
    elif problem == "gap":
        changed = changed.drop(index=20)
    else:
        changed = pd.concat([changed, changed.iloc[[0]]])
    with pytest.raises(ValueError):
        validate_demand(changed)


def test_cannot_substitute_sales_for_missing_demand(history):
    changed = history.rename(columns={"customer_demand": "units_sold"})
    with pytest.raises(ValueError, match="customer_demand"):
        validate_demand(changed)


def test_metrics_and_zero_denominator():
    result = metrics([10, 20], [8, 24])
    assert result["mae"] == 3
    assert result["rmse"] == pytest.approx(np.sqrt(10))
    assert result["wape_pct"] == 20
    assert np.isnan(metrics([0, 0], [0, 1])["wape_pct"])
    with pytest.raises(ValueError):
        metrics([1, np.nan], [1, 2])


def test_selection_ignores_final_test_scores():
    table = pd.DataFrame([
        {"split": "validation", "scope": "overall", "method": "A",
         "mae": 1, "wape_pct": 10},
        {"split": "validation", "scope": "overall", "method": "B",
         "mae": 2, "wape_pct": 20},
        {"split": "test", "scope": "overall", "method": "A",
         "mae": 9999, "wape_pct": 9999},
        {"split": "test", "scope": "overall", "method": "B",
         "mae": 0, "wape_pct": 0},
    ])
    assert select_method(table) == "A"


def test_regressor_reproducibility_and_artifact_round_trip(history, tmp_path):
    parameters = {
        "n_estimators": 5, "max_depth": 4, "min_samples_leaf": 2,
        "random_state": 42, "n_jobs": 1,
    }
    first = fit_regressor(history, "2024-02-29", parameters)
    second = fit_regressor(history, "2024-02-29", parameters)
    expected = forecast(history, "2024-03-01", "random_forest", first)
    pd.testing.assert_frame_equal(
        expected, forecast(history, "2024-03-01", "random_forest", second)
    )
    path = tmp_path / "test.joblib"
    joblib.dump(first, path)
    pd.testing.assert_frame_equal(
        expected, forecast(
            history, "2024-03-01", "random_forest", joblib.load(path)
        )
    )


def test_auto_source_fallback_is_explicit(history, tmp_path, monkeypatch):
    path = tmp_path / "demand.csv"
    history.to_csv(path, index=False)

    def unavailable():
        raise DatabaseError("Unavailable")

    monkeypatch.setattr(data, "load_mysql_sales", unavailable)
    with pytest.warns(RuntimeWarning, match="CSV"):
        result, source = data.load_demand("auto", path)
    assert source == "csv"
    assert fingerprint(history) == fingerprint(result)
    with pytest.raises(DatabaseError):
        data.load_demand("mysql", path)


def test_invalid_mysql_data_does_not_silently_fall_back(history, monkeypatch):
    monkeypatch.setattr(data, "load_mysql_sales", lambda: history.iloc[1:])
    with pytest.raises(ValueError, match="coverage"):
        data.load_demand("auto", Path("not-read.csv"))


def test_group_metrics_include_each_store_product(history):
    result = backtest(history, "2024-03-01", "2024-03-07", "moving_average")
    result["split"] = "validation"
    table = evaluation_table(result)
    assert len(table[table.scope.eq("store_product")]) == 2
    assert len(table[table.scope.eq("horizon")]) == 7
