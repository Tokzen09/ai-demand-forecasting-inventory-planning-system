"""Saved-artifact integrity, selector behavior, and filtered forecast views."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.forecasting.dashboard_data import (
    ForecastArtifactError, artifact_revision, chart_data, comparison_metrics,
    load_forecast_artifacts, verify_history,
)
from src.forecasting.data import fingerprint
from src.forecasting.engine import (
    METHODS, backtest, fit_regressor, forecast_all_methods,
)
from src.forecasting.metrics import evaluation_table, metrics


@pytest.fixture
def saved(tmp_path):
    rows = []
    for store, scale in (("S1", 1), ("S2", 10)):
        dates = pd.date_range("2024-01-01", periods=90)
        product = "P1" if store == "S1" else "P2"
        for position, date in enumerate(dates):
            rows.append({
                "date": date, "store_id": store, "product_id": product,
                "product_name": f"Synthetic product {product}",
                "customer_demand": scale * (10 + position % 7),
            })
    history = pd.DataFrame(rows)
    model = fit_regressor(history, "2024-01-31", {
        "n_estimators": 5, "random_state": 42, "n_jobs": 1,
        "min_samples_leaf": 2, "max_depth": 4,
    })
    future = forecast_all_methods(history, "2024-03-31", model)
    future.to_csv(tmp_path / "next_7_days_all_methods.csv", index=False)
    backtests = []
    for split, start, end in (
        ("validation", "2024-02-01", "2024-02-07"),
        ("test", "2024-03-01", "2024-03-07"),
    ):
        for method in METHODS:
            prediction = backtest(history, start, end, method, model)
            prediction["split"] = split
            backtests.append(prediction)
    backtests = pd.concat(backtests, ignore_index=True)
    backtests.to_csv(tmp_path / "backtest_predictions.csv", index=False)
    evaluation_table(backtests).to_csv(
        tmp_path / "evaluation.csv", index=False
    )
    metadata = {
        "target": "customer_demand", "horizon": 7, "series": 2,
        "selected_method": "random_forest",
        "data_sha256": fingerprint(history),
        "future_forecast": ["2024-03-31", "2024-04-06"],
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))
    return tmp_path, history, future


def test_all_methods_have_complete_forecasts(saved):
    directory, history, future = saved
    assert len(future) == 42
    assert set(future.method) == set(METHODS)
    artifacts = load_forecast_artifacts(directory)
    verify_history(artifacts, history)
    assert len(artifact_revision(directory)) == 4


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "nan", "method"]
)
def test_invalid_forecasts_are_rejected(saved, corruption):
    directory, _, frame = saved
    if corruption == "missing":
        frame = frame.iloc[1:]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == "nan":
        frame.loc[0, "prediction"] = np.nan
    else:
        frame = frame[frame.method.ne("moving_average")]
    frame.to_csv(directory / "next_7_days_all_methods.csv", index=False)
    with pytest.raises(ForecastArtifactError):
        load_forecast_artifacts(directory)


def test_missing_artifacts_have_actionable_error(tmp_path):
    with pytest.raises(ForecastArtifactError, match="Run python"):
        artifact_revision(tmp_path)


def test_modified_history_is_stale(saved):
    directory, history, _ = saved
    artifacts = load_forecast_artifacts(directory)
    history.loc[0, "customer_demand"] += 1
    with pytest.raises(ForecastArtifactError, match="do not match"):
        verify_history(artifacts, history)


def test_corrupt_saved_metrics_are_rejected(saved):
    directory, _, _ = saved
    path = directory / "evaluation.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.scope.eq("overall"), "mae"] = 9999
    frame.to_csv(path, index=False)
    with pytest.raises(ForecastArtifactError):
        load_forecast_artifacts(directory)


def test_subset_metrics_are_computed_from_saved_errors(saved):
    directory, history, _ = saved
    artifacts = load_forecast_artifacts(directory)
    keys = history[history.store_id.eq("S1")][["store_id", "product_id"]]
    comparison = comparison_metrics(artifacts, keys, "test")
    raw = artifacts.backtests
    for row in comparison.itertuples():
        subset = raw[
            raw.store_id.eq("S1") & raw.split.eq("test")
            & raw.method.eq(row.method)
        ]
        expected = metrics(subset.actual, subset.prediction)
        assert row.mae == expected["mae"]
        assert row.wape_pct == expected["wape_pct"]
        assert row.n == 7
    overall = comparison_metrics(artifacts, history, "test")
    assert overall.n.eq(14).all()


def test_chart_separates_history_and_future_and_filters_series(saved):
    directory, history, _ = saved
    artifacts = load_forecast_artifacts(directory)
    keys = history[history.store_id.eq("S1")]
    data = chart_data(history, artifacts.forecasts, keys, "seasonal_naive")
    actual = data[data.series.eq("Historical customer demand")]
    future = data[data.series.eq("Seven-day forecast")]
    assert len(actual) == 56
    assert len(future) == 8  # Seven forecasts plus an observed bridge point.
    assert actual.date.max() == pd.Timestamp("2024-03-30")
    assert future.date.max() == pd.Timestamp("2024-04-06")
    assert actual.demand.max() < 100


def panel_app():
    """AppTest extracts this function body into an independent app script."""
    import pandas as pd
    import streamlit as st
    from dashboard import forecast_panel

    history = pd.read_csv(
        forecast_panel.ARTIFACT_DIR / "test_history.csv", parse_dates=["date"]
    )
    stores = st.sidebar.multiselect(
        "Stores", ["S1", "S2"], default=["S1", "S2"], key="panel_stores"
    )
    products = st.sidebar.multiselect(
        "Products", ["P1", "P2"], default=["P1", "P2"],
        key="panel_products",
    )
    selected = history[
        history.store_id.isin(stores) & history.product_id.isin(products)
    ]
    forecast_panel.render_forecast_panel(history, selected)


def test_forecast_panel_model_store_and_period_controls(saved, monkeypatch):
    from dashboard import forecast_panel

    directory, history, future = saved
    history.to_csv(directory / "test_history.csv", index=False)
    monkeypatch.setattr(forecast_panel, "ARTIFACT_DIR", directory)
    st.cache_data.clear()
    app = AppTest.from_function(panel_app).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="forecast_model").value == "random_forest"
    assert len(app.dataframe[0].value) == 14
    app.sidebar.multiselect(key="panel_products").set_value(["P1"]).run()
    assert len(app.dataframe[0].value) == 7
    assert app.dataframe[0].value.product_id.eq("P1").all()
    app.sidebar.multiselect(key="panel_stores").set_value(["S1"]).run()
    app.selectbox(key="forecast_model").set_value("moving_average").run()
    assert not app.exception
    daily = app.dataframe[0].value
    assert len(daily) == 7
    expected = future[
        future.store_id.eq("S1") & future.method.eq("moving_average")
    ].prediction.to_numpy()
    np.testing.assert_allclose(daily.prediction, expected)
    app.segmented_control(key="forecast_evaluation_period").set_value(
        "validation"
    ).run()
    assert not app.exception
    assert len(app.dataframe[1].value) == 3
    assert app.dataframe[1].value.n.eq(7).all()


def test_missing_results_leave_historical_dashboard_available(monkeypatch):
    from dashboard import forecast_panel

    def missing(_):
        raise ForecastArtifactError("Missing")

    monkeypatch.setenv("DASHBOARD_DATA_SOURCE", "csv")
    monkeypatch.setattr(forecast_panel, "artifact_revision", missing)
    st.cache_data.clear()
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "380,002"
    assert any(
        "Forecasts are missing" in warning.value for warning in app.warning
    )


def test_refresh_revision_changes_when_forecasts_change(saved):
    directory, _, future = saved
    first = artifact_revision(directory)
    future.to_csv(directory / "next_7_days_all_methods.csv", index=False)
    second = artifact_revision(directory)
    assert first != second
