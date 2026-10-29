"""Saved-results dashboard section; no inference or training on UI reruns."""

import plotly.express as px
import streamlit as st

from src.forecasting.dashboard_data import (
    ARTIFACT_DIR, MODEL_LABELS, ForecastArtifactError, artifact_revision,
    chart_data, comparison_metrics, filter_series, load_forecast_artifacts,
    verify_history,
)


@st.cache_data(ttl=60, max_entries=4)
def cached_artifacts(directory, revision):
    return load_forecast_artifacts(directory)


@st.cache_data(ttl=60, max_entries=4)
def verified_history(history, fingerprint):
    from src.forecasting.data import fingerprint as data_fingerprint

    return data_fingerprint(history) == fingerprint


def render_forecast_panel(history, filtered):
    st.subheader("Seven-day customer-demand forecasts")
    try:
        revision = artifact_revision(ARTIFACT_DIR)
        artifacts = cached_artifacts(str(ARTIFACT_DIR), revision)
        if not verified_history(history, artifacts.metadata["data_sha256"]):
            verify_history(artifacts, history)
    except (ForecastArtifactError, ValueError):
        st.warning(
            "Forecasts are missing, invalid, or stale for this history. "
            "Run python -m src.forecast --source mysql (or --source csv), "
            "then refresh data. Historical exploration remains available."
        )
        return

    methods = list(MODEL_LABELS)
    selected = artifacts.metadata["selected_method"]
    method = st.sidebar.selectbox(
        "Forecast model", methods, index=methods.index(selected),
        format_func=MODEL_LABELS.get, key="forecast_model",
    )
    st.caption(
        f"Globally selected by validation WAPE: {MODEL_LABELS[selected]}. "
        "Changing the displayed model does not change that selection."
    )
    first, last = artifacts.metadata["future_forecast"]
    origin = artifacts.forecasts.origin.iloc[0].date()
    st.info(
        f"Saved forecasts: {first} to {last}, as of {origin}. "
        "These follow the historical dataset, not the current calendar week. "
        "Future actual demand and promotion plans are unavailable."
    )
    keys = filtered[["store_id", "product_id"]].drop_duplicates()
    forecasts = filter_series(artifacts.forecasts, keys)
    forecasts = forecasts[forecasts.method.eq(method)]
    st.metric(
        "Forecast demand over seven days", f"{forecasts.prediction.sum():,.1f}"
    )
    st.caption(
        "Demand is summed across the selected stores and products. "
        "Point forecasts have no uncertainty bands; all data is synthetic."
    )
    series = chart_data(history, artifacts.forecasts, keys, method)
    figure = px.line(
        series, x="date", y="demand", color="series", line_dash="series",
        labels={"date": "Date", "demand": "Customer demand (units)",
                "series": ""},
        color_discrete_map={
            "Historical customer demand": "#087f8c",
            "Seven-day forecast": "#e07a32",
        },
    )
    st.plotly_chart(
        figure, width="stretch", key="demand_forecast_chart",
        alt="Historical customer demand and the selected seven-day forecast",
    )
    st.subheader("Daily demand forecasts")
    names = filtered[["product_id", "product_name"]].drop_duplicates()
    daily = forecasts.merge(names, on="product_id", validate="many_to_one")
    daily = daily[[
        "date", "store_id", "product_id", "product_name", "method",
        "prediction",
    ]].sort_values(["date", "store_id", "product_id"])
    daily["method"] = daily.method.map(MODEL_LABELS)
    st.dataframe(
        daily, hide_index=True, width="stretch", key="daily_forecast_table",
        column_config={"prediction": st.column_config.NumberColumn(
            "Forecast demand", format="%.2f"
        )}, alt="Saved daily forecasts for the selected stores and products",
    )
    st.download_button(
        "Download selected forecasts", daily.to_csv(index=False),
        file_name="selected_demand_forecasts.csv", mime="text/csv",
        key="download_forecasts",
    )
    st.subheader("Model comparison")
    split = st.segmented_control(
        "Evaluation period", ["validation", "test"], default="test",
        format_func=lambda value: "Validation" if value == "validation"
        else "Final test", key="forecast_evaluation_period",
    )
    if split is None:
        st.info("Choose an evaluation period to compare models.")
        return
    period = artifacts.metadata.get(f"{split}_period")
    if period:
        st.caption(f"Saved {split} period: {period[0]} to {period[1]}")
    comparison = comparison_metrics(artifacts, keys, split)
    comparison["selected_on_validation"] = comparison.method.eq(selected)
    comparison["method"] = comparison.method.map(MODEL_LABELS)
    st.dataframe(
        comparison, hide_index=True, width="stretch", key="comparison_table",
        column_config={
            "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
            "wape_pct": st.column_config.NumberColumn(
                "WAPE (%)", format="%.3f"
            ),
        }, alt="Validation or final test metrics for all three saved methods",
    )
    st.caption(
        "Metrics describe historical seven-day backtests, not accuracy on "
        "future forecast dates. Subset metrics use saved per-series errors; "
        "WAPE is not averaged across products. Lower errors are better. "
        "Viewing test results does not change validation-based selection."
    )
