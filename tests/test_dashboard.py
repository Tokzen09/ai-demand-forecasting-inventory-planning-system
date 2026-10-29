"""Exercise the exploratory dashboard without a running browser server."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "data/raw/synthetic_retail_sales.csv").exists(),
    reason="Generate the default dataset before the dashboard smoke test.",
)
def test_dashboard_filters(monkeypatch):
    monkeypatch.setenv("DASHBOARD_DATA_SOURCE", "csv")
    app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[0].label == "Units sold"
    app.sidebar.multiselect[0].set_value([]).run(timeout=30)
    assert not app.exception
    assert app.info[0].value == "Select at least one store and product."


def test_mysql_dashboard_filters(monkeypatch):
    import pandas as pd
    from src import data_access

    frame = pd.read_csv(
        ROOT / "data/raw/synthetic_retail_sales.csv", parse_dates=["date"]
    )
    monkeypatch.setenv("DASHBOARD_DATA_SOURCE", "mysql")
    monkeypatch.setattr(data_access, "load_mysql_sales", lambda: frame)
    app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "380,002"
    app.sidebar.multiselect[0].set_value(["S001"]).run(timeout=30)
    assert not app.exception
    expected = frame[frame.store_id.eq("S001")].units_sold.sum()
    assert app.metric[0].value == f"{expected:,}"
    app.sidebar.multiselect[1].set_value([]).run(timeout=30)
    assert app.info[0].value == "Select at least one store and product."


def test_mysql_dashboard_connection_error(monkeypatch):
    from src import data_access
    from src.database import DatabaseError

    monkeypatch.setenv("DASHBOARD_DATA_SOURCE", "mysql")

    def fail():
        raise DatabaseError("MySQL unavailable; configure connection.")

    monkeypatch.setattr(data_access, "load_mysql_sales", fail)
    # A different file revision forces an independent cache key.
    import streamlit as st
    st.cache_data.clear()
    app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert "MySQL unavailable" in app.error[0].value
    st.cache_data.clear()
