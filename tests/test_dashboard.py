"""Exercise the exploratory dashboard without a running browser server."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "data/raw/synthetic_retail_sales.csv").exists(),
    reason="Generate the default dataset before the dashboard smoke test.",
)
def test_dashboard_filters():
    app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[0].label == "Units sold"
    app.sidebar.multiselect[0].set_value([]).run(timeout=30)
    assert not app.exception
    assert app.info[0].value == "Select at least one store and product."
