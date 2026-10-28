"""Verify the generated artifacts and SQLite integration."""

import sqlite3

from src.data_generation import generate_sales
from src.eda import run_eda


def test_eda_pipeline(tmp_path):
    path = tmp_path / "input.csv"
    generate_sales(start="2024-01-01", end="2024-02-01").to_csv(
        path, index=False
    )
    output = tmp_path / "reports"
    summary = run_eda(path, output)
    assert summary["rows"] == 32 * 18
    for name in (
        "daily_sales.csv", "missing_values.csv", "product_performance.csv",
        "demand_variability.csv", "eda_overview.png", "sales_trend.html",
        "summary.json", "retail.db",
    ):
        assert (output / name).stat().st_size > 0
    with sqlite3.connect(output / "retail.db") as connection:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM sales"
        ).fetchone()[0]
        assert row_count == 576
    # Re-running must replace derived data without duplicating database rows.
    assert run_eda(path, output) == summary
