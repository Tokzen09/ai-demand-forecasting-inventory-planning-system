"""Explicitly opt-in read-only round-trip test against configured MySQL."""

import os

import pytest

from src.database import load_mysql_sales
from src.import_mysql import read_import_data, verify_import


@pytest.mark.skipif(
    os.environ.get("RUN_MYSQL_READ_TEST") != "1",
    reason="Set RUN_MYSQL_READ_TEST=1 after configuring and importing MySQL.",
)
def test_live_mysql_round_trip():
    """Compare imported database records to the CSV without DDL or writes."""
    verify_import(read_import_data(), load_mysql_sales())
