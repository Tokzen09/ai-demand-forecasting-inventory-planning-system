"""Offline importer tests; no local database credentials are required."""

from dataclasses import replace
from unittest.mock import MagicMock

import mysql.connector
import pandas as pd
import pytest

from src import database
from src.data_generation import generate_sales
from src.import_mysql import read_import_data, verify_import


@pytest.fixture
def small_frame():
    return generate_sales(start="2024-01-01", end="2024-01-03")


def test_settings_private_and_scoped(tmp_path, monkeypatch):
    for key in (
        "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_PORT"
    ):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text("MYSQL_USER=local\nMYSQL_PASSWORD='a#${literal}'\n")
    settings = database.database_settings(path)
    assert settings.password == "a#${literal}"
    assert settings.password not in repr(settings)
    monkeypatch.setenv("MYSQL_DATABASE", "unrelated_database")
    with pytest.raises(database.DatabaseError, match="Only"):
        database.database_settings(path)


def test_missing_password(tmp_path, monkeypatch):
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)
    path = tmp_path / ".env"
    path.write_text("MYSQL_USER=local\nMYSQL_PASSWORD=\n")
    with pytest.raises(database.DatabaseError, match="Configure"):
        database.database_settings(path)


def test_connection_rejects_other_database():
    settings = database.DatabaseSettings(
        "localhost", 3306, database.DATABASE_NAME, "test", "secret"
    )
    with pytest.raises(database.DatabaseError, match="Only"):
        with database.connect(replace(settings, database="unrelated")):
            pytest.fail("Must not connect")


def test_connection_error_is_sanitized(monkeypatch):
    settings = database.DatabaseSettings(
        "localhost", 3306, database.DATABASE_NAME, "test", "hidden"
    )
    mock = MagicMock(side_effect=mysql.connector.Error("hidden", errno=1045))
    monkeypatch.setattr(database.mysql.connector, "connect", mock)
    with pytest.raises(database.DatabaseError) as caught:
        with database.connect(settings):
            pass
    assert "1045" in str(caught.value)
    assert "hidden" not in str(caught.value)


def test_import_validation_before_writes(tmp_path, small_frame):
    small_frame.loc[0, "lost_sales"] += 1
    path = tmp_path / "bad.csv"
    small_frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_import_data(path)


def test_import_rejects_conflicting_product_names(tmp_path, small_frame):
    small_frame.loc[0, "product_name"] = "Conflicting name"
    path = tmp_path / "bad.csv"
    small_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one product_name"):
        read_import_data(path)


def test_atomic_write_and_parameter_binding(small_frame):
    connection = MagicMock()
    database.write_sales(connection, small_frame)
    connection.start_transaction.assert_called_once()
    connection.commit.assert_called_once()
    connection.rollback.assert_not_called()
    calls = connection.cursor.return_value.executemany.call_args_list
    sql, rows = calls[2].args
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "%s" in sql
    assert len(rows) == len(small_frame)
    assert rows[0][0].isoformat() == "2024-01-01"
    connection.cursor.return_value.close.assert_called_once()


def test_write_rollback(small_frame):
    connection = MagicMock()
    connection.cursor.return_value.executemany.side_effect = (
        mysql.connector.Error("simulated failure")
    )
    with pytest.raises(mysql.connector.Error):
        database.write_sales(connection, small_frame)
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()


def test_verification_detects_missing_and_changed_rows(small_frame):
    verify_import(small_frame, small_frame.copy())
    with pytest.raises(AssertionError):
        verify_import(small_frame, small_frame.iloc[1:])
    changed = small_frame.copy()
    changed.loc[0, "unit_price"] += 1
    with pytest.raises(AssertionError):
        verify_import(small_frame, changed)
    with pytest.raises(AssertionError, match="empty"):
        verify_import(small_frame, pd.DataFrame())


def test_schema_refuses_wrong_database():
    connection = MagicMock()
    connection.cursor.return_value.fetchone.return_value = ("other", "8.0.40")
    with pytest.raises(database.DatabaseError, match="Refusing"):
        database.ensure_schema(connection)
    assert connection.cursor.return_value.execute.call_count == 1


def test_read_restores_dashboard_types(monkeypatch):
    connection = MagicMock()
    connection.cursor.return_value.fetchall.return_value = [
        {"date": "2024-01-01", "unit_price": "24.99",
         "estimated_customer_demand": "29.4450"}
    ]
    context = MagicMock()
    context.__enter__.return_value = connection
    monkeypatch.setattr(database, "connect", lambda _: context)
    frame = database.load_mysql_sales()
    assert pd.api.types.is_datetime64_any_dtype(frame.date)
    assert frame.unit_price.iloc[0] == 24.99
