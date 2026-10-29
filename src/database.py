"""MySQL access scoped to the existing demand_forecasting_db database."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import os
from pathlib import Path
import re

from dotenv import dotenv_values
import mysql.connector
import pandas as pd

from src.config import ROOT

DATABASE_NAME = "demand_forecasting_db"
SCHEMA_PATH = ROOT / "sql" / "mysql_schema.sql"
SALES_COLUMNS = [
    "date", "product_id", "store_id", "units_sold", "unit_price",
    "promotion_flag", "stock_on_hand", "supplier_lead_time_days",
    "estimated_customer_demand", "customer_demand", "lost_sales",
    "opening_stock", "units_received", "order_quantity",
]
READ_SALES = """
SELECT s.date, s.product_id, p.product_name, s.store_id, s.units_sold,
       s.unit_price, s.promotion_flag, s.stock_on_hand,
       s.supplier_lead_time_days, s.estimated_customer_demand,
       s.customer_demand, s.lost_sales, s.opening_stock,
       s.units_received, s.order_quantity
FROM df_daily_sales AS s
JOIN df_products AS p ON p.product_id = s.product_id
ORDER BY s.date, s.store_id, s.product_id
"""


class DatabaseError(RuntimeError):
    """Sanitized database error suitable for CLI and dashboard display."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Credentials are omitted from repr and never logged."""

    host: str
    port: int
    database: str
    user: str = field(repr=False)
    password: str = field(repr=False)


def local_settings(env_path: Path = ROOT / ".env") -> dict:
    """Read local .env without mutating process environment; env wins."""
    return {**dotenv_values(env_path, interpolate=False), **os.environ}


def database_settings(env_path: Path = ROOT / ".env") -> DatabaseSettings:
    values = local_settings(env_path)
    if values.get("MYSQL_DATABASE", DATABASE_NAME) != DATABASE_NAME:
        raise DatabaseError(f"Only {DATABASE_NAME} is supported.")
    user = values.get("MYSQL_USER", "")
    password = values.get("MYSQL_PASSWORD", "")
    if not user or not password:
        raise DatabaseError(
            "Configure MYSQL_USER and MYSQL_PASSWORD in the local .env file."
        )
    try:
        port = int(values.get("MYSQL_PORT", "3306"))
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError) as error:
        raise DatabaseError(
            "MYSQL_PORT must be between 1 and 65535."
        ) from error
    return DatabaseSettings(
        values.get("MYSQL_HOST", "localhost"), port, DATABASE_NAME,
        user, password,
    )


@contextmanager
def connect(settings=None):
    """Open one scoped connection; never share a connection across sessions."""
    settings = settings or database_settings()
    if settings.database != DATABASE_NAME:
        raise DatabaseError(f"Only {DATABASE_NAME} is supported.")
    connection = None
    try:
        connection = mysql.connector.connect(
            host=settings.host, port=settings.port,
            database=settings.database, user=settings.user,
            password=settings.password, connection_timeout=5,
            autocommit=False, charset="utf8mb4", use_unicode=True,
        )
        yield connection
    except mysql.connector.Error as error:
        # Do not expose connector messages, SQL values, or credentials.
        raise DatabaseError(
            f"MySQL operation failed (code {error.errno}). Check connection, "
            "credentials, table schema, and account permissions."
        ) from None
    finally:
        if connection is not None:
            connection.close()


def ensure_schema(connection):
    """Create only application tables; DDL commits independently in MySQL."""
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT DATABASE(), VERSION()")
        database, version = cursor.fetchone()
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
        if database != DATABASE_NAME:
            raise DatabaseError("Refusing schema changes outside project DB.")
        if (
            not match or "mariadb" in version.lower()
            or tuple(map(int, match.groups())) < (8, 0, 19)
        ):
            raise DatabaseError("MySQL 8.0.19 or newer is required.")
        source = SCHEMA_PATH.read_text(encoding="utf-8")
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("--")
        )
        for statement in source.split(";"):
            if statement.strip():
                cursor.execute(statement)
        connection.commit()
    finally:
        cursor.close()


def upsert_statement(table, columns, key_columns):
    """Build SQL from module-owned identifiers, never user input."""
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{column}=incoming.{column}" for column in columns
        if column not in key_columns
    )
    if not updates:
        updates = f"{columns[0]}=incoming.{columns[0]}"
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders}) AS incoming "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )


def write_sales(connection, frame):
    """Atomically upsert dimensions and facts without deleting rows."""
    cursor = connection.cursor()
    try:
        connection.start_transaction()
        products = frame[["product_id", "product_name"]].drop_duplicates()
        cursor.executemany(
            upsert_statement(
                "df_products", ["product_id", "product_name"], {"product_id"}
            ), list(products.itertuples(index=False, name=None)),
        )
        cursor.executemany(
            upsert_statement("df_stores", ["store_id"], {"store_id"}),
            [(value,) for value in sorted(frame["store_id"].unique())],
        )
        rows = []
        for values in frame[SALES_COLUMNS].itertuples(index=False, name=None):
            row = list(values)
            row[0] = pd.Timestamp(row[0]).date()
            row[4] = Decimal(str(row[4]))
            row[8] = Decimal(str(row[8]))
            rows.append(tuple(row))
        sql = upsert_statement(
            "df_daily_sales", SALES_COLUMNS, {"date", "store_id", "product_id"}
        )
        for offset in range(0, len(rows), 1000):
            cursor.executemany(sql, rows[offset:offset + 1000])
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def load_mysql_sales(settings=None):
    """Return the original CSV-shaped dataset, keeping chart code unchanged."""
    with connect(settings) as connection:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(READ_SALES)
            frame = pd.DataFrame(cursor.fetchall())
        finally:
            cursor.close()
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("unit_price", "estimated_customer_demand"):
            frame[column] = frame[column].astype(float)
    return frame
