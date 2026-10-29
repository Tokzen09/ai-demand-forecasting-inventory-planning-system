"""Read-only dashboard data access with an explicit offline CSV option."""

import pandas as pd

from src.config import DEFAULT_DATA_PATH
from src.database import DatabaseError, load_mysql_sales


def load_dashboard_sales(source):
    if source == "mysql":
        return load_mysql_sales()
    if source == "csv":
        return pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["date"])
    raise DatabaseError("Choose mysql or csv as the dashboard data source.")
