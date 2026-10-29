"""Read-only forecast sources and a strict daily target contract."""

import hashlib
import warnings

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_PATH
from src.database import DatabaseError, load_mysql_sales

KEYS = ["date", "store_id", "product_id"]
TARGET = "customer_demand"


def validate_demand(frame):
    """Reject gaps and missing targets; never replace demand with sales."""
    required = KEYS + [TARGET]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(
            "Nonempty date/store/product/customer_demand data is required."
        )
    result = frame[required].copy()
    if result.isna().any().any():
        raise ValueError("Forecast keys and demand cannot be missing.")
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Forecast observations must be daily dates.")
    result[TARGET] = pd.to_numeric(result[TARGET], errors="raise")
    if not np.isfinite(result[TARGET]).all() or result[TARGET].lt(0).any():
        raise ValueError("Customer demand must be finite and nonnegative.")
    if result.duplicated(KEYS).any():
        raise ValueError("Duplicate store/product/date keys.")
    expected = pd.date_range(result.date.min(), result.date.max())
    for _, group in result.groupby(["store_id", "product_id"]):
        if not group.date.sort_values().reset_index(drop=True).equals(
            pd.Series(expected, name="date")
        ):
            raise ValueError(
                "Every series must have complete shared daily coverage."
            )
    return result.sort_values(KEYS).reset_index(drop=True)


def load_demand(source="auto", csv_path=DEFAULT_DATA_PATH):
    """Auto falls back on connection errors, not invalid MySQL records."""
    if source not in {"auto", "mysql", "csv"}:
        raise ValueError("Source must be auto, mysql, or csv.")
    if source in {"auto", "mysql"}:
        try:
            frame = load_mysql_sales()
        except DatabaseError:
            if source == "mysql":
                raise
            warnings.warn(
                "MySQL unavailable; using the existing synthetic CSV.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            return validate_demand(frame), "mysql"
    return validate_demand(pd.read_csv(csv_path)), "csv"


def fingerprint(frame):
    """Canonical target/key hash, identical across CSV and MySQL sources."""
    content = validate_demand(frame).to_csv(
        index=False, date_format="%Y-%m-%d", float_format="%.8f"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
