"""Validate and upsert synthetic CSV records into the project MySQL DB."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DATA_PATH
from src.database import (
    DatabaseError, connect, ensure_schema, load_mysql_sales, write_sales,
)
from src.quality import validate_sales


def read_import_data(path=DEFAULT_DATA_PATH):
    """Validate before any database writes or DDL are attempted."""
    frame = pd.read_csv(path, parse_dates=["date"])
    validate_sales(frame)
    names = frame.groupby("product_id")["product_name"].nunique()
    if names.gt(1).any():
        raise ValueError("Each product_id must have one product_name.")
    for column in ("product_id", "store_id"):
        if frame[column].str.len().gt(32).any():
            raise ValueError(
                f"{column} exceeds the 32-character schema limit."
            )
    if frame["product_name"].str.len().gt(255).any():
        raise ValueError("product_name exceeds the schema limit.")
    return frame


def verify_import(expected, actual):
    """Compare every imported value; allow existing non-overlapping facts."""
    if actual.empty or not set(expected.columns).issubset(actual.columns):
        raise AssertionError("Database is empty or missing required columns.")
    keys = ["date", "store_id", "product_id"]
    left = expected.set_index(keys).sort_index()
    right = actual.set_index(keys).reindex(left.index)[left.columns]
    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False,
        rtol=0, atol=0.00001,
    )


def import_sales(path=DEFAULT_DATA_PATH):
    frame = read_import_data(path)
    with connect() as connection:
        ensure_schema(connection)
        write_sales(connection, frame)
    verify_import(frame, load_mysql_sales())
    return len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            frame = read_import_data(args.csv)
            verify_import(frame, load_mysql_sales())
            count = len(frame)
        else:
            count = import_sales(args.csv)
    except (DatabaseError, ValueError, OSError, AssertionError) as error:
        # Assertion details may contain data; never print raw exceptions.
        if isinstance(error, DatabaseError):
            parser.exit(1, f"{error}\n")
        parser.exit(1, "Import/verification failed. Check CSV and schema.\n")
    print(f"Verified {count:,} synthetic records in demand_forecasting_db.")


if __name__ == "__main__":
    main()
