"""Explicit checks of the synthetic retail data contract."""

import numpy as np
import pandas as pd

REQUIRED = {
    "date", "product_id", "product_name", "store_id", "units_sold",
    "unit_price", "promotion_flag", "stock_on_hand",
    "supplier_lead_time_days",
    "estimated_customer_demand", "customer_demand", "lost_sales",
    "opening_stock", "units_received", "order_quantity",
}


def validate_sales(frame):
    """Raise ValueError for incomplete, duplicate, or invalid observations."""
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing columns: {REQUIRED - set(frame.columns)}")
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Sales data must be nonempty and contain no nulls.")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be valid daily dates.")
    keys = frame[["store_id", "product_id"]].copy()
    keys["date"] = dates
    if keys.duplicated().any():
        raise ValueError("Duplicate date/store/product observations.")
    for column in (
        "units_sold", "unit_price", "stock_on_hand", "supplier_lead_time_days",
        "estimated_customer_demand", "customer_demand", "lost_sales",
        "opening_stock", "units_received", "order_quantity",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        minimum = 0.000001 if column in (
            "unit_price", "supplier_lead_time_days"
        ) else 0
        if not np.isfinite(values).all() or (values < minimum).any():
            raise ValueError(f"Invalid numeric values in {column}.")
        if column not in ("unit_price", "estimated_customer_demand") and not (
            values.eq(np.floor(values)).all()
        ):
            raise ValueError(f"Expected whole numbers in {column}.")
    if not frame["promotion_flag"].isin([0, 1]).all():
        raise ValueError("Promotion flag must be 0 or 1.")
    if not frame["customer_demand"].eq(
        frame["units_sold"] + frame["lost_sales"]
    ).all():
        raise ValueError("Demand must equal units sold plus lost sales.")
    available = frame["opening_stock"] + frame["units_received"]
    if not frame["stock_on_hand"].eq(available - frame["units_sold"]).all():
        raise ValueError("Inventory balance is inconsistent.")
    if not frame["units_sold"].eq(
        np.minimum(available, frame["customer_demand"])
    ).all():
        raise ValueError("Sales must equal demand constrained by stock.")
    ordered = frame.assign(date=dates).sort_values("date")
    for _, group in ordered.groupby(["store_id", "product_id"]):
        if not group["opening_stock"].iloc[1:].reset_index(drop=True).eq(
            group["stock_on_hand"].iloc[:-1].reset_index(drop=True)
        ).all():
            raise ValueError("Opening stock must equal prior closing stock.")
    for _, group in keys.groupby(["store_id", "product_id"]):
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        if len(group) != len(expected):
            raise ValueError("Each product/store series must cover every day.")
