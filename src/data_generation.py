"""Generate synthetic sales with inventory-constrained observations."""

import argparse

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_END,
    DEFAULT_SEED,
    DEFAULT_START,
)

# SKU, description, price (illustrative currency units), daily demand,
# supplier lead time, seasonal amplitude, seasonal peak day of year.
PRODUCTS = (
    ("P001", "Wholegrain Bread", 24.99, 32, 2, 0.08, 180),
    ("P002", "Fresh Milk 1L", 18.99, 42, 3, 0.06, 180),
    ("P003", "Ground Coffee 250g", 79.99, 14, 7, 0.20, 180),
    ("P004", "Bottled Water 1.5L", 14.99, 30, 5, 0.35, 15),
    ("P005", "Breakfast Cereal 500g", 49.99, 18, 7, 0.12, 30),
    ("P006", "Laundry Detergent 2L", 99.99, 10, 10, 0.10, 330),
)
STORES = {"S001": 1.0, "S002": 0.75, "S003": 1.3}


def generate_sales(
    seed: int = DEFAULT_SEED,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Return deterministic daily records; stock is measured at day end.

    Replenishment uses an illustrative reorder-point policy. Sales are capped
    by available stock. Demand and lost sales are synthetic ground truth;
    typical real-world sales datasets do not directly observe these values.
    This is a simulation, not an estimated demand model or a production policy.
    """
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError("End date must be on or after start date.")
    rng = np.random.default_rng(seed)
    records = []
    for store_id, store_scale in STORES.items():
        for sku, name, price, base, lead, amplitude, peak in PRODUCTS:
            average = base * store_scale
            stock = int(average * (lead + 7))
            orders = []
            for day_index, date in enumerate(dates):
                opening_stock = stock
                receipts = sum(qty for due, qty in orders if due == day_index)
                stock += receipts
                orders = [(due, qty) for due, qty in orders if due > day_index]
                promotion = int(rng.random() < 0.09)
                weekend = 1.22 if date.dayofweek >= 5 else 1.0
                seasonal = 1 + amplitude * np.cos(
                    2 * np.pi * (date.dayofyear - peak) / 365.25
                )
                trend = 1 + 0.06 * day_index / 365.25
                expected = average * weekend * seasonal * trend
                expected *= 1.45 if promotion else 1.0
                # Gamma-Poisson variation represents noisy retail demand.
                demand = int(rng.poisson(rng.gamma(20, expected / 20)))
                sold = min(stock, demand)
                stock -= sold
                inventory_position = stock + sum(qty for _, qty in orders)
                reorder_point = average * (lead + 3) * trend
                quantity = 0
                if inventory_position <= reorder_point:
                    target = average * (lead + 10) * trend
                    quantity = max(
                        1, int(np.ceil(target - inventory_position))
                    )
                    orders.append((day_index + lead, quantity))
                records.append(
                    (date, sku, name, store_id, sold,
                     round(price * (0.85 if promotion else 1), 2),
                     promotion, stock, lead, round(expected, 4), demand,
                     demand - sold, opening_stock, receipts, quantity)
                )
    return pd.DataFrame(
        records,
        columns=[
            "date", "product_id", "product_name", "store_id",
            "units_sold", "unit_price", "promotion_flag", "stock_on_hand",
            "supplier_lead_time_days",
            "estimated_customer_demand", "customer_demand", "lost_sales",
            "opening_stock", "units_received", "order_quantity",
        ],
    ).sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)


def main():
    """Write the reproducible default synthetic dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_sales(seed=args.seed)
    frame.to_csv(DEFAULT_DATA_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(frame):,} synthetic rows to {DEFAULT_DATA_PATH}")


if __name__ == "__main__":
    main()
