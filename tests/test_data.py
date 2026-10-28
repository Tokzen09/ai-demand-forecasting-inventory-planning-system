"""Data quality, simulation behavior, and reproducibility tests."""

import pandas as pd
import pytest

from src.data_generation import PRODUCTS, generate_sales
from src.quality import validate_sales


@pytest.fixture(scope="module")
def sales():
    return generate_sales()


def test_quality(sales):
    validate_sales(sales)


def test_coverage(sales):
    assert len(sales) == 731 * 6 * 3
    assert sales.date.nunique() == 731
    assert sales.product_id.nunique() == 6
    assert sales.store_id.nunique() == 3


def test_reproducibility(sales):
    pd.testing.assert_frame_equal(sales, generate_sales(seed=42))


def test_different_seeds(sales):
    assert not sales.equals(generate_sales(seed=43))


def test_promotion_and_weekend_patterns(sales):
    means = sales.groupby("promotion_flag").units_sold.mean()
    assert means.loc[1] > means.loc[0] * 1.15
    weekend = sales.date.dt.dayofweek >= 5
    assert sales.loc[weekend, "units_sold"].mean() > sales.loc[
        ~weekend, "units_sold"
    ].mean()


def test_water_seasonality(sales):
    water = sales[sales.product_id.eq("P004")]
    summer = water[water.date.dt.month.isin([12, 1, 2])].units_sold.mean()
    winter = water[water.date.dt.month.isin([6, 7, 8])].units_sold.mean()
    assert summer > winter * 1.2


def test_prices_and_supplier_lead_times(sales):
    for sku, _, price, _, lead, _, _ in PRODUCTS:
        product = sales[sales.product_id.eq(sku)]
        assert product.supplier_lead_time_days.eq(lead).all()
        assert product.loc[
            product.promotion_flag.eq(0), "unit_price"
        ].eq(price).all()
        assert product.loc[
            product.promotion_flag.eq(1), "unit_price"
        ].eq(round(price * 0.85, 2)).all()


@pytest.mark.parametrize("kind", ["null", "duplicate", "negative", "gap"])
def test_rejects_bad_data(sales, kind):
    bad = sales.copy()
    if kind == "null":
        bad.loc[0, "product_name"] = None
    elif kind == "duplicate":
        bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
    elif kind == "negative":
        bad.loc[0, "units_sold"] = -1
    else:
        bad = bad.drop(index=100)
    with pytest.raises(ValueError):
        validate_sales(bad)


def test_invalid_date_range():
    with pytest.raises(ValueError):
        generate_sales(start="2025-01-02", end="2025-01-01")


def test_inventory_and_demand_balance(sales):
    available = sales.opening_stock + sales.units_received
    assert (sales.units_sold <= available).all()
    assert sales.stock_on_hand.eq(available - sales.units_sold).all()
    assert sales.customer_demand.eq(sales.units_sold + sales.lost_sales).all()
    assert sales.lost_sales.gt(0).any()
    for _, group in sales.groupby(["store_id", "product_id"]):
        assert group.opening_stock.iloc[1:].tolist() == (
            group.stock_on_hand.iloc[:-1].tolist()
        )
        lead = int(group.supplier_lead_time_days.iloc[0])
        expected_receipts = group.order_quantity.shift(lead, fill_value=0)
        assert group.units_received.tolist() == expected_receipts.tolist()


@pytest.mark.parametrize(
    "column", ["lost_sales", "stock_on_hand", "opening_stock"]
)
def test_rejects_inconsistent_balance(sales, column):
    bad = sales.copy()
    bad.loc[0, column] += 1
    with pytest.raises(ValueError):
        validate_sales(bad)
