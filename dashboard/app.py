"""Phase 1 descriptive dashboard; forecasting arrives in Phase 2."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "synthetic_retail_sales.csv"
)
st.set_page_config(page_title="Demand & Inventory Planning", layout="wide")
st.title("AI Demand Forecasting & Inventory Planning")
st.caption("Phase 1 · Synthetic data exploration · No trained models")
if not DATA_PATH.exists():
    st.info("Run python main.py in the project folder to generate data.")
    st.stop()
data = pd.read_csv(DATA_PATH, parse_dates=["date"])
st.warning("All records are simulated. This is not actual company data.")
stores = st.sidebar.multiselect(
    "Stores", sorted(data.store_id.unique()),
    default=sorted(data.store_id.unique()),
)
products = st.sidebar.multiselect(
    "Products", sorted(data.product_name.unique()),
    default=sorted(data.product_name.unique()),
)
filtered = data[data.store_id.isin(stores) & data.product_name.isin(products)]
if filtered.empty:
    st.info("Select at least one store and product.")
    st.stop()
left, middle, right = st.columns(3)
left.metric("Units sold", f"{filtered.units_sold.sum():,}")
middle.metric("Products", filtered.product_id.nunique())
right.metric("Store-product days at zero closing stock", int(
    filtered.stock_on_hand.eq(0).sum()
))
st.metric("Simulated lost sales (units)", f"{filtered.lost_sales.sum():,}")
daily = filtered.groupby("date", as_index=False)[[
    "units_sold", "customer_demand"
]].sum()
st.plotly_chart(
    px.line(daily, x="date", y=["units_sold", "customer_demand"]),
    use_container_width=True,
)
st.subheader("Product performance")
st.dataframe(
    filtered.groupby("product_name").units_sold.agg(["sum", "mean", "std"])
)
st.subheader("Data preview")
st.dataframe(filtered.head(100), use_container_width=True)
st.caption(
    "Zero closing stock is a stockout-risk signal, not a lost-sales measure."
)
