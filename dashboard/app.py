"""Phase 1 descriptive dashboard; forecasting arrives in Phase 2."""

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_PATH
from src.data_access import load_dashboard_sales
from src.database import DatabaseError, local_settings

st.set_page_config(page_title="Demand & Inventory Planning", layout="wide")
st.title("AI Demand Forecasting & Inventory Planning")
st.caption("Phase 1 · Synthetic data exploration · No trained models")
default_source = local_settings().get("DASHBOARD_DATA_SOURCE", "mysql")
sources = ["mysql", "csv"]
source = st.sidebar.selectbox(
    "Data source", sources,
    index=sources.index(default_source) if default_source in sources else 0,
)


@st.cache_data(ttl=60, max_entries=4)
def cached_sales(data_source, revision):
    """Cache records only; credentials and connections are never cached."""
    return load_dashboard_sales(data_source)


if st.sidebar.button("Refresh data"):
    cached_sales.clear()
revision_file = ROOT / ".env" if source == "mysql" else DEFAULT_DATA_PATH
revision = revision_file.stat().st_mtime_ns if revision_file.exists() else 0
try:
    data = cached_sales(source, revision)
except DatabaseError as error:
    st.error(str(error))
    st.info(
        "Configure .env and run python -m src.import_mysql, then refresh. "
        "Choose csv for offline exploration."
    )
    st.stop()
except (OSError, ValueError):
    st.error("Unable to read the synthetic CSV. Check its file and format.")
    st.stop()
if data.empty:
    st.info("No records found. Run python -m src.import_mysql, then refresh.")
    st.stop()
st.caption(f"Data source: {source} · Cached for up to 60 seconds")
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
    width="stretch",
)
st.subheader("Product performance")
st.dataframe(
    filtered.groupby("product_name").units_sold.agg(["sum", "mean", "std"])
)
st.subheader("Data preview")
st.dataframe(filtered.head(100), width="stretch")
st.caption(
    "Zero closing stock is a stockout-risk signal, not a lost-sales measure."
)
