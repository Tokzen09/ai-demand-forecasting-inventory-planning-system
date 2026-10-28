"""Generate descriptive tables, charts, and a local SQLite sales table."""

import json
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from src.config import DEFAULT_DATA_PATH, PROCESSED_DIR
from src.quality import validate_sales


def run_eda(data_path=DEFAULT_DATA_PATH, output_dir=PROCESSED_DIR):
    """Save EDA artifacts without fitting a forecasting model."""
    frame = pd.read_csv(data_path, parse_dates=["date"])
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = frame.isna().sum().rename("missing_count")
    missing.to_csv(output_dir / "missing_values.csv")
    validate_sales(frame)
    frame["revenue"] = frame["units_sold"] * frame["unit_price"]
    daily = frame.groupby("date")[[
        "units_sold", "customer_demand", "lost_sales", "revenue"
    ]].sum()
    daily.to_csv(output_dir / "daily_sales.csv")
    performance = frame.groupby(["product_id", "product_name"]).agg(
        total_units=("units_sold", "sum"), revenue=("revenue", "sum")
    ).sort_values("revenue", ascending=False)
    performance.to_csv(output_dir / "product_performance.csv")
    variability = frame.groupby(["store_id", "product_id"]).agg(
        mean_daily_units=("units_sold", "mean"),
        std_daily_units=("units_sold", "std"),
        zero_stock_days=("stock_on_hand", lambda x: int(x.eq(0).sum())),
        total_demand=("customer_demand", "sum"),
        lost_sales=("lost_sales", "sum"),
    )
    variability["coefficient_of_variation"] = (
        variability["std_daily_units"]
        / variability["mean_daily_units"].replace(0, float("nan"))
    )
    variability.to_csv(output_dir / "demand_variability.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    axes[0].plot(
        daily.index, daily["units_sold"], alpha=0.3, label="Daily units"
    )
    axes[0].plot(
        daily.index, daily["units_sold"].rolling(28).mean(),
        label="28-day moving average", color="#087f8c"
    )
    axes[0].set(title="Synthetic retail sales trends", ylabel="Units sold")
    axes[0].legend()
    axes[1].barh(
        performance.index.get_level_values("product_name"),
        performance["total_units"], color="#087f8c",
    )
    axes[1].set(
        title="Product sales performance", xlabel="Units sold", ylabel=""
    )
    fig.savefig(output_dir / "eda_overview.png", dpi=150)
    plt.close(fig)
    px.line(
        daily.reset_index(), x="date", y=["units_sold", "customer_demand"],
        title="Synthetic daily sales and customer demand",
    ).write_html(output_dir / "sales_trend.html", include_plotlyjs=True)
    with sqlite3.connect(output_dir / "retail.db") as connection:
        frame.to_sql("sales", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE UNIQUE INDEX sales_key "
            "ON sales(date, store_id, product_id)"
        )
    summary = {
        "synthetic": True, "rows": len(frame),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "products": int(frame["product_id"].nunique()),
        "stores": int(frame["store_id"].nunique()),
        "total_units": int(frame["units_sold"].sum()),
        "total_customer_demand": int(frame["customer_demand"].sum()),
        "total_lost_sales": int(frame["lost_sales"].sum()),
        "stockout_days": int(frame["lost_sales"].gt(0).sum()),
        "promotion_rate": float(frame["promotion_flag"].mean()),
        "missing_values": int(missing.sum()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
