# Synthetic retail data dictionary

**Simulation only. Not actual company data.** Grain: one date/product/store row.

| Column | Type / meaning |
|---|---|
| date | ISO calendar date |
| product_id | Stable synthetic SKU P001–P006 |
| product_name | Synthetic product description |
| store_id | Synthetic store S001–S003 |
| units_sold | Nonnegative integer fulfilled customer units |
| unit_price | Positive price per unit; includes promotion discount; unspecified currency |
| promotion_flag | 1 if promoted, otherwise 0 |
| stock_on_hand | Nonnegative integer closing stock after sales |
| supplier_lead_time_days | Positive fixed calendar-day lead time per SKU |
| estimated_customer_demand | Nonnegative simulator mean before random noise; not an ML prediction |
| customer_demand | Nonnegative integer sampled demand before stock constraints |
| lost_sales | Nonnegative integer demand not fulfilled; no backorders |
| opening_stock | Nonnegative integer prior-day closing stock, before receipts |
| units_received | Nonnegative integer supplier receipts before today's demand |
| order_quantity | Nonnegative integer ordered after today's sales, due after lead time |

Demand fields and lost sales are known only because the data is simulated.
They are retained to demonstrate sales censoring and test inventory logic.
There are no missing observations; zero sales represent a recorded zero.
