-- Read-only checks for MySQL Workbench after the synthetic import.
USE demand_forecasting_db;

SELECT VERSION() AS mysql_version, DATABASE() AS selected_database;
SHOW TABLES LIKE 'df_%';

SELECT 'products' AS entity, COUNT(*) AS records FROM df_products
UNION ALL SELECT 'stores', COUNT(*) FROM df_stores
UNION ALL SELECT 'daily_sales', COUNT(*) FROM df_daily_sales;
-- Expected on a fresh import: 6 products, 3 stores, 13158 daily sales rows.

SELECT MIN(date) AS first_date, MAX(date) AS last_date,
       SUM(customer_demand) AS customer_demand,
       SUM(units_sold) AS units_sold, SUM(lost_sales) AS lost_sales
FROM df_daily_sales;
-- 2024-01-01, 2025-12-31, 380466, 380002, 464 for seed 42.

SELECT COUNT(*) AS balance_violations
FROM df_daily_sales
WHERE opening_stock + units_received <> units_sold + stock_on_hand
   OR customer_demand <> units_sold + lost_sales
   OR units_sold <> LEAST(customer_demand, opening_stock + units_received);
-- Expected: 0.

SELECT COUNT(*) AS continuity_violations
FROM (
    SELECT opening_stock,
           LAG(stock_on_hand) OVER (
               PARTITION BY store_id, product_id ORDER BY date
           ) AS previous_closing_stock
    FROM df_daily_sales
) AS history
WHERE previous_closing_stock IS NOT NULL
  AND opening_stock <> previous_closing_stock;
-- Expected: 0.

SELECT COUNT(*) AS receipt_timing_violations
FROM df_daily_sales AS today
LEFT JOIN df_daily_sales AS ordered
  ON ordered.product_id = today.product_id
 AND ordered.store_id = today.store_id
 AND ordered.date = DATE_SUB(
     today.date, INTERVAL today.supplier_lead_time_days DAY
 )
WHERE today.units_received <> COALESCE(ordered.order_quantity, 0);
-- Expected: 0 for the original full synthetic horizon.

SELECT s.date, s.store_id, p.product_name, s.customer_demand,
       s.units_sold, s.lost_sales, s.stock_on_hand
FROM df_daily_sales AS s
JOIN df_products AS p ON p.product_id = s.product_id
ORDER BY s.date, s.store_id, s.product_id
LIMIT 20;
