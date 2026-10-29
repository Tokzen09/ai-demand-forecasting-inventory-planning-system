-- MySQL 8.0.19+. Run only inside the existing demand_forecasting_db database.
-- df_ names are reserved for this application's synthetic retail records.
USE demand_forecasting_db;

CREATE TABLE IF NOT EXISTS df_products (
    product_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS df_stores (
    store_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin PRIMARY KEY
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS df_daily_sales (
    date DATE NOT NULL,
    product_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    store_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    units_sold INT UNSIGNED NOT NULL,
    unit_price DECIMAL(12, 2) NOT NULL,
    promotion_flag TINYINT UNSIGNED NOT NULL,
    stock_on_hand INT UNSIGNED NOT NULL,
    supplier_lead_time_days SMALLINT UNSIGNED NOT NULL,
    estimated_customer_demand DECIMAL(14, 4) NOT NULL,
    customer_demand INT UNSIGNED NOT NULL,
    lost_sales INT UNSIGNED NOT NULL,
    opening_stock INT UNSIGNED NOT NULL,
    units_received INT UNSIGNED NOT NULL,
    order_quantity INT UNSIGNED NOT NULL,
    PRIMARY KEY (date, store_id, product_id),
    INDEX idx_df_product_date (product_id, date),
    INDEX idx_df_store_date (store_id, date),
    CONSTRAINT fk_df_product FOREIGN KEY (product_id)
        REFERENCES df_products (product_id),
    CONSTRAINT fk_df_store FOREIGN KEY (store_id)
        REFERENCES df_stores (store_id),
    CONSTRAINT ck_df_price CHECK (unit_price > 0),
    CONSTRAINT ck_df_promotion CHECK (promotion_flag IN (0, 1)),
    CONSTRAINT ck_df_lead CHECK (supplier_lead_time_days > 0),
    CONSTRAINT ck_df_expected_demand CHECK (estimated_customer_demand >= 0),
    CONSTRAINT ck_df_demand CHECK (customer_demand = units_sold + lost_sales),
    CONSTRAINT ck_df_inventory CHECK (
        opening_stock + units_received = units_sold + stock_on_hand
    ),
    CONSTRAINT ck_df_fulfillment CHECK (
        units_sold = LEAST(customer_demand, opening_stock + units_received)
    )
) ENGINE=InnoDB;
