# AI Demand Forecasting & Inventory Planning System

A Python portfolio project applying industrial engineering principles to retail
sales analysis and, in later phases, demand forecasting and replenishment.

**Status: Phase 1 implemented. No machine learning models have been trained.**
**All data is synthetic. No actual company or confidential data is used.**

## Business problem and objectives

Retailers face a trade-off between stock availability and the cost of excess
inventory. Sales records alone can understate demand when stock is unavailable.
This project establishes a tested data foundation to investigate that problem.
Future work will forecast demand, measure forecast error, and compare inventory
policies using service level, lost units, inventory holding, and stockout metrics.
Those benefits are objectives, not results demonstrated by Phase 1.

## Technology

Python 3.11+; Pandas and NumPy for processing and simulation; Matplotlib and
Plotly for charts; SQLite (Python standard library) for local storage; Streamlit
for an exploratory dashboard; Pytest for tests. Scikit-learn is installed for
Phase 2 but is not used for model training. Git/GitHub are intended for version
control; no remote repository is created or pushed automatically.

MySQL Connector/Python and python-dotenv provide local MySQL integration.
SQLite remains available as the original EDA export.

## Structure

```text
data/
  raw/                  # Reproducible synthetic CSV and data dictionary
  processed/            # Derived summaries, charts, and SQLite database
src/
  config.py             # Portable project paths and defaults
  data_generation.py    # Seeded demand and replenishment simulation
  quality.py            # Dataset contract validation
  eda.py                # Descriptive reports and database export
  database.py           # Scoped MySQL connection, schema, and SQL operations
  import_mysql.py       # Validated repeatable MySQL import and verification
  data_access.py        # Dashboard MySQL/CSV read adapter
sql/                    # MySQL schema and Workbench verification queries
.env.example            # Connection template; real .env is ignored
models/                 # Reserved for Phase 2; intentionally empty
dashboard/app.py        # Phase 1 Streamlit explorer
tests/                  # Data, inventory, reproducibility, EDA tests
main.py                 # Rebuild data and reports
requirements.txt        # Direct dependency ranges
requirements-lock.txt   # Exact installed versions, when installation succeeds
pytest.ini
.gitignore
README.md
outputs/                # Handoff report and source archive (ignored by Git)
work/                   # Local scratch/temp files (ignored by Git)
```

## Setup and execution (Windows PowerShell)

Run from this project folder with a working Python 3.11+ installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Activation is optional; explicitly invoking the environment avoids changing PATH
or PowerShell execution policy. On macOS/Linux use `.venv/bin/python`.
Use `-r requirements-lock.txt` instead of `-r requirements.txt` to reproduce the
validated Windows/Python 3.12 environment once the lock file is available.
Locks may contain platform-specific dependencies; use the ranges on other systems.

If Windows blocks its default temporary directory, use a project-local one:

```powershell
New-Item -ItemType Directory -Force work/tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) 'work/tmp'
$env:TMP = $env:TEMP
```

These variables apply only to the current shell. Generated data/reports are
replaced deterministically by `main.py`; it does not modify source files.
Standalone steps:

```powershell
.\.venv\Scripts\python.exe -m src.data_generation --seed 42
.\.venv\Scripts\python.exe -m src.eda
```

## Synthetic dataset and assumptions

Default seed: **42**. Coverage: **2024-01-01 to 2025-12-31**, including leap day;
731 days × six products × three stores = **13,158 rows**. Each row represents one
store-product-day. No network access is needed to generate the data.

Demand combines store size, product baseline, annual seasonality, 22% weekend
uplift, 45% promotional uplift, 6% annual trend, and Gamma-Poisson noise.
Promotions occur independently with probability 9% and reduce price by 15%.
Seasonality is illustrative of Southern Hemisphere retail (summer water demand),
not calibrated to a company. Prices use unspecified illustrative currency units.

Opening stock is the previous day's closing stock. Supplier deliveries arrive
before demand on the due date. A reorder-point rule uses inventory position
(closing stock plus outstanding orders) and places orders for fixed product lead
times of 2–10 days. The first day starts with a known stock allocation and no
outstanding orders. Orders due after the dataset ends remain outside its horizon.

```text
available_stock = opening_stock + units_received
units_sold = min(customer_demand, available_stock)
stock_on_hand = available_stock - units_sold
lost_sales = customer_demand - units_sold
```

`estimated_customer_demand` is the simulator's expected mean before random noise;
`customer_demand` is the sampled customer request quantity; `units_sold` is the
quantity fulfilled. The first two are simulation diagnostics, not learned model
predictions. Lost sales are assumed permanently lost, not backordered. Zero
closing stock alone does not prove lost sales; `lost_sales > 0` does.

See `data/raw/DATA_DICTIONARY.md` for every column. Simulated demand and lost
sales must not be treated as observed business data or future model inputs.
Future feature engineering must avoid target leakage, including same-day closing
stock and same-day outcomes. Promotions are assumed known before the selling day.

The simulation omits expiry, returns, capacity limits, supplier disruptions,
substitution, inter-store transfers, and correlated promotions. Bread and milk
are illustrative SKUs; their expiry and waste are not modeled.

## EDA and SQL

`main.py` creates missing-value counts, daily trends, product volume/revenue,
store-SKU variability (standard deviation and coefficient of variation), lost
sales, an overview PNG, standalone interactive Plotly HTML, and summary JSON.
Variability describes observed sales and is affected by inventory constraints.
SQLite output is `data/processed/retail.db`, with a `sales` table and a unique
date/store/product index. Dates are stored as timestamp text. Example SQL:

```sql
SELECT product_id, SUM(units_sold) AS sold, SUM(lost_sales) AS lost
FROM sales
GROUP BY product_id
ORDER BY sold DESC;
```

## Validation

Tests cover complete daily coverage, missing values, valid quantities, unique
keys, seeded reproducibility, different seeds, promotion/weekend/seasonality
signals, inventory continuity, receipts matching orders at their lead times,
demand/sales/lost-sales balance, rejection of bad data, EDA output, and repeatable
SQLite export. These are simulation and pipeline tests, not proof of commercial
forecast accuracy. The exploratory Streamlit app is available but browser checks
must be reported separately from automated tests.

## Planned phases

1. **Data foundation (current):** setup, synthetic data, EDA, SQLite, tests.
2. **Forecasting (requires approval):** temporal train/validation/test splits,
   naive and seasonal baselines, lagged features, scikit-learn models, MAE/RMSE
   and bias evaluation. No random time-series split.
3. **Inventory planning:** lead-time demand, safety stock, reorder policies,
   service/cost trade-offs and historical simulation.
4. **Product polish:** forecasting dashboard, documentation, CI, and an approved
   GitHub publishing workflow.

Do not start Phase 2 or create/push a GitHub repository without user approval.

## MySQL integration

Use the existing **demand_forecasting_db** on **localhost:3306**. The importer
does not create databases, drop tables, truncate records, or access another
database. MySQL **8.0.19+** is required (enforced CHECK constraints and row-alias
upserts). Tables prefixed `df_` are owned by this project; do not reuse those
names for unrelated data. Existing tables are not altered by this script;
incompatible existing definitions must be reviewed rather than overwritten.

### Configure the connection

Install dependencies in the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

A local `.env` has been prepared with a blank password. Edit it in VS Code;
do not paste credentials into a terminal command, source file, or chat. For a
new clone, copy `.env.example` to `.env` only if `.env` does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
code .env
```

Set `MYSQL_HOST=localhost`, `MYSQL_PORT=3306`,
`MYSQL_DATABASE=demand_forecasting_db`, and your local MySQL user/password.
The sample user is `root`; substitute your own account. Prefer a dedicated
account whose permissions are restricted to this database. Schema creation
needs CREATE, REFERENCES, SELECT, INSERT, and UPDATE permissions; the dashboard
only reads and can use a SELECT-only account after import. No user accounts or
grants are created automatically.

Use `MYSQL_PASSWORD='your local password'` when it contains `#`, spaces, or `$`.
The parser disables variable interpolation to preserve literal dollar signs.
Do not use a blank password. Real process environment variables override `.env`.
The credentials are neither logged nor cached by the dashboard.
The `.gitignore` excludes `.env` and permits the password-free `.env.example`.

### Schema and repeatable import

```powershell
.\.venv\Scripts\python.exe -m src.import_mysql
# Run again: existing natural keys are updated, not duplicated.
.\.venv\Scripts\python.exe -m src.import_mysql
# Compare every original CSV value with the database without writing:
.\.venv\Scripts\python.exe -m src.import_mysql --verify-only
```

The importer reads the existing CSV without regenerating or changing it. It
validates daily coverage, keys, quantities, inventory continuity, fulfillment,
and demand/lost-sales balance before connecting. One name is required per SKU.
Products, stores, and facts are then upserted in a single InnoDB transaction;
an insertion failure rolls back all data writes in that import. MySQL schema
DDL commits separately, so empty created tables can remain if data loading fails.
Post-commit verification reports any round-trip mismatch; it does not undo an
already committed transaction. Commands exit nonzero on failure with sanitized
messages. Re-running an import updates matching keys; it does not remove rows
absent from the CSV. Do not use this utility for unrelated business data.

| Table | Role and key |
|---|---|
| df_products | Product ID primary key; product name |
| df_stores | Store ID primary key |
| df_daily_sales | Daily observations; primary key `(date, store_id, product_id)` |

Facts reference both dimension tables. Prices and expected demand use DECIMAL;
dates use DATE; nonnegative quantities use unsigned integers. Supplier lead time
and price are retained per day to preserve historical observations. SQL CHECK
constraints enforce row-level inventory/demand balance. Cross-day continuity
is validated in Python and by the Workbench query, not by a row-level CHECK.

### Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

MySQL is the default source. The existing store/product filters, metrics, sales
and demand chart, product performance, and preview are preserved. Choose `csv`
in the sidebar for explicit offline exploration. MySQL errors are displayed
without raw credentials or connector diagnostics; there is no silent fallback.
Use **Refresh data** after import or allow the 60-second record cache to expire.
The dashboard does not create tables or import records. SQLite EDA outputs and
`main.py` remain unchanged; neither is required for the MySQL import.

### Verify in MySQL Workbench

Connect to your local server and open `sql/verify_mysql.sql`. Execute it using
the lightning button (or Ctrl+Shift+Enter for the script). On the original seed-42
dataset expect 6 products, 3 stores, 13,158 facts, 380,466 demand units, 380,002
sold units, and 464 lost units. Balance, continuity, and receipt timing violation
counts should all be zero. Execute the import twice and rerun the counts to
confirm repeatability. The script is read-only and selects the project database.

### Tests and troubleshooting

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Automated tests cover local configuration, database scoping, secret redaction,
parameter binding, rollback, round-trip verification, and MySQL/CSV dashboard
behavior using mocks. They do not require credentials or modify a real database.
Actual MySQL validation is provided by the import and `--verify-only` commands;
mock tests alone do not prove live server compatibility.

After a successful import, opt into the real read-only round-trip test:

```powershell
$env:RUN_MYSQL_READ_TEST = '1'
.\.venv\Scripts\python.exe -m pytest tests/test_mysql_live.py -q
Remove-Item Env:RUN_MYSQL_READ_TEST
```

If prompted to configure credentials, edit `.env`. Connector error 1045 usually
means authentication or grants; 1049 means the database does not exist; 2003 means
the server/port is unreachable; 1146 means import has not created the tables.
Check Workbench for detailed diagnostics locally without sharing passwords.
If MySQL is older than 8.0.19, update the local server before importing.

The upsert behavior follows the official [MySQL INSERT documentation](https://dev.mysql.com/doc/refman/8.0/en/insert.html),
and transaction handling follows [Connector/Python commit](https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlconnection-commit.html).
