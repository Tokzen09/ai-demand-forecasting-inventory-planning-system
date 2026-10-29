# AI Demand Forecasting & Inventory Planning System

A Python portfolio project applying industrial engineering principles to retail
sales analysis and, in later phases, demand forecasting and replenishment.

**Status: Phase 3B saved-forecast dashboard integration implemented.**
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
for an exploratory dashboard; Pytest for tests. Scikit-learn trains the Phase 3A
regression candidate; Joblib saves artifacts. Git/GitHub are intended for version
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
  forecast.py           # Phase 3A forecasting CLI
  forecasting/          # Sources, features, recursion, evaluation, pipeline
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

1. **Data foundation (completed):** setup, synthetic data, EDA, SQLite, tests,
   and MySQL integration.
2. **Forecasting (implemented in Phase 3A):** chronological splits, baselines,
   causal demand features, regression, MAE/RMSE/WAPE, and saved artifacts.
3. **Inventory planning (future review):** lead-time demand, safety stock, policies,
   service/cost trade-offs and historical simulation.
4. **Product polish:** forecasting dashboard, documentation, CI, and an approved
   GitHub publishing workflow.

Phases 3A and 3B are authorized. Stop after Phase 3B for review. Do not create or
push a GitHub repository without user approval.

## Phase 3A: seven-day customer-demand forecasting

Run from the project folder with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.forecast --source mysql
# Explicit offline source:
.\.venv\Scripts\python.exe -m src.forecast --source csv
# MySQL first, with a visible warning and CSV fallback on connection failure:
.\.venv\Scripts\python.exe -m src.forecast --source auto
.\.venv\Scripts\python.exe -m pytest -q
```

No new package installation was necessary in the validated environment: Joblib
was already installed with scikit-learn and is now a direct requirement.
MySQL access is read-only. The original CSV, schema, importer, SQLite EDA,
`main.py`, and dashboard are preserved. The CLI refuses incomplete 2024–2025
date coverage. Missing targets/dates, negative/nonfinite values, and duplicate
series/date keys are rejected rather than imputed or replaced with sales.
Invalid MySQL data raises an error instead of silently switching datasets.

### Target, features, and promotion assumptions

The target is **customer_demand**, not units_sold. Stockouts censor recorded
sales; demand is known here only because the dataset is simulated. This is not
proof that lost customer demand can be recovered from ordinary company data.
The simulator's `estimated_customer_demand`, same-day units_sold, closing stock,
lost_sales, price, and promotion_flag are excluded from predictors.

Features include store/product one-hot encoding, weekday, weekend, month,
annual sine/cosine, elapsed calendar days, demand lags 1/7/14/28, and demand
means and population standard deviations over the prior 7/28 days. Every lag
and rolling window ends at t-1. The first 28 days per series are warm-up only.
One-hot categories and forest fitting use training rows only.

Although the simulator generated promotions, the dataset does not record an
advance schedule or the time at which each promotion became known. No future
promotions or discounted prices are supplied to this experiment. Forecasts
therefore represent demand with unknown promotional effects, not an invented
no-promotion schedule. Calendar values are known for the forecast horizon.

### Candidates and recursive seven-day evaluation

- **seasonal_naive:** demand on the same weekday one week earlier.
- **moving_average:** mean of the most recent 28 daily demands; subsequent
  horizon steps append and use the baseline's own predictions.
- **random_forest:** a global RandomForestRegressor with 100 trees, depth 12,
  minimum leaf size 8, fixed seed 42, and one worker. Parameters are fixed before
  evaluating the held-out periods; there is no test-based hyperparameter search.

At each forecast origin the forecast function discards all rows from the first
forecast day onward. It generates seven steps, appending only its own predictions
to its private series histories. Actuals inside the current horizon are never
used as lags. The next weekly origin may use observations that have become
available since the previous origin; this simulates weekly operational planning.
Each store/product has a separate history, without cross-series rolling windows.

| Stage | Available fitting labels | Evaluation period |
|---|---|---|
| Validation | 2024-01-01–2025-06-30 | 2025-07-01–2025-09-30 |
| Final test | Refit through 2025-09-30 | 2025-10-01–2025-12-31 |
| Production refit | All observed 2024–2025 data, after test scoring | 2026-01-01–2026-01-07 |

There are 13 non-overlapping weekly origins per held-out period, seven steps per
origin, and 18 series: **1,638 scored rows per candidate per period**. The final
single day of each period (September 30 and December 31) cannot form a complete
seven-day window at the chosen stride and is not scored. It remains available
as historical data when later refits or origins occur.

Selection uses **overall validation WAPE**, then MAE, then method name to break
ties. The selected name is frozen before test scoring. All three candidates are
reported on test for comparison; the winner is not switched based on test results.
The test forest is fitted on train+validation, never test labels. A separate
production artifact is fitted on all data only after scoring is complete.

MAE and RMSE are in daily demand units. WAPE (%) is
`100 * sum(abs(actual - forecast)) / sum(abs(actual))`; it is volume-weighted
across observations, not the average of individual percentage errors. WAPE is
undefined (NaN in CSV / null in JSON) for zero total demand. Forecasts are
nonnegative continuous estimates, not rounded unit counts. Evaluation includes
overall, store, product, store-product, and horizon-day breakdowns.

Measured seed-42 results (not estimated accuracy claims):

| Candidate | Validation WAPE % | Test MAE | Test RMSE | Test WAPE % |
|---|---:|---:|---:|---:|
| Seasonal naive | 35.0375 | 10.7589 | 14.9146 | 35.0623 |
| Moving average | 26.2734 | 8.1285 | 11.3749 | 26.4903 |
| Random forest | 26.1281 | 8.1744 | 11.3014 | 26.6397 |

Random forest was selected by validation WAPE. Moving average's lower test WAPE
is reported honestly; no test-based switch was made. The complete suite passed
51 tests with the live MySQL check enabled and no warnings. Independent MySQL
and CSV full runs produced identical metric and prediction CSVs.

### Generated artifacts and reproducibility

`data/processed/phase3a/` contains:

- `overall_metrics.csv` and `evaluation.csv` (all metric breakdowns).
- `backtest_predictions.csv` (origin, date, horizon, series, method, actual,
  prediction, and split for audit).
- `next_7_days.csv` (126 selected-method forecasts).
- `next_7_days_all_methods.csv` (378 forecasts across all three methods).
- `feature_importance.csv` (forest impurity importances; descriptive, not causal).
- `run_metadata.json` (data fingerprint, dates, seed, source, feature definitions,
  fixed parameters, environment versions, selection, and measured metrics).

`models/phase3a/` contains separate validation, test, and production regressors,
plus `selected_forecaster.joblib` containing the selected method, optional model,
and observed demand history. Both baselines can be selected and persisted.
Load only artifacts produced by this trusted project: Joblib loading can execute
code. Use the recorded dependency versions for model compatibility.

The next-seven-day file is **as of 2025-12-31**, the dataset's last observation.
It is a historical demonstration for January 1–7, 2026, not a forecast for the
current real-world week. Future actuals and promotion plans are unavailable.
Reruns replace these generated artifacts; they do not change raw data. Generated
models and evaluation files remain excluded from Git by the existing rules.

### Limitations and review boundary

There are only two synthetic years and a small validation difference between
the forest and moving-average baseline. The forest's slight validation advantage
does not guarantee better final-test WAPE. The model cannot anticipate unscheduled
promotion spikes or extrapolate long-term trends reliably. No confidence intervals,
inventory-policy changes, or dashboard redesign are included. Only one fixed
forest specification is evaluated, and the final test is a one-time assessment,
not a development tuning loop. Phase 3B now exposes these results in the dashboard.

The chronology follows scikit-learn's [lagged time-series example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
Model parameters follow the [RandomForestRegressor reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

## Phase 3B: saved forecasts in the Streamlit dashboard

The existing historical exploration is preserved. A forecasting section follows
the original sales metrics, chart, product performance, and data preview.

```powershell
# Generate or refresh saved results from either supported source:
.\.venv\Scripts\python.exe -m src.forecast --source mysql
# Or: .\.venv\Scripts\python.exe -m src.forecast --source csv
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The installed Streamlit 1.66.1 supports the chart accessibility and widget APIs
used here; the direct dependency now declares that minimum. No additional
packages or database changes are required.

### Controls and views

- **Data source:** MySQL or CSV, preserving the existing explicit offline mode.
- **Stores / Products:** existing sidebar multiselects also filter forecasts and
  model comparisons. Empty selections show an informative message.
- **Forecast model:** choose seasonal naive, moving average, or random forest.
  The default is the globally selected validation winner stored in metadata;
  choosing another model changes the view, not the selection decision.
- **Historical demand versus forecast:** 56 observed days followed by the actual
  saved seven daily predictions, summed across the selected store-product pairs.
  The forecast line includes one last-observed endpoint as a visual bridge;
  this does not add an eighth forecast day or invent future actual demand.
- **Daily demand forecasts:** dated, store/product-level predictions for the
  selected method, with a CSV download and a seven-day demand total.
- **Model comparison:** switch between validation and final test. All three
  methods display MAE, RMSE, WAPE (%), observation count, and the original global
  validation-selection flag. MAE/RMSE remain daily per-series errors, even when
  the chart sums multiple series; cross-series errors are not cancelled out.

With all series selected, the comparison reads the saved overall evaluation
rows. For a subset, it recomputes the metrics from the selected saved backtest
prediction/actual rows. It never averages store/product WAPEs. Comparisons are
historical backtests, not measured accuracy on the future forecast dates.

The forecast remains a historical demonstration for **January 1–7, 2026, as of
December 31, 2025**, not today's calendar week. Actual future demand, known future
promotion schedules, and uncertainty intervals are not available. All original
synthetic-data disclaimers remain visible. No accuracy numbers are invented.

### Saved artifacts and stale-data handling

The offline pipeline now saves all three methods to
`data/processed/phase3a/next_7_days_all_methods.csv`, using the same history and
cutoff for each. The existing selected-method file is preserved. The dashboard
reads CSV/JSON artifacts only; it never loads Joblib files, trains, or performs
model inference on a widget rerun.

`src/forecasting/dashboard_data.py` validates files, complete seven-day/method
coverage, duplicate keys, finite forecasts, date ranges, and overall evaluation
consistency. A canonical demand-history fingerprint must match the active
MySQL/CSV source. Missing, incomplete, stale, or inconsistent results show an
actionable warning while historical exploration remains usable. Regenerate
artifacts with the pipeline, then click **Refresh data**. Record caches are
bounded to four entries and expire after 60 seconds; revisions of every relevant
artifact also invalidate cached results when files change.

`dashboard/forecast_panel.py` contains presentation and cache logic, separate
from the pure data transformations. Forecasting data and evaluations are not
written back to MySQL; the schema and importer are unchanged.

### Phase 3B tests

```powershell
$env:RUN_MYSQL_READ_TEST = '1'
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_MYSQL_READ_TEST
```

Tests exercise all-method output, missing/corrupt/stale artifacts, metric
consistency, correctly filtered subset metrics, history/future chart separation,
model/store/evaluation-period widget interactions, cache invalidation, and graceful
fallback to historical exploration. Existing forecast leakage/reproducibility,
SQLite, MySQL, and dashboard tests remain in the full suite. Streamlit AppTest
checks behavior without a manual browser visual review. No commits or pushes
are made by these commands. Stop for review after this phase.

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
