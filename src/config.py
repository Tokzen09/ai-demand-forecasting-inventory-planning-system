"""Project paths and reproducible simulation defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_DATA_PATH = RAW_DIR / "synthetic_retail_sales.csv"
DEFAULT_SEED = 42
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"
