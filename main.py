"""Phase 1 entry point: generate data and perform exploratory analysis."""

from src.config import DEFAULT_DATA_PATH
from src.data_generation import generate_sales
from src.eda import run_eda


def main():
    """Rebuild deterministic data and derived reports."""
    DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_sales().to_csv(
        DEFAULT_DATA_PATH, index=False, date_format="%Y-%m-%d"
    )
    run_eda()


if __name__ == "__main__":
    main()
