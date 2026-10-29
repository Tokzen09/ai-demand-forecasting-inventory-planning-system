"""CLI: python -m src.forecast --source mysql|csv|auto."""

import argparse

from src.forecasting.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["auto", "mysql", "csv"], default="auto"
    )
    args = parser.parse_args()
    run_pipeline(args.source)


if __name__ == "__main__":
    main()
