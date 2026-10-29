"""Chronological experiment and separately refitted forecast artifact."""

import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import ROOT
from src.forecasting.data import fingerprint, load_demand, validate_demand
from src.forecasting.engine import (
    METHODS, MODEL_PARAMETERS, backtest, fit_regressor, forecast_all_methods,
)
from src.forecasting.features import training_examples
from src.forecasting.metrics import evaluation_table, select_method

TRAIN_END = "2025-06-30"
VALIDATION_START, VALIDATION_END = "2025-07-01", "2025-09-30"
TEST_START, TEST_END = "2025-10-01", "2025-12-31"


def run_pipeline(source="auto", output_dir=None, model_dir=None):
    """No test-based tuning; all candidate specifications are fixed in code."""
    output_dir = Path(output_dir or ROOT / "data/processed/phase3a")
    model_dir = Path(model_dir or ROOT / "models/phase3a")
    for path in (output_dir, model_dir):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Artifacts must stay inside the project.")
        path.mkdir(parents=True, exist_ok=True)
    frame, actual_source = load_demand(source)
    frame = validate_demand(frame)
    if (
        frame.date.min() != pd.Timestamp("2024-01-01")
        or frame.date.max() != pd.Timestamp(TEST_END)
    ):
        raise ValueError("Phase 3A requires the complete 2024-2025 dataset.")

    validation_model = fit_regressor(frame, TRAIN_END)
    joblib.dump(validation_model, model_dir / "validation_regressor.joblib")
    validation = []
    for method in METHODS:
        prediction = backtest(
            frame, VALIDATION_START, VALIDATION_END, method, validation_model
        )
        prediction["split"] = "validation"
        validation.append(prediction)
    validation = pd.concat(validation, ignore_index=True)
    selected = select_method(evaluation_table(validation))
    # Freeze selection before fitting on validation or inspecting test scores.
    test_model = fit_regressor(frame, VALIDATION_END)
    joblib.dump(test_model, model_dir / "test_regressor.joblib")
    test = []
    for method in METHODS:
        prediction = backtest(frame, TEST_START, TEST_END, method, test_model)
        prediction["split"] = "test"
        test.append(prediction)
    all_predictions = pd.concat([validation, *test], ignore_index=True)
    evaluations = evaluation_table(all_predictions)
    all_predictions.to_csv(
        output_dir / "backtest_predictions.csv", index=False
    )
    evaluations.to_csv(output_dir / "evaluation.csv", index=False)
    overall = evaluations[evaluations.scope.eq("overall")]
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)

    # Production refit happens after the untouched final test has been scored.
    production_model = fit_regressor(frame, TEST_END)
    joblib.dump(production_model, model_dir / "production_regressor.joblib")
    joblib.dump({
        "method": selected,
        "model": production_model if selected == "random_forest" else None,
        "history": frame, "horizon": 7, "target": "customer_demand",
    }, model_dir / "selected_forecaster.joblib", compress=3)
    all_future = forecast_all_methods(frame, "2026-01-01", production_model)
    all_future.to_csv(output_dir / "next_7_days_all_methods.csv", index=False)
    future = all_future[all_future.method.eq(selected)].reset_index(drop=True)
    future.to_csv(output_dir / "next_7_days.csv", index=False)
    features, _ = training_examples(frame, TEST_END)
    names = production_model.named_steps["features"].get_feature_names_out()
    pd.DataFrame({
        "feature": names,
        "importance": production_model.named_steps[
            "regressor"
        ].feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    summary = {
        "target": "customer_demand", "horizon": 7, "seed": 42,
        "requested_source": source, "actual_source": actual_source,
        "rows": len(frame), "series": frame.groupby(
            ["store_id", "product_id"]
        ).ngroups,
        "data_sha256": fingerprint(frame),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "joblib": joblib.__version__,
        "pandas": pd.__version__, "model_parameters": MODEL_PARAMETERS,
        "feature_columns": features.columns.to_list(),
        "train_end": TRAIN_END,
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, TEST_END],
        "origin_stride_days": 7, "origins_per_period": 13,
        "unscored_trailing_days": [VALIDATION_END, TEST_END],
        "selection_metric": "validation overall WAPE, then MAE, then name",
        "selected_method": selected,
        "saved_forecast_methods": list(METHODS),
        "promotion_policy": "No advance schedule; exclude promotion and price",
        "future_forecast": ["2026-01-01", "2026-01-07"],
        "future_forecast_as_of": "2025-12-31 (dataset end, not today's date)",
        "overall_metrics": json.loads(overall.to_json(orient="records")),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (model_dir / "run_metadata.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(overall[["split", "method", "mae", "rmse", "wape_pct"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    print(f"Selected using validation only: {selected}")
    return summary
