"""Stage 5B temporal replication on the public EVE Pilot Renault Zoe.

EVE contributes measured module temperature and a different manufacturer, but
only one vehicle.  Consequently, all results use expanding temporal folds and
are labelled cross-manufacturer replication rather than fleet validation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .rwth_external import (
    _conformal_summary,
    _finite_metrics,
    _kan_residual_predictions,
    _preprocessor,
    _regression_pipeline,
)


EVE_DRIVER = (
    "speed_mean_kmh",
    "speed_p95_kmh",
    "speed_std_kmh",
    "accel_positive_rms_mps2",
    "decel_rms_mps2",
    "jerk_rms_mps3",
    "harsh_accel_fraction",
    "harsh_decel_fraction",
    "stop_fraction",
)
EVE_CONTEXT = (
    "duration_h",
    "distance_km",
    "ambient_mean_c",
    "soc_mean_pct",
)
EVE_RESPONSE_TARGETS = (
    "c_rate_abs_p95",
    "pack_temp_rise_c",
    "pack_temp_max_c",
    "energy_kwh_per_100km",
    "electrothermal_intensity",
    "dbsi",
)

CHECKPOINT_AGING = ("calendar_days", "odometer_proxy_km")
CHECKPOINT_PHYSICS = (
    "cum_efc",
    "cum_electrothermal_dose_h",
    "cum_high_soc_hot_time_h",
    "cum_stress_dose_h",
    "cum_mean_pack_temp_mean_c",
    "cum_mean_c_rate_abs_p95",
)
CHECKPOINT_DRIVER = (
    "cum_mean_speed_mean_kmh",
    "cum_mean_accel_positive_rms_mps2",
    "cum_mean_decel_rms_mps2",
    "cum_mean_jerk_rms_mps3",
)
INTERVAL_CONTEXT = (
    "calendar_days",
    "recent_duration_h",
    "recent_distance_km",
    "previous_soh_trend_pct",
)
INTERVAL_PHYSICS = (
    "recent_efc",
    "recent_electrothermal_dose_h",
    "recent_high_soc_hot_time_h",
    "recent_stress_dose_h",
    "recent_mean_pack_temp_mean_c",
    "recent_mean_c_rate_abs_p95",
)
INTERVAL_DRIVER = (
    "recent_mean_speed_mean_kmh",
    "recent_mean_accel_positive_rms_mps2",
    "recent_mean_decel_rms_mps2",
    "recent_mean_jerk_rms_mps3",
)


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def expanding_splits(n_rows: int, *, min_train: int, folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows <= min_train:
        return []
    remaining = n_rows - min_train
    block = max(1, math.ceil(remaining / folds))
    result = []
    for start in range(min_train, n_rows, block):
        stop = min(n_rows, start + block)
        result.append((np.arange(0, start), np.arange(start, stop)))
    return result[:folds]


def run_trip_replication(
    trips: pd.DataFrame,
    output: Path,
    *,
    include_ebm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = trips.sort_values("trip_end").reset_index(drop=True).copy()
    table["electrothermal_intensity"] = table.electrothermal_dose_h / table.duration_h.clip(lower=1e-4)
    folds = expanding_splits(len(table), min_train=max(100, int(0.60 * len(table))), folds=4)
    specifications = {
        "Context-Ridge": (EVE_CONTEXT, "ridge"),
        "DriverContext-Ridge": (EVE_DRIVER + EVE_CONTEXT, "ridge"),
        "DriverContext-XGBoost": (EVE_DRIVER + EVE_CONTEXT, "xgboost"),
    }
    try:
        if not include_ebm:
            raise ImportError
        import interpret  # noqa: F401

        specifications["DriverContext-EBM"] = (EVE_DRIVER + EVE_CONTEXT, "ebm")
    except ImportError:
        pass
    records = []
    for target in EVE_RESPONSE_TARGETS:
        for fold, (train_indices, test_indices) in enumerate(folds):
            train = table.iloc[train_indices].dropna(subset=[target])
            test = table.iloc[test_indices].dropna(subset=[target])
            if train.empty or test.empty:
                continue
            for model_name, (features, kind) in specifications.items():
                model = _regression_pipeline(train, features, kind, seed=42 + fold)
                model.fit(train[list(features)], train[target])
                prediction = model.predict(test[list(features)])
                for row_id, true, pred in zip(test.index, test[target], prediction):
                    records.append({
                        "task": target,
                        "outer_fold": fold,
                        "row_id": int(row_id),
                        "trip_key": table.loc[row_id, "trip_key"],
                        "model": model_name,
                        "true": float(true),
                        "prediction": float(pred),
                    })
        update(f"EVE temporal response target complete: {target}")
    predictions = pd.DataFrame(records)
    predictions.to_csv(output / "eve_trip_response_temporal_predictions.csv", index=False)
    metrics = []
    for (task, model), group in predictions.groupby(["task", "model"]):
        metrics.append({"task": task, "model": model, **_finite_metrics(group.true, group.prediction)})
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "eve_trip_response_temporal_metrics.csv", index=False)
    return predictions, metrics_frame


def run_stress_classification(trips: pd.DataFrame, output: Path) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    table = trips.sort_values("trip_end").reset_index(drop=True)
    folds = expanding_splits(len(table), min_train=max(100, int(0.60 * len(table))), folds=4)
    features = EVE_DRIVER + EVE_CONTEXT
    rows = []
    for fold, (train_indices, test_indices) in enumerate(folds):
        train, test = table.iloc[train_indices], table.iloc[test_indices]
        low, high = train.dbsi.quantile([1 / 3, 2 / 3]).to_numpy(float)
        y_train = np.select([train.dbsi <= low, train.dbsi >= high], [0, 2], default=1).astype(int)
        y_test = np.select([test.dbsi <= low, test.dbsi >= high], [0, 2], default=1).astype(int)
        model = Pipeline([
            ("preprocess", _preprocessor(train, features, scale=False)),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42 + fold,
                n_jobs=-1,
            )),
        ])
        model.fit(train[list(features)], y_train)
        probability = model.predict_proba(test[list(features)])
        prediction = probability.argmax(axis=1)
        for row_id, true, pred, prob in zip(test.index, y_test, prediction, probability):
            rows.append({
                "outer_fold": fold,
                "row_id": int(row_id),
                "threshold_low": low,
                "threshold_high": high,
                "true_class": int(true),
                "predicted_class": int(pred),
                "probability_low": float(prob[0]),
                "probability_moderate": float(prob[1]),
                "probability_high": float(prob[2]),
            })
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "eve_stress_class_temporal_predictions.csv", index=False)
    metrics = pd.DataFrame([{
        "n": len(predictions),
        "macro_f1": f1_score(predictions.true_class, predictions.predicted_class, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(predictions.true_class, predictions.predicted_class),
        "log_loss": log_loss(
            predictions.true_class,
            predictions[["probability_low", "probability_moderate", "probability_high"]],
            labels=[0, 1, 2],
        ),
        "thresholds_fitted_inside_each_temporal_fold": True,
    }])
    metrics.to_csv(output / "eve_stress_class_temporal_metrics.csv", index=False)
    return metrics


def driver_response_associations(trips: pd.DataFrame, output: Path) -> pd.DataFrame:
    table = trips.copy()
    table["electrothermal_intensity"] = table.electrothermal_dose_h / table.duration_h.clip(lower=1e-4)
    controls = EVE_CONTEXT
    rows = []
    for driver in EVE_DRIVER:
        for target in EVE_RESPONSE_TARGETS:
            clean = table[[driver, target, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(clean) < 30:
                continue
            x_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            y_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            x_model.fit(clean[list(controls)], clean[driver])
            y_model.fit(clean[list(controls)], clean[target])
            x_residual = clean[driver] - x_model.predict(clean[list(controls)])
            y_residual = clean[target] - y_model.predict(clean[list(controls)])
            rho, p_value = spearmanr(x_residual, y_residual)
            rows.append({
                "driver_feature": driver,
                "battery_response": target,
                "n": len(clean),
                "partial_spearman": rho,
                "p_value_descriptive_only": p_value,
                "controls": ",".join(controls),
                "claim_scope": "single-vehicle context-adjusted association; not causation",
            })
    result = pd.DataFrame(rows).sort_values(
        "partial_spearman", key=lambda values: values.abs(), ascending=False
    )
    result.to_csv(output / "eve_driver_battery_partial_associations.csv", index=False)
    return result


def _temporal_health_models(
    table: pd.DataFrame,
    *,
    id_column: str,
    target: str,
    context: tuple[str, ...],
    physics: tuple[str, ...],
    driver: tuple[str, ...],
    prefix: str,
    output: Path,
    device: torch.device,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = table.sort_values("checkpoint_time").reset_index(drop=True)
    folds = expanding_splits(len(table), min_train=max(10, int(0.50 * len(table))), folds=4)
    specifications = {
        "Context-Ridge": (context, "ridge"),
        "ContextPhysics-Ridge": (context + physics, "ridge"),
        "ContextPhysicsDriver-Ridge": (context + physics + driver, "ridge"),
    }
    rows = []
    for fold, (train_indices, test_indices) in enumerate(folds):
        train = table.iloc[train_indices].dropna(subset=[target])
        test = table.iloc[test_indices].dropna(subset=[target])
        if train.empty or test.empty:
            continue
        for model_name, (features, kind) in specifications.items():
            model = _regression_pipeline(train, features, kind, seed=42 + fold)
            model.fit(train[list(features)], train[target])
            prediction = model.predict(test[list(features)])
            for identifier, true, pred in zip(test[id_column], test[target], prediction):
                rows.append({
                    "outer_fold": fold,
                    "row_id": str(identifier),
                    "model": model_name,
                    "true": float(true),
                    "prediction": float(pred),
                })
        for model_name, concepts in (
            ("CorrectedPhysics-AdditiveKAN", physics),
            ("CorrectedPhysicsDriver-AdditiveKAN", physics + driver),
        ):
            prediction, _, _, _ = _kan_residual_predictions(
                train,
                test,
                baseline_features=context,
                kan_features=concepts,
                target=target,
                device=device,
                bound_pp=3.0,
                epochs=epochs,
            )
            for identifier, true, pred in zip(test[id_column], test[target], prediction):
                rows.append({
                    "outer_fold": fold,
                    "row_id": str(identifier),
                    "model": model_name,
                    "true": float(true),
                    "prediction": float(pred),
                })
        update(f"EVE {prefix}: temporal fold {fold + 1}/{len(folds)}")
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / f"{prefix}_temporal_predictions.csv", index=False)
    metrics = pd.DataFrame([
        {"task": target, "model": model, **_finite_metrics(group.true, group.prediction)}
        for model, group in predictions.groupby("model")
    ]).sort_values("mae")
    metrics.to_csv(output / f"{prefix}_temporal_metrics.csv", index=False)
    return predictions, metrics


def run_health_replication(
    checkpoints: pd.DataFrame,
    intervals: pd.DataFrame,
    output: Path,
    *,
    device: torch.device,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_table = checkpoints[checkpoints.trip_count_cumulative.gt(0)].copy()
    current_predictions, current_metrics = _temporal_health_models(
        checkpoint_table,
        id_column="checkpoint_id",
        target="bms_soh_trend_pct",
        context=CHECKPOINT_AGING,
        physics=CHECKPOINT_PHYSICS,
        driver=CHECKPOINT_DRIVER,
        prefix="eve_bms_soh_trend",
        output=output,
        device=device,
        epochs=epochs,
    )
    interval_table = intervals[intervals.trip_count_recent_31d.gt(0)].copy()
    interval_predictions, interval_metrics = _temporal_health_models(
        interval_table,
        id_column="checkpoint_id",
        target="signed_trend_loss_pp",
        context=INTERVAL_CONTEXT,
        physics=INTERVAL_PHYSICS,
        driver=INTERVAL_DRIVER,
        prefix="eve_signed_monthly_trend_loss",
        output=output,
        device=device,
        epochs=epochs,
    )
    persistence = checkpoint_table.sort_values("checkpoint_time").copy()
    persistence["prediction"] = persistence.bms_soh_trend_pct.shift()
    persistence = persistence.dropna(subset=["prediction"])
    pd.DataFrame([{
        "task": "bms_soh_trend_pct",
        "model": "PreviousMonth-Persistence",
        **_finite_metrics(persistence.bms_soh_trend_pct, persistence.prediction),
    }]).to_csv(output / "eve_bms_soh_persistence_metrics.csv", index=False)
    uncertainty = pd.concat([
        _conformal_summary(current_predictions, "bms_soh_trend_pct"),
        _conformal_summary(interval_predictions, "signed_monthly_trend_loss_pp"),
    ], ignore_index=True)
    uncertainty.to_csv(output / "eve_temporal_uncertainty_exploratory.csv", index=False)
    return current_predictions, current_metrics, interval_predictions, interval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-features", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--kan-epochs", type=int, default=300)
    parser.add_argument("--skip-ebm", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    trips = pd.read_parquet(args.trip_features)
    checkpoints = pd.read_csv(args.checkpoints, parse_dates=["checkpoint_time"])
    intervals = pd.read_csv(args.intervals, parse_dates=["checkpoint_time"])
    trip_predictions, trip_metrics = run_trip_replication(
        trips, output, include_ebm=not args.skip_ebm
    )
    stress_metrics = run_stress_classification(trips, output)
    associations = driver_response_associations(trips, output)
    current_predictions, current_metrics, interval_predictions, interval_metrics = run_health_replication(
        checkpoints, intervals, output, device=device, epochs=args.kan_epochs
    )

    metric_index = current_metrics.set_index("model")
    interval_index = interval_metrics.set_index("model")
    response_index = trip_metrics.set_index(["task", "model"])
    decision = {
        "dataset": "EVE Pilot Renault Zoe",
        "vehicle_count": 1,
        "measured_module_temperature": True,
        "current_label": "past-only 3-month rolling median of cleaned BMS SOH",
        "label_pooled_with_rwth_capacity_soh": False,
        "driver_improves_measured_temp_rise_over_context_ridge": bool(
            response_index.loc[("pack_temp_rise_c", "DriverContext-Ridge"), "mae"]
            < response_index.loc[("pack_temp_rise_c", "Context-Ridge"), "mae"]
        ),
        "physics_kan_beats_aging_current_soh": bool(
            metric_index.loc["CorrectedPhysics-AdditiveKAN", "mae"]
            < metric_index.loc["Context-Ridge", "mae"]
        ),
        "physics_kan_beats_context_monthly_loss": bool(
            interval_index.loc["CorrectedPhysics-AdditiveKAN", "mae"]
            < interval_index.loc["Context-Ridge", "mae"]
        ),
        "cross_manufacturer_replication": True,
        "independent_fleet_claim_authorized": False,
        "causal_claim_authorized": False,
        "nissan_locked_test_used": False,
    }
    (output / "EVE_EXTERNAL_ACCEPTANCE.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    report = f"""# EVE Stage 5B temporal replication

EVE provides a Renault Zoe with measured battery-module temperature, speed,
current, voltage, SOC, ambient temperature, and BMS SOH.  It replicates the
driver-to-battery-response pathway across a manufacturer and chemistry context
different from Nissan/RWTH, but it is still one vehicle.  Expanding temporal
folds are therefore used and no fleet-generalisation or causal claim is made.

Driver variables improve measured temperature-rise MAE over context-only Ridge:
{decision['driver_improves_measured_temp_rise_over_context_ridge']}.

The BMS-SOH label is cleaned and summarized monthly, with a past-only
three-month rolling median used as the current-health trend.  This label is not
pooled with controlled capacity SOH from RWTH.
"""
    (output / "EVE_EXTERNAL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "acceptance": decision,
        "trip_metrics": trip_metrics.to_dict(orient="records"),
        "stress_metrics": stress_metrics.to_dict(orient="records"),
        "current_soh_metrics": current_metrics.to_dict(orient="records"),
        "monthly_loss_metrics": interval_metrics.to_dict(orient="records"),
        "top_associations": associations.head(30).to_dict(orient="records"),
    }
    (output / "EVE_EXTERNAL_VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    update("Stage 5B complete; Nissan locked test was never loaded")


if __name__ == "__main__":
    main()
