"""Fold-safe log/robust-normalised additive-KAN sensitivity for Stage 5.

This analysis is deliberately labelled post-hoc.  It does not overwrite the
primary external-validation decisions.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from dbpatchkan.analysis.eve_external import expanding_splits
from dbpatchkan.analysis.rwth_external import SEEDS, _fit_additive_kan, _regression_pipeline


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def concept_values(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    log_columns: tuple[str, ...],
) -> pd.DataFrame:
    values = numeric(frame, columns)
    for column in log_columns:
        if column in values:
            values[column] = np.log1p(values[column].clip(lower=0))
    return values


def one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    context: tuple[str, ...],
    concepts: tuple[str, ...],
    log_concepts: tuple[str, ...],
    target: str,
    device: torch.device,
    epochs: int,
) -> dict[str, np.ndarray]:
    baseline = _regression_pipeline(train, context, "ridge", seed=42)
    baseline.fit(train[list(context)], train[target])
    baseline_train = baseline.predict(train[list(context)])
    baseline_test = baseline.predict(test[list(context)])

    concept_transform = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
    ])
    x_train = concept_transform.fit_transform(
        concept_values(train, concepts, log_concepts)
    ).astype(np.float32)
    x_test = concept_transform.transform(
        concept_values(test, concepts, log_concepts)
    ).astype(np.float32)
    x_train = np.clip(x_train, -3.0, 3.0)
    x_test = np.clip(x_test, -3.0, 3.0)
    residual_train = train[target].to_numpy(np.float32) - baseline_train.astype(np.float32)

    ridge_residual = Ridge(alpha=10.0).fit(x_train, residual_train)
    ridge_prediction = baseline_test + ridge_residual.predict(x_test)
    bound = float(np.clip(np.nanstd(residual_train), 0.5, 4.0))
    seed_predictions = []
    for seed in SEEDS:
        correction, _ = _fit_additive_kan(
            x_train,
            residual_train,
            x_test,
            seed=seed,
            device=device,
            bound_pp=bound,
            epochs=epochs,
        )
        seed_predictions.append(baseline_test + correction)
    return {
        "Context-Ridge": baseline_test,
        "LogRobustResidual-Ridge": ridge_prediction,
        "LogRobustResidual-AdditiveKAN": np.mean(seed_predictions, axis=0),
    }


def metrics(true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(true) & np.isfinite(prediction)
    y, p = true[valid], prediction[valid]
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(p - y)),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else math.nan,
    }


def evaluate(
    name: str,
    table: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    id_column: str,
    group_column: str,
    context: tuple[str, ...],
    concepts: tuple[str, ...],
    log_concepts: tuple[str, ...],
    target: str,
    device: torch.device,
    epochs: int,
    output: Path,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splits):
        train = table.iloc[train_index].dropna(subset=[target]).copy()
        test = table.iloc[test_index].dropna(subset=[target]).copy()
        if train.empty or test.empty:
            continue
        predictions = one_fold(
            train,
            test,
            context=context,
            concepts=concepts,
            log_concepts=log_concepts,
            target=target,
            device=device,
            epochs=epochs,
        )
        for model, prediction in predictions.items():
            for identifier, group, true, pred in zip(
                test[id_column], test[group_column], test[target], prediction
            ):
                records.append({
                    "task": name,
                    "target": target,
                    "fold": fold,
                    "row_id": str(identifier),
                    "group": str(group),
                    "model": model,
                    "true": float(true),
                    "prediction": float(pred),
                })
        update(f"{name}: fold {fold + 1}/{len(splits)}")
    predictions = pd.DataFrame(records)
    predictions.to_csv(output / f"{name}_predictions.csv", index=False)
    rows = []
    for model, group in predictions.groupby("model"):
        summary = metrics(group.true.to_numpy(float), group.prediction.to_numpy(float))
        group_mae = group.groupby("group").apply(
            lambda frame: mean_absolute_error(frame.true, frame.prediction),
            include_groups=False,
        )
        rows.append({
            "task": name,
            "target": target,
            "model": model,
            **summary,
            "group_macro_mae": float(group_mae.mean()),
            "group_mae_std": float(group_mae.std(ddof=1)),
            "analysis_status": "posthoc_fold_safe_sensitivity",
        })
    result = pd.DataFrame(rows).sort_values("group_macro_mae")
    result.to_csv(output / f"{name}_metrics.csv", index=False)
    return result


def loov_splits(table: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    vehicles = table.vehicle_id.astype(str).to_numpy()
    for vehicle in sorted(set(vehicles)):
        output.append((np.flatnonzero(vehicles != vehicle), np.flatnonzero(vehicles == vehicle)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwth-checkpoints", type=Path, required=True)
    parser.add_argument("--rwth-intervals", type=Path, required=True)
    parser.add_argument("--eve-checkpoints", type=Path, required=True)
    parser.add_argument("--eve-intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    update(f"Post-Block-11 sensitivity starts on {device}")

    rwth_cp = pd.read_csv(args.rwth_checkpoints)
    rwth_cp = rwth_cp[rwth_cp.trip_count.gt(0)].copy().reset_index(drop=True)
    rwth_iv = pd.read_csv(args.rwth_intervals)
    rwth_iv = rwth_iv[rwth_iv.trip_count.gt(0)].copy().reset_index(drop=True)
    eve_cp = pd.read_csv(args.eve_checkpoints, parse_dates=["checkpoint_time"])
    eve_cp = eve_cp[eve_cp.trip_count_cumulative.gt(0)].sort_values("checkpoint_time").reset_index(drop=True)
    eve_cp["temporal_group"] = np.arange(len(eve_cp))
    eve_iv = pd.read_csv(args.eve_intervals, parse_dates=["checkpoint_time"])
    eve_iv = eve_iv[eve_iv.trip_count_recent_31d.gt(0)].sort_values("checkpoint_time").reset_index(drop=True)
    eve_iv["temporal_group"] = np.arange(len(eve_iv))

    results = []
    results.append(evaluate(
        "rwth_capacity_soh", rwth_cp, loov_splits(rwth_cp),
        id_column="checkpoint_id", group_column="vehicle_id",
        context=("observation_days", "odometer_km", "vehicle_family"),
        concepts=(
            "sum_efc", "sum_electrothermal_proxy_dose_h", "mean_high_c_rate_fraction",
            "mean_soc_high_fraction", "mean_arrhenius_ambient_factor_mean",
            "field_distance_coverage", "mean_accel_positive_rms_mps2",
            "mean_decel_rms_mps2", "mean_jerk_rms_mps3",
        ),
        log_concepts=("sum_efc", "sum_electrothermal_proxy_dose_h"),
        target="capacity_soh_pct", device=device, epochs=args.epochs, output=output,
    ))
    results.append(evaluate(
        "rwth_signed_degradation", rwth_iv, loov_splits(rwth_iv),
        id_column="interval_id", group_column="vehicle_id",
        context=("interval_days", "odometer_delta_km", "start_soh_pct", "field_distance_coverage", "vehicle_family"),
        concepts=(
            "sum_efc", "sum_electrothermal_proxy_dose_h", "sum_joule_c2_dose_h",
            "mean_high_c_rate_fraction", "mean_soc_high_fraction",
            "mean_arrhenius_ambient_factor_mean", "mean_accel_positive_rms_mps2",
            "mean_decel_rms_mps2", "mean_jerk_rms_mps3",
        ),
        log_concepts=("sum_efc", "sum_electrothermal_proxy_dose_h", "sum_joule_c2_dose_h"),
        target="soh_loss_pp", device=device, epochs=args.epochs, output=output,
    ))
    eve_cp_splits = expanding_splits(len(eve_cp), min_train=max(10, int(0.50 * len(eve_cp))), folds=4)
    results.append(evaluate(
        "eve_bms_soh_trend", eve_cp, eve_cp_splits,
        id_column="checkpoint_id", group_column="temporal_group",
        context=("calendar_days", "odometer_proxy_km"),
        concepts=(
            "cum_efc", "cum_electrothermal_dose_h", "cum_high_soc_hot_time_h",
            "cum_stress_dose_h", "cum_mean_pack_temp_mean_c", "cum_mean_c_rate_abs_p95",
            "cum_mean_accel_positive_rms_mps2", "cum_mean_decel_rms_mps2",
            "cum_mean_jerk_rms_mps3",
        ),
        log_concepts=(
            "cum_efc", "cum_electrothermal_dose_h", "cum_high_soc_hot_time_h", "cum_stress_dose_h",
        ),
        target="bms_soh_trend_pct", device=device, epochs=args.epochs, output=output,
    ))
    eve_iv_splits = expanding_splits(len(eve_iv), min_train=max(10, int(0.50 * len(eve_iv))), folds=4)
    results.append(evaluate(
        "eve_signed_monthly_trend_loss", eve_iv, eve_iv_splits,
        id_column="checkpoint_id", group_column="temporal_group",
        context=("calendar_days", "recent_duration_h", "recent_distance_km", "previous_soh_trend_pct"),
        concepts=(
            "recent_efc", "recent_electrothermal_dose_h", "recent_high_soc_hot_time_h",
            "recent_stress_dose_h", "recent_mean_pack_temp_mean_c", "recent_mean_c_rate_abs_p95",
            "recent_mean_accel_positive_rms_mps2", "recent_mean_decel_rms_mps2",
            "recent_mean_jerk_rms_mps3",
        ),
        log_concepts=(
            "recent_efc", "recent_electrothermal_dose_h", "recent_high_soc_hot_time_h", "recent_stress_dose_h",
        ),
        target="signed_trend_loss_pp", device=device, epochs=args.epochs, output=output,
    ))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output / "POST_BLOCK11_NORMALIZATION_SENSITIVITY_METRICS.csv", index=False)
    decisions = []
    for task, group in combined.groupby("task"):
        indexed = group.set_index("model")
        kan = indexed.loc["LogRobustResidual-AdditiveKAN", "group_macro_mae"]
        baseline = indexed.loc["Context-Ridge", "group_macro_mae"]
        ridge = indexed.loc["LogRobustResidual-Ridge", "group_macro_mae"]
        decisions.append({
            "task": task,
            "kan_group_macro_mae": float(kan),
            "context_group_macro_mae": float(baseline),
            "transformed_ridge_group_macro_mae": float(ridge),
            "kan_beats_context": bool(kan < baseline),
            "kan_beats_transformed_ridge": bool(kan < ridge),
            "primary_acceptance_changed": False,
        })
    decision = {
        "analysis_status": "posthoc_fold_safe_sensitivity",
        "transform": "training-fold log1p -> median impute -> RobustScaler(10,90) -> clip[-3,3]",
        "primary_stage5_acceptance_changed": False,
        "nissan_locked_test_used": False,
        "tasks": decisions,
    }
    (output / "POST_BLOCK11_NORMALIZATION_SENSITIVITY_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(combined.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
