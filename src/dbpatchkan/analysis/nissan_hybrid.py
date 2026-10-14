"""Ensemble, conformal uncertainty, baselines, and mediation for Nissan PI-DB-PatchKAN."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dbpatchkan.models import NISSAN_PHYSICS_CONCEPT_NAMES, NISSAN_RESPONSE_TARGET_NAMES


AGE_FEATURES = ["calendar_days_since_start", "odometer_clean_km"]
DRIVER_FEATURES = [
    "accel_pos_rms", "accel_pos_p95", "decel_abs_rms", "decel_abs_p95",
    "jerk_rms", "jerk_p95", "pedal_mean", "pedal_p95",
    "brake_active_fraction", "harsh_accel_per_min", "harsh_brake_per_min",
    "speed_p95_kmh",
]
BATTERY_FEATURES = [
    "c_rate_abs_p95", "c_rate_abs_peak", "pack_temp_max_c", "pack_temp_rise_c",
    "pack_temp_spread_p95_c", "high_c_rate_time_h", "soc_high_hot_time_h",
    "c_rate_squared_dose_h", "electrothermal_dose_h", "ah_throughput", "efc",
    "bsi_rate", "bsi_thermal", "bsi_soc", "bsi_transient", "bsi_interaction",
]


def finite_sample_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    level = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, level, method="higher"))


def _regression_metrics(true: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    valid = true.notna() & prediction.notna()
    y = true.loc[valid].to_numpy(float)
    p = prediction.loc[valid].to_numpy(float)
    error = p - y
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(error))) if len(y) else math.nan,
        "rmse": float(np.sqrt(np.mean(error**2))) if len(y) else math.nan,
        "bias": float(np.mean(error)) if len(y) else math.nan,
        "r2": float(r2_score(y, p)) if len(y) >= 2 else math.nan,
    }


def build_ensemble(run_root: Path, output: Path, coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(run_root.glob("seed_*/predictions.csv")):
        seed = int(path.parent.name.split("_")[-1])
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed predictions found below {run_root}")
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(output / "all_seed_predictions.csv", index=False)
    key_columns = ["split", "concept_source", "trip_key", "Trip"]
    numeric_prediction_columns = [
        column for column in all_predictions
        if column.endswith("_pred") or column.startswith("stress_probability_")
    ]
    truth_columns = [column for column in all_predictions if column.endswith("_true")]
    ensemble = all_predictions.groupby(key_columns, as_index=False)[
        numeric_prediction_columns
    ].agg(["mean", "std"])
    ensemble.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column for column in ensemble.columns
    ]
    truth = all_predictions.groupby(key_columns, as_index=False)[truth_columns].first()
    ensemble = truth.merge(ensemble, on=key_columns, validate="one_to_one")

    calibration_rows = []
    for target in ("current_soh", "loss_30d", "loss_500km"):
        calibration = ensemble.loc[
            ensemble["split"].eq("calibration") & ensemble["concept_source"].eq("predicted")
        ]
        true = pd.to_numeric(calibration[f"{target}_true"], errors="coerce")
        pred = pd.to_numeric(calibration[f"{target}_pred_mean"], errors="coerce")
        valid = true.notna() & pred.notna()
        q = finite_sample_quantile((true[valid] - pred[valid]).abs().to_numpy(), coverage)
        calibration_rows.append({
            "target": target, "coverage_requested": coverage,
            "calibration_n": int(valid.sum()), "absolute_residual_quantile": q,
            "calibration_source": "last_20_percent_of_training_trips",
        })
        ensemble[f"{target}_interval_low"] = ensemble[f"{target}_pred_mean"] - q
        ensemble[f"{target}_interval_high"] = ensemble[f"{target}_pred_mean"] + q
        if target.startswith("loss_"):
            ensemble[f"{target}_interval_low"] = ensemble[f"{target}_interval_low"].clip(lower=0)
        else:
            ensemble[f"{target}_interval_low"] = ensemble[f"{target}_interval_low"].clip(0, 100)
            ensemble[f"{target}_interval_high"] = ensemble[f"{target}_interval_high"].clip(0, 100)
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(output / "conformal_calibration.csv", index=False)
    ensemble.to_csv(output / "ensemble_predictions_with_intervals.csv", index=False)
    return ensemble, calibration_frame


def ensemble_metrics(ensemble: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    validation = ensemble.loc[
        ensemble["split"].eq("validation") & ensemble["concept_source"].eq("predicted")
    ]
    for target in ("current_soh", "loss_30d", "loss_500km"):
        metrics = _regression_metrics(
            pd.to_numeric(validation[f"{target}_true"], errors="coerce"),
            pd.to_numeric(validation[f"{target}_pred_mean"], errors="coerce"),
        )
        true = pd.to_numeric(validation[f"{target}_true"], errors="coerce")
        low = pd.to_numeric(validation[f"{target}_interval_low"], errors="coerce")
        high = pd.to_numeric(validation[f"{target}_interval_high"], errors="coerce")
        valid = true.notna() & low.notna() & high.notna()
        metrics["interval_coverage"] = float(((true[valid] >= low[valid]) & (true[valid] <= high[valid])).mean()) if valid.any() else math.nan
        metrics["mean_interval_width"] = float((high[valid] - low[valid]).mean()) if valid.any() else math.nan
        rows.append({"task": target, "model": "PI-DB-PatchKAN ensemble", **metrics})
    stress_true = validation["stress_class_true"].astype(int)
    probability_columns = [f"stress_probability_{label}_mean" for label in ("low", "moderate", "high")]
    stress_pred = np.argmax(validation[probability_columns].to_numpy(), axis=1)
    rows.append({
        "task": "stress_class", "model": "PI-DB-PatchKAN ensemble", "n": len(validation),
        "macro_f1": float(f1_score(stress_true, stress_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(stress_true, stress_pred)),
    })
    for target in NISSAN_RESPONSE_TARGET_NAMES:
        rows.append({
            "task": target, "model": "PI-DB-PatchKAN ensemble",
            **_regression_metrics(
                pd.to_numeric(validation[f"{target}_true"], errors="coerce"),
                pd.to_numeric(validation[f"{target}_pred_mean"], errors="coerce"),
            ),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    return metrics


def run_baselines(table: pd.DataFrame, output: Path) -> pd.DataFrame:
    train = table.loc[table["split"].eq("train")].sort_values("trip_end_utc").copy()
    fit = train.iloc[: int(math.ceil(0.80 * len(train)))].copy()
    validation = table.loc[table["split"].eq("validation")].sort_values("trip_end_utc").copy()
    feature_sets = {
        "AgeMileage-Ridge": AGE_FEATURES,
        "DriverOnly-Ridge": DRIVER_FEATURES,
        "BatteryOperational-Ridge": BATTERY_FEATURES,
        "PhysicsConcept-Ridge": list(NISSAN_PHYSICS_CONCEPT_NAMES),
        "DriverBatteryPhysics-Ridge": list(dict.fromkeys(AGE_FEATURES + DRIVER_FEATURES + BATTERY_FEATURES + list(NISSAN_PHYSICS_CONCEPT_NAMES))),
        "DBSIOnly-Ridge": ["bsi_rate", "bsi_thermal", "bsi_soc", "bsi_cycle", "bsi_transient", "bsi_interaction", "stress_dose"],
    }
    target_map = {
        "current_soh": "current_soh_target",
        "loss_30d": "target_loss_30d_pp",
        "loss_500km": "target_loss_500km_pp",
    }
    prediction_rows = []
    metric_rows = []
    for task, target in target_map.items():
        y_fit = pd.to_numeric(fit[target], errors="coerce")
        y_val = pd.to_numeric(validation[target], errors="coerce")
        for model_name, features in feature_sets.items():
            valid_features = [feature for feature in features if feature in table]
            labelled = y_fit.notna()
            if labelled.sum() < 3:
                continue
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
            model.fit(fit.loc[labelled, valid_features], y_fit.loc[labelled])
            prediction = pd.Series(model.predict(validation[valid_features]), index=validation.index)
            if task.startswith("loss_"):
                prediction = prediction.clip(lower=0)
            metrics = _regression_metrics(y_val, prediction)
            metric_rows.append({"task": task, "model": model_name, **metrics})
            for index in validation.index:
                prediction_rows.append({
                    "task": task, "model": model_name, "trip_key": validation.loc[index, "trip_key"],
                    "true": y_val.loc[index], "prediction": prediction.loc[index],
                })
        persistence = (
            pd.to_numeric(validation["last_known_soh"], errors="coerce")
            if task == "current_soh" else pd.Series(0.0, index=validation.index)
        )
        metric_rows.append({"task": task, "model": "Persistence", **_regression_metrics(y_val, persistence)})
    pd.DataFrame(prediction_rows).to_csv(output / "baseline_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "baseline_metrics.csv", index=False)
    return metrics


def _zscore(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    std = values.std(ddof=0).replace(0, 1)
    return ((values - values.mean()) / std).mean(axis=1).to_numpy(float)


def _linear_coefficients(y: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *columns])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def mediation_analysis(table: pd.DataFrame, output: Path, seed: int = 42) -> pd.DataFrame:
    train = table.loc[table["split"].eq("train")].sort_values("trip_end_utc").copy()
    fit = train.iloc[: int(math.floor(0.80 * len(train)))].copy()
    fit["c2_intensity"] = fit["c_rate_squared_dose_h"] / fit["physics_duration_h"].replace(0, np.nan)
    fit["electrothermal_intensity"] = fit["electrothermal_dose_h"] / fit["physics_duration_h"].replace(0, np.nan)
    driver = _zscore(fit, DRIVER_FEATURES)
    stress_columns = [
        "c_rate_abs_peak", "c2_intensity", "electrothermal_intensity",
        "pack_temp_rise_c", "c_rate_transient_dose", "bsi_intensity",
    ]
    stress = _zscore(fit, stress_columns)
    age = _zscore(fit, AGE_FEATURES)
    rows = []
    rho, p_value = spearmanr(driver, stress)
    rows.append({
        "outcome": "battery_stress_index", "path": "a_driver_to_stress",
        "estimate": float(rho), "p_value_descriptive": float(p_value), "n": len(fit),
    })
    rng = np.random.default_rng(seed)
    for outcome_name, source_column in (
        ("current_degradation_from_100_pp", "current_degradation_from_100_pp"),
        ("future_500km_loss_pp", "target_loss_500km_pp"),
    ):
        outcome = pd.to_numeric(fit[source_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(outcome) & np.isfinite(driver) & np.isfinite(stress) & np.isfinite(age)
        y, d, m, a = outcome[valid], driver[valid], stress[valid], age[valid]
        if len(y) < 6:
            continue
        path_a = _linear_coefficients(m, [d, a])[1]
        outcome_model = _linear_coefficients(y, [d, m, a])
        direct = outcome_model[1]
        path_b = outcome_model[2]
        total = _linear_coefficients(y, [d, a])[1]
        indirect = path_a * path_b
        bootstrap = []
        for _ in range(1000):
            indices = rng.integers(0, len(y), len(y))
            try:
                aa = _linear_coefficients(m[indices], [d[indices], a[indices]])[1]
                bb = _linear_coefficients(y[indices], [d[indices], m[indices], a[indices]])[2]
                bootstrap.append(aa * bb)
            except np.linalg.LinAlgError:
                continue
        low, high = np.quantile(bootstrap, [0.025, 0.975]) if bootstrap else (math.nan, math.nan)
        for path, estimate in (
            ("a_driver_to_stress", path_a), ("b_stress_to_soh", path_b),
            ("indirect_driver_stress_soh", indirect), ("direct_driver_to_soh", direct),
            ("total_driver_to_soh", total),
        ):
            rows.append({
                "outcome": outcome_name, "path": path, "estimate": float(estimate),
                "bootstrap_ci_low": float(low) if path.startswith("indirect") else math.nan,
                "bootstrap_ci_high": float(high) if path.startswith("indirect") else math.nan,
                "n": int(len(y)), "claim": "exploratory_association_not_causal",
            })
    result = pd.DataFrame(rows)
    result.to_csv(output / "driver_stress_soh_mediation.csv", index=False)
    return result


def write_report(
    output: Path, metrics: pd.DataFrame, baselines: pd.DataFrame,
    calibration: pd.DataFrame, mediation: pd.DataFrame,
) -> None:
    current = metrics.loc[metrics["task"].eq("current_soh")].iloc[0]
    text = f"""# PI-DB-PatchKAN Nissan development report

## Registered architecture

The SOH outputs receive no unrestricted Transformer embedding. Driver and battery
streams interact through bidirectional cross-attention, predict twenty named
physics concepts, and then enter positive monotonic degradation paths plus a
bounded additive KAN interaction term.

Auxiliary outputs are ambient-adjusted pack-temperature rise, energy/100 km,
peak C-rate, maximum pack temperature, high-current duration,
high-SOC/high-temperature duration, and Low/Moderate/High D-BSI stress class.

## Development result

Five-seed predicted-concept ensemble current-SOH validation MAE:
**{current['mae']:.4f} percentage points** on **{int(current['n'])} trips**.

Intervals use absolute-residual conformal calibration on the final 20% of the
training chronology. Validation and the locked final 26-trip test set were not
used to set interval widths.

## Claim boundary

This single-vehicle Nissan analysis estimates a temporally ordered,
physics-mediated association. It does not establish a causal driver effect.
The locked test has not been loaded. External multi-vehicle validation remains
necessary before an architectural superiority claim.
"""
    (output / "NISSAN_HYBRID_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--physics-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.90)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(args.physics_table) if args.physics_table.suffix == ".parquet" else pd.read_csv(args.physics_table)
    ensemble, calibration = build_ensemble(args.run_root, output, args.coverage)
    metrics = ensemble_metrics(ensemble, output)
    baselines = run_baselines(table, output)
    mediation = mediation_analysis(table, output)
    write_report(output, metrics, baselines, calibration, mediation)
    summary = {
        "architecture": "PI-DB-PatchKAN", "seeds": 5,
        "test_used": False, "uncertainty_coverage_requested": args.coverage,
        "validation_metrics": metrics.to_dict(orient="records"),
    }
    (output / "NISSAN_HYBRID_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
