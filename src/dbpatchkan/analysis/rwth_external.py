"""Stage 5A external validation on the RWTH commercial-EV fleet.

The module deliberately separates three questions:

1. Can driver kinematics predict immediate electrical/thermal-proxy stress on
   vehicles that were never present in model fitting?
2. Do accumulated physical exposures add capacity-SOH information beyond age
   and mileage?
3. Can the same exposures predict the *signed* capacity change between
   controlled tests?

The third task is an explicitly small-sample proof of concept.  Capacity-test
recalibration can produce negative apparent loss, so no monotonic target clamp
is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dbpatchkan.models.kan import ConceptKAN


SEEDS = (42, 52, 62, 72, 82)

TRIP_DRIVER_FEATURES = (
    "speed_mean_kmh",
    "speed_max_kmh",
    "speed_std_kmh",
    "accel_positive_rms_mps2",
    "decel_rms_mps2",
    "jerk_rms_mps3",
    "harsh_accel_fraction",
    "harsh_decel_fraction",
    "high_speed_fraction",
    "stop_fraction",
)
TRIP_CONTEXT_FEATURES = (
    "duration_observed_h",
    "distance_km",
    "soc_mean_pct",
    "dod_pct",
    "ambient_temp_mean_c",
    "vehicle_family",
)
TRIP_TARGETS = (
    "c_rate_peak",
    "high_current_fraction",
    "electrothermal_intensity",
    "thermal_proxy_peak",
    "energy_kwh_per_100km",
    "dbsi_partial",
)

CHECKPOINT_AGING = (
    "observation_days",
    "odometer_km",
    "vehicle_family",
)
CHECKPOINT_PHYSICS = (
    "sum_efc",
    "sum_electrothermal_proxy_dose_h",
    "mean_high_c_rate_fraction",
    "mean_soc_high_fraction",
    "mean_arrhenius_ambient_factor_mean",
    "field_distance_coverage",
)
CHECKPOINT_DRIVER = (
    "mean_speed_mean_kmh",
    "mean_accel_positive_rms_mps2",
    "mean_decel_rms_mps2",
    "mean_jerk_rms_mps3",
    "mean_harsh_accel_fraction",
    "mean_harsh_decel_fraction",
)

INTERVAL_CONTEXT = (
    "interval_days",
    "odometer_delta_km",
    "start_soh_pct",
    "field_distance_coverage",
    "vehicle_family",
)
INTERVAL_PHYSICS = (
    "efc_per_observed_hour",
    "electrothermal_intensity",
    "c2_intensity",
    "mean_high_c_rate_fraction",
    "mean_soc_high_fraction",
    "mean_arrhenius_ambient_factor_mean",
)
INTERVAL_DRIVER = (
    "mean_speed_mean_kmh",
    "mean_accel_positive_rms_mps2",
    "mean_decel_rms_mps2",
    "mean_jerk_rms_mps3",
    "mean_harsh_accel_fraction",
    "mean_harsh_decel_fraction",
)


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_metrics(true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(true) & np.isfinite(prediction)
    y, p = true[valid], prediction[valid]
    if not len(y):
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan, "r2": math.nan}
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(p - y)),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else math.nan,
    }


def _preprocessor(frame: pd.DataFrame, columns: tuple[str, ...], *, scale: bool) -> ColumnTransformer:
    categorical = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    numeric = [column for column in columns if column not in categorical]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _regression_pipeline(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    kind: str,
    seed: int,
) -> Pipeline:
    if kind == "median":
        estimator = DummyRegressor(strategy="median")
        scale = False
    elif kind == "ridge":
        estimator = Ridge(alpha=10.0)
        scale = True
    elif kind == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        )
        scale = False
    elif kind == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        scale = False
    elif kind == "ebm":
        from interpret.glassbox import ExplainableBoostingRegressor

        estimator = ExplainableBoostingRegressor(
            interactions=0,
            max_bins=128,
            min_samples_leaf=5,
            learning_rate=0.04,
            max_rounds=600,
            early_stopping_rounds=40,
            random_state=seed,
            n_jobs=-1,
        )
        scale = False
    else:
        raise ValueError(f"Unknown regression model: {kind}")
    return Pipeline([
        ("preprocess", _preprocessor(frame, columns, scale=scale)),
        ("model", estimator),
    ])


def _trip_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if "vehicle_family" not in table:
        table["vehicle_family"] = np.where(
            table.vehicle_id.astype(str).str.startswith("Smart-"), "Smart", "iMiEV_iOn"
        )
    duration = pd.to_numeric(table["duration_observed_h"], errors="coerce").clip(lower=1e-4)
    distance = pd.to_numeric(table["distance_km"], errors="coerce").clip(lower=0.1)
    table["c_rate_peak"] = pd.to_numeric(table["c_rate_discharge_peak"], errors="coerce")
    table["high_current_fraction"] = pd.to_numeric(table["high_c_rate_fraction"], errors="coerce")
    table["electrothermal_intensity"] = (
        pd.to_numeric(table["electrothermal_proxy_dose_h"], errors="coerce") / duration
    )
    table["thermal_proxy_peak"] = pd.to_numeric(table["thermal_memory_proxy_peak"], errors="coerce")
    table["energy_kwh_per_100km"] = (
        100.0 * pd.to_numeric(table["energy_discharge_kwh"], errors="coerce") / distance
    )
    table["dbsi_partial"] = pd.to_numeric(table["dbsi_partial_renormalized"], errors="coerce")
    return table.replace([np.inf, -np.inf], np.nan)


def _vehicle_bootstrap_delta(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    seed: int = 42,
    draws: int = 2000,
) -> dict[str, float]:
    subset = predictions[predictions.model.isin([model_a, model_b])]
    pivot = subset.pivot_table(
        index=["vehicle_id", "row_id"], columns="model", values=["true", "prediction"], aggfunc="first"
    ).dropna()
    if pivot.empty:
        return {"delta_mae_b_minus_a": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    per_vehicle = []
    for vehicle, group in pivot.groupby(level=0):
        true = group[("true", model_a)].to_numpy(float)
        a = group[("prediction", model_a)].to_numpy(float)
        b = group[("prediction", model_b)].to_numpy(float)
        per_vehicle.append((vehicle, np.mean(np.abs(b - true)) - np.mean(np.abs(a - true))))
    values = np.asarray([value for _, value in per_vehicle], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)])
    return {
        "delta_mae_b_minus_a": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def run_trip_response_validation(
    table: pd.DataFrame,
    output: Path,
    *,
    include_ebm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    update(f"RWTH trip response validation: {len(table):,} trips, {table.vehicle_id.nunique()} vehicles")
    specifications = {
        "Context-Ridge": (TRIP_CONTEXT_FEATURES, "ridge"),
        "Driver-Ridge": (TRIP_DRIVER_FEATURES, "ridge"),
        "DriverContext-Ridge": (TRIP_DRIVER_FEATURES + TRIP_CONTEXT_FEATURES, "ridge"),
        "DriverContext-XGBoost": (TRIP_DRIVER_FEATURES + TRIP_CONTEXT_FEATURES, "xgboost"),
    }
    try:
        if not include_ebm:
            raise ImportError
        import interpret  # noqa: F401

        specifications["DriverContext-EBM"] = (
            TRIP_DRIVER_FEATURES + TRIP_CONTEXT_FEATURES,
            "ebm",
        )
    except ImportError:
        update("interpret is unavailable; EBM is skipped but all mandatory models continue")

    records: list[dict[str, object]] = []
    vehicles = sorted(table.vehicle_id.unique())
    for target in TRIP_TARGETS:
        for fold, held_out in enumerate(vehicles):
            train = table[table.vehicle_id.ne(held_out) & table[target].notna()].copy()
            test = table[table.vehicle_id.eq(held_out) & table[target].notna()].copy()
            if train.empty or test.empty:
                continue
            for model_name, (features, kind) in specifications.items():
                model = _regression_pipeline(train, features, kind, seed=42 + fold)
                model.fit(train[list(features)], train[target])
                prediction = model.predict(test[list(features)])
                for row_id, vehicle_id, true, pred in zip(
                    test.index, test.vehicle_id, test[target], prediction
                ):
                    records.append({
                        "task": target,
                        "outer_fold": fold,
                        "held_out_vehicle": held_out,
                        "vehicle_id": vehicle_id,
                        "row_id": int(row_id),
                        "model": model_name,
                        "true": float(true),
                        "prediction": float(pred),
                    })
            update(f"trip target={target}: held out {held_out} ({fold + 1}/{len(vehicles)})")

    predictions = pd.DataFrame(records)
    predictions.to_csv(output / "rwth_trip_response_loov_predictions.csv", index=False)
    metric_rows = []
    for (task, model), group in predictions.groupby(["task", "model"]):
        pooled = _finite_metrics(group.true.to_numpy(), group.prediction.to_numpy())
        vehicle_mae = group.groupby("vehicle_id").apply(
            lambda x: mean_absolute_error(x.true, x.prediction), include_groups=False
        )
        metric_rows.append({
            "task": task,
            "model": model,
            **pooled,
            "vehicle_macro_mae": float(vehicle_mae.mean()),
            "vehicle_mae_std": float(vehicle_mae.std(ddof=1)),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "rwth_trip_response_loov_metrics.csv", index=False)

    deltas = []
    for target in TRIP_TARGETS:
        task_predictions = predictions[predictions.task.eq(target)]
        for challenger in ("DriverContext-Ridge", "DriverContext-XGBoost", "DriverContext-EBM"):
            if challenger not in set(task_predictions.model):
                continue
            delta = _vehicle_bootstrap_delta(task_predictions, "Context-Ridge", challenger)
            deltas.append({
                "task": target,
                "reference": "Context-Ridge",
                "challenger": challenger,
                **delta,
                "driver_adds_value": bool(delta["ci_high"] < 0),
            })
    increments = pd.DataFrame(deltas)
    increments.to_csv(output / "rwth_incremental_driver_value.csv", index=False)
    return predictions, metrics


def run_stress_classification(table: pd.DataFrame, output: Path) -> pd.DataFrame:
    update("RWTH fold-safe Low/Moderate/High stress classification")
    features = TRIP_DRIVER_FEATURES + TRIP_CONTEXT_FEATURES
    records = []
    vehicles = sorted(table.vehicle_id.unique())
    for fold, held_out in enumerate(vehicles):
        train = table[table.vehicle_id.ne(held_out) & table.dbsi_partial.notna()].copy()
        test = table[table.vehicle_id.eq(held_out) & table.dbsi_partial.notna()].copy()
        low, high = train.dbsi_partial.quantile([1 / 3, 2 / 3]).to_numpy(float)
        y_train = np.select(
            [train.dbsi_partial <= low, train.dbsi_partial >= high], [0, 2], default=1
        ).astype(int)
        y_test = np.select(
            [test.dbsi_partial <= low, test.dbsi_partial >= high], [0, 2], default=1
        ).astype(int)
        preprocess = _preprocessor(train, features, scale=False)
        model = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42 + fold,
                n_jobs=-1,
            )),
        ])
        model.fit(train[list(features)], y_train)
        probabilities = model.predict_proba(test[list(features)])
        prediction = probabilities.argmax(axis=1)
        for row_id, vehicle, true, pred, prob in zip(
            test.index, test.vehicle_id, y_test, prediction, probabilities
        ):
            records.append({
                "outer_fold": fold,
                "held_out_vehicle": held_out,
                "vehicle_id": vehicle,
                "row_id": int(row_id),
                "threshold_low": low,
                "threshold_high": high,
                "true_class": int(true),
                "predicted_class": int(pred),
                "probability_low": float(prob[0]),
                "probability_moderate": float(prob[1]),
                "probability_high": float(prob[2]),
            })
    predictions = pd.DataFrame(records)
    predictions.to_csv(output / "rwth_stress_class_loov_predictions.csv", index=False)
    metrics = {
        "n": int(len(predictions)),
        "macro_f1": float(f1_score(predictions.true_class, predictions.predicted_class, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(predictions.true_class, predictions.predicted_class)),
        "log_loss": float(log_loss(
            predictions.true_class,
            predictions[["probability_low", "probability_moderate", "probability_high"]],
            labels=[0, 1, 2],
        )),
        "thresholds_fitted_inside_each_outer_fold": True,
        "vehicle_id_used_as_feature": False,
    }
    pd.DataFrame([metrics]).to_csv(output / "rwth_stress_class_loov_metrics.csv", index=False)
    return pd.DataFrame([metrics])


def _partial_driver_associations(table: pd.DataFrame, output: Path) -> pd.DataFrame:
    controls = ("duration_observed_h", "distance_km", "soc_mean_pct", "dod_pct", "ambient_temp_mean_c")
    rows = []
    for driver in TRIP_DRIVER_FEATURES:
        for target in TRIP_TARGETS:
            per_vehicle = []
            for vehicle, group in table.groupby("vehicle_id"):
                columns = [driver, target, *controls]
                clean = group[columns].apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean) < 30 or clean[driver].nunique() < 3 or clean[target].nunique() < 3:
                    continue
                x_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
                y_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
                x_model.fit(clean[list(controls)], clean[driver])
                y_model.fit(clean[list(controls)], clean[target])
                x_residual = clean[driver] - x_model.predict(clean[list(controls)])
                y_residual = clean[target] - y_model.predict(clean[list(controls)])
                rho, _ = spearmanr(x_residual, y_residual)
                if np.isfinite(rho):
                    per_vehicle.append((vehicle, len(clean), float(rho)))
            if not per_vehicle:
                continue
            rhos = np.asarray([item[2] for item in per_vehicle])
            weights = np.asarray([item[1] for item in per_vehicle], dtype=float)
            rows.append({
                "driver_feature": driver,
                "battery_response": target,
                "vehicles": len(per_vehicle),
                "rows": int(weights.sum()),
                "vehicle_median_partial_spearman": float(np.median(rhos)),
                "row_weighted_partial_spearman": float(np.average(rhos, weights=weights)),
                "same_direction_vehicle_fraction": float(max(np.mean(rhos > 0), np.mean(rhos < 0))),
                "controls": ",".join(controls),
                "claim_scope": "context-adjusted association, not causation",
            })
    result = pd.DataFrame(rows).sort_values(
        "vehicle_median_partial_spearman", key=lambda values: values.abs(), ascending=False
    )
    result.to_csv(output / "rwth_driver_battery_partial_associations.csv", index=False)
    return result


def _fit_additive_kan(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    bound_pp: float,
    epochs: int,
) -> tuple[np.ndarray, ConceptKAN]:
    seed_everything(seed)
    model = ConceptKAN((x_train.shape[1], 1), grid_size=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.05)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(residual_train, dtype=torch.float32, device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = bound_pp * torch.tanh(model(x).squeeze(-1))
        loss = torch.nn.functional.huber_loss(prediction, y, delta=1.0)
        loss = loss + 1e-2 * model.regularization_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        test = torch.tensor(x_test, dtype=torch.float32, device=device)
        prediction = bound_pp * torch.tanh(model(test).squeeze(-1))
    return prediction.cpu().numpy(), model


def _kan_residual_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    baseline_features: tuple[str, ...],
    kan_features: tuple[str, ...],
    target: str,
    device: torch.device,
    bound_pp: float,
    epochs: int,
) -> tuple[np.ndarray, list[ConceptKAN], Pipeline, Pipeline]:
    baseline = _regression_pipeline(train, baseline_features, "ridge", seed=42)
    baseline.fit(train[list(baseline_features)], train[target])
    baseline_train = baseline.predict(train[list(baseline_features)])
    baseline_test = baseline.predict(test[list(baseline_features)])
    transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    x_train = transformer.fit_transform(train[list(kan_features)]).astype(np.float32)
    x_test = transformer.transform(test[list(kan_features)]).astype(np.float32)
    residual = train[target].to_numpy(np.float32) - baseline_train.astype(np.float32)
    predictions, models = [], []
    for seed in SEEDS:
        correction, model = _fit_additive_kan(
            x_train,
            residual,
            x_test,
            seed=seed,
            device=device,
            bound_pp=bound_pp,
            epochs=epochs,
        )
        predictions.append(baseline_test + correction)
        models.append(model.cpu())
    return np.mean(predictions, axis=0), models, baseline, transformer


def _health_loov(
    table: pd.DataFrame,
    *,
    id_column: str,
    target: str,
    context: tuple[str, ...],
    physics: tuple[str, ...],
    driver: tuple[str, ...],
    output_prefix: str,
    output: Path,
    device: torch.device,
    epochs: int,
    bound_pp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = {
        "Context-Ridge": (context, "ridge"),
        "ContextPhysics-Ridge": (context + physics, "ridge"),
        "ContextPhysicsDriver-Ridge": (context + physics + driver, "ridge"),
        "ContextPhysics-ExtraTrees": (context + physics, "extra_trees"),
    }
    records = []
    vehicles = sorted(table.vehicle_id.unique())
    for fold, held_out in enumerate(vehicles):
        train = table[table.vehicle_id.ne(held_out) & table[target].notna()].copy()
        test = table[table.vehicle_id.eq(held_out) & table[target].notna()].copy()
        if train.empty or test.empty:
            continue
        for model_name, (features, kind) in specifications.items():
            model = _regression_pipeline(train, features, kind, seed=42 + fold)
            model.fit(train[list(features)], train[target])
            prediction = model.predict(test[list(features)])
            for identifier, true, pred in zip(test[id_column], test[target], prediction):
                records.append({
                    "outer_fold": fold,
                    "held_out_vehicle": held_out,
                    "vehicle_id": held_out,
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
                bound_pp=bound_pp,
                epochs=epochs,
            )
            for identifier, true, pred in zip(test[id_column], test[target], prediction):
                records.append({
                    "outer_fold": fold,
                    "held_out_vehicle": held_out,
                    "vehicle_id": held_out,
                    "row_id": str(identifier),
                    "model": model_name,
                    "true": float(true),
                    "prediction": float(pred),
                })
        update(f"{output_prefix}: held out {held_out} ({fold + 1}/{len(vehicles)})")

    predictions = pd.DataFrame(records)
    predictions.to_csv(output / f"{output_prefix}_loov_predictions.csv", index=False)
    rows = []
    for model, group in predictions.groupby("model"):
        stats = _finite_metrics(group.true.to_numpy(), group.prediction.to_numpy())
        per_vehicle = group.groupby("vehicle_id").apply(
            lambda x: mean_absolute_error(x.true, x.prediction), include_groups=False
        )
        rows.append({
            "task": target,
            "model": model,
            **stats,
            "vehicle_macro_mae": float(per_vehicle.mean()),
            "vehicle_mae_std": float(per_vehicle.std(ddof=1)),
        })
    metrics = pd.DataFrame(rows).sort_values("vehicle_macro_mae")
    metrics.to_csv(output / f"{output_prefix}_loov_metrics.csv", index=False)
    return predictions, metrics


def run_checkpoint_validation(
    frame: pd.DataFrame,
    output: Path,
    *,
    device: torch.device,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = frame[frame.trip_count.gt(0)].copy()
    update(f"RWTH capacity-SOH LOOV: {len(table)} checkpoints across {table.vehicle_id.nunique()} vehicles")
    predictions, metrics = _health_loov(
        table,
        id_column="checkpoint_id",
        target="capacity_soh_pct",
        context=CHECKPOINT_AGING,
        physics=CHECKPOINT_PHYSICS,
        driver=CHECKPOINT_DRIVER,
        output_prefix="rwth_capacity_soh",
        output=output,
        device=device,
        epochs=epochs,
        bound_pp=3.0,
    )
    chronological = table.sort_values(["vehicle_id", "test_number"]).copy()
    chronological["persistence_prediction"] = chronological.groupby("vehicle_id")["capacity_soh_pct"].shift()
    persistence = chronological.dropna(subset=["persistence_prediction"])
    persistence_stats = _finite_metrics(
        persistence.capacity_soh_pct.to_numpy(), persistence.persistence_prediction.to_numpy()
    )
    pd.DataFrame([{
        "task": "capacity_soh_pct",
        "model": "PreviousCapacity-Persistence",
        **persistence_stats,
        "scope": "only checkpoints with an earlier controlled capacity test on the same vehicle",
    }]).to_csv(output / "rwth_capacity_persistence_metrics.csv", index=False)
    return predictions, metrics


def run_interval_validation(
    frame: pd.DataFrame,
    output: Path,
    *,
    device: torch.device,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = frame[frame.trip_count.gt(0)].copy()
    duration = pd.to_numeric(table.sum_duration_observed_h, errors="coerce").clip(lower=1e-4)
    table["efc_per_observed_hour"] = pd.to_numeric(table.sum_efc, errors="coerce") / duration
    table["electrothermal_intensity"] = (
        pd.to_numeric(table.sum_electrothermal_proxy_dose_h, errors="coerce") / duration
    )
    table["c2_intensity"] = pd.to_numeric(table.sum_joule_c2_dose_h, errors="coerce") / duration
    update(
        f"RWTH signed degradation LOOV proof-of-concept: {len(table)} intervals across "
        f"{table.vehicle_id.nunique()} vehicles"
    )
    return _health_loov(
        table,
        id_column="interval_id",
        target="soh_loss_pp",
        context=INTERVAL_CONTEXT,
        physics=INTERVAL_PHYSICS,
        driver=INTERVAL_DRIVER,
        output_prefix="rwth_signed_degradation",
        output=output,
        device=device,
        epochs=epochs,
        bound_pp=4.0,
    )


def _conformal_summary(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        residual = np.abs(group.true.to_numpy(float) - group.prediction.to_numpy(float))
        residual = residual[np.isfinite(residual)]
        if not len(residual):
            continue
        rank = min(len(residual) - 1, math.ceil((len(residual) + 1) * 0.90) - 1)
        radius = float(np.sort(residual)[rank])
        coverage = float(np.mean(residual <= radius))
        rows.append({
            "task": task,
            "model": model,
            "n_loov_residuals": len(residual),
            "nominal_coverage": 0.90,
            "loov_residual_radius_pp": radius,
            "empirical_coverage": coverage,
            "status": "exploratory; same LOOV residuals used to estimate and audit width",
        })
    return pd.DataFrame(rows)


def _fit_final_kan_explanation(
    table: pd.DataFrame,
    output: Path,
    *,
    device: torch.device,
    epochs: int,
) -> None:
    features = CHECKPOINT_PHYSICS
    prediction, models, baseline, transform = _kan_residual_predictions(
        table,
        table,
        baseline_features=CHECKPOINT_AGING,
        kan_features=features,
        target="capacity_soh_pct",
        device=device,
        bound_pp=3.0,
        epochs=epochs,
    )
    joblib.dump(baseline, output / "rwth_capacity_final_age_mileage_baseline.joblib")
    joblib.dump(transform, output / "rwth_capacity_final_physics_transform.joblib")
    for seed, model in zip(SEEDS, models):
        torch.save({
            "state_dict": model.state_dict(),
            "features": features,
            "dimensions": model.dimensions,
            "grid_size": 3,
            "label": "capacity_soh_pct",
            "label_definition": "published fixed-voltage-window 1C discharge capacity",
        }, output / f"rwth_capacity_additive_kan_seed_{seed}.pt")

    grid = np.linspace(-3.0, 3.0, 121, dtype=np.float32)
    rows = []
    for feature_index, feature in enumerate(features):
        values = np.zeros((len(grid), len(features)), dtype=np.float32)
        values[:, feature_index] = grid
        seed_curves = []
        for model in models:
            model.eval()
            with torch.no_grad():
                raw = model(torch.tensor(values)).squeeze(-1)
                seed_curves.append((3.0 * torch.tanh(raw)).numpy())
        curves = np.stack(seed_curves)
        for position, standardized_value in enumerate(grid):
            rows.append({
                "feature": feature,
                "standardized_value": float(standardized_value),
                "soh_correction_pp_mean": float(curves[:, position].mean()),
                "soh_correction_pp_std": float(curves[:, position].std(ddof=1)),
                "interpretation": "additive correction around age/mileage baseline; descriptive full-data fit",
            })
    pd.DataFrame(rows).to_csv(output / "rwth_capacity_kan_spline_curves.csv", index=False)
    pd.DataFrame({
        "checkpoint_id": table.checkpoint_id,
        "capacity_soh_pct": table.capacity_soh_pct,
        "full_data_fitted_prediction": prediction,
    }).to_csv(output / "rwth_capacity_final_fitted_predictions_DIAGNOSTIC.csv", index=False)


def _acceptance(
    trip_metrics: pd.DataFrame,
    increments: pd.DataFrame,
    capacity_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
) -> dict[str, object]:
    capacity = capacity_metrics.set_index("model")
    interval = interval_metrics.set_index("model")
    driver_tasks = increments.groupby("task").driver_adds_value.any() if len(increments) else pd.Series(dtype=bool)
    return {
        "dataset": "RWTH commercial EV fleet",
        "external_vehicles": 9,
        "trip_response_rows": int(trip_metrics.n.max()) if len(trip_metrics) else 0,
        "driver_increment_significant_tasks": [str(task) for task, value in driver_tasks.items() if value],
        "capacity_checkpoints_with_exposure": 16,
        "capacity_physics_kan_beats_aging_ridge": bool(
            capacity.loc["CorrectedPhysics-AdditiveKAN", "vehicle_macro_mae"]
            < capacity.loc["Context-Ridge", "vehicle_macro_mae"]
        ),
        "capacity_driver_direct_addition_beats_physics_kan": bool(
            capacity.loc["CorrectedPhysicsDriver-AdditiveKAN", "vehicle_macro_mae"]
            < capacity.loc["CorrectedPhysics-AdditiveKAN", "vehicle_macro_mae"]
        ),
        "degradation_intervals_with_exposure": 8,
        "interval_physics_kan_beats_context_ridge": bool(
            interval.loc["CorrectedPhysics-AdditiveKAN", "vehicle_macro_mae"]
            < interval.loc["Context-Ridge", "vehicle_macro_mae"]
        ),
        "causal_claim_authorized": False,
        "fleet_scale_architecture_claim_authorized": False,
        "nissan_locked_test_used": False,
        "interpretation": (
            "Trip results test the driver-to-stress pathway. Capacity and interval results are "
            "small-sample external proof-of-concept and must be reported with vehicle-wise LOOV metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-features", type=Path, required=True)
    parser.add_argument("--checkpoint-exposures", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--kan-epochs", type=int, default=300)
    parser.add_argument("--skip-ebm", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    update(f"Stage 5A starts on {device}; KAN splines remain float32")

    trips = _trip_table(pd.read_parquet(args.trip_features))
    checkpoints = pd.read_csv(args.checkpoint_exposures)
    intervals = pd.read_csv(args.intervals)
    trip_predictions, trip_metrics = run_trip_response_validation(
        trips, output, include_ebm=not args.skip_ebm
    )
    stress_metrics = run_stress_classification(trips, output)
    associations = _partial_driver_associations(trips, output)
    capacity_predictions, capacity_metrics = run_checkpoint_validation(
        checkpoints, output, device=device, epochs=args.kan_epochs
    )
    interval_predictions, interval_metrics = run_interval_validation(
        intervals, output, device=device, epochs=args.kan_epochs
    )
    uncertainty = pd.concat([
        _conformal_summary(capacity_predictions, "capacity_soh_pct"),
        _conformal_summary(interval_predictions, "signed_soh_loss_pp"),
    ], ignore_index=True)
    uncertainty.to_csv(output / "rwth_loov_uncertainty_exploratory.csv", index=False)
    usable_checkpoints = checkpoints[checkpoints.trip_count.gt(0)].copy()
    _fit_final_kan_explanation(usable_checkpoints, output, device=device, epochs=args.kan_epochs)

    increments = pd.read_csv(output / "rwth_incremental_driver_value.csv")
    decision = _acceptance(trip_metrics, increments, capacity_metrics, interval_metrics)
    (output / "RWTH_EXTERNAL_ACCEPTANCE.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    report = f"""# RWTH Stage 5A external validation

The external pathway is evaluated in two evidence layers.  First, {len(trips):,} trips
from {trips.vehicle_id.nunique()} vehicles test whether driver kinematics add predictive
value for immediate electrical and latent-thermal stress after holding out each vehicle.
Second, 16 capacity checkpoints and 8 exposure-bearing degradation intervals test whether
the accumulated physical concepts add information beyond age, mileage, and observation
context.

The controlled-test label is capacity SOH in a fixed voltage window.  It is not pooled
with Nissan BMS SOH.  Pack temperature is absent from RWTH field telemetry; all thermal
features are explicitly latent C-rate/ambient proxies.  Apparent capacity gains remain
signed because controlled capacity measurements can fluctuate.  No causal claim is made.

Driver-increment tasks with a vehicle-bootstrap 95% interval below zero:
{decision['driver_increment_significant_tasks']}

Capacity physics KAN beats aging/context Ridge: {decision['capacity_physics_kan_beats_aging_ridge']}.
Signed-interval physics KAN beats context Ridge: {decision['interval_physics_kan_beats_context_ridge']}.
"""
    (output / "RWTH_EXTERNAL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "acceptance": decision,
        "trip_metrics": trip_metrics.to_dict(orient="records"),
        "stress_metrics": stress_metrics.to_dict(orient="records"),
        "capacity_metrics": capacity_metrics.to_dict(orient="records"),
        "interval_metrics": interval_metrics.to_dict(orient="records"),
        "top_associations": associations.head(30).to_dict(orient="records"),
    }
    (output / "RWTH_EXTERNAL_VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    update("Stage 5A complete; Nissan locked test was never loaded")


if __name__ == "__main__":
    main()
