"""CPU-only Stage 9 attribution and pack-aware semi-empirical sensitivity.

This module never reads the Nissan locked-test tensor.  It operates on:

* the first 101 chronological Nissan trips and their development health anchors;
* RWTH charge-to-charge trip metadata, field-trip physics features, and
  controlled capacity checkpoints;
* EVE monthly development/external intervals.

The purpose is scientific attribution and sensitivity analysis, not post-test
architecture selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


R_GAS = 8.31446261815324
REFERENCE_T_K = 298.15

CALENDAR = ["calendar_days"]
USAGE = ["distance_km", "efc", "ah_throughput"]
CHARGING = [
    "charge_event_count",
    "charge_depth_pct",
    "charge_high_soc_count",
    "charge_plug_in_soc_mean_pct",
]
TEMPERATURE = [
    "ambient_mean_c",
    "pack_temp_mean_c",
    "pack_temp_max_c",
    "pack_temp_spread_c",
    "hot_exposure_fraction",
]
DRIVER = [
    "speed_mean_kmh",
    "speed_p95_kmh",
    "accel_rms_mps2",
    "decel_rms_mps2",
    "jerk_rms_mps3",
    "harsh_accel_fraction",
    "harsh_decel_fraction",
    "c_rate_p95",
    "high_c_rate_fraction",
    "c_rate_squared_dose_h",
    "electrothermal_dose_h",
    "transient_dose",
    "dbsi_mean",
    "stress_dose_h",
]

NESTED_BLOCKS: list[tuple[str, list[str]]] = [
    ("Persistence_zero_loss", []),
    ("Calendar", CALENDAR),
    ("Calendar+Usage", CALENDAR + USAGE),
    ("Calendar+Usage+Charging", CALENDAR + USAGE + CHARGING),
    (
        "Calendar+Usage+Charging+Temperature",
        CALENDAR + USAGE + CHARGING + TEMPERATURE,
    ),
    (
        "Full+DriverResidual",
        CALENDAR + USAGE + CHARGING + TEMPERATURE + DRIVER,
    ),
    ("DriverOnly", DRIVER),
]


@dataclass(frozen=True)
class Stage9Inputs:
    nissan_trips: Path
    nissan_health: Path
    rwth_trips: Path
    rwth_checkpoints: Path
    rwth_trip_metadata_dir: Path
    eve_trips: Path
    eve_intervals: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _weighted_mean(
    frame: pd.DataFrame, value: str, weight: str = "duration_h", default: float = np.nan
) -> float:
    values = _numeric(frame, value)
    weights = _numeric(frame, weight, 1.0).fillna(0.0).clip(lower=0.0)
    valid = values.notna() & weights.gt(0.0)
    if not valid.any():
        valid = values.notna()
        if not valid.any():
            return float(default)
        return float(values.loc[valid].mean())
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _sum(frame: pd.DataFrame, name: str) -> float:
    return float(_numeric(frame, name, 0.0).fillna(0.0).sum())


def _max(frame: pd.DataFrame, name: str) -> float:
    values = _numeric(frame, name)
    return float(values.max()) if values.notna().any() else np.nan


def _safe_days(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(float((end - start).total_seconds() / 86400.0), 0.0)


def _arrhenius(temp_c: np.ndarray | pd.Series | float, ea_j_mol: float) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    kelvin = np.clip(temp + 273.15, 250.0, 350.0)
    exponent = (ea_j_mol / R_GAS) * (1.0 / REFERENCE_T_K - 1.0 / kelvin)
    return np.exp(np.clip(exponent, -20.0, 20.0))


def _canonical_exposure(
    frame: pd.DataFrame,
    *,
    duration_col: str,
    distance_col: str,
    efc_col: str,
    ah_col: str,
    ambient_col: str,
    pack_mean_col: str,
    pack_max_col: str,
    pack_spread_col: str,
    hot_col: str,
    speed_mean_col: str,
    speed_p95_col: str,
    accel_col: str,
    decel_col: str,
    jerk_col: str,
    harsh_accel_col: str,
    harsh_decel_col: str,
    c_rate_p95_col: str,
    high_c_col: str,
    c2_col: str,
    et_col: str,
    transient_col: str,
    dbsi_col: str,
    stress_col: str,
    dod_col: str,
    soc_col: str,
) -> dict[str, float]:
    return {
        "distance_km": _sum(frame, distance_col),
        "efc": _sum(frame, efc_col),
        "ah_throughput": _sum(frame, ah_col),
        "ambient_mean_c": _weighted_mean(frame, ambient_col, duration_col),
        "pack_temp_mean_c": _weighted_mean(frame, pack_mean_col, duration_col),
        "pack_temp_max_c": _max(frame, pack_max_col),
        "pack_temp_spread_c": _weighted_mean(frame, pack_spread_col, duration_col),
        "hot_exposure_fraction": _weighted_mean(frame, hot_col, duration_col, 0.0),
        "speed_mean_kmh": _weighted_mean(frame, speed_mean_col, duration_col),
        "speed_p95_kmh": _weighted_mean(frame, speed_p95_col, duration_col),
        "accel_rms_mps2": _weighted_mean(frame, accel_col, duration_col),
        "decel_rms_mps2": _weighted_mean(frame, decel_col, duration_col),
        "jerk_rms_mps3": _weighted_mean(frame, jerk_col, duration_col),
        "harsh_accel_fraction": _weighted_mean(frame, harsh_accel_col, duration_col, 0.0),
        "harsh_decel_fraction": _weighted_mean(frame, harsh_decel_col, duration_col, 0.0),
        "c_rate_p95": _weighted_mean(frame, c_rate_p95_col, duration_col),
        "high_c_rate_fraction": _weighted_mean(frame, high_c_col, duration_col, 0.0),
        "c_rate_squared_dose_h": _sum(frame, c2_col),
        "electrothermal_dose_h": _sum(frame, et_col),
        "transient_dose": _sum(frame, transient_col),
        "dbsi_mean": _weighted_mean(frame, dbsi_col, duration_col),
        "stress_dose_h": _sum(frame, stress_col),
        "dod_mean_pct": _weighted_mean(frame, dod_col, duration_col),
        "soc_mean_pct": _weighted_mean(frame, soc_col, duration_col),
        "exposure_duration_h": _sum(frame, duration_col),
        "trip_count": int(len(frame)),
    }


def build_nissan_intervals(trips_path: Path, health_path: Path) -> pd.DataFrame:
    trips = pd.read_csv(trips_path)
    health = pd.read_csv(health_path)
    for col in ("trip_start_utc", "trip_end_utc"):
        trips[col] = pd.to_datetime(trips[col], utc=True, format="mixed")
    health["anchor_time_utc"] = pd.to_datetime(
        health["anchor_time_utc"], utc=True, format="mixed"
    )

    trips = trips.sort_values("trip_start_utc").reset_index(drop=True)
    dev_trips = trips.iloc[:101].copy()
    cutoff = dev_trips["trip_end_utc"].max()
    dev_health = (
        health.loc[health["anchor_time_utc"].le(cutoff)]
        .sort_values("anchor_time_utc")
        .drop_duplicates("anchor_time_utc", keep="last")
        .reset_index(drop=True)
    )

    previous_soc_end = _numeric(dev_trips, "soc_end_pct").shift(1)
    gap_h = (
        dev_trips["trip_start_utc"] - dev_trips["trip_end_utc"].shift(1)
    ).dt.total_seconds() / 3600.0
    soc_gain = (_numeric(dev_trips, "soc_start_pct") - previous_soc_end).clip(lower=0.0)
    plausible = gap_h.between(0.05, 24.0 * 30.0) & soc_gain.ge(2.0)
    dev_trips["charge_soc_gain_pct"] = soc_gain.where(plausible, 0.0)
    dev_trips["charge_event_proxy"] = plausible.astype(float)
    dev_trips["charge_high_soc_proxy"] = (
        plausible & _numeric(dev_trips, "soc_start_pct").ge(80.0)
    ).astype(float)
    dev_trips["charge_plug_in_soc_proxy"] = _numeric(
        dev_trips, "soc_end_pct"
    ).shift(1).where(plausible)
    dev_trips["duration_h"] = _numeric(dev_trips, "duration_s") / 3600.0
    dev_trips["c_rate_p95"] = _numeric(dev_trips, "current_dis_p95_a") / 111.11
    dev_trips["high_c_rate_fraction"] = (
        dev_trips["c_rate_p95"].sub(1.0).clip(lower=0.0)
    )
    dev_trips["c_rate_squared_dose_h"] = (
        dev_trips["c_rate_p95"].fillna(0.0).pow(2) * dev_trips["duration_h"]
    )
    dev_trips["electrothermal_dose_h"] = (
        dev_trips["c_rate_squared_dose_h"]
        * _arrhenius(_numeric(dev_trips, "pack_temp_mean_c", 25.0).fillna(25.0), 35_000.0)
    )
    dev_trips["transient_dose"] = (
        _numeric(dev_trips, "bsi_transient", 0.0).fillna(0.0)
        * dev_trips["duration_h"]
    )
    dev_trips["soc_mean_pct"] = (
        _numeric(dev_trips, "soc_start_pct") + _numeric(dev_trips, "soc_end_pct")
    ) / 2.0

    rows: list[dict[str, object]] = []
    for idx in range(1, len(dev_health)):
        previous = dev_health.iloc[idx - 1]
        current = dev_health.iloc[idx]
        start = previous["anchor_time_utc"]
        end = current["anchor_time_utc"]
        exposure = dev_trips.loc[
            dev_trips["trip_end_utc"].gt(start) & dev_trips["trip_end_utc"].le(end)
        ].copy()
        if exposure.empty:
            continue
        row: dict[str, object] = {
            "dataset": "Nissan-development",
            "vehicle_id": "leaf_01",
            "interval_id": f"Nissan:{idx - 1}-{idx}",
            "interval_start": start,
            "interval_end": end,
            "calendar_days": _safe_days(start, end),
            "signed_loss_pp": float(
                pd.to_numeric(previous["soh_target_trend_pct"], errors="coerce")
                - pd.to_numeric(current["soh_target_trend_pct"], errors="coerce")
            ),
            "charging_observation_status": "inferred_from_intertrip_SOC_gain_not_measured",
            "charge_event_count": _sum(exposure, "charge_event_proxy"),
            "charge_depth_pct": _sum(exposure, "charge_soc_gain_pct"),
            "charge_high_soc_count": _sum(exposure, "charge_high_soc_proxy"),
            "charge_plug_in_soc_mean_pct": float(
                _numeric(exposure, "charge_plug_in_soc_proxy").mean()
            ),
        }
        row.update(
            _canonical_exposure(
                exposure,
                duration_col="duration_h",
                distance_col="distance_km",
                efc_col="efc",
                ah_col="ah_throughput",
                ambient_col="ambient_mean_c",
                pack_mean_col="pack_temp_mean_c",
                pack_max_col="pack_temp_max_c",
                pack_spread_col="pack_temp_spread_p95_c",
                hot_col="hot_time_fraction",
                speed_mean_col="speed_mean_kmh",
                speed_p95_col="speed_p95_kmh",
                accel_col="accel_pos_rms",
                decel_col="decel_abs_rms",
                jerk_col="jerk_rms",
                harsh_accel_col="harsh_accel_per_min",
                harsh_decel_col="harsh_brake_per_min",
                c_rate_p95_col="c_rate_p95",
                high_c_col="high_c_rate_fraction",
                c2_col="c_rate_squared_dose_h",
                et_col="electrothermal_dose_h",
                transient_col="transient_dose",
                dbsi_col="bsi_intensity",
                stress_col="stress_dose",
                dod_col="dod_pct",
                soc_col="soc_mean_pct",
            )
        )
        row["positive_loss_pp"] = max(float(row["signed_loss_pp"]), 0.0)
        row["temperature_observation_status"] = "measured_pack_sensors"
        rows.append(row)
    return pd.DataFrame(rows)


def _load_rwth_charge_level(directory: Path) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*_charge.parquet")):
        table = pd.read_parquet(path)
        table["source_file"] = path.name
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    charge = pd.concat(tables, ignore_index=True)
    charge["starttime"] = pd.to_datetime(charge["starttime"])
    charge["endtime"] = charge["starttime"] + pd.to_timedelta(
        pd.to_numeric(charge["duration_seconds"], errors="coerce"), unit="s"
    )
    return charge


def build_rwth_intervals(
    trips_path: Path, checkpoints_path: Path, metadata_dir: Path
) -> pd.DataFrame:
    trips = pd.read_csv(trips_path)
    checkpoints = pd.read_csv(checkpoints_path)
    charge = _load_rwth_charge_level(metadata_dir)
    trips["trip_start"] = pd.to_datetime(trips["trip_start"], format="mixed")
    trips["trip_end"] = pd.to_datetime(trips["trip_end"], format="mixed")
    checkpoints["checkpoint_start"] = pd.to_datetime(
        checkpoints["checkpoint_start"], format="mixed"
    )
    checkpoints["checkpoint_end"] = pd.to_datetime(
        checkpoints["checkpoint_end"], format="mixed"
    )
    trips["duration_h"] = _numeric(trips, "duration_observed_h")
    trips["c_rate_p95"] = _numeric(trips, "c_rate_discharge_peak")
    trips["transient_dose"] = (
        _numeric(trips, "bsi_transient", 0.0).fillna(0.0) * trips["duration_h"]
    )
    trips["soc_mean_for_model"] = _numeric(trips, "soc_mean_pct")

    rows: list[dict[str, object]] = []
    for vehicle_id, group in checkpoints.groupby("vehicle_id", sort=True):
        group = group.sort_values("checkpoint_start").reset_index(drop=True)
        for idx in range(1, len(group)):
            previous = group.iloc[idx - 1]
            current = group.iloc[idx]
            start = previous["checkpoint_end"]
            end = current["checkpoint_start"]
            exposure = trips.loc[
                trips["vehicle_id"].eq(vehicle_id)
                & trips["trip_end"].gt(start)
                & trips["trip_end"].le(end)
            ].copy()
            if exposure.empty:
                continue
            charge_exposure = charge.loc[
                charge["vehicle"].astype(str).eq(str(vehicle_id))
                & charge["endtime"].gt(start)
                & charge["endtime"].le(end)
            ].copy()
            plug_soc = _numeric(charge_exposure, "soc_stop")
            start_soc = _numeric(charge_exposure, "soc_start")
            charge_depth = (start_soc - plug_soc).clip(lower=0.0)
            row: dict[str, object] = {
                "dataset": "RWTH",
                "vehicle_id": str(vehicle_id),
                "interval_id": f"{vehicle_id}:{idx - 1}-{idx}",
                "interval_start": start,
                "interval_end": end,
                "calendar_days": _safe_days(start, end),
                "signed_loss_pp": float(
                    previous["capacity_soh_pct"] - current["capacity_soh_pct"]
                ),
                "positive_loss_pp": max(
                    float(previous["capacity_soh_pct"] - current["capacity_soh_pct"]),
                    0.0,
                ),
                "charging_observation_status": (
                    "charge_to_charge_metadata_observed_process_power_unobserved"
                ),
                "charge_event_count": int(len(charge_exposure)),
                "charge_depth_pct": float(charge_depth.sum()),
                "charge_high_soc_count": int(start_soc.ge(90.0).sum()),
                "charge_plug_in_soc_mean_pct": float(plug_soc.mean()),
                "temperature_observation_status": (
                    "ambient_measured_pack_temperature_unobserved"
                ),
            }
            row.update(
                _canonical_exposure(
                    exposure,
                    duration_col="duration_h",
                    distance_col="distance_km",
                    efc_col="efc",
                    ah_col="ah_throughput",
                    ambient_col="ambient_temp_mean_c",
                    pack_mean_col="ambient_temp_mean_c",
                    pack_max_col="ambient_temp_mean_c",
                    pack_spread_col="__missing_pack_spread",
                    hot_col="__missing_hot",
                    speed_mean_col="speed_mean_kmh",
                    speed_p95_col="speed_max_kmh",
                    accel_col="accel_positive_rms_mps2",
                    decel_col="decel_rms_mps2",
                    jerk_col="jerk_rms_mps3",
                    harsh_accel_col="harsh_accel_fraction",
                    harsh_decel_col="harsh_decel_fraction",
                    c_rate_p95_col="c_rate_p95",
                    high_c_col="high_c_rate_fraction",
                    c2_col="joule_c2_dose_h",
                    et_col="electrothermal_proxy_dose_h",
                    transient_col="transient_dose",
                    dbsi_col="dbsi_partial_renormalized",
                    stress_col="__missing_stress_dose",
                    dod_col="dod_pct",
                    soc_col="soc_mean_for_model",
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_eve_intervals(trips_path: Path, intervals_path: Path) -> pd.DataFrame:
    trips = pd.read_csv(trips_path)
    intervals = pd.read_csv(intervals_path)
    trips["trip_start"] = pd.to_datetime(trips["trip_start"], format="mixed")
    trips["trip_end"] = pd.to_datetime(trips["trip_end"], format="mixed")
    intervals["checkpoint_time"] = pd.to_datetime(
        intervals["checkpoint_time"], format="mixed"
    )
    intervals = intervals.sort_values("checkpoint_time").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx in range(1, len(intervals)):
        previous = intervals.iloc[idx - 1]
        current = intervals.iloc[idx]
        start = previous["checkpoint_time"]
        end = current["checkpoint_time"]
        exposure = trips.loc[
            trips["trip_end"].gt(start) & trips["trip_end"].le(end)
        ].copy()
        if exposure.empty:
            continue
        exposure["duration_for_model_h"] = _numeric(exposure, "duration_h")
        exposure["transient_for_model"] = (
            _numeric(exposure, "transient_dose", 0.0).fillna(0.0)
            * exposure["duration_for_model_h"]
        )
        exposure["soc_for_model"] = _numeric(exposure, "soc_mean_pct")
        row: dict[str, object] = {
            "dataset": "EVE",
            "vehicle_id": "Renault-Zoe-Q90-1",
            "interval_id": f"EVE:{idx - 1}-{idx}",
            "interval_start": start,
            "interval_end": end,
            "calendar_days": _safe_days(start, end),
            "signed_loss_pp": float(current["signed_trend_loss_pp"]),
            "positive_loss_pp": max(float(current["signed_trend_loss_pp"]), 0.0),
            "charging_observation_status": "unavailable",
            "charge_event_count": np.nan,
            "charge_depth_pct": np.nan,
            "charge_high_soc_count": np.nan,
            "charge_plug_in_soc_mean_pct": np.nan,
            "temperature_observation_status": "measured_module_temperatures",
        }
        row.update(
            _canonical_exposure(
                exposure,
                duration_col="duration_for_model_h",
                distance_col="distance_km",
                efc_col="efc",
                ah_col="ah_throughput",
                ambient_col="ambient_mean_c",
                pack_mean_col="pack_temp_mean_c",
                pack_max_col="pack_temp_max_c",
                pack_spread_col="pack_temp_spread_mean_c",
                hot_col="high_soc_hot_time_h",
                speed_mean_col="speed_mean_kmh",
                speed_p95_col="speed_p95_kmh",
                accel_col="accel_positive_rms_mps2",
                decel_col="decel_rms_mps2",
                jerk_col="jerk_rms_mps3",
                harsh_accel_col="harsh_accel_fraction",
                harsh_decel_col="harsh_decel_fraction",
                c_rate_p95_col="c_rate_abs_p95",
                high_c_col="high_c_rate_fraction",
                c2_col="c_rate_squared_dose_h",
                et_col="electrothermal_dose_h",
                transient_col="transient_for_model",
                dbsi_col="dbsi",
                stress_col="stress_dose_h",
                dod_col="__missing_dod",
                soc_col="soc_for_model",
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _expanding_splits(n_rows: int, folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < 8:
        return []
    initial = max(6, int(math.floor(n_rows * 0.40)))
    remaining = n_rows - initial
    chunks = [chunk for chunk in np.array_split(np.arange(initial, n_rows), folds) if len(chunk)]
    return [(np.arange(0, int(chunk[0])), chunk) for chunk in chunks]


def _outer_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if frame["dataset"].iloc[0] == "RWTH":
        result = []
        for vehicle in sorted(frame["vehicle_id"].astype(str).unique()):
            test = np.flatnonzero(frame["vehicle_id"].astype(str).eq(vehicle).to_numpy())
            train = np.flatnonzero(~frame["vehicle_id"].astype(str).eq(vehicle).to_numpy())
            if len(test) and len(train) >= 4:
                result.append((train, test, vehicle))
        return result
    return [
        (train, test, f"time_fold_{idx + 1}")
        for idx, (train, test) in enumerate(_expanding_splits(len(frame)))
    ]


def _feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=frame.index)
    for feature in features:
        matrix[feature] = _numeric(frame, feature)
    return matrix


def _fit_ridge(
    train: pd.DataFrame,
    y_train: np.ndarray,
    test: pd.DataFrame,
    features: Sequence[str],
    alpha: float = 10.0,
) -> np.ndarray:
    if not features:
        return np.zeros(len(test), dtype=float)
    x_train = _feature_matrix(train, features)
    x_test = _feature_matrix(test, features)
    available = [
        feature
        for feature in features
        if x_train[feature].notna().any() and x_train[feature].nunique(dropna=True) > 1
    ]
    if not available:
        return np.repeat(float(np.nanmean(y_train)), len(test))
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=False)),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(x_train[available], y_train)
    return model.predict(x_test[available])


def evaluate_nested_attribution(
    frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    for raw in frames:
        frame = raw.sort_values("interval_end").reset_index(drop=True)
        if frame.empty:
            continue
        for train_idx, test_idx, fold in _outer_splits(frame):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            y_train = _numeric(train, "signed_loss_pp").to_numpy(dtype=float)
            for model_name, features in NESTED_BLOCKS:
                prediction = _fit_ridge(train, y_train, test, features)
                for local, pred in enumerate(prediction):
                    source = test.iloc[local]
                    prediction_rows.append(
                        {
                            "dataset": source["dataset"],
                            "vehicle_id": source["vehicle_id"],
                            "interval_id": source["interval_id"],
                            "outer_fold": fold,
                            "model": model_name,
                            "observed_signed_loss_pp": float(source["signed_loss_pp"]),
                            "predicted_signed_loss_pp": float(pred),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, object]] = []
    if not predictions.empty:
        for (dataset, model), group in predictions.groupby(["dataset", "model"]):
            truth = group["observed_signed_loss_pp"].to_numpy()
            pred = group["predicted_signed_loss_pp"].to_numpy()
            metric_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n": len(group),
                    "mae_pp": mean_absolute_error(truth, pred),
                    "rmse_pp": math.sqrt(mean_squared_error(truth, pred)),
                    "bias_pp": float(np.mean(pred - truth)),
                    "r2": r2_score(truth, pred) if len(group) > 1 else np.nan,
                }
            )
    metrics = pd.DataFrame(metric_rows)

    increments: list[dict[str, object]] = []
    order = [name for name, _ in NESTED_BLOCKS[:6]]
    for dataset, group in predictions.groupby("dataset"):
        wide = group.pivot_table(
            index=["vehicle_id", "interval_id"],
            columns="model",
            values=["observed_signed_loss_pp", "predicted_signed_loss_pp"],
            aggfunc="first",
        )
        truth = wide["observed_signed_loss_pp"].iloc[:, 0].to_numpy()
        previous = order[0]
        for current in order[1:]:
            pred_previous = wide["predicted_signed_loss_pp"][previous].to_numpy()
            pred_current = wide["predicted_signed_loss_pp"][current].to_numpy()
            sse_previous = float(np.sum((truth - pred_previous) ** 2))
            sse_current = float(np.sum((truth - pred_current) ** 2))
            residual_fraction = (
                1.0 - sse_current / sse_previous if sse_previous > 0 else np.nan
            )
            increments.append(
                {
                    "dataset": dataset,
                    "added_block": current.replace(previous, "").lstrip("+")
                    if current.startswith(previous)
                    else current,
                    "reference_model": previous,
                    "challenger_model": current,
                    "n": len(truth),
                    "delta_mae_pp": mean_absolute_error(truth, pred_current)
                    - mean_absolute_error(truth, pred_previous),
                    "delta_r2": (
                        r2_score(truth, pred_current) - r2_score(truth, pred_previous)
                    ),
                    "heldout_residual_sse_fraction_removed": residual_fraction,
                }
            )
            previous = current
    return predictions, metrics, pd.DataFrame(increments)


def _moving_block_bootstrap_delta(
    group: pd.DataFrame,
    *,
    reference: str,
    challenger: str,
    seed: int = 20260723,
    iterations: int = 10_000,
) -> dict[str, float]:
    wide = group.pivot_table(
        index=["vehicle_id", "interval_id"],
        columns="model",
        values=["observed_signed_loss_pp", "predicted_signed_loss_pp"],
        aggfunc="first",
    ).reset_index()
    truth = wide["observed_signed_loss_pp"].iloc[:, 0].to_numpy(dtype=float)
    ref = wide["predicted_signed_loss_pp"][reference].to_numpy(dtype=float)
    chal = wide["predicted_signed_loss_pp"][challenger].to_numpy(dtype=float)
    n = len(truth)
    observed = float(np.mean(np.abs(truth - chal) - np.abs(truth - ref)))
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    vehicle_values = wide["vehicle_id"]
    if isinstance(vehicle_values, pd.DataFrame):
        vehicle_values = vehicle_values.iloc[:, 0]
    vehicles = vehicle_values.astype(str).to_numpy()
    unique_vehicles = np.unique(vehicles)
    if len(unique_vehicles) > 1:
        clusters = {v: np.flatnonzero(vehicles == v) for v in unique_vehicles}
        for i in range(iterations):
            sampled = rng.choice(unique_vehicles, size=len(unique_vehicles), replace=True)
            idx = np.concatenate([clusters[v] for v in sampled])
            values[i] = np.mean(np.abs(truth[idx] - chal[idx]) - np.abs(truth[idx] - ref[idx]))
    else:
        block = min(3, n)
        for i in range(iterations):
            idx_parts: list[np.ndarray] = []
            while sum(len(x) for x in idx_parts) < n:
                start = int(rng.integers(0, max(n - block + 1, 1)))
                idx_parts.append(np.arange(start, min(start + block, n)))
            idx = np.concatenate(idx_parts)[:n]
            values[i] = np.mean(np.abs(truth[idx] - chal[idx]) - np.abs(truth[idx] - ref[idx]))
    return {
        "delta_mae_pp": observed,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_challenger_better": float(np.mean(values < 0.0)),
    }


def driver_increment_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        result = _moving_block_bootstrap_delta(
            group,
            reference="Calendar+Usage+Charging+Temperature",
            challenger="Full+DriverResidual",
        )
        rows.append(
            {
                "dataset": dataset,
                "reference": "Calendar+Usage+Charging+Temperature",
                "challenger": "Full+DriverResidual",
                **result,
            }
        )
    return pd.DataFrame(rows)


def source_attribution_summary(
    metrics: pd.DataFrame,
    increments: pd.DataFrame,
    driver_ci: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_lookup = metrics.set_index(["dataset", "model"])
    driver_lookup = driver_ci.set_index("dataset")
    for _, increment in increments.iterrows():
        dataset = str(increment["dataset"])
        reference = str(increment["reference_model"])
        challenger = str(increment["challenger_model"])
        reference_mae = float(metric_lookup.loc[(dataset, reference), "mae_pp"])
        delta_mae = float(increment["delta_mae_pp"])
        block = str(increment["added_block"])
        row: dict[str, object] = {
            "dataset": dataset,
            "block_added_last": block,
            "reference_model": reference,
            "challenger_model": challenger,
            "n": int(increment["n"]),
            "delta_mae_pp": delta_mae,
            "relative_mae_change_pct": (
                100.0 * delta_mae / reference_mae if reference_mae > 0 else np.nan
            ),
            "heldout_residual_sse_fraction_removed": float(
                increment["heldout_residual_sse_fraction_removed"]
            ),
            "heldout_residual_sse_percent_removed": float(
                100.0 * increment["heldout_residual_sse_fraction_removed"]
            ),
            "delta_r2": float(increment["delta_r2"]),
            "charging_data_status": (
                "unavailable"
                if dataset == "EVE" and block == "Charging"
                else "available_with_dataset_specific_limitations"
            ),
            "order_dependent": True,
        }
        if block == "Full+DriverResidual" and dataset in driver_lookup.index:
            ci = driver_lookup.loc[dataset]
            row["paired_ci95_low"] = float(ci["ci95_low"])
            row["paired_ci95_high"] = float(ci["ci95_high"])
            row["paired_mae_improvement_supported"] = bool(ci["ci95_high"] < 0.0)
            row["positive_heldout_variance_removed"] = bool(
                increment["heldout_residual_sse_fraction_removed"] > 0.0
            )
        else:
            row["paired_ci95_low"] = np.nan
            row["paired_ci95_high"] = np.nan
            row["paired_mae_improvement_supported"] = np.nan
            row["positive_heldout_variance_removed"] = bool(
                increment["heldout_residual_sse_fraction_removed"] > 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_pack_se_nmc_stress(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic Pack-SE-NMC-v1 stress terms.

    This is an explicit, pack-calibrated sensitivity law.  It deliberately
    does not copy cell-specific coefficients from Schmalstieg/Ecker.
    Coefficients mapping stress to capacity loss are fitted inside each
    training fold.
    """

    out = frame.copy()
    days = _numeric(out, "calendar_days", 0.0).fillna(0.0).clip(lower=0.0)
    temp = _numeric(out, "pack_temp_mean_c")
    temp = temp.fillna(_numeric(out, "ambient_mean_c", 25.0)).fillna(25.0)
    soc = _numeric(out, "soc_mean_pct", 50.0).fillna(50.0).clip(0.0, 100.0)
    efc = _numeric(out, "efc", 0.0).fillna(0.0).clip(lower=0.0)
    dod = _numeric(out, "dod_mean_pct", 50.0).fillna(50.0).clip(1.0, 100.0)
    c2 = _numeric(out, "c_rate_squared_dose_h", 0.0).fillna(0.0).clip(lower=0.0)
    duration = _numeric(out, "exposure_duration_h", 1.0).fillna(1.0).clip(lower=1e-6)
    charge_count = _numeric(out, "charge_event_count", 0.0).fillna(0.0).clip(lower=0.0)
    charge_depth = _numeric(out, "charge_depth_pct", 0.0).fillna(0.0).clip(lower=0.0)
    plug_soc = _numeric(out, "charge_plug_in_soc_mean_pct", 50.0).fillna(50.0)
    spread = _numeric(out, "pack_temp_spread_c")
    measured_spread = spread.notna()
    spread = spread.fillna(2.0).clip(lower=0.0)
    transient = _numeric(out, "transient_dose", 0.0).fillna(0.0).clip(lower=0.0)

    calendar = (
        np.sqrt(days / 365.25)
        * _arrhenius(temp, 35_000.0)
        * (1.0 + 1.5 * np.square(soc / 100.0))
    )
    cycle = (
        efc
        * _arrhenius(temp, 25_000.0)
        * np.power(dod / 80.0, 1.3)
        * (1.0 + 0.5 * c2 / duration)
    )
    charging = (
        charge_count
        * (charge_depth / np.maximum(charge_count, 1.0))
        / 100.0
        * (1.0 + np.square(np.clip((plug_soc - 80.0) / 20.0, 0.0, 1.0)))
    )
    imbalance = 1.0 + 0.5 * np.square(spread / 10.0)
    transient_term = np.sqrt(transient + 1e-12)

    out["packse_calendar_stress"] = calendar
    out["packse_cycle_stress"] = cycle * imbalance
    out["packse_charging_stress"] = charging
    out["packse_transient_stress"] = transient_term
    out["packse_homogeneous_damage_index"] = calendar + cycle + charging + transient_term
    out["packse_pack_corrected_damage_index"] = (
        calendar + cycle * imbalance + charging + transient_term
    )
    out["pack_temperature_spread_measured"] = measured_spread
    return out


PACKSE_FEATURES = [
    "packse_calendar_stress",
    "packse_cycle_stress",
    "packse_charging_stress",
    "packse_transient_stress",
]


def _fit_nnls_packse(
    train: pd.DataFrame, test: pd.DataFrame, target: str = "positive_loss_pp"
) -> np.ndarray:
    x_train = _feature_matrix(train, PACKSE_FEATURES).fillna(0.0).to_numpy(dtype=float)
    x_test = _feature_matrix(test, PACKSE_FEATURES).fillna(0.0).to_numpy(dtype=float)
    y_train = _numeric(train, target, 0.0).fillna(0.0).clip(lower=0.0).to_numpy()
    scales = np.nanmedian(np.abs(x_train), axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
    coefficients, _ = nnls(x_train / scales, y_train)
    return (x_test / scales) @ coefficients


def evaluate_cpu_comparators(
    frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for raw in frames:
        frame = add_pack_se_nmc_stress(raw.sort_values("interval_end").reset_index(drop=True))
        if frame.empty:
            continue
        for train_idx, test_idx, fold in _outer_splits(frame):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            y_train = _numeric(train, "positive_loss_pp").fillna(0.0).to_numpy()
            models = {
                "Persistence-zero-loss": np.zeros(len(test)),
                (
                    "D2B-style-PackSE-cumulative-physics"
                    "-NOT-an-exact-D2B-reproduction"
                ): _fit_nnls_packse(train, test),
                (
                    "EV-DOS-style-physics-Ridge"
                    "-NOT-an-EV-DOS-LSTM-reproduction"
                ): np.clip(
                    _fit_ridge(
                        train,
                        y_train,
                        test,
                        CALENDAR + USAGE + CHARGING + TEMPERATURE + DRIVER,
                    ),
                    0.0,
                    None,
                ),
                "DBSI-concept-Ridge-CPU-sensitivity": np.clip(
                    _fit_ridge(
                        train,
                        y_train,
                        test,
                        CALENDAR
                        + USAGE
                        + CHARGING
                        + TEMPERATURE
                        + ["dbsi_mean", "stress_dose_h"],
                    ),
                    0.0,
                    None,
                ),
            }
            for model, pred in models.items():
                for local, value in enumerate(pred):
                    source = test.iloc[local]
                    rows.append(
                        {
                            "dataset": source["dataset"],
                            "vehicle_id": source["vehicle_id"],
                            "interval_id": source["interval_id"],
                            "outer_fold": fold,
                            "model": model,
                            "observed_positive_loss_pp": float(source["positive_loss_pp"]),
                            "predicted_positive_loss_pp": float(value),
                        }
                    )
    predictions = pd.DataFrame(rows)
    metric_rows = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        truth = group["observed_positive_loss_pp"].to_numpy()
        pred = group["predicted_positive_loss_pp"].to_numpy()
        metric_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(group),
                "mae_pp": mean_absolute_error(truth, pred),
                "rmse_pp": math.sqrt(mean_squared_error(truth, pred)),
                "bias_pp": float(np.mean(pred - truth)),
                "r2": r2_score(truth, pred) if len(group) > 1 else np.nan,
            }
        )
    return predictions, pd.DataFrame(metric_rows)


def pack_uncertainty_analysis(
    frames: Iterable[pd.DataFrame],
    *,
    simulations: int = 2_000,
    seed: int = 20260723,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    interval_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    sample_parameters = [
        "ea_calendar_kj_mol",
        "ea_cycle_kj_mol",
        "temperature_bias_c",
        "capacity_scale",
        "pack_heterogeneity",
        "soc_bias_pct",
    ]
    for raw in frames:
        deterministic = add_pack_se_nmc_stress(raw)
        for row_idx, row in deterministic.iterrows():
            days = max(float(row.get("calendar_days", 0.0) or 0.0), 0.0)
            temp = row.get("pack_temp_mean_c", np.nan)
            if not np.isfinite(temp):
                temp = row.get("ambient_mean_c", 25.0)
            temp = 25.0 if not np.isfinite(temp) else float(temp)
            soc = row.get("soc_mean_pct", 50.0)
            soc = 50.0 if not np.isfinite(soc) else float(soc)
            efc = row.get("efc", 0.0)
            efc = 0.0 if not np.isfinite(efc) else max(float(efc), 0.0)
            dod = row.get("dod_mean_pct", 50.0)
            dod = 50.0 if not np.isfinite(dod) else float(np.clip(dod, 1.0, 100.0))
            c2 = row.get("c_rate_squared_dose_h", 0.0)
            c2 = 0.0 if not np.isfinite(c2) else max(float(c2), 0.0)
            duration = row.get("exposure_duration_h", 1.0)
            duration = 1.0 if not np.isfinite(duration) else max(float(duration), 1e-6)
            charge_count = row.get("charge_event_count", 0.0)
            charge_count = 0.0 if not np.isfinite(charge_count) else max(float(charge_count), 0.0)
            charge_depth = row.get("charge_depth_pct", 0.0)
            charge_depth = 0.0 if not np.isfinite(charge_depth) else max(float(charge_depth), 0.0)
            plug_soc = row.get("charge_plug_in_soc_mean_pct", 50.0)
            plug_soc = 50.0 if not np.isfinite(plug_soc) else float(plug_soc)
            spread = row.get("pack_temp_spread_c", np.nan)
            spread_measured = bool(np.isfinite(spread))
            spread = 2.0 if not spread_measured else max(float(spread), 0.0)
            transient = row.get("transient_dose", 0.0)
            transient = 0.0 if not np.isfinite(transient) else max(float(transient), 0.0)

            ea_cal = np.clip(rng.normal(35.0, 5.0, simulations), 20.0, 50.0)
            ea_cycle = np.clip(rng.normal(25.0, 5.0, simulations), 10.0, 45.0)
            temp_bias = np.clip(rng.normal(0.0, 2.0, simulations), -6.0, 6.0)
            capacity_scale = np.clip(rng.normal(1.0, 0.05, simulations), 0.85, 1.15)
            heterogeneity = np.clip(rng.normal(1.0, 0.10, simulations), 0.70, 1.30)
            soc_bias = np.clip(rng.normal(0.0, 3.0, simulations), -9.0, 9.0)
            uncertain_spread = (
                np.clip(rng.normal(spread, 0.5, simulations), 0.0, 12.0)
                if spread_measured
                else np.clip(rng.normal(2.0, 1.5, simulations), 0.0, 10.0)
            )
            t = temp + temp_bias
            s = np.clip(soc + soc_bias, 0.0, 100.0)
            calendar = (
                np.sqrt(days / 365.25)
                * _arrhenius(t, ea_cal * 1000.0)
                * (1.0 + 1.5 * np.square(s / 100.0))
            )
            cycle = (
                (efc / capacity_scale)
                * _arrhenius(t, ea_cycle * 1000.0)
                * np.power(dod / 80.0, 1.3)
                * (1.0 + 0.5 * c2 / duration)
            )
            charge_stress = (
                charge_count
                * (charge_depth / max(charge_count, 1.0))
                / 100.0
                * (1.0 + np.square(np.clip((plug_soc - 80.0) / 20.0, 0.0, 1.0)))
            )
            pack_factor = heterogeneity * (
                1.0 + 0.5 * np.square(uncertain_spread / 10.0)
            )
            damage = calendar + cycle * pack_factor + charge_stress + math.sqrt(transient + 1e-12)
            homogeneous = float(row["packse_homogeneous_damage_index"])
            p05, p50, p95 = np.quantile(damage, [0.05, 0.50, 0.95])
            interval_rows.append(
                {
                    "dataset": row["dataset"],
                    "vehicle_id": row["vehicle_id"],
                    "interval_id": row["interval_id"],
                    "simulations": simulations,
                    "pack_temperature_spread_measured": spread_measured,
                    "homogeneous_damage_index": homogeneous,
                    "pack_damage_p05": float(p05),
                    "pack_damage_p50": float(p50),
                    "pack_damage_p95": float(p95),
                    "relative_90pct_width": float((p95 - p05) / max(abs(p50), 1e-9)),
                    "median_pack_correction_pct": float(
                        100.0 * (p50 - homogeneous) / max(abs(homogeneous), 1e-9)
                    ),
                }
            )
            parameter_values = {
                "ea_calendar_kj_mol": ea_cal,
                "ea_cycle_kj_mol": ea_cycle,
                "temperature_bias_c": temp_bias,
                "capacity_scale": capacity_scale,
                "pack_heterogeneity": heterogeneity,
                "soc_bias_pct": soc_bias,
            }
            for parameter in sample_parameters:
                rho = spearmanr(parameter_values[parameter], damage).statistic
                sensitivity_rows.append(
                    {
                        "dataset": row["dataset"],
                        "interval_id": row["interval_id"],
                        "parameter": parameter,
                        "spearman_rho": float(rho),
                        "absolute_spearman_rho": abs(float(rho)),
                    }
                )

    intervals = pd.DataFrame(interval_rows)
    sensitivities = pd.DataFrame(sensitivity_rows)
    summary = (
        intervals.groupby("dataset")
        .agg(
            intervals=("interval_id", "count"),
            measured_spread_fraction=("pack_temperature_spread_measured", "mean"),
            median_relative_90pct_width=("relative_90pct_width", "median"),
            median_pack_correction_pct=("median_pack_correction_pct", "median"),
        )
        .reset_index()
    )
    return intervals, sensitivities, summary


def pack_sensitivity_summary(sensitivities: pd.DataFrame) -> pd.DataFrame:
    summary = (
        sensitivities.groupby(["dataset", "parameter"])
        .agg(
            intervals=("interval_id", "count"),
            median_spearman_rho=("spearman_rho", "median"),
            median_absolute_spearman_rho=("absolute_spearman_rho", "median"),
            p90_absolute_spearman_rho=(
                "absolute_spearman_rho",
                lambda values: float(np.quantile(values, 0.90)),
            ),
        )
        .reset_index()
    )
    summary["sensitivity_rank"] = summary.groupby("dataset")[
        "median_absolute_spearman_rho"
    ].rank(method="dense", ascending=False)
    return summary.sort_values(["dataset", "sensitivity_rank", "parameter"])


def comparison_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "criterion": "Primary evidence",
                "D2B": "Generated multi-year daily scenarios from travel demand and learned driving policies",
                "EV_DOS": "One year of real second-by-second operation plus hybrid simulation",
                "DBPatchKAN": "Real telemetry from Nissan, nine RWTH vehicles, and EVE",
            },
            {
                "criterion": "Driver representation",
                "D2B": "High-level PPO driving policy and generated cycles",
                "EV_DOS": "Dynamic operating features consumed by an LSTM",
                "DBPatchKAN": "Dedicated raw driver stream plus discovered trip behaviour",
            },
            {
                "criterion": "Battery physics",
                "D2B": "Powertrain/energy-system scenario simulation with calendar and cycling aging",
                "EV_DOS": "Energy transfer, heat transfer, charge/discharge, and state-estimation modules",
                "DBPatchKAN": "Measured pack response, D-BSI, electrothermal dose, and Pack-SE-NMC sensitivity",
            },
            {
                "criterion": "Interpretability interface",
                "D2B": "Scenario-level comparison of driving policies",
                "EV_DOS": "Hybrid physical modules; LSTM health head",
                "DBPatchKAN": "Named D-BSI concepts and inspectable KAN splines",
            },
            {
                "criterion": "Forecast anchor",
                "D2B": "Long-horizon simulated health trajectory",
                "EV_DOS": "Direct LSTM SOH estimate",
                "DBPatchKAN": "Strict persistence for current SOH; anchor minus non-negative loss for future SOH",
            },
            {
                "criterion": "Persistence baseline",
                "D2B": "Not identified in accessible primary-paper summary",
                "EV_DOS": "Not identified in accessible primary-paper summary",
                "DBPatchKAN": "Mandatory on Nissan, RWTH, and EVE; significant locked 500-km improvement",
            },
            {
                "criterion": "External validation",
                "D2B": "Scenario/region comparison",
                "EV_DOS": "Reported real-operation validation",
                "DBPatchKAN": "RWTH leave-one-vehicle-out and EVE chronological validation",
            },
            {
                "criterion": "Charging information",
                "D2B": "Fixed nightly Level-1 charging in the described scenarios",
                "EV_DOS": "Charging rate, depth, V2G, and HVAC experiments",
                "DBPatchKAN": "RWTH plug-in metadata; Nissan SOC-gain proxy; EVE charging unavailable",
            },
            {
                "criterion": "Exact reproduction status",
                "D2B": "Not reproduced: simulator/DRL environment and ESM parameters unavailable locally",
                "EV_DOS": "Not reproduced: exact physical modules, LSTM data, and training code unavailable locally",
                "DBPatchKAN": "Complete local code and archived Stage 0-8 artifacts",
            },
        ]
    )


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No eligible rows."
    clean = frame.copy()
    clean.columns = [str(column).replace("|", "\\|") for column in clean.columns]
    rows = [
        "| " + " | ".join(clean.columns) + " |",
        "| " + " | ".join("---" for _ in clean.columns) + " |",
    ]
    for _, record in clean.iterrows():
        values = []
        for value in record:
            if pd.isna(value):
                rendered = "NA"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _metrics_markdown(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "No eligible rows."
    view = metrics.copy()
    for col in ("mae_pp", "rmse_pp", "bias_pp", "r2"):
        view[col] = view[col].map(lambda value: f"{value:.5f}" if np.isfinite(value) else "NA")
    return _frame_to_markdown(view)


def _write_report(
    output: Path,
    counts: dict[str, int],
    nested_metrics: pd.DataFrame,
    increments: pd.DataFrame,
    driver_ci: pd.DataFrame,
    comparator_metrics: pd.DataFrame,
    pack_summary: pd.DataFrame,
) -> None:
    report = f"""# Stage 9 CPU Attribution and Pack-Level Sensitivity

Generated: {_utc_now()}

## Scope and lock discipline

The Nissan locked test was not opened. Nissan analyses use only the first 101
chronological trips and their eligible development health intervals. RWTH uses
leave-one-vehicle-out folds. EVE uses expanding chronological folds.

## Eligible interval counts

{_frame_to_markdown(pd.DataFrame([counts]))}

## Nested signed-loss attribution

{_metrics_markdown(nested_metrics)}

## Sequential contribution

{_frame_to_markdown(increments)}

The driver contribution is the final step from
`Calendar+Usage+Charging+Temperature` to `Full+DriverResidual`. Negative
delta-MAE is beneficial. `heldout_residual_sse_fraction_removed` is the fraction
of the previous held-out squared residual removed after adding the block; it can
be negative.

## Driver residual paired uncertainty

{_frame_to_markdown(driver_ci)}

## CPU comparator sensitivity

{_metrics_markdown(comparator_metrics)}

The D2B-style and EV-DOS-style rows are matched CPU surrogates, not exact
reproductions. Their purpose is to isolate cumulative physics and hybrid
physics-feature designs on exactly the same folds.

## Pack-SE-NMC-v1 uncertainty

{_frame_to_markdown(pack_summary)}

Pack-SE-NMC-v1 is a sensitivity model with explicit calendar, cycling,
charging-proxy, transient, Arrhenius, and pack-heterogeneity terms. It does not
transfer cell-specific Schmalstieg/Ecker coefficients directly. Mapping from
stress to loss is fitted inside each training fold with non-negative
coefficients. The uncertainty analysis varies activation energies, temperature
bias, usable capacity, SOC bias, and pack heterogeneity.

## Charging observability

- Nissan: inferred SOC gain between trips; charging current and power are not
  observed.
- RWTH: charge-to-charge mobility intervals identify plug-in events and SOC at
  plug-in, but the field logger explicitly excludes the charging process.
- EVE: no defensible charging-session variable in the harmonised analysis.

## Interpretation rule

Driver behaviour is interpreted as a candidate explanation of residual
variation after calendar, usage, charging proxies, and temperature. It is not
assumed to be the dominant degradation source, and a negative held-out
increment is reported rather than hidden.
"""
    (output / "STAGE9_CPU_REPORT.md").write_text(report, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(inputs: Stage9Inputs, output: Path, *, simulations: int = 2_000) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    nissan = build_nissan_intervals(inputs.nissan_trips, inputs.nissan_health)
    rwth = build_rwth_intervals(
        inputs.rwth_trips, inputs.rwth_checkpoints, inputs.rwth_trip_metadata_dir
    )
    eve = build_eve_intervals(inputs.eve_trips, inputs.eve_intervals)
    frames = [nissan, rwth, eve]

    nissan.to_csv(output / "nissan_development_attribution_intervals.csv", index=False)
    rwth.to_csv(output / "rwth_capacity_attribution_intervals.csv", index=False)
    eve.to_csv(output / "eve_monthly_attribution_intervals.csv", index=False)

    predictions, metrics, increments = evaluate_nested_attribution(frames)
    driver_ci = driver_increment_bootstrap(predictions)
    attribution_summary = source_attribution_summary(metrics, increments, driver_ci)
    comparator_predictions, comparator_metrics = evaluate_cpu_comparators(frames)
    pack_intervals, pack_sensitivity, pack_summary = pack_uncertainty_analysis(
        frames, simulations=simulations
    )
    pack_sensitivity_aggregate = pack_sensitivity_summary(pack_sensitivity)
    matrix = comparison_matrix()

    predictions.to_csv(output / "nested_attribution_predictions.csv", index=False)
    metrics.to_csv(output / "nested_attribution_metrics.csv", index=False)
    increments.to_csv(output / "nested_attribution_incremental_value.csv", index=False)
    driver_ci.to_csv(output / "driver_residual_bootstrap.csv", index=False)
    attribution_summary.to_csv(output / "source_attribution_summary.csv", index=False)
    comparator_predictions.to_csv(output / "cpu_comparator_predictions.csv", index=False)
    comparator_metrics.to_csv(output / "cpu_comparator_metrics.csv", index=False)
    pack_intervals.to_csv(output / "pack_uncertainty_intervals.csv", index=False)
    pack_sensitivity.to_csv(output / "pack_parameter_sensitivity.csv", index=False)
    pack_sensitivity_aggregate.to_csv(
        output / "pack_parameter_sensitivity_summary.csv", index=False
    )
    pack_summary.to_csv(output / "pack_uncertainty_summary.csv", index=False)
    matrix.to_csv(output / "D2B_EV_DOS_experimental_differentiation.csv", index=False)
    literature_sources = {
        "D2B": {
            "title": "Cross-temporal framework for driving behavior impact on electric vehicle battery health",
            "doi": "10.26599/COMMTR.2026.9640013",
            "url": "https://www.sciopen.com/article/10.26599/COMMTR.2026.9640013",
            "locally_reproduced": False,
            "accessible_method_facts_used": [
                "three-stage data collection, scenario generation, and scenario simulation",
                "PPO-based high-level driving policies",
                "daily driving cycles accumulated in a physics-based powertrain system",
                "calendar and cycling aging over multi-year scenarios",
                "fixed nightly Level-1 charging in the described comparison",
            ],
        },
        "EV_DOS": {
            "title": "A data-driven and physics-based model for assessing real-world usage behavior impacts on electric vehicle battery life",
            "doi": "10.1016/j.est.2025.119680",
            "url": "https://doi.org/10.1016/j.est.2025.119680",
            "locally_reproduced": False,
            "accessible_method_facts_used": [
                "real-world operating data",
                "powertrain energy-transfer and physical response modules",
                "LSTM State-of-Health module",
                "charging-rate, charging-depth, HVAC, ambient, and V2G scenarios",
            ],
        },
        "RWTH_source": {
            "title": "Battery Electric Vehicles in Commercial Fleets: Use profiles, battery aging, and open-access data",
            "doi": "10.1016/j.est.2024.111030",
            "url": "https://publications.rwth-aachen.de/record/983655/files/983655.pdf",
            "facts_used": [
                "charge-level trip starts at unplugging and ends at the next plug-in",
                "field logger excludes the charging process",
                "SOC at plug-in is observed but charging current and power are not",
            ],
        },
    }
    (output / "literature_sources.json").write_text(
        json.dumps(literature_sources, indent=2), encoding="utf-8"
    )

    counts = {
        "nissan_development_intervals": int(len(nissan)),
        "rwth_capacity_intervals": int(len(rwth)),
        "eve_monthly_intervals": int(len(eve)),
    }
    _write_report(
        output,
        counts,
        metrics,
        increments,
        driver_ci,
        comparator_metrics,
        pack_summary,
    )

    summary: dict[str, object] = {
        "created_utc": _utc_now(),
        "status": "complete",
        "locked_nissan_test_used": False,
        "counts": counts,
        "simulations_per_interval": simulations,
        "charging_observability": {
            "Nissan-development": "inferred inter-trip SOC-gain proxy",
            "RWTH": "charge-to-charge metadata; process current/power unavailable",
            "EVE": "unavailable",
        },
        "semiempirical_model": {
            "name": "Pack-SE-NMC-v1 sensitivity model",
            "cell_parameters_transferred_directly": False,
            "pack_mapping_fitted_inside_training_fold": True,
            "uncertain_parameters": [
                "calendar activation energy",
                "cycle activation energy",
                "temperature bias",
                "usable capacity scale",
                "pack heterogeneity",
                "SOC bias",
                "temperature spread",
            ],
        },
        "exact_D2B_reproduction": False,
        "exact_EV_DOS_reproduction": False,
        "reason_exact_reproduction_not_possible": (
            "The original simulation environments, supplementary physical parameters, "
            "training data, and executable code were not available in the local project."
        ),
    }
    (output / "stage9_cpu_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    inventory_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_inventory.csv":
            inventory_rows.append(
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    pd.DataFrame(inventory_rows).to_csv(output / "artifact_inventory.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nissan-trips", type=Path, required=True)
    parser.add_argument("--nissan-health", type=Path, required=True)
    parser.add_argument("--rwth-trips", type=Path, required=True)
    parser.add_argument("--rwth-checkpoints", type=Path, required=True)
    parser.add_argument("--rwth-trip-metadata-dir", type=Path, required=True)
    parser.add_argument("--eve-trips", type=Path, required=True)
    parser.add_argument("--eve-intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = Stage9Inputs(
        nissan_trips=args.nissan_trips,
        nissan_health=args.nissan_health,
        rwth_trips=args.rwth_trips,
        rwth_checkpoints=args.rwth_checkpoints,
        rwth_trip_metadata_dir=args.rwth_trip_metadata_dir,
        eve_trips=args.eve_trips,
        eve_intervals=args.eve_intervals,
    )
    summary = run(inputs, args.output_dir, simulations=args.simulations)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
