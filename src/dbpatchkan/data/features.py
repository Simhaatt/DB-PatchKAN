"""Auditable telemetry feature engineering and health-window construction."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .dbsi import DBSIConfig, compute_dbsi
from .preprocessing import NISSAN_SOURCE_COLUMNS, QC_FLAGS, QC_RULES, iter_fixed_windows


_RAW_META: Mapping[str, tuple[str, str, str, str, str]] = {
    "index": ("Original row identifier", "count", "lineage", "Never a predictor", "Preserve as source_row_id"),
    "Timestamp": ("Unix epoch timestamp", "microseconds", "ordering", "Raw absolute time excluded from predictors", "Convert to UTC"),
    "Speed": ("Vehicle speed", "km/h", "driver", "0..200", "Clean; derive distance, acceleration and jerk"),
    "Odometer": ("Cumulative vehicle distance", "km", "aging", "non-negative; inspect decreases/jumps", "Aging control and distance audit only"),
    "RPM": ("Signed traction-motor speed", "rpm", "driver", "16383 is sentinel", "Clean; derive absolute percentiles"),
    "SOC": ("BMS state of charge", "%", "battery", "0..100", "Battery input and SOC/DoD stress"),
    "SOH": ("BMS-reported state of health", "%", "target", "plausibility checked", "Target only; never contemporaneous input"),
    "Torque": ("Signed motor torque", "N m", "driver", "physical plausibility", "Separate traction and regenerative demand"),
    "Motor_Temp": ("Motor temperature", "C", "context", "-40 sentinel; -30..200", "Clean; optional context"),
    "TP_FR": ("Front-right tyre pressure", "kPa", "context", "0 sentinel; 100..500", "Secondary rolling-resistance context"),
    "TP_FL": ("Front-left tyre pressure", "kPa", "context", "0 sentinel; 100..500", "Secondary rolling-resistance context"),
    "TP_RR": ("Rear-right tyre pressure", "kPa", "context", "0 sentinel; 100..500", "Secondary rolling-resistance context"),
    "TP_RL": ("Rear-left tyre pressure", "kPa", "context", "0 sentinel; 100..500", "Secondary rolling-resistance context"),
    "Aux_Pwr_100W": ("Auxiliary electrical load", "W (100-W steps)", "context", "0..20000", "Integrate auxiliary energy"),
    "AC_Pwr_250W": ("Measured HVAC load", "W (250-W steps)", "context", "0..20000", "Primary HVAC power and energy"),
    "EstPwr_AC_50W": ("Estimated HVAC load", "W (50-W steps)", "context", "0..20000", "Fallback/audit for measured HVAC"),
    "HV_Bat_Voltage": ("High-voltage pack voltage", "V", "battery", "0/655.35 sentinel; 240..420", "Clean; recompute battery power"),
    "Brake": ("Undocumented brake command/count", "raw count", "driver", ">=0", "Activity, duration and event features; not labelled percent"),
    "Acc_Pedal": ("Accelerator position/count", "0..200 count", "driver", "0..200", "Normalize to 0..1 and derive intensity"),
    "ECO": ("Eco-mode state", "binary", "driver", "0 or 1", "Usage fraction and interaction"),
    "ePedal": ("e-Pedal state", "binary", "driver", "0 or 1", "Usage fraction and regen interaction"),
    "HV_Bat_Current_2": ("Raw pack current", "A", "battery", "absolute <=1000", "Invert sign so positive means discharge"),
    "Ambient_C_Temp": ("Ambient temperature", "C", "context", "-50..70", "Thermal/context control"),
    "Pack_1_C_Temp": ("Battery temperature sensor 1", "C", "battery", "-30..80", "Recompute pack mean and spread"),
    "Pack_2_C_Temp": ("Battery temperature sensor 2", "C", "battery", "-30..80", "Recompute pack mean and spread"),
    "Pack_4_C_Temp": ("Battery temperature sensor 4", "C", "battery", "-30..80", "Recompute pack mean and spread"),
    "Pack_Avg_C_Temp": ("Source-computed pack mean", "C", "diagnostic", "audit against sensors", "Audit only; exclude source version"),
    "kWh_Remaining": ("Estimated remaining battery energy", "kWh", "leakage", "non-negative", "Descriptive leakage benchmark only"),
    "Horsepower": ("Power-derived horsepower", "hp", "redundant", "audit", "Diagnostic only; exclude from models"),
    "Instant_kW": ("Source-computed signed battery power", "kW", "diagnostic", "audit V x I", "Audit only; replace with clean power"),
    "Driver_ID": ("Pseudonymous driver identifier", "category", "target/group", "0..6 in Nissan data", "Secondary identity target and grouping only"),
    "Trip": ("Globally unique trip identifier", "category", "group", "not missing", "Segmentation and leakage-control key only"),
}


def _raw_dictionary_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in NISSAN_SOURCE_COLUMNS:
        definition, unit, role, valid, action = _RAW_META[col]
        rows.append({
            "Field_ID": f"RAW_{len(rows)+1:03d}", "Table": "source_telemetry",
            "Source_or_Derived": "source", "Column_Name": col,
            "Definition": definition, "Unit": unit, "Data_Type": "numeric",
            "Formula_or_Algorithm": "source value", "Valid_Range_or_QC": valid,
            "Missing_Treatment": "short-gap interpolation only after QC; never across trips",
            "Aggregation": "see action", "Model_Role": role,
            "Leakage_Risk": "high" if role in {"target", "leakage", "group"} else "low",
            "Core_Model_Use": action, "Notes": "",
        })
    return rows


_DERIVED_META: tuple[tuple[str, str, str, str], ...] = (
    ("dataset_id", "Source dataset identifier", "text", "fixed by adapter"),
    ("vehicle_id", "Vehicle identifier", "text", "fixed by adapter/source"),
    ("source_row_id", "Immutable source lineage key", "count", "copy of index"),
    ("datetime_utc", "Parsed UTC timestamp", "datetime", "Timestamp interpreted as Unix microseconds"),
    ("trip_key", "Dataset/vehicle/trip compound key", "text", "dataset_id:vehicle_id:Trip"),
    ("segment_id", "Gap-safe temporal segment", "category", "break at trip change, dt<=0, dt>2 s or invalid time"),
    ("dt_s", "Native integration interval", "s", "valid within-segment timestamp difference"),
    ("qc_bitmask", "All failed row-level QC rules", "bitmask", "bitwise QC_FLAGS"),
    ("key_sensor_valid", "Core-sensor validity indicator", "binary", "speed/SOC/voltage/current/pack temperature all present"),
    ("speed_mps", "Vehicle speed", "m/s", "Speed / 3.6"),
    ("speed_1hz_mps", "Median-smoothed 1-Hz speed", "m/s", "1-Hz median then 3-point rolling median"),
    ("accel_mps2", "Longitudinal acceleration", "m/s2", "within-segment derivative of 1-Hz speed"),
    ("jerk_mps3", "Longitudinal jerk", "m/s3", "within-segment derivative of acceleration"),
    ("accel_pedal_norm", "Normalized accelerator position", "fraction", "Acc_Pedal / 200 clipped to [0,1]"),
    ("brake_active", "Brake activity indicator", "binary", "Brake > 0"),
    ("current_std_a", "Canonical pack current; discharge positive", "A", "-HV_Bat_Current_2"),
    ("p_batt_kw_clean", "Canonical pack power; discharge positive", "kW", "voltage * current_std_a / 1000"),
    ("p_dis_kw", "Discharge power", "kW", "max(p_batt_kw_clean,0)"),
    ("p_regen_kw", "Regenerative power magnitude", "kW", "max(-p_batt_kw_clean,0)"),
    ("pack_temp_avg_clean_c", "Audited pack mean temperature", "C", "mean of at least two valid pack sensors"),
    ("pack_temp_spread_c", "Pack sensor spread", "C", "max minus min valid pack sensor"),
    ("c_dis", "Discharge C-rate", "C", "max(current_std_a,0) / nominal Ah"),
    ("c_regen", "Regenerative C-rate", "C", "max(-current_std_a,0) / nominal Ah"),
    ("c_rate_slew", "Absolute signed C-rate slew", "C/s", "within-segment absolute derivative"),
    ("distance_step_km", "Native integrated distance", "km", "Speed * dt / 3600"),
    ("energy_dis_step_kwh", "Native integrated discharge energy", "kWh", "p_dis_kw * dt / 3600"),
    ("energy_regen_step_kwh", "Native integrated regen energy", "kWh", "p_regen_kw * dt / 3600"),
    ("ah_throughput_step", "Native absolute charge throughput", "Ah", "abs(current_std_a) * dt / 3600"),
    ("efc_step", "Equivalent-full-cycle increment", "cycles", "Ah throughput / (2 * nominal Ah)"),
    ("bsi_rate", "Electrical-rate D-BSI component", "0..1", "0.75 discharge severity + 0.25 regen severity"),
    ("bsi_thermal", "Thermal D-BSI component", "0..1", "0.70 hot + 0.20 cold-regen + 0.10 spread"),
    ("bsi_soc", "SOC D-BSI component", "0..1", "0.70 high-SOC + 0.30 low-SOC"),
    ("bsi_cycle", "Cycling D-BSI component", "0..1", "0.50 DoD + 0.50 EFC-rate severity"),
    ("bsi_transient", "Transient D-BSI component", "0..1", "time-weighted C-rate-slew severity"),
    ("bsi_interaction", "Interaction D-BSI component", "0..1", "hot-discharge and boundary-regen interaction"),
    ("bsi_intensity", "Driving Battery Stress Index", "0..100", "100 * weighted sum of six components"),
    ("stress_dose", "Duration-scaled D-BSI exposure", "stress-hours", "bsi_intensity/100 * duration_hours"),
    ("cumulative_stress_dose", "Chronological cumulative stress dose", "stress-hours", "past/current cumulative sum"),
    ("soh_target_raw_pct", "Trip/checkpoint median BMS SOH", "%", "target only"),
    ("soh_target_trend_pct", "Past-only monotone checkpoint trend", "%", "trailing-three checkpoint median capped at prior trend"),
    ("target_soh_30d_pct", "Nearest future SOH 30+/-7 days away", "%", "future target only"),
    ("target_loss_30d_pp", "SOH loss at the 30-day horizon", "percentage points", "current trend minus target_soh_30d_pct"),
    ("target_soh_500km_pct", "Nearest future SOH 500+/-100 km away", "%", "future target only"),
    ("target_loss_rate_pp_per_1000km", "Mileage-normalized future SOH loss", "pp/1000 km", "loss / travelled km * 1000"),
)


def _derived_dictionary_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, definition, unit, formula in _DERIVED_META:
        rows.append({
            "Field_ID": f"DER_{len(rows)+1:03d}", "Table": "derived_tables",
            "Source_or_Derived": "derived", "Column_Name": name,
            "Definition": definition, "Unit": unit, "Data_Type": "derived",
            "Formula_or_Algorithm": formula, "Valid_Range_or_QC": "finite or explicitly missing",
            "Missing_Treatment": "retain missingness and masks", "Aggregation": "defined by row grain",
            "Model_Role": "target" if name.startswith("target_") or name.startswith("soh_target") else "feature/audit",
            "Leakage_Risk": "high" if name.startswith("target_") or name.startswith("soh_target") else "low",
            "Core_Model_Use": "as specified by feature set", "Notes": "D-BSI components are named concepts",
        })
    return rows


RAW_DATA_DICTIONARY = tuple(_raw_dictionary_records())
DERIVED_DATA_DICTIONARY = tuple(_derived_dictionary_records())


def data_dictionary_frame() -> pd.DataFrame:
    """Return the complete source/derived column dictionary."""

    return pd.DataFrame([*RAW_DATA_DICTIONARY, *DERIVED_DATA_DICTIONARY])


def _percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _rms(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan


def _first_valid(series: pd.Series) -> float:
    values = series.dropna()
    return values.iloc[0] if len(values) else np.nan


def _last_valid(series: pd.Series) -> float:
    values = series.dropna()
    return values.iloc[-1] if len(values) else np.nan


def _sum(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").sum(min_count=1)
    return float(value) if pd.notna(value) else 0.0


def _event_rate(flag: pd.Series, segment: pd.Series, duration_s: float) -> float:
    active = flag.fillna(0).astype(bool)
    previous = active.groupby(segment, sort=False).shift(fill_value=False)
    events = int((active & ~previous).sum())
    return events / max(duration_s / 60.0, 1e-12)


def _base_feature_record(native: pd.DataFrame, one: pd.DataFrame) -> dict[str, Any]:
    n = native.sort_values("datetime_utc", kind="mergesort") if "datetime_utc" in native else native
    o = one.sort_values("datetime_utc", kind="mergesort") if "datetime_utc" in one else one
    source = n if not n.empty else o
    duration_s = _sum(n["dt_s"]) if "dt_s" in n else _sum(o.get("dt_1hz_s", pd.Series(dtype=float)))
    distance = _sum(n.get("distance_step_km", pd.Series(dtype=float)))
    discharge = _sum(n.get("energy_dis_step_kwh", pd.Series(dtype=float)))
    regen = _sum(n.get("energy_regen_step_kwh", pd.Series(dtype=float)))
    net_energy = _sum(n.get("net_energy_step_kwh", pd.Series(dtype=float)))
    ah = _sum(n.get("ah_throughput_step", pd.Series(dtype=float)))
    efc = _sum(n.get("efc_step", pd.Series(dtype=float)))
    soc_series = pd.to_numeric(n.get("SOC", pd.Series(dtype=float)), errors="coerce")
    soc_start, soc_end = _first_valid(soc_series), _last_valid(soc_series)
    dod = float(soc_series.max() - soc_series.min()) if soc_series.notna().any() else np.nan
    moving_time = float(o.get("Speed", pd.Series(dtype=float)).ge(1).sum()) if not o.empty else 0.0

    record: dict[str, Any] = {
        "dataset_id": _first_valid(source.get("dataset_id", pd.Series(dtype=object))),
        "vehicle_id": _first_valid(source.get("vehicle_id", pd.Series(dtype=object))),
        "trip_key": _first_valid(source.get("trip_key", pd.Series(dtype=object))),
        "Trip": _first_valid(source.get("Trip", pd.Series(dtype=float))),
        "Driver_ID": _first_valid(source.get("Driver_ID", pd.Series(dtype=float))),
        "trip_start_utc": source["datetime_utc"].min() if "datetime_utc" in source else pd.NaT,
        "trip_end_utc": source["datetime_utc"].max() if "datetime_utc" in source else pd.NaT,
        "duration_s": duration_s,
        "moving_time_s": moving_time,
        "source_rows": len(n),
        "coverage_pct": float(n.get("key_sensor_valid", pd.Series(False, index=n.index)).fillna(False).mean() * 100) if len(n) else np.nan,
        "soh_target_raw_pct": float(pd.to_numeric(n.get("SOH", pd.Series(dtype=float)), errors="coerce").median()) if len(n) else np.nan,
        "soh_variation_flag": bool(pd.to_numeric(n.get("SOH", pd.Series(dtype=float)), errors="coerce").nunique(dropna=True) > 1),
        "soc_start_pct": soc_start, "soc_end_pct": soc_end, "dod_pct": dod,
        "ambient_mean_c": _mean(n.get("Ambient_C_Temp", pd.Series(dtype=float))),
        "eco_fraction": _mean(n.get("ECO", pd.Series(dtype=float))),
        "epedal_fraction": _mean(n.get("ePedal", pd.Series(dtype=float))),
        "hvac_energy_kwh": _sum(n.get("hvac_energy_step_kwh", pd.Series(dtype=float))),
        "aux_energy_kwh": _sum(n.get("aux_energy_step_kwh", pd.Series(dtype=float))),
        "speed_mean_kmh": _mean(o.get("Speed", pd.Series(dtype=float))),
        "speed_std_kmh": float(pd.to_numeric(o.get("Speed", pd.Series(dtype=float)), errors="coerce").std()) if len(o) else np.nan,
        "speed_p95_kmh": _percentile(o.get("Speed", pd.Series(dtype=float)), 0.95),
        "speed_max_kmh": float(pd.to_numeric(o.get("Speed", pd.Series(dtype=float)), errors="coerce").max()) if len(o) else np.nan,
        "stop_fraction": _mean(o.get("stop_flag", o.get("Speed", pd.Series(dtype=float)).lt(1))),
        "high_speed_fraction": _mean(o.get("high_speed_flag", o.get("Speed", pd.Series(dtype=float)).ge(100))),
        "distance_km": distance, "energy_dis_kwh": discharge,
        "energy_regen_kwh": regen, "net_energy_kwh": net_energy,
        "energy_kwh_per_100km": discharge / distance * 100.0 if distance > 0 else np.nan,
        "regen_fraction": float(np.clip(regen / discharge, 0, 1)) if discharge > 0 else np.nan,
        "ah_throughput": ah, "efc": efc,
        "efc_rate_per_h": efc / (duration_s / 3600.0) if duration_s > 0 else np.nan,
        "current_dis_p95_a": _percentile(n.get("current_std_a", pd.Series(dtype=float)).clip(lower=0), 0.95),
        "current_regen_p95_a": _percentile((-n.get("current_std_a", pd.Series(dtype=float))).clip(lower=0), 0.95),
        "power_dis_p95_kw": _percentile(n.get("p_dis_kw", pd.Series(dtype=float)), 0.95),
        "power_regen_p95_kw": _percentile(n.get("p_regen_kw", pd.Series(dtype=float)), 0.95),
        "pack_temp_mean_c": _mean(n.get("pack_temp_avg_clean_c", pd.Series(dtype=float))),
        "pack_temp_max_c": float(pd.to_numeric(n.get("pack_temp_avg_clean_c", pd.Series(dtype=float)), errors="coerce").max()) if len(n) else np.nan,
        "pack_temp_rise_c": _last_valid(n.get("pack_temp_avg_clean_c", pd.Series(dtype=float))) - _first_valid(n.get("pack_temp_avg_clean_c", pd.Series(dtype=float))),
        "pack_temp_spread_p95_c": _percentile(n.get("pack_temp_spread_c", pd.Series(dtype=float)), 0.95),
        "hot_time_fraction": _mean(n.get("hot_pack_flag", pd.Series(dtype=float))),
        "cold_time_fraction": _mean(n.get("cold_pack_flag", pd.Series(dtype=float))),
        "odometer_clean_km": float(pd.to_numeric(n.get("Odometer", pd.Series(dtype=float)), errors="coerce").median()) if len(n) else np.nan,
    }

    accel = pd.to_numeric(o.get("accel_mps2", pd.Series(dtype=float)), errors="coerce")
    positive_accel = accel.where(accel > 0)
    decel = (-accel).where(accel < 0)
    jerk = pd.to_numeric(o.get("jerk_mps3", pd.Series(dtype=float)), errors="coerce").abs()
    record.update({
        "accel_pos_rms": _rms(positive_accel), "accel_pos_p95": _percentile(positive_accel, 0.95),
        "decel_abs_rms": _rms(decel), "decel_abs_p95": _percentile(decel, 0.95),
        "jerk_rms": _rms(jerk), "jerk_p95": _percentile(jerk, 0.95),
        "pedal_mean": _mean(o.get("accel_pedal_norm", pd.Series(dtype=float))),
        "pedal_p95": _percentile(o.get("accel_pedal_norm", pd.Series(dtype=float)), 0.95),
        "brake_active_fraction": _mean(o.get("brake_active", pd.Series(dtype=float))),
        "torque_pos_p95": _percentile(o.get("Torque", pd.Series(dtype=float)).clip(lower=0), 0.95),
        "torque_regen_p95": _percentile((-o.get("Torque", pd.Series(dtype=float))).clip(lower=0), 0.95),
        "rpm_abs_p95": _percentile(o.get("RPM", pd.Series(dtype=float)).abs(), 0.95),
    })
    segment = o.get("segment_id", pd.Series(0, index=o.index))
    record["harsh_accel_per_min"] = _event_rate(o.get("harsh_accel_flag", pd.Series(False, index=o.index)), segment, duration_s)
    record["harsh_brake_per_min"] = _event_rate(o.get("harsh_brake_flag", pd.Series(False, index=o.index)), segment, duration_s)
    coverage = record["coverage_pct"]
    record["qc_grade"] = "A" if coverage >= 95 else "B" if coverage >= 80 else "C" if coverage >= 50 else "D"
    record.update(compute_dbsi(n if not n.empty else o))
    return record


def aggregate_trip_features(
    native_frame: pd.DataFrame,
    *,
    one_hz: pd.DataFrame | None = None,
    nominal_capacity_ah: float = 111.11,
    dbsi_config: DBSIConfig | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one leakage-auditable feature row per trip."""

    if native_frame.empty:
        return pd.DataFrame()
    group_col = "trip_key" if "trip_key" in native_frame else "Trip"
    records: list[dict[str, Any]] = []
    for key, native in native_frame.groupby(group_col, sort=False, dropna=False):
        if one_hz is None:
            from .preprocessing import resample_telemetry_1hz
            one = resample_telemetry_1hz(native)
        else:
            one_group_col = group_col if group_col in one_hz else "Trip"
            one = one_hz.loc[one_hz[one_group_col].eq(key)]
        record = _base_feature_record(native, one)
        # Recalculate D-BSI with the caller's nominal capacity.
        if dbsi_config is None:
            effective_dbsi: DBSIConfig | Mapping[str, Any] = DBSIConfig(
                nominal_capacity_ah=nominal_capacity_ah
            )
        elif isinstance(dbsi_config, DBSIConfig):
            effective_dbsi = dbsi_config
        else:
            effective_dbsi = {**dict(dbsi_config), "nominal_capacity_ah": nominal_capacity_ah}
        record.update(compute_dbsi(native, effective_dbsi))
        records.append(record)
    out = pd.DataFrame(records).sort_values("trip_start_utc", kind="mergesort").reset_index(drop=True)
    if "stress_dose" in out:
        vehicle_cols = [c for c in ("dataset_id", "vehicle_id") if c in out]
        if vehicle_cols:
            out["cumulative_stress_dose"] = out.groupby(
                vehicle_cols, sort=False, dropna=False
            )["stress_dose"].transform(lambda values: values.fillna(0).cumsum())
        else:
            out["cumulative_stress_dose"] = out["stress_dose"].fillna(0).cumsum()
    return out


def aggregate_window_features(
    one_hz: pd.DataFrame,
    *,
    native_frame: pd.DataFrame | None = None,
    length_s: int = 256,
    stride_s: int = 128,
    min_coverage: float = 0.80,
    min_moving_s: float = 10.0,
    nominal_capacity_ah: float = 111.11,
    dbsi_config: DBSIConfig | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one feature row for each valid fixed-length Transformer window."""

    if one_hz.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for window in iter_fixed_windows(
        one_hz, length_s=length_s, stride_s=stride_s,
        min_coverage=min_coverage, include_partial=True,
    ):
        o = window.data
        moving_s = float(pd.to_numeric(o.get("Speed", pd.Series(dtype=float)), errors="coerce").ge(1).sum())
        if moving_s < min_moving_s:
            continue
        if native_frame is not None and not native_frame.empty:
            n = native_frame.loc[
                native_frame["segment_id"].eq(window.segment_id)
                & native_frame["datetime_utc"].ge(window.start_utc)
                & native_frame["datetime_utc"].lt(window.end_utc + pd.Timedelta(seconds=1))
            ]
        else:
            n = o.loc[o["observed_1hz"].fillna(0).gt(0)].copy()
            if "dt_s" not in n:
                n["dt_s"] = n.get("dt_1hz_s", 1.0)
        record = _base_feature_record(n, o)
        record.update({
            "window_id": window.window_id, "window_start_utc": window.start_utc,
            "window_end_utc": window.end_utc, "segment_id": window.segment_id,
            "coverage_pct": window.coverage_pct, "moving_time_s": moving_s,
            "window_length_s": length_s, "window_stride_s": stride_s,
        })
        if dbsi_config is None:
            effective_dbsi: DBSIConfig | Mapping[str, Any] = DBSIConfig(
                nominal_capacity_ah=nominal_capacity_ah
            )
        elif isinstance(dbsi_config, DBSIConfig):
            effective_dbsi = dbsi_config
        else:
            effective_dbsi = {**dict(dbsi_config), "nominal_capacity_ah": nominal_capacity_ah}
        record.update(compute_dbsi(n, effective_dbsi))
        records.append(record)
    return pd.DataFrame(records)


def build_health_windows(
    trip_features: pd.DataFrame,
    *,
    day_horizon: float = 30.0,
    day_tolerance: float = 7.0,
    distance_horizon_km: float = 500.0,
    distance_tolerance_km: float = 100.0,
    collapse_plateaus: bool = True,
    _single_vehicle: bool = False,
) -> pd.DataFrame:
    """Construct past-only health checkpoints and future SOH targets.

    Every chronology, trend, exposure history, and future match is isolated by
    dataset/vehicle.  Duplicate anchors may be collapsed for modelling, but all
    plateau observations remain eligible as future horizon targets.
    """

    if trip_features.empty:
        return pd.DataFrame()
    group_cols = [c for c in ("dataset_id", "vehicle_id") if c in trip_features]
    if not _single_vehicle and group_cols:
        pieces = [
            build_health_windows(
                group,
                day_horizon=day_horizon,
                day_tolerance=day_tolerance,
                distance_horizon_km=distance_horizon_km,
                distance_tolerance_km=distance_tolerance_km,
                collapse_plateaus=collapse_plateaus,
                _single_vehicle=True,
            )
            for _, group in trip_features.groupby(group_cols, sort=False, dropna=False)
        ]
        pieces = [piece for piece in pieces if not piece.empty]
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    trips = trip_features.copy()
    trips["trip_start_utc"] = pd.to_datetime(
        trips["trip_start_utc"], utc=True, errors="coerce", format="mixed"
    )
    if "trip_end_utc" in trips:
        trips["trip_end_utc"] = pd.to_datetime(
            trips["trip_end_utc"], utc=True, errors="coerce", format="mixed"
        )
    anchor_col = "trip_end_utc" if "trip_end_utc" in trips else "trip_start_utc"
    trips = trips.sort_values(anchor_col, kind="mergesort").reset_index(drop=True)
    recorded_distance_step = (
        pd.to_numeric(trips["distance_km"], errors="coerce")
        if "distance_km" in trips
        else pd.Series(0.0, index=trips.index)
    )
    recorded_distance = recorded_distance_step.fillna(0).cumsum()
    odometer = (
        pd.to_numeric(trips["odometer_clean_km"], errors="coerce")
        if "odometer_clean_km" in trips
        else pd.Series(np.nan, index=trips.index, dtype="float64")
    )
    # Forecast horizons represent vehicle distance travelled, including drives
    # absent from the logger. Prefer monotonic odometer progression; fall back to
    # integrated recorded-trip distance only where odometer data are unusable.
    valid_odometer = odometer.notna() & odometer.ge(0)
    odometer_axis = odometer - odometer.loc[valid_odometer].iloc[0] if valid_odometer.any() else pd.Series(np.nan, index=trips.index)
    odometer_monotonic = bool(valid_odometer.sum() >= 2 and odometer.loc[valid_odometer].is_monotonic_increasing)
    if odometer_monotonic:
        trips["cumulative_distance_km"] = odometer_axis.interpolate(limit_direction="both")
        trips["distance_axis_source"] = "odometer"
    else:
        trips["cumulative_distance_km"] = recorded_distance
        trips["distance_axis_source"] = "recorded_trip_integration"
    soh = (
        pd.to_numeric(trips["soh_target_raw_pct"], errors="coerce")
        if "soh_target_raw_pct" in trips
        else pd.Series(np.nan, index=trips.index, dtype="float64")
    )
    timeline = trips.loc[soh.notna() & trips[anchor_col].notna()].copy().reset_index(drop=True)
    if timeline.empty:
        return timeline

    trend_values: list[float] = []
    raw_history: list[float] = []
    previous = np.inf
    for raw in pd.to_numeric(timeline["soh_target_raw_pct"], errors="coerce"):
        raw_history.append(float(raw))
        candidate = float(np.median(raw_history[-3:]))
        previous = min(previous, candidate)
        trend_values.append(previous)
    timeline["soh_target_trend_pct"] = trend_values
    raw_timeline_soh = pd.to_numeric(timeline["soh_target_raw_pct"], errors="coerce")
    anchor_mask = raw_timeline_soh.ne(raw_timeline_soh.shift()) if collapse_plateaus else pd.Series(True, index=timeline.index)
    checkpoints = timeline.loc[anchor_mask].copy()
    checkpoints["_timeline_position"] = checkpoints.index.astype(int)
    checkpoints = checkpoints.reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for i, anchor in checkpoints.iterrows():
        timeline_position = int(anchor["_timeline_position"])
        anchor_time = anchor[anchor_col]
        anchor_distance = float(anchor["cumulative_distance_km"])
        row = anchor.to_dict()
        row.pop("_timeline_position", None)
        dataset_id = anchor.get("dataset_id", "dataset")
        vehicle_id = anchor.get("vehicle_id", "vehicle")
        row["health_window_id"] = f"{dataset_id}:{vehicle_id}:{i:04d}"
        row["anchor_time_utc"] = anchor_time
        row["days_since_start"] = (anchor_time - timeline[anchor_col].iloc[0]).total_seconds() / 86400.0
        if i > 0:
            previous_cp = checkpoints.iloc[i - 1]
            row["days_since_prev"] = (anchor_time - previous_cp[anchor_col]).total_seconds() / 86400.0
            row["distance_since_prev_km"] = anchor_distance - float(previous_cp["cumulative_distance_km"])
            row["soh_change_pp"] = float(anchor["soh_target_trend_pct"] - previous_cp["soh_target_trend_pct"])
            row["loss_rate_pp_per_1000km"] = (
                -row["soh_change_pp"] / row["distance_since_prev_km"] * 1000.0
                if row["distance_since_prev_km"] > 0 else np.nan
            )
        else:
            row.update({"days_since_prev": np.nan, "distance_since_prev_km": np.nan, "soh_change_pp": np.nan, "loss_rate_pp_per_1000km": np.nan})

        history = trips.loc[trips[anchor_col].le(anchor_time)]
        for days in (7, 30, 90):
            hist = history.loc[history[anchor_col].ge(anchor_time - pd.Timedelta(days=days))]
            prefix = f"hist_{days}d"
            row[f"{prefix}_distance_km"] = pd.to_numeric(hist.get("distance_km", 0), errors="coerce").sum()
            row[f"{prefix}_ah"] = pd.to_numeric(hist.get("ah_throughput", 0), errors="coerce").sum()
            row[f"{prefix}_efc"] = pd.to_numeric(hist.get("efc", 0), errors="coerce").sum()
            row[f"{prefix}_bsi_dose"] = pd.to_numeric(hist.get("stress_dose", 0), errors="coerce").sum()
            row[f"{prefix}_temp_max_c"] = pd.to_numeric(hist.get("pack_temp_max_c"), errors="coerce").max()
            if "behavior_level" in hist:
                for level in ("eco", "balanced", "aggressive"):
                    row[f"{prefix}_behavior_share_{level}"] = hist["behavior_level"].astype(str).str.lower().eq(level).mean()

        # Plateau SOH rows remain valid future observations even when duplicate
        # anchors are removed, avoiding degradation-event selection bias.
        future = timeline.iloc[timeline_position + 1 :].copy()
        row.update({
            "target_soh_30d_pct": np.nan, "target_loss_30d_pp": np.nan,
            "target_soh_500km_pct": np.nan, "target_loss_500km_pp": np.nan,
            "target_loss_rate_pp_per_1000km": np.nan,
            "target_quality_flag": "unavailable",
        })
        found: list[str] = []
        if not future.empty:
            future["delta_days"] = (future[anchor_col] - anchor_time).dt.total_seconds() / 86400.0
            day_candidates = future.loc[future["delta_days"].between(day_horizon - day_tolerance, day_horizon + day_tolerance)]
            if not day_candidates.empty:
                target = day_candidates.iloc[(day_candidates["delta_days"] - day_horizon).abs().argmin()]
                row["target_soh_30d_pct"] = float(target["soh_target_trend_pct"])
                row["target_loss_30d_pp"] = max(0.0, float(anchor["soh_target_trend_pct"] - target["soh_target_trend_pct"]))
                found.append("30d")
            future["delta_km"] = future["cumulative_distance_km"] - anchor_distance
            km_candidates = future.loc[future["delta_km"].between(distance_horizon_km - distance_tolerance_km, distance_horizon_km + distance_tolerance_km)]
            if not km_candidates.empty:
                target = km_candidates.iloc[(km_candidates["delta_km"] - distance_horizon_km).abs().argmin()]
                delta_km = float(target["delta_km"])
                loss = max(0.0, float(anchor["soh_target_trend_pct"] - target["soh_target_trend_pct"]))
                row["target_soh_500km_pct"] = float(target["soh_target_trend_pct"])
                row["target_loss_500km_pp"] = loss
                row["target_loss_rate_pp_per_1000km"] = loss / delta_km * 1000.0 if delta_km > 0 else np.nan
                found.append("500km")
        row["target_quality_flag"] = "+".join(found) if found else "unavailable"
        records.append(row)
    return pd.DataFrame(records)


def _sentinel_count(series: pd.Series, column: str) -> int:
    values = pd.to_numeric(series, errors="coerce")
    if column == "RPM":
        return int(values.abs().eq(16383).sum())
    if column == "Motor_Temp":
        return int(values.le(-40).sum())
    if column == "HV_Bat_Voltage":
        return int((values.eq(0) | np.isclose(values, 655.35)).sum())
    if column == "Odometer":
        return int(values.eq(255).sum())
    if column.startswith("TP_"):
        return int(values.eq(0).sum())
    return 0


def build_source_profile(frame: pd.DataFrame, *, total_rows: int | None = None) -> pd.DataFrame:
    """Profile every source column; quantiles may use a bounded pipeline sample."""

    records: list[dict[str, Any]] = []
    reported_rows = int(total_rows if total_rows is not None else len(frame))
    for col in NISSAN_SOURCE_COLUMNS:
        series = frame[col] if col in frame else pd.Series(dtype=float)
        numeric = pd.to_numeric(series, errors="coerce")
        records.append({
            "Column_Name": col, "Row_Count": reported_rows,
            "Profile_Sample_Rows": len(series), "Data_Type": str(series.dtype),
            "Null_Count_In_Sample": int(series.isna().sum()),
            "Sentinel_Count_In_Sample": _sentinel_count(series, col),
            "Unique_Count_In_Sample": int(series.nunique(dropna=True)),
            "Min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "P01": _percentile(numeric, 0.01), "P05": _percentile(numeric, 0.05),
            "Median": _percentile(numeric, 0.50), "P95": _percentile(numeric, 0.95),
            "P99": _percentile(numeric, 0.99),
            "Max": float(numeric.max()) if numeric.notna().any() else np.nan,
            "Action": _RAW_META[col][4],
        })
    return pd.DataFrame(records)


def build_qc_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize row and trip effects of every frozen QC rule."""

    bits = pd.to_numeric(frame.get("qc_bitmask", pd.Series(0, index=frame.index)), errors="coerce").fillna(0).astype("uint32")
    trip_col = "trip_key" if "trip_key" in frame else "Trip" if "Trip" in frame else None
    records: list[dict[str, Any]] = []
    for rule in QC_RULES:
        bit = QC_FLAGS[str(rule["Name"])]
        affected = bits.map(lambda x: bool(int(x) & bit))
        records.append({
            **rule, "Affected_Rows": int(affected.sum()),
            "Affected_Trips": int(frame.loc[affected, trip_col].nunique(dropna=True)) if trip_col else np.nan,
            "Rows_Remaining": int((~affected).sum()),
        })
    return pd.DataFrame(records)


__all__ = [
    "DERIVED_DATA_DICTIONARY", "RAW_DATA_DICTIONARY", "aggregate_trip_features",
    "aggregate_window_features", "build_health_windows", "build_qc_summary",
    "build_source_profile", "data_dictionary_frame",
]
