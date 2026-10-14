"""Audited Nissan driver-to-battery physics features.

All quantities are calculated independently within a trip.  No target values,
future trips, validation rows, or test rows are used to estimate a physical
feature.  Cumulative/history features are then constructed in chronological,
past-only order within each vehicle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


NISSAN_NOMINAL_CAPACITY_AH = 111.11
ARRHENIUS_ACTIVATION_ENERGY_J_MOL = 35_000.0
GAS_CONSTANT_J_MOL_K = 8.314462618
REFERENCE_TEMPERATURE_C = 25.0


def relative_arrhenius_factor(
    temperature_c: pd.Series | np.ndarray,
    *,
    activation_energy_j_mol: float = ARRHENIUS_ACTIVATION_ENERGY_J_MOL,
    reference_temperature_c: float = REFERENCE_TEMPERATURE_C,
) -> np.ndarray:
    """Return a temperature acceleration factor relative to the reference."""

    values = np.asarray(temperature_c, dtype=float)
    kelvin = values + 273.15
    reference = reference_temperature_c + 273.15
    valid = np.isfinite(kelvin) & (kelvin > 0)
    result = np.full(values.shape, np.nan, dtype=float)
    exponent = activation_energy_j_mol / GAS_CONSTANT_J_MOL_K * (
        1.0 / reference - 1.0 / kelvin[valid]
    )
    result[valid] = np.exp(np.clip(exponent, -20.0, 20.0))
    return result


def estimate_trip_resistance(
    frame: pd.DataFrame,
    *,
    minimum_current_step_a: float = 10.0,
    maximum_step_seconds: float = 2.0,
    resistance_bounds_ohm: tuple[float, float] = (0.005, 0.30),
) -> tuple[float, int, float]:
    """Estimate pack resistance from within-trip current/voltage transients.

    With discharge-positive current, an increased current normally causes a
    voltage drop, hence ``R0 = -dV/dI``.  Only same-segment, short-time steps
    inside conservative pack-resistance bounds contribute.
    """

    current = pd.to_numeric(frame.get("current_std_a"), errors="coerce")
    voltage = pd.to_numeric(frame.get("voltage_v"), errors="coerce")
    time = pd.to_datetime(frame.get("datetime_utc"), utc=True, errors="coerce")
    delta_i = current.diff()
    delta_v = voltage.diff()
    delta_t = time.diff().dt.total_seconds()
    same_segment = pd.Series(True, index=frame.index)
    if "segment_id" in frame:
        segment = frame["segment_id"]
        same_segment = segment.eq(segment.shift())
    resistance = -delta_v / delta_i
    valid = (
        same_segment
        & delta_i.abs().ge(minimum_current_step_a)
        & delta_t.gt(0)
        & delta_t.le(maximum_step_seconds)
        & resistance.between(*resistance_bounds_ohm)
    )
    samples = resistance.loc[valid].replace([np.inf, -np.inf], np.nan).dropna()
    if samples.empty:
        return np.nan, 0, np.nan
    median = float(samples.median())
    mad = float((samples - median).abs().median())
    return median, int(len(samples)), mad


def nissan_trip_physics_features(
    frame: pd.DataFrame,
    *,
    nominal_capacity_ah: float = NISSAN_NOMINAL_CAPACITY_AH,
) -> dict[str, Any]:
    """Calculate one trip's named electrical/thermal degradation exposures."""

    if frame.empty:
        raise ValueError("Cannot calculate trip physics from an empty frame")
    ordered = frame.sort_values("datetime_utc", kind="mergesort").copy()
    dt = pd.to_numeric(ordered.get("dt_s"), errors="coerce").clip(lower=0, upper=10)
    current = pd.to_numeric(ordered.get("current_std_a"), errors="coerce")
    voltage = pd.to_numeric(ordered.get("voltage_v"), errors="coerce")
    soc = pd.to_numeric(ordered.get("soc_pct"), errors="coerce")
    pack_temp = pd.to_numeric(ordered.get("pack_temp_avg_clean_c"), errors="coerce")
    ambient = pd.to_numeric(ordered.get("ambient_temp_c"), errors="coerce")
    c_rate = current.abs() / float(nominal_capacity_ah)
    valid_electrical = dt.notna() & current.notna() & voltage.notna()
    valid_thermal = dt.notna() & pack_temp.notna()
    resistance, resistance_n, resistance_mad = estimate_trip_resistance(ordered)
    arrhenius = pd.Series(relative_arrhenius_factor(pack_temp), index=ordered.index)
    hours = dt / 3600.0
    c2 = c_rate.pow(2)

    same_segment = pd.Series(True, index=ordered.index)
    if "segment_id" in ordered:
        same_segment = ordered["segment_id"].eq(ordered["segment_id"].shift())
    delta_c = c_rate.diff().abs().where(same_segment)
    signed_power = voltage * current / 1000.0
    power_sign = np.sign(signed_power.where(signed_power.abs().ge(0.5)))
    reversal = power_sign.ne(power_sign.shift()) & power_sign.notna() & power_sign.shift().notna()
    reversal &= same_segment

    def dose(values: pd.Series) -> float:
        valid = values.notna() & hours.notna()
        return float((values.loc[valid] * hours.loc[valid]).sum())

    duration_h = float(hours.loc[hours.notna()].sum())
    joule_kwh = (
        float((current.pow(2) * resistance * dt).loc[valid_electrical].sum() / 3.6e6)
        if np.isfinite(resistance)
        else np.nan
    )
    output: dict[str, Any] = {
        "dataset_id": str(ordered["dataset_id"].iloc[0]),
        "vehicle_id": str(ordered["vehicle_id"].iloc[0]),
        "trip_key": str(ordered["trip_key"].iloc[0]),
        "Trip": ordered["Trip"].iloc[0] if "Trip" in ordered else np.nan,
        "trip_start_utc": pd.to_datetime(ordered["datetime_utc"].iloc[0], utc=True),
        "trip_end_utc": pd.to_datetime(ordered["datetime_utc"].iloc[-1], utc=True),
        "physics_duration_h": duration_h,
        "physics_electrical_coverage": float(valid_electrical.mean()),
        "physics_thermal_coverage": float(valid_thermal.mean()),
        "resistance_estimate_ohm": resistance,
        "resistance_step_count": resistance_n,
        "resistance_mad_ohm": resistance_mad,
        "joule_heat_estimate_kwh": joule_kwh,
        "c_rate_abs_mean": float(c_rate.mean()),
        "c_rate_abs_p95": float(c_rate.quantile(0.95)),
        "c_rate_abs_peak": float(c_rate.max()),
        "c_rate_squared_dose_h": dose(c2),
        "electrothermal_dose_h": dose(c2 * arrhenius),
        "arrhenius_pack_dose_h": dose(arrhenius),
        "pack_temp_above_25_degree_h": dose((pack_temp - 25.0).clip(lower=0)),
        "pack_temp_above_35_time_h": dose(pack_temp.ge(35.0).astype(float)),
        "pack_ambient_delta_mean_c": float((pack_temp - ambient).mean()),
        "soc_high_dose_h": dose(((soc - 80.0) / 20.0).clip(lower=0).pow(2)),
        "soc_low_dose_h": dose(((20.0 - soc) / 20.0).clip(lower=0).pow(2)),
        "soc_high_hot_time_h": dose((soc.ge(80.0) & pack_temp.ge(35.0)).astype(float)),
        "high_c_rate_time_h": dose(c_rate.ge(1.0).astype(float)),
        "c_rate_transient_dose": float(delta_c.sum(skipna=True)),
        "power_reversal_count": int(reversal.sum()),
        "power_reversal_per_h": int(reversal.sum()) / duration_h if duration_h > 0 else np.nan,
        "physics_equations_version": "Nissan-Physics-v1.0",
        "past_only_feature": True,
    }
    return output


_CUMULATIVE_COLUMNS = (
    "distance_km",
    "ah_throughput",
    "efc",
    "stress_dose",
    "joule_heat_estimate_kwh",
    "c_rate_squared_dose_h",
    "electrothermal_dose_h",
    "arrhenius_pack_dose_h",
    "pack_temp_above_25_degree_h",
    "pack_temp_above_35_time_h",
    "soc_high_dose_h",
    "soc_low_dose_h",
    "soc_high_hot_time_h",
    "high_c_rate_time_h",
    "c_rate_transient_dose",
    "power_reversal_count",
)


def append_past_only_exposures(
    trip_features: pd.DataFrame,
    *,
    history_days: Iterable[int] = (7, 30, 90),
) -> pd.DataFrame:
    """Append cumulative and rolling features using present/past trips only."""

    result = trip_features.copy()
    result["trip_end_utc"] = pd.to_datetime(
        result["trip_end_utc"], utc=True, errors="coerce", format="mixed"
    )
    group_columns = [c for c in ("dataset_id", "vehicle_id") if c in result]
    pieces: list[pd.DataFrame] = []
    groups = result.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, result)]
    for _, group in groups:
        group = group.sort_values("trip_end_utc", kind="mergesort").copy()
        for column in _CUMULATIVE_COLUMNS:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
                group[f"cum_{column}"] = values.cumsum()
        first_time = group["trip_end_utc"].iloc[0]
        group["calendar_days_since_start"] = (
            group["trip_end_utc"] - first_time
        ).dt.total_seconds() / 86400.0
        for row_index, row in group.iterrows():
            for days in history_days:
                start = row["trip_end_utc"] - pd.Timedelta(days=int(days))
                history = group.loc[
                    group["trip_end_utc"].between(start, row["trip_end_utc"], inclusive="both")
                ]
                for column in _CUMULATIVE_COLUMNS:
                    if column in history:
                        group.loc[row_index, f"hist_{days}d_{column}"] = pd.to_numeric(
                            history[column], errors="coerce"
                        ).sum(min_count=1)
                for column in (
                    "accel_pos_rms", "decel_abs_rms", "jerk_rms", "pedal_mean",
                    "brake_active_fraction", "harsh_accel_per_min",
                    "harsh_brake_per_min", "speed_p95_kmh", "bsi_intensity",
                ):
                    if column in history:
                        group.loc[row_index, f"hist_{days}d_mean_{column}"] = pd.to_numeric(
                            history[column], errors="coerce"
                        ).mean()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else result


def nissan_physics_parameters() -> dict[str, Any]:
    return {
        "nominal_capacity_ah": NISSAN_NOMINAL_CAPACITY_AH,
        "arrhenius_activation_energy_j_mol": ARRHENIUS_ACTIVATION_ENERGY_J_MOL,
        "gas_constant_j_mol_k": GAS_CONSTANT_J_MOL_K,
        "reference_temperature_c": REFERENCE_TEMPERATURE_C,
        "resistance_formula": "R0=-delta_voltage/delta_discharge_positive_current",
        "joule_heat_formula": "integral(I^2*R0*dt)",
        "c_rate_squared_formula": "integral((abs(I)/Q_nominal)^2*dt)",
        "electrothermal_formula": "integral(C_rate^2*relative_Arrhenius(pack_temperature)*dt)",
        "causal_order": "driver_action->electrical_response->thermal/SOC/cycle/transient_dose->SOH",
        "pack_temperature_observed": True,
        "target_used_in_feature_engineering": False,
    }


__all__ = [
    "ARRHENIUS_ACTIVATION_ENERGY_J_MOL",
    "NISSAN_NOMINAL_CAPACITY_AH",
    "append_past_only_exposures",
    "estimate_trip_resistance",
    "nissan_physics_parameters",
    "nissan_trip_physics_features",
    "relative_arrhenius_factor",
]
