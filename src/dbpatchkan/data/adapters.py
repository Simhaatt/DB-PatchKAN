"""Dataset adapters for the common DB-PatchKAN telemetry schema.

External publications do not use one stable set of column names.  RWTH and
Nature adapters therefore accept an explicit ``column_map`` while also resolving
common aliases.  No silent sign inference is performed: callers declare the raw
current convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .preprocessing import PreprocessConfig, preprocess_nissan_frame


CANONICAL_COLUMNS: tuple[str, ...] = (
    "dataset_id", "vehicle_id", "source_row_id", "datetime_utc", "trip_id",
    "trip_key", "driver_id", "speed_kmh", "odometer_km", "soc_pct", "soh_pct",
    "voltage_v", "current_a", "power_kw", "pack_temp_c", "ambient_temp_c",
    "nominal_capacity_ah", "measured_capacity_ah", "capacity_soh_pct",
    "sampling_interval_s",
)

CORE_SEQUENCE_COLUMNS: tuple[str, ...] = (
    "speed_kmh", "current_a", "voltage_v", "soc_pct", "pack_temp_c",
)


DEFAULT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "vehicle_id": ("vehicle_id", "Vehicle_ID", "vehicle", "car_id", "vin_hash"),
    "source_row_id": ("source_row_id", "index", "row_id", "record_id"),
    "timestamp": ("datetime_utc", "timestamp", "Timestamp", "time", "datetime", "DateTime"),
    "trip_id": ("trip_id", "Trip", "trip", "drive_id", "journey_id"),
    "driver_id": ("driver_id", "Driver_ID", "driver"),
    "speed_kmh": ("speed_kmh", "Speed", "speed", "vehicle_speed"),
    "odometer_km": ("odometer_km", "Odometer", "odometer", "mileage_km", "mileage"),
    "soc_pct": ("soc_pct", "SOC", "soc", "state_of_charge"),
    "soh_pct": ("soh_pct", "SOH", "soh", "state_of_health"),
    "voltage_v": ("voltage_v", "HV_Bat_Voltage", "pack_voltage", "battery_voltage", "voltage"),
    "current_raw": ("current_a", "HV_Bat_Current_2", "pack_current", "battery_current", "current"),
    "power_kw": ("power_kw", "p_batt_kw_clean", "battery_power_kw", "pack_power_kw"),
    "pack_temp_c": ("pack_temp_c", "pack_temp_avg_clean_c", "Pack_Avg_C_Temp", "battery_temperature", "battery_temp"),
    "ambient_temp_c": ("ambient_temp_c", "Ambient_C_Temp", "ambient_temperature", "outside_temp"),
    "nominal_capacity_ah": ("nominal_capacity_ah", "rated_capacity_ah", "capacity_nominal_ah"),
    "measured_capacity_ah": ("measured_capacity_ah", "usable_capacity_ah", "capacity_ah"),
    "capacity_soh_pct": ("capacity_soh_pct", "capacity_based_soh", "capacity_soh"),
}


class CanonicalAdapter(ABC):
    """Interface implemented by each source-dataset adapter."""

    dataset_id: str

    @abstractmethod
    def adapt(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a frame containing every :data:`CANONICAL_COLUMNS` field."""

    def validate(self, frame: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
        return validate_canonical_frame(frame, strict=strict)


def _resolve(frame: pd.DataFrame, canonical: str, column_map: Mapping[str, str]) -> str | None:
    explicit = column_map.get(canonical)
    if explicit is not None:
        if explicit not in frame:
            raise ValueError(f"Mapped source column {explicit!r} for {canonical!r} is absent")
        return explicit
    for candidate in DEFAULT_ALIASES.get(canonical, (canonical,)):
        if candidate in frame:
            return candidate
    return None


def _value(frame: pd.DataFrame, source: str | None, default: Any = np.nan) -> pd.Series:
    if source is None:
        return pd.Series(default, index=frame.index)
    return frame[source]


def _parse_timestamp(series: pd.Series, unit: str | None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if unit is not None:
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    # Numeric magnitudes allow deterministic unit selection without consulting
    # future rows.  Explicit timestamp_unit remains recommended for publication.
    finite = numeric.dropna().abs()
    magnitude = float(finite.median()) if len(finite) else 0.0
    inferred = "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
    parsed_numeric = pd.to_datetime(numeric, unit=inferred, utc=True, errors="coerce")
    parsed_text = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed_numeric.where(numeric.notna(), parsed_text)


@dataclass
class MappingAdapter(CanonicalAdapter):
    """Generic explicit/alias-based adapter used by external datasets."""

    dataset_id: str
    column_map: Mapping[str, str] = field(default_factory=dict)
    timestamp_unit: str | None = None
    current_positive_discharge: bool = True
    default_vehicle_id: str = "vehicle_01"
    default_trip_id: str = "trip_0001"
    default_nominal_capacity_ah: float | None = None
    retain_source_columns: bool = False

    def adapt(self, frame: pd.DataFrame) -> pd.DataFrame:
        source = frame.copy() if self.retain_source_columns else pd.DataFrame(index=frame.index)
        out = source
        timestamp_source = _resolve(frame, "timestamp", self.column_map)
        if timestamp_source is None:
            raise ValueError("An external adapter requires a timestamp column")
        out["dataset_id"] = self.dataset_id
        out["vehicle_id"] = _value(frame, _resolve(frame, "vehicle_id", self.column_map), self.default_vehicle_id).fillna(self.default_vehicle_id).astype(str)
        source_id = _value(frame, _resolve(frame, "source_row_id", self.column_map), np.nan)
        out["source_row_id"] = source_id.where(source_id.notna(), pd.Series(np.arange(len(frame)), index=frame.index))
        out["datetime_utc"] = _parse_timestamp(frame[timestamp_source], self.timestamp_unit)
        out["trip_id"] = _value(frame, _resolve(frame, "trip_id", self.column_map), self.default_trip_id).fillna(self.default_trip_id).astype(str)
        out["trip_key"] = out["dataset_id"] + ":" + out["vehicle_id"] + ":" + out["trip_id"]
        out["driver_id"] = _value(frame, _resolve(frame, "driver_id", self.column_map))

        for canonical, alias_key in (
            ("speed_kmh", "speed_kmh"), ("odometer_km", "odometer_km"),
            ("soc_pct", "soc_pct"), ("soh_pct", "soh_pct"),
            ("voltage_v", "voltage_v"), ("pack_temp_c", "pack_temp_c"),
            ("ambient_temp_c", "ambient_temp_c"),
            ("measured_capacity_ah", "measured_capacity_ah"),
            ("capacity_soh_pct", "capacity_soh_pct"),
        ):
            out[canonical] = pd.to_numeric(_value(frame, _resolve(frame, alias_key, self.column_map)), errors="coerce")

        current = pd.to_numeric(_value(frame, _resolve(frame, "current_raw", self.column_map)), errors="coerce")
        out["current_a"] = current if self.current_positive_discharge else -current
        calculated_power = out["voltage_v"] * out["current_a"] / 1000.0
        power_source = _resolve(frame, "power_kw", self.column_map)
        supplied_power = (
            pd.to_numeric(_value(frame, power_source), errors="coerce")
            if power_source
            else pd.Series(np.nan, index=frame.index, dtype="float64")
        )
        # Canonical power always follows the declared discharge-positive current
        # convention. Supplied power is preserved solely as an audit signal.
        out["source_power_kw_audit"] = supplied_power
        out["power_kw_recalculated"] = calculated_power
        out["source_power_error_kw"] = supplied_power - calculated_power
        out["power_kw"] = calculated_power

        nominal_source = _resolve(frame, "nominal_capacity_ah", self.column_map)
        out["nominal_capacity_ah"] = pd.to_numeric(
            _value(frame, nominal_source, self.default_nominal_capacity_ah), errors="coerce"
        )
        calculated_capacity_soh = out["measured_capacity_ah"] / out["nominal_capacity_ah"] * 100.0
        out["capacity_soh_pct"] = out["capacity_soh_pct"].fillna(calculated_capacity_soh)
        # Establish deterministic, globally unique temporal segments before
        # any sequence export.  A break is declared for non-increasing time or
        # a gap exceeding three native sampling intervals (with a 30 s floor).
        out = out.sort_values(
            ["dataset_id", "vehicle_id", "trip_key", "datetime_utc", "source_row_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        dt = out.groupby("trip_key", sort=False)["datetime_utc"].diff().dt.total_seconds()
        typical = dt.where(dt.gt(0)).groupby(out["trip_key"], sort=False).transform("median")
        threshold = (3.0 * typical).clip(lower=30.0).fillna(30.0)
        segment_break = dt.isna() | dt.le(0) | dt.gt(threshold)
        local_segment = segment_break.groupby(out["trip_key"], sort=False).cumsum().astype("int64")
        pairs = pd.MultiIndex.from_arrays([out["trip_key"].astype(str), local_segment])
        out["segment_id"] = pd.factorize(pairs, sort=True)[0].astype("int64")
        out["sampling_interval_s"] = dt.where(dt.gt(0))

        # Shared canonical-to-model/D-BSI aliases. Optional driver channels stay
        # explicitly missing and are handled through model missingness masks.
        out["speed_mps"] = out["speed_kmh"] / 3.6
        out["current_discharge_a"] = out["current_a"]
        out["current_std_a"] = out["current_a"]
        out["pack_temp_avg_c"] = out["pack_temp_c"]
        out["pack_temp_avg_clean_c"] = out["pack_temp_c"]
        out["SOC"] = out["soc_pct"]
        out["dt_s"] = out["sampling_interval_s"].fillna(0.0)
        out["valid_fraction"] = out[
            ["speed_kmh", "current_a", "voltage_v", "soc_pct", "pack_temp_c"]
        ].notna().all(axis=1).astype("float32")
        out["missing_fraction"] = out[
            ["speed_kmh", "current_a", "voltage_v", "soc_pct", "pack_temp_c"]
        ].isna().mean(axis=1).astype("float32")
        out["accel_mps2"] = out.groupby("segment_id", sort=False)["speed_mps"].diff() / out[
            "sampling_interval_s"
        ].replace(0, np.nan)
        out["jerk_mps3"] = out.groupby("segment_id", sort=False)["accel_mps2"].diff() / out[
            "sampling_interval_s"
        ].replace(0, np.nan)
        out["key_sensor_valid"] = out[
            ["speed_mps", "current_discharge_a", "voltage_v", "soc_pct", "pack_temp_avg_c"]
        ].notna().all(axis=1)
        for optional in (
            "torque_nm", "accel_pedal_norm",
            "brake_active", "rpm_abs", "eco", "epedal", "pack_temp_spread_c",
            "hvac_power_w",
        ):
            if optional not in out:
                out[optional] = np.nan

        for column in CANONICAL_COLUMNS:
            if column not in out:
                out[column] = np.nan
        canonical_first = list(CANONICAL_COLUMNS) + [c for c in out if c not in CANONICAL_COLUMNS]
        return out[canonical_first]


@dataclass
class NissanAdapter(CanonicalAdapter):
    dataset_id: str = "nissan_leaf_real_world"
    vehicle_id: str = "nissan_leaf_2018_40kwh_01"
    nominal_capacity_ah: float = 111.11
    retain_source_columns: bool = True

    def adapt(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = preprocess_nissan_frame(
            frame,
            config=PreprocessConfig(
                dataset_id=self.dataset_id, vehicle_id=self.vehicle_id,
                nominal_capacity_ah=self.nominal_capacity_ah,
            ),
        )
        canonical = pd.DataFrame(index=cleaned.index)
        canonical["dataset_id"] = cleaned["dataset_id"]
        canonical["vehicle_id"] = cleaned["vehicle_id"]
        canonical["source_row_id"] = cleaned["source_row_id"]
        canonical["datetime_utc"] = cleaned["datetime_utc"]
        canonical["trip_id"] = cleaned["Trip"].astype("Int64").astype(str)
        canonical["trip_key"] = cleaned["trip_key"]
        canonical["driver_id"] = cleaned["Driver_ID"]
        canonical["speed_kmh"] = cleaned["Speed"]
        canonical["odometer_km"] = cleaned["Odometer"]
        canonical["soc_pct"] = cleaned["SOC"]
        canonical["soh_pct"] = cleaned["SOH"]
        canonical["voltage_v"] = cleaned["HV_Bat_Voltage"]
        canonical["current_a"] = cleaned["current_std_a"]
        canonical["power_kw"] = cleaned["p_batt_kw_clean"]
        canonical["pack_temp_c"] = cleaned["pack_temp_avg_clean_c"]
        canonical["ambient_temp_c"] = cleaned["Ambient_C_Temp"]
        canonical["nominal_capacity_ah"] = self.nominal_capacity_ah
        canonical["measured_capacity_ah"] = np.nan
        canonical["capacity_soh_pct"] = np.nan
        canonical["sampling_interval_s"] = cleaned["dt_s"].where(cleaned["dt_s"].gt(0))
        if self.retain_source_columns:
            extras = cleaned.drop(columns=[c for c in canonical if c in cleaned], errors="ignore")
            canonical = pd.concat([canonical, extras], axis=1)
        return canonical


@dataclass
class RWTHAdapter(MappingAdapter):
    """Adapter for RWTH commercial-fleet telemetry/capacity tables."""

    dataset_id: str = "rwth_commercial_ev_fleet"
    current_positive_discharge: bool = True


@dataclass
class NatureEVAdapter(MappingAdapter):
    """Adapter for the low-cadence 300-EV Nature Communications dataset."""

    dataset_id: str = "nature_300_ev"
    current_positive_discharge: bool = True


def validate_canonical_frame(frame: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return field-level diagnostics and optionally reject invalid core data."""

    diagnostics: list[dict[str, Any]] = []
    for column in CANONICAL_COLUMNS:
        missing_col = column not in frame
        null_fraction = 1.0 if missing_col or len(frame) == 0 else float(frame[column].isna().mean())
        diagnostics.append({
            "Column_Name": column, "Present": not missing_col,
            "Null_Fraction": null_fraction,
            "Core_Sequence_Field": column in CORE_SEQUENCE_COLUMNS,
        })
    result = pd.DataFrame(diagnostics)
    if strict:
        absent = result.loc[~result["Present"] | (result["Core_Sequence_Field"] & result["Null_Fraction"].eq(1)), "Column_Name"].tolist()
        if absent:
            raise ValueError(f"Canonical frame is missing required usable fields: {absent}")
        if "datetime_utc" in frame and not isinstance(frame["datetime_utc"].dtype, pd.DatetimeTZDtype):
            raise ValueError("datetime_utc must be timezone-aware")
    return result


def adapt_nissan(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return NissanAdapter(**kwargs).adapt(frame)


def adapt_rwth(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return RWTHAdapter(**kwargs).adapt(frame)


def adapt_nature_ev(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return NatureEVAdapter(**kwargs).adapt(frame)


__all__ = [
    "CANONICAL_COLUMNS", "CORE_SEQUENCE_COLUMNS", "CanonicalAdapter",
    "MappingAdapter", "NatureEVAdapter", "NissanAdapter", "RWTHAdapter",
    "adapt_nature_ev", "adapt_nissan", "adapt_rwth", "validate_canonical_frame",
]
